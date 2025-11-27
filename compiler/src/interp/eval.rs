//! Tree-walking interpreter for HIR

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use miette::{Result, miette};

use crate::hir::*;

use super::env::Environment;
use super::value::{ControlFlow, Value};

/// Tree-walking interpreter
pub struct Interpreter {
    /// Variable environment
    env: Environment,
    /// Function definitions (by name)
    functions: HashMap<String, Rc<HirFn>>,
    /// Struct definitions (by name)
    structs: HashMap<String, HirStruct>,
    /// Enum definitions (by name)
    enums: HashMap<String, HirEnum>,
    /// Output buffer for testing
    output: Vec<String>,
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        Interpreter {
            env: Environment::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            output: Vec::new(),
        }
    }

    /// Get captured output (for testing)
    pub fn get_output(&self) -> &[String] {
        &self.output
    }

    /// Clear output buffer
    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    /// Interpret an HIR program
    pub fn interpret(&mut self, hir: &Hir) -> Result<Value> {
        // First pass: collect all definitions
        for item in &hir.items {
            match item {
                HirItem::Function(f) => {
                    self.functions.insert(f.name.clone(), Rc::new(f.clone()));
                }
                HirItem::Struct(s) => {
                    self.structs.insert(s.name.clone(), s.clone());
                }
                HirItem::Enum(e) => {
                    self.enums.insert(e.name.clone(), e.clone());
                }
                _ => {}
            }
        }

        // Look for main function
        if let Some(main_fn) = self.functions.get("main").cloned() {
            self.call_function(&main_fn, vec![])
        } else {
            // No main, evaluate last expression or return unit
            Ok(Value::Unit)
        }
    }

    /// Call a function with arguments
    fn call_function(&mut self, func: &HirFn, args: Vec<Value>) -> Result<Value> {
        self.env.push_scope();

        // Bind parameters
        for (param, arg) in func.ty.params.iter().zip(args.into_iter()) {
            self.env.define(param.name.clone(), arg);
        }

        // Execute body
        let result = self.eval_block(&func.body);

        self.env.pop_scope();

        match result {
            Ok(v) => Ok(v),
            Err(ControlFlow::Return(v)) => Ok(v),
            Err(ControlFlow::Break(_)) => Err(miette!("break outside loop")),
            Err(ControlFlow::Continue) => Err(miette!("continue outside loop")),
        }
    }

    /// Evaluate a block
    fn eval_block(&mut self, block: &HirBlock) -> Result<Value, ControlFlow> {
        self.env.push_scope();

        let mut result = Value::Unit;

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == block.stmts.len() - 1;

            match stmt {
                HirStmt::Let { name, value, .. } => {
                    let val = if let Some(expr) = value {
                        self.eval_expr(expr)?
                    } else {
                        Value::Unit
                    };
                    self.env.define(name.clone(), val);
                }
                HirStmt::Expr(expr) => {
                    result = self.eval_expr(expr)?;
                }
                HirStmt::Assign { target, value } => {
                    let val = self.eval_expr(value)?;
                    self.assign_target(target, val)?;
                }
            }
        }

        self.env.pop_scope();
        Ok(result)
    }

    /// Evaluate an expression
    fn eval_expr(&mut self, expr: &HirExpr) -> Result<Value, ControlFlow> {
        match &expr.kind {
            HirExprKind::Literal(lit) => Ok(self.eval_literal(lit)),

            HirExprKind::Local(name) => {
                // First check local variables
                if let Some(val) = self.env.get(name) {
                    return Ok(val);
                }
                // Then check if it's a function name
                if let Some(func) = self.functions.get(name).cloned() {
                    return Ok(Value::Function {
                        func,
                        captures: HashMap::new(),
                    });
                }
                // Not found
                Err(ControlFlow::Return(Value::Unit))
            }

            HirExprKind::Global(name) => {
                // Check if it's a function
                if let Some(func) = self.functions.get(name).cloned() {
                    Ok(Value::Function {
                        func,
                        captures: HashMap::new(),
                    })
                } else {
                    self.env
                        .get(name)
                        .ok_or_else(|| ControlFlow::Return(Value::Unit))
                }
            }

            HirExprKind::Binary { op, left, right } => {
                let lhs = self.eval_expr(left)?;

                // Short-circuit for And/Or
                match op {
                    HirBinaryOp::And => {
                        if !lhs.is_truthy() {
                            return Ok(Value::Bool(false));
                        }
                        let rhs = self.eval_expr(right)?;
                        return Ok(Value::Bool(rhs.is_truthy()));
                    }
                    HirBinaryOp::Or => {
                        if lhs.is_truthy() {
                            return Ok(Value::Bool(true));
                        }
                        let rhs = self.eval_expr(right)?;
                        return Ok(Value::Bool(rhs.is_truthy()));
                    }
                    _ => {}
                }

                let rhs = self.eval_expr(right)?;
                self.eval_binary(*op, lhs, rhs)
            }

            HirExprKind::Unary { op, expr: inner } => {
                let val = self.eval_expr(inner)?;
                self.eval_unary(*op, val)
            }

            HirExprKind::Call { func, args } => {
                let callee = self.eval_expr(func)?;
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.eval_expr(arg)?);
                }
                self.eval_call(callee, arg_values)
            }

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = self.eval_expr(condition)?;
                if cond.is_truthy() {
                    self.eval_block(then_branch)
                } else if let Some(else_expr) = else_branch {
                    self.eval_expr(else_expr)
                } else {
                    Ok(Value::Unit)
                }
            }

            HirExprKind::Block(block) => self.eval_block(block),

            HirExprKind::Loop(block) => loop {
                match self.eval_block(block) {
                    Ok(_) => continue,
                    Err(ControlFlow::Continue) => continue,
                    Err(ControlFlow::Break(val)) => {
                        return Ok(val.unwrap_or(Value::Unit));
                    }
                    Err(ControlFlow::Return(v)) => {
                        return Err(ControlFlow::Return(v));
                    }
                }
            },

            HirExprKind::Return(value) => {
                let val = if let Some(expr) = value {
                    self.eval_expr(expr)?
                } else {
                    Value::Unit
                };
                Err(ControlFlow::Return(val))
            }

            HirExprKind::Break(value) => {
                let val = if let Some(expr) = value {
                    Some(self.eval_expr(expr)?)
                } else {
                    None
                };
                Err(ControlFlow::Break(val))
            }

            HirExprKind::Continue => Err(ControlFlow::Continue),

            HirExprKind::Tuple(elements) => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expr(elem)?);
                }
                Ok(Value::Tuple(values))
            }

            HirExprKind::Array(elements) => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expr(elem)?);
                }
                Ok(Value::Array(Rc::new(RefCell::new(values))))
            }

            HirExprKind::Struct { name, fields } => {
                let mut field_values = HashMap::new();
                for (field_name, field_expr) in fields {
                    field_values.insert(field_name.clone(), self.eval_expr(field_expr)?);
                }
                Ok(Value::Struct {
                    name: name.clone(),
                    fields: field_values,
                })
            }

            HirExprKind::Variant {
                enum_name,
                variant,
                fields,
            } => {
                let mut field_values = Vec::new();
                for field_expr in fields {
                    field_values.push(self.eval_expr(field_expr)?);
                }
                Ok(Value::Variant {
                    enum_name: enum_name.clone(),
                    variant_name: variant.clone(),
                    fields: field_values,
                })
            }

            HirExprKind::Field { base, field } => {
                let base_val = self.eval_expr(base)?;
                match base_val {
                    Value::Struct { fields, .. } => fields
                        .get(field)
                        .cloned()
                        .ok_or_else(|| ControlFlow::Return(Value::Unit)),
                    Value::Ref(r) => {
                        let inner = r.borrow();
                        if let Value::Struct { ref fields, .. } = *inner {
                            fields
                                .get(field)
                                .cloned()
                                .ok_or_else(|| ControlFlow::Return(Value::Unit))
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::TupleField { base, index } => {
                let base_val = self.eval_expr(base)?;
                match base_val {
                    Value::Tuple(elements) => elements
                        .get(*index)
                        .cloned()
                        .ok_or_else(|| ControlFlow::Return(Value::Unit)),
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::Index { base, index } => {
                let base_val = self.eval_expr(base)?;
                let idx_val = self.eval_expr(index)?;

                let idx = idx_val
                    .as_int()
                    .ok_or_else(|| ControlFlow::Return(Value::Unit))?
                    as usize;

                match base_val {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        arr.get(idx)
                            .cloned()
                            .ok_or_else(|| ControlFlow::Return(Value::Unit))
                    }
                    Value::String(s) => s
                        .chars()
                        .nth(idx)
                        .map(|c| Value::String(c.to_string()))
                        .ok_or_else(|| ControlFlow::Return(Value::Unit)),
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::Ref {
                mutable: _,
                expr: inner,
            } => {
                let val = self.eval_expr(inner)?;
                Ok(Value::Ref(Rc::new(RefCell::new(val))))
            }

            HirExprKind::Deref(inner) => {
                let val = self.eval_expr(inner)?;
                match val {
                    Value::Ref(r) => Ok(r.borrow().clone()),
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::Match { scrutinee, arms } => {
                let val = self.eval_expr(scrutinee)?;

                for arm in arms {
                    if let Some(bindings) = self.match_pattern(&arm.pattern, &val) {
                        // Check guard if present
                        if let Some(guard) = &arm.guard {
                            self.env.push_scope();
                            for (name, value) in &bindings {
                                self.env.define(name.clone(), value.clone());
                            }
                            let guard_result = self.eval_expr(guard)?;
                            self.env.pop_scope();

                            if !guard_result.is_truthy() {
                                continue;
                            }
                        }

                        // Execute arm body with bindings
                        self.env.push_scope();
                        for (name, value) in bindings {
                            self.env.define(name, value);
                        }
                        let result = self.eval_expr(&arm.body);
                        self.env.pop_scope();
                        return result;
                    }
                }

                // No match found - this should be a runtime error
                Err(ControlFlow::Return(Value::Unit))
            }

            HirExprKind::Cast { expr: inner, .. } => {
                // For now, just evaluate the inner expression
                // Real casting would convert types
                self.eval_expr(inner)
            }

            HirExprKind::Closure { params, body } => {
                // Capture current environment
                let captures = self.env.capture_all();

                // Create a synthetic HirFn for the closure
                let closure_fn = HirFn {
                    id: crate::common::NodeId::dummy(),
                    name: "<closure>".to_string(),
                    ty: HirFnType {
                        params: params.clone(),
                        return_type: Box::new(body.ty.clone()),
                        effects: Vec::new(),
                    },
                    body: HirBlock {
                        stmts: vec![HirStmt::Expr(body.as_ref().clone())],
                        ty: body.ty.clone(),
                    },
                };

                Ok(Value::Function {
                    func: Rc::new(closure_fn),
                    captures,
                })
            }

            HirExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let recv = self.eval_expr(receiver)?;
                let mut arg_values = vec![recv.clone()];
                for arg in args {
                    arg_values.push(self.eval_expr(arg)?);
                }

                // Handle built-in methods
                match (recv, method.as_str()) {
                    (Value::Array(arr), "len") => Ok(Value::Int(arr.borrow().len() as i64)),
                    (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
                    (Value::Array(arr), "push") => {
                        if let Some(val) = arg_values.get(1) {
                            arr.borrow_mut().push(val.clone());
                        }
                        Ok(Value::Unit)
                    }
                    (Value::Array(arr), "pop") => Ok(arr.borrow_mut().pop().unwrap_or(Value::None)),
                    _ => {
                        // Try to find a function with method name
                        if let Some(func) = self.functions.get(method).cloned() {
                            self.call_function(&func, arg_values)
                                .map_err(|e| ControlFlow::Return(Value::Unit))
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
                }
            }

            // Effect operations - not fully implemented
            HirExprKind::Perform { .. } | HirExprKind::Handle { .. } | HirExprKind::Sample(_) => {
                Ok(Value::Unit)
            }
        }
    }

    /// Evaluate a literal
    fn eval_literal(&self, lit: &HirLiteral) -> Value {
        match lit {
            HirLiteral::Unit => Value::Unit,
            HirLiteral::Bool(b) => Value::Bool(*b),
            HirLiteral::Int(n) => Value::Int(*n),
            HirLiteral::Float(f) => Value::Float(*f),
            HirLiteral::Char(c) => Value::String(c.to_string()),
            HirLiteral::String(s) => Value::String(s.clone()),
        }
    }

    /// Evaluate a binary operation
    fn eval_binary(&self, op: HirBinaryOp, lhs: Value, rhs: Value) -> Result<Value, ControlFlow> {
        match op {
            HirBinaryOp::Add => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Sub => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - b as f64)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Mul => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * b as f64)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Div => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 {
                        Err(ControlFlow::Return(Value::Unit)) // Division by zero
                    } else {
                        Ok(Value::Int(a / b))
                    }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / b as f64)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Rem => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 {
                        Err(ControlFlow::Return(Value::Unit))
                    } else {
                        Ok(Value::Int(a % b))
                    }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Eq => Ok(Value::Bool(lhs == rhs)),
            HirBinaryOp::Ne => Ok(Value::Bool(lhs != rhs)),
            HirBinaryOp::Lt => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a < b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Le => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a <= b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Gt => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a > b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Ge => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a >= b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::And => Ok(Value::Bool(lhs.is_truthy() && rhs.is_truthy())),
            HirBinaryOp::Or => Ok(Value::Bool(lhs.is_truthy() || rhs.is_truthy())),
            HirBinaryOp::BitAnd => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::BitOr => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::BitXor => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Shl => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a << b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Shr => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a >> b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
        }
    }

    /// Evaluate a unary operation
    fn eval_unary(&self, op: HirUnaryOp, val: Value) -> Result<Value, ControlFlow> {
        match op {
            HirUnaryOp::Neg => match val {
                Value::Int(n) => Ok(Value::Int(-n)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirUnaryOp::Not => match val {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                Value::Int(n) => Ok(Value::Int(!n)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirUnaryOp::Ref | HirUnaryOp::RefMut => Ok(Value::Ref(Rc::new(RefCell::new(val)))),
            HirUnaryOp::Deref => match val {
                Value::Ref(r) => Ok(r.borrow().clone()),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
        }
    }

    /// Evaluate a function call
    fn eval_call(&mut self, callee: Value, args: Vec<Value>) -> Result<Value, ControlFlow> {
        match callee {
            Value::Function { func, captures } => {
                // Set up environment with captures
                self.env.push_scope();
                for (name, value) in captures {
                    self.env.define(name, value);
                }

                // Bind parameters
                for (param, arg) in func.ty.params.iter().zip(args.into_iter()) {
                    self.env.define(param.name.clone(), arg);
                }

                // Execute body
                let result = self.eval_block(&func.body);

                self.env.pop_scope();

                match result {
                    Ok(v) => Ok(v),
                    Err(ControlFlow::Return(v)) => Ok(v),
                    Err(cf) => Err(cf),
                }
            }
            _ => {
                // Check if it's a builtin by looking at the callee name
                // For now, handle common cases
                self.call_builtin_by_args(&args)
            }
        }
    }

    /// Try calling a builtin function by examining arguments
    fn call_builtin_by_args(&mut self, args: &[Value]) -> Result<Value, ControlFlow> {
        // Default: return unit
        Ok(Value::Unit)
    }

    /// Call a named builtin function
    pub fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, ControlFlow> {
        match name {
            "print" => {
                let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
                let line = output.join(" ");
                print!("{}", line);
                self.output.push(line);
                Ok(Value::Unit)
            }
            "println" => {
                let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
                let line = output.join(" ");
                println!("{}", line);
                self.output.push(line);
                Ok(Value::Unit)
            }
            "assert" => {
                if let Some(val) = args.first() {
                    if !val.is_truthy() {
                        return Err(ControlFlow::Return(Value::Unit)); // Assertion failed
                    }
                }
                Ok(Value::Unit)
            }
            "assert_eq" => {
                if args.len() >= 2 {
                    if args[0] != args[1] {
                        return Err(ControlFlow::Return(Value::Unit)); // Assertion failed
                    }
                }
                Ok(Value::Unit)
            }
            "len" => {
                if let Some(val) = args.first() {
                    match val {
                        Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                        Value::String(s) => Ok(Value::Int(s.len() as i64)),
                        Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
                        _ => Ok(Value::Int(0)),
                    }
                } else {
                    Ok(Value::Int(0))
                }
            }
            "type_of" => {
                if let Some(val) = args.first() {
                    Ok(Value::String(val.type_name().to_string()))
                } else {
                    Ok(Value::String("unknown".to_string()))
                }
            }
            "Some" => {
                if let Some(val) = args.into_iter().next() {
                    Ok(Value::Some(Box::new(val)))
                } else {
                    Ok(Value::Some(Box::new(Value::Unit)))
                }
            }
            "None" => Ok(Value::None),
            "Ok" => {
                if let Some(val) = args.into_iter().next() {
                    Ok(Value::Ok(Box::new(val)))
                } else {
                    Ok(Value::Ok(Box::new(Value::Unit)))
                }
            }
            "Err" => {
                if let Some(val) = args.into_iter().next() {
                    Ok(Value::Err(Box::new(val)))
                } else {
                    Ok(Value::Err(Box::new(Value::Unit)))
                }
            }
            _ => {
                // Try to find function by name
                if let Some(func) = self.functions.get(name).cloned() {
                    self.call_function(&func, args)
                        .map_err(|_| ControlFlow::Return(Value::Unit))
                } else {
                    Ok(Value::Unit)
                }
            }
        }
    }

    /// Match a pattern against a value, returning bindings if successful
    fn match_pattern(&self, pattern: &HirPattern, value: &Value) -> Option<Vec<(String, Value)>> {
        match pattern {
            HirPattern::Wildcard => Some(vec![]),

            HirPattern::Binding { name, .. } => Some(vec![(name.clone(), value.clone())]),

            HirPattern::Literal(lit) => {
                let lit_val = self.eval_literal(lit);
                if lit_val == *value {
                    Some(vec![])
                } else {
                    None
                }
            }

            HirPattern::Tuple(patterns) => {
                if let Value::Tuple(values) = value {
                    if patterns.len() != values.len() {
                        return None;
                    }
                    let mut bindings = Vec::new();
                    for (pat, val) in patterns.iter().zip(values.iter()) {
                        bindings.extend(self.match_pattern(pat, val)?);
                    }
                    Some(bindings)
                } else {
                    None
                }
            }

            HirPattern::Struct { name, fields } => {
                if let Value::Struct {
                    name: struct_name,
                    fields: struct_fields,
                } = value
                {
                    if name != struct_name {
                        return None;
                    }
                    let mut bindings = Vec::new();
                    for (field_name, field_pat) in fields {
                        let field_val = struct_fields.get(field_name)?;
                        bindings.extend(self.match_pattern(field_pat, field_val)?);
                    }
                    Some(bindings)
                } else {
                    None
                }
            }

            HirPattern::Variant {
                enum_name,
                variant,
                patterns,
            } => {
                if let Value::Variant {
                    enum_name: e,
                    variant_name: v,
                    fields,
                } = value
                {
                    if enum_name != e || variant != v {
                        return None;
                    }
                    if patterns.len() != fields.len() {
                        return None;
                    }
                    let mut bindings = Vec::new();
                    for (pat, val) in patterns.iter().zip(fields.iter()) {
                        bindings.extend(self.match_pattern(pat, val)?);
                    }
                    Some(bindings)
                } else {
                    None
                }
            }

            HirPattern::Or(patterns) => {
                for pat in patterns {
                    if let Some(bindings) = self.match_pattern(pat, value) {
                        return Some(bindings);
                    }
                }
                None
            }
        }
    }

    /// Assign to a target expression
    fn assign_target(&mut self, target: &HirExpr, value: Value) -> Result<(), ControlFlow> {
        match &target.kind {
            HirExprKind::Local(name) => {
                self.env.assign(name, value);
                Ok(())
            }
            HirExprKind::Field { base, field } => {
                let base_val = self.eval_expr(base)?;
                if let Value::Ref(r) = base_val {
                    if let Value::Struct { ref mut fields, .. } = *r.borrow_mut() {
                        fields.insert(field.clone(), value);
                    }
                }
                Ok(())
            }
            HirExprKind::Index { base, index } => {
                let base_val = self.eval_expr(base)?;
                let idx = self.eval_expr(index)?.as_int().unwrap_or(0) as usize;

                if let Value::Array(arr) = base_val {
                    let mut arr = arr.borrow_mut();
                    if idx < arr.len() {
                        arr[idx] = value;
                    }
                }
                Ok(())
            }
            HirExprKind::Deref(inner) => {
                let inner_val = self.eval_expr(inner)?;
                if let Value::Ref(r) = inner_val {
                    *r.borrow_mut() = value;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}
