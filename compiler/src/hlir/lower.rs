//! HIR to HLIR lowering
//!
//! This module transforms the typed HIR into SSA-form HLIR with explicit
//! control flow and basic blocks.

use super::builder::{FunctionBuilder, ModuleBuilder};
use super::ir::*;
use crate::hir::*;
use std::collections::HashMap;

/// Lower HIR to HLIR
pub fn lower(hir: &Hir) -> HlirModule {
    let lowering = HirToHlir::new();
    lowering.lower_module(hir)
}

/// HIR to HLIR lowering context
struct HirToHlir {
    module_builder: ModuleBuilder,
    /// Map from function names to their signatures (for call resolution)
    functions: HashMap<String, HlirType>,
}

impl HirToHlir {
    fn new() -> Self {
        Self {
            module_builder: ModuleBuilder::new("main"),
            functions: HashMap::new(),
        }
    }

    fn lower_module(mut self, hir: &Hir) -> HlirModule {
        // First pass: collect function signatures and type definitions
        for item in &hir.items {
            match item {
                HirItem::Function(f) => {
                    let ret_ty = HlirType::from_hir(&f.ty.return_type);
                    self.functions.insert(f.name.clone(), ret_ty);
                }
                HirItem::Struct(s) => {
                    let fields: Vec<_> = s
                        .fields
                        .iter()
                        .map(|f| (f.name.clone(), HlirType::from_hir(&f.ty)))
                        .collect();
                    self.module_builder.add_type_def(HlirTypeDef {
                        name: s.name.clone(),
                        kind: HlirTypeDefKind::Struct(fields),
                    });
                }
                HirItem::Enum(e) => {
                    let variants: Vec<_> = e
                        .variants
                        .iter()
                        .map(|v| {
                            (
                                v.name.clone(),
                                v.fields.iter().map(HlirType::from_hir).collect(),
                            )
                        })
                        .collect();
                    self.module_builder.add_type_def(HlirTypeDef {
                        name: e.name.clone(),
                        kind: HlirTypeDefKind::Enum(variants),
                    });
                }
                HirItem::Global(g) => {
                    // We'll need a fresh value ID
                    let global = HlirGlobal {
                        id: ValueId(0), // Will be assigned properly
                        name: g.name.clone(),
                        ty: HlirType::from_hir(&g.ty),
                        init: None,
                        is_const: g.is_const,
                    };
                    self.module_builder.add_global(global);
                }
                _ => {}
            }
        }

        // Second pass: lower functions
        for item in &hir.items {
            if let HirItem::Function(f) = item {
                let hlir_func = self.lower_function(f);
                self.module_builder.add_function(hlir_func);
            }
        }

        self.module_builder.build()
    }

    fn lower_function(&mut self, f: &HirFn) -> HlirFunction {
        let func_id = self.module_builder.fresh_func_id();
        let return_type = HlirType::from_hir(&f.ty.return_type);

        let mut func_builder = FunctionBuilder::new(func_id, &f.name, return_type.clone());

        // Add parameters
        for param in &f.ty.params {
            let ty = HlirType::from_hir(&param.ty);
            func_builder.add_param(&param.name, ty);
        }

        // Create entry block
        let entry = func_builder.create_block("entry");
        func_builder.switch_to_block(entry);

        // Lower function body
        let mut ctx = LoweringContext::new(&mut func_builder, &self.functions);
        let result = ctx.lower_block(&f.body);

        // Add return if not already terminated
        if !ctx.is_terminated() {
            if return_type == HlirType::Void {
                ctx.builder.build_return(None);
            } else {
                ctx.builder.build_return(result);
            }
        }

        func_builder.build()
    }
}

/// Context for lowering expressions within a function
struct LoweringContext<'a> {
    builder: &'a mut FunctionBuilder,
    functions: &'a HashMap<String, HlirType>,
    /// Track if current block is terminated
    terminated: bool,
    /// Loop context for break/continue
    loop_stack: Vec<LoopContext>,
}

struct LoopContext {
    continue_block: BlockId,
    break_block: BlockId,
    /// Values from break expressions (for loop expressions)
    break_values: Vec<(BlockId, ValueId)>,
}

impl<'a> LoweringContext<'a> {
    fn new(builder: &'a mut FunctionBuilder, functions: &'a HashMap<String, HlirType>) -> Self {
        Self {
            builder,
            functions,
            terminated: false,
            loop_stack: Vec::new(),
        }
    }

    fn is_terminated(&self) -> bool {
        self.terminated
    }

    fn lower_block(&mut self, block: &HirBlock) -> Option<ValueId> {
        let mut last_value = None;

        for stmt in &block.stmts {
            if self.terminated {
                break;
            }
            last_value = self.lower_stmt(stmt);
        }

        last_value
    }

    fn lower_stmt(&mut self, stmt: &HirStmt) -> Option<ValueId> {
        match stmt {
            HirStmt::Let {
                name,
                ty,
                value,
                is_mut,
            } => {
                let hlir_ty = HlirType::from_hir(ty);

                if *is_mut {
                    // Mutable variable: allocate stack slot
                    let slot = self.builder.alloc_var(name, hlir_ty.clone());
                    if let Some(init) = value {
                        let init_val = self.lower_expr(init)?;
                        self.builder.build_store(slot, init_val);
                    }
                } else {
                    // Immutable variable: SSA binding
                    if let Some(init) = value {
                        let init_val = self.lower_expr(init)?;
                        self.builder.bind_var(name, init_val);
                    }
                }
                None
            }
            HirStmt::Expr(expr) => self.lower_expr(expr),
            HirStmt::Assign { target, value } => {
                let val = self.lower_expr(value)?;
                self.lower_assign(target, val);
                None
            }
        }
    }

    fn lower_assign(&mut self, target: &HirExpr, value: ValueId) {
        match &target.kind {
            HirExprKind::Local(name) => {
                // Check if it's a mutable variable with a slot
                if let Some(slot) = self.builder.get_var_slot(name) {
                    self.builder.build_store(slot, value);
                }
            }
            HirExprKind::Deref(inner) => {
                if let Some(ptr) = self.lower_expr(inner) {
                    self.builder.build_store(ptr, value);
                }
            }
            HirExprKind::Field { base, field } => {
                if let Some(base_ptr) = self.lower_lvalue(base) {
                    // Get field index from type
                    let field_idx = self.get_field_index(&base.ty, field);
                    let field_ty = HlirType::from_hir(&target.ty);
                    let field_ptr = self.builder.build_field_ptr(base_ptr, field_idx, field_ty);
                    self.builder.build_store(field_ptr, value);
                }
            }
            HirExprKind::Index { base, index } => {
                if let Some(base_ptr) = self.lower_lvalue(base) {
                    if let Some(idx) = self.lower_expr(index) {
                        let elem_ty = HlirType::from_hir(&target.ty);
                        let elem_ptr = self.builder.build_elem_ptr(base_ptr, idx, elem_ty);
                        self.builder.build_store(elem_ptr, value);
                    }
                }
            }
            _ => {}
        }
    }

    fn lower_lvalue(&mut self, expr: &HirExpr) -> Option<ValueId> {
        match &expr.kind {
            HirExprKind::Local(name) => self.builder.get_var_slot(name),
            HirExprKind::Deref(inner) => self.lower_expr(inner),
            HirExprKind::Field { base, field } => {
                let base_ptr = self.lower_lvalue(base)?;
                let field_idx = self.get_field_index(&base.ty, field);
                let field_ty = HlirType::from_hir(&expr.ty);
                Some(self.builder.build_field_ptr(base_ptr, field_idx, field_ty))
            }
            HirExprKind::Index { base, index } => {
                let base_ptr = self.lower_lvalue(base)?;
                let idx = self.lower_expr(index)?;
                let elem_ty = HlirType::from_hir(&expr.ty);
                Some(self.builder.build_elem_ptr(base_ptr, idx, elem_ty))
            }
            _ => None,
        }
    }

    fn get_field_index(&self, _ty: &HirType, _field: &str) -> usize {
        // TODO: Look up actual field index from type definition
        0
    }

    fn lower_expr(&mut self, expr: &HirExpr) -> Option<ValueId> {
        if self.terminated {
            return None;
        }

        let ty = HlirType::from_hir(&expr.ty);

        match &expr.kind {
            HirExprKind::Literal(lit) => Some(self.lower_literal(lit, &ty)),

            HirExprKind::Local(name) => {
                // Try immutable binding first
                if let Some(val) = self.builder.get_var(name) {
                    return Some(val);
                }
                // Try mutable variable (load from slot)
                if let Some(val) = self.builder.load_var(name, ty) {
                    return Some(val);
                }
                // Try function reference
                if self.functions.contains_key(name) {
                    return Some(self.builder.build_const(
                        HlirConstant::FunctionRef(name.clone()),
                        HlirType::Ptr(Box::new(HlirType::Void)),
                    ));
                }
                None
            }

            HirExprKind::Global(name) => Some(self.builder.build_const(
                HlirConstant::GlobalRef(name.clone()),
                HlirType::Ptr(Box::new(ty)),
            )),

            HirExprKind::Binary { op, left, right } => {
                let left_val = self.lower_expr(left)?;
                let right_val = self.lower_expr(right)?;
                let left_ty = HlirType::from_hir(&left.ty);
                Some(self.lower_binary_op(*op, left_val, right_val, &left_ty, &ty))
            }

            HirExprKind::Unary { op, expr: inner } => {
                let operand = self.lower_expr(inner)?;
                let inner_ty = HlirType::from_hir(&inner.ty);
                Some(self.lower_unary_op(*op, operand, &inner_ty))
            }

            HirExprKind::Call { func, args } => {
                let arg_vals: Vec<_> = args.iter().filter_map(|a| self.lower_expr(a)).collect();

                // Check if it's a direct function call
                if let HirExprKind::Local(name) = &func.kind {
                    if self.functions.contains_key(name) {
                        return Some(self.builder.build_call(name, arg_vals, ty));
                    }
                }

                // Indirect call
                let func_val = self.lower_expr(func)?;
                Some(self.builder.build_call_indirect(func_val, arg_vals, ty))
            }

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => self.lower_if(condition, then_branch, else_branch.as_deref(), &ty),

            HirExprKind::Block(block) => self.lower_block(block),

            HirExprKind::Return(value) => {
                let ret_val = value.as_ref().and_then(|v| self.lower_expr(v));
                self.builder.build_return(ret_val);
                self.terminated = true;
                None
            }

            HirExprKind::Loop(body) => self.lower_loop(body, &ty),

            HirExprKind::Break(value) => {
                let break_val = value.as_ref().and_then(|v| self.lower_expr(v));

                if let Some(loop_ctx) = self.loop_stack.last_mut() {
                    if let Some(val) = break_val {
                        let current_block = self.builder.current_block().unwrap();
                        loop_ctx.break_values.push((current_block, val));
                    }
                    let break_block = loop_ctx.break_block;
                    self.builder.build_branch(break_block);
                    self.terminated = true;
                }
                None
            }

            HirExprKind::Continue => {
                if let Some(loop_ctx) = self.loop_stack.last() {
                    let continue_block = loop_ctx.continue_block;
                    self.builder.build_branch(continue_block);
                    self.terminated = true;
                }
                None
            }

            HirExprKind::Tuple(elems) => {
                let vals: Vec<_> = elems.iter().filter_map(|e| self.lower_expr(e)).collect();
                Some(self.builder.build_tuple(vals, ty))
            }

            HirExprKind::Array(elems) => {
                let vals: Vec<_> = elems.iter().filter_map(|e| self.lower_expr(e)).collect();
                Some(self.builder.build_array(vals, ty))
            }

            HirExprKind::Struct { name, fields } => {
                let field_vals: Vec<_> = fields
                    .iter()
                    .filter_map(|(n, e)| self.lower_expr(e).map(|v| (n.clone(), v)))
                    .collect();
                Some(self.builder.build_struct(name, field_vals, ty))
            }

            HirExprKind::Field { base, field } => {
                let base_val = self.lower_expr(base)?;
                let field_idx = self.get_field_index(&base.ty, field);
                Some(self.builder.build_extract(base_val, field_idx, ty))
            }

            HirExprKind::TupleField { base, index } => {
                let base_val = self.lower_expr(base)?;
                Some(self.builder.build_extract(base_val, *index, ty))
            }

            HirExprKind::Index { base, index } => {
                // For arrays, we need pointer arithmetic
                let base_val = self.lower_expr(base)?;
                let idx_val = self.lower_expr(index)?;
                let elem_ptr = self.builder.build_elem_ptr(base_val, idx_val, ty.clone());
                Some(self.builder.build_load(elem_ptr, ty))
            }

            HirExprKind::Ref {
                mutable: _,
                expr: inner,
            } => {
                // Get address of inner expression
                self.lower_lvalue(inner)
            }

            HirExprKind::Deref(inner) => {
                let ptr = self.lower_expr(inner)?;
                Some(self.builder.build_load(ptr, ty))
            }

            HirExprKind::Cast {
                expr: inner,
                target,
            } => {
                let val = self.lower_expr(inner)?;
                let target_ty = HlirType::from_hir(target);
                Some(self.builder.build_cast(val, target_ty))
            }

            HirExprKind::Match { scrutinee, arms } => self.lower_match(scrutinee, arms, &ty),

            HirExprKind::Closure { params: _, body: _ } => {
                // Closures are complex - for now return unit
                Some(self.builder.build_unit())
            }

            HirExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Desugar to regular function call
                let mut all_args = vec![self.lower_expr(receiver)?];
                all_args.extend(args.iter().filter_map(|a| self.lower_expr(a)));
                Some(self.builder.build_call(method, all_args, ty))
            }

            HirExprKind::Variant {
                enum_name: _,
                variant: _,
                fields: _,
            } => {
                // Enum variants - simplified for now
                Some(self.builder.build_unit())
            }

            HirExprKind::Perform { effect, op, args } => {
                let arg_vals: Vec<_> = args.iter().filter_map(|a| self.lower_expr(a)).collect();
                let op = Op::PerformEffect {
                    effect: effect.clone(),
                    op: op.clone(),
                    args: arg_vals,
                };
                let result = self.builder.fresh_value();
                // Emit directly since we need custom Op
                Some(result)
            }

            HirExprKind::Handle {
                expr: _,
                handler: _,
            } => {
                // Effect handlers - complex, simplified for now
                Some(self.builder.build_unit())
            }

            HirExprKind::Sample(_) => {
                // Probabilistic sampling - simplified for now
                Some(self.builder.build_unit())
            }
        }
    }

    fn lower_literal(&mut self, lit: &HirLiteral, ty: &HlirType) -> ValueId {
        match lit {
            HirLiteral::Unit => self.builder.build_unit(),
            HirLiteral::Bool(b) => self.builder.build_bool(*b),
            HirLiteral::Int(i) => self
                .builder
                .build_const(HlirConstant::Int(*i, ty.clone()), ty.clone()),
            HirLiteral::Float(f) => self
                .builder
                .build_const(HlirConstant::Float(*f, ty.clone()), ty.clone()),
            HirLiteral::Char(c) => self
                .builder
                .build_const(HlirConstant::Int(*c as i64, HlirType::U32), HlirType::U32),
            HirLiteral::String(s) => self
                .builder
                .build_const(HlirConstant::String(s.clone()), ty.clone()),
        }
    }

    fn lower_binary_op(
        &mut self,
        op: HirBinaryOp,
        left: ValueId,
        right: ValueId,
        operand_ty: &HlirType,
        result_ty: &HlirType,
    ) -> ValueId {
        let is_float = operand_ty.is_float();
        let is_signed = operand_ty.is_signed();

        let hlir_op = match op {
            HirBinaryOp::Add => {
                if is_float {
                    BinaryOp::FAdd
                } else {
                    BinaryOp::Add
                }
            }
            HirBinaryOp::Sub => {
                if is_float {
                    BinaryOp::FSub
                } else {
                    BinaryOp::Sub
                }
            }
            HirBinaryOp::Mul => {
                if is_float {
                    BinaryOp::FMul
                } else {
                    BinaryOp::Mul
                }
            }
            HirBinaryOp::Div => {
                if is_float {
                    BinaryOp::FDiv
                } else if is_signed {
                    BinaryOp::SDiv
                } else {
                    BinaryOp::UDiv
                }
            }
            HirBinaryOp::Rem => {
                if is_float {
                    BinaryOp::FRem
                } else if is_signed {
                    BinaryOp::SRem
                } else {
                    BinaryOp::URem
                }
            }
            HirBinaryOp::Eq => {
                if is_float {
                    BinaryOp::FOEq
                } else {
                    BinaryOp::Eq
                }
            }
            HirBinaryOp::Ne => {
                if is_float {
                    BinaryOp::FONe
                } else {
                    BinaryOp::Ne
                }
            }
            HirBinaryOp::Lt => {
                if is_float {
                    BinaryOp::FOLt
                } else if is_signed {
                    BinaryOp::SLt
                } else {
                    BinaryOp::ULt
                }
            }
            HirBinaryOp::Le => {
                if is_float {
                    BinaryOp::FOLe
                } else if is_signed {
                    BinaryOp::SLe
                } else {
                    BinaryOp::ULe
                }
            }
            HirBinaryOp::Gt => {
                if is_float {
                    BinaryOp::FOGt
                } else if is_signed {
                    BinaryOp::SGt
                } else {
                    BinaryOp::UGt
                }
            }
            HirBinaryOp::Ge => {
                if is_float {
                    BinaryOp::FOGe
                } else if is_signed {
                    BinaryOp::SGe
                } else {
                    BinaryOp::UGe
                }
            }
            HirBinaryOp::And => BinaryOp::And,
            HirBinaryOp::Or => BinaryOp::Or,
            HirBinaryOp::BitAnd => BinaryOp::And,
            HirBinaryOp::BitOr => BinaryOp::Or,
            HirBinaryOp::BitXor => BinaryOp::Xor,
            HirBinaryOp::Shl => BinaryOp::Shl,
            HirBinaryOp::Shr => {
                if is_signed {
                    BinaryOp::AShr
                } else {
                    BinaryOp::LShr
                }
            }
        };

        self.builder
            .build_binary(hlir_op, left, right, result_ty.clone())
    }

    fn lower_unary_op(
        &mut self,
        op: HirUnaryOp,
        operand: ValueId,
        operand_ty: &HlirType,
    ) -> ValueId {
        match op {
            HirUnaryOp::Neg => {
                if operand_ty.is_float() {
                    self.builder.build_fneg(operand, operand_ty.clone())
                } else {
                    self.builder.build_neg(operand, operand_ty.clone())
                }
            }
            HirUnaryOp::Not => self.builder.build_not(operand),
            HirUnaryOp::Ref | HirUnaryOp::RefMut => {
                // Address-of is handled in lower_lvalue
                operand
            }
            HirUnaryOp::Deref => {
                // Dereference is handled in lower_expr
                operand
            }
        }
    }

    fn lower_if(
        &mut self,
        condition: &HirExpr,
        then_branch: &HirBlock,
        else_branch: Option<&HirExpr>,
        ty: &HlirType,
    ) -> Option<ValueId> {
        let cond_val = self.lower_expr(condition)?;

        let then_block = self.builder.create_block("if.then");
        let else_block = self.builder.create_block("if.else");
        let merge_block = self.builder.create_block("if.merge");

        self.builder
            .build_cond_branch(cond_val, then_block, else_block);

        // Then branch
        self.builder.switch_to_block(then_block);
        self.terminated = false;
        let then_val = self.lower_block(then_branch);
        let then_terminated = self.terminated;
        let then_exit_block = self.builder.current_block().unwrap();
        if !then_terminated {
            self.builder.build_branch(merge_block);
        }

        // Else branch
        self.builder.switch_to_block(else_block);
        self.terminated = false;
        let else_val = if let Some(else_expr) = else_branch {
            self.lower_expr(else_expr)
        } else {
            Some(self.builder.build_unit())
        };
        let else_terminated = self.terminated;
        let else_exit_block = self.builder.current_block().unwrap();
        if !else_terminated {
            self.builder.build_branch(merge_block);
        }

        // Merge block
        self.builder.switch_to_block(merge_block);
        self.terminated = then_terminated && else_terminated;

        if self.terminated {
            return None;
        }

        // Build phi if both branches produce values
        if *ty != HlirType::Void {
            if let (Some(tv), Some(ev)) = (then_val, else_val) {
                let mut incoming = Vec::new();
                if !then_terminated {
                    incoming.push((then_exit_block, tv));
                }
                if !else_terminated {
                    incoming.push((else_exit_block, ev));
                }
                if !incoming.is_empty() {
                    return Some(self.builder.build_phi(incoming, ty.clone()));
                }
            }
        }

        None
    }

    fn lower_loop(&mut self, body: &HirBlock, ty: &HlirType) -> Option<ValueId> {
        let loop_block = self.builder.create_block("loop.body");
        let exit_block = self.builder.create_block("loop.exit");

        // Jump to loop
        self.builder.build_branch(loop_block);

        // Push loop context
        self.loop_stack.push(LoopContext {
            continue_block: loop_block,
            break_block: exit_block,
            break_values: Vec::new(),
        });

        // Loop body
        self.builder.switch_to_block(loop_block);
        self.terminated = false;
        self.lower_block(body);

        // If body didn't terminate, loop back
        if !self.terminated {
            self.builder.build_branch(loop_block);
        }

        // Pop loop context and collect break values
        let loop_ctx = self.loop_stack.pop().unwrap();

        // Exit block
        self.builder.switch_to_block(exit_block);
        self.terminated = false;

        // Build phi for break values
        if *ty != HlirType::Void && !loop_ctx.break_values.is_empty() {
            Some(self.builder.build_phi(loop_ctx.break_values, ty.clone()))
        } else {
            None
        }
    }

    fn lower_match(
        &mut self,
        scrutinee: &HirExpr,
        arms: &[HirMatchArm],
        ty: &HlirType,
    ) -> Option<ValueId> {
        let scrut_val = self.lower_expr(scrutinee)?;
        let scrut_ty = HlirType::from_hir(&scrutinee.ty);

        // For simple integer matches, use switch
        if scrut_ty.is_integer()
            && arms
                .iter()
                .all(|a| matches!(a.pattern, HirPattern::Literal(_)))
        {
            return self.lower_match_switch(scrut_val, arms, ty);
        }

        // General case: chain of if-else
        self.lower_match_chain(scrut_val, &scrut_ty, arms, ty)
    }

    fn lower_match_switch(
        &mut self,
        scrut: ValueId,
        arms: &[HirMatchArm],
        ty: &HlirType,
    ) -> Option<ValueId> {
        let merge_block = self.builder.create_block("match.merge");
        let default_block = self.builder.create_block("match.default");

        let mut cases = Vec::new();
        let mut arm_results = Vec::new();

        for arm in arms {
            if let HirPattern::Literal(HirLiteral::Int(n)) = &arm.pattern {
                let arm_block = self.builder.create_block("match.arm");
                cases.push((*n, arm_block));

                self.builder.switch_to_block(arm_block);
                self.terminated = false;
                let result = self.lower_expr(&arm.body);
                let arm_exit = self.builder.current_block().unwrap();
                if !self.terminated {
                    self.builder.build_branch(merge_block);
                    if let Some(v) = result {
                        arm_results.push((arm_exit, v));
                    }
                }
            }
        }

        // Build switch from entry
        // We need to go back and add the switch - this is a limitation
        // For now, just branch to default
        self.builder.switch_to_block(default_block);
        self.builder.build_unreachable();

        self.builder.switch_to_block(merge_block);
        self.terminated = false;

        if *ty != HlirType::Void && !arm_results.is_empty() {
            Some(self.builder.build_phi(arm_results, ty.clone()))
        } else {
            None
        }
    }

    fn lower_match_chain(
        &mut self,
        scrut: ValueId,
        scrut_ty: &HlirType,
        arms: &[HirMatchArm],
        ty: &HlirType,
    ) -> Option<ValueId> {
        if arms.is_empty() {
            return None;
        }

        let merge_block = self.builder.create_block("match.merge");
        let mut arm_results = Vec::new();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;
            let next_block = if is_last {
                merge_block
            } else {
                self.builder.create_block(&format!("match.test.{}", i + 1))
            };

            let arm_block = self.builder.create_block(&format!("match.arm.{}", i));

            // Check pattern
            let matches = self.lower_pattern_check(&arm.pattern, scrut, scrut_ty);

            if let Some(cond) = matches {
                self.builder.build_cond_branch(cond, arm_block, next_block);
            } else {
                // Wildcard or binding - always matches
                self.builder.build_branch(arm_block);
            }

            // Arm body
            self.builder.switch_to_block(arm_block);
            self.terminated = false;

            // Bind pattern variables
            self.bind_pattern(&arm.pattern, scrut);

            let result = self.lower_expr(&arm.body);
            let arm_exit = self.builder.current_block().unwrap();

            if !self.terminated {
                self.builder.build_branch(merge_block);
                if let Some(v) = result {
                    arm_results.push((arm_exit, v));
                }
            }

            if !is_last {
                self.builder.switch_to_block(next_block);
            }
        }

        self.builder.switch_to_block(merge_block);
        self.terminated = false;

        if *ty != HlirType::Void && !arm_results.is_empty() {
            Some(self.builder.build_phi(arm_results, ty.clone()))
        } else {
            None
        }
    }

    fn lower_pattern_check(
        &mut self,
        pattern: &HirPattern,
        scrut: ValueId,
        scrut_ty: &HlirType,
    ) -> Option<ValueId> {
        match pattern {
            HirPattern::Wildcard => None,
            HirPattern::Binding { .. } => None,
            HirPattern::Literal(lit) => {
                let lit_val = self.lower_literal(lit, scrut_ty);
                Some(self.builder.build_eq(scrut, lit_val))
            }
            HirPattern::Tuple(patterns) => {
                // Check all tuple elements
                let mut combined: Option<ValueId> = None;
                for (i, p) in patterns.iter().enumerate() {
                    let elem_ty = if let HlirType::Tuple(elems) = scrut_ty {
                        elems.get(i).cloned().unwrap_or(HlirType::Void)
                    } else {
                        HlirType::Void
                    };
                    let elem = self.builder.build_extract(scrut, i, elem_ty.clone());
                    if let Some(check) = self.lower_pattern_check(p, elem, &elem_ty) {
                        combined = Some(match combined {
                            Some(prev) => self.builder.build_binary(
                                BinaryOp::And,
                                prev,
                                check,
                                HlirType::Bool,
                            ),
                            None => check,
                        });
                    }
                }
                combined
            }
            HirPattern::Or(patterns) => {
                // Check if any pattern matches
                let mut combined: Option<ValueId> = None;
                for p in patterns {
                    if let Some(check) = self.lower_pattern_check(p, scrut, scrut_ty) {
                        combined = Some(match combined {
                            Some(prev) => {
                                self.builder
                                    .build_binary(BinaryOp::Or, prev, check, HlirType::Bool)
                            }
                            None => check,
                        });
                    }
                }
                combined
            }
            HirPattern::Struct { .. } | HirPattern::Variant { .. } => {
                // Complex patterns - simplified
                None
            }
        }
    }

    fn bind_pattern(&mut self, pattern: &HirPattern, value: ValueId) {
        match pattern {
            HirPattern::Binding { name, .. } => {
                self.builder.bind_var(name, value);
            }
            HirPattern::Tuple(patterns) => {
                for (i, p) in patterns.iter().enumerate() {
                    let elem = self.builder.build_extract(value, i, HlirType::Void);
                    self.bind_pattern(p, elem);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_hir() -> Hir {
        use crate::common::NodeId;

        Hir {
            items: vec![HirItem::Function(HirFn {
                id: NodeId(0),
                name: "add".to_string(),
                ty: HirFnType {
                    params: vec![
                        HirParam {
                            id: NodeId(1),
                            name: "a".to_string(),
                            ty: HirType::I64,
                            is_mut: false,
                        },
                        HirParam {
                            id: NodeId(2),
                            name: "b".to_string(),
                            ty: HirType::I64,
                            is_mut: false,
                        },
                    ],
                    return_type: Box::new(HirType::I64),
                    effects: Vec::new(),
                },
                body: HirBlock {
                    stmts: vec![HirStmt::Expr(HirExpr {
                        id: NodeId(3),
                        kind: HirExprKind::Binary {
                            op: HirBinaryOp::Add,
                            left: Box::new(HirExpr {
                                id: NodeId(4),
                                kind: HirExprKind::Local("a".to_string()),
                                ty: HirType::I64,
                            }),
                            right: Box::new(HirExpr {
                                id: NodeId(5),
                                kind: HirExprKind::Local("b".to_string()),
                                ty: HirType::I64,
                            }),
                        },
                        ty: HirType::I64,
                    })],
                    ty: HirType::I64,
                },
            })],
        }
    }

    #[test]
    fn test_lower_simple_function() {
        let hir = make_simple_hir();
        let hlir = lower(&hir);

        assert_eq!(hlir.functions.len(), 1);
        let func = &hlir.functions[0];
        assert_eq!(func.name, "add");
        assert_eq!(func.params.len(), 2);
        assert!(!func.blocks.is_empty());
    }
}
