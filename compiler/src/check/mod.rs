//! Type checker for Demetrios
//!
//! This module implements type checking and produces HIR from the AST.
//! It handles:
//! - Type inference (bidirectional)
//! - Name resolution
//! - Effect checking
//! - Ownership/borrow checking
//! - Unit checking

use crate::ast::*;
use crate::common::{NodeId, Span};
use crate::hir::*;
use crate::types::{self, Type, TypeVar, effects::EffectInference, units::UnitChecker};
use miette::Result;
use std::collections::HashMap;

/// Type check an AST and produce HIR
pub fn check(ast: &Ast) -> Result<Hir> {
    let mut checker = TypeChecker::new();
    checker.check_program(ast)
}

/// Type checker state
pub struct TypeChecker {
    /// Type environment (variable -> type)
    env: TypeEnv,
    /// Type definitions
    type_defs: HashMap<String, TypeDef>,
    /// Effect inference context
    effects: EffectInference,
    /// Unit checker
    units: UnitChecker,
    /// Fresh type variable counter
    next_type_var: u32,
    /// Type constraints for unification
    constraints: Vec<TypeConstraint>,
    /// Errors accumulated during checking
    errors: Vec<TypeError>,
}

/// Type environment with scopes
#[derive(Default)]
pub struct TypeEnv {
    scopes: Vec<Scope>,
}

#[derive(Default)]
struct Scope {
    bindings: HashMap<String, TypeBinding>,
}

/// Binding in environment
#[derive(Clone)]
struct TypeBinding {
    ty: Type,
    mutable: bool,
    used: bool,
}

/// Type definition (struct, enum, type alias)
#[derive(Clone)]
enum TypeDef {
    Struct {
        fields: Vec<(String, Type)>,
        linear: bool,
        affine: bool,
    },
    Enum {
        variants: Vec<(String, Vec<Type>)>,
        linear: bool,
        affine: bool,
    },
    Alias(Type),
}

/// Type constraint for unification
#[derive(Debug)]
struct TypeConstraint {
    expected: Type,
    actual: Type,
    span: Span,
}

/// Type error
#[derive(Debug)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::default(),
            type_defs: HashMap::new(),
            effects: EffectInference::new(),
            units: UnitChecker::new(),
            next_type_var: 0,
            constraints: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Generate a fresh type variable
    fn fresh_type_var(&mut self) -> Type {
        let var = TypeVar(self.next_type_var);
        self.next_type_var += 1;
        Type::Var(var)
    }

    /// Add a type constraint
    fn constrain(&mut self, expected: Type, actual: Type, span: Span) {
        self.constraints.push(TypeConstraint {
            expected,
            actual,
            span,
        });
    }

    /// Report a type error
    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.errors.push(TypeError {
            message: message.into(),
            span,
        });
    }

    pub fn check_program(&mut self, ast: &Ast) -> Result<Hir> {
        let mut items = Vec::new();

        // First pass: collect type definitions
        for item in &ast.items {
            self.collect_type_def(item);
        }

        // Second pass: register function signatures in environment
        self.env.push_scope();
        for item in &ast.items {
            if let Item::Function(f) = item {
                let params: Vec<Type> = f
                    .params
                    .iter()
                    .map(|p| self.lower_type_expr(&p.ty))
                    .collect();
                let return_type = f
                    .return_type
                    .as_ref()
                    .map(|t| self.lower_type_expr(t))
                    .unwrap_or(Type::Unit);
                let fn_type = Type::Function {
                    params,
                    return_type: Box::new(return_type),
                    effects: types::EffectSet::new(),
                };
                self.env.bind(f.name.clone(), fn_type, false);
            }
        }

        // Third pass: type check items
        for item in &ast.items {
            if let Some(hir_item) = self.check_item(item)? {
                items.push(hir_item);
            }
        }

        self.env.pop_scope();

        // Solve type constraints
        self.solve_constraints()?;

        if !self.errors.is_empty() {
            let messages: Vec<_> = self.errors.iter().map(|e| e.message.clone()).collect();
            return Err(miette::miette!("Type errors:\n{}", messages.join("\n")));
        }

        Ok(Hir { items })
    }

    fn collect_type_def(&mut self, item: &Item) {
        match item {
            Item::Struct(s) => {
                let fields: Vec<_> = s
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), self.lower_type_expr(&f.ty)))
                    .collect();
                self.type_defs.insert(
                    s.name.clone(),
                    TypeDef::Struct {
                        fields,
                        linear: s.modifiers.linear,
                        affine: s.modifiers.affine,
                    },
                );
            }
            Item::Enum(e) => {
                let variants: Vec<_> = e
                    .variants
                    .iter()
                    .map(|v| {
                        let types = match &v.data {
                            VariantData::Unit => Vec::new(),
                            VariantData::Tuple(types) => {
                                types.iter().map(|t| self.lower_type_expr(t)).collect()
                            }
                            VariantData::Struct(fields) => {
                                fields.iter().map(|f| self.lower_type_expr(&f.ty)).collect()
                            }
                        };
                        (v.name.clone(), types)
                    })
                    .collect();
                self.type_defs.insert(
                    e.name.clone(),
                    TypeDef::Enum {
                        variants,
                        linear: e.modifiers.linear,
                        affine: e.modifiers.affine,
                    },
                );
            }
            Item::TypeAlias(t) => {
                let ty = self.lower_type_expr(&t.ty);
                self.type_defs.insert(t.name.clone(), TypeDef::Alias(ty));
            }
            _ => {}
        }
    }

    fn check_item(&mut self, item: &Item) -> Result<Option<HirItem>> {
        match item {
            Item::Function(f) => {
                let hir_fn = self.check_function(f)?;
                Ok(Some(HirItem::Function(hir_fn)))
            }
            Item::Struct(s) => {
                let hir_struct = self.check_struct(s)?;
                Ok(Some(HirItem::Struct(hir_struct)))
            }
            Item::Enum(e) => {
                let hir_enum = self.check_enum(e)?;
                Ok(Some(HirItem::Enum(hir_enum)))
            }
            Item::Effect(e) => {
                let hir_effect = self.check_effect_def(e)?;
                Ok(Some(HirItem::Effect(hir_effect)))
            }
            Item::Handler(h) => {
                let hir_handler = self.check_handler_def(h)?;
                Ok(Some(HirItem::Handler(hir_handler)))
            }
            Item::Global(g) => {
                let hir_global = self.check_global(g)?;
                Ok(Some(HirItem::Global(hir_global)))
            }
            _ => Ok(None),
        }
    }

    fn check_function(&mut self, f: &FnDef) -> Result<HirFn> {
        self.env.push_scope();

        // Process parameters
        let mut params = Vec::new();
        for param in &f.params {
            let ty = self.lower_type_expr(&param.ty);
            let hir_ty = self.type_to_hir(&ty);

            // Bind parameter in environment
            if let Pattern::Binding { name, .. } = &param.pattern {
                self.env.bind(name.clone(), ty.clone(), param.is_mut);
            }

            params.push(HirParam {
                id: param.id,
                name: self.pattern_name(&param.pattern),
                ty: hir_ty,
                is_mut: param.is_mut,
            });
        }

        // Process return type
        let return_type = f
            .return_type
            .as_ref()
            .map(|t| self.lower_type_expr(t))
            .unwrap_or(Type::Unit);

        // Check body
        let body = self.check_block(&f.body, Some(&return_type))?;

        self.env.pop_scope();

        Ok(HirFn {
            id: f.id,
            name: f.name.clone(),
            ty: HirFnType {
                params: params.clone(),
                return_type: Box::new(self.type_to_hir(&return_type)),
                effects: Vec::new(), // TODO: convert effects
            },
            body,
        })
    }

    fn check_struct(&mut self, s: &StructDef) -> Result<HirStruct> {
        let fields: Vec<_> = s
            .fields
            .iter()
            .map(|f| {
                let ty = self.lower_type_expr(&f.ty);
                HirField {
                    id: f.id,
                    name: f.name.clone(),
                    ty: self.type_to_hir(&ty),
                }
            })
            .collect();

        Ok(HirStruct {
            id: s.id,
            name: s.name.clone(),
            fields,
            is_linear: s.modifiers.linear,
            is_affine: s.modifiers.affine,
        })
    }

    fn check_enum(&mut self, e: &EnumDef) -> Result<HirEnum> {
        let variants: Vec<_> = e
            .variants
            .iter()
            .map(|v| {
                let fields = match &v.data {
                    VariantData::Unit => Vec::new(),
                    VariantData::Tuple(types) => types
                        .iter()
                        .map(|t| self.type_to_hir(&self.lower_type_expr(t)))
                        .collect(),
                    VariantData::Struct(fields) => fields
                        .iter()
                        .map(|f| self.type_to_hir(&self.lower_type_expr(&f.ty)))
                        .collect(),
                };
                HirVariant {
                    id: v.id,
                    name: v.name.clone(),
                    fields,
                }
            })
            .collect();

        Ok(HirEnum {
            id: e.id,
            name: e.name.clone(),
            variants,
            is_linear: e.modifiers.linear,
            is_affine: e.modifiers.affine,
        })
    }

    fn check_effect_def(&mut self, e: &EffectDef) -> Result<HirEffect> {
        let operations: Vec<_> = e
            .operations
            .iter()
            .map(|op| {
                let params: Vec<_> = op
                    .params
                    .iter()
                    .map(|p| self.type_to_hir(&self.lower_type_expr(&p.ty)))
                    .collect();
                let return_type = op
                    .return_type
                    .as_ref()
                    .map(|t| self.type_to_hir(&self.lower_type_expr(t)))
                    .unwrap_or(HirType::Unit);

                HirEffectOp {
                    id: op.id,
                    name: op.name.clone(),
                    params,
                    return_type,
                }
            })
            .collect();

        Ok(HirEffect {
            id: e.id,
            name: e.name.clone(),
            operations,
        })
    }

    fn check_handler_def(&mut self, h: &HandlerDef) -> Result<HirHandler> {
        let cases: Vec<_> = h
            .cases
            .iter()
            .map(|case| {
                let params: Vec<_> = case
                    .params
                    .iter()
                    .map(|p| self.pattern_name(&p.pattern))
                    .collect();

                // TODO: properly check handler case body
                HirHandlerCase {
                    id: case.id,
                    op_name: case.name.clone(),
                    params,
                    body: HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Literal(HirLiteral::Unit),
                        ty: HirType::Unit,
                    },
                }
            })
            .collect();

        Ok(HirHandler {
            id: h.id,
            name: h.name.clone(),
            effect: h.effect.to_string(),
            cases,
        })
    }

    fn check_global(&mut self, g: &GlobalDef) -> Result<HirGlobal> {
        let ty =
            g.ty.as_ref()
                .map(|t| self.lower_type_expr(t))
                .unwrap_or_else(|| self.fresh_type_var());

        // TODO: properly check global value expression
        let value = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Literal(HirLiteral::Unit),
            ty: self.type_to_hir(&ty),
        };

        Ok(HirGlobal {
            id: g.id,
            name: self.pattern_name(&g.pattern),
            ty: self.type_to_hir(&ty),
            value,
            is_const: g.is_const,
        })
    }

    fn check_block(&mut self, block: &Block, expected: Option<&Type>) -> Result<HirBlock> {
        self.env.push_scope();

        let mut stmts = Vec::new();
        let mut result_ty = Type::Unit;

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == block.stmts.len() - 1;

            match stmt {
                Stmt::Let {
                    is_mut,
                    pattern,
                    ty,
                    value,
                } => {
                    let declared_ty = ty
                        .as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or_else(|| self.fresh_type_var());

                    let value_expr = value
                        .as_ref()
                        .map(|v| self.check_expr(v, Some(&declared_ty)))
                        .transpose()?;

                    if let Pattern::Binding { name, .. } = pattern {
                        self.env.bind(name.clone(), declared_ty.clone(), *is_mut);
                    }

                    stmts.push(HirStmt::Let {
                        name: self.pattern_name(pattern),
                        ty: self.type_to_hir(&declared_ty),
                        value: value_expr,
                        is_mut: *is_mut,
                    });
                }
                Stmt::Expr { expr, has_semi } => {
                    let expr_result = self.check_expr(expr, None)?;

                    if is_last && !has_semi {
                        result_ty = self.hir_type_to_type(&expr_result.ty);
                    }

                    stmts.push(HirStmt::Expr(expr_result));
                }
                Stmt::Assign { target, op, value } => {
                    let target_expr = self.check_expr(target, None)?;
                    let value_expr =
                        self.check_expr(value, Some(&self.hir_type_to_type(&target_expr.ty)))?;

                    stmts.push(HirStmt::Assign {
                        target: target_expr,
                        value: value_expr,
                    });
                }
                Stmt::Empty => {}
            }
        }

        if let Some(exp) = expected {
            self.constrain(exp.clone(), result_ty.clone(), Span::dummy());
        }

        self.env.pop_scope();

        Ok(HirBlock {
            stmts,
            ty: self.type_to_hir(&result_ty),
        })
    }

    fn check_expr(&mut self, expr: &Expr, expected: Option<&Type>) -> Result<HirExpr> {
        let (kind, ty) = match expr {
            Expr::Literal { id, value } => {
                let (lit, ty) = self.check_literal(value);
                (HirExprKind::Literal(lit), ty)
            }

            Expr::Path { id, path } => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    if let Some(binding) = self.env.lookup(name) {
                        let ty = binding.ty.clone();
                        (HirExprKind::Local(name.clone()), self.type_to_hir(&ty))
                    } else {
                        self.error(format!("Unknown variable: {}", name), Span::dummy());
                        (HirExprKind::Local(name.clone()), HirType::Error)
                    }
                } else {
                    // Qualified path - could be enum variant, module path, etc.
                    (
                        HirExprKind::Global(path.to_string()),
                        HirType::Error, // TODO: proper resolution
                    )
                }
            }

            Expr::Binary {
                id,
                op,
                left,
                right,
            } => {
                let left_expr = self.check_expr(left, None)?;
                let right_expr =
                    self.check_expr(right, Some(&self.hir_type_to_type(&left_expr.ty)))?;

                let result_ty = self.binary_result_type(*op, &left_expr.ty, &right_expr.ty);
                let hir_op = self.lower_binary_op(*op);

                (
                    HirExprKind::Binary {
                        op: hir_op,
                        left: Box::new(left_expr),
                        right: Box::new(right_expr),
                    },
                    result_ty,
                )
            }

            Expr::Unary {
                id,
                op,
                expr: inner,
            } => {
                let inner_expr = self.check_expr(inner, None)?;
                let result_ty = self.unary_result_type(*op, &inner_expr.ty);
                let hir_op = self.lower_unary_op(*op);

                (
                    HirExprKind::Unary {
                        op: hir_op,
                        expr: Box::new(inner_expr),
                    },
                    result_ty,
                )
            }

            Expr::Call { id, callee, args } => {
                let callee_expr = self.check_expr(callee, None)?;
                let checked_args: Vec<_> = args
                    .iter()
                    .map(|a| self.check_expr(a, None))
                    .collect::<Result<_>>()?;

                // Extract return type from function type
                let result_ty = match &callee_expr.ty {
                    HirType::Fn { return_type, .. } => *return_type.clone(),
                    _ => HirType::Unit,
                };

                (
                    HirExprKind::Call {
                        func: Box::new(callee_expr),
                        args: checked_args,
                    },
                    result_ty,
                )
            }

            Expr::If {
                id,
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_expr = self.check_expr(condition, Some(&Type::Bool))?;
                let then_block = self.check_block(then_branch, expected)?;

                let else_expr = else_branch
                    .as_ref()
                    .map(|e| self.check_expr(e, expected))
                    .transpose()?;

                let result_ty = if else_expr.is_some() {
                    then_block.ty.clone()
                } else {
                    HirType::Unit
                };

                (
                    HirExprKind::If {
                        condition: Box::new(cond_expr),
                        then_branch: then_block,
                        else_branch: else_expr.map(Box::new),
                    },
                    result_ty,
                )
            }

            Expr::Block { id, block } => {
                let hir_block = self.check_block(block, expected)?;
                let ty = hir_block.ty.clone();
                (HirExprKind::Block(hir_block), ty)
            }

            Expr::Return { id, value } => {
                let val = value
                    .as_ref()
                    .map(|v| self.check_expr(v, expected))
                    .transpose()?;

                // Return has Never type since control doesn't continue
                (HirExprKind::Return(val.map(Box::new)), HirType::Never)
            }

            Expr::Tuple { id, elements } => {
                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, None))
                    .collect::<Result<_>>()?;

                let tys: Vec<_> = exprs.iter().map(|e| e.ty.clone()).collect();
                let result_ty = HirType::Tuple(tys);

                (HirExprKind::Tuple(exprs), result_ty)
            }

            Expr::Array { id, elements } => {
                let elem_ty = expected
                    .and_then(|t| {
                        if let Type::Array { element, .. } = t {
                            Some(element.as_ref().clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| self.fresh_type_var());

                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, Some(&elem_ty)))
                    .collect::<Result<_>>()?;

                let elem_hir_ty = if exprs.is_empty() {
                    self.type_to_hir(&elem_ty)
                } else {
                    exprs[0].ty.clone()
                };

                let result_ty = HirType::Array {
                    element: Box::new(elem_hir_ty),
                    size: Some(exprs.len()),
                };

                (HirExprKind::Array(exprs), result_ty)
            }

            Expr::Index { id, base, index } => {
                let base_expr = self.check_expr(base, None)?;
                let index_expr = self.check_expr(index, Some(&Type::I64))?;

                // Extract element type from array type
                let elem_ty = match &base_expr.ty {
                    HirType::Array { element, .. } => *element.clone(),
                    HirType::String => HirType::Char,
                    _ => HirType::Error,
                };

                (
                    HirExprKind::Index {
                        base: Box::new(base_expr),
                        index: Box::new(index_expr),
                    },
                    elem_ty,
                )
            }

            Expr::Field { id, base, field } => {
                let base_expr = self.check_expr(base, None)?;

                // Look up field type from struct definition
                let field_ty = if let HirType::Named { name, .. } = &base_expr.ty {
                    if let Some(TypeDef::Struct { fields, .. }) = self.type_defs.get(name) {
                        fields
                            .iter()
                            .find(|(n, _)| n == field)
                            .map(|(_, t)| self.type_to_hir(t))
                            .unwrap_or(HirType::Error)
                    } else {
                        HirType::Error
                    }
                } else {
                    HirType::Error
                };

                (
                    HirExprKind::Field {
                        base: Box::new(base_expr),
                        field: field.clone(),
                    },
                    field_ty,
                )
            }

            Expr::TupleField { id, base, index } => {
                let base_expr = self.check_expr(base, None)?;

                // Extract element type from tuple type
                let elem_ty = match &base_expr.ty {
                    HirType::Tuple(elements) => {
                        elements.get(*index).cloned().unwrap_or(HirType::Error)
                    }
                    _ => HirType::Error,
                };

                (
                    HirExprKind::TupleField {
                        base: Box::new(base_expr),
                        index: *index,
                    },
                    elem_ty,
                )
            }

            Expr::StructLit { id, path, fields } => {
                let struct_name = path.segments.last().cloned().unwrap_or_default();
                let checked_fields: Vec<_> = fields
                    .iter()
                    .map(|(name, expr)| {
                        let expr = self.check_expr(expr, None)?;
                        Ok((name.clone(), expr))
                    })
                    .collect::<Result<_>>()?;

                (
                    HirExprKind::Struct {
                        name: struct_name.clone(),
                        fields: checked_fields,
                    },
                    HirType::Named {
                        name: struct_name,
                        args: vec![],
                    },
                )
            }

            Expr::Loop { id, body } => {
                let body_block = self.check_block(body, None)?;
                (HirExprKind::Loop(body_block), HirType::Unit)
            }

            Expr::While {
                id,
                condition,
                body,
            } => {
                let cond_expr = self.check_expr(condition, Some(&Type::Bool))?;
                let body_block = self.check_block(body, None)?;

                // Desugar while to loop with if/break
                (
                    HirExprKind::Loop(HirBlock {
                        stmts: vec![
                            HirStmt::Expr(HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::If {
                                    condition: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Unary {
                                            op: HirUnaryOp::Not,
                                            expr: Box::new(cond_expr),
                                        },
                                        ty: HirType::Bool,
                                    }),
                                    then_branch: HirBlock {
                                        stmts: vec![HirStmt::Expr(HirExpr {
                                            id: NodeId::dummy(),
                                            kind: HirExprKind::Break(None),
                                            ty: HirType::Never,
                                        })],
                                        ty: HirType::Never,
                                    },
                                    else_branch: None,
                                },
                                ty: HirType::Unit,
                            }),
                            HirStmt::Expr(HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Block(body_block),
                                ty: HirType::Unit,
                            }),
                        ],
                        ty: HirType::Unit,
                    }),
                    HirType::Unit,
                )
            }

            Expr::Break { id, value } => {
                let val = value
                    .as_ref()
                    .map(|v| self.check_expr(v, None))
                    .transpose()?;
                (HirExprKind::Break(val.map(Box::new)), HirType::Never)
            }

            Expr::Continue { id } => (HirExprKind::Continue, HirType::Never),

            // Simplified handling for other expressions
            _ => {
                // For now, return a placeholder
                (HirExprKind::Literal(HirLiteral::Unit), HirType::Unit)
            }
        };

        let id = match expr {
            Expr::Literal { id, .. }
            | Expr::Path { id, .. }
            | Expr::Binary { id, .. }
            | Expr::Unary { id, .. }
            | Expr::Call { id, .. }
            | Expr::If { id, .. }
            | Expr::Block { id, .. }
            | Expr::Return { id, .. }
            | Expr::Tuple { id, .. }
            | Expr::Array { id, .. } => *id,
            _ => NodeId::dummy(),
        };

        Ok(HirExpr { id, kind, ty })
    }

    fn check_literal(&self, lit: &Literal) -> (HirLiteral, HirType) {
        match lit {
            Literal::Unit => (HirLiteral::Unit, HirType::Unit),
            Literal::Bool(b) => (HirLiteral::Bool(*b), HirType::Bool),
            Literal::Int(i) => (HirLiteral::Int(*i), HirType::I64),
            Literal::Float(f) => (HirLiteral::Float(*f), HirType::F64),
            Literal::Char(c) => (HirLiteral::Char(*c), HirType::Char),
            Literal::String(s) => (HirLiteral::String(s.clone()), HirType::String),
        }
    }

    fn binary_result_type(&self, op: BinaryOp, left: &HirType, right: &HirType) -> HirType {
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                left.clone()
            }
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::And
            | BinaryOp::Or => HirType::Bool,
            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => left.clone(),
        }
    }

    fn unary_result_type(&self, op: UnaryOp, operand: &HirType) -> HirType {
        match op {
            UnaryOp::Neg => operand.clone(),
            UnaryOp::Not => {
                if *operand == HirType::Bool {
                    HirType::Bool
                } else {
                    operand.clone()
                }
            }
            UnaryOp::Ref => HirType::Ref {
                mutable: false,
                inner: Box::new(operand.clone()),
            },
            UnaryOp::RefMut => HirType::Ref {
                mutable: true,
                inner: Box::new(operand.clone()),
            },
            UnaryOp::Deref => {
                if let HirType::Ref { inner, .. } = operand {
                    *inner.clone()
                } else {
                    HirType::Error
                }
            }
        }
    }

    fn lower_binary_op(&self, op: BinaryOp) -> HirBinaryOp {
        match op {
            BinaryOp::Add => HirBinaryOp::Add,
            BinaryOp::Sub => HirBinaryOp::Sub,
            BinaryOp::Mul => HirBinaryOp::Mul,
            BinaryOp::Div => HirBinaryOp::Div,
            BinaryOp::Rem => HirBinaryOp::Rem,
            BinaryOp::Eq => HirBinaryOp::Eq,
            BinaryOp::Ne => HirBinaryOp::Ne,
            BinaryOp::Lt => HirBinaryOp::Lt,
            BinaryOp::Le => HirBinaryOp::Le,
            BinaryOp::Gt => HirBinaryOp::Gt,
            BinaryOp::Ge => HirBinaryOp::Ge,
            BinaryOp::And => HirBinaryOp::And,
            BinaryOp::Or => HirBinaryOp::Or,
            BinaryOp::BitAnd => HirBinaryOp::BitAnd,
            BinaryOp::BitOr => HirBinaryOp::BitOr,
            BinaryOp::BitXor => HirBinaryOp::BitXor,
            BinaryOp::Shl => HirBinaryOp::Shl,
            BinaryOp::Shr => HirBinaryOp::Shr,
        }
    }

    fn lower_unary_op(&self, op: UnaryOp) -> HirUnaryOp {
        match op {
            UnaryOp::Neg => HirUnaryOp::Neg,
            UnaryOp::Not => HirUnaryOp::Not,
            UnaryOp::Ref => HirUnaryOp::Ref,
            UnaryOp::RefMut => HirUnaryOp::RefMut,
            UnaryOp::Deref => HirUnaryOp::Deref,
        }
    }

    fn lower_type_expr(&self, ty: &TypeExpr) -> Type {
        match ty {
            TypeExpr::Unit => Type::Unit,
            TypeExpr::Named { path, args, .. } => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    match name.as_str() {
                        "bool" => Type::Bool,
                        "i8" => Type::I8,
                        "i16" => Type::I16,
                        "i32" => Type::I32,
                        "i64" => Type::I64,
                        "i128" => Type::I128,
                        "isize" => Type::Isize,
                        "u8" => Type::U8,
                        "u16" => Type::U16,
                        "u32" => Type::U32,
                        "u64" => Type::U64,
                        "u128" => Type::U128,
                        "usize" => Type::Usize,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "char" => Type::Char,
                        "str" => Type::Str,
                        "String" => Type::String,
                        _ => Type::Named {
                            name: name.clone(),
                            args: args.iter().map(|a| self.lower_type_expr(a)).collect(),
                        },
                    }
                } else {
                    Type::Named {
                        name: path.to_string(),
                        args: args.iter().map(|a| self.lower_type_expr(a)).collect(),
                    }
                }
            }
            TypeExpr::Reference { mutable, inner } => Type::Ref {
                mutable: *mutable,
                lifetime: None,
                inner: Box::new(self.lower_type_expr(inner)),
            },
            TypeExpr::Array { element, size } => Type::Array {
                element: Box::new(self.lower_type_expr(element)),
                size: None, // TODO: evaluate const expression
            },
            TypeExpr::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.lower_type_expr(e)).collect())
            }
            TypeExpr::Function {
                params,
                return_type,
                ..
            } => Type::Function {
                params: params.iter().map(|p| self.lower_type_expr(p)).collect(),
                return_type: Box::new(self.lower_type_expr(return_type)),
                effects: types::EffectSet::new(),
            },
            TypeExpr::Infer => Type::Unknown,
            TypeExpr::SelfType => Type::SelfType,
        }
    }

    fn type_to_hir(&self, ty: &Type) -> HirType {
        match ty {
            Type::Unit => HirType::Unit,
            Type::Bool => HirType::Bool,
            Type::I8 => HirType::I8,
            Type::I16 => HirType::I16,
            Type::I32 => HirType::I32,
            Type::I64 => HirType::I64,
            Type::I128 => HirType::I128,
            Type::Isize => HirType::Isize,
            Type::U8 => HirType::U8,
            Type::U16 => HirType::U16,
            Type::U32 => HirType::U32,
            Type::U64 => HirType::U64,
            Type::U128 => HirType::U128,
            Type::Usize => HirType::Usize,
            Type::F32 => HirType::F32,
            Type::F64 => HirType::F64,
            Type::Char => HirType::Char,
            Type::Str | Type::String => HirType::String,
            Type::Ref { mutable, inner, .. } => HirType::Ref {
                mutable: *mutable,
                inner: Box::new(self.type_to_hir(inner)),
            },
            Type::Array { element, size } => HirType::Array {
                element: Box::new(self.type_to_hir(element)),
                size: *size,
            },
            Type::Tuple(elems) => {
                HirType::Tuple(elems.iter().map(|e| self.type_to_hir(e)).collect())
            }
            Type::Function {
                params,
                return_type,
                ..
            } => HirType::Fn {
                params: params.iter().map(|p| self.type_to_hir(p)).collect(),
                return_type: Box::new(self.type_to_hir(return_type)),
            },
            Type::Named { name, args } => HirType::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.type_to_hir(a)).collect(),
            },
            Type::Var(v) => HirType::Var(v.0),
            Type::Forall { inner, .. } => self.type_to_hir(inner),
            Type::Never | Type::Unknown | Type::Error | Type::SelfType => HirType::Error,
        }
    }

    fn hir_type_to_type(&self, ty: &HirType) -> Type {
        match ty {
            HirType::Unit => Type::Unit,
            HirType::Bool => Type::Bool,
            HirType::I8 => Type::I8,
            HirType::I16 => Type::I16,
            HirType::I32 => Type::I32,
            HirType::I64 => Type::I64,
            HirType::I128 => Type::I128,
            HirType::Isize => Type::Isize,
            HirType::U8 => Type::U8,
            HirType::U16 => Type::U16,
            HirType::U32 => Type::U32,
            HirType::U64 => Type::U64,
            HirType::U128 => Type::U128,
            HirType::Usize => Type::Usize,
            HirType::F32 => Type::F32,
            HirType::F64 => Type::F64,
            HirType::Char => Type::Char,
            HirType::String => Type::String,
            HirType::Ref { mutable, inner } => Type::Ref {
                mutable: *mutable,
                lifetime: None,
                inner: Box::new(self.hir_type_to_type(inner)),
            },
            HirType::Array { element, size } => Type::Array {
                element: Box::new(self.hir_type_to_type(element)),
                size: *size,
            },
            HirType::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.hir_type_to_type(e)).collect())
            }
            HirType::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.hir_type_to_type(a)).collect(),
            },
            HirType::Fn {
                params,
                return_type,
            } => Type::Function {
                params: params.iter().map(|p| self.hir_type_to_type(p)).collect(),
                return_type: Box::new(self.hir_type_to_type(return_type)),
                effects: types::EffectSet::new(),
            },
            HirType::Var(v) => Type::Var(TypeVar(*v)),
            HirType::Never => Type::Never,
            HirType::Error => Type::Error,
        }
    }

    fn pattern_name(&self, pattern: &Pattern) -> String {
        match pattern {
            Pattern::Binding { name, .. } => name.clone(),
            Pattern::Wildcard => "_".to_string(),
            _ => "_".to_string(),
        }
    }

    fn solve_constraints(&mut self) -> Result<()> {
        // Simple unification - a real implementation would be more sophisticated
        // Collect errors first to avoid borrow issues
        let errors: Vec<_> = self
            .constraints
            .iter()
            .filter(|c| !self.types_compatible(&c.expected, &c.actual))
            .map(|c| {
                (
                    format!(
                        "Type mismatch: expected {:?}, found {:?}",
                        c.expected, c.actual
                    ),
                    c.span,
                )
            })
            .collect();

        for (msg, span) in errors {
            self.errors.push(TypeError { message: msg, span });
        }
        Ok(())
    }

    fn types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        match (t1, t2) {
            (Type::Var(_), _) | (_, Type::Var(_)) => true, // Type variables unify with anything
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            (Type::Error, _) | (_, Type::Error) => true,
            (Type::Never, _) | (_, Type::Never) => true, // Never is subtype of all types
            (Type::Unit, Type::Unit) => true,
            (Type::Bool, Type::Bool) => true,
            (Type::I8, Type::I8) => true,
            (Type::I16, Type::I16) => true,
            (Type::I32, Type::I32) => true,
            (Type::I64, Type::I64) => true,
            (Type::I128, Type::I128) => true,
            (Type::Isize, Type::Isize) => true,
            (Type::U8, Type::U8) => true,
            (Type::U16, Type::U16) => true,
            (Type::U32, Type::U32) => true,
            (Type::U64, Type::U64) => true,
            (Type::U128, Type::U128) => true,
            (Type::Usize, Type::Usize) => true,
            (Type::F32, Type::F32) => true,
            (Type::F64, Type::F64) => true,
            (Type::Char, Type::Char) => true,
            (Type::Str, Type::Str) => true,
            (Type::String, Type::String) => true,
            (
                Type::Ref {
                    mutable: m1,
                    inner: i1,
                    ..
                },
                Type::Ref {
                    mutable: m2,
                    inner: i2,
                    ..
                },
            ) => m1 == m2 && self.types_compatible(i1, i2),
            (
                Type::Array {
                    element: e1,
                    size: s1,
                },
                Type::Array {
                    element: e2,
                    size: s2,
                },
            ) => s1 == s2 && self.types_compatible(e1, e2),
            (Type::Tuple(t1), Type::Tuple(t2)) => {
                t1.len() == t2.len()
                    && t1
                        .iter()
                        .zip(t2.iter())
                        .all(|(a, b)| self.types_compatible(a, b))
            }
            (Type::Named { name: n1, args: a1 }, Type::Named { name: n2, args: a2 }) => {
                n1 == n2
                    && a1.len() == a2.len()
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(a, b)| self.types_compatible(a, b))
            }
            _ => false,
        }
    }
}

impl TypeEnv {
    fn push_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: String, ty: Type, mutable: bool) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(
                name,
                TypeBinding {
                    ty,
                    mutable,
                    used: false,
                },
            );
        }
    }

    fn lookup(&self, name: &str) -> Option<&TypeBinding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.bindings.get(name) {
                return Some(binding);
            }
        }
        None
    }

    fn lookup_mut(&mut self, name: &str) -> Option<&mut TypeBinding> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.bindings.get_mut(name) {
                return Some(binding);
            }
        }
        None
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
