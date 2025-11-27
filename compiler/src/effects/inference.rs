//! Effect inference pass
//!
//! Infers effects for expressions and checks that all effects are declared
//! in function signatures.

use crate::ast::{self, Ast, BinaryOp, Expr, Item, Stmt};
use crate::common::Span;
use crate::resolve::{DefId, SymbolTable};
use crate::types::core::{Effect, EffectSet};
use std::collections::HashMap;

/// Effect inference context
pub struct EffectChecker<'a> {
    symbols: &'a SymbolTable,
    /// Inferred effects per function DefId
    fn_effects: HashMap<DefId, EffectSet>,
    /// Current function's declared effects
    declared: EffectSet,
    /// Current function's inferred effects
    inferred: EffectSet,
    /// Current function span (for error reporting)
    current_fn_span: Span,
    /// Errors
    errors: Vec<EffectError>,
}

/// Effect error
#[derive(Debug, Clone)]
pub struct EffectError {
    pub kind: EffectErrorKind,
    pub span: Span,
    pub fn_span: Span,
}

/// Kind of effect error
#[derive(Debug, Clone)]
pub enum EffectErrorKind {
    /// Effect used but not declared in function signature
    UndeclaredEffect { effect: String },
    /// Effect not handled
    UnhandledEffect { effect: String },
    /// Effectful operation in pure context
    EffectInPureContext { effect: String },
}

impl<'a> EffectChecker<'a> {
    pub fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            fn_effects: HashMap::new(),
            declared: EffectSet::new(),
            inferred: EffectSet::new(),
            current_fn_span: Span::dummy(),
            errors: Vec::new(),
        }
    }

    /// Check effects for entire program
    pub fn check_program(&mut self, ast: &Ast) -> Result<(), Vec<EffectError>> {
        // First pass: collect declared effects for all functions
        for item in &ast.items {
            if let Item::Function(f) = item {
                self.collect_function_effects(f);
            }
        }

        // Second pass: infer and check effects in function bodies
        for item in &ast.items {
            if let Item::Function(f) = item {
                self.check_function(f);
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn collect_function_effects(&mut self, f: &ast::FnDef) {
        let mut effects = EffectSet::new();
        for eff_ref in &f.effects {
            let effect = self.resolve_effect_ref(eff_ref);
            effects.add(effect);
        }

        if let Some(def_id) = self.symbols.def_for_node(f.id) {
            self.fn_effects.insert(def_id, effects);
        }
    }

    fn check_function(&mut self, f: &ast::FnDef) {
        // Set declared effects for this function
        self.declared = EffectSet::new();
        for eff_ref in &f.effects {
            let effect = self.resolve_effect_ref(eff_ref);
            self.declared.add(effect);
        }

        // Reset inferred effects
        self.inferred = EffectSet::new();
        self.current_fn_span = f.span;

        // Infer effects from body
        self.infer_block(&f.body);

        // Check that all inferred effects are declared
        for effect_name in &self.inferred.effects.clone() {
            if !self.declared.contains(effect_name) {
                self.errors.push(EffectError {
                    kind: EffectErrorKind::UndeclaredEffect {
                        effect: effect_name.clone(),
                    },
                    span: f.span, // TODO: more precise span
                    fn_span: f.span,
                });
            }
        }
    }

    fn infer_block(&mut self, block: &ast::Block) -> EffectSet {
        let mut block_effects = EffectSet::new();

        for stmt in &block.stmts {
            let stmt_effects = self.infer_stmt(stmt);
            block_effects = block_effects.union(&stmt_effects);
        }

        self.inferred = self.inferred.union(&block_effects);
        block_effects
    }

    fn infer_stmt(&mut self, stmt: &Stmt) -> EffectSet {
        match stmt {
            Stmt::Let { value, .. } => {
                if let Some(init) = value {
                    self.infer_expr(init)
                } else {
                    EffectSet::new()
                }
            }
            Stmt::Expr { expr, .. } => self.infer_expr(expr),
            Stmt::Assign { target, value, .. } => {
                let mut effects = self.infer_expr(target);
                effects = effects.union(&self.infer_expr(value));
                // Assignment implies Mut effect if target is mutable
                effects
            }
            Stmt::Empty => EffectSet::new(),
        }
    }

    fn infer_expr(&mut self, expr: &Expr) -> EffectSet {
        match expr {
            Expr::Literal { .. } => EffectSet::new(),

            Expr::Path { .. } => EffectSet::new(),

            Expr::Binary {
                op, left, right, ..
            } => {
                let mut effects = self.infer_expr(left);
                effects = effects.union(&self.infer_expr(right));

                // Division and remainder may panic
                if matches!(op, BinaryOp::Div | BinaryOp::Rem) {
                    effects.add(Effect {
                        name: "Panic".to_string(),
                        args: Vec::new(),
                    });
                }

                effects
            }

            Expr::Unary { expr, .. } => self.infer_expr(expr),

            Expr::Call { callee, args, .. } => {
                let mut effects = self.infer_expr(callee);

                // Get callee's declared effects
                let callee_effects = self.get_callee_effects(callee);
                effects = effects.union(&callee_effects);

                // Infer argument effects
                for arg in args {
                    effects = effects.union(&self.infer_expr(arg));
                }

                effects
            }

            Expr::MethodCall { receiver, args, .. } => {
                let mut effects = self.infer_expr(receiver);
                for arg in args {
                    effects = effects.union(&self.infer_expr(arg));
                }
                // TODO: look up method effects
                effects
            }

            Expr::Field { base, .. } => self.infer_expr(base),

            Expr::TupleField { base, .. } => self.infer_expr(base),

            Expr::Index { base, index, .. } => {
                let mut effects = self.infer_expr(base);
                effects = effects.union(&self.infer_expr(index));
                // Indexing may panic
                effects.add(Effect {
                    name: "Panic".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Cast { expr, .. } => self.infer_expr(expr),

            Expr::Block { block, .. } => self.infer_block(block),

            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let mut effects = self.infer_expr(condition);
                effects = effects.union(&self.infer_block(then_branch));
                if let Some(else_expr) = else_branch {
                    effects = effects.union(&self.infer_expr(else_expr));
                }
                effects
            }

            Expr::Match {
                scrutinee, arms, ..
            } => {
                let mut effects = self.infer_expr(scrutinee);
                for arm in arms {
                    effects = effects.union(&self.infer_expr(&arm.body));
                    if let Some(guard) = &arm.guard {
                        effects = effects.union(&self.infer_expr(guard));
                    }
                }
                effects
            }

            Expr::Loop { body, .. } => {
                let mut effects = self.infer_block(body);
                // Loops may diverge
                effects.add(Effect {
                    name: "Div".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::While {
                condition, body, ..
            } => {
                let mut effects = self.infer_expr(condition);
                effects = effects.union(&self.infer_block(body));
                // Loops may diverge
                effects.add(Effect {
                    name: "Div".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::For { iter, body, .. } => {
                let mut effects = self.infer_expr(iter);
                effects = effects.union(&self.infer_block(body));
                effects
            }

            Expr::Return { value, .. } => {
                if let Some(val) = value {
                    self.infer_expr(val)
                } else {
                    EffectSet::new()
                }
            }

            Expr::Break { value, .. } => {
                if let Some(val) = value {
                    self.infer_expr(val)
                } else {
                    EffectSet::new()
                }
            }

            Expr::Continue { .. } => EffectSet::new(),

            Expr::Closure { body, .. } => {
                // Closures capture their effects
                self.infer_expr(body)
            }

            Expr::Tuple { elements, .. } => {
                let mut effects = EffectSet::new();
                for elem in elements {
                    effects = effects.union(&self.infer_expr(elem));
                }
                effects
            }

            Expr::Array { elements, .. } => {
                let mut effects = EffectSet::new();
                for elem in elements {
                    effects = effects.union(&self.infer_expr(elem));
                }
                effects
            }

            Expr::StructLit { fields, .. } => {
                let mut effects = EffectSet::new();
                for (_, expr) in fields {
                    effects = effects.union(&self.infer_expr(expr));
                }
                effects
            }

            Expr::Try { expr, .. } => {
                let mut effects = self.infer_expr(expr);
                // Try can propagate Panic
                effects.add(Effect {
                    name: "Panic".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Perform { effect, args, .. } => {
                let mut effects = EffectSet::new();
                // Add the performed effect
                if let Some(name) = effect.name() {
                    effects.add(Effect {
                        name: name.to_string(),
                        args: Vec::new(),
                    });
                }
                for arg in args {
                    effects = effects.union(&self.infer_expr(arg));
                }
                effects
            }

            Expr::Handle { expr, handler, .. } => {
                let body_effects = self.infer_expr(expr);
                // Handler removes the handled effect
                let handled_name = handler.name().unwrap_or("").to_string();
                let mut result = EffectSet::new();
                for eff in &body_effects.effects {
                    if eff != &handled_name {
                        result.effects.insert(eff.clone());
                    }
                }
                result
            }

            Expr::Sample { distribution, .. } => {
                let mut effects = self.infer_expr(distribution);
                // Sample has Prob effect
                effects.add(Effect {
                    name: "Prob".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Await { expr, .. } => {
                let mut effects = self.infer_expr(expr);
                // Await has Async effect
                effects.add(Effect {
                    name: "Async".to_string(),
                    args: Vec::new(),
                });
                effects
            }
        }
    }

    fn get_callee_effects(&self, callee: &Expr) -> EffectSet {
        if let Expr::Path { path, id } = callee {
            if path.is_simple() {
                // Look up the function by NodeId reference
                if let Some(def_id) = self.symbols.ref_for_node(*id) {
                    if let Some(effects) = self.fn_effects.get(&def_id) {
                        return effects.clone();
                    }
                }
            }
        }
        EffectSet::new()
    }

    fn resolve_effect_ref(&self, eff_ref: &ast::EffectRef) -> Effect {
        let name = eff_ref.name.name().unwrap_or("Unknown").to_string();
        Effect {
            name,
            args: Vec::new(),
        }
    }

    /// Get the inferred effects for a function
    pub fn get_function_effects(&self, def_id: DefId) -> Option<&EffectSet> {
        self.fn_effects.get(&def_id)
    }
}

impl std::fmt::Display for EffectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            EffectErrorKind::UndeclaredEffect { effect } => {
                write!(f, "Effect `{}` not declared in function signature", effect)
            }
            EffectErrorKind::UnhandledEffect { effect } => {
                write!(f, "Unhandled effect `{}`", effect)
            }
            EffectErrorKind::EffectInPureContext { effect } => {
                write!(f, "Cannot perform `{}` in pure context", effect)
            }
        }
    }
}
