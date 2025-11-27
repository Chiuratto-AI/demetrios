//! Ownership checker implementation
//!
//! Checks ownership, borrowing, and linearity rules.

use crate::ast::{self, Ast, BinaryOp, Expr, Item, Stmt, TypeExpr, UnaryOp};
use crate::common::Span;
use crate::diagnostics::{CompileError, SourceFile};
use crate::resolve::{DefId, SymbolTable};

use super::state::*;
use std::collections::HashMap;

/// Ownership and borrow checker
pub struct OwnershipChecker<'a> {
    symbols: &'a SymbolTable,
    source: &'a SourceFile,
    /// Scope stack
    scopes: Vec<ScopeState>,
    /// Type linearity cache (for structs)
    linearity_cache: HashMap<DefId, Linearity>,
    /// Errors
    errors: Vec<CompileError>,
}

impl<'a> OwnershipChecker<'a> {
    pub fn new(symbols: &'a SymbolTable, source: &'a SourceFile) -> Self {
        Self {
            symbols,
            source,
            scopes: vec![ScopeState::new()],
            linearity_cache: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Check entire program
    pub fn check_program(&mut self, ast: &Ast) -> Result<(), Vec<CompileError>> {
        // Build linearity cache from struct definitions
        for item in &ast.items {
            if let Item::Struct(s) = item {
                if let Some(def_id) = self.symbols.def_for_node(s.id) {
                    let linearity = if s.modifiers.linear {
                        Linearity::Linear
                    } else if s.modifiers.affine {
                        Linearity::Affine
                    } else {
                        Linearity::Unrestricted
                    };
                    self.linearity_cache.insert(def_id, linearity);
                }
            }
        }

        // Check functions
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

    fn check_function(&mut self, f: &ast::FnDef) {
        self.push_scope();

        // Track parameters
        for param in &f.params {
            if let Some(def_id) = self.symbols.def_for_node(param.id) {
                let linearity = self.get_type_linearity(&param.ty);
                let name = self.get_pattern_name(&param.pattern);
                self.track_value(def_id, name, linearity, get_param_span(param));
            }
        }

        // Check body
        self.check_block(&f.body);

        // Check linear values consumed
        self.check_scope_end(f.span);
        self.pop_scope();
    }

    fn check_block(&mut self, block: &ast::Block) {
        self.push_scope();

        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }

        self.check_scope_end(Span::dummy());
        self.pop_scope();
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                pattern, ty, value, ..
            } => {
                // Check initializer first
                if let Some(init) = value {
                    self.check_expr(init, UseKind::Move);
                }

                // Track the binding
                let name = self.get_pattern_name(pattern);
                if let Some(def_id) = self.get_pattern_def_id(pattern) {
                    let linearity = if let Some(ty_expr) = ty {
                        self.get_type_linearity(ty_expr)
                    } else {
                        Linearity::Unrestricted
                    };
                    self.track_value(def_id, name, linearity, get_pattern_span(pattern));
                }
            }

            Stmt::Expr { expr, .. } => {
                self.check_expr(expr, UseKind::Move);
            }

            Stmt::Assign { target, value, .. } => {
                self.check_expr(value, UseKind::Move);
                // Target is being written to, not consumed
            }

            Stmt::Empty => {}
        }
    }

    fn check_expr(&mut self, expr: &Expr, use_kind: UseKind) {
        match expr {
            Expr::Literal { .. } => {}

            Expr::Path { path, id } => {
                if path.is_simple() {
                    if let Some(def_id) = self.symbols.ref_for_node(*id) {
                        self.use_value(def_id, use_kind, get_expr_span(expr));
                    }
                }
            }

            Expr::Binary {
                op, left, right, ..
            } => {
                // Assignment: RHS is moved, LHS is target
                if matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                ) {
                    self.check_expr(left, UseKind::Copy);
                    self.check_expr(right, UseKind::Copy);
                } else {
                    self.check_expr(left, UseKind::Copy);
                    self.check_expr(right, UseKind::Copy);
                }
            }

            Expr::Unary {
                op, expr: inner, ..
            } => {
                match op {
                    UnaryOp::Ref => {
                        // Shared borrow
                        if let Some(place) = self.expr_to_place(inner) {
                            self.borrow_shared(place, get_expr_span(expr));
                        }
                    }
                    UnaryOp::RefMut => {
                        // Exclusive borrow (&!)
                        if let Some(place) = self.expr_to_place(inner) {
                            self.borrow_exclusive(place, get_expr_span(expr));
                        }
                    }
                    UnaryOp::Deref => {
                        self.check_expr(inner, UseKind::Copy);
                    }
                    _ => {
                        self.check_expr(inner, use_kind);
                    }
                }
            }

            Expr::Call { callee, args, .. } => {
                self.check_expr(callee, UseKind::Copy);
                for arg in args {
                    // TODO: check parameter ownership annotations
                    self.check_expr(arg, UseKind::Move);
                }
            }

            Expr::MethodCall { receiver, args, .. } => {
                // TODO: check method receiver ownership
                self.check_expr(receiver, UseKind::Copy);
                for arg in args {
                    self.check_expr(arg, UseKind::Move);
                }
            }

            Expr::Field { base, .. } => {
                self.check_expr(base, UseKind::Copy);
            }

            Expr::TupleField { base, .. } => {
                self.check_expr(base, UseKind::Copy);
            }

            Expr::Index { base, index, .. } => {
                self.check_expr(base, UseKind::Copy);
                self.check_expr(index, UseKind::Copy);
            }

            Expr::Cast { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::Block { block, .. } => {
                self.check_block(block);
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.check_expr(condition, UseKind::Copy);
                self.check_block(then_branch);
                if let Some(else_expr) = else_branch {
                    self.check_expr(else_expr, use_kind);
                }
            }

            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.check_expr(scrutinee, UseKind::Move);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard, UseKind::Copy);
                    }
                    self.check_expr(&arm.body, use_kind);
                }
            }

            Expr::Loop { body, .. } => {
                self.check_block(body);
            }

            Expr::While {
                condition, body, ..
            } => {
                self.check_expr(condition, UseKind::Copy);
                self.check_block(body);
            }

            Expr::For { iter, body, .. } => {
                self.check_expr(iter, UseKind::Move);
                self.push_scope();
                // TODO: bind pattern variables
                self.check_block(body);
                self.check_scope_end(Span::dummy());
                self.pop_scope();
            }

            Expr::Return { value, .. } => {
                if let Some(val) = value {
                    self.check_expr(val, UseKind::Move);
                }
            }

            Expr::Break { value, .. } => {
                if let Some(val) = value {
                    self.check_expr(val, UseKind::Move);
                }
            }

            Expr::Continue { .. } => {}

            Expr::Closure { body, .. } => {
                // TODO: check captures
                self.check_expr(body, UseKind::Move);
            }

            Expr::Tuple { elements, .. } => {
                for elem in elements {
                    self.check_expr(elem, use_kind);
                }
            }

            Expr::Array { elements, .. } => {
                for elem in elements {
                    self.check_expr(elem, use_kind);
                }
            }

            Expr::StructLit { fields, .. } => {
                for (_, field_expr) in fields {
                    self.check_expr(field_expr, UseKind::Move);
                }
            }

            Expr::Try { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::Perform { args, .. } => {
                for arg in args {
                    self.check_expr(arg, UseKind::Move);
                }
            }

            Expr::Handle { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::Sample { distribution, .. } => {
                self.check_expr(distribution, UseKind::Move);
            }

            Expr::Await { expr, .. } => {
                self.check_expr(expr, use_kind);
            }
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(ScopeState::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn current_scope(&mut self) -> &mut ScopeState {
        self.scopes.last_mut().unwrap()
    }

    fn track_value(&mut self, def_id: DefId, name: String, linearity: Linearity, span: Span) {
        let value = TrackedValue::new(def_id, name, linearity, span);
        self.current_scope().track(value);
    }

    fn use_value(&mut self, def_id: DefId, use_kind: UseKind, span: Span) {
        // Look up in all scopes
        for scope in self.scopes.iter_mut().rev() {
            if let Some(value) = scope.get_mut(def_id) {
                // Check if already moved
                if let OwnershipState::Moved { to } = &value.state {
                    self.errors.push(CompileError::UseAfterMove {
                        name: value.name.clone(),
                        use_span: span.into(),
                        move_span: (*to).into(),
                        src: self.source.to_named_source(),
                    });
                    return;
                }

                // Record use
                value.record_use(span);

                // Update state for move
                if use_kind == UseKind::Move && value.linearity != Linearity::Unrestricted {
                    value.state = OwnershipState::Moved { to: span };
                }

                return;
            }
        }
    }

    fn borrow_shared(&mut self, place: Place, span: Span) {
        // Check if exclusively borrowed
        for scope in &self.scopes {
            let active = scope.active_borrows(&place);
            for borrow in active {
                if borrow.exclusive {
                    self.errors.push(CompileError::AlreadyBorrowed {
                        name: place.to_string(),
                        span: span.into(),
                        prev_span: borrow.span.into(),
                        src: self.source.to_named_source(),
                    });
                    return;
                }
            }
        }

        // Add borrow
        self.current_scope().add_borrow(BorrowState {
            place,
            exclusive: false,
            span,
            active: true,
        });
    }

    fn borrow_exclusive(&mut self, place: Place, span: Span) {
        // Check if any borrows exist
        for scope in &self.scopes {
            let active = scope.active_borrows(&place);
            if !active.is_empty() {
                let prev = active[0];
                if prev.exclusive {
                    self.errors.push(CompileError::DoubleMutBorrow {
                        name: place.to_string(),
                        span: span.into(),
                        first_span: prev.span.into(),
                        src: self.source.to_named_source(),
                    });
                } else {
                    self.errors.push(CompileError::AlreadyBorrowed {
                        name: place.to_string(),
                        span: span.into(),
                        prev_span: prev.span.into(),
                        src: self.source.to_named_source(),
                    });
                }
                return;
            }
        }

        // Add borrow
        self.current_scope().add_borrow(BorrowState {
            place,
            exclusive: true,
            span,
            active: true,
        });
    }

    fn check_scope_end(&mut self, scope_end_span: Span) {
        let errors = self.current_scope().check_all_linear();

        for error in errors {
            match error {
                LinearityError::NotConsumed {
                    name, decl_span, ..
                } => {
                    self.errors.push(CompileError::LinearNotConsumed {
                        name,
                        decl_span: decl_span.into(),
                        scope_end: scope_end_span.into(),
                        src: self.source.to_named_source(),
                    });
                }
                LinearityError::MultipleUse {
                    name,
                    first,
                    second,
                    ..
                } => {
                    self.errors.push(CompileError::LinearMultipleUse {
                        name,
                        first_span: first.into(),
                        second_span: second.into(),
                        src: self.source.to_named_source(),
                    });
                }
            }
        }
    }

    fn get_type_linearity(&self, ty: &TypeExpr) -> Linearity {
        match ty {
            TypeExpr::Named { path, .. } => {
                if let Some(name) = path.name() {
                    // Check if it's a known linear/affine type
                    for (def_id, linearity) in &self.linearity_cache {
                        if let Some(sym) = self.symbols.get(*def_id) {
                            if sym.name == name {
                                return *linearity;
                            }
                        }
                    }
                }
                Linearity::Unrestricted
            }
            TypeExpr::Reference { .. } => Linearity::Unrestricted,
            TypeExpr::Tuple(elems) => {
                // If any element is linear, the tuple is linear
                for elem in elems {
                    let elem_lin = self.get_type_linearity(elem);
                    if elem_lin == Linearity::Linear {
                        return Linearity::Linear;
                    }
                }
                Linearity::Unrestricted
            }
            _ => Linearity::Unrestricted,
        }
    }

    fn expr_to_place(&self, expr: &Expr) -> Option<Place> {
        match expr {
            Expr::Path { path, id } => {
                if path.is_simple() {
                    if let Some(def_id) = self.symbols.ref_for_node(*id) {
                        return Some(Place::var(def_id));
                    }
                }
                None
            }
            Expr::Field { base, field, .. } => {
                self.expr_to_place(base).map(|p| p.field(field.clone()))
            }
            Expr::Unary { op, expr, .. } if matches!(op, UnaryOp::Deref) => {
                self.expr_to_place(expr).map(|p| p.deref())
            }
            _ => None,
        }
    }

    fn get_pattern_name(&self, pattern: &ast::Pattern) -> String {
        match pattern {
            ast::Pattern::Binding { name, .. } => name.clone(),
            _ => "<pattern>".to_string(),
        }
    }

    fn get_pattern_def_id(&self, pattern: &ast::Pattern) -> Option<DefId> {
        match pattern {
            ast::Pattern::Binding { name, .. } => {
                // Look up the binding in symbols
                // This is a simplification - in practice we'd have NodeId on patterns
                self.symbols.lookup(name)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UseKind {
    Move,
    Copy,
}

fn get_expr_span(_expr: &Expr) -> Span {
    // TODO: get actual span from expression
    Span::dummy()
}

fn get_pattern_span(_pattern: &ast::Pattern) -> Span {
    // TODO: get actual span from pattern
    Span::dummy()
}

fn get_param_span(_param: &ast::Param) -> Span {
    // TODO: get actual span from parameter
    Span::dummy()
}
