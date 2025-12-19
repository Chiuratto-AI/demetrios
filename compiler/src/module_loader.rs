//! Module loader and import resolver.
//!
//! Loads a root source file, resolves its imports, and returns a single AST
//! with all imported modules merged. Qualified paths that match imported
//! module prefixes are rewritten to unqualified names to match the current
//! single-namespace compiler pipeline.

use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};

use miette::Result;

use crate::ast::Path as AstPath;
use crate::ast::*;
use crate::lexer;
use crate::parser;

pub fn load_program_ast(entry_path: &StdPath) -> Result<Ast> {
    let mut loader = ModuleLoader::new()?;
    let root_id = loader.load_module(entry_path)?;
    loader.into_ast(root_id)
}

struct ModuleLoader {
    next_node_id: u32,
    stdlib_dir: PathBuf,
    modules: Vec<ModuleData>,
    path_to_id: HashMap<PathBuf, usize>,
    load_stack: Vec<PathBuf>,
}

struct ModuleData {
    id: usize,
    path: PathBuf,
    ast: Ast,
    import_paths: Vec<Vec<String>>,
}

impl ModuleLoader {
    fn new() -> Result<Self> {
        Ok(Self {
            next_node_id: 1,
            stdlib_dir: find_stdlib_path(),
            modules: Vec::new(),
            path_to_id: HashMap::new(),
            load_stack: Vec::new(),
        })
    }

    fn load_module(&mut self, path: &StdPath) -> Result<usize> {
        let canonical = path.canonicalize().map_err(|e| {
            miette::miette!("Failed to resolve module path {}: {}", path.display(), e)
        })?;

        if let Some(existing) = self.path_to_id.get(&canonical) {
            return Ok(*existing);
        }

        if self.load_stack.contains(&canonical) {
            return Err(miette::miette!(
                "Circular import detected: {}",
                canonical.display()
            ));
        }

        self.load_stack.push(canonical.clone());

        let source = std::fs::read_to_string(&canonical)
            .map_err(|e| miette::miette!("Failed to read {}: {}", canonical.display(), e))?;
        let tokens = lexer::lex(&source)?;
        let (mut ast, next_id) = parser::parse_with_id_start(&tokens, &source, self.next_node_id)?;
        self.next_node_id = next_id;

        let import_paths = collect_import_paths(&ast);
        let module_prefixes = module_prefixes(&import_paths, &canonical);
        rewrite_paths_in_ast(&mut ast, &module_prefixes);

        let id = self.modules.len();
        self.modules.push(ModuleData {
            id,
            path: canonical.clone(),
            ast,
            import_paths: import_paths.clone(),
        });
        self.path_to_id.insert(canonical.clone(), id);

        let import_paths_owned = import_paths;
        for import_path in &import_paths_owned {
            let import_file = resolve_import_path(&canonical, import_path, &self.stdlib_dir)?;
            let _ = self.load_module(&import_file)?;
        }

        self.load_stack.pop();

        Ok(id)
    }

    fn into_ast(mut self, root_id: usize) -> Result<Ast> {
        let root_module_name = self
            .modules
            .get(root_id)
            .and_then(|m| m.ast.module_name.clone());

        let mut items = Vec::new();
        let mut node_spans = std::collections::HashMap::new();

        let mut defined: HashMap<String, PathBuf> = HashMap::new();

        for module in &mut self.modules {
            for item in &module.ast.items {
                if let Some(name) = item_name(item) {
                    if let Some(prev_path) = defined.get(&name) {
                        return Err(miette::miette!(
                            "Duplicate definition `{}` in {} and {}",
                            name,
                            prev_path.display(),
                            module.path.display()
                        ));
                    }
                    defined.insert(name, module.path.clone());
                }
            }

            items.append(&mut module.ast.items);
            node_spans.extend(module.ast.node_spans.drain());
        }

        Ok(Ast {
            module_name: root_module_name,
            items,
            node_spans,
        })
    }
}

fn find_stdlib_path() -> PathBuf {
    if let Ok(path) = std::env::var("DEMETRIOS_STDLIB") {
        return PathBuf::from(path);
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let stdlib = parent.join("stdlib");
            if stdlib.exists() {
                return stdlib;
            }
        }
    }

    PathBuf::from("/usr/share/demetrios/stdlib")
}

fn collect_import_paths(ast: &Ast) -> Vec<Vec<String>> {
    ast.items
        .iter()
        .filter_map(|item| match item {
            Item::Import(import_def) => Some(import_def.path.segments.clone()),
            _ => None,
        })
        .collect()
}

fn module_prefixes(import_paths: &[Vec<String>], module_path: &StdPath) -> Vec<Vec<String>> {
    let mut prefixes = Vec::new();

    for import_path in import_paths {
        if import_path.is_empty() {
            continue;
        }
        if import_path.len() == 1 {
            prefixes.push(import_path.clone());
        } else {
            prefixes.push(import_path[..import_path.len() - 1].to_vec());
        }
    }

    if let Some(stem) = module_path.file_stem().and_then(|s| s.to_str()) {
        prefixes.push(vec![stem.to_string()]);
    }

    prefixes
}

fn resolve_import_path(
    current_path: &StdPath,
    import_path: &[String],
    stdlib_dir: &StdPath,
) -> Result<PathBuf> {
    if import_path.is_empty() {
        return Err(miette::miette!(
            "Empty import path in {}",
            current_path.display()
        ));
    }

    let (base_dir, segments) = if import_path[0] == "std" {
        (stdlib_dir.to_path_buf(), &import_path[1..])
    } else {
        (
            current_path
                .parent()
                .unwrap_or_else(|| StdPath::new("."))
                .to_path_buf(),
            import_path,
        )
    };

    if segments.is_empty() {
        return Err(miette::miette!(
            "Invalid import path `std` in {}",
            current_path.display()
        ));
    }

    let mut candidates = Vec::new();
    let path_joined = segments.join("/");
    candidates.push(base_dir.join(format!("{}.d", path_joined)));
    candidates.push(base_dir.join(&path_joined).join("mod.d"));
    candidates.push(base_dir.join(&path_joined).join("lib.d"));

    if let Some(last) = segments.last() {
        let mut lowered = segments.to_vec();
        lowered[segments.len() - 1] = last.to_lowercase();
        let lowered_joined = lowered.join("/");
        candidates.push(base_dir.join(format!("{}.d", lowered_joined)));
        candidates.push(base_dir.join(&lowered_joined).join("mod.d"));
        candidates.push(base_dir.join(&lowered_joined).join("lib.d"));
    }

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(miette::miette!(
        "Import not found: `{}` (from {})",
        import_path.join("::"),
        current_path.display()
    ))
}

fn item_name(item: &Item) -> Option<String> {
    match item {
        Item::Function(f) => Some(f.name.clone()),
        Item::Struct(s) => Some(s.name.clone()),
        Item::Enum(e) => Some(e.name.clone()),
        Item::Trait(t) => Some(t.name.clone()),
        Item::TypeAlias(t) => Some(t.name.clone()),
        Item::Effect(e) => Some(e.name.clone()),
        Item::Handler(h) => Some(h.name.clone()),
        Item::Global(g) => match &g.pattern {
            Pattern::Binding { name, .. } => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn rewrite_paths_in_ast(ast: &mut Ast, prefixes: &[Vec<String>]) {
    for item in &mut ast.items {
        rewrite_paths_in_item(item, prefixes);
    }
}

fn rewrite_paths_in_item(item: &mut Item, prefixes: &[Vec<String>]) {
    match item {
        Item::Function(f) => rewrite_fn_def(f, prefixes),
        Item::Struct(s) => rewrite_struct_def(s, prefixes),
        Item::Enum(e) => rewrite_enum_def(e, prefixes),
        Item::Trait(t) => rewrite_trait_def(t, prefixes),
        Item::Impl(i) => rewrite_impl_def(i, prefixes),
        Item::TypeAlias(t) => {
            rewrite_generics(&mut t.generics, prefixes);
            rewrite_type_expr(&mut t.ty, prefixes);
        }
        Item::Effect(e) => rewrite_effect_def(e, prefixes),
        Item::Handler(h) => rewrite_handler_def(h, prefixes),
        Item::Extern(e) => rewrite_extern_block(e, prefixes),
        Item::Global(g) => rewrite_global_def(g, prefixes),
        Item::OdeDef(o) => rewrite_ode_def(o, prefixes),
        Item::PdeDef(p) => rewrite_pde_def(p, prefixes),
        Item::CausalModel(c) => rewrite_causal_model_def(c, prefixes),
        _ => {}
    }
}

fn rewrite_fn_def(def: &mut FnDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    for param in &mut def.params {
        rewrite_pattern(&mut param.pattern, prefixes);
        rewrite_type_expr(&mut param.ty, prefixes);
    }
    if let Some(ret) = &mut def.return_type {
        rewrite_type_expr(ret, prefixes);
    }
    for effect in &mut def.effects {
        rewrite_effect_ref(effect, prefixes);
    }
    rewrite_block(&mut def.body, prefixes);
}

fn rewrite_struct_def(def: &mut StructDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for field in &mut def.fields {
        rewrite_type_expr(&mut field.ty, prefixes);
    }
}

fn rewrite_enum_def(def: &mut EnumDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for variant in &mut def.variants {
        match &mut variant.data {
            VariantData::Tuple(types) => {
                for ty in types {
                    rewrite_type_expr(ty, prefixes);
                }
            }
            VariantData::Struct(fields) => {
                for field in fields {
                    rewrite_type_expr(&mut field.ty, prefixes);
                }
            }
            VariantData::Unit => {}
        }
    }
}

fn rewrite_trait_def(def: &mut TraitDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    for supertrait in &mut def.supertraits {
        rewrite_path(supertrait, prefixes);
    }
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for item in &mut def.items {
        match item {
            TraitItem::Fn(f) => rewrite_trait_fn_def(f, prefixes),
            TraitItem::Type(ty) => {
                for bound in &mut ty.bounds {
                    rewrite_path(bound, prefixes);
                }
                if let Some(default) = &mut ty.default {
                    rewrite_type_expr(default, prefixes);
                }
            }
        }
    }
}

fn rewrite_trait_fn_def(def: &mut TraitFnDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for param in &mut def.params {
        rewrite_pattern(&mut param.pattern, prefixes);
        rewrite_type_expr(&mut param.ty, prefixes);
    }
    if let Some(ret) = &mut def.return_type {
        rewrite_type_expr(ret, prefixes);
    }
    for effect in &mut def.effects {
        rewrite_effect_ref(effect, prefixes);
    }
    if let Some(body) = &mut def.default_body {
        rewrite_block(body, prefixes);
    }
}

fn rewrite_impl_def(def: &mut ImplDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    if let Some(trait_ref) = &mut def.trait_ref {
        rewrite_path(trait_ref, prefixes);
    }
    rewrite_type_expr(&mut def.target_type, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for item in &mut def.items {
        match item {
            ImplItem::Fn(f) => rewrite_fn_def(f, prefixes),
            ImplItem::Type(t) => rewrite_type_expr(&mut t.ty, prefixes),
        }
    }
}

fn rewrite_effect_def(def: &mut EffectDef, prefixes: &[Vec<String>]) {
    for op in &mut def.operations {
        for param in &mut op.params {
            rewrite_pattern(&mut param.pattern, prefixes);
            rewrite_type_expr(&mut param.ty, prefixes);
        }
        if let Some(ret) = &mut op.return_type {
            rewrite_type_expr(ret, prefixes);
        }
    }
}

fn rewrite_handler_def(def: &mut HandlerDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_path(&mut def.effect, prefixes);
    for case in &mut def.cases {
        for param in &mut case.params {
            rewrite_pattern(&mut param.pattern, prefixes);
            rewrite_type_expr(&mut param.ty, prefixes);
        }
        rewrite_expr(&mut case.body, prefixes);
    }
}

fn rewrite_extern_block(block: &mut ExternBlock, prefixes: &[Vec<String>]) {
    for item in &mut block.items {
        match item {
            ExternItem::Fn(f) => {
                for param in &mut f.params {
                    rewrite_type_expr(&mut param.ty, prefixes);
                }
                if let Some(ret) = &mut f.return_type {
                    rewrite_type_expr(ret, prefixes);
                }
            }
            ExternItem::Static(s) => {
                rewrite_type_expr(&mut s.ty, prefixes);
            }
            ExternItem::Type(_) => {}
        }
    }
}

fn rewrite_global_def(def: &mut GlobalDef, prefixes: &[Vec<String>]) {
    rewrite_pattern(&mut def.pattern, prefixes);
    if let Some(ty) = &mut def.ty {
        rewrite_type_expr(ty, prefixes);
    }
    rewrite_expr(&mut def.value, prefixes);
}

fn rewrite_ode_def(def: &mut OdeDef, prefixes: &[Vec<String>]) {
    for param in &mut def.params {
        rewrite_type_expr(&mut param.ty, prefixes);
        if let Some(default) = &mut param.default {
            rewrite_expr(default, prefixes);
        }
    }
    for state in &mut def.state {
        rewrite_type_expr(&mut state.ty, prefixes);
    }
    for eq in &mut def.equations {
        rewrite_expr(&mut eq.rhs, prefixes);
    }
}

fn rewrite_pde_def(def: &mut PdeDef, prefixes: &[Vec<String>]) {
    for param in &mut def.params {
        rewrite_type_expr(&mut param.ty, prefixes);
    }
    for dim in &mut def.domain.dimensions {
        rewrite_expr(&mut dim.min, prefixes);
        rewrite_expr(&mut dim.max, prefixes);
    }
    rewrite_expr(&mut def.equation.rhs, prefixes);
    for bc in &mut def.boundary_conditions {
        rewrite_expr(&mut bc.boundary.value, prefixes);
        match &mut bc.condition {
            BoundaryConditionType::Dirichlet(expr) | BoundaryConditionType::Neumann(expr) => {
                rewrite_expr(expr, prefixes);
            }
            BoundaryConditionType::Robin { a, b, value } => {
                rewrite_expr(a, prefixes);
                rewrite_expr(b, prefixes);
                rewrite_expr(value, prefixes);
            }
            BoundaryConditionType::Periodic => {}
        }
    }
    if let Some(init) = &mut def.initial_condition {
        rewrite_expr(init, prefixes);
    }
}

fn rewrite_causal_model_def(def: &mut CausalModelDef, prefixes: &[Vec<String>]) {
    for node in &mut def.nodes {
        if let Some(ty) = &mut node.ty {
            rewrite_type_expr(ty, prefixes);
        }
    }
    for eq in &mut def.equations {
        rewrite_expr(&mut eq.rhs, prefixes);
    }
}

fn rewrite_generics(generics: &mut Generics, prefixes: &[Vec<String>]) {
    for param in &mut generics.params {
        match param {
            GenericParam::Type {
                bounds, default, ..
            } => {
                for bound in bounds {
                    rewrite_path(bound, prefixes);
                }
                if let Some(default) = default {
                    rewrite_type_expr(default, prefixes);
                }
            }
            GenericParam::Const { ty, .. } => {
                rewrite_type_expr(ty, prefixes);
            }
        }
    }
}

fn rewrite_where_clause(preds: &mut [WherePredicate], prefixes: &[Vec<String>]) {
    for pred in preds {
        rewrite_type_expr(&mut pred.ty, prefixes);
        for bound in &mut pred.bounds {
            rewrite_path(bound, prefixes);
        }
    }
}

fn rewrite_effect_ref(effect: &mut EffectRef, prefixes: &[Vec<String>]) {
    rewrite_path(&mut effect.name, prefixes);
    for arg in &mut effect.args {
        rewrite_type_expr(arg, prefixes);
    }
}

fn rewrite_pattern(pat: &mut Pattern, prefixes: &[Vec<String>]) {
    match pat {
        Pattern::Tuple(items) | Pattern::Or(items) => {
            for item in items {
                rewrite_pattern(item, prefixes);
            }
        }
        Pattern::Struct { path, fields } => {
            rewrite_path(path, prefixes);
            for (_, field_pat) in fields {
                rewrite_pattern(field_pat, prefixes);
            }
        }
        Pattern::Enum { path, patterns } => {
            rewrite_path(path, prefixes);
            if let Some(patterns) = patterns {
                for pattern in patterns {
                    rewrite_pattern(pattern, prefixes);
                }
            }
        }
        _ => {}
    }
}

fn rewrite_block(block: &mut Block, prefixes: &[Vec<String>]) {
    for stmt in &mut block.stmts {
        rewrite_stmt(stmt, prefixes);
    }
}

fn rewrite_stmt(stmt: &mut Stmt, prefixes: &[Vec<String>]) {
    match stmt {
        Stmt::Let {
            pattern, ty, value, ..
        } => {
            rewrite_pattern(pattern, prefixes);
            if let Some(ty) = ty {
                rewrite_type_expr(ty, prefixes);
            }
            if let Some(value) = value {
                rewrite_expr(value, prefixes);
            }
        }
        Stmt::Expr { expr, .. } => rewrite_expr(expr, prefixes),
        Stmt::Assign { target, value, .. } => {
            rewrite_expr(target, prefixes);
            rewrite_expr(value, prefixes);
        }
        Stmt::MacroInvocation(_) | Stmt::Empty => {}
    }
}

fn rewrite_expr(expr: &mut Expr, prefixes: &[Vec<String>]) {
    match expr {
        Expr::Path { path, .. } => rewrite_path(path, prefixes),
        Expr::Binary { left, right, .. } => {
            rewrite_expr(left, prefixes);
            rewrite_expr(right, prefixes);
        }
        Expr::Unary { expr, .. } => rewrite_expr(expr, prefixes),
        Expr::Call { callee, args, .. } => {
            rewrite_expr(callee, prefixes);
            for arg in args {
                rewrite_expr(arg, prefixes);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            rewrite_expr(receiver, prefixes);
            for arg in args {
                rewrite_expr(arg, prefixes);
            }
        }
        Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
            rewrite_expr(base, prefixes);
        }
        Expr::Index { base, index, .. } => {
            rewrite_expr(base, prefixes);
            rewrite_expr(index, prefixes);
        }
        Expr::Cast { expr, ty, .. } => {
            rewrite_expr(expr, prefixes);
            rewrite_type_expr(ty, prefixes);
        }
        Expr::Block { block, .. } => rewrite_block(block, prefixes),
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            rewrite_expr(condition, prefixes);
            rewrite_block(then_branch, prefixes);
            if let Some(else_expr) = else_branch {
                rewrite_expr(else_expr, prefixes);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            rewrite_expr(scrutinee, prefixes);
            for arm in arms {
                rewrite_pattern(&mut arm.pattern, prefixes);
                if let Some(guard) = &mut arm.guard {
                    rewrite_expr(guard, prefixes);
                }
                rewrite_expr(&mut arm.body, prefixes);
            }
        }
        Expr::Loop { body, .. } => rewrite_block(body, prefixes),
        Expr::While {
            condition, body, ..
        } => {
            rewrite_expr(condition, prefixes);
            rewrite_block(body, prefixes);
        }
        Expr::For {
            pattern,
            iter,
            body,
            ..
        } => {
            rewrite_pattern(pattern, prefixes);
            rewrite_expr(iter, prefixes);
            rewrite_block(body, prefixes);
        }
        Expr::Return { value, .. } | Expr::Break { value, .. } => {
            if let Some(value) = value {
                rewrite_expr(value, prefixes);
            }
        }
        Expr::Closure {
            params,
            return_type,
            body,
            ..
        }
        | Expr::AsyncClosure {
            params,
            return_type,
            body,
            ..
        } => {
            for (_, ty) in params {
                if let Some(ty) = ty {
                    rewrite_type_expr(ty, prefixes);
                }
            }
            if let Some(ret) = return_type {
                rewrite_type_expr(ret, prefixes);
            }
            rewrite_expr(body, prefixes);
        }
        Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
            for elem in elements {
                rewrite_expr(elem, prefixes);
            }
        }
        Expr::Range { start, end, .. } => {
            if let Some(start) = start {
                rewrite_expr(start, prefixes);
            }
            if let Some(end) = end {
                rewrite_expr(end, prefixes);
            }
        }
        Expr::StructLit { path, fields, .. } => {
            rewrite_path(path, prefixes);
            for (_, expr) in fields {
                rewrite_expr(expr, prefixes);
            }
        }
        Expr::Try { expr, .. } | Expr::Await { expr, .. } | Expr::Spawn { expr, .. } => {
            rewrite_expr(expr, prefixes);
        }
        Expr::Perform { effect, args, .. } => {
            rewrite_path(effect, prefixes);
            for arg in args {
                rewrite_expr(arg, prefixes);
            }
        }
        Expr::Handle { expr, handler, .. } => {
            rewrite_expr(expr, prefixes);
            rewrite_path(handler, prefixes);
        }
        Expr::Sample { distribution, .. } => rewrite_expr(distribution, prefixes),
        Expr::AsyncBlock { block, .. } => rewrite_block(block, prefixes),
        Expr::Select { arms, .. } => {
            for arm in arms {
                rewrite_expr(&mut arm.future, prefixes);
                rewrite_pattern(&mut arm.pattern, prefixes);
                if let Some(guard) = &mut arm.guard {
                    rewrite_expr(guard, prefixes);
                }
                rewrite_expr(&mut arm.body, prefixes);
            }
        }
        Expr::Join { futures, .. } => {
            for future in futures {
                rewrite_expr(future, prefixes);
            }
        }
        Expr::Do { interventions, .. } => {
            for (_, value) in interventions {
                rewrite_expr(value, prefixes);
            }
        }
        Expr::Counterfactual {
            factual,
            intervention,
            outcome,
            ..
        } => {
            rewrite_expr(factual, prefixes);
            rewrite_expr(intervention, prefixes);
            rewrite_expr(outcome, prefixes);
        }
        Expr::KnowledgeExpr {
            value,
            epsilon,
            validity,
            provenance,
            ..
        } => {
            rewrite_expr(value, prefixes);
            if let Some(eps) = epsilon {
                rewrite_expr(eps, prefixes);
            }
            if let Some(validity) = validity {
                rewrite_expr(validity, prefixes);
            }
            if let Some(provenance) = provenance {
                rewrite_expr(provenance, prefixes);
            }
        }
        Expr::Uncertain {
            value, uncertainty, ..
        } => {
            rewrite_expr(value, prefixes);
            rewrite_expr(uncertainty, prefixes);
        }
        Expr::GpuAnnotated {
            expr, annotation, ..
        } => {
            rewrite_expr(expr, prefixes);
            for (_, param_expr) in &mut annotation.params {
                rewrite_expr(param_expr, prefixes);
            }
        }
        Expr::Observe {
            data, distribution, ..
        } => {
            rewrite_expr(data, prefixes);
            rewrite_expr(distribution, prefixes);
        }
        Expr::Query {
            target,
            given,
            interventions,
            ..
        } => {
            rewrite_expr(target, prefixes);
            for g in given {
                rewrite_expr(g, prefixes);
            }
            for (_, value) in interventions {
                rewrite_expr(value, prefixes);
            }
        }
        Expr::Literal { .. }
        | Expr::Continue { .. }
        | Expr::MacroInvocation(_)
        | Expr::OntologyTerm { .. } => {}
    }
}

fn rewrite_type_expr(ty: &mut TypeExpr, prefixes: &[Vec<String>]) {
    match ty {
        TypeExpr::Named { path, args, .. } => {
            rewrite_path(path, prefixes);
            for arg in args {
                rewrite_type_expr(arg, prefixes);
            }
        }
        TypeExpr::Reference { inner, .. }
        | TypeExpr::RawPointer { inner, .. }
        | TypeExpr::Linear { inner, .. }
        | TypeExpr::Effected { inner, .. } => rewrite_type_expr(inner, prefixes),
        TypeExpr::Array { element, size } => {
            rewrite_type_expr(element, prefixes);
            if let Some(size) = size {
                rewrite_expr(size, prefixes);
            }
        }
        TypeExpr::Tuple(elements) => {
            for elem in elements {
                rewrite_type_expr(elem, prefixes);
            }
        }
        TypeExpr::Function {
            params,
            return_type,
            effects,
        } => {
            for param in params {
                rewrite_type_expr(param, prefixes);
            }
            rewrite_type_expr(return_type, prefixes);
            for effect in effects {
                rewrite_effect_ref(effect, prefixes);
            }
        }
        TypeExpr::Knowledge {
            value_type,
            epsilon,
            validity,
            provenance,
        } => {
            rewrite_type_expr(value_type, prefixes);
            if let Some(epsilon) = epsilon {
                rewrite_expr(&mut epsilon.value, prefixes);
            }
            if let Some(validity) = validity {
                rewrite_expr(&mut validity.condition, prefixes);
            }
            if let Some(prov) = provenance {
                if let Some(expr) = &mut prov.source {
                    rewrite_expr(expr, prefixes);
                }
            }
        }
        TypeExpr::Quantity { numeric_type, .. } => rewrite_type_expr(numeric_type, prefixes),
        TypeExpr::Tensor {
            element_type,
            shape,
        } => {
            rewrite_type_expr(element_type, prefixes);
            for dim in shape {
                if let TensorDim::Expr(expr) = dim {
                    rewrite_expr(expr, prefixes);
                }
            }
        }
        TypeExpr::Tile { element_type, .. } => rewrite_type_expr(element_type, prefixes),
        TypeExpr::SelfType | TypeExpr::Infer | TypeExpr::Unit | TypeExpr::Ontology { .. } => {}
    }
}

fn rewrite_path(path: &mut AstPath, prefixes: &[Vec<String>]) {
    if path.segments.len() <= 1 {
        return;
    }

    let module_part = &path.segments[..path.segments.len() - 1];
    if prefixes
        .iter()
        .any(|prefix| module_part == prefix.as_slice())
    {
        if let Some(last) = path.segments.last().cloned() {
            path.segments = vec![last];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolves_imports_and_rewrites_qualified_paths() {
        let dir = tempfile::tempdir().unwrap();
        let module_a = dir.path().join("a.d");
        let module_b = dir.path().join("b.d");

        fs::write(&module_b, "type Foo = i32;").unwrap();
        fs::write(
            &module_a,
            "import b;\n\nfn use(foo: b.Foo) -> b.Foo { return foo }\n",
        )
        .unwrap();

        let ast = load_program_ast(&module_a).unwrap();

        let func = ast.items.iter().find_map(|item| match item {
            Item::Function(f) if f.name == "use" => Some(f),
            _ => None,
        });
        let func = func.expect("missing use() function");

        let param_ty = &func.params[0].ty;
        match param_ty {
            TypeExpr::Named { path, .. } => {
                assert_eq!(path.segments, vec!["Foo"]);
            }
            _ => panic!("expected named type for param"),
        }

        let ret_ty = func.return_type.as_ref().expect("missing return type");
        match ret_ty {
            TypeExpr::Named { path, .. } => {
                assert_eq!(path.segments, vec!["Foo"]);
            }
            _ => panic!("expected named type for return"),
        }
    }
}
