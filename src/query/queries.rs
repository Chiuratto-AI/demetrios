//! Concrete query definitions for the compiler

use std::path::PathBuf;
use std::sync::Arc;

use super::database::{Durability, QueryDatabase, QueryKey};
use crate::ast::Ast;
use crate::hir::Hir;
use crate::resolve::ResolvedAst;
use crate::types::Type;

/// Query key for file contents
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FileContents(pub PathBuf);

impl QueryKey for FileContents {
    type Value = Arc<String>;
}

/// Query key for parsed AST
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ParsedAst(pub PathBuf);

impl QueryKey for ParsedAst {
    type Value = Arc<Ast>;
}

/// Query key for resolved AST
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ResolvedAstQuery(pub PathBuf);

impl QueryKey for ResolvedAstQuery {
    type Value = Arc<ResolvedAst>;
}

/// Query key for type-checked AST (produces HIR)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TypeCheckedAst(pub PathBuf);

impl QueryKey for TypeCheckedAst {
    type Value = Arc<Hir>;
}

/// Query key for HIR (alias for type-checked AST)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct HirQuery(pub PathBuf);

impl QueryKey for HirQuery {
    type Value = Arc<Hir>;
}

/// Query key for function signature
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FunctionSignature {
    pub file: PathBuf,
    pub name: String,
}

impl QueryKey for FunctionSignature {
    type Value = Arc<Type>;
}

/// Query key for module dependencies
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ModuleDependencies(pub PathBuf);

impl QueryKey for ModuleDependencies {
    type Value = Arc<Vec<PathBuf>>;
}

/// Extension trait for QueryDatabase with compiler queries
pub trait CompilerQueries {
    /// Get file contents (input)
    fn file_contents(&self, path: PathBuf) -> Arc<String>;

    /// Set file contents (input)
    fn set_file_contents(&self, path: PathBuf, contents: String);

    /// Get parsed AST
    fn parsed_ast(&self, path: PathBuf) -> Arc<Ast>;

    /// Get resolved AST
    fn resolved_ast(&self, path: PathBuf) -> Arc<ResolvedAst>;

    /// Get type-checked AST (HIR)
    fn type_checked_ast(&self, path: PathBuf) -> Arc<Hir>;

    /// Get HIR
    fn hir(&self, path: PathBuf) -> Arc<Hir>;

    /// Get function signature (approximate)
    fn function_signature(&self, file: PathBuf, name: String) -> Arc<Type>;

    /// Get module dependencies
    fn module_dependencies(&self, path: PathBuf) -> Arc<Vec<PathBuf>>;
}

impl CompilerQueries for QueryDatabase {
    fn file_contents(&self, path: PathBuf) -> Arc<String> {
        self.query(FileContents(path.clone()), |_db, key| {
            let contents = std::fs::read_to_string(&key.0).unwrap_or_default();
            Arc::new(contents)
        })
    }

    fn set_file_contents(&self, path: PathBuf, contents: String) {
        self.set_input(FileContents(path), Arc::new(contents), Durability::Low);
    }

    fn parsed_ast(&self, path: PathBuf) -> Arc<Ast> {
        self.query(ParsedAst(path.clone()), |db, key| {
            let contents = db.file_contents(key.0.clone());
            let tokens = match crate::lexer::lex(&contents) {
                Ok(tokens) => tokens,
                Err(_) => return Arc::new(Ast::default()),
            };
            let ast = crate::parser::parse(&tokens, &contents).unwrap_or_default();
            Arc::new(ast)
        })
    }

    fn resolved_ast(&self, path: PathBuf) -> Arc<ResolvedAst> {
        self.query(ResolvedAstQuery(path.clone()), |db, key| {
            let ast = db.parsed_ast(key.0.clone());
            let resolved =
                crate::resolve::resolve((*ast).clone()).unwrap_or_else(|_| ResolvedAst {
                    ast: Ast::default(),
                    symbols: crate::resolve::SymbolTable::default(),
                });
            Arc::new(resolved)
        })
    }

    fn type_checked_ast(&self, path: PathBuf) -> Arc<Hir> {
        self.query(TypeCheckedAst(path.clone()), |db, key| {
            let ast = db.parsed_ast(key.0.clone());
            let hir = crate::check::check(&ast).unwrap_or_else(|_| Hir { items: Vec::new() });
            Arc::new(hir)
        })
    }

    fn hir(&self, path: PathBuf) -> Arc<Hir> {
        self.query(HirQuery(path.clone()), |db, key| {
            db.type_checked_ast(key.0.clone())
        })
    }

    fn function_signature(&self, file: PathBuf, name: String) -> Arc<Type> {
        self.query(
            FunctionSignature {
                file: file.clone(),
                name: name.clone(),
            },
            |db, key| {
                let ast = db.parsed_ast(key.file.clone());

                for item in &ast.items {
                    if let crate::ast::Item::Function(f) = item {
                        if f.name == key.name {
                            // Represent the signature as a function type using return type if available
                            if let Some(ret) = &f.return_type {
                                let ret_ty = Type::Named {
                                    name: format!("{ret:?}"),
                                    args: Vec::new(),
                                };
                                return Arc::new(Type::Function {
                                    params: vec![Type::Unknown; f.params.len()],
                                    return_type: Box::new(ret_ty),
                                    effects: crate::types::EffectSet::default(),
                                });
                            }
                        }
                    }
                }

                Arc::new(Type::Unknown)
            },
        )
    }

    fn module_dependencies(&self, path: PathBuf) -> Arc<Vec<PathBuf>> {
        self.query(ModuleDependencies(path.clone()), |db, key| {
            let ast = db.parsed_ast(key.0.clone());

            let mut deps = Vec::new();
            for item in &ast.items {
                if let crate::ast::Item::Import(import) = item {
                    if !import.path.segments.is_empty() {
                        let dep_path = import.path.segments.join("/") + ".d";
                        deps.push(PathBuf::from(dep_path));
                    }
                }
            }

            Arc::new(deps)
        })
    }
}
