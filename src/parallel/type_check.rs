//! Parallel type checking with dependency-aware scheduling (simplified)

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::ast::*;
use crate::hir::Hir;
use rayon::prelude::*;

/// Dependency graph for parallel type checking
#[derive(Debug)]
pub struct DependencyGraph {
    /// Nodes in the graph
    nodes: HashMap<NodeKey, DependencyNode>,

    /// Edges: key depends on value
    edges: HashMap<NodeKey, HashSet<NodeKey>>,

    /// Reverse edges: key is depended on by value
    reverse_edges: HashMap<NodeKey, HashSet<NodeKey>>,
}

/// A key identifying a compilation unit
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum NodeKey {
    /// A file
    File(PathBuf),

    /// A function in a file
    Function(PathBuf, String),

    /// A type in a file
    Type(PathBuf, String),
}

/// A node in the dependency graph
#[derive(Debug)]
struct DependencyNode {
    key: NodeKey,
    status: NodeStatus,
}

/// Status of a node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    /// Not yet processed
    Pending,

    /// Currently being processed
    InProgress,

    /// Successfully completed
    Complete,

    /// Failed with error
    Failed,
}

impl DependencyGraph {
    pub fn new() -> Self {
        DependencyGraph {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, key: NodeKey) {
        if !self.nodes.contains_key(&key) {
            self.nodes.insert(
                key.clone(),
                DependencyNode {
                    key: key.clone(),
                    status: NodeStatus::Pending,
                },
            );
            self.edges.insert(key.clone(), HashSet::new());
            self.reverse_edges.insert(key, HashSet::new());
        }
    }

    /// Add an edge: `from` depends on `to`
    pub fn add_edge(&mut self, from: NodeKey, to: NodeKey) {
        self.add_node(from.clone());
        self.add_node(to.clone());

        self.edges.get_mut(&from).unwrap().insert(to.clone());
        self.reverse_edges.get_mut(&to).unwrap().insert(from);
    }

    /// Get nodes with no pending dependencies
    pub fn ready_nodes(&self) -> Vec<NodeKey> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.status == NodeStatus::Pending)
            .filter(|(key, _)| {
                self.edges
                    .get(*key)
                    .map(|deps| {
                        deps.iter().all(|dep| {
                            self.nodes
                                .get(dep)
                                .map(|n| n.status == NodeStatus::Complete)
                                .unwrap_or(true)
                        })
                    })
                    .unwrap_or(true)
            })
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Mark a node as in progress
    pub fn mark_in_progress(&mut self, key: &NodeKey) {
        if let Some(node) = self.nodes.get_mut(key) {
            node.status = NodeStatus::InProgress;
        }
    }

    /// Mark a node as complete
    pub fn mark_complete(&mut self, key: &NodeKey) {
        if let Some(node) = self.nodes.get_mut(key) {
            node.status = NodeStatus::Complete;
        }
    }

    /// Mark a node as failed
    pub fn mark_failed(&mut self, key: &NodeKey) {
        if let Some(node) = self.nodes.get_mut(key) {
            node.status = NodeStatus::Failed;
        }
    }

    /// Check if all nodes are complete
    pub fn is_complete(&self) -> bool {
        self.nodes
            .values()
            .all(|n| n.status == NodeStatus::Complete || n.status == NodeStatus::Failed)
    }

    /// Topological sort
    pub fn topological_order(&self) -> Vec<NodeKey> {
        let mut result = Vec::new();
        let mut in_degree: HashMap<&NodeKey, usize> = HashMap::new();

        for key in self.nodes.keys() {
            in_degree.insert(key, self.edges.get(key).map(|e| e.len()).unwrap_or(0));
        }

        let mut queue: VecDeque<&NodeKey> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(k, _)| *k)
            .collect();

        while let Some(key) = queue.pop_front() {
            result.push(key.clone());

            if let Some(dependents) = self.reverse_edges.get(key) {
                for dep in dependents {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
        }

        result
    }
}

/// Parallel type checker
pub struct ParallelTypeChecker {
    /// Dependency graph
    graph: Arc<Mutex<DependencyGraph>>,

    /// Number of worker threads
    num_threads: usize,
}

impl ParallelTypeChecker {
    pub fn new(num_threads: usize) -> Self {
        ParallelTypeChecker {
            graph: Arc::new(Mutex::new(DependencyGraph::new())),
            num_threads,
        }
    }

    /// Build dependency graph from ASTs
    pub fn build_graph(&self, files: &[(PathBuf, Ast)]) {
        let mut graph = self.graph.lock().unwrap();

        for (path, ast) in files {
            // Add file node
            graph.add_node(NodeKey::File(path.clone()));

            // Analyze items for dependencies
            for item in &ast.items {
                match item {
                    Item::Function(f) => {
                        let key = NodeKey::Function(path.clone(), f.name.clone());
                        graph.add_node(key.clone());

                        // Add dependency on file
                        graph.add_edge(key.clone(), NodeKey::File(path.clone()));

                        // Add dependencies on used types
                        self.collect_type_deps(&f.return_type, path, &key, &mut graph);
                        for param in &f.params {
                            self.collect_type_deps(&Some(param.ty.clone()), path, &key, &mut graph);
                        }
                    }

                    Item::Struct(s) => {
                        let key = NodeKey::Type(path.clone(), s.name.clone());
                        graph.add_node(key.clone());

                        // Add dependency on file
                        graph.add_edge(key.clone(), NodeKey::File(path.clone()));

                        // Add dependencies on field types
                        for field in &s.fields {
                            self.collect_type_deps(&Some(field.ty.clone()), path, &key, &mut graph);
                        }
                    }

                    _ => {}
                }
            }
        }
    }

    fn collect_type_deps(
        &self,
        ty: &Option<TypeExpr>,
        current_file: &PathBuf,
        from: &NodeKey,
        graph: &mut DependencyGraph,
    ) {
        let Some(ty) = ty else {
            return;
        };
        match ty {
            TypeExpr::Named { path, args, .. } => {
                let name = path.segments.last().cloned().unwrap_or_default();
                let dep = NodeKey::Type(current_file.clone(), name);
                if graph.nodes.contains_key(&dep) {
                    graph.add_edge(from.clone(), dep);
                }

                for arg in args {
                    self.collect_type_deps(&Some(arg.clone()), current_file, from, graph);
                }
            }

            TypeExpr::Function {
                params,
                return_type,
                effects: _,
            } => {
                for param in params {
                    self.collect_type_deps(&Some(param.clone()), current_file, from, graph);
                }
                self.collect_type_deps(&Some(*return_type.clone()), current_file, from, graph);
            }

            TypeExpr::Reference { inner, .. } => {
                self.collect_type_deps(&Some(*inner.clone()), current_file, from, graph);
            }

            TypeExpr::Array { element, .. } => {
                self.collect_type_deps(&Some(*element.clone()), current_file, from, graph);
            }

            TypeExpr::Tuple(inner) => {
                for elem in inner {
                    self.collect_type_deps(&Some(elem.clone()), current_file, from, graph);
                }
            }

            _ => {}
        }
    }

    /// Type check all files in parallel. Returns a map from path to HIR.
    pub fn check_all(&self, files: Vec<(PathBuf, Ast)>) -> HashMap<PathBuf, Hir> {
        self.build_graph(&files);

        let results: Arc<Mutex<HashMap<PathBuf, Hir>>> = Arc::new(Mutex::new(HashMap::new()));

        // Create work-stealing thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .unwrap();

        pool.scope(|s| {
            for (path, ast) in files {
                let results = Arc::clone(&results);
                s.spawn(move |_| {
                    if let Ok(hir) = crate::check::check(&ast) {
                        results.lock().unwrap().insert(path, hir);
                    }
                });
            }
        });

        Arc::try_unwrap(results).unwrap().into_inner().unwrap()
    }
}
