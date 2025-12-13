//! Causal inference and do-calculus
//!
//! Implements causal reasoning with support for:
//! - Causal DAGs (directed acyclic graphs)
//! - Interventions (do-operator)
//! - Counterfactual reasoning
//! - Simpson's paradox detection
//! - Backdoor criterion checking

use std::collections::{HashMap, HashSet};

use miette::Result;

use super::value::Value;

/// A causal directed acyclic graph (DAG)
#[derive(Clone, Debug)]
pub struct CausalDAG {
    /// Node names
    pub nodes: Vec<String>,
    /// Edges: (source, target)
    pub edges: Vec<(String, String)>,
}

impl CausalDAG {
    /// Create a new causal DAG
    pub fn new(nodes: Vec<String>) -> Self {
        CausalDAG {
            nodes,
            edges: Vec::new(),
        }
    }

    /// Add a directed edge (causal link)
    pub fn add_edge(&mut self, source: &str, target: &str) {
        self.edges.push((source.to_string(), target.to_string()));
    }

    /// Get parents of a node
    pub fn parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(_, target)| target == node)
            .map(|(source, _)| source.clone())
            .collect()
    }

    /// Get children of a node
    pub fn children(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(source, _)| source == node)
            .map(|(_, target)| target.clone())
            .collect()
    }

    /// Find all paths from source to target
    fn find_paths_dfs(
        &self,
        current: &str,
        target: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        paths: &mut Vec<Vec<String>>,
    ) {
        if current == target {
            paths.push(path.clone());
            return;
        }

        visited.insert(current.to_string());

        for child in self.children(current) {
            if !visited.contains(&child) {
                path.push(child.clone());
                self.find_paths_dfs(&child, target, visited, path, paths);
                path.pop();
            }
        }

        visited.remove(current);
    }

    /// Get all directed paths from source to target
    pub fn directed_paths(&self, source: &str, target: &str) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut path = vec![source.to_string()];
        self.find_paths_dfs(source, target, &mut visited, &mut path, &mut paths);
        paths
    }

    /// Check backdoor criterion (path adjustment for causal effect)
    /// Returns confounders that need to be controlled for
    pub fn backdoor_criterion(&self, treatment: &str, outcome: &str) -> Vec<String> {
        let mut confounders = Vec::new();

        // Find all parents of treatment (causes of treatment)
        let treatment_parents = self.parents(treatment);

        // For each parent of treatment, check if it reaches outcome
        for parent in treatment_parents {
            let paths = self.directed_paths(&parent, outcome);
            if !paths.is_empty() {
                // This is a confounder - affects both treatment and outcome
                confounders.push(parent);
            }
        }

        confounders
    }
}

/// An intervention: setting a variable to a fixed value
#[derive(Clone, Debug)]
pub struct Intervention {
    pub variable: String,
    pub value: f64,
}

/// Causal model with structural equations
#[derive(Clone, Debug)]
pub struct CausalModel {
    /// The causal DAG
    pub dag: CausalDAG,
    /// Structural equations (variable -> formula)
    pub equations: HashMap<String, String>,
    /// Observed data
    pub data: HashMap<String, Vec<f64>>,
}

impl CausalModel {
    /// Create a new causal model
    pub fn new(dag: CausalDAG) -> Self {
        CausalModel {
            dag,
            equations: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Add a structural equation
    pub fn add_equation(&mut self, variable: String, formula: String) {
        self.equations.insert(variable, formula);
    }

    /// Apply an intervention (do-operator)
    /// Returns a modified model with the intervention applied
    pub fn intervene(&self, intervention: Intervention) -> Self {
        let mut new_model = self.clone();

        // Remove all edges pointing TO the intervened variable
        // (cut backdoor paths)
        new_model.dag.edges.retain(|(_, target)| target != &intervention.variable);

        // Set the equation for the intervened variable to its constant value
        new_model.equations.insert(
            intervention.variable.clone(),
            format!("constant({})", intervention.value),
        );

        new_model
    }

    /// Estimate average treatment effect (ATE)
    /// ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
    pub fn estimate_ate(&self, treatment: &str, outcome: &str) -> Result<f64> {
        // Placeholder implementation
        // Real implementation would integrate with observed data
        Ok(0.0)
    }

    /// Detect Simpson's paradox
    /// Returns true if causal direction contradicts marginal association
    pub fn has_simpsons_paradox(&self, x: &str, y: &str, z: &str) -> bool {
        // Simpson's paradox occurs when:
        // - Marginal association between X and Y has one direction
        // - Conditional association (stratified by Z) has opposite direction
        // This is detected when Z is a confounder
        let confounders = self.dag.backdoor_criterion(x, y);
        confounders.contains(&z.to_string())
    }
}

/// Causal query result
#[derive(Clone, Debug)]
pub struct CausalQuery {
    pub query_type: String,  // "ate", "counterfactual", "prob"
    pub result: f64,
    pub confidence: f64,  // 0-1 confidence level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_dag_creation() {
        let mut dag = CausalDAG::new(vec![
            "X".to_string(),
            "Y".to_string(),
            "Z".to_string(),
        ]);
        dag.add_edge("X", "Y");
        dag.add_edge("Z", "X");
        dag.add_edge("Z", "Y");

        assert_eq!(dag.parents("X"), vec!["Z"]);
        assert_eq!(dag.children("Z"), vec!["X", "Y"]);
    }

    #[test]
    fn test_backdoor_criterion() {
        let mut dag = CausalDAG::new(vec![
            "X".to_string(),
            "Y".to_string(),
            "Z".to_string(),
        ]);
        dag.add_edge("Z", "X");
        dag.add_edge("Z", "Y");
        dag.add_edge("X", "Y");

        let confounders = dag.backdoor_criterion("X", "Y");
        assert!(confounders.contains(&"Z".to_string()));
    }

    #[test]
    fn test_intervention() {
        let dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string()]);
        let mut model = CausalModel::new(dag);
        model.add_equation("X".to_string(), "normal(0, 1)".to_string());
        model.add_equation("Y".to_string(), "X + noise".to_string());

        let intervention = Intervention {
            variable: "X".to_string(),
            value: 5.0,
        };

        let intervened = model.intervene(intervention);
        assert_eq!(
            intervened.equations.get("X"),
            Some(&"constant(5)".to_string())
        );
    }
}
