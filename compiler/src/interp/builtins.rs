//! Builtin function registry for the interpreter
//!
//! This module provides a flexible system for registering and calling builtin functions
//! that bridge the D interpreter with Rust runtime modules (ODE solvers, probabilistic
//! inference, symbolic math, etc).

use std::collections::HashMap;
use std::rc::Rc;

use crate::interp::value::Value;

/// Type alias for a builtin function handler
pub type BuiltinHandler = Rc<dyn Fn(&[Value]) -> Result<Value, String>>;

/// Registry of builtin functions
pub struct BuiltinRegistry {
    /// Map from function name to handler
    handlers: HashMap<String, BuiltinHandler>,
}

impl BuiltinRegistry {
    /// Create a new registry with standard builtins
    pub fn new() -> Self {
        let mut registry = BuiltinRegistry {
            handlers: HashMap::new(),
        };

        // Register I/O builtins
        registry.register_io_builtins();

        // Register math builtins
        registry.register_math_builtins();

        // Register utility builtins
        registry.register_utility_builtins();

        // Scientific primitives will be registered here
        // For now: placeholder for ODE, prob, symbolic, etc
        registry.register_scientific_builtins();

        registry
    }

    /// Register a builtin function
    pub fn register(&mut self, name: &str, handler: BuiltinHandler) {
        self.handlers.insert(name.to_string(), handler);
    }

    /// Check if a name is a registered builtin
    pub fn is_builtin(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Call a builtin function
    pub fn call(&self, name: &str, args: &[Value]) -> Result<Value, String> {
        match self.handlers.get(name) {
            Some(handler) => handler(args),
            None => Err(format!("Unknown builtin: {}", name)),
        }
    }

    /// Get all registered builtin names
    pub fn names(&self) -> Vec<String> {
        self.handlers.keys().cloned().collect()
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Builtin Categories
// ============================================================================

impl BuiltinRegistry {
    /// Register I/O builtins (print, println, etc)
    fn register_io_builtins(&mut self) {
        self.register("print", Rc::new(|args| {
            let output = args
                .iter()
                .map(|v| format!("{}", v))
                .collect::<Vec<_>>()
                .join("");
            print!("{}", output);
            Ok(Value::Unit)
        }));

        self.register("println", Rc::new(|args| {
            let output = args
                .iter()
                .map(|v| format!("{}", v))
                .collect::<Vec<_>>()
                .join("");
            println!("{}", output);
            Ok(Value::Unit)
        }));
    }

    /// Register math builtins (sqrt, sin, cos, etc)
    fn register_math_builtins(&mut self) {
        self.register("sqrt", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("sqrt expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.sqrt())),
                Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
                _ => Err(format!("sqrt expects numeric argument, got {}", args[0].type_name())),
            }
        }));

        self.register("abs", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("abs expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.abs())),
                Value::Int(n) => Ok(Value::Int(n.abs())),
                _ => Err(format!("abs expects numeric argument")),
            }
        }));

        self.register("sin", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("sin expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.sin())),
                Value::Int(n) => Ok(Value::Float((*n as f64).sin())),
                _ => Err(format!("sin expects numeric argument")),
            }
        }));

        self.register("cos", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("cos expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.cos())),
                Value::Int(n) => Ok(Value::Float((*n as f64).cos())),
                _ => Err(format!("cos expects numeric argument")),
            }
        }));

        self.register("tan", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("tan expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.tan())),
                Value::Int(n) => Ok(Value::Float((*n as f64).tan())),
                _ => Err(format!("tan expects numeric argument")),
            }
        }));

        self.register("exp", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("exp expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.exp())),
                Value::Int(n) => Ok(Value::Float((*n as f64).exp())),
                _ => Err(format!("exp expects numeric argument")),
            }
        }));

        self.register("log", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("log expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.ln())),
                Value::Int(n) => Ok(Value::Float((*n as f64).ln())),
                _ => Err(format!("log expects numeric argument")),
            }
        }));

        self.register("pow", Rc::new(|args| {
            if args.len() != 2 {
                return Err(format!("pow expects 2 arguments, got {}", args.len()));
            }
            let base = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(n) => *n as f64,
                _ => return Err(format!("pow expects numeric arguments")),
            };
            let exp = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(n) => *n as f64,
                _ => return Err(format!("pow expects numeric arguments")),
            };
            Ok(Value::Float(base.powf(exp)))
        }));

        self.register("floor", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("floor expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.floor())),
                Value::Int(n) => Ok(Value::Int(*n)),
                _ => Err(format!("floor expects numeric argument")),
            }
        }));

        self.register("ceil", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("ceil expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.ceil())),
                Value::Int(n) => Ok(Value::Int(*n)),
                _ => Err(format!("ceil expects numeric argument")),
            }
        }));

        self.register("round", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("round expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.round())),
                Value::Int(n) => Ok(Value::Int(*n)),
                _ => Err(format!("round expects numeric argument")),
            }
        }));

        self.register("min", Rc::new(|args| {
            if args.len() != 2 {
                return Err(format!("min expects 2 arguments, got {}", args.len()));
            }
            let a = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(n) => *n as f64,
                _ => return Err(format!("min expects numeric arguments")),
            };
            let b = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(n) => *n as f64,
                _ => return Err(format!("min expects numeric arguments")),
            };
            Ok(Value::Float(a.min(b)))
        }));

        self.register("max", Rc::new(|args| {
            if args.len() != 2 {
                return Err(format!("max expects 2 arguments, got {}", args.len()));
            }
            let a = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(n) => *n as f64,
                _ => return Err(format!("max expects numeric arguments")),
            };
            let b = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(n) => *n as f64,
                _ => return Err(format!("max expects numeric arguments")),
            };
            Ok(Value::Float(a.max(b)))
        }));
    }

    /// Register utility builtins (len, type_of, assert, etc)
    fn register_utility_builtins(&mut self) {
        self.register("len", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("len expects 1 argument, got {}", args.len()));
            }
            match &args[0] {
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
                _ => Err(format!("len expects string, array, or tuple")),
            }
        }));

        self.register("type_of", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("type_of expects 1 argument, got {}", args.len()));
            }
            Ok(Value::String(args[0].type_name().to_string()))
        }));

        self.register("assert", Rc::new(|args| {
            if args.len() != 1 {
                return Err(format!("assert expects 1 argument, got {}", args.len()));
            }
            if args[0].is_truthy() {
                Ok(Value::Unit)
            } else {
                Err("Assertion failed".to_string())
            }
        }));

        self.register("assert_eq", Rc::new(|args| {
            if args.len() != 2 {
                return Err(format!("assert_eq expects 2 arguments, got {}", args.len()));
            }
            if format!("{}", args[0]) == format!("{}", args[1]) {
                Ok(Value::Unit)
            } else {
                Err(format!("Assertion failed: {:?} != {:?}", args[0], args[1]))
            }
        }));

        self.register("panic", Rc::new(|args| {
            let msg = if args.is_empty() {
                "panic".to_string()
            } else {
                args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join("")
            };
            Err(format!("panic: {}", msg))
        }));

        self.register("dbg", Rc::new(|args| {
            for arg in args {
                eprintln!("[DEBUG] {}", arg);
            }
            if args.len() == 1 {
                Ok(args[0].clone())
            } else {
                Ok(Value::Unit)
            }
        }));
    }

    /// Register scientific builtins
    fn register_scientific_builtins(&mut self) {
        use crate::interp::value::{SolverStats, Distribution};

        // ODE solver (stub implementation for testing)
        self.register("solve_ode", Rc::new(|args| {
            if args.len() < 3 {
                return Err("solve_ode expects: closure, initial_values, time_span".to_string());
            }

            // For now, return a simple ODE solution
            // Later: integrate with runtime::ode::solve
            let t = vec![0.0, 0.5, 1.0, 1.5, 2.0];
            let y = vec![
                vec![1.0, 1.0],
                vec![0.9, 1.1],
                vec![0.8, 1.2],
                vec![0.7, 1.3],
                vec![0.6, 1.4],
            ];
            let stats = SolverStats {
                steps: 100,
                accepted_steps: 100,
                rejected_steps: 0,
            };

            Ok(Value::ODESolution { t, y, stats })
        }));

        // Probabilistic sampling (stub)
        self.register("sample", Rc::new(|args| {
            if args.is_empty() {
                return Err("sample expects distribution argument".to_string());
            }

            match &args[0] {
                Value::Distribution(d) => {
                    // Return a sample from the distribution
                    match d {
                        Distribution::Normal { mean, std: _ } => Ok(Value::Float(*mean)),
                        Distribution::Uniform { a, b } => {
                            Ok(Value::Float((a + b) / 2.0)) // Return midpoint for now
                        }
                        Distribution::Beta { alpha, beta } => {
                            // Return expected value E[Beta(a,b)] = a/(a+b)
                            Ok(Value::Float(alpha / (alpha + beta)))
                        }
                        Distribution::Exponential { lambda } => {
                            Ok(Value::Float(1.0 / lambda)) // Return mean
                        }
                        Distribution::Categorical { probs } => {
                            // Return index of max probability
                            let idx = probs
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                .map(|(i, _)| i)
                                .unwrap_or(0);
                            Ok(Value::Int(idx as i64))
                        }
                    }
                }
                _ => Err("sample expects a Distribution".to_string()),
            }
        }));

        // Symbolic differentiation (stub)
        self.register("differentiate", Rc::new(|args| {
            if args.len() < 2 {
                return Err("differentiate expects: expression, variable".to_string());
            }

            match (&args[0], &args[1]) {
                (Value::SymbolicExpr(expr), Value::String(var)) => {
                    use crate::interp::symbolic::Expr as SymbolicExpr;
                    let derivative = expr.differentiate(var);
                    Ok(Value::SymbolicExpr(std::rc::Rc::new(derivative)))
                }
                _ => Err("differentiate expects: SymbolicExpr, String".to_string()),
            }
        }));

        // Array/matrix operations
        self.register("zeros", Rc::new(|args| {
            if args.is_empty() {
                return Ok(Value::Float(0.0));
            }

            match &args[0] {
                Value::Int(n) => {
                    let data = vec![0.0; *n as usize];
                    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                        data.into_iter().map(Value::Float).collect(),
                    ))))
                }
                _ => Err("zeros expects integer dimension".to_string()),
            }
        }));

        self.register("ones", Rc::new(|args| {
            if args.is_empty() {
                return Ok(Value::Float(1.0));
            }

            match &args[0] {
                Value::Int(n) => {
                    let data = vec![1.0; *n as usize];
                    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                        data.into_iter().map(Value::Float).collect(),
                    ))))
                }
                _ => Err("ones expects integer dimension".to_string()),
            }
        }));

        // Gradient computation (reverse-mode autodiff)
        self.register("grad", Rc::new(|args| {
            if args.len() < 2 {
                return Err("grad expects: function, parameters".to_string());
            }

            // For now, return a placeholder gradient
            // Full implementation requires integration with interpreter
            match &args[1] {
                Value::Float(_) => {
                    // Return gradient as a float (scalar case)
                    Ok(Value::Float(0.0))  // Placeholder
                }
                Value::Array(_) => {
                    // Return gradient as an array
                    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![]))))
                }
                _ => Err("grad expects scalar or array parameters".to_string()),
            }
        }));

        // Jacobian matrix (gradient for vector functions)
        self.register("jacobian", Rc::new(|args| {
            if args.len() < 2 {
                return Err("jacobian expects: function, parameters".to_string());
            }

            // Placeholder for Jacobian computation
            // Returns a matrix of partial derivatives
            Ok(Value::Tensor {
                data: vec![],
                shape: vec![0, 0],
            })
        }));

        // Hessian matrix (second derivatives)
        self.register("hessian", Rc::new(|args| {
            if args.len() < 2 {
                return Err("hessian expects: function, parameters".to_string());
            }

            // Placeholder for Hessian computation
            // Returns a matrix of second partial derivatives
            Ok(Value::Tensor {
                data: vec![],
                shape: vec![0, 0],
            })
        }));

        // Causal do-operator: do(model, interventions)
        self.register("do", Rc::new(|args| {
            if args.len() < 2 {
                return Err("do expects: causal_model, interventions".to_string());
            }

            // For now, return a placeholder causal model
            // Full implementation requires model integration
            match &args[0] {
                Value::CausalModel(_) => {
                    Ok(Value::CausalModel("intervened_model".to_string()))
                }
                _ => Err("do expects a CausalModel".to_string()),
            }
        }));

        // Counterfactual reasoning: counterfactual(model, factual, intervention, query)
        self.register("counterfactual", Rc::new(|args| {
            if args.len() < 3 {
                return Err("counterfactual expects: model, factual_evidence, intervention".to_string());
            }

            // Placeholder for counterfactual computation
            // Returns the counterfactual value
            Ok(Value::Float(0.0))
        }));

        // Estimate average treatment effect (ATE)
        self.register("estimate_ate", Rc::new(|args| {
            if args.len() < 3 {
                return Err("estimate_ate expects: data, treatment, outcome".to_string());
            }

            // Placeholder for ATE estimation
            // Would compute E[Y | do(X=1)] - E[Y | do(X=0)]
            Ok(Value::Float(0.0))
        }));

        // Detect Simpson's paradox
        self.register("simpsons_paradox", Rc::new(|args| {
            if args.len() < 3 {
                return Err("simpsons_paradox expects: data, x, y, stratified_by_z".to_string());
            }

            // Placeholder for Simpson's paradox detection
            // Returns true if paradox is detected
            Ok(Value::Bool(false))
        }));
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = BuiltinRegistry::new();
        assert!(registry.is_builtin("print"));
        assert!(registry.is_builtin("sqrt"));
        assert!(!registry.is_builtin("unknown_function"));
    }

    #[test]
    fn test_math_builtins() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("sqrt", &[Value::Float(4.0)]).unwrap();
        assert_eq!(format!("{}", result), "2");

        let result = registry.call("abs", &[Value::Int(-5)]).unwrap();
        assert_eq!(format!("{}", result), "5");

        let result = registry.call("max", &[Value::Float(3.0), Value::Float(7.0)]).unwrap();
        assert_eq!(format!("{}", result), "7");
    }

    #[test]
    fn test_builtin_error_handling() {
        let registry = BuiltinRegistry::new();

        // Wrong number of arguments
        let result = registry.call("sqrt", &[]);
        assert!(result.is_err());

        // Wrong argument type
        let result = registry.call("sqrt", &[Value::String("hello".to_string())]);
        assert!(result.is_err());
    }

    #[test]
    fn test_scientific_builtins_registry() {
        let registry = BuiltinRegistry::new();

        // Check that scientific builtins are registered
        assert!(registry.is_builtin("solve_ode"));
        assert!(registry.is_builtin("sample"));
        assert!(registry.is_builtin("differentiate"));
        assert!(registry.is_builtin("zeros"));
        assert!(registry.is_builtin("ones"));
    }

    #[test]
    fn test_solve_ode_error_handling() {
        let registry = BuiltinRegistry::new();

        // solve_ode with insufficient arguments
        let result = registry.call("solve_ode", &[Value::Float(1.0)]);
        assert!(result.is_err());

        // solve_ode with correct number of args returns ODE solution
        let f = Value::Float(1.0);  // Dummy closure representation
        let y0 = Value::Float(1.0);
        let t_span = Value::Float(2.0);

        let result = registry.call("solve_ode", &[f, y0, t_span]);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert_eq!(solution.type_name(), "ODESolution");
    }

    #[test]
    fn test_sample_from_distribution() {
        use crate::interp::value::Distribution;

        let registry = BuiltinRegistry::new();

        // Sample from Normal distribution
        let normal = Value::Distribution(Distribution::Normal {
            mean: 5.0,
            std: 1.0,
        });

        let result = registry.call("sample", &[normal]);
        assert!(result.is_ok());

        let sample_val = result.unwrap();
        match sample_val {
            Value::Float(f) => {
                // Sample should be near the mean
                assert!((f - 5.0).abs() < 10.0);
            }
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_differentiate_symbolic() {
        use crate::interp::symbolic;
        let registry = BuiltinRegistry::new();

        // Parse "x^2" into an Expr
        let parsed_expr = symbolic::Expr::parse("x^2").expect("Failed to parse expression");
        let expr = Value::SymbolicExpr(std::rc::Rc::new(parsed_expr));
        let var = Value::String("x".to_string());

        let result = registry.call("differentiate", &[expr, var]);
        assert!(result.is_ok());

        let deriv = result.unwrap();
        assert_eq!(deriv.type_name(), "SymbolicExpr");
    }

    #[test]
    fn test_zeros_creates_array() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("zeros", &[Value::Int(5)]);
        assert!(result.is_ok());

        let arr = result.unwrap();
        assert_eq!(arr.type_name(), "array");
    }

    #[test]
    fn test_ones_creates_array() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("ones", &[Value::Int(3)]);
        assert!(result.is_ok());

        let arr = result.unwrap();
        assert_eq!(arr.type_name(), "array");
    }

    #[test]
    fn test_utility_builtins() {
        let registry = BuiltinRegistry::new();

        // Test len()
        let arr = Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
            Value::Float(1.0),
            Value::Float(2.0),
        ])));
        let result = registry.call("len", &[arr]);
        assert!(result.is_ok());

        let len_val = result.unwrap();
        match len_val {
            Value::Int(n) => assert_eq!(n, 2),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_type_of_builtin() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("type_of", &[Value::Float(3.14)]);
        assert!(result.is_ok());

        let type_name = result.unwrap();
        match type_name {
            Value::String(s) => assert_eq!(s, "float"),
            _ => panic!("Expected String"),
        }
    }
}
