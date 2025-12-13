//! Reverse-mode automatic differentiation (backpropagation)
//!
//! Implements tape-based reverse-mode AD for computing gradients efficiently.
//! This is the foundation for neural networks and optimization.
//!
//! Algorithm:
//! 1. Forward pass: Record all operations on a tape
//! 2. Backward pass: Backpropagate gradients from output to inputs
//! 3. Result: Gradients of output w.r.t. each input

use std::collections::HashMap;
use std::rc::Rc;

use miette::Result;

use super::value::Value;

/// An operation recorded on the autodiff tape
#[derive(Clone, Debug)]
pub enum TapeOp {
    /// Constant value (no dependencies)
    Const { output_id: usize, value: f64 },

    /// Addition: z = x + y
    Add { output_id: usize, left_id: usize, right_id: usize },

    /// Subtraction: z = x - y
    Sub { output_id: usize, left_id: usize, right_id: usize },

    /// Multiplication: z = x * y
    Mul { output_id: usize, left_id: usize, right_id: usize },

    /// Division: z = x / y
    Div { output_id: usize, left_id: usize, right_id: usize },

    /// Power: z = x ^ y (y must be constant)
    Pow { output_id: usize, base_id: usize, exponent: f64 },

    /// Sine: z = sin(x)
    Sin { output_id: usize, input_id: usize },

    /// Cosine: z = cos(x)
    Cos { output_id: usize, input_id: usize },

    /// Exponential: z = exp(x)
    Exp { output_id: usize, input_id: usize },

    /// Natural logarithm: z = ln(x)
    Ln { output_id: usize, input_id: usize },

    /// Square root: z = sqrt(x)
    Sqrt { output_id: usize, input_id: usize },

    /// Absolute value: z = |x|
    Abs { output_id: usize, input_id: usize },
}

/// Autodiff tape for recording computation graph
#[derive(Clone, Debug)]
pub struct Tape {
    /// All operations in order
    operations: Vec<TapeOp>,
    /// Values computed during forward pass
    values: Vec<f64>,
    /// Node ID counter
    next_id: usize,
    /// Map from variable name to node ID (for inputs)
    variables: HashMap<String, usize>,
}

impl Tape {
    /// Create a new tape
    pub fn new() -> Self {
        Tape {
            operations: Vec::new(),
            values: vec![],
            next_id: 0,
            variables: HashMap::new(),
        }
    }

    /// Allocate a new node ID
    fn allocate(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.values.push(0.0);  // Placeholder
        id
    }

    /// Record a constant value
    pub fn constant(&mut self, value: f64) -> TapeNode {
        let id = self.allocate();
        self.values[id] = value;
        self.operations.push(TapeOp::Const {
            output_id: id,
            value,
        });
        TapeNode { tape: None, id }
    }

    /// Record a variable (independent input)
    pub fn variable(&mut self, value: f64) -> TapeNode {
        let id = self.allocate();
        self.values[id] = value;
        TapeNode { tape: None, id }
    }

    /// Get value at a node
    pub fn get_value(&self, node_id: usize) -> f64 {
        self.values[node_id]
    }

    /// Backpropagate from output node to compute gradients
    pub fn backward(&mut self, output_id: usize) -> Result<HashMap<usize, f64>> {
        let mut gradients = HashMap::new();
        gradients.insert(output_id, 1.0);  // dL/dL = 1

        // Iterate operations in reverse order
        for op in self.operations.iter().rev() {
            match op {
                TapeOp::Const { .. } => {
                    // Constants have no upstream gradients
                }

                TapeOp::Add { output_id, left_id, right_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        // dL/dx = dL/dz * dz/dx = dL/dz * 1
                        // dL/dy = dL/dz * dz/dy = dL/dz * 1
                        *gradients.entry(*left_id).or_insert(0.0) += grad_output;
                        *gradients.entry(*right_id).or_insert(0.0) += grad_output;
                    }
                }

                TapeOp::Sub { output_id, left_id, right_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        // dL/dx = dL/dz * dz/dx = dL/dz * 1
                        // dL/dy = dL/dz * dz/dy = dL/dz * (-1)
                        *gradients.entry(*left_id).or_insert(0.0) += grad_output;
                        *gradients.entry(*right_id).or_insert(0.0) -= grad_output;
                    }
                }

                TapeOp::Mul { output_id, left_id, right_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*left_id];
                        let y = self.values[*right_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * y
                        // dL/dy = dL/dz * dz/dy = dL/dz * x
                        *gradients.entry(*left_id).or_insert(0.0) += grad_output * y;
                        *gradients.entry(*right_id).or_insert(0.0) += grad_output * x;
                    }
                }

                TapeOp::Div { output_id, left_id, right_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*left_id];
                        let y = self.values[*right_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * (1/y)
                        // dL/dy = dL/dz * dz/dy = dL/dz * (-x/y²)
                        let dy_inv = 1.0 / y;
                        *gradients.entry(*left_id).or_insert(0.0) += grad_output * dy_inv;
                        *gradients.entry(*right_id).or_insert(0.0) += grad_output * (-x / (y * y));
                    }
                }

                TapeOp::Pow { output_id, base_id, exponent } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*base_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * y * x^(y-1)
                        let grad = grad_output * exponent * x.powf(exponent - 1.0);
                        *gradients.entry(*base_id).or_insert(0.0) += grad;
                    }
                }

                TapeOp::Sin { output_id, input_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*input_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * cos(x)
                        let grad = grad_output * x.cos();
                        *gradients.entry(*input_id).or_insert(0.0) += grad;
                    }
                }

                TapeOp::Cos { output_id, input_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*input_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * (-sin(x))
                        let grad = grad_output * (-x.sin());
                        *gradients.entry(*input_id).or_insert(0.0) += grad;
                    }
                }

                TapeOp::Exp { output_id, input_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let z = self.values[*output_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * exp(x) = dL/dz * z
                        let grad = grad_output * z;
                        *gradients.entry(*input_id).or_insert(0.0) += grad;
                    }
                }

                TapeOp::Ln { output_id, input_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*input_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * (1/x)
                        let grad = grad_output / x;
                        *gradients.entry(*input_id).or_insert(0.0) += grad;
                    }
                }

                TapeOp::Sqrt { output_id, input_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*input_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * (1/(2*sqrt(x)))
                        let grad = grad_output / (2.0 * x.sqrt());
                        *gradients.entry(*input_id).or_insert(0.0) += grad;
                    }
                }

                TapeOp::Abs { output_id, input_id } => {
                    let grad_output = gradients.get(output_id).copied().unwrap_or(0.0);
                    if grad_output != 0.0 {
                        let x = self.values[*input_id];
                        // dL/dx = dL/dz * dz/dx = dL/dz * sign(x)
                        let sign = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
                        let grad = grad_output * sign;
                        *gradients.entry(*input_id).or_insert(0.0) += grad;
                    }
                }
            }
        }

        Ok(gradients)
    }
}

/// A node in the autodiff tape (represents a computation)
#[derive(Clone)]
pub struct TapeNode {
    /// Reference to the tape (None for temporary values)
    pub tape: Option<Rc<std::cell::RefCell<Tape>>>,
    /// Node ID in the tape
    pub id: usize,
}

impl TapeNode {
    /// Add two tape nodes
    pub fn add(&self, other: &TapeNode, tape: &mut Tape) -> TapeNode {
        let output_id = tape.allocate();
        let left_val = tape.get_value(self.id);
        let right_val = tape.get_value(other.id);
        tape.values[output_id] = left_val + right_val;
        tape.operations.push(TapeOp::Add {
            output_id,
            left_id: self.id,
            right_id: other.id,
        });
        TapeNode { tape: None, id: output_id }
    }

    /// Multiply two tape nodes
    pub fn mul(&self, other: &TapeNode, tape: &mut Tape) -> TapeNode {
        let output_id = tape.allocate();
        let left_val = tape.get_value(self.id);
        let right_val = tape.get_value(other.id);
        tape.values[output_id] = left_val * right_val;
        tape.operations.push(TapeOp::Mul {
            output_id,
            left_id: self.id,
            right_id: other.id,
        });
        TapeNode { tape: None, id: output_id }
    }

    /// Get the value at this node
    pub fn value(&self, tape: &Tape) -> f64 {
        tape.get_value(self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_addition() {
        let mut tape = Tape::new();
        let x = tape.variable(2.0);
        let y = tape.variable(3.0);
        let z = x.add(&y, &mut tape);

        // z = 2 + 3 = 5
        assert_eq!(tape.get_value(z.id), 5.0);

        // Compute gradients
        let grads = tape.backward(z.id).unwrap();
        assert_eq!(grads.get(&x.id).copied().unwrap_or(0.0), 1.0);
        assert_eq!(grads.get(&y.id).copied().unwrap_or(0.0), 1.0);
    }

    #[test]
    fn test_simple_multiplication() {
        let mut tape = Tape::new();
        let x = tape.variable(3.0);
        let y = tape.variable(4.0);
        let z = x.mul(&y, &mut tape);

        // z = 3 * 4 = 12
        assert_eq!(tape.get_value(z.id), 12.0);

        // Compute gradients: dz/dx = y = 4, dz/dy = x = 3
        let grads = tape.backward(z.id).unwrap();
        assert_eq!(grads.get(&x.id).copied().unwrap_or(0.0), 4.0);
        assert_eq!(grads.get(&y.id).copied().unwrap_or(0.0), 3.0);
    }
}
