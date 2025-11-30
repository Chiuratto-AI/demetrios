//! Automatic differentiation library

use linalg::{Vector, Matrix}

/// Forward mode AD using dual numbers
pub mod forward {
    /// Dual number: value + epsilon * derivative
    #[derive(Clone, Copy, Debug)]
    pub struct Dual {
        /// The primal value
        pub val: f64,

        /// The tangent (derivative)
        pub dot: f64,
    }

    impl Dual {
        pub fn new(val: f64, dot: f64) -> Self {
            Dual { val, dot }
        }

        /// Create a constant (derivative = 0)
        pub fn constant(val: f64) -> Self {
            Dual { val, dot: 0.0 }
        }

        /// Create a variable (derivative = 1)
        pub fn variable(val: f64) -> Self {
            Dual { val, dot: 1.0 }
        }

        /// Extract value
        pub fn value(&self) -> f64 { self.val }

        /// Extract derivative
        pub fn derivative(&self) -> f64 { self.dot }

        // Arithmetic operations
        pub fn add(self, other: Dual) -> Dual {
            Dual {
                val: self.val + other.val,
                dot: self.dot + other.dot,
            }
        }

        pub fn sub(self, other: Dual) -> Dual {
            Dual {
                val: self.val - other.val,
                dot: self.dot - other.dot,
            }
        }

        pub fn mul(self, other: Dual) -> Dual {
            // Product rule: (f*g)' = f'*g + f*g'
            Dual {
                val: self.val * other.val,
                dot: self.dot * other.val + self.val * other.dot,
            }
        }

        pub fn div(self, other: Dual) -> Dual {
            // Quotient rule: (f/g)' = (f'*g - f*g') / g^2
            let g2 = other.val * other.val;
            Dual {
                val: self.val / other.val,
                dot: (self.dot * other.val - self.val * other.dot) / g2,
            }
        }

        pub fn neg(self) -> Dual {
            Dual { val: -self.val, dot: -self.dot }
        }

        // Mathematical functions
        pub fn sqrt(self) -> Dual {
            let v = self.val.sqrt();
            Dual {
                val: v,
                dot: self.dot / (2.0 * v),
            }
        }

        pub fn exp(self) -> Dual {
            let e = self.val.exp();
            Dual {
                val: e,
                dot: self.dot * e,
            }
        }

        pub fn ln(self) -> Dual {
            Dual {
                val: self.val.ln(),
                dot: self.dot / self.val,
            }
        }

        pub fn pow(self, n: f64) -> Dual {
            let v = self.val.powf(n);
            Dual {
                val: v,
                dot: self.dot * n * self.val.powf(n - 1.0),
            }
        }

        pub fn sin(self) -> Dual {
            Dual {
                val: self.val.sin(),
                dot: self.dot * self.val.cos(),
            }
        }

        pub fn cos(self) -> Dual {
            Dual {
                val: self.val.cos(),
                dot: -self.dot * self.val.sin(),
            }
        }

        pub fn tan(self) -> Dual {
            let c = self.val.cos();
            Dual {
                val: self.val.tan(),
                dot: self.dot / (c * c),
            }
        }

        pub fn abs(self) -> Dual {
            Dual {
                val: self.val.abs(),
                dot: self.dot * self.val.signum(),
            }
        }

        pub fn max(self, other: Dual) -> Dual {
            if self.val >= other.val {
                self
            } else {
                other
            }
        }

        pub fn min(self, other: Dual) -> Dual {
            if self.val <= other.val {
                self
            } else {
                other
            }
        }
    }

    // Operator implementations
    impl Add for Dual {
        type Output = Dual;
        fn add(self, other: Dual) -> Dual { self.add(other) }
    }

    impl Sub for Dual {
        type Output = Dual;
        fn sub(self, other: Dual) -> Dual { self.sub(other) }
    }

    impl Mul for Dual {
        type Output = Dual;
        fn mul(self, other: Dual) -> Dual { self.mul(other) }
    }

    impl Div for Dual {
        type Output = Dual;
        fn div(self, other: Dual) -> Dual { self.div(other) }
    }

    impl Neg for Dual {
        type Output = Dual;
        fn neg(self) -> Dual { self.neg() }
    }

    /// Compute gradient using forward mode
    pub fn gradient<F>(f: F, x: &Vector<f64>) -> Vector<f64>
    where F: Fn(&Vector<Dual>) -> Dual
    {
        let n = x.len();
        let mut grad = Vector::new(n);

        for i in 0..n {
            // Set i-th variable as the differentiation variable
            let mut x_dual = Vector::new(n);
            for j in 0..n {
                x_dual[j] = if i == j {
                    Dual::variable(x[j])
                } else {
                    Dual::constant(x[j])
                };
            }

            let result = f(&x_dual);
            grad[i] = result.derivative();
        }

        grad
    }

    /// Compute directional derivative
    pub fn directional_derivative<F>(f: F, x: &Vector<f64>, v: &Vector<f64>) -> f64
    where F: Fn(&Vector<Dual>) -> Dual
    {
        let n = x.len();
        let mut x_dual = Vector::new(n);

        for i in 0..n {
            x_dual[i] = Dual::new(x[i], v[i]);
        }

        f(&x_dual).derivative()
    }

    /// Compute Jacobian-vector product (JVP)
    pub fn jvp<F>(f: F, x: &Vector<f64>, v: &Vector<f64>) -> Vector<f64>
    where F: Fn(&Vector<Dual>) -> Vector<Dual>
    {
        let n = x.len();
        let mut x_dual = Vector::new(n);

        for i in 0..n {
            x_dual[i] = Dual::new(x[i], v[i]);
        }

        let result = f(&x_dual);
        let m = result.len();
        let mut jvp_result = Vector::new(m);

        for i in 0..m {
            jvp_result[i] = result[i].derivative();
        }

        jvp_result
    }
}

/// Reverse mode AD using a tape (Wengert list)
pub mod reverse {
    use std::cell::RefCell;

    /// Node in the computation graph
    #[derive(Clone, Debug)]
    struct Node {
        /// Value at this node
        value: f64,

        /// Accumulated adjoint (∂L/∂v)
        adjoint: f64,

        /// Parent indices and local gradients
        parents: Vec<(usize, f64)>,
    }

    /// Tape recording computation graph
    thread_local! {
        static TAPE: RefCell<Vec<Node>> = RefCell::new(Vec::new());
    }

    /// Tracked value for reverse mode AD
    #[derive(Clone, Copy)]
    pub struct Var {
        /// Index in the tape
        idx: usize,

        /// Cached value
        val: f64,
    }

    impl Var {
        /// Create a new input variable
        pub fn new(val: f64) -> Self {
            TAPE.with(|tape| {
                let mut t = tape.borrow_mut();
                let idx = t.len();
                t.push(Node {
                    value: val,
                    adjoint: 0.0,
                    parents: Vec::new(),
                });
                Var { idx, val }
            })
        }

        /// Get the value
        pub fn value(&self) -> f64 { self.val }

        /// Create a node with parents
        fn from_op(val: f64, parents: Vec<(usize, f64)>) -> Self {
            TAPE.with(|tape| {
                let mut t = tape.borrow_mut();
                let idx = t.len();
                t.push(Node {
                    value: val,
                    adjoint: 0.0,
                    parents,
                });
                Var { idx, val }
            })
        }

        /// Run backward pass from this node
        pub fn backward(&self) {
            TAPE.with(|tape| {
                let mut t = tape.borrow_mut();

                // Set the adjoint of the output to 1
                t[self.idx].adjoint = 1.0;

                // Backward pass in reverse topological order
                for i in (0..=self.idx).rev() {
                    let node = &t[i];
                    let adj = node.adjoint;
                    let parents = node.parents.clone();

                    for (parent_idx, local_grad) in parents {
                        t[parent_idx].adjoint += adj * local_grad;
                    }
                }
            })
        }

        /// Get gradient after backward pass
        pub fn grad(&self) -> f64 {
            TAPE.with(|tape| {
                tape.borrow()[self.idx].adjoint
            })
        }

        // Arithmetic operations
        pub fn add(self, other: Var) -> Var {
            Var::from_op(
                self.val + other.val,
                vec![(self.idx, 1.0), (other.idx, 1.0)]
            )
        }

        pub fn sub(self, other: Var) -> Var {
            Var::from_op(
                self.val - other.val,
                vec![(self.idx, 1.0), (other.idx, -1.0)]
            )
        }

        pub fn mul(self, other: Var) -> Var {
            Var::from_op(
                self.val * other.val,
                vec![(self.idx, other.val), (other.idx, self.val)]
            )
        }

        pub fn div(self, other: Var) -> Var {
            let g2 = other.val * other.val;
            Var::from_op(
                self.val / other.val,
                vec![
                    (self.idx, 1.0 / other.val),
                    (other.idx, -self.val / g2)
                ]
            )
        }

        pub fn neg(self) -> Var {
            Var::from_op(-self.val, vec![(self.idx, -1.0)])
        }

        // Mathematical functions
        pub fn exp(self) -> Var {
            let e = self.val.exp();
            Var::from_op(e, vec![(self.idx, e)])
        }

        pub fn ln(self) -> Var {
            Var::from_op(
                self.val.ln(),
                vec![(self.idx, 1.0 / self.val)]
            )
        }

        pub fn pow(self, n: f64) -> Var {
            let v = self.val.powf(n);
            Var::from_op(
                v,
                vec![(self.idx, n * self.val.powf(n - 1.0))]
            )
        }

        pub fn sqrt(self) -> Var {
            let v = self.val.sqrt();
            Var::from_op(v, vec![(self.idx, 0.5 / v)])
        }

        pub fn sin(self) -> Var {
            Var::from_op(
                self.val.sin(),
                vec![(self.idx, self.val.cos())]
            )
        }

        pub fn cos(self) -> Var {
            Var::from_op(
                self.val.cos(),
                vec![(self.idx, -self.val.sin())]
            )
        }

        pub fn tanh(self) -> Var {
            let t = self.val.tanh();
            Var::from_op(t, vec![(self.idx, 1.0 - t * t)])
        }

        pub fn sigmoid(self) -> Var {
            let s = 1.0 / (1.0 + (-self.val).exp());
            Var::from_op(s, vec![(self.idx, s * (1.0 - s))])
        }

        pub fn relu(self) -> Var {
            if self.val > 0.0 {
                Var::from_op(self.val, vec![(self.idx, 1.0)])
            } else {
                Var::from_op(0.0, vec![(self.idx, 0.0)])
            }
        }
    }

    // Operator implementations
    impl Add for Var {
        type Output = Var;
        fn add(self, other: Var) -> Var { self.add(other) }
    }

    impl Sub for Var {
        type Output = Var;
        fn sub(self, other: Var) -> Var { self.sub(other) }
    }

    impl Mul for Var {
        type Output = Var;
        fn mul(self, other: Var) -> Var { self.mul(other) }
    }

    impl Div for Var {
        type Output = Var;
        fn div(self, other: Var) -> Var { self.div(other) }
    }

    impl Neg for Var {
        type Output = Var;
        fn neg(self) -> Var { self.neg() }
    }

    /// Clear the tape
    pub fn reset_tape() {
        TAPE.with(|tape| {
            tape.borrow_mut().clear();
        })
    }

    /// Compute gradient using reverse mode
    pub fn gradient<F>(f: F, x: &Vector<f64>) -> Vector<f64>
    where F: Fn(&Vector<Var>) -> Var
    {
        reset_tape();

        let n = x.len();
        let mut x_vars = Vector::new(n);

        for i in 0..n {
            x_vars[i] = Var::new(x[i]);
        }

        let result = f(&x_vars);
        result.backward();

        let mut grad = Vector::new(n);
        for i in 0..n {
            grad[i] = x_vars[i].grad();
        }

        grad
    }

    /// Compute vector-Jacobian product (VJP)
    pub fn vjp<F>(f: F, x: &Vector<f64>, v: &Vector<f64>) -> Vector<f64>
    where F: Fn(&Vector<Var>) -> Vector<Var>
    {
        reset_tape();

        let n = x.len();
        let mut x_vars = Vector::new(n);

        for i in 0..n {
            x_vars[i] = Var::new(x[i]);
        }

        let result = f(&x_vars);
        let m = result.len();

        // Set adjoints according to v
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            for i in 0..m {
                t[result[i].idx].adjoint = v[i];
            }

            // Backward pass
            let max_idx = result.iter().map(|r| r.idx).max().unwrap_or(0);
            for i in (0..=max_idx).rev() {
                let adj = t[i].adjoint;
                let parents = t[i].parents.clone();

                for (parent_idx, local_grad) in parents {
                    t[parent_idx].adjoint += adj * local_grad;
                }
            }
        });

        let mut vjp_result = Vector::new(n);
        for i in 0..n {
            vjp_result[i] = x_vars[i].grad();
        }

        vjp_result
    }

    /// Compute full Jacobian matrix
    pub fn jacobian<F>(f: F, x: &Vector<f64>) -> Matrix<f64>
    where F: Fn(&Vector<Var>) -> Vector<Var> + Clone
    {
        reset_tape();

        let n = x.len();

        // First pass to get output dimension
        let mut x_vars = Vector::new(n);
        for i in 0..n {
            x_vars[i] = Var::new(x[i]);
        }
        let result = f(&x_vars);
        let m = result.len();

        // Compute Jacobian column by column using VJP
        let mut jac = Matrix::zeros(m, n);

        for i in 0..m {
            let mut v = Vector::zeros(m);
            v[i] = 1.0;

            let col = vjp(f.clone(), x, &v);
            for j in 0..n {
                jac[(i, j)] = col[j];
            }
        }

        jac
    }

    /// Compute Hessian matrix
    pub fn hessian<F>(f: F, x: &Vector<f64>) -> Matrix<f64>
    where F: Fn(&Vector<Var>) -> Var + Clone
    {
        let n = x.len();
        let mut hess = Matrix::zeros(n, n);

        // Use finite differences on the gradient
        let eps = 1e-7;
        let grad_at_x = gradient(f.clone(), x);

        for i in 0..n {
            let mut x_plus = x.clone();
            x_plus[i] += eps;

            let grad_plus = gradient(f.clone(), &x_plus);

            for j in 0..n {
                hess[(i, j)] = (grad_plus[j] - grad_at_x[j]) / eps;
            }
        }

        // Symmetrize
        for i in 0..n {
            for j in i+1..n {
                let avg = (hess[(i, j)] + hess[(j, i)]) / 2.0;
                hess[(i, j)] = avg;
                hess[(j, i)] = avg;
            }
        }

        hess
    }
}
