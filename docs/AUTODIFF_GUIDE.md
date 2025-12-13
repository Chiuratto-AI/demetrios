# Automatic Differentiation in Demetrios

## Overview

Demetrios provides **native automatic differentiation (autodiff)** as a language feature. Unlike external libraries, autodiff is built into the compiler and runtime, enabling seamless gradient computation for machine learning and scientific computing.

## Two Modes of Autodiff

### 1. Forward-Mode (Dual Numbers)

Forward-mode computes derivatives as you evaluate the function. Efficient for functions with **few inputs and many outputs**.

```d
// Forward-mode automatically tracks derivatives through computation
let f = |x: f64| -> f64 { x * x * x };
let result = f(dual(2.0, 1.0));  // Returns both value and derivative
// result.value = 8.0, result.deriv = 12.0 (3x² = 3*4 = 12)
```

**Use when:** Computing full Jacobians of vector functions, sensitivity analysis.

### 2. Reverse-Mode (Backpropagation)

Reverse-mode (backpropagation) computes all gradients in one backward pass. Efficient for functions with **many inputs and few outputs**.

```d
// Reverse-mode: compute gradient of scalar output
let f = |x: [f64]| -> f64 {
    let a = x[0] * x[1];
    let b = a + x[2];
    return b * b;
};

let x = [1.0, 2.0, 3.0];
let grads = grad(f, x);  // ∇f = [∂f/∂x₀, ∂f/∂x₁, ∂f/∂x₂]
```

**Use when:** Training neural networks, optimization, loss minimization.

## Computing Gradients

### Scalar Gradients

For a scalar function f(x: f64) → f64:

```d
fn f(x: f64) -> f64 {
    return x * x * x - 2.0 * x * x + x;
}

let x = 2.0;
let df_dx = grad(f, x);  // Returns f64
println("df/dx at x={} is {}", x, df_dx);
```

Analytical gradient:
```
f(x) = x³ - 2x² + x
df/dx = 3x² - 4x + 1
df/dx(2) = 3(4) - 4(2) + 1 = 12 - 8 + 1 = 5
```

### Vector Gradients (Jacobian)

For a vector function f(x: [f64; n]) → [f64; m]:

```d
fn f(x: [f64]) -> [f64] {
    return [x[0] * x[1], x[1] + x[2]];
}

let x = [1.0, 2.0, 3.0];
let jacobian = jacobian(f, x);  // Returns matrix (m × n)
```

### Higher-Order Derivatives

Second derivatives (Hessian matrix):

```d
fn loss(x: [f64]) -> f64 {
    return x[0]*x[0] + x[1]*x[1];  // Simple quadratic
}

let x = [1.0, 2.0];
let hess = hessian(loss, x);  // 2×2 matrix of second derivatives
```

## Applications

### 1. Neural Network Training

```d
fn neural_net(params: [f64], input: [f64]) -> f64 {
    // Simple 1-hidden-layer network
    let hidden = tanh(matrix_mul(input, params[0..4]));
    let output = matrix_mul(hidden, params[4..]);
    return output[0];
}

fn loss_fn(params: [f64]) -> f64 {
    let batch_loss = 0.0;
    for i in 0..batch_size {
        let pred = neural_net(params, data[i]);
        let error = pred - labels[i];
        batch_loss = batch_loss + error * error;
    }
    return batch_loss / batch_size as f64;
}

// Training loop
let mut params = initialize_params();
for epoch in 0..num_epochs {
    let grads = grad(loss_fn, params);
    params = update(params, grads, learning_rate);
}
```

### 2. Optimization (Gradient Descent)

```d
fn objective(x: [f64]) -> f64 {
    // Minimize: (x₀ - 3)² + (x₁ + 2)²
    let dx0 = x[0] - 3.0;
    let dx1 = x[1] + 2.0;
    return dx0*dx0 + dx1*dx1;
}

let mut x = [0.0, 0.0];
let learning_rate = 0.1;

for iter in 0..100 {
    let grads = grad(objective, x);
    for i in 0..2 {
        x[i] = x[i] - learning_rate * grads[i];
    }
    if iter % 10 == 0 {
        println("Iteration {}: loss = {}", iter, objective(x));
    }
}

println("Optimal point: {:?}", x);  // Should be [3.0, -2.0]
```

### 3. Physics-Informed Neural Networks (PINNs)

Solve PDEs using neural networks with physics constraints:

```d
fn physics_loss(params: [f64]) -> f64 {
    // PINN loss = data loss + physics loss

    // Data loss: network predictions match observations
    let data_loss = 0.0;
    for (x, y_true) in data {
        let y_pred = network(x, params);
        data_loss = data_loss + (y_pred - y_true)^2;
    }

    // Physics loss: PDE is satisfied
    // ∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²)
    let physics_loss = 0.0;
    for x in collocation_points {
        let dudt = grad_t(network(x, params));  // ∂u/∂t
        let dudx = grad_x(network(x, params));  // ∂u/∂x
        let d2udx2 = hessian_xx(network(x, params));  // ∂²u/∂x²

        let residual = dudt + u*dudx - nu*d2udx2;
        physics_loss = physics_loss + residual^2;
    }

    return data_loss + physics_loss;
}
```

### 4. Variational Inference

```d
fn evidence_lower_bound(params: [f64]) -> f64 {
    // ELBO = E_q[log p(x,z)] - E_q[log q(z)]
    // Maximize ELBO = minimize -ELBO

    let mut elbo = 0.0;

    // Sample from variational distribution
    for sample in 0..num_samples {
        let z = sample_variational(params);
        let log_pxz = log_likelihood(x, z);
        let log_qz = log_variational(z, params);
        elbo = elbo + (log_pxz - log_qz);
    }

    return -elbo / num_samples as f64;
}

// Optimize variational parameters
let mut params = initialize();
for step in 0..num_steps {
    let grads = grad(evidence_lower_bound, params);
    params = params - learning_rate * grads;
}
```

## Computational Complexity

### Time Complexity

- **Forward-mode**: O(n) for n input variables
- **Reverse-mode**: O(m) for m output variables

For neural networks (many parameters, scalar loss):
- Reverse-mode is ~3-4x the cost of forward evaluation
- Computing all gradients takes same time as computing one output

### Memory Complexity

- **Forward-mode**: O(1) additional memory
- **Reverse-mode**: O(n) to store computation graph

## Implementation Details

### The Tape

Reverse-mode tracks operations on a **computation tape**:

1. **Forward pass**: Record operations
   - Allocate node for each operation
   - Compute and store value
   - Store edges in computation graph

2. **Backward pass**: Backpropagate gradients
   - Start with ∇output = 1.0
   - For each operation in reverse:
     - Compute local gradient: ∂output/∂input
     - Accumulate: ∇input += ∇output × (∂output/∂input)

Example for multiplication z = x * y:
```
Forward:  z.value = x.value * y.value
Backward: grad_x += grad_z * y.value
          grad_y += grad_z * x.value
```

### Supported Operations

Current autodiff supports:
- Arithmetic: +, -, *, /
- Trigonometric: sin, cos, tan
- Exponential: exp, ln, sqrt
- Absolute value
- Power: x^n (n constant)
- Matrix operations: dot product, matrix multiplication

## Best Practices

### 1. Batch Processing

```d
fn batch_loss(params: [f64], batch: [(InputType, OutputType)]) -> f64 {
    let mut total_loss = 0.0;
    for (x, y) in batch {
        let pred = model(params, x);
        let loss = (pred - y)^2;
        total_loss = total_loss + loss;
    }
    return total_loss / batch.len() as f64;
}

let grads = grad(batch_loss, params, batch);  // Efficient!
```

### 2. Gradient Accumulation

```d
let mut accumulated_grads = zeros(param_count);

for batch in batches {
    let grads = grad(loss_fn, params, batch);
    accumulated_grads = accumulated_grads + grads;
}

// Apply accumulated gradients
params = params - learning_rate * (accumulated_grads / num_batches as f64);
```

### 3. Numerical Stability

```d
// Use log-sum-exp trick for numerical stability
fn stable_softmax_loss(logits: [f64], labels: [f64]) -> f64 {
    let max_logit = max(logits);
    let shifted = [logits[i] - max_logit for i in 0..n];
    let exp_shifted = [exp(shifted[i]) for i in 0..n];
    let sum_exp = sum(exp_shifted);
    let log_softmax = shifted - log(sum_exp);
    return -sum(labels * log_softmax);
}
```

## Debugging Gradients

### Numerical Gradient Checking

Verify autodiff gradients against finite differences:

```d
fn numerical_gradient(f: fn([f64])->f64, x: [f64], epsilon: f64) -> [f64] {
    let mut grads = zeros(x.len());
    for i in 0..x.len() {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] = x_plus[i] + epsilon;
        x_minus[i] = x_minus[i] - epsilon;

        grads[i] = (f(x_plus) - f(x_minus)) / (2.0 * epsilon);
    }
    return grads;
}

let auto_grads = grad(loss, params);
let num_grads = numerical_gradient(loss, params, 1e-5);

// Check they're close (within tolerance)
for i in 0..params.len() {
    let rel_error = abs(auto_grads[i] - num_grads[i]) /
                    max(abs(auto_grads[i]), abs(num_grads[i]));
    assert(rel_error < 1e-4);  // Should be small
}
```

## Performance Tips

1. **Use reverse-mode** for scalar outputs (optimization, training)
2. **Use forward-mode** for Jacobians of vector functions
3. **Vectorize** operations - GPU acceleration is automatic
4. **Cache** intermediate values when computing multiple gradients
5. **Profile** - use `dc profile` to identify bottlenecks

## Limitations and Future Work

- Nested autodiff (hessian via autodiff) coming in v0.70
- Sparse jacobians (coming v0.71)
- GPU autodiff (in development)
- Symbolic simplification of gradients (future)

## See Also

- [CAUSAL_GUIDE.md](CAUSAL_GUIDE.md) - Causal inference
- [UNITS_GUIDE.md](UNITS_GUIDE.md) - Physical units
- [Examples](../examples/) - wave2_* examples
