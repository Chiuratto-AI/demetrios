# Automatic Differentiation in Demetrios

Demetrios provides built-in support for **automatic differentiation (autodiff)** using forward-mode dual numbers. This enables exact computation of derivatives, gradients, Jacobians, and Hessians without numerical approximation errors.

## Overview

Automatic differentiation is a technique that computes derivatives by applying the chain rule systematically through a computation. Unlike:
- **Symbolic differentiation**: Which produces expressions (can be complex)
- **Numerical differentiation**: Which uses finite differences (introduces errors)

Autodiff computes **exact derivatives** at the cost of augmented arithmetic operations.

## Dual Numbers

A **dual number** is a pair `(value, derivative)` where:
- `value` is the actual function value
- `derivative` tracks how the value changes with respect to an input variable

### Dual Number Arithmetic

```d
// Dual number type: dual
let x = dual(3.0, 1.0);  // value=3.0, derivative=1.0 (seed)

// Arithmetic follows calculus rules:
// Addition: (a, a') + (b, b') = (a + b, a' + b')
// Subtraction: (a, a') - (b, b') = (a - b, a' - b')
// Multiplication (product rule): (a, a') * (b, b') = (a*b, a'*b + a*b')
// Division (quotient rule): (a, a') / (b, b') = (a/b, (a'*b - a*b') / b^2)
```

## Basic Usage

### Computing Simple Derivatives

```d
// f(x) = x^2
// f'(x) = 2x
// At x = 3: f'(3) = 6

let x = dual(3.0, 1.0);  // Seed derivative with 1.0
let f_x = x * x;         // Product rule: (3, 1) * (3, 1) = (9, 6)

let value = dual_value(f_x);  // 9.0
let deriv = dual_deriv(f_x);  // 6.0
```

### Chain Rule Composition

```d
// g(x) = (2x + 1)^2
// g'(x) = 2 * (2x + 1) * 2 = 4(2x + 1)
// At x = 1: g'(1) = 4 * 3 = 12

let x = dual(1.0, 1.0);
let two = dual(2.0, 0.0);   // Constant (derivative = 0)
let one = dual(1.0, 0.0);

let inner = two * x + one;   // 2x + 1
let g_x = inner * inner;     // (2x + 1)^2

let deriv = dual_deriv(g_x); // 12.0
```

## Builtin Functions

### `dual(value: f64, derivative: f64) -> dual`

Creates a dual number with specified value and derivative.

```d
let x = dual(5.0, 1.0);  // Variable: dx/dx = 1
let c = dual(2.0, 0.0);  // Constant: dc/dx = 0
```

### `dual_value(d: dual) -> f64`

Extracts the value component from a dual number.

```d
let x = dual(3.0, 1.0);
let v = dual_value(x);  // 3.0
```

### `dual_deriv(d: dual) -> f64`

Extracts the derivative component from a dual number.

```d
let x = dual(3.0, 1.0);
let d = dual_deriv(x);  // 1.0
```

### `grad(f: fn(dual) -> dual, x: f64) -> f64`

Computes the gradient (derivative) of a scalar function at a point.

```d
fn square(x: dual) -> dual {
    x * x
}

let derivative = grad(square, 3.0);  // 6.0 (d/dx[x^2] at x=3)
```

### `jacobian(f: fn([dual]) -> [dual], x: [f64]) -> [[f64]]`

Computes the Jacobian matrix of a vector-valued function.

```d
fn vector_func(x: [dual]) -> [dual] {
    // f1(x, y) = x^2 + y
    // f2(x, y) = x * y
    [x[0] * x[0] + x[1], x[0] * x[1]]
}

// Jacobian at (1, 2):
// | df1/dx  df1/dy |   | 2x  1 |   | 2  1 |
// | df2/dx  df2/dy | = |  y  x | = | 2  1 |
let J = jacobian(vector_func, [1.0, 2.0]);
```

### `hessian(f: fn([dual]) -> dual, x: [f64]) -> [[f64]]`

Computes the Hessian matrix (second derivatives) of a scalar function.

```d
fn quadratic(x: [dual]) -> dual {
    // f(x, y) = x^2 + 2xy + y^2
    x[0] * x[0] + dual(2.0, 0.0) * x[0] * x[1] + x[1] * x[1]
}

// Hessian:
// | d^2f/dx^2   d^2f/dxdy |   | 2  2 |
// | d^2f/dydx   d^2f/dy^2 | = | 2  2 |
let H = hessian(quadratic, [1.0, 1.0]);
```

## Mathematical Functions

All standard mathematical functions support dual number propagation:

| Function | Derivative Rule |
|----------|----------------|
| `sqrt(a, a')` | `(sqrt(a), a' / (2*sqrt(a)))` |
| `exp(a, a')` | `(exp(a), exp(a) * a')` |
| `log(a, a')` | `(log(a), a' / a)` |
| `sin(a, a')` | `(sin(a), cos(a) * a')` |
| `cos(a, a')` | `(cos(a), -sin(a) * a')` |
| `tan(a, a')` | `(tan(a), a' / cos^2(a))` |
| `abs(a, a')` | `(\|a\|, sign(a) * a')` |
| `pow(a, n)` | `(a^n, n * a^(n-1) * a')` |

## Applications

### Optimization (Gradient Descent)

```d
fn loss(params: [dual]) -> dual {
    // Mean squared error or other loss function
    let prediction = model(params);
    let error = prediction - target;
    error * error
}

fn optimize(initial: [f64], learning_rate: f64, iterations: i32) -> [f64] {
    var params = initial;
    for i in 0..iterations {
        let grads = grad(loss, params);
        for j in 0..len(params) {
            params[j] = params[j] - learning_rate * grads[j];
        }
    }
    params
}
```

### Physics Simulation (ODE Sensitivity)

```d
// Compute how ODE solution changes with respect to parameters
fn sensitivity_analysis(
    model: fn([dual]) -> [dual],
    params: [f64],
    initial_state: [f64]
) -> [[f64]] {
    jacobian(model, params)
}
```

### Neural Networks (Backpropagation)

Forward-mode autodiff is efficient for computing derivatives when:
- Number of inputs < Number of outputs
- Computing directional derivatives

For neural networks with many parameters, reverse-mode (backpropagation) is typically preferred, but forward-mode can be used for:
- Small networks
- Hessian-vector products
- Jacobian columns

## Implementation Details

### Memory Layout

Dual numbers are stored as 128-bit values (two 64-bit floats):
- Lane 0: Value component
- Lane 1: Derivative component

This enables SIMD operations on x86-64 using SSE/AVX F64X2 vectors.

### Performance

Forward-mode autodiff has overhead proportional to the number of input variables. For computing a gradient of `n` inputs:
- Forward mode: `O(n)` function evaluations
- Reverse mode: `O(1)` function evaluations (but more memory)

Use forward mode when:
- `n` is small (< 10-20 inputs)
- You need Jacobian columns or directional derivatives
- Memory is limited

## Comparison with Numerical Differentiation

```d
// Numerical (finite differences) - approximate, O(n) evaluations
fn numerical_grad(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

// Autodiff - exact, single evaluation
fn autodiff_grad(f: fn(dual) -> dual, x: f64) -> f64 {
    dual_deriv(f(dual(x, 1.0)))
}
```

Advantages of autodiff:
- **Exact**: No truncation or round-off errors
- **Efficient**: Computes derivative in same pass as function
- **Robust**: Works for any differentiable function

## Future Work

- **Reverse-mode autodiff**: For efficient gradients with many inputs
- **Higher-order derivatives**: Nested dual numbers for d^n/dx^n
- **Automatic vectorization**: SIMD-parallel derivative computation
- **GPU acceleration**: Compute derivatives on GPU for large-scale problems

## References

1. Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*
2. Baydin, A. G., et al. (2018). "Automatic Differentiation in Machine Learning: a Survey"
3. Rall, L. B. (1981). *Automatic Differentiation: Techniques and Applications*
