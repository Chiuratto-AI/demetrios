# Neurosymbolic Programming in Demetrios

## Overview

**Demetrios Wave 3** introduces **native neurosymbolic programming** as a language feature. Rather than treating neural networks and symbolic reasoning as separate, incompatible tools, Demetrios unifies them at the type system and runtime level.

Neurosymbolic models combine the **flexibility of neural learning** with the **interpretability and structure of symbolic reasoning**, unlocking new capabilities:
- Hybrid models that are both powerful and explainable
- Physics-informed neural networks that enforce domain knowledge
- Symbolic regression that discovers laws from data
- Transparent decision-making in AI systems

## What is Neurosymbolic Programming?

### The Problem with Pure Approaches

**Neural Networks Alone:**
- ✓ Flexible, high-capacity function approximators
- ✓ End-to-end learning from data
- ✗ Black box: difficult to interpret
- ✗ Require large amounts of training data
- ✗ May learn spurious correlations
- ✗ Hard to enforce domain constraints (conservation laws, symmetries)

**Symbolic Systems Alone:**
- ✓ Transparent and interpretable
- ✓ Guaranteed to follow logical rules
- ✓ Work well with limited data
- ✗ Brittle: hand-coded rules fail on variations
- ✗ Difficult to incorporate learned patterns
- ✗ Require domain expertise to formalize

### The Neurosymbolic Solution

Combine both approaches:
- **Neural components** handle flexible pattern recognition and learning
- **Symbolic components** enforce structure, interpretability, and domain knowledge
- **Unified training** optimizes hybrid objectives: accuracy + interpretability

## Core Concepts

### 1. Symbolic Expressions

A **symbolic expression** is a mathematical formula that can be:
- **Parsed** from string notation
- **Differentiated** symbolically (exact derivatives, not numeric approximation)
- **Simplified** algebraically
- **Evaluated** with variable bindings
- **Compiled** to efficient numeric code

```d
// Parse expression
let expr = symbolic::parse("x^2 + 2*x + 1");

// Differentiate symbolically
let deriv = diff(expr, "x");  // Returns: 2*x + 2

// Evaluate with specific values
let result = expr.eval({x: 3.0});  // Returns: 16.0

// Simplify
let simplified = simplify((x + 0) * (y / y));  // Returns: x
```

### 2. Hybrid Models

A **hybrid model** combines neural and symbolic components:

```d
type HybridModel = {
    neural_params: [f64],      // Learned parameters
    symbolic_expr: Expr,       // Structural form
    fusion: FusionStrategy,    // How to combine them
}
```

**Fusion strategies:**

1. **WeightedSum**: `α * neural(x) + (1-α) * symbolic(x)`
   - Learns an interpolation weight
   - Best for: Gradually improving symbolic models

2. **LearnedGate**: `gate(x) * neural + (1-gate(x)) * symbolic`
   - Learns when to use each component
   - Best for: Domain-specific vs. catch-all behaviors

3. **Product**: `neural(x) * symbolic(x)`
   - Multiplicative combination
   - Best for: Modulation and scaling

4. **ProductResidual**: `neural(x) * symbolic(x) + residual(x)`
   - Uses symbolic as base, neural as correction
   - Best for: Refining known models

### 3. Physics-Informed Neural Networks (PINNs)

Solve differential equations by encoding the PDE as a **soft constraint**:

```d
fn pinn_loss(params: [f64]) -> f64 {
    // Data loss: match observations
    let data_loss = 0.0;
    for (x, t, u_obs) in observations {
        let u_pred = network(x, t, params);
        data_loss += (u_pred - u_obs)^2;
    }

    // Physics loss: PDE residual
    let physics_loss = 0.0;
    for (x, t) in collocation_points {
        let u = network(x, t, params);
        let dudt = grad_t(u);
        let d2udx2 = hessian_xx(u);
        let residual = dudt - d2udx2;  // ∂u/∂t - ∂²u/∂x² = 0
        physics_loss += residual^2;
    }

    return data_loss + physics_loss;
}
```

No mesh, no finite-difference stencil—just the PDE in symbolic form!

### 4. Symbolic Regression

Discover equations from data:

```d
// Data: x, y pairs following some unknown law
let data = [(1.0, 1.0), (2.0, 4.0), (3.0, 9.0), (4.0, 16.0)];

// Try candidate forms
let candidates = [
    "x",
    "x^2",
    "x^3",
    "sin(x)",
    "exp(x)",
];

// For each form, fit parameters and evaluate error
// Best fit reveals: y = x^2
```

Classic application: **Kepler's Third Law**
- Data: orbital periods and semi-major axes of planets
- Discovered law: T² ∝ a³
- This unified celestial mechanics!

### 5. Explainable AI via Symbolic Approximation

Interpret neural networks by fitting symbolic models to their learned functions:

```d
// Step 1: Train black-box neural network
let nn = train_neural_network(data);

// Step 2: Sample neural network predictions
let samples = (0..1000)
    .map(|i| (i as f64, nn.predict(i as f64)))
    .collect();

// Step 3: Fit symbolic model to neural outputs
let approximation = fit_polynomial(samples, degree: 2);
// Result: clear formula that explains neural network

// Step 4: Interpret
println!("Neural network is approximately: {}", approximation);
// "price ≈ 150 * sqft + 5000"
// Much more interpretable than "8-layer network"!
```

## Symbolic Expression API

### Parsing

```d
fn parse(source: &str) -> Result<Expr>
```

Parse a mathematical expression from a string.

**Supported syntax:**
- Arithmetic: `+`, `-`, `*`, `/`, `^`
- Functions: `sin()`, `cos()`, `tan()`, `exp()`, `ln()`, `sqrt()`, `abs()`
- Variables: any identifier (e.g., `x`, `y`, `theta`)
- Numbers: integers and floats
- Parentheses for grouping

**Examples:**
```d
parse("x^2 + 2*x + 1")
parse("sin(x) * cos(y)")
parse("exp(-x^2/2) / sqrt(2*pi)")
parse("(a + b)^3")
```

### Differentiation

```d
fn differentiate(&self, var: &str) -> Expr
```

Compute the symbolic derivative with respect to a variable.

Uses **automatic symbolic differentiation** with rules:
- Constant rule: ∂c/∂x = 0
- Power rule: ∂(x^n)/∂x = n*x^(n-1)
- Sum rule: ∂(f+g)/∂x = ∂f/∂x + ∂g/∂x
- Product rule: ∂(f*g)/∂x = f*∂g/∂x + g*∂f/∂x
- Chain rule: ∂f(g(x))/∂x = ∂f/∂u * ∂g/∂x (where u = g(x))
- Quotient rule: ∂(f/g)/∂x = (g*∂f/∂x - f*∂g/∂x) / g²

**Example:**
```d
let f = parse("x^3 - 2*x + 1").unwrap();
let df = f.differentiate("x");
// df is: 3*x^2 - 2

let d2f = df.differentiate("x");
// d2f is: 6*x (second derivative)
```

### Evaluation

```d
fn evaluate(&self, vars: &HashMap<String, f64>) -> Result<f64>
```

Evaluate the expression with specific variable values.

**Example:**
```d
let expr = parse("x^2 + y").unwrap();
let value = expr.evaluate(&[("x", 3.0), ("y", 2.0)].iter().cloned().collect());
// value = 11.0
```

### Simplification

```d
fn simplify(&self) -> Expr
```

Perform algebraic simplification:
- Identity: `x + 0 → x`, `x * 1 → x`, `x * 0 → 0`
- Self-cancellation: `x - x → 0`, `x / x → 1`
- Constant folding: `2.0 * 3.0 → 6.0`
- Associativity: `(x + 1) + 2 → x + 3`

**Example:**
```d
let expr = parse("(x + 0) * (y / y)").unwrap();
let simp = expr.simplify();
// simp is: x
```

### Variable Extraction

```d
fn variables(&self) -> Vec<String>
```

Extract all free variables in the expression.

**Example:**
```d
let expr = parse("x^2 + 2*x*y + z").unwrap();
let vars = expr.variables();
// vars = ["x", "y", "z"]
```

## Hybrid Model API

### Creating Hybrid Models

```d
fn hybrid_model(
    neural_params: [f64],
    symbolic_expr: Expr,
    fusion: HybridFusion
) -> HybridModel
```

Create a hybrid model combining neural and symbolic components.

**Example:**
```d
let symbolic = parse("a*x^2 + b*x + c").unwrap();
let params = [0.1, 0.2, 0.5];  // Initial guesses
let model = hybrid_model(params, symbolic, HybridFusion::WeightedSum);
```

### Forward Pass

```d
fn forward(&self, x: f64) -> f64
```

Evaluate the hybrid model at input `x`.

Behavior depends on fusion strategy:
- **WeightedSum**: Interpolates between neural and symbolic outputs
- **Product**: Multiplies neural and symbolic
- Etc.

### Training Hybrid Models

```d
fn train(
    model: &mut HybridModel,
    data: &[(f64, f64)],
    epochs: usize,
    learning_rate: f64
) -> TrainingStats
```

Train using gradient descent on the unified loss.

**Example:**
```d
let loss_fn = |params: [f64]| -> f64 {
    let mut loss = 0.0;
    for (x, y_true) in &data {
        let y_pred = model.forward(x);
        loss += (y_pred - y_true)^2;
    }
    return loss / data.len() as f64;
};

let grads = grad(loss_fn, model.neural_params);
model.neural_params -= learning_rate * grads;
```

## Unified Autodiff

### The Challenge

Demetrios Wave 2 provides two separate autodiff systems:
- **Numeric autodiff** (reverse-mode): for neural network training
- **Symbolic autodiff**: for symbolic expressions

Wave 3 unifies them:

```d
// Same grad() function works everywhere!

// Numeric: gradient of f(x) = x^2 at x=3
let df_numeric = grad(|x| x * x, 3.0);  // Returns: 6.0

// Symbolic: derivative of f(x) = x^2
let expr = parse("x^2").unwrap();
let df_symbolic = expr.differentiate("x");  // Returns: 2*x

// Hybrid: gradient through hybrid model
let loss_fn = |params: [f64]| hybrid_loss(model, data, params);
let grads = grad(loss_fn, model.neural_params);
```

### How Unified Autodiff Works

For hybrid functions:

```
f(x) = α * neural(x; θ) + (1-α) * symbolic(x)
     = α * neural(x; θ) + (1-α) * expr.eval(x)

∂f/∂θ = α * (∂neural/∂θ)                      [numeric autodiff]
∂f/∂x = α * (∂neural/∂x) + (1-α) * (∂symbolic/∂x)  [mixed]
```

The gradient computation automatically selects the right method for each component!

## Applications

### 1. Hybrid Regression

Fit models where you know the symbolic structure but not the parameters:

```d
// Known: data follows ax² + bx + c
// Learn: coefficients a, b, c

let form = parse("a*x^2 + b*x + c").unwrap();
let model = hybrid_model([0.1, 0.1, 0.1], form, WeightedSum);

// Train to find a ≈ 2, b ≈ -3, c ≈ 1
train(model, data, epochs: 100, lr: 0.01);
```

### 2. Physics-Informed Learning

Solve PDEs with data + constraints:

```d
// Heat equation: ∂u/∂t = ∂²u/∂x²
let loss = |params| {
    let mut data_loss = 0.0;
    let mut pde_loss = 0.0;

    for (x, t, u_obs) in observations {
        let u = nn(x, t, params);
        data_loss += (u - u_obs)^2;
    }

    for (x, t) in collocation {
        let u = nn(x, t, params);
        let dudt = autodiff::grad_t(|t| nn(x, t, params), t);
        let d2udx2 = autodiff::hessian_xx(|x| nn(x, t, params), x);
        let residual = dudt - d2udx2;
        pde_loss += residual^2;
    }

    return data_loss + pde_loss;
};

// No mesh generation needed!
```

### 3. Symbolic Regression

Discover laws from data:

```d
fn symbolic_regression(data: &[(f64, f64)], max_complexity: usize) -> (Expr, f64) {
    let mut best = (parse("x").unwrap(), f64::INFINITY);

    for complexity in 1..=max_complexity {
        for candidate in generate_candidates(complexity) {
            let expr = parse(&candidate).ok()?;
            let error = evaluate_fit(&expr, data);
            if error < best.1 {
                best = (expr, error);
            }
        }
    }

    return best;
}

// Discovers: y = x^2 (Kepler's law, quadratic regression, etc.)
```

### 4. Explainable AI

Make neural networks interpretable:

```d
// Train neural network
let nn = train_neural_network(data);

// Approximate with polynomial
let approx = fit_polynomial(
    |x| nn.predict(x),
    degree: 2,
    domain: (0.0, 10.0)
);

println!("Neural network ≈ {}", approx);
// "0.001*x^2 + 0.15*x + 0.01"
```

## Best Practices

### 1. Choose the Right Fusion Strategy

- **WeightedSum**: General-purpose, good for gradual improvement
- **LearnedGate**: When components handle different regimes
- **Product**: When factors are multiplicative (e.g., amplitude * phase)
- **ProductResidual**: Refining a known base model

### 2. Balance Data and Physics

```d
let loss = data_loss + λ * physics_loss;
```

Choose λ based on:
- λ = 0: Pure data-driven (flexible, may overfit)
- λ = 1: Equal weight (common starting point)
- λ > 1: Prioritize physics (good when data is noisy)

### 3. Symbolic Simplification

Always simplify expressions to understand them:

```d
let expr = differentiate(f);
let simplified = expr.simplify();
println!("{}", simplified);  // Makes patterns visible
```

### 4. Validate Against Domain Knowledge

```d
// Does the learned formula make sense?
// Does it satisfy boundary conditions?
// Does it preserve symmetries?
// Does it pass physical unit analysis?
```

### 5. Use Symbolic Regression on Small Feature Sets

Symbolic regression scales as O(C^n) where C = candidates, n = complexity.
- Keep n ≤ 4-5 terms
- Keep variables ≤ 5 for exhaustive search
- Use genetic algorithms for larger spaces

## Limitations and Future Work

### Current Limitations

1. **Symbolic Expression Complexity**: Limited to compositions of elementary functions
2. **Symbolic Integration**: Only differentiation fully implemented; integration symbolic stubs
3. **Scalability**: Symbolic regression becomes expensive with many variables
4. **Automatic Candidate Generation**: Limited; mostly hand-specified forms

### Planned Enhancements (v0.70+)

- [ ] Automatic expression simplification heuristics
- [ ] Symbolic integration (symbolic antiderivatives)
- [ ] Genetic programming for symbolic regression
- [ ] Lie algebra symmetry detection
- [ ] Automatic conservation law verification
- [ ] Mixed-precision symbolic-numeric computation

## See Also

- [AUTODIFF_GUIDE.md](AUTODIFF_GUIDE.md) - Numeric automatic differentiation
- [CAUSAL_GUIDE.md](CAUSAL_GUIDE.md) - Causal reasoning
- [Examples](../examples/) - wave3_* examples
- "Machine Learning for Scientific Discovery" - Cranmer et al. (2020)
- "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems" - Raissi et al. (2019)
