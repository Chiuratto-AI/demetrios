# Scientific Computing API Reference

## Module: `linalg`

### Core Types

#### `Matrix<T, const L: Layout>`
Dense matrix with configurable memory layout.

```d
pub struct Matrix<T, const L: Layout = Layout::RowMajor> {
    // Implementation details hidden
}

impl<T: Clone + Default, const L: Layout> Matrix<T, L> {
    pub fn new(nrows: usize, ncols: usize) -> Self;
    pub fn zeros(nrows: usize, ncols: usize) -> Self where T: num::Zero;
    pub fn eye(n: usize) -> Self where T: num::Zero + num::One;
    pub fn from_nested(data: &[[T]]) -> Self;
    
    // Accessors
    pub fn nrows(&self) -> usize;
    pub fn ncols(&self) -> usize;
    pub fn shape(&self) -> (usize, usize);
    
    // Operations
    pub fn t(&self) -> MatrixView<T, {L.transpose()}>;
    pub fn clone(&self) -> Self;
    pub fn norm_fro(&self) -> f64 where T: Into<f64> + Clone;
}
```

#### `Vector<T>`
Type alias for single-column matrix: `type Vector<T> = Matrix<T, Layout::ColMajor>`

### BLAS Operations

#### Level 1 (Vector-Vector)
```d
pub fn daxpy(alpha: f64, x: &Vector<f64>, y: &!Vector<f64>) with IO;
pub fn ddot(x: &Vector<f64>, y: &Vector<f64>) -> f64 with IO;
pub fn dnrm2(x: &Vector<f64>) -> f64 with IO;
pub fn dscal(alpha: f64, x: &!Vector<f64>) with IO;
```

#### Level 2 (Matrix-Vector)
```d
pub fn dgemv(
    trans: Transpose,
    alpha: f64,
    a: &Matrix<f64>,
    x: &Vector<f64>,
    beta: f64,
    y: &!Vector<f64>
) with IO;
```

#### Level 3 (Matrix-Matrix)
```d
pub fn dgemm(
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: f64,
    a: &Matrix<f64>,
    b: &Matrix<f64>,
    beta: f64,
    c: &!Matrix<f64>
) with IO;
```

### LAPACK Decompositions

#### LU Decomposition
```d
pub struct LU {
    pub factors: Matrix<f64>,
    pub pivots: Vector<i32>,
}

pub fn lu(a: &Matrix<f64>) -> Result<LU, string> with IO;
pub fn solve(a: &Matrix<f64>, b: &Vector<f64>) -> Result<Vector<f64>, string> with IO;
```

#### Cholesky Decomposition
```d
pub struct Cholesky {
    pub factor: Matrix<f64>,
}

pub fn cholesky(a: &Matrix<f64>) -> Result<Cholesky, string> with IO;
```

#### SVD
```d
pub struct SVD {
    pub u: Matrix<f64>,
    pub s: Vector<f64>,
    pub vt: Matrix<f64>,
}

pub fn svd(a: &Matrix<f64>) -> Result<SVD, string> with IO;
```

## Module: `numerics`

### ODE Solvers

#### Core Trait
```d
pub trait ODESystem {
    fn eval(&self, t: f64, y: &Vector<f64>, dydt: &!Vector<f64>);
    fn dim(&self) -> usize;
    fn jacobian(&self, t: f64, y: &Vector<f64>) -> Option<Matrix<f64>>;
}
```

#### Solvers
```d
pub struct RKF45<S: ODESystem>;
pub struct BDF<S: ODESystem>;

pub fn odeint<S: ODESystem>(
    system: S,
    y0: &Vector<f64>,
    t_span: (f64, f64),
    method: &str,
) -> ODESolution with Alloc;
```

### Optimization

#### Algorithms
```d
pub struct GradientDescent;
pub struct BFGS;

impl BFGS {
    pub fn minimize<F>(&self, f: F, x0: &Vector<f64>) -> OptResult
    where F: Fn(&Vector<Var>) -> Var + Clone;
}
```

### Integration

#### Quadrature
```d
pub fn quad<F>(f: F, a: f64, b: f64, config: IntegrationConfig) -> IntegrationResult
where F: Fn(f64) -> f64;

pub fn simpson<F>(f: F, a: f64, b: f64, tol: f64) -> IntegrationResult
where F: Fn(f64) -> f64;
```

## Module: `autodiff`

### Forward Mode
```d
pub struct Dual {
    pub val: f64,
    pub dot: f64,
}

impl Dual {
    pub fn new(val: f64, dot: f64) -> Self;
    pub fn constant(val: f64) -> Self;
    pub fn variable(val: f64) -> Self;
    
    // Mathematical operations
    pub fn add(self, other: Dual) -> Dual;
    pub fn mul(self, other: Dual) -> Dual;
    pub fn exp(self) -> Dual;
    pub fn sin(self) -> Dual;
    // ... more operations
}

pub fn gradient<F>(f: F, x: &Vector<f64>) -> Vector<f64>
where F: Fn(&Vector<Dual>) -> Dual;
```

### Reverse Mode
```d
pub struct Var;

impl Var {
    pub fn new(val: f64) -> Self;
    pub fn value(&self) -> f64;
    pub fn backward(&self);
    pub fn grad(&self) -> f64;
    
    // Operations (same interface as Dual)
}

pub fn gradient<F>(f: F, x: &Vector<f64>) -> Vector<f64>
where F: Fn(&Vector<Var>) -> Var + Clone;
```

## Module: `prob`

### Distributions

#### Core Trait
```d
pub trait Distribution<T> {
    fn sample(&self, rng: &!Rng) -> T with Prob;
    fn pdf(&self, x: T) -> f64;
    fn log_pdf(&self, x: T) -> f64;
    fn cdf(&self, x: T) -> f64;
    fn mean(&self) -> T;
    fn variance(&self) -> f64;
}
```

#### Common Distributions
```d
pub struct Normal { pub mu: f64, pub sigma: f64 }
pub struct MultivariateNormal { pub mean: Vector<f64>, pub cov: Matrix<f64> }
pub struct Gamma { pub shape: f64, pub rate: f64 }
pub struct Beta { pub alpha: f64, pub beta: f64 }
pub struct Poisson { pub lambda: f64 }
```

### MCMC Samplers

#### Metropolis-Hastings
```d
pub struct MetropolisHastings {
    pub proposal_cov: Matrix<f64>,
}

impl MetropolisHastings {
    pub fn sample<F>(
        &mut self,
        log_prob: F,
        x0: &Vector<f64>,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> MCMCSample
    where F: Fn(&Vector<f64>) -> f64;
}
```

#### Hamiltonian Monte Carlo
```d
pub struct HMC {
    pub epsilon: f64,
    pub l: usize,
}

impl HMC {
    pub fn sample<F>(
        &self,
        log_prob: F,
        x0: &Vector<f64>,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> MCMCSample
    where F: Fn(&Vector<Var>) -> Var + Clone;
}
```

## Module: `pkpd`

### Compartment Models

#### PK Parameters
```d
pub struct PKParameters {
    pub cl: f64: L_h,      // Clearance
    pub v1: f64: L,        // Central volume
    pub v2: Option<f64: L>, // Peripheral volume
    pub q2: Option<f64: L_h>, // Inter-compartmental clearance
    pub ka: Option<f64: h_inv>, // Absorption rate
    pub f: f64,            // Bioavailability
}

impl PKParameters {
    pub fn one_compartment(cl: f64: L_h, v: f64: L) -> Self;
    pub fn two_compartment(cl: f64: L_h, v1: f64: L, v2: f64: L, q: f64: L_h) -> Self;
    pub fn half_life(&self) -> f64: h;
}
```

#### Dosing
```d
pub struct DoseEvent {
    pub time: f64: h,
    pub amount: f64: mg,
    pub cmt: usize,
    pub duration: f64: h,
}

impl DoseEvent {
    pub fn iv_bolus(time: f64: h, amount: f64: mg) -> Self;
    pub fn oral(time: f64: h, amount: f64: mg) -> Self;
}
```

#### Simulation
```d
pub fn simulate_pk(
    params: &PKParameters,
    doses: &[DoseEvent],
    times: &[f64],
) -> PKResult with Alloc;
```

### Non-compartmental Analysis
```d
pub fn nca_analysis(
    time: &Vector<f64>,
    concentration: &Vector<f64>,
    dose: f64: mg,
    dose_time: f64: h,
    n_terminal_points: usize,
) -> NCAResult;

pub struct NCAResult {
    pub auc_last: f64: mg_L * h,
    pub auc_inf: f64: mg_L * h,
    pub cmax: f64: mg_L,
    pub tmax: f64: h,
    pub t_half: f64: h,
    pub cl: f64: L_h,
    pub vz: f64: L,
    // ... more parameters
}
```

## Error Handling

All fallible operations return `Result<T, string>` with descriptive error messages:

```d
match lu(&matrix) {
    Ok(decomp) => { /* use decomp */ },
    Err(msg) => eprintln!("LU decomposition failed: {}", msg),
}
```

## Effect System Integration

Functions are annotated with their computational effects:

- `with IO`: BLAS/LAPACK calls, file operations
- `with Prob`: Random number generation, sampling
- `with Alloc`: Memory allocation for large arrays
- `with GPU`: GPU kernel launches (when available)

## Units of Measure

Pharmacokinetic parameters use compile-time unit checking:

```d
let clearance: f64: L_h = 10.0;  // Liters per hour
let volume: f64: L = 50.0;       // Liters
let concentration: f64: mg_L = dose / volume;  // mg/L
```

Unit mismatches are caught at compile time, preventing dimensional analysis errors common in scientific computing.
