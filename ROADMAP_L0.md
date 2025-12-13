# Demetrios L0 Roadmap: A Linguagem L0 Cientifica

## Visao

**Demetrios (D)** e uma linguagem **L0 cientifica** - o mesmo nivel que C/Assembly, mas com primitivas cientificas nativas que nenhuma outra linguagem oferece.

```
Linguagens L0 tradicionais:     Demetrios L0 Cientifica:
├── C: ponteiros, structs       ├── tudo de C/Rust +
├── Rust: ownership, lifetimes  ├── autodiff nativo
└── Assembly: registros         ├── probabilistico nativo
                                ├── descoberta de modelos
                                ├── inferencia causal
                                ├── computacao simbolica
                                └── incerteza nativa
```

**Proposta unica:** Primeira linguagem L0 onde ciencia e cidada de primeira classe.

---

## Estado Atual (v0.61.0)

### ✅ Primitivas L0 Implementadas

| Primitiva | Modulo | Status | Linhas |
|-----------|--------|--------|--------|
| `grad`/`jacobian`/`hessian` | `autodiff.rs` | ✅ Completo | ~300 |
| `uncertain<T>` com ± | `uncertain.rs` | ✅ Completo | ~500 |
| `Tensor<T, Shape>` verificado | `tensor.rs` | ✅ Completo | ~400 |
| `sample`/`observe`/`infer` | `prob.rs` | ✅ Completo | ~600 |
| `ode`/`solve` (Euler, RK4, RK45) | `ode.rs` | ✅ Completo | ~400 |
| `solve_stiff` (BDF, LSODA) | `stiff.rs` | ✅ Completo | ~1200 |
| `discover`/`sindy` | `discover.rs` | ✅ Completo | ~500 |
| `do`/`counterfactual`/`ate` | `causal.rs` | ✅ Completo | ~600 |
| `symbolic`/`simplify`/`diff` | `symbolic.rs` | ✅ Completo | ~700 |
| `heat`/`wave`/`advection` PDEs | `pde.rs` | ✅ Completo | ~900 |
| `einsum` Einstein notation | `einsum.rs` | ✅ Completo | ~450 |
| GPU kernels (CUDA/Metal/WebGPU) | `gpu_scientific.rs` | ✅ Completo | ~1000 |

**Total Runtime Cientifico:** ~7,550 linhas, 1695 testes

---

## Primitivas L0 Cientificas

### 1. **Diferenciacao Automatica** ✅ IMPLEMENTADO

```d
fn loss(params: Tensor<f64>) -> f64 {
    return sum((predict(params) - data)^2)
}

fn main() {
    let theta = [1.0, 2.0, 3.0];
    
    // Primitivas nativas
    let g = grad(loss, theta);           // Gradiente
    let J = jacobian(f, theta);          // Jacobiano  
    let H = hessian(loss, theta);        // Hessiano
}
```

### 2. **Computacao Probabilistica** ✅ IMPLEMENTADO

```d
fn bayesian_model(data: [f64]) -> f64 with Prob {
    let mu = sample Normal(0.0, 10.0);
    let sigma = sample Gamma(1.0, 1.0);
    observe data ~ Normal(mu, sigma);
    return mu
}

fn main() {
    let posterior = infer(bayesian_model, data, 
                          method: HMC, 
                          samples: 10000);
}
```

### 3. **Descoberta de Modelos** ✅ IMPLEMENTADO

```d
fn discover_dynamics(data: Tensor<f64>, dt: f64) -> ODE {
    let library = polynomial_library(3) + dynamics_library();
    let model = sindy(data, library, threshold: 0.1);
    return model
}
```

### 4. **Inferencia Causal** ✅ IMPLEMENTADO

```d
fn causal_analysis(model: CausalModel, data: DataFrame) {
    let effect = do(model, X = 1.0);
    let cf = counterfactual(model, observed: {X: 0}, intervention: {X: 1});
    let ate = estimate_ate(model, treatment: X, outcome: Y);
}
```

### 5. **Computacao Simbolica** ✅ IMPLEMENTADO

```d
fn symbolic_math() {
    let x = symbol("x");
    let expr = x^2 + 2*x + 1;
    let simplified = simplify(expr);     // (x + 1)^2
    let derivative = differentiate(expr, x);  // 2x + 2
    let integral = integrate(expr, x);   // x^3/3 + x^2 + x
}
```

### 6. **Propagacao de Incerteza** ✅ IMPLEMENTADO

```d
fn experiment() -> uncertain<f64> {
    let mass = 5.0 +- 0.1;         // 5.0 kg +/- 0.1
    let velocity = 10.0 +- 0.5;    // 10.0 m/s +/- 0.5
    let energy = 0.5 * mass * velocity^2;  // Propagacao automatica!
    return energy  // 250.0 +/- 27.5 J
}
```

### 7. **Tensores Verificados** ✅ IMPLEMENTADO

```d
fn matrix_ops() {
    let A: Tensor<f64, [3, 4]> = zeros();
    let B: Tensor<f64, [4, 5]> = ones();
    let C = A @ B;  // [3,4] @ [4,5] = [3,5] - verificado em compilacao!
    let E = einsum("ij,jk->ik", A, B);
}
```

### 8. **Solvers ODE/PDE** ✅ IMPLEMENTADO

```d
fn simulate() {
    // ODE
    let sol = solve(lotka_volterra, y0, t_span, method: RK45);
    
    // Stiff ODE
    let sol = solve_stiff(robertson, y0, t_span);
    
    // PDE
    let heat = heat_equation_1d(&domain, &boundary, alpha, initial, t_final);
    let wave = wave_equation_1d(&domain, &boundary, c, u0, v0, t_final);
}
```

---

## Proximo Passo: Integracao no Compilador

As primitivas existem no runtime. Agora precisamos:

### Fase 5: Syntax Sugar e Integracao (Atual)

| Feature | Prioridade | Status |
|---------|------------|--------|
| Syntax `x +- y` para uncertain | P0 | 🔴 Parser |
| Syntax `ode { }` block | P0 | 🔴 Parser |
| Syntax `pde { }` block | P0 | 🔴 Parser |
| Type inference para Tensor shapes | P1 | 🟡 Parcial |
| Efeito handlers para Prob | P1 | 🟡 Parcial |
| Efeito handlers para Causal | P1 | 🔴 Nao iniciado |
| Codegen para autodiff | P1 | 🟡 HLIR pass |
| LLVM backend para primitivas | P2 | 🔴 Nao iniciado |
| Julia backend para primitivas | P2 | 🟡 Parcial |

### Fase 6: Otimizacoes

| Feature | Prioridade | Status |
|---------|------------|--------|
| Fusion de operacoes tensoriais | P2 | 🔴 |
| Paralelizacao automatica de ODEs | P2 | 🔴 |
| GPU dispatch automatico | P2 | 🟡 Kernels prontos |
| Sparse tensor support | P3 | 🔴 |
| Mixed precision autodiff | P3 | 🔴 |

### Fase 7: Ecossistema

| Feature | Prioridade | Status |
|---------|------------|--------|
| LSP com inferencia de shapes | P2 | 🔴 |
| Visualizacao de DAGs causais | P3 | 🔴 |
| Export para Stan/PyMC | P3 | 🔴 |
| Import de ONNX | P3 | 🔴 |
| Notebooks interativos | P3 | 🔴 |

---

## Comparacao Final

| Feature | D | Julia | Python | Rust | C++ |
|---------|---|-------|--------|------|-----|
| **L0 (compilada, sem runtime)** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Autodiff nativo** | ✅ | Pkg | Pkg | Pkg | ❌ |
| **Probabilistico nativo** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Descoberta de modelos** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Inferencia causal** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Incerteza nativa** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Unidades verificadas** | ✅ | Pkg | Pkg | Pkg | ❌ |
| **Ontologias** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Shapes verificados** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **PDEs nativos** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Einstein notation** | ✅ | Pkg | Pkg | ❌ | ❌ |
| **GPU multi-backend** | ✅ | ❌ | ❌ | ❌ | ❌ |

**Demetrios e a primeira e unica linguagem L0 projetada para ciencia.**

---

## Changelog

### v0.61.0 (2025-12-11)
- ✅ PDE solvers (Heat, Wave, Advection, Diffusion-Reaction)
- ✅ Einstein notation (einsum)
- ✅ Stiff ODE solvers (BDF, LSODA, Rosenbrock)
- ✅ GPU scientific kernels

### v0.60.0 (2025-12-10)
- ✅ Symbolic computation
- ✅ Causal inference (do-calculus)
- ✅ Model discovery (SINDy)

### v0.59.0 (2025-12-09)
- ✅ Autodiff (dual numbers)
- ✅ uncertain<T> type
- ✅ Tensor<T, Shape>
- ✅ Prob effect runtime
- ✅ ODE solvers (Euler, RK4, RK45)

---

## Citacao

```bibtex
@software{demetrios2025,
  author = {Agourakis, Demetrios Chiuratto and Agourakis, Dionisio Chiuratto},
  title = {Demetrios: A Scientific L0 Programming Language},
  year = {2025},
  url = {https://github.com/Chiuratto-AI/demetrios}
}
```

---

*"A linguagem que a ciencia merecia desde o inicio."*
