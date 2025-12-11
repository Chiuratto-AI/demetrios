# Demetrios L0 Roadmap: A Linguagem L0 Científica

## Visão

**Demetrios (D)** é uma linguagem **L0 científica** - o mesmo nível que C/Assembly, mas com primitivas científicas nativas que nenhuma outra linguagem oferece.

```
Linguagens L0 tradicionais:     Demetrios L0 Científica:
├── C: ponteiros, structs       ├── tudo de C/Rust +
├── Rust: ownership, lifetimes  ├── autodiff nativo
└── Assembly: registros         ├── probabilístico nativo
                                ├── descoberta de modelos
                                ├── inferência causal
                                ├── computação simbólica
                                └── incerteza nativa
```

**Proposta única:** Primeira linguagem L0 onde ciência é cidadã de primeira classe.

---

## Primitivas L0 Científicas

### 1. **Diferenciação Automática** (`grad`, `jacobian`, `hessian`)

```d
// Nativo na linguagem - não biblioteca!
fn loss(params: Tensor<f64>) -> f64 {
    return sum((predict(params) - data)^2)
}

fn main() {
    let θ = [1.0, 2.0, 3.0];
    
    // Primitivas nativas
    let g = grad(loss, θ);           // Gradiente
    let J = jacobian(f, θ);          // Jacobiano  
    let H = hessian(loss, θ);        // Hessiano
    
    // Forward e reverse mode automático
    let θ_new = θ - 0.01 * g;
}
```

**Implementação:** Dual numbers + source transformation no HLIR

### 2. **Computação Probabilística** (`sample`, `observe`, `infer`)

```d
// Efeito Prob é primitiva L0
fn bayesian_model(data: [f64]) -> f64 with Prob {
    // Sampling é operação primitiva
    let μ = sample Normal(0.0, 10.0);
    let σ = sample Gamma(1.0, 1.0);
    
    // Conditioning é primitivo
    observe data ~ Normal(μ, σ);
    
    return μ
}

fn main() {
    let data = [1.2, 1.5, 1.3, 1.8];
    
    // Inferência como primitiva
    let posterior = infer(bayesian_model, data, 
                          method: HMC, 
                          samples: 10000);
    
    let μ_estimate = posterior.mean();
    let credible = posterior.hdi(0.95);
}
```

**Backends nativos:** HMC, NUTS, Variational Inference, SMC

### 3. **Descoberta de Modelos** (`discover`, `sparse`)

```d
// SINDy-like como primitiva L0
fn discover_dynamics(data: Tensor<f64>, dt: f64) -> ODE with Discover {
    // Biblioteca de funções candidatas
    let library = [
        poly(1), poly(2), poly(3),  // x, x², x³
        sin, cos, exp,               // transcendentais  
        |x, y| x * y,               // interações
    ];
    
    // Descoberta esparsa é primitiva
    let model = discover ode from data 
                with library: library,
                     sparsity: 0.1,
                     threshold: 0.05;
    
    return model  // Retorna ODE simbólica!
}

// O compilador otimiza a ODE descoberta
fn main() {
    let trajectory = load_csv("experiment.csv");
    let ode = discover_dynamics(trajectory, 0.01);
    
    // ode é agora um tipo ODE verificado em tempo de compilação
    print(ode.symbolic_form());  // "dx/dt = -0.5*x + 0.1*x²"
    
    let prediction = solve(ode, x0: 1.0, t_span: [0, 10]);
}
```

### 4. **Inferência Causal** (`do`, `counterfactual`, `intervene`)

```d
// Causalidade como primitiva L0
struct CausalModel {
    graph: DAG,
    mechanisms: [fn],
}

fn causal_analysis(model: CausalModel, data: DataFrame) with Causal {
    // Operador do() - intervenção
    let effect = do(model, X = 1.0) {
        observe Y
    };
    
    // Counterfactual como primitiva
    let cf = counterfactual(model, 
        observed: {X: 0, Y: 1},
        intervention: {X: 1}
    ) {
        query Y  // "O que teria acontecido?"
    };
    
    // Identificação causal automática
    let ate = identify(model, 
        treatment: X,
        outcome: Y,
        method: BackdoorCriterion
    );
}
```

### 5. **Computação Simbólica** (`symbolic`, `simplify`, `solve_symbolic`)

```d
// Expressões simbólicas são tipos de primeira classe
fn symbolic_math() with Symbolic {
    // Declaração de símbolos
    let x = symbolic("x");
    let y = symbolic("y");
    
    // Manipulação simbólica nativa
    let expr = x^2 + 2*x*y + y^2;
    let simplified = simplify(expr);     // (x + y)²
    
    // Resolução simbólica
    let equation = x^2 - 4 == 0;
    let solutions = solve_symbolic(equation, x);  // [-2, 2]
    
    // Integração simbólica
    let integral = integrate(x^2, x);    // x³/3
    
    // Conversão para função numérica
    let f = compile(expr);               // fn(f64, f64) -> f64
}
```

### 6. **Propagação de Incerteza** (`uncertain<T>`, `±`)

```d
// Incerteza como tipo primitivo
fn experiment() -> uncertain<f64> {
    // Medições com incerteza
    let mass: uncertain<f64> = 5.0 ± 0.1;      // 5.0 kg ± 0.1
    let velocity: uncertain<f64> = 10.0 ± 0.5; // 10.0 m/s ± 0.5
    
    // Propagação automática (Monte Carlo ou analítica)
    let energy = 0.5 * mass * velocity^2;
    
    // energy automaticamente tem incerteza propagada!
    print(energy);  // 250.0 ± 27.5 J
    
    return energy
}

// Também funciona com distribuições completas
fn bayesian_propagation() {
    let x: Distribution = Normal(10.0, 2.0);
    let y: Distribution = Gamma(2.0, 1.0);
    
    // Operações preservam distribuições
    let z = x * y;  // z é uma distribuição derivada
    
    print(z.mean());
    print(z.std());
    print(z.percentile(0.95));
}
```

### 7. **Tensores com Dimensões Verificadas** (`Tensor<T, Shape>`)

```d
// Shapes verificados em tempo de compilação
fn matrix_ops() {
    let A: Tensor<f64, [3, 4]> = zeros();
    let B: Tensor<f64, [4, 5]> = ones();
    
    // Compilador verifica compatibilidade
    let C = A @ B;  // OK: [3,4] @ [4,5] = [3,5]
    
    // let D = B @ A;  // ERRO DE COMPILAÇÃO: [4,5] @ [3,4] inválido
    
    // Broadcasting verificado
    let v: Tensor<f64, [4]> = [1, 2, 3, 4];
    let D = A + v;  // OK: broadcast [4] para [3, 4]
    
    // Einstein notation nativa
    let E = einsum("ij,jk->ik", A, B);
}
```

### 8. **Solvers de ODE/PDE Nativos** (`ode`, `pde`, `solve`)

```d
// ODEs como tipos de primeira classe
ode LotkaVolterra {
    params: { α: f64, β: f64, γ: f64, δ: f64 }
    state: { prey: f64, predator: f64 }
    
    d(prey)/dt = α * prey - β * prey * predator
    d(predator)/dt = δ * prey * predator - γ * predator
}

fn simulate() {
    let model = LotkaVolterra {
        α: 1.1, β: 0.4, γ: 0.4, δ: 0.1
    };
    
    let solution = solve(model,
        initial: { prey: 10.0, predator: 5.0 },
        t_span: [0.0, 50.0],
        method: DormandPrince,      // ou Tsit5, Rodas5, etc.
        abstol: 1e-8,
        reltol: 1e-6
    );
    
    // Events nativos
    let events = solve(model, ...,
        events: [
            when prey < 1.0 then stop,
            when predator > 20.0 then { 
                predator = 15.0  // reset
            }
        ]
    );
}

// PDEs também
pde HeatEquation {
    params: { α: f64 }  // difusividade
    domain: Rectangle([0, 1], [0, 1])
    
    ∂u/∂t = α * (∂²u/∂x² + ∂²u/∂y²)
    
    boundary: {
        x = 0: u = 0,
        x = 1: u = 0,
        y = 0: ∂u/∂n = 0,  // Neumann
        y = 1: u = sin(π * x)
    }
}
```

---

## Arquitetura de Implementação

### Novos Efeitos Algébricos

```d
// Efeitos L0 científicos
effect Prob {
    sample<D: Distribution>(d: D) -> D::Output
    observe<D: Distribution>(value: D::Output, d: D) -> ()
    factor(log_weight: f64) -> ()
}

effect Discover {
    propose_term(library: [fn]) -> Term
    evaluate_fitness(model: Model, data: Data) -> f64
    select_sparse(coefficients: [f64], threshold: f64) -> [f64]
}

effect Causal {
    intervene<T>(variable: Var<T>, value: T) -> ()
    observe_under_intervention<T>(target: Var<T>) -> T
    counterfactual_query<T>(factual: Evidence, intervention: Evidence) -> T
}

effect Symbolic {
    create_symbol(name: str) -> Expr
    differentiate(expr: Expr, var: Symbol) -> Expr
    integrate(expr: Expr, var: Symbol) -> Expr
    simplify(expr: Expr) -> Expr
}

effect Autodiff {
    dual<T: Numeric>(value: T) -> Dual<T>
    grad<F: Differentiable>(f: F, at: [f64]) -> [f64]
    jacobian<F: Differentiable>(f: F, at: [f64]) -> [[f64]]
}
```

### Hierarquia de Tipos Científicos

```d
// Tipos base
trait Numeric { ... }
trait Differentiable: Numeric { ... }
trait Probabilistic { ... }

// Tipos com incerteza
type uncertain<T: Numeric> = struct {
    value: T,
    uncertainty: T,  // ou Distribution<T>
}

// Tensores tipados
type Tensor<T: Numeric, Shape: [usize]> = struct {
    data: [T],
    shape: Shape,
    strides: [usize],
}

// Distribuições
trait Distribution {
    type Output;
    fn sample(self, rng: &mut RNG) -> Self::Output;
    fn log_prob(self, x: Self::Output) -> f64;
}

// Expressões simbólicas
enum Expr {
    Symbol(String),
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Fn(String, Vec<Expr>),
    Derivative(Box<Expr>, String),
}
```

---

## Estado Atual vs. Objetivo

### ✅ Já Implementado (v0.59)

| Componente | Status |
|------------|--------|
| Lexer/Parser | ✅ Completo |
| AST | ✅ Completo |
| Type Checker | ✅ Funcional |
| Unidades de Medida | ⚠️ Parsing OK |
| Efeito `Prob` | ⚠️ Declarado |
| Efeito `GPU` | ⚠️ Declarado |
| Ontologias | ✅ Avançado |

### 🎯 Objetivo: Primitivas Científicas

| Primitiva | Prioridade | Complexidade | Status |
|-----------|------------|--------------|--------|
| `grad`/`jacobian` | P0 | Alta | 🔴 Não iniciado |
| `sample`/`observe`/`infer` | P0 | Alta | 🟡 Efeito existe |
| `uncertain<T>` | P1 | Média | 🔴 Não iniciado |
| `Tensor<T, Shape>` | P1 | Média | 🟡 Tipo existe |
| `ode`/`solve` | P1 | Alta | 🔴 Não iniciado |
| `discover` | P2 | Muito Alta | 🔴 Não iniciado |
| `do`/`counterfactual` | P2 | Muito Alta | 🟡 Keywords existem |
| `symbolic`/`simplify` | P3 | Muito Alta | 🔴 Não iniciado |

---

## Plano de Implementação

### Fase 1: Fundação (4-6 semanas)
1. **Autodiff básico** - dual numbers para `grad`
2. **Tensores verificados** - shapes em tempo de compilação
3. **`uncertain<T>`** - tipo com propagação de erro

### Fase 2: Probabilístico (4-6 semanas)
1. **Runtime para `Prob`** - handlers de efeito
2. **Distribuições básicas** - Normal, Gamma, Beta, etc.
3. **Inferência HMC** - sampler nativo

### Fase 3: Dinâmico (4-6 semanas)
1. **Tipo `ode`** - ODEs como valores
2. **Solvers nativos** - RK45, BDF
3. **Descoberta básica** - SINDy simplificado

### Fase 4: Causal + Simbólico (6-8 semanas)
1. **Operador `do`** - intervenções
2. **Counterfactuals** - queries contrafactuais
3. **Expressões simbólicas** - manipulação básica

---

## Comparação Final

| Feature | D | Julia | Python | Rust | C++ |
|---------|---|-------|--------|------|-----|
| **L0 (compilada, sem runtime)** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Autodiff nativo** | ✅ | Pkg | Pkg | Pkg | ❌ |
| **Probabilístico nativo** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Descoberta de modelos** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Inferência causal** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Incerteza nativa** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Unidades verificadas** | ✅ | Pkg | Pkg | Pkg | ❌ |
| **Ontologias** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Shapes verificados** | ✅ | ❌ | ❌ | ❌ | ❌ |

**Demetrios é a primeira e única linguagem L0 projetada para ciência.**

---

## Citação

Se você usar Demetrios em pesquisa, cite:

```bibtex
@software{demetrios2025,
  author = {Agourakis, Demetrios Chiuratto and Agourakis, Dionisio Chiuratto},
  title = {Demetrios: A Scientific L0 Programming Language},
  year = {2025},
  url = {https://github.com/demetrios-lang/demetrios}
}
```

---

*"A linguagem que a ciência merecia desde o início."*
