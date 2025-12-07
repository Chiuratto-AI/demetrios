# Implementation Roadmap: Novel GPU Paradigms in Demetrios

**Companion Document to:** GPU_COMPUTING_PARADIGMS_2024_2025.md  
**Focus:** Concrete implementation strategies and compiler architecture

---

## Overview

This document provides actionable implementation steps for incorporating cutting-edge GPU computing paradigms into the Demetrios compiler. Each section maps research findings to specific compiler passes, IR extensions, and backend implementations.

---

## Phase 1: Foundation (Months 1-3)

### 1.1 Extended Effect System

**Goal:** Add domain-specific effects for semantic computation

**Compiler Changes:**
```
compiler/src/effects/
├── mod.rs (existing)
├── builtin.rs (existing: IO, Mut, Alloc, Panic, Async, GPU)
├── domain_specific.rs (NEW)
│   ├── Parabolic, Hyperbolic, Elliptic (PDE types)
│   ├── Spatial<Distance> (locality-aware)
│   ├── Irregular (graph/sparse)
│   ├── Differentiable (autodiff)
│   └── Reversible, Thermodynamic (energy-aware)
└── effect_inference.rs (EXTEND)
    └── Infer domain effects from code patterns
```

**Example IR Extension:**
```rust
// compiler/src/effects/domain_specific.rs
#[derive(Debug, Clone, PartialEq)]
pub enum DomainEffect {
    // PDE classification
    Parabolic { stability_constraint: Option<StabilityBound> },
    Hyperbolic { cfl_condition: Option<CFLBound> },
    Elliptic,
    
    // Spatial computing
    Spatial { cutoff_radius: Option<Distance> },
    
    // Irregular computation
    Irregular { sparsity_hint: SparsityPattern },
    
    // Differentiability
    Differentiable { mode: ADMode },
    
    // Energy-aware
    Reversible,
    Thermodynamic,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ADMode {
    Forward,
    Reverse,
    Mixed,
    Auto, // Compiler selects
}
```

**Type Checker Integration:**
```rust
// compiler/src/check/mod.rs
impl TypeChecker {
    fn check_pde_stability(&mut self, expr: &Expr) -> Result<(), TypeError> {
        if let Some(parabolic) = self.infer_parabolic_pde(expr) {
            let dt = parabolic.time_step;
            let dx = parabolic.spatial_step;
            let alpha = parabolic.diffusivity;
            
            // Check stability: dt < dx^2 / (2 * alpha)
            if dt >= dx.pow(2) / (2.0 * alpha) {
                return Err(TypeError::StabilityViolation {
                    condition: "dt < dx^2 / (2*alpha)",
                    max_dt: dx.pow(2) / (2.0 * alpha),
                    actual_dt: dt,
                });
            }
        }
        Ok(())
    }
}
```

### 1.2 Uncertainty Types

**Goal:** First-class uncertain values with adaptive precision

**Type System Extension:**
```rust
// compiler/src/types/mod.rs
#[derive(Debug, Clone)]
pub struct UncertainType {
    pub base_type: TypeId,
    pub epistemic: bool,  // Reducible with more data
    pub aleatoric: bool,  // Irreducible randomness
}

impl Type {
    pub fn make_uncertain(base: TypeId) -> TypeId {
        TypeId::Uncertain(Box::new(UncertainType {
            base_type: base,
            epistemic: true,
            aleatoric: false,
        }))
    }
}
```

**HIR Lowering:**
```rust
// compiler/src/hir/lower_uncertain.rs
impl HirBuilder {
    fn lower_uncertain_op(&mut self, op: &UncertainOp) -> HirExpr {
        match op {
            UncertainOp::Sample(dist) => {
                // Generate both mean and std computations
                HirExpr::Struct {
                    fields: vec![
                        ("mean", self.lower_mean(dist)),
                        ("epistemic_std", self.lower_epistemic_std(dist)),
                        ("aleatoric_std", self.lower_aleatoric_std(dist)),
                    ]
                }
            }
            UncertainOp::AdaptivePrecision(expr) => {
                // Generate precision selection based on uncertainty
                HirExpr::If {
                    cond: Box::new(HirExpr::Binary {
                        op: BinOp::Gt,
                        lhs: Box::new(self.get_epistemic_std(expr)),
                        rhs: Box::new(HirExpr::Const(0.1)), // Threshold
                    }),
                    then_branch: Box::new(self.lower_with_precision(expr, Precision::FP16)),
                    else_branch: Box::new(self.lower_with_precision(expr, Precision::FP64)),
                }
            }
        }
    }
}
```

### 1.3 Spatial Array Type

**Goal:** Memory layouts encoding physical proximity

**AST Extension:**
```rust
// compiler/src/ast/types.rs
#[derive(Debug, Clone)]
pub struct SpatialArrayType {
    pub element_type: TypeId,
    pub domain: DomainSpec,
    pub layout: SpatialLayout,
}

#[derive(Debug, Clone)]
pub enum SpatialLayout {
    Morton,    // Z-order curve
    Hilbert,   // Hilbert curve
    CellList { cell_size: Distance },
    Octree,
    KDTree,
}

#[derive(Debug, Clone)]
pub struct DomainSpec {
    pub dimension: usize,
    pub bounds: BoundingBox,
    pub geometry: Option<GeometryType>,
}
```

**Codegen for Spatial Queries:**
```rust
// compiler/src/codegen/spatial.rs
impl CodeGenerator {
    fn generate_neighbor_query(
        &mut self,
        array: &SpatialArray,
        point: &Point,
        radius: &Distance,
    ) -> Vec<Instruction> {
        match array.layout {
            SpatialLayout::CellList { cell_size } => {
                // Generate cell-based neighbor search
                vec![
                    Instruction::ComputeCellIndex(point, cell_size),
                    Instruction::IterateNeighborCells(radius / cell_size),
                    Instruction::FilterByDistance(radius),
                ]
            }
            SpatialLayout::Octree => {
                // Generate tree traversal
                vec![
                    Instruction::OctreeRangeQuery(point, radius),
                ]
            }
            _ => todo!("Other spatial layouts"),
        }
    }
}
```

---

## Phase 2: Semantic Compilation (Months 4-6)

### 2.1 PDE Pattern Recognition

**Goal:** Identify equation types and select optimal solvers

**Pattern Matcher:**
```rust
// compiler/src/analysis/pde_patterns.rs
pub struct PDEPatternMatcher {
    patterns: Vec<PDEPattern>,
}

#[derive(Debug, Clone)]
pub struct PDEPattern {
    pub name: String,
    pub equation_type: EquationType,
    pub optimal_methods: Vec<NumericalMethod>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EquationType {
    HeatEquation { diffusivity: f64 },
    WaveEquation { wave_speed: f64 },
    PoissonEquation,
    NavierStokes { reynolds_number: f64 },
    SchrodingerEquation,
}

impl PDEPatternMatcher {
    pub fn recognize(&self, expr: &Expr) -> Option<PDEPattern> {
        // Pattern match against known PDE forms
        if self.matches_heat_equation(expr) {
            Some(PDEPattern {
                name: "Heat Equation".to_string(),
                equation_type: EquationType::HeatEquation {
                    diffusivity: self.extract_diffusivity(expr),
                },
                optimal_methods: vec![
                    NumericalMethod::ImplicitEuler,
                    NumericalMethod::CrankNicolson,
                    NumericalMethod::ADI, // For multidimensional
                ],
            })
        } else if self.matches_wave_equation(expr) {
            Some(PDEPattern {
                name: "Wave Equation".to_string(),
                equation_type: EquationType::WaveEquation {
                    wave_speed: self.extract_wave_speed(expr),
                },
                optimal_methods: vec![
                    NumericalMethod::LeapfrogIntegrator,
                    NumericalMethod::DiscontinuousGalerkin,
                ],
            })
        } else {
            None
        }
    }
    
    fn matches_heat_equation(&self, expr: &Expr) -> bool {
        // Look for pattern: du/dt = alpha * laplacian(u)
        matches!(expr, Expr::Binary {
            op: BinOp::Assign,
            lhs: box Expr::TimeDerivative(_),
            rhs: box Expr::Binary {
                op: BinOp::Mul,
                lhs: _,
                rhs: box Expr::Call { 
                    func: box Expr::Var(name), 
                    .. 
                } if name == "laplacian",
            },
        })
    }
}
```

**Optimization Pass:**
```rust
// compiler/src/optimize/pde_select.rs
pub struct PDESolverSelector {
    matcher: PDEPatternMatcher,
}

impl OptimizationPass for PDESolverSelector {
    fn run(&mut self, hir: &mut Hir) -> OptimizationResult {
        for func in &mut hir.functions {
            if let Some(pattern) = self.matcher.recognize(&func.body) {
                // Replace generic solver with specialized one
                func.body = self.generate_specialized_solver(
                    &pattern,
                    &func.body,
                );
                
                // Add stability checks
                self.insert_stability_assertions(&mut func.body, &pattern);
            }
        }
        OptimizationResult::Modified
    }
    
    fn generate_specialized_solver(
        &self,
        pattern: &PDEPattern,
        original: &Expr,
    ) -> Expr {
        match pattern.equation_type {
            EquationType::HeatEquation { diffusivity } => {
                // Generate Crank-Nicolson solver
                Expr::Call {
                    func: Box::new(Expr::Var("crank_nicolson_heat".to_string())),
                    args: vec![
                        self.extract_initial_condition(original),
                        Expr::Const(diffusivity),
                        self.extract_time_step(original),
                    ],
                }
            }
            _ => original.clone(),
        }
    }
}
```

### 2.2 Algorithm Database

**Goal:** Map equation characteristics to GPU implementations

**Database Schema:**
```rust
// compiler/src/analysis/algorithm_db.rs
pub struct AlgorithmDatabase {
    entries: HashMap<EquationSignature, Vec<AlgorithmEntry>>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct EquationSignature {
    pub equation_type: EquationType,
    pub dimension: usize,
    pub boundary_conditions: BoundaryType,
    pub domain_shape: DomainShape,
}

#[derive(Debug, Clone)]
pub struct AlgorithmEntry {
    pub method: NumericalMethod,
    pub complexity: Complexity,
    pub memory_usage: MemoryUsage,
    pub gpu_efficiency: f64,  // 0.0-1.0
    pub stability_requirements: Vec<StabilityConstraint>,
    pub code_generator: fn(&Expr) -> GpuKernel,
}

impl AlgorithmDatabase {
    pub fn select_best(
        &self,
        signature: &EquationSignature,
        hardware: &GpuCapabilities,
    ) -> Option<&AlgorithmEntry> {
        let candidates = self.entries.get(signature)?;
        
        // Score each algorithm
        candidates.iter()
            .max_by_key(|entry| {
                self.score_algorithm(entry, hardware)
            })
    }
    
    fn score_algorithm(
        &self,
        entry: &AlgorithmEntry,
        hardware: &GpuCapabilities,
    ) -> i32 {
        let mut score = 0;
        
        // Prefer algorithms with high GPU efficiency
        score += (entry.gpu_efficiency * 100.0) as i32;
        
        // Penalize if memory usage exceeds GPU capacity
        if entry.memory_usage.estimate() > hardware.memory_bytes {
            score -= 1000;
        }
        
        // Prefer lower computational complexity
        score -= entry.complexity.estimate() / 1000;
        
        score
    }
}
```

---

## Phase 3: Automatic Differentiation (Months 4-7)

### 3.1 `grad()` Operator

**Goal:** Language-level automatic differentiation

**AST Extension:**
```rust
// compiler/src/ast/expr.rs
#[derive(Debug, Clone)]
pub enum Expr {
    // ... existing variants
    Grad {
        func: Box<Expr>,
        wrt: Vec<String>,  // Variables to differentiate wrt
        mode: ADMode,
    },
}
```

**Type Checking:**
```rust
// compiler/src/check/mod.rs
impl TypeChecker {
    fn check_grad(&mut self, expr: &GradExpr) -> Result<TypeId, TypeError> {
        // Check that function is differentiable
        if !self.has_effect(&expr.func, Effect::Differentiable) {
            return Err(TypeError::NotDifferentiable {
                func: expr.func.clone(),
            });
        }
        
        // Input: f: T -> U
        let func_type = self.check_expr(&expr.func)?;
        
        // Output: grad(f): T -> U × (∂U/∂T)
        if expr.wrt.len() == 1 {
            // Single variable: return gradient type
            let input_type = self.extract_input_type(&func_type)?;
            let output_type = self.extract_output_type(&func_type)?;
            
            Ok(Type::Function {
                params: vec![input_type.clone()],
                ret: Type::Tuple(vec![output_type, input_type]),
            })
        } else {
            // Multiple variables: return Jacobian
            todo!("Multiple variable gradients")
        }
    }
}
```

### 3.2 AD Lowering Pass

**Goal:** Generate forward/reverse mode AD code

**HIR Transformation:**
```rust
// compiler/src/hir/autodiff.rs
pub struct AutodiffLowering {
    mode: ADMode,
    tape: Vec<ADInstruction>,
}

impl AutodiffLowering {
    pub fn lower_grad(&mut self, expr: &GradExpr) -> HirExpr {
        match expr.mode {
            ADMode::Forward => self.generate_forward_mode(expr),
            ADMode::Reverse => self.generate_reverse_mode(expr),
            ADMode::Auto => {
                // Heuristic: reverse mode if many inputs
                if expr.wrt.len() > 10 {
                    self.generate_reverse_mode(expr)
                } else {
                    self.generate_forward_mode(expr)
                }
            }
            _ => todo!(),
        }
    }
    
    fn generate_reverse_mode(&mut self, expr: &GradExpr) -> HirExpr {
        // 1. Forward pass: record operations on tape
        let forward_pass = self.build_tape(&expr.func);
        
        // 2. Backward pass: traverse tape in reverse
        let backward_pass = self.build_backward_pass();
        
        HirExpr::Block {
            stmts: vec![forward_pass],
            expr: Some(Box::new(backward_pass)),
        }
    }
    
    fn build_tape(&mut self, expr: &Expr) -> HirStmt {
        // Recursively walk expression tree, recording operations
        match expr {
            Expr::Binary { op, lhs, rhs } => {
                let lhs_id = self.record(lhs);
                let rhs_id = self.record(rhs);
                
                self.tape.push(ADInstruction::BinaryOp {
                    op: *op,
                    lhs: lhs_id,
                    rhs: rhs_id,
                    derivative: self.get_binary_op_derivative(*op),
                });
                
                // ... generate HIR for tape recording
                todo!()
            }
            _ => todo!("Other expression types"),
        }
    }
    
    fn build_backward_pass(&self) -> HirExpr {
        // Generate code to traverse tape backwards
        let mut stmts = Vec::new();
        
        for instruction in self.tape.iter().rev() {
            match instruction {
                ADInstruction::BinaryOp { op, lhs, rhs, derivative } => {
                    // Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)
                    stmts.push(HirStmt::Assign {
                        lhs: format!("grad_{}", lhs),
                        rhs: HirExpr::Binary {
                            op: BinOp::Mul,
                            lhs: Box::new(derivative.apply_lhs()),
                            rhs: Box::new(HirExpr::Var(format!("grad_out"))),
                        },
                    });
                }
            }
        }
        
        HirExpr::Block { stmts, expr: None }
    }
}
```

### 3.3 GPU Kernel Differentiation

**Goal:** Integrate with Enzyme-like LLVM pass

**MLIR Integration:**
```rust
// compiler/src/mlir/autodiff.rs
use mlir_sys::*;

pub struct EnzymeIntegration {
    context: MlirContext,
}

impl EnzymeIntegration {
    pub fn differentiate_kernel(
        &self,
        kernel: &MlirOperation,
        wrt: &[MlirValue],
    ) -> Result<MlirOperation, MlirError> {
        // Create Enzyme attribute
        let enzyme_attr = mlir_attribute_parse(
            self.context,
            c_str!("#enzyme.gradient"),
        );
        
        // Apply Enzyme pass to kernel
        let pass_manager = mlir_pass_manager_create(self.context);
        mlir_pass_manager_add_owned_pass(
            pass_manager,
            c_str!("enzyme-differentiate"),
        );
        
        // Run pass
        mlir_pass_manager_run(pass_manager, kernel);
        
        // Extract differentiated kernel
        todo!("Extract from pass manager")
    }
}
```

---

## Phase 4: GPU Optimization (Months 5-8)

### 4.1 Irregular Computation Optimization

**Goal:** Handle sparse/graph workloads efficiently

**Sparse Format Selection:**
```rust
// compiler/src/codegen/sparse.rs
pub struct SparseFormatSelector {
    profiler: RuntimeProfiler,
}

#[derive(Debug, Clone)]
pub enum SparseFormat {
    COO,  // Coordinate
    CSR,  // Compressed Sparse Row
    CSC,  // Compressed Sparse Column
    ELL,  // ELLPACK
    HYB,  // Hybrid (ELL + COO)
}

impl SparseFormatSelector {
    pub fn select_format(&self, matrix: &SparseMatrix) -> SparseFormat {
        // Decision based on matrix characteristics
        let avg_nnz_per_row = matrix.nnz as f64 / matrix.rows as f64;
        let variance = self.compute_nnz_variance(matrix);
        
        if variance < 0.2 * avg_nnz_per_row {
            // Regular sparsity pattern → ELL
            SparseFormat::ELL
        } else if avg_nnz_per_row < 8.0 {
            // Very sparse → COO
            SparseFormat::COO
        } else {
            // Mixed pattern → HYB
            SparseFormat::HYB
        }
    }
}
```

**Graph Kernel Optimization:**
```rust
// compiler/src/codegen/graph.rs
pub struct GraphKernelOptimizer;

impl GraphKernelOptimizer {
    pub fn optimize_gnn_kernel(&self, kernel: &GnnKernel) -> OptimizedKernel {
        // Analyze graph structure
        let degree_distribution = self.analyze_degree_distribution(&kernel.graph);
        
        if degree_distribution.is_power_law() {
            // Use vertex-centric for high-degree vertices
            // Use edge-centric for low-degree vertices
            self.generate_hybrid_kernel(kernel)
        } else {
            // Uniform degree distribution → simple vertex-centric
            self.generate_vertex_centric_kernel(kernel)
        }
    }
    
    fn generate_hybrid_kernel(&self, kernel: &GnnKernel) -> OptimizedKernel {
        // Split vertices by degree
        let high_degree_threshold = 32;
        
        OptimizedKernel {
            preprocessing: vec![
                Instruction::PartitionVerticesByDegree(high_degree_threshold),
            ],
            kernels: vec![
                // High-degree vertices: vertex-centric with warp-level reduction
                GpuKernel {
                    name: "gnn_high_degree",
                    launch_params: LaunchParams {
                        blocks: kernel.graph.high_degree_vertices.len() / 256,
                        threads: 256,
                    },
                    body: self.generate_vertex_centric_body(true),
                },
                // Low-degree vertices: edge-centric
                GpuKernel {
                    name: "gnn_low_degree",
                    launch_params: LaunchParams {
                        blocks: kernel.graph.num_edges / 256,
                        threads: 256,
                    },
                    body: self.generate_edge_centric_body(),
                },
            ],
        }
    }
}
```

### 4.2 Memory Layout Optimization

**Goal:** Transform irregular → regular for GPU efficiency

**Spatial Hash Grid:**
```rust
// compiler/src/codegen/spatial_hash.rs
pub struct SpatialHashGridTransform;

impl SpatialHashGridTransform {
    pub fn transform_irregular_access(
        &self,
        kernel: &Kernel,
    ) -> TransformedKernel {
        // Identify irregular neighbor queries
        let neighbor_queries = self.find_neighbor_queries(kernel);
        
        if neighbor_queries.is_empty() {
            return kernel.clone();
        }
        
        // Insert grid construction before kernel
        let grid_construction = self.generate_grid_construction(
            &neighbor_queries,
        );
        
        // Replace neighbor queries with grid lookups
        let transformed_body = self.replace_with_grid_lookups(
            &kernel.body,
            &neighbor_queries,
        );
        
        TransformedKernel {
            preprocessing: vec![grid_construction],
            main_kernel: Kernel {
                body: transformed_body,
                ..kernel.clone()
            },
        }
    }
    
    fn generate_grid_construction(&self, queries: &[NeighborQuery]) -> GpuKernel {
        // Determine grid cell size
        let cell_size = queries.iter()
            .map(|q| q.radius)
            .min()
            .unwrap();
        
        GpuKernel {
            name: "build_spatial_hash_grid",
            body: vec![
                // 1. Compute cell indices for all points
                Instruction::ParallelFor {
                    var: "i",
                    range: "0..num_points",
                    body: vec![
                        Instruction::Assign {
                            lhs: "cell_idx[i]",
                            rhs: "hash(points[i] / cell_size)",
                        },
                    ],
                },
                // 2. Sort points by cell index
                Instruction::Call {
                    func: "gpu_sort_by_key",
                    args: vec!["cell_idx", "points"],
                },
                // 3. Build cell start/end arrays
                Instruction::Call {
                    func: "compute_cell_offsets",
                    args: vec!["cell_idx", "cell_start", "cell_end"],
                },
            ],
        }
    }
}
```

### 4.3 Kernel Fusion

**Goal:** Merge operations to reduce memory traffic

**Fusion Pass:**
```rust
// compiler/src/optimize/kernel_fusion.rs
pub struct KernelFusionPass;

impl OptimizationPass for KernelFusionPass {
    fn run(&mut self, hir: &mut Hir) -> OptimizationResult {
        let mut fused = false;
        
        for func in &mut hir.functions {
            // Find fusable kernel sequences
            let sequences = self.find_fusable_sequences(&func.kernels);
            
            for seq in sequences {
                if self.should_fuse(&seq) {
                    let fused_kernel = self.fuse_kernels(&seq);
                    func.kernels.replace_range(
                        seq.start..seq.end,
                        vec![fused_kernel],
                    );
                    fused = true;
                }
            }
        }
        
        if fused {
            OptimizationResult::Modified
        } else {
            OptimizationResult::Unchanged
        }
    }
    
    fn should_fuse(&self, seq: &KernelSequence) -> bool {
        // Fusion criteria:
        // 1. Temporary array is only used between kernels
        // 2. Combined kernel fits in register/shared memory
        // 3. No synchronization required between operations
        
        let total_registers = seq.kernels.iter()
            .map(|k| k.register_usage)
            .sum::<usize>();
        
        let max_registers_per_thread = 255;
        
        seq.is_producer_consumer() 
            && total_registers < max_registers_per_thread
            && !seq.requires_sync()
    }
    
    fn fuse_kernels(&self, seq: &KernelSequence) -> GpuKernel {
        // Inline kernel bodies into single kernel
        let mut fused_body = Vec::new();
        
        for kernel in &seq.kernels {
            // Remove loads from global memory (will be in registers)
            let inlined = self.remove_global_loads(&kernel.body);
            fused_body.extend(inlined);
        }
        
        // Optimize: dead code elimination, register allocation
        self.optimize_fused_kernel(fused_body)
    }
}
```

---

## Phase 5: Advanced Features (Months 9-12)

### 5.1 Topological Data Analysis Primitives

**Goal:** First-class persistent homology support

**AST Types:**
```rust
// compiler/src/ast/types.rs
#[derive(Debug, Clone)]
pub struct SimplicialComplexType {
    pub max_dimension: usize,
}

#[derive(Debug, Clone)]
pub struct BarcodeType {
    pub dimension: usize,
}
```

**Codegen:**
```rust
// compiler/src/codegen/tda.rs
pub struct TDACodegen;

impl TDACodegen {
    pub fn generate_persistent_homology(
        &self,
        complex: &SimplicialComplex,
    ) -> GpuKernel {
        // Use GPU-accelerated matrix reduction
        GpuKernel {
            name: "persistent_homology",
            body: vec![
                // 1. Construct boundary matrices (sparse)
                Instruction::Call {
                    func: "build_boundary_matrices",
                    args: vec!["complex"],
                },
                // 2. Parallel reduction (critical for performance)
                Instruction::Call {
                    func: "gpu_sparse_matrix_reduction",
                    args: vec!["boundary_matrices"],
                },
                // 3. Extract persistence pairs
                Instruction::Call {
                    func: "extract_barcode",
                    args: vec!["reduced_matrices"],
                },
            ],
        }
    }
}
```

### 5.2 Multiscale Coupling

**Goal:** Seamlessly integrate atomic/continuum scales

**Type System:**
```rust
// compiler/src/types/scales.rs
pub trait Scale {
    type LengthUnit: Unit;
    type TimeUnit: Unit;
}

pub struct Atomic;
impl Scale for Atomic {
    type LengthUnit = Angstrom;
    type TimeUnit = Femtosecond;
}

pub struct Continuum;
impl Scale for Continuum {
    type LengthUnit = Meter;
    type TimeUnit = Second;
}

pub struct CoupledSystem<S1: Scale, S2: Scale> {
    pub fine_region: Domain<S1>,
    pub coarse_region: Domain<S2>,
    pub interface: Interface<S1, S2>,
}
```

**Code Generation:**
```rust
// compiler/src/codegen/multiscale.rs
impl CodeGenerator {
    fn generate_multiscale_step(
        &self,
        system: &CoupledSystem,
    ) -> Vec<GpuKernel> {
        vec![
            // 1. Fine-scale update (MD)
            self.generate_md_kernel(&system.fine_region),
            
            // 2. Coarse-scale update (FEM)
            self.generate_fem_kernel(&system.coarse_region),
            
            // 3. Interface coupling
            self.generate_coupling_kernel(&system.interface),
        ]
    }
    
    fn generate_coupling_kernel(
        &self,
        interface: &Interface,
    ) -> GpuKernel {
        GpuKernel {
            name: "interface_coupling",
            body: vec![
                // Atomic → Continuum: average atomic quantities
                Instruction::Call {
                    func: "atomic_to_continuum_average",
                    args: vec!["fine_forces", "coarse_forces"],
                },
                // Continuum → Atomic: interpolate continuum fields
                Instruction::Call {
                    func: "continuum_to_atomic_interpolate",
                    args: vec!["coarse_fields", "fine_fields"],
                },
            ],
        }
    }
}
```

---

## Testing Strategy

### Unit Tests

```rust
// compiler/tests/uncertainty_types.rs
#[test]
fn test_uncertain_type_inference() {
    let src = r#"
        fn sample() -> Uncertain<f64> with Prob {
            gaussian(0.0, 1.0)
        }
    "#;
    
    let ast = parse(src).unwrap();
    let typed = type_check(ast).unwrap();
    
    assert!(matches!(
        typed.return_type,
        Type::Uncertain { base: Type::F64, .. }
    ));
}

#[test]
fn test_pde_stability_check() {
    let src = r#"
        fn unstable_heat() {
            let dx = 0.1 * m;
            let dt = 1.0 * s;  // Too large!
            let alpha = 1.0 * m^2/s;
            
            solve_heat(u0, alpha, dt)
        }
    "#;
    
    let result = type_check(parse(src).unwrap());
    assert!(matches!(result, Err(TypeError::StabilityViolation { .. })));
}
```

### Integration Tests

```rust
// compiler/tests/integration/autodiff.rs
#[test]
fn test_grad_operator() {
    let src = r#"
        fn f(x: f64) -> f64 with Pure {
            x * x + 2.0 * x
        }
        
        let df = grad(f);
        let result = df(3.0);
    "#;
    
    let compiled = compile(src).unwrap();
    let result = execute(compiled).unwrap();
    
    assert_eq!(result, 8.0);  // df/dx = 2x + 2, at x=3: 2*3 + 2 = 8
}
```

### Benchmark Suite

```rust
// compiler/benches/sparse_kernels.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_sparse_matvec(c: &mut Criterion) {
    let src = r#"
        kernel fn sparse_matvec(
            matrix: &SparseMatrix,
            x: &GPUArray<f64>,
            y: &!GPUArray<f64>
        ) with GPU, Irregular {
            for row in matrix.rows {
                let mut sum = 0.0;
                for (col, val) in matrix.nonzeros(row) {
                    sum += val * x[col];
                }
                y[row] = sum;
            }
        }
    "#;
    
    let compiled = compile(src).unwrap();
    
    c.bench_function("sparse_matvec_coo", |b| {
        b.iter(|| execute_with_format(compiled, SparseFormat::COO))
    });
    
    c.bench_function("sparse_matvec_csr", |b| {
        b.iter(|| execute_with_format(compiled, SparseFormat::CSR))
    });
}

criterion_group!(benches, benchmark_sparse_matvec);
criterion_main!(benches);
```

---

## Milestones and Deliverables

### Month 3 Milestone
- [ ] Extended effect system with domain-specific effects
- [ ] `Uncertain<T>` type implementation
- [ ] `SpatialArray<T, D>` type implementation
- [ ] Basic PDE pattern recognition
- [ ] Unit tests passing (>90% coverage)

### Month 6 Milestone
- [ ] Semantic PDE compilation working
- [ ] Algorithm database with 10+ equation types
- [ ] `grad()` operator implemented
- [ ] Forward/reverse AD code generation
- [ ] Integration tests passing

### Month 9 Milestone
- [ ] Irregular computation optimizations
- [ ] Kernel fusion pass
- [ ] Spatial hash grid transformation
- [ ] Benchmark suite showing performance gains
- [ ] Documentation for all features

### Month 12 Milestone
- [ ] TDA primitives implemented
- [ ] Multiscale coupling support
- [ ] Full compiler pipeline tested
- [ ] Performance competitive with hand-written CUDA
- [ ] Research paper draft on novel language features

---

## Performance Targets

**Compared to hand-written CUDA:**
- Generic code: 70-90% performance
- Specialized (with semantic hints): 90-110% performance
- Automatic optimizations (fusion, layout): 80-95% performance

**Compared to Julia:**
- Startup time: <100ms (vs. ~1s for Julia)
- Peak performance: ≥100% (better due to static compilation)
- Memory usage: ~50% (no JIT overhead)

**Compared to PyTorch/JAX:**
- Compilation time: 2-5× faster (no Python overhead)
- Inference: 100-120% (ahead-of-time compilation)
- Training: 90-110% (automatic differentiation)

---

## Next Steps

1. **Immediate (This Week):**
   - Create `compiler/src/effects/domain_specific.rs`
   - Implement basic `Parabolic`, `Hyperbolic`, `Elliptic` effects
   - Write unit tests for effect inference

2. **Short Term (This Month):**
   - Implement `Uncertain<T>` type in type system
   - Add uncertainty propagation rules
   - Create `SpatialArray` AST nodes

3. **Medium Term (Next Quarter):**
   - Build PDE pattern matcher
   - Implement algorithm database
   - Begin automatic differentiation infrastructure

4. **Long Term (This Year):**
   - Complete all Phase 1-5 features
   - Extensive testing and benchmarking
   - Prepare research publication

---

**Document Version:** 1.0  
**Last Updated:** December 6, 2025  
**Status:** Ready for Implementation
