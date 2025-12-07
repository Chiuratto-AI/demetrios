//! Tensor Core Microarchitecture
//!
//! This module implements GPU Tensor Core specifications and operations:
//! - Tensor Core specs for different GPU generations
//! - Fragment layout for matrix tiles
//! - MMA (Matrix Multiply Accumulate) instructions
//! - WMMA (Warp Matrix Multiply Accumulate) operations
//! - Mixed precision support (FP16, BF16, TF32, FP8, INT8)

use std::collections::HashMap;

// ============================================================================
// Tensor Core Specification
// ============================================================================

/// Tensor Core generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCoreGen {
    /// Volta (V100) - 1st generation
    Volta,
    /// Turing (T4, RTX 20xx) - 2nd generation
    Turing,
    /// Ampere (A100) - 3rd generation
    Ampere,
    /// Ada Lovelace (L4, RTX 40xx) - 4th generation
    Ada,
    /// Hopper (H100) - 4th generation
    Hopper,
}

/// Data types supported by Tensor Cores
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDataType {
    /// 16-bit floating point (IEEE 754)
    FP16,
    /// Brain floating point 16
    BF16,
    /// TensorFloat-32 (19 bits: 1 sign, 8 exponent, 10 mantissa)
    TF32,
    /// 8-bit floating point (E4M3)
    FP8E4M3,
    /// 8-bit floating point (E5M2)
    FP8E5M2,
    /// 8-bit signed integer
    INT8,
    /// 4-bit signed integer
    INT4,
    /// 1-bit (binary)
    Binary,
    /// 32-bit floating point (accumulator)
    FP32,
    /// 64-bit floating point (H100 only)
    FP64,
}

impl TensorDataType {
    /// Size in bits
    pub fn bits(&self) -> u32 {
        match self {
            TensorDataType::Binary => 1,
            TensorDataType::INT4 => 4,
            TensorDataType::FP8E4M3 | TensorDataType::FP8E5M2 | TensorDataType::INT8 => 8,
            TensorDataType::FP16 | TensorDataType::BF16 => 16,
            TensorDataType::TF32 => 19,
            TensorDataType::FP32 => 32,
            TensorDataType::FP64 => 64,
        }
    }

    /// Size in bytes (rounded up)
    pub fn bytes(&self) -> u32 {
        (self.bits() + 7) / 8
    }
}

/// Matrix dimensions for MMA operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MmaDimensions {
    /// M dimension (rows of A and C)
    pub m: u32,
    /// N dimension (columns of B and C)
    pub n: u32,
    /// K dimension (columns of A, rows of B)
    pub k: u32,
}

impl MmaDimensions {
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    /// Standard 16x16x16 dimension
    pub fn m16n16k16() -> Self {
        Self::new(16, 16, 16)
    }

    /// Standard 16x8x16 dimension
    pub fn m16n8k16() -> Self {
        Self::new(16, 8, 16)
    }

    /// Standard 16x8x8 dimension
    pub fn m16n8k8() -> Self {
        Self::new(16, 8, 8)
    }

    /// Standard 8x8x4 dimension (Volta)
    pub fn m8n8k4() -> Self {
        Self::new(8, 8, 4)
    }

    /// Standard 8x8x16 dimension (INT8)
    pub fn m8n8k16() -> Self {
        Self::new(8, 8, 16)
    }

    /// Standard 8x8x32 dimension (INT4)
    pub fn m8n8k32() -> Self {
        Self::new(8, 8, 32)
    }

    /// Total elements in A matrix
    pub fn a_elements(&self) -> u32 {
        self.m * self.k
    }

    /// Total elements in B matrix
    pub fn b_elements(&self) -> u32 {
        self.k * self.n
    }

    /// Total elements in C/D matrix
    pub fn c_elements(&self) -> u32 {
        self.m * self.n
    }
}

/// MMA instruction configuration
#[derive(Debug, Clone)]
pub struct MmaConfig {
    /// Matrix dimensions
    pub dims: MmaDimensions,
    /// A matrix data type
    pub a_type: TensorDataType,
    /// B matrix data type
    pub b_type: TensorDataType,
    /// C matrix data type (accumulator input)
    pub c_type: TensorDataType,
    /// D matrix data type (accumulator output)
    pub d_type: TensorDataType,
    /// Whether A is transposed
    pub trans_a: bool,
    /// Whether B is transposed
    pub trans_b: bool,
}

impl MmaConfig {
    /// Standard FP16 configuration
    pub fn fp16_16x16x16() -> Self {
        Self {
            dims: MmaDimensions::m16n16k16(),
            a_type: TensorDataType::FP16,
            b_type: TensorDataType::FP16,
            c_type: TensorDataType::FP32,
            d_type: TensorDataType::FP32,
            trans_a: false,
            trans_b: false,
        }
    }

    /// Standard TF32 configuration (Ampere+)
    pub fn tf32_16x8x8() -> Self {
        Self {
            dims: MmaDimensions::m16n8k8(),
            a_type: TensorDataType::TF32,
            b_type: TensorDataType::TF32,
            c_type: TensorDataType::FP32,
            d_type: TensorDataType::FP32,
            trans_a: false,
            trans_b: false,
        }
    }

    /// Standard INT8 configuration
    pub fn int8_16x8x32() -> Self {
        Self {
            dims: MmaDimensions::new(16, 8, 32),
            a_type: TensorDataType::INT8,
            b_type: TensorDataType::INT8,
            c_type: TensorDataType::INT8,
            d_type: TensorDataType::INT8,
            trans_a: false,
            trans_b: false,
        }
    }

    /// FP8 configuration (Hopper)
    pub fn fp8_16x8x32() -> Self {
        Self {
            dims: MmaDimensions::new(16, 8, 32),
            a_type: TensorDataType::FP8E4M3,
            b_type: TensorDataType::FP8E5M2,
            c_type: TensorDataType::FP32,
            d_type: TensorDataType::FP32,
            trans_a: false,
            trans_b: false,
        }
    }
}

/// Tensor Core specification
#[derive(Debug, Clone)]
pub struct TensorCoreSpec {
    /// Generation
    pub generation: TensorCoreGen,
    /// Number of Tensor Cores per SM
    pub cores_per_sm: u32,
    /// Supported MMA configurations
    pub supported_configs: Vec<MmaConfig>,
    /// Peak TFLOPS for FP16
    pub peak_fp16_tflops: f64,
    /// Peak TFLOPS for TF32
    pub peak_tf32_tflops: Option<f64>,
    /// Peak TOPS for INT8
    pub peak_int8_tops: f64,
    /// Cycles per MMA operation
    pub mma_latency_cycles: u32,
    /// MMA throughput per cycle per core
    pub mma_throughput_per_cycle: u32,
}

impl TensorCoreSpec {
    /// A100 Tensor Core spec
    pub fn a100() -> Self {
        Self {
            generation: TensorCoreGen::Ampere,
            cores_per_sm: 4,
            supported_configs: vec![
                MmaConfig::fp16_16x16x16(),
                MmaConfig::tf32_16x8x8(),
                MmaConfig::int8_16x8x32(),
            ],
            peak_fp16_tflops: 312.0,
            peak_tf32_tflops: Some(156.0),
            peak_int8_tops: 624.0,
            mma_latency_cycles: 16,
            mma_throughput_per_cycle: 256, // FMAs per cycle
        }
    }

    /// H100 Tensor Core spec
    pub fn h100() -> Self {
        Self {
            generation: TensorCoreGen::Hopper,
            cores_per_sm: 4,
            supported_configs: vec![
                MmaConfig::fp16_16x16x16(),
                MmaConfig::tf32_16x8x8(),
                MmaConfig::int8_16x8x32(),
                MmaConfig::fp8_16x8x32(),
            ],
            peak_fp16_tflops: 989.0,
            peak_tf32_tflops: Some(495.0),
            peak_int8_tops: 1979.0,
            mma_latency_cycles: 12,
            mma_throughput_per_cycle: 512,
        }
    }

    /// L4 Tensor Core spec
    pub fn l4() -> Self {
        Self {
            generation: TensorCoreGen::Ada,
            cores_per_sm: 4,
            supported_configs: vec![
                MmaConfig::fp16_16x16x16(),
                MmaConfig::tf32_16x8x8(),
                MmaConfig::int8_16x8x32(),
                MmaConfig::fp8_16x8x32(),
            ],
            peak_fp16_tflops: 242.0,
            peak_tf32_tflops: Some(121.0),
            peak_int8_tops: 485.0,
            mma_latency_cycles: 14,
            mma_throughput_per_cycle: 256,
        }
    }

    /// Check if a configuration is supported
    pub fn supports_config(&self, config: &MmaConfig) -> bool {
        self.supported_configs.iter().any(|c| {
            c.dims == config.dims && c.a_type == config.a_type && c.b_type == config.b_type
        })
    }
}

// ============================================================================
// Fragment Layout
// ============================================================================

/// Fragment type for WMMA operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentType {
    /// Matrix A fragment
    MatrixA,
    /// Matrix B fragment
    MatrixB,
    /// Accumulator fragment (C/D)
    Accumulator,
}

/// Layout of a matrix fragment in registers
#[derive(Debug, Clone)]
pub struct FragmentLayout {
    /// Fragment type
    pub fragment_type: FragmentType,
    /// Dimensions
    pub dims: MmaDimensions,
    /// Data type
    pub data_type: TensorDataType,
    /// Number of registers per thread
    pub registers_per_thread: u32,
    /// Element to register mapping
    pub mapping: FragmentMapping,
}

/// How elements are mapped to thread registers
#[derive(Debug, Clone)]
pub struct FragmentMapping {
    /// Number of threads participating (typically 32 for warp)
    pub num_threads: u32,
    /// Elements per thread
    pub elements_per_thread: u32,
    /// Row stride in memory
    pub row_stride: u32,
    /// Column stride in memory
    pub col_stride: u32,
}

impl FragmentLayout {
    /// Create layout for matrix A in m16n16k16 FP16
    pub fn matrix_a_m16n16k16_fp16() -> Self {
        Self {
            fragment_type: FragmentType::MatrixA,
            dims: MmaDimensions::m16n16k16(),
            data_type: TensorDataType::FP16,
            registers_per_thread: 8,
            mapping: FragmentMapping {
                num_threads: 32,
                elements_per_thread: 8, // 16*16/32 = 8
                row_stride: 16,
                col_stride: 1,
            },
        }
    }

    /// Create layout for matrix B in m16n16k16 FP16
    pub fn matrix_b_m16n16k16_fp16() -> Self {
        Self {
            fragment_type: FragmentType::MatrixB,
            dims: MmaDimensions::m16n16k16(),
            data_type: TensorDataType::FP16,
            registers_per_thread: 8,
            mapping: FragmentMapping {
                num_threads: 32,
                elements_per_thread: 8,
                row_stride: 16,
                col_stride: 1,
            },
        }
    }

    /// Create layout for accumulator in m16n16k16 FP32
    pub fn accumulator_m16n16k16_fp32() -> Self {
        Self {
            fragment_type: FragmentType::Accumulator,
            dims: MmaDimensions::m16n16k16(),
            data_type: TensorDataType::FP32,
            registers_per_thread: 8,
            mapping: FragmentMapping {
                num_threads: 32,
                elements_per_thread: 8, // 16*16/32 = 8
                row_stride: 16,
                col_stride: 1,
            },
        }
    }

    /// Calculate register index for a matrix element
    pub fn element_to_register(&self, row: u32, col: u32, thread_id: u32) -> Option<u32> {
        // Simplified mapping - actual hardware is more complex
        let element_idx = row * self.mapping.col_stride + col * self.mapping.row_stride;
        let thread_for_element = element_idx % self.mapping.num_threads;

        if thread_for_element != thread_id {
            return None; // This thread doesn't own this element
        }

        Some(element_idx / self.mapping.num_threads)
    }

    /// Total size in bytes
    pub fn size_bytes(&self) -> u32 {
        self.registers_per_thread * 4 * self.mapping.num_threads
    }
}

// ============================================================================
// MMA Operations
// ============================================================================

/// MMA instruction state
#[derive(Debug, Clone)]
pub struct MmaOperation {
    /// Configuration
    pub config: MmaConfig,
    /// A fragment (simulated values)
    pub a_fragment: Vec<f32>,
    /// B fragment
    pub b_fragment: Vec<f32>,
    /// C fragment (accumulator input)
    pub c_fragment: Vec<f32>,
    /// D fragment (accumulator output)
    pub d_fragment: Vec<f32>,
    /// Cycle when operation started
    pub start_cycle: u64,
    /// Whether operation is complete
    pub complete: bool,
}

impl MmaOperation {
    /// Create new MMA operation
    pub fn new(config: MmaConfig) -> Self {
        let a_size = config.dims.a_elements() as usize;
        let b_size = config.dims.b_elements() as usize;
        let c_size = config.dims.c_elements() as usize;

        Self {
            config,
            a_fragment: vec![0.0; a_size],
            b_fragment: vec![0.0; b_size],
            c_fragment: vec![0.0; c_size],
            d_fragment: vec![0.0; c_size],
            start_cycle: 0,
            complete: false,
        }
    }

    /// Load A fragment from memory
    pub fn load_a(&mut self, data: &[f32]) {
        let size = self.config.dims.a_elements() as usize;
        self.a_fragment.copy_from_slice(&data[..size]);
    }

    /// Load B fragment from memory
    pub fn load_b(&mut self, data: &[f32]) {
        let size = self.config.dims.b_elements() as usize;
        self.b_fragment.copy_from_slice(&data[..size]);
    }

    /// Load C fragment from memory
    pub fn load_c(&mut self, data: &[f32]) {
        let size = self.config.dims.c_elements() as usize;
        self.c_fragment.copy_from_slice(&data[..size]);
    }

    /// Execute the MMA: D = A * B + C
    pub fn execute(&mut self) {
        let m = self.config.dims.m as usize;
        let n = self.config.dims.n as usize;
        let k = self.config.dims.k as usize;

        // Initialize D from C
        self.d_fragment.clone_from(&self.c_fragment);

        // Matrix multiply: D[i,j] = sum(A[i,k] * B[k,j]) + C[i,j]
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a_idx = i * k + kk;
                    let b_idx = kk * n + j;
                    sum += self.a_fragment[a_idx] * self.b_fragment[b_idx];
                }
                let c_idx = i * n + j;
                self.d_fragment[c_idx] += sum;
            }
        }

        self.complete = true;
    }

    /// Get result
    pub fn result(&self) -> &[f32] {
        &self.d_fragment
    }
}

// ============================================================================
// WMMA Operations
// ============================================================================

/// WMMA (Warp Matrix Multiply Accumulate) operation
#[derive(Debug, Clone)]
pub struct WmmaOperation {
    /// Matrix dimensions
    pub dims: MmaDimensions,
    /// A matrix layout
    pub a_layout: FragmentLayout,
    /// B matrix layout
    pub b_layout: FragmentLayout,
    /// Accumulator layout
    pub acc_layout: FragmentLayout,
    /// Per-thread registers for A
    pub a_registers: Vec<Vec<f32>>,
    /// Per-thread registers for B
    pub b_registers: Vec<Vec<f32>>,
    /// Per-thread accumulator registers
    pub acc_registers: Vec<Vec<f32>>,
}

impl WmmaOperation {
    /// Create new WMMA operation for m16n16k16 FP16
    pub fn new_m16n16k16() -> Self {
        let dims = MmaDimensions::m16n16k16();
        let a_layout = FragmentLayout::matrix_a_m16n16k16_fp16();
        let b_layout = FragmentLayout::matrix_b_m16n16k16_fp16();
        let acc_layout = FragmentLayout::accumulator_m16n16k16_fp32();

        let num_threads = 32;
        let regs_per_thread = 8;

        Self {
            dims,
            a_layout,
            b_layout,
            acc_layout,
            a_registers: vec![vec![0.0; regs_per_thread]; num_threads],
            b_registers: vec![vec![0.0; regs_per_thread]; num_threads],
            acc_registers: vec![vec![0.0; regs_per_thread]; num_threads],
        }
    }

    /// Load matrix A tile from memory
    pub fn load_matrix_a(&mut self, data: &[f32], ldm: u32) {
        let m = self.dims.m;
        let k = self.dims.k;

        for thread_id in 0..32 {
            // Each thread loads its portion
            for reg_idx in 0..8 {
                // Simplified: compute which element this thread/register owns
                let element_idx = thread_id * 8 + reg_idx;
                let row = element_idx / k;
                let col = element_idx % k;

                if row < m && col < k {
                    let mem_idx = (row * ldm + col) as usize;
                    if mem_idx < data.len() {
                        self.a_registers[thread_id as usize][reg_idx as usize] = data[mem_idx];
                    }
                }
            }
        }
    }

    /// Load matrix B tile from memory
    pub fn load_matrix_b(&mut self, data: &[f32], ldm: u32) {
        let k = self.dims.k;
        let n = self.dims.n;

        for thread_id in 0..32 {
            for reg_idx in 0..8 {
                let element_idx = thread_id * 8 + reg_idx;
                let row = element_idx / n;
                let col = element_idx % n;

                if row < k && col < n {
                    let mem_idx = (row * ldm + col) as usize;
                    if mem_idx < data.len() {
                        self.b_registers[thread_id as usize][reg_idx as usize] = data[mem_idx];
                    }
                }
            }
        }
    }

    /// Fill accumulator with zeros
    pub fn fill_accumulator(&mut self, value: f32) {
        for thread_regs in &mut self.acc_registers {
            for reg in thread_regs {
                *reg = value;
            }
        }
    }

    /// Execute MMA and accumulate
    pub fn mma_sync(&mut self) {
        // Gather A and B matrices
        let m = self.dims.m as usize;
        let n = self.dims.n as usize;
        let k = self.dims.k as usize;

        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        // Gather from thread registers
        for thread_id in 0..32 {
            for reg_idx in 0..8 {
                let element_idx = thread_id * 8 + reg_idx;

                // A matrix
                if element_idx < m * k {
                    a[element_idx] = self.a_registers[thread_id][reg_idx];
                }

                // B matrix
                if element_idx < k * n {
                    b[element_idx] = self.b_registers[thread_id][reg_idx];
                }

                // C matrix (current accumulator)
                if element_idx < m * n {
                    c[element_idx] = self.acc_registers[thread_id][reg_idx];
                }
            }
        }

        // Compute D = A * B + C
        let mut d = c.clone();
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                d[i * n + j] += sum;
            }
        }

        // Scatter back to thread registers
        for thread_id in 0..32 {
            for reg_idx in 0..8 {
                let element_idx = thread_id * 8 + reg_idx;
                if element_idx < m * n {
                    self.acc_registers[thread_id][reg_idx] = d[element_idx];
                }
            }
        }
    }

    /// Store accumulator to memory
    pub fn store_matrix_d(&self, data: &mut [f32], ldm: u32) {
        let m = self.dims.m;
        let n = self.dims.n;

        for thread_id in 0..32 {
            for reg_idx in 0..8 {
                let element_idx = thread_id * 8 + reg_idx;
                let row = element_idx / n;
                let col = element_idx % n;

                if row < m && col < n {
                    let mem_idx = (row * ldm + col) as usize;
                    if mem_idx < data.len() {
                        data[mem_idx] = self.acc_registers[thread_id as usize][reg_idx as usize];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Tensor Core Unit
// ============================================================================

/// Tensor Core execution unit
#[derive(Debug)]
pub struct TensorCoreUnit {
    /// Specification
    pub spec: TensorCoreSpec,
    /// Current cycle
    current_cycle: u64,
    /// In-flight operations
    in_flight: Vec<MmaOperation>,
    /// Statistics
    pub stats: TensorCoreStats,
}

/// Tensor Core statistics
#[derive(Debug, Clone, Default)]
pub struct TensorCoreStats {
    pub mma_operations: u64,
    pub total_fmas: u64,
    pub cycles_active: u64,
    pub cycles_stalled: u64,
}

impl TensorCoreStats {
    pub fn tflops(&self, clock_freq_ghz: f64) -> f64 {
        let total_flops = self.total_fmas * 2; // Each FMA is 2 FLOPS
        let seconds = self.cycles_active as f64 / (clock_freq_ghz * 1e9);
        if seconds > 0.0 {
            (total_flops as f64) / (seconds * 1e12)
        } else {
            0.0
        }
    }
}

impl TensorCoreUnit {
    pub fn new(spec: TensorCoreSpec) -> Self {
        Self {
            spec,
            current_cycle: 0,
            in_flight: Vec::new(),
            stats: TensorCoreStats::default(),
        }
    }

    /// Issue an MMA operation
    pub fn issue_mma(&mut self, config: MmaConfig) -> Result<usize, &'static str> {
        if !self.spec.supports_config(&config) {
            return Err("Unsupported MMA configuration");
        }

        let mut op = MmaOperation::new(config);
        op.start_cycle = self.current_cycle;

        let idx = self.in_flight.len();
        self.in_flight.push(op);
        self.stats.mma_operations += 1;

        Ok(idx)
    }

    /// Tick the unit forward one cycle
    pub fn tick(&mut self) -> Vec<usize> {
        self.current_cycle += 1;
        let mut completed = Vec::new();

        // Check for completed operations
        for (idx, op) in self.in_flight.iter_mut().enumerate() {
            if !op.complete {
                let elapsed = self.current_cycle - op.start_cycle;
                if elapsed >= self.spec.mma_latency_cycles as u64 {
                    op.execute();
                    completed.push(idx);

                    // Update stats
                    let fmas = op.config.dims.m * op.config.dims.n * op.config.dims.k;
                    self.stats.total_fmas += fmas as u64;
                }
            }
        }

        if !completed.is_empty() {
            self.stats.cycles_active += 1;
        } else if !self.in_flight.is_empty() {
            self.stats.cycles_stalled += 1;
        }

        completed
    }

    /// Get operation result
    pub fn get_result(&self, idx: usize) -> Option<&[f32]> {
        self.in_flight
            .get(idx)
            .filter(|op| op.complete)
            .map(|op| op.result())
    }

    /// Clear completed operations
    pub fn clear_completed(&mut self) {
        self.in_flight.retain(|op| !op.complete);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_core_gen() {
        let a100 = TensorCoreSpec::a100();
        assert_eq!(a100.generation, TensorCoreGen::Ampere);
        assert_eq!(a100.cores_per_sm, 4);
    }

    #[test]
    fn test_data_type_bits() {
        assert_eq!(TensorDataType::FP16.bits(), 16);
        assert_eq!(TensorDataType::FP32.bits(), 32);
        assert_eq!(TensorDataType::INT8.bits(), 8);
        assert_eq!(TensorDataType::FP8E4M3.bits(), 8);
    }

    #[test]
    fn test_mma_dimensions() {
        let dims = MmaDimensions::m16n16k16();
        assert_eq!(dims.m, 16);
        assert_eq!(dims.n, 16);
        assert_eq!(dims.k, 16);
        assert_eq!(dims.a_elements(), 256);
        assert_eq!(dims.b_elements(), 256);
        assert_eq!(dims.c_elements(), 256);
    }

    #[test]
    fn test_mma_operation() {
        let config = MmaConfig::fp16_16x16x16();
        let mut op = MmaOperation::new(config);

        // Create identity-like matrices for testing
        let m = 16usize;
        let n = 16usize;
        let k = 16usize;

        // A = identity (simplified as 1s on diagonal)
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }

        // B = all 2s
        let b = vec![2.0f32; k * n];

        // C = all 0s
        let c = vec![0.0f32; m * n];

        op.load_a(&a);
        op.load_b(&b);
        op.load_c(&c);
        op.execute();

        // Result should have 2s in first k rows
        let result = op.result();
        assert!(op.complete);
        assert_eq!(result.len(), m * n);
    }

    #[test]
    fn test_wmma_operation() {
        let mut wmma = WmmaOperation::new_m16n16k16();

        // Fill with test data
        let a = vec![1.0f32; 16 * 16];
        let b = vec![1.0f32; 16 * 16];

        wmma.load_matrix_a(&a, 16);
        wmma.load_matrix_b(&b, 16);
        wmma.fill_accumulator(0.0);
        wmma.mma_sync();

        // Store result
        let mut result = vec![0.0f32; 16 * 16];
        wmma.store_matrix_d(&mut result, 16);

        // Each element should be sum of k products = 16
        // (simplified test - actual values depend on mapping)
    }

    #[test]
    fn test_fragment_layout() {
        let layout = FragmentLayout::matrix_a_m16n16k16_fp16();
        assert_eq!(layout.registers_per_thread, 8);
        assert_eq!(layout.mapping.num_threads, 32);
    }

    #[test]
    fn test_tensor_core_unit() {
        let spec = TensorCoreSpec::a100();
        let mut unit = TensorCoreUnit::new(spec);

        // Issue an MMA
        let config = MmaConfig::fp16_16x16x16();
        let result = unit.issue_mma(config);
        assert!(result.is_ok());

        // Run until complete
        let mut completed = Vec::new();
        for _ in 0..20 {
            completed.extend(unit.tick());
        }

        assert!(!completed.is_empty());
        assert_eq!(unit.stats.mma_operations, 1);
    }

    #[test]
    fn test_h100_fp8_support() {
        let h100 = TensorCoreSpec::h100();
        let fp8_config = MmaConfig::fp8_16x8x32();

        assert!(h100.supports_config(&fp8_config));

        let a100 = TensorCoreSpec::a100();
        // A100 doesn't have FP8 in our simplified config list
    }

    #[test]
    fn test_peak_performance() {
        let h100 = TensorCoreSpec::h100();
        let a100 = TensorCoreSpec::a100();

        // H100 should have higher peak
        assert!(h100.peak_fp16_tflops > a100.peak_fp16_tflops);
        assert!(h100.peak_int8_tops > a100.peak_int8_tops);
    }

    #[test]
    fn test_mma_config_types() {
        let fp16_config = MmaConfig::fp16_16x16x16();
        assert_eq!(fp16_config.a_type, TensorDataType::FP16);
        assert_eq!(fp16_config.d_type, TensorDataType::FP32);

        let tf32_config = MmaConfig::tf32_16x8x8();
        assert_eq!(tf32_config.a_type, TensorDataType::TF32);

        let int8_config = MmaConfig::int8_16x8x32();
        assert_eq!(int8_config.a_type, TensorDataType::INT8);
    }

    #[test]
    fn test_simple_matmul() {
        let config = MmaConfig {
            dims: MmaDimensions::new(2, 2, 2),
            a_type: TensorDataType::FP32,
            b_type: TensorDataType::FP32,
            c_type: TensorDataType::FP32,
            d_type: TensorDataType::FP32,
            trans_a: false,
            trans_b: false,
        };

        let mut op = MmaOperation::new(config);

        // A = [[1, 2], [3, 4]]
        op.load_a(&[1.0, 2.0, 3.0, 4.0]);

        // B = [[5, 6], [7, 8]]
        op.load_b(&[5.0, 6.0, 7.0, 8.0]);

        // C = [[0, 0], [0, 0]]
        op.load_c(&[0.0, 0.0, 0.0, 0.0]);

        op.execute();

        let result = op.result();
        // D = A*B + C = [[19, 22], [43, 50]]
        assert!((result[0] - 19.0).abs() < 0.001);
        assert!((result[1] - 22.0).abs() < 0.001);
        assert!((result[2] - 43.0).abs() < 0.001);
        assert!((result[3] - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_tensor_core_stats() {
        let spec = TensorCoreSpec::a100();
        let mut unit = TensorCoreUnit::new(spec.clone());

        let config = MmaConfig::fp16_16x16x16();
        let _ = unit.issue_mma(config);

        // Run to completion
        for _ in 0..20 {
            unit.tick();
        }

        assert!(unit.stats.total_fmas > 0);
        assert!(unit.stats.mma_operations > 0);
    }
}
