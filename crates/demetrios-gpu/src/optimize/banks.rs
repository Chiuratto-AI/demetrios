//! Shared Memory Bank Conflict Analysis
//!
//! Shared memory is divided into 32 banks (same as warp size). Each bank
//! can service one request per cycle. When multiple threads access the
//! same bank (but different addresses), accesses are serialized.
//!
//! This module detects and suggests fixes for bank conflicts.

/// Shared memory access pattern
#[derive(Debug, Clone)]
pub enum SharedAccessPattern {
    /// addr = base + tid * stride
    Linear { base: u32, stride: u32 },
    /// addr = base + indices[tid]
    Indexed { base: u32, indices: Vec<u32> },
    /// All threads access same address (free broadcast)
    Broadcast { addr: u32 },
}

/// Bank conflict report
#[derive(Debug, Clone)]
pub struct BankConflictReport {
    /// Maximum threads hitting same bank (1 = no conflict, 32 = worst)
    pub conflict_degree: u32,
    /// Efficiency (1/conflict_degree)
    pub efficiency: f32,
    /// Pattern classification
    pub pattern_type: ConflictPatternType,
    /// Suggested fix
    pub suggestion: Option<BankConflictFix>,
}

impl BankConflictReport {
    /// Check if there are any conflicts
    pub fn has_conflicts(&self) -> bool {
        self.conflict_degree > 1
    }

    /// Check if conflicts are severe (more than 4-way)
    pub fn is_severe(&self) -> bool {
        self.conflict_degree > 4
    }
}

/// Classification of conflict pattern
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictPatternType {
    /// No conflicts - perfect
    NoConflict,
    /// All threads read same address - free broadcast
    Broadcast,
    /// All threads hit same bank - 32-way conflict
    FullConflict,
    /// Regular strided pattern with known degree
    Strided { degree: u32 },
    /// Irregular pattern
    Irregular,
}

/// Suggested fix for bank conflicts
#[derive(Debug, Clone)]
pub enum BankConflictFix {
    /// Add padding to array rows
    Padding { bytes: usize },
    /// Use XOR-based indexing
    XorIndexing { mask: u32 },
    /// Rearrange data layout (swizzle)
    Swizzle,
    /// No fix needed or possible
    None,
}

/// Greatest common divisor
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Shared memory bank conflict analyzer
pub struct BankConflictAnalyzer {
    /// Number of banks (32 on modern NVIDIA)
    num_banks: u32,
    /// Bank width in bytes (4 on modern NVIDIA)
    bank_width: u32,
    /// Warp size
    warp_size: u32,
}

impl BankConflictAnalyzer {
    /// Create a new analyzer with default settings (NVIDIA GPU)
    pub fn new() -> Self {
        Self {
            num_banks: 32,
            bank_width: 4,
            warp_size: 32,
        }
    }

    /// Create analyzer with custom bank configuration
    pub fn with_config(num_banks: u32, bank_width: u32, warp_size: u32) -> Self {
        Self {
            num_banks,
            bank_width,
            warp_size,
        }
    }

    /// Which bank does an address map to?
    pub fn address_to_bank(&self, addr: u32) -> u32 {
        (addr / self.bank_width) % self.num_banks
    }

    /// Analyze shared memory access pattern
    pub fn analyze_access(&self, pattern: &SharedAccessPattern) -> BankConflictReport {
        match pattern {
            SharedAccessPattern::Linear { base, stride } => self.analyze_linear(*base, *stride),
            SharedAccessPattern::Indexed { base, indices } => self.analyze_indexed(*base, indices),
            SharedAccessPattern::Broadcast { .. } => {
                // Broadcast is free
                BankConflictReport {
                    conflict_degree: 1,
                    efficiency: 1.0,
                    pattern_type: ConflictPatternType::Broadcast,
                    suggestion: Some(BankConflictFix::None),
                }
            }
        }
    }

    /// Analyze linear access: addr = base + tid * stride
    fn analyze_linear(&self, base: u32, stride: u32) -> BankConflictReport {
        // Calculate which bank each thread hits
        let mut bank_counts = [0u32; 32];

        for tid in 0..self.warp_size {
            let addr = base.wrapping_add(tid.wrapping_mul(stride));
            let bank = self.address_to_bank(addr);
            bank_counts[bank as usize] += 1;
        }

        let max_conflicts = *bank_counts.iter().max().unwrap();

        // Check for special patterns
        let pattern_type = if max_conflicts == 1 {
            ConflictPatternType::NoConflict
        } else if stride % (self.bank_width * self.num_banks) == 0 {
            // All threads hit the same bank!
            ConflictPatternType::FullConflict
        } else if stride % self.bank_width == 0 {
            // Regular strided conflict
            let effective_stride = stride / self.bank_width;
            let g = gcd(effective_stride, self.num_banks);
            ConflictPatternType::Strided {
                degree: self.num_banks / g,
            }
        } else {
            ConflictPatternType::Irregular
        };

        let suggestion = self.suggest_fix(&pattern_type, stride);

        BankConflictReport {
            conflict_degree: max_conflicts,
            efficiency: 1.0 / max_conflicts as f32,
            pattern_type,
            suggestion,
        }
    }

    /// Analyze indexed access: addr = base + indices[tid]
    fn analyze_indexed(&self, base: u32, indices: &[u32]) -> BankConflictReport {
        let mut bank_counts = [0u32; 32];

        for &idx in indices.iter().take(self.warp_size as usize) {
            let addr = base.wrapping_add(idx);
            let bank = self.address_to_bank(addr);
            bank_counts[bank as usize] += 1;
        }

        let max_conflicts = *bank_counts.iter().max().unwrap_or(&1);

        BankConflictReport {
            conflict_degree: max_conflicts,
            efficiency: 1.0 / max_conflicts as f32,
            pattern_type: ConflictPatternType::Irregular,
            suggestion: Some(BankConflictFix::Padding { bytes: 4 }),
        }
    }

    /// Suggest fix for bank conflicts
    fn suggest_fix(&self, pattern: &ConflictPatternType, stride: u32) -> Option<BankConflictFix> {
        match pattern {
            ConflictPatternType::NoConflict => Some(BankConflictFix::None),
            ConflictPatternType::Broadcast => Some(BankConflictFix::None),

            ConflictPatternType::FullConflict => {
                // Add 4 bytes padding to each row
                Some(BankConflictFix::Padding {
                    bytes: self.bank_width as usize,
                })
            }

            ConflictPatternType::Strided { degree } if *degree > 4 => {
                // XOR-based indexing can help
                Some(BankConflictFix::XorIndexing {
                    mask: self.compute_xor_mask(stride),
                })
            }

            ConflictPatternType::Strided { .. } => Some(BankConflictFix::Padding {
                bytes: self.bank_width as usize,
            }),

            ConflictPatternType::Irregular => Some(BankConflictFix::Padding {
                bytes: self.bank_width as usize,
            }),
        }
    }

    /// Compute XOR mask to reduce conflicts
    fn compute_xor_mask(&self, stride: u32) -> u32 {
        // XOR indexing: bank = ((addr/4) ^ (addr/128)) % 32
        // Helps with power-of-2 strides
        if stride.is_power_of_two() {
            stride.trailing_zeros()
        } else {
            0
        }
    }

    /// Calculate padded array size to avoid conflicts
    pub fn padded_size(&self, rows: usize, cols: usize, elem_size: usize) -> usize {
        // Add one element padding per row
        let padded_cols = cols + (self.bank_width as usize / elem_size);
        rows * padded_cols * elem_size
    }

    /// Calculate padded row stride
    pub fn padded_stride(&self, cols: usize, elem_size: usize) -> usize {
        let original_stride = cols * elem_size;
        // Add one bank width of padding
        original_stride + self.bank_width as usize
    }

    /// Check if a stride will cause full conflicts
    pub fn causes_full_conflict(&self, stride: u32) -> bool {
        stride % (self.bank_width * self.num_banks) == 0
    }

    /// Get optimal stride for conflict-free access
    pub fn optimal_stride(&self, element_size: usize) -> usize {
        // Stride should be coprime with num_banks * bank_width
        // Adding one element usually works
        element_size + self.bank_width as usize
    }
}

impl Default for BankConflictAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Matrix transpose helper - common source of bank conflicts
pub struct TransposeHelper {
    analyzer: BankConflictAnalyzer,
}

impl TransposeHelper {
    /// Create a new transpose helper
    pub fn new() -> Self {
        Self {
            analyzer: BankConflictAnalyzer::new(),
        }
    }

    /// Calculate tile dimensions for conflict-free transpose
    pub fn optimal_tile_size(&self, element_size: usize) -> (usize, usize) {
        // 32x33 tiles avoid bank conflicts for 4-byte elements
        let tile_dim = 32;
        let padded_dim = tile_dim + (self.analyzer.bank_width as usize / element_size);
        (tile_dim, padded_dim)
    }

    /// Check if a tile configuration has bank conflicts
    pub fn check_tile_conflicts(&self, tile_width: usize, element_size: usize) -> bool {
        let stride = (tile_width * element_size) as u32;
        self.analyzer.causes_full_conflict(stride)
    }

    /// Get recommended shared memory allocation for transpose
    pub fn shared_memory_size(&self, tile_dim: usize, element_size: usize) -> usize {
        let (width, height) = self.optimal_tile_size(element_size);
        width * height * element_size
    }
}

impl Default for TransposeHelper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_conflict() {
        let analyzer = BankConflictAnalyzer::new();

        // Unit stride (4 bytes) - each thread hits different bank
        let report = analyzer.analyze_linear(0, 4);

        assert_eq!(report.conflict_degree, 1);
        assert_eq!(report.efficiency, 1.0);
        assert!(matches!(
            report.pattern_type,
            ConflictPatternType::NoConflict
        ));
    }

    #[test]
    fn test_full_conflict() {
        let analyzer = BankConflictAnalyzer::new();

        // Stride of 128 bytes (32 banks × 4 bytes) - all threads hit same bank!
        let report = analyzer.analyze_linear(0, 128);

        assert_eq!(report.conflict_degree, 32);
        assert_eq!(report.efficiency, 1.0 / 32.0);
        assert!(matches!(
            report.pattern_type,
            ConflictPatternType::FullConflict
        ));
    }

    #[test]
    fn test_two_way_conflict() {
        let analyzer = BankConflictAnalyzer::new();

        // Stride of 8 bytes - 2-way conflict
        // Thread 0 → bank 0, Thread 1 → bank 2, ..., Thread 16 → bank 0
        let report = analyzer.analyze_linear(0, 8);

        assert_eq!(report.conflict_degree, 2);
        assert_eq!(report.efficiency, 0.5);
    }

    #[test]
    fn test_matrix_transpose_conflict() {
        let analyzer = BankConflictAnalyzer::new();

        // Column access in 32×32 float matrix - classic 32-way conflict
        // Stride = 32 * 4 = 128 bytes
        let report = analyzer.analyze_linear(0, 128);

        assert_eq!(report.conflict_degree, 32);
        assert!(report.suggestion.is_some());
    }

    #[test]
    fn test_padded_matrix() {
        let analyzer = BankConflictAnalyzer::new();

        // Column access in 33×32 float matrix (padded)
        // Stride = 33 * 4 = 132 bytes
        let report = analyzer.analyze_linear(0, 132);

        // 132 / 4 = 33, gcd(33, 32) = 1 → no conflicts!
        assert_eq!(report.conflict_degree, 1);
    }

    #[test]
    fn test_broadcast() {
        let analyzer = BankConflictAnalyzer::new();

        let pattern = SharedAccessPattern::Broadcast { addr: 0x100 };
        let report = analyzer.analyze_access(&pattern);

        assert_eq!(report.conflict_degree, 1);
        assert!(matches!(
            report.pattern_type,
            ConflictPatternType::Broadcast
        ));
    }

    #[test]
    fn test_indexed_access() {
        let analyzer = BankConflictAnalyzer::new();

        // All threads access same bank (worst case)
        let indices: Vec<u32> = (0..32).map(|i| i * 128).collect();
        let pattern = SharedAccessPattern::Indexed { base: 0, indices };
        let report = analyzer.analyze_access(&pattern);

        assert!(report.has_conflicts());
        assert!(report.is_severe());
    }

    #[test]
    fn test_padded_size() {
        let analyzer = BankConflictAnalyzer::new();

        // 32x32 matrix of floats
        let original = 32 * 32 * 4;
        let padded = analyzer.padded_size(32, 32, 4);

        // Should add one float per row
        assert_eq!(padded, 32 * 33 * 4);
        assert!(padded > original);
    }

    #[test]
    fn test_transpose_helper() {
        let helper = TransposeHelper::new();

        // Check that 32-wide tiles cause conflicts
        assert!(helper.check_tile_conflicts(32, 4));

        // Check that 33-wide tiles don't
        assert!(!helper.check_tile_conflicts(33, 4));

        // Optimal tile size
        let (width, padded) = helper.optimal_tile_size(4);
        assert_eq!(width, 32);
        assert_eq!(padded, 33);
    }

    #[test]
    fn test_address_to_bank() {
        let analyzer = BankConflictAnalyzer::new();

        // First 32 4-byte addresses should map to banks 0-31
        for i in 0..32 {
            assert_eq!(analyzer.address_to_bank(i * 4), i);
        }

        // Address 128 should wrap to bank 0
        assert_eq!(analyzer.address_to_bank(128), 0);
    }
}
