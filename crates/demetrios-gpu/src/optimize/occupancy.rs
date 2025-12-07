//! Occupancy Calculator
//!
//! Occupancy = Active Warps per SM / Maximum Warps per SM
//!
//! Higher occupancy means better latency hiding. The GPU can switch to
//! other warps when one is waiting for memory.
//!
//! Occupancy is limited by three resources:
//! 1. Registers per thread - More registers → fewer warps
//! 2. Shared memory per block - More shared → fewer blocks
//! 3. Threads per block - Fewer threads → fewer warps

/// GPU architecture limits
#[derive(Debug, Clone)]
pub struct ArchLimits {
    /// Architecture name
    pub name: &'static str,
    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),

    // SM limits
    /// Maximum threads per SM
    pub max_threads_per_sm: u32,
    /// Maximum warps per SM
    pub max_warps_per_sm: u32,
    /// Maximum blocks per SM
    pub max_blocks_per_sm: u32,
    /// Maximum shared memory per SM (bytes)
    pub max_shared_per_sm: usize,
    /// Total registers per SM
    pub registers_per_sm: u32,

    // Block limits
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block (bytes)
    pub max_shared_per_block: usize,
    /// Maximum registers per block
    pub max_registers_per_block: u32,

    // Thread limits
    /// Warp size
    pub warp_size: u32,
    /// Register allocation granularity
    pub register_granularity: u32,
    /// Shared memory allocation granularity (bytes)
    pub shared_granularity: usize,
}

impl ArchLimits {
    /// NVIDIA A100 (SM 8.0)
    pub fn a100() -> Self {
        Self {
            name: "A100",
            compute_capability: (8, 0),
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            max_shared_per_sm: 164 * 1024, // 164 KB configurable
            registers_per_sm: 65536,
            max_threads_per_block: 1024,
            max_shared_per_block: 99 * 1024, // 99 KB
            max_registers_per_block: 65536,
            warp_size: 32,
            register_granularity: 8, // Registers allocated in multiples of 8 per thread
            shared_granularity: 256,
        }
    }

    /// NVIDIA H100 (SM 9.0)
    pub fn h100() -> Self {
        Self {
            name: "H100",
            compute_capability: (9, 0),
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            max_shared_per_sm: 228 * 1024, // 228 KB configurable
            registers_per_sm: 65536,
            max_threads_per_block: 1024,
            max_shared_per_block: 227 * 1024, // 227 KB
            max_registers_per_block: 65536,
            warp_size: 32,
            register_granularity: 8, // Registers allocated in multiples of 8 per thread
            shared_granularity: 256,
        }
    }

    /// NVIDIA L4 (SM 8.9)
    pub fn l4() -> Self {
        Self {
            name: "L4",
            compute_capability: (8, 9),
            max_threads_per_sm: 1536,
            max_warps_per_sm: 48,
            max_blocks_per_sm: 24,
            max_shared_per_sm: 100 * 1024,
            registers_per_sm: 65536,
            max_threads_per_block: 1024,
            max_shared_per_block: 99 * 1024,
            max_registers_per_block: 65536,
            warp_size: 32,
            register_granularity: 8, // Registers allocated in multiples of 8 per thread
            shared_granularity: 256,
        }
    }

    /// NVIDIA RTX 4090 (SM 8.9)
    pub fn rtx4090() -> Self {
        Self {
            name: "RTX 4090",
            ..Self::l4()
        }
    }

    /// NVIDIA V100 (SM 7.0)
    pub fn v100() -> Self {
        Self {
            name: "V100",
            compute_capability: (7, 0),
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            max_shared_per_sm: 96 * 1024,
            registers_per_sm: 65536,
            max_threads_per_block: 1024,
            max_shared_per_block: 96 * 1024,
            max_registers_per_block: 65536,
            warp_size: 32,
            register_granularity: 8, // Registers allocated in multiples of 8 per thread
            shared_granularity: 256,
        }
    }

    /// Generic SM 7.x (Volta/Turing)
    pub fn sm_7x() -> Self {
        Self::v100()
    }

    /// Generic SM 8.x (Ampere/Ada)
    pub fn sm_8x() -> Self {
        Self::a100()
    }

    /// Generic SM 9.x (Hopper)
    pub fn sm_9x() -> Self {
        Self::h100()
    }
}

/// Kernel resource requirements
#[derive(Debug, Clone)]
pub struct KernelResources {
    /// Threads per block
    pub threads_per_block: u32,
    /// Registers per thread
    pub registers_per_thread: u32,
    /// Shared memory per block (bytes)
    pub shared_per_block: usize,
}

impl KernelResources {
    /// Create new kernel resources
    pub fn new(threads: u32, registers: u32, shared: usize) -> Self {
        Self {
            threads_per_block: threads,
            registers_per_thread: registers,
            shared_per_block: shared,
        }
    }

    /// Create with just thread count (minimal resources)
    pub fn minimal(threads: u32) -> Self {
        Self {
            threads_per_block: threads,
            registers_per_thread: 32,
            shared_per_block: 0,
        }
    }
}

/// What's limiting occupancy?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitingFactor {
    /// Register usage is the bottleneck
    Registers,
    /// Shared memory usage is the bottleneck
    SharedMemory,
    /// Block size is too small
    BlockSize,
    /// Maximum blocks per SM limit reached
    MaxBlocksPerSm,
    /// Already at maximum occupancy
    Optimal,
}

impl std::fmt::Display for LimitingFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LimitingFactor::Registers => write!(f, "register usage"),
            LimitingFactor::SharedMemory => write!(f, "shared memory usage"),
            LimitingFactor::BlockSize => write!(f, "block size"),
            LimitingFactor::MaxBlocksPerSm => write!(f, "max blocks per SM"),
            LimitingFactor::Optimal => write!(f, "optimal"),
        }
    }
}

/// Suggestion for improving occupancy
#[derive(Debug, Clone)]
pub enum OccupancySuggestion {
    /// Reduce register usage
    ReduceRegisters {
        current: u32,
        target: u32,
        occupancy_gain: f32,
    },
    /// Reduce shared memory usage
    ReduceShared {
        current: usize,
        target: usize,
        occupancy_gain: f32,
    },
    /// Increase block size
    IncreaseBlockSize { current: u32, target: u32 },
    /// Decrease block size
    DecreaseBlockSize { current: u32, target: u32 },
}

impl std::fmt::Display for OccupancySuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OccupancySuggestion::ReduceRegisters {
                current,
                target,
                occupancy_gain,
            } => {
                write!(
                    f,
                    "Reduce registers from {} to {} for {:.0}% occupancy gain",
                    current,
                    target,
                    occupancy_gain * 100.0
                )
            }
            OccupancySuggestion::ReduceShared {
                current,
                target,
                occupancy_gain,
            } => {
                write!(
                    f,
                    "Reduce shared memory from {} to {} bytes for {:.0}% occupancy gain",
                    current,
                    target,
                    occupancy_gain * 100.0
                )
            }
            OccupancySuggestion::IncreaseBlockSize { current, target } => {
                write!(f, "Increase block size from {} to {}", current, target)
            }
            OccupancySuggestion::DecreaseBlockSize { current, target } => {
                write!(f, "Decrease block size from {} to {}", current, target)
            }
        }
    }
}

/// Occupancy analysis result
#[derive(Debug, Clone)]
pub struct OccupancyReport {
    /// Achieved occupancy (0.0 - 1.0)
    pub occupancy: f32,

    /// Active warps per SM
    pub active_warps: u32,

    /// Active blocks per SM
    pub active_blocks: u32,

    /// What limits occupancy?
    pub limiting_factor: LimitingFactor,

    /// Maximum theoretical warps
    pub theoretical_warps: u32,
    /// Warps limited by registers
    pub register_limited_warps: u32,
    /// Warps limited by shared memory
    pub shared_limited_warps: u32,
    /// Warps limited by block count
    pub block_limited_warps: u32,

    /// Suggestions for improvement
    pub suggestions: Vec<OccupancySuggestion>,
}

impl OccupancyReport {
    /// Check if occupancy is acceptable (> 50%)
    pub fn is_acceptable(&self) -> bool {
        self.occupancy >= 0.5
    }

    /// Check if occupancy is good (> 75%)
    pub fn is_good(&self) -> bool {
        self.occupancy >= 0.75
    }

    /// Get occupancy as percentage
    pub fn percentage(&self) -> f32 {
        self.occupancy * 100.0
    }
}

impl std::fmt::Display for OccupancyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Occupancy: {:.1}%", self.percentage())?;
        writeln!(
            f,
            "Active warps: {} / {}",
            self.active_warps, self.theoretical_warps
        )?;
        writeln!(f, "Active blocks: {}", self.active_blocks)?;
        writeln!(f, "Limited by: {}", self.limiting_factor)?;
        if !self.suggestions.is_empty() {
            writeln!(f, "Suggestions:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "  - {}", suggestion)?;
            }
        }
        Ok(())
    }
}

/// Occupancy calculator
pub struct OccupancyCalculator {
    arch: ArchLimits,
}

impl OccupancyCalculator {
    /// Create a new occupancy calculator for the given architecture
    pub fn new(arch: ArchLimits) -> Self {
        Self { arch }
    }

    /// Create calculator for A100
    pub fn for_a100() -> Self {
        Self::new(ArchLimits::a100())
    }

    /// Create calculator for H100
    pub fn for_h100() -> Self {
        Self::new(ArchLimits::h100())
    }

    /// Create calculator for L4
    pub fn for_l4() -> Self {
        Self::new(ArchLimits::l4())
    }

    /// Get the architecture limits
    pub fn arch(&self) -> &ArchLimits {
        &self.arch
    }

    /// Calculate occupancy for given kernel resources
    pub fn calculate(&self, resources: &KernelResources) -> OccupancyReport {
        let warps_per_block =
            (resources.threads_per_block + self.arch.warp_size - 1) / self.arch.warp_size;

        // Calculate blocks per SM limited by each resource
        let blocks_by_warps = self.arch.max_warps_per_sm / warps_per_block;
        let blocks_by_threads = self.arch.max_threads_per_sm / resources.threads_per_block;
        let blocks_by_registers = self.blocks_limited_by_registers(resources);
        let blocks_by_shared = self.blocks_limited_by_shared(resources);
        let blocks_by_limit = self.arch.max_blocks_per_sm;

        // Take minimum
        let active_blocks = blocks_by_warps
            .min(blocks_by_threads)
            .min(blocks_by_registers)
            .min(blocks_by_shared)
            .min(blocks_by_limit);

        let active_warps = active_blocks * warps_per_block;
        let occupancy = active_warps as f32 / self.arch.max_warps_per_sm as f32;

        // Determine limiting factor
        let limiting_factor = if active_blocks == blocks_by_limit && active_blocks < blocks_by_warps
        {
            LimitingFactor::MaxBlocksPerSm
        } else if active_blocks == blocks_by_registers
            && blocks_by_registers < blocks_by_warps.min(blocks_by_shared)
        {
            LimitingFactor::Registers
        } else if active_blocks == blocks_by_shared
            && blocks_by_shared < blocks_by_warps.min(blocks_by_registers)
        {
            LimitingFactor::SharedMemory
        } else if active_blocks == blocks_by_warps {
            if occupancy >= 0.99 {
                LimitingFactor::Optimal
            } else {
                LimitingFactor::BlockSize
            }
        } else {
            LimitingFactor::Optimal
        };

        let suggestions = self.generate_suggestions(resources, &limiting_factor, occupancy);

        OccupancyReport {
            occupancy,
            active_warps,
            active_blocks,
            limiting_factor,
            theoretical_warps: self.arch.max_warps_per_sm,
            register_limited_warps: blocks_by_registers * warps_per_block,
            shared_limited_warps: blocks_by_shared * warps_per_block,
            block_limited_warps: blocks_by_warps * warps_per_block,
            suggestions,
        }
    }

    /// Calculate blocks limited by registers
    fn blocks_limited_by_registers(&self, resources: &KernelResources) -> u32 {
        if resources.registers_per_thread == 0 {
            return self.arch.max_blocks_per_sm;
        }

        // Registers are allocated at warp granularity, rounded up
        let regs_per_thread = ((resources.registers_per_thread as usize
            + self.arch.register_granularity as usize
            - 1)
            / self.arch.register_granularity as usize)
            * self.arch.register_granularity as usize;

        let regs_per_block = regs_per_thread * resources.threads_per_block as usize;

        if regs_per_block == 0 {
            self.arch.max_blocks_per_sm
        } else {
            (self.arch.registers_per_sm as usize / regs_per_block) as u32
        }
    }

    /// Calculate blocks limited by shared memory
    fn blocks_limited_by_shared(&self, resources: &KernelResources) -> u32 {
        if resources.shared_per_block == 0 {
            return self.arch.max_blocks_per_sm;
        }

        // Round up to shared memory granularity
        let shared_per_block = ((resources.shared_per_block + self.arch.shared_granularity - 1)
            / self.arch.shared_granularity)
            * self.arch.shared_granularity;

        (self.arch.max_shared_per_sm / shared_per_block) as u32
    }

    /// Generate improvement suggestions
    fn generate_suggestions(
        &self,
        resources: &KernelResources,
        limiting_factor: &LimitingFactor,
        current_occupancy: f32,
    ) -> Vec<OccupancySuggestion> {
        let mut suggestions = Vec::new();

        match limiting_factor {
            LimitingFactor::Registers => {
                // Find register count for next occupancy level
                for target_occupancy in [0.5, 0.625, 0.75, 0.875, 1.0] {
                    if target_occupancy <= current_occupancy {
                        continue;
                    }

                    let target_warps =
                        (target_occupancy * self.arch.max_warps_per_sm as f32) as u32;
                    let warps_per_block = (resources.threads_per_block + self.arch.warp_size - 1)
                        / self.arch.warp_size;
                    let target_blocks = (target_warps + warps_per_block - 1) / warps_per_block;

                    if target_blocks == 0 {
                        continue;
                    }

                    let max_regs =
                        self.arch.registers_per_sm / (target_blocks * resources.threads_per_block);

                    if max_regs < resources.registers_per_thread {
                        suggestions.push(OccupancySuggestion::ReduceRegisters {
                            current: resources.registers_per_thread,
                            target: max_regs,
                            occupancy_gain: target_occupancy - current_occupancy,
                        });
                        break;
                    }
                }
            }

            LimitingFactor::SharedMemory => {
                let target_occupancy = (current_occupancy + 0.125).min(1.0);
                let target_warps = (target_occupancy * self.arch.max_warps_per_sm as f32) as u32;
                let warps_per_block =
                    (resources.threads_per_block + self.arch.warp_size - 1) / self.arch.warp_size;
                let target_blocks = (target_warps + warps_per_block - 1) / warps_per_block;

                if target_blocks > 0 {
                    let max_shared = self.arch.max_shared_per_sm / target_blocks as usize;

                    if max_shared < resources.shared_per_block {
                        suggestions.push(OccupancySuggestion::ReduceShared {
                            current: resources.shared_per_block,
                            target: max_shared,
                            occupancy_gain: target_occupancy - current_occupancy,
                        });
                    }
                }
            }

            LimitingFactor::BlockSize => {
                // Suggest larger block size
                let next_size = ((resources.threads_per_block + 32) / 32) * 32;
                if next_size <= self.arch.max_threads_per_block {
                    suggestions.push(OccupancySuggestion::IncreaseBlockSize {
                        current: resources.threads_per_block,
                        target: next_size.min(self.arch.max_threads_per_block),
                    });
                }
            }

            LimitingFactor::MaxBlocksPerSm => {
                // Suggest smaller block size to fit more blocks
                let half_size = resources.threads_per_block / 2;
                if half_size >= 32 {
                    suggestions.push(OccupancySuggestion::DecreaseBlockSize {
                        current: resources.threads_per_block,
                        target: half_size,
                    });
                }
            }

            LimitingFactor::Optimal => {
                // Already optimal, no suggestions
            }
        }

        suggestions
    }

    /// Find optimal block size for given kernel
    pub fn optimal_block_size(&self, registers_per_thread: u32, shared_per_thread: usize) -> u32 {
        let mut best_occupancy = 0.0;
        let mut best_block_size = 128;

        for block_size in (32..=self.arch.max_threads_per_block).step_by(32) {
            let resources = KernelResources {
                threads_per_block: block_size,
                registers_per_thread,
                shared_per_block: shared_per_thread * block_size as usize,
            };

            let report = self.calculate(&resources);

            if report.occupancy > best_occupancy {
                best_occupancy = report.occupancy;
                best_block_size = block_size;
            }
        }

        best_block_size
    }

    /// Calculate minimum block size for target occupancy
    pub fn min_block_size_for_occupancy(&self, target_occupancy: f32) -> u32 {
        let target_warps = (target_occupancy * self.arch.max_warps_per_sm as f32) as u32;

        // Minimum block size to achieve target warps
        // blocks = warps / warps_per_block
        // For max blocks = max_blocks_per_sm, we need:
        // warps_per_block >= target_warps / max_blocks_per_sm

        let min_warps_per_block =
            (target_warps + self.arch.max_blocks_per_sm - 1) / self.arch.max_blocks_per_sm;
        min_warps_per_block * self.arch.warp_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_occupancy() {
        let calc = OccupancyCalculator::new(ArchLimits::a100());

        let resources = KernelResources {
            threads_per_block: 256,
            registers_per_thread: 32,
            shared_per_block: 0,
        };

        let report = calc.calculate(&resources);

        // 256 threads = 8 warps per block
        // With low register pressure, should achieve good occupancy
        assert!(report.active_warps > 0);
        assert!(report.occupancy > 0.0);
    }

    #[test]
    fn test_register_limited() {
        let calc = OccupancyCalculator::new(ArchLimits::a100());

        let resources = KernelResources {
            threads_per_block: 256,
            registers_per_thread: 128, // High register usage
            shared_per_block: 0,
        };

        let report = calc.calculate(&resources);

        // High register usage should limit occupancy
        assert!(report.occupancy < 1.0);
    }

    #[test]
    fn test_shared_limited() {
        let calc = OccupancyCalculator::new(ArchLimits::a100());

        let resources = KernelResources {
            threads_per_block: 256,
            registers_per_thread: 32,
            shared_per_block: 80 * 1024, // 80 KB shared - very high
        };

        let report = calc.calculate(&resources);

        // High shared memory usage should limit blocks
        assert!(report.active_blocks <= 2);
    }

    #[test]
    fn test_optimal_block_size() {
        let calc = OccupancyCalculator::new(ArchLimits::a100());

        // Low register pressure
        let optimal = calc.optimal_block_size(32, 0);

        // Should suggest a valid block size
        assert!(optimal >= 32);
        assert!(optimal <= 1024);
        assert_eq!(optimal % 32, 0);
    }

    #[test]
    fn test_different_architectures() {
        let a100 = OccupancyCalculator::new(ArchLimits::a100());
        let h100 = OccupancyCalculator::new(ArchLimits::h100());

        let resources = KernelResources::minimal(256);

        let a100_report = a100.calculate(&resources);
        let h100_report = h100.calculate(&resources);

        // Both should achieve reasonable occupancy with minimal resources
        assert!(a100_report.occupancy > 0.0);
        assert!(h100_report.occupancy > 0.0);
    }

    #[test]
    fn test_suggestions() {
        let calc = OccupancyCalculator::new(ArchLimits::a100());

        let resources = KernelResources {
            threads_per_block: 256,
            registers_per_thread: 128,
            shared_per_block: 0,
        };

        let report = calc.calculate(&resources);

        // Should have suggestions for improving occupancy
        assert!(!report.suggestions.is_empty());
    }

    #[test]
    fn test_min_block_size() {
        let calc = OccupancyCalculator::new(ArchLimits::a100());

        // For 50% occupancy on A100 (32 warps)
        let min_size = calc.min_block_size_for_occupancy(0.5);

        // Should be at least 32 (one warp)
        assert!(min_size >= 32);
        assert_eq!(min_size % 32, 0);
    }
}
