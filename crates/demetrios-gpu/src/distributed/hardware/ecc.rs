//! Error Correcting Codes for GPU Memory
//!
//! ECC is critical for HPC reliability. This module models:
//! - SECDED (Single Error Correct, Double Error Detect) codes
//! - Error injection and detection
//! - Performance overhead modeling
//! - Error handling strategies
//!
//! At HPC scale (1000+ GPUs), bit flips happen daily due to:
//! - Cosmic rays and alpha particles
//! - Voltage noise
//! - Retention failures
//!
//! ECC adds ~12.5% storage overhead (72 bits for 64 bits data)
//! but is essential for correctness in scientific computing.

use std::collections::{HashMap, VecDeque};

// ============================================================================
// ECC Fundamentals
// ============================================================================

/// Hamming code for SECDED (Single Error Correct, Double Error Detect)
///
/// For 64-bit data, uses 8 parity bits (7 + 1 overall) = 72 bits total
/// - Can correct any single-bit error
/// - Can detect (but not correct) any double-bit error
#[derive(Debug, Clone)]
pub struct HammingCode {
    /// Data bits
    pub data_bits: u32,
    /// Parity bits (excluding overall parity)
    pub parity_bits: u32,
    /// Total codeword bits
    pub codeword_bits: u32,
    /// Parity check matrix positions
    parity_positions: Vec<u32>,
}

impl HammingCode {
    /// Standard (64, 72) SECDED code for GPU memory
    pub fn secded_64() -> Self {
        // For 64 data bits, we need 7 parity bits (2^7 - 1 = 127 > 64 + 7)
        // Plus 1 overall parity bit for double error detection
        Self {
            data_bits: 64,
            parity_bits: 8, // 7 + 1 overall
            codeword_bits: 72,
            parity_positions: vec![1, 2, 4, 8, 16, 32, 64],
        }
    }

    /// Encode data to codeword
    pub fn encode(&self, data: u64) -> u128 {
        let mut codeword = 0u128;
        let mut data_idx = 0u32;

        // Place data bits (skip power-of-2 positions for parity)
        // Position 0 is unused, positions 1,2,4,8,16,32,64 are parity, position 72 is overall parity
        for pos in 1..self.codeword_bits {
            if !self.parity_positions.contains(&pos) {
                if data_idx < self.data_bits {
                    if (data >> data_idx) & 1 == 1 {
                        codeword |= 1u128 << (pos - 1);
                    }
                    data_idx += 1;
                }
            }
        }

        // Calculate parity bits
        for &parity_pos in &self.parity_positions {
            let mut parity = 0u32;
            for pos in 1..self.codeword_bits {
                if pos & parity_pos != 0 {
                    if (codeword >> (pos - 1)) & 1 == 1 {
                        parity ^= 1;
                    }
                }
            }
            if parity == 1 {
                codeword |= 1u128 << (parity_pos - 1);
            }
        }

        // Overall parity (bit 72) - count all 71 bits
        let mut total_ones = 0u32;
        for i in 0..(self.codeword_bits - 1) {
            if (codeword >> i) & 1 == 1 {
                total_ones += 1;
            }
        }
        if total_ones % 2 == 1 {
            codeword |= 1u128 << (self.codeword_bits - 1);
        }

        codeword
    }

    /// Decode codeword, detecting and correcting errors
    pub fn decode(&self, codeword: u128) -> EccResult {
        // Calculate syndrome
        let mut syndrome = 0u32;
        for (i, &parity_pos) in self.parity_positions.iter().enumerate() {
            let mut parity = 0u32;
            for pos in 1..self.codeword_bits {
                if pos & parity_pos != 0 {
                    if (codeword >> (pos - 1)) & 1 == 1 {
                        parity ^= 1;
                    }
                }
            }
            if parity == 1 {
                syndrome |= 1 << i;
            }
        }

        // Check overall parity
        let mut total_ones = 0u32;
        for i in 0..self.codeword_bits {
            if (codeword >> i) & 1 == 1 {
                total_ones += 1;
            }
        }
        let overall_parity_error = total_ones % 2 == 1;

        if syndrome == 0 && !overall_parity_error {
            // No error
            EccResult::NoError {
                data: self.extract_data(codeword),
            }
        } else if syndrome != 0 && overall_parity_error {
            // Single-bit error (correctable)
            let error_pos = syndrome as usize;
            if error_pos > 0 && error_pos <= self.codeword_bits as usize {
                let corrected = codeword ^ (1u128 << (error_pos - 1));
                EccResult::SingleBitError {
                    data: self.extract_data(corrected),
                    error_position: error_pos,
                }
            } else {
                EccResult::DoubleBitError { syndrome }
            }
        } else if syndrome != 0 && !overall_parity_error {
            // Double-bit error (detected but not correctable)
            EccResult::DoubleBitError { syndrome }
        } else {
            // Error in overall parity bit only
            EccResult::ParityBitError {
                data: self.extract_data(codeword),
            }
        }
    }

    /// Extract data bits from codeword
    fn extract_data(&self, codeword: u128) -> u64 {
        let mut data = 0u64;
        let mut data_idx = 0u32;

        for pos in 1..self.codeword_bits {
            if !self.parity_positions.contains(&pos) {
                if (codeword >> (pos - 1)) & 1 == 1 {
                    data |= 1u64 << data_idx;
                }
                data_idx += 1;
                if data_idx >= self.data_bits {
                    break;
                }
            }
        }

        data
    }

    /// Bandwidth overhead ratio
    pub fn bandwidth_overhead(&self) -> f64 {
        self.codeword_bits as f64 / self.data_bits as f64
    }
}

/// ECC decode result
#[derive(Debug, Clone)]
pub enum EccResult {
    /// No error detected
    NoError { data: u64 },
    /// Single-bit error corrected
    SingleBitError { data: u64, error_position: usize },
    /// Double-bit error detected (uncorrectable)
    DoubleBitError { syndrome: u32 },
    /// Error only in parity bit
    ParityBitError { data: u64 },
}

// ============================================================================
// Memory Error Modeling
// ============================================================================

/// Memory error types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryErrorType {
    /// Single-bit upset (soft error, correctable)
    SingleBitUpset,
    /// Multi-bit upset in same word
    MultiBitUpset,
    /// Stuck-at fault (hard error)
    StuckAt { bit: u32, value: bool },
    /// Row hammer induced error
    RowHammer,
    /// Retention failure
    RetentionFailure,
}

/// Error rate model based on FIT (Failures In Time)
#[derive(Debug, Clone)]
pub struct ErrorRateModel {
    /// Base FIT rate (Failures In Time per billion device-hours)
    pub fit_rate: f64,
    /// Temperature coefficient (errors increase with temp)
    pub temp_coefficient: f64,
    /// Altitude coefficient (cosmic rays increase at altitude)
    pub altitude_coefficient: f64,
    /// Technology node (nm)
    pub tech_node_nm: u32,
}

impl ErrorRateModel {
    /// HBM2e error model (A100)
    pub fn hbm2e() -> Self {
        Self {
            fit_rate: 1000.0, // ~1000 FIT per Gbit
            temp_coefficient: 1.1,
            altitude_coefficient: 1.0,
            tech_node_nm: 16,
        }
    }

    /// HBM3 error model (H100)
    pub fn hbm3() -> Self {
        Self {
            fit_rate: 1500.0, // Higher density = more susceptible
            temp_coefficient: 1.15,
            altitude_coefficient: 1.0,
            tech_node_nm: 5,
        }
    }

    /// GDDR6 error model
    pub fn gddr6() -> Self {
        Self {
            fit_rate: 800.0,
            temp_coefficient: 1.08,
            altitude_coefficient: 1.0,
            tech_node_nm: 10,
        }
    }

    /// Calculate error probability per bit per hour
    pub fn error_probability_per_bit_hour(&self, temp_c: f64, altitude_m: f64) -> f64 {
        let base_rate = self.fit_rate / 1e9; // FIT to per-hour

        // Temperature adjustment (reference: 55°C)
        let temp_factor = self.temp_coefficient.powf((temp_c - 55.0) / 10.0);

        // Altitude adjustment (cosmic ray flux doubles every ~1500m)
        let altitude_factor = 2.0_f64.powf(altitude_m / 1500.0) * self.altitude_coefficient;

        base_rate * temp_factor * altitude_factor
    }

    /// Expected errors per GPU per day
    pub fn expected_errors_per_day(&self, memory_gb: u64, temp_c: f64, altitude_m: f64) -> f64 {
        let bits = memory_gb * 8 * 1_000_000_000;
        let prob_per_bit_hour = self.error_probability_per_bit_hour(temp_c, altitude_m);
        bits as f64 * prob_per_bit_hour * 24.0
    }

    /// Expected errors across cluster per day
    pub fn cluster_errors_per_day(
        &self,
        num_gpus: u64,
        memory_per_gpu_gb: u64,
        temp_c: f64,
        altitude_m: f64,
    ) -> f64 {
        self.expected_errors_per_day(memory_per_gpu_gb, temp_c, altitude_m) * num_gpus as f64
    }
}

// ============================================================================
// ECC Memory Controller
// ============================================================================

/// ECC-enabled memory controller
#[derive(Debug)]
pub struct EccMemoryController {
    /// ECC codec
    codec: HammingCode,
    /// Error statistics
    pub stats: EccStats,
    /// Memory scrubber
    scrubber: MemoryScrubber,
    /// Error log
    error_log: VecDeque<MemoryError>,
    /// Configuration
    config: EccConfig,
    /// Maximum error log size
    max_log_size: usize,
}

/// ECC statistics
#[derive(Debug, Default, Clone)]
pub struct EccStats {
    /// Total reads
    pub reads: u64,
    /// Total writes
    pub writes: u64,
    /// Corrected single-bit errors
    pub corrected_errors: u64,
    /// Detected uncorrectable errors (UE)
    pub uncorrectable_errors: u64,
    /// Parity errors
    pub parity_errors: u64,
    /// Scrub corrections
    pub scrub_corrections: u64,
}

impl EccStats {
    /// Correctable error rate
    pub fn correctable_error_rate(&self) -> f64 {
        if self.reads == 0 {
            0.0
        } else {
            self.corrected_errors as f64 / self.reads as f64
        }
    }

    /// Uncorrectable error rate
    pub fn uncorrectable_error_rate(&self) -> f64 {
        if self.reads == 0 {
            0.0
        } else {
            self.uncorrectable_errors as f64 / self.reads as f64
        }
    }
}

/// ECC configuration
#[derive(Debug, Clone)]
pub struct EccConfig {
    /// Enable ECC
    pub enabled: bool,
    /// Scrub interval (hours)
    pub scrub_interval_hours: f64,
    /// Poison on UE (mark page as bad)
    pub poison_on_ue: bool,
    /// Report threshold (log after N errors)
    pub report_threshold: u64,
    /// Retire row on repeated errors
    pub row_retirement_threshold: u32,
}

impl Default for EccConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scrub_interval_hours: 24.0,
            poison_on_ue: true,
            report_threshold: 1,
            row_retirement_threshold: 10,
        }
    }
}

/// Memory error record
#[derive(Debug, Clone)]
pub struct MemoryError {
    /// Address where error occurred
    pub address: u64,
    /// Error type
    pub error_type: MemoryErrorType,
    /// Timestamp (cycles)
    pub timestamp: u64,
    /// Was error corrected?
    pub corrected: bool,
}

/// Memory scrubber (background error detection/correction)
#[derive(Debug)]
pub struct MemoryScrubber {
    /// Current scrub position
    current_address: u64,
    /// Total memory size
    memory_size: u64,
    /// Last scrub completion cycle
    last_complete_cycle: u64,
    /// Scrub rate (bytes per cycle)
    rate_bytes_per_cycle: f64,
    /// Enabled
    enabled: bool,
}

impl MemoryScrubber {
    pub fn new(memory_size: u64, rate_mbps: f64) -> Self {
        // Convert Mbps to bytes per cycle (assuming 1.5 GHz)
        let rate_bytes_per_cycle = rate_mbps * 1e6 / 8.0 / 1.5e9;

        Self {
            current_address: 0,
            memory_size,
            last_complete_cycle: 0,
            rate_bytes_per_cycle: rate_bytes_per_cycle.max(0.001),
            enabled: true,
        }
    }

    /// Full scrub cycles needed
    pub fn full_scrub_cycles(&self) -> u64 {
        (self.memory_size as f64 / self.rate_bytes_per_cycle) as u64
    }

    /// Advance scrubber
    pub fn advance(&mut self, cycles: u64, current_cycle: u64) -> Vec<u64> {
        if !self.enabled {
            return Vec::new();
        }

        let bytes_to_scrub = (cycles as f64 * self.rate_bytes_per_cycle) as u64;
        let mut addresses = Vec::new();

        let start = self.current_address;
        let end = (start + bytes_to_scrub).min(self.memory_size);

        // Return cache-line aligned addresses
        for addr in (start..end).step_by(128) {
            addresses.push(addr);
        }

        self.current_address = if end >= self.memory_size {
            self.last_complete_cycle = current_cycle;
            0
        } else {
            end
        };

        addresses
    }

    /// Progress through current scrub pass
    pub fn progress(&self) -> f64 {
        if self.memory_size == 0 {
            0.0
        } else {
            self.current_address as f64 / self.memory_size as f64
        }
    }
}

impl EccMemoryController {
    pub fn new(memory_size_gb: u64, config: EccConfig) -> Self {
        let memory_size = memory_size_gb * 1024 * 1024 * 1024;

        Self {
            codec: HammingCode::secded_64(),
            stats: EccStats::default(),
            scrubber: MemoryScrubber::new(memory_size, 100.0),
            error_log: VecDeque::new(),
            config,
            max_log_size: 1000,
        }
    }

    /// Read with ECC check
    pub fn read(&mut self, address: u64, codeword: u128, cycle: u64) -> Result<u64, MemoryError> {
        self.stats.reads += 1;

        if !self.config.enabled {
            return Ok(self.codec.extract_data(codeword));
        }

        match self.codec.decode(codeword) {
            EccResult::NoError { data } => Ok(data),

            EccResult::SingleBitError {
                data,
                error_position,
            } => {
                self.stats.corrected_errors += 1;

                let error = MemoryError {
                    address,
                    error_type: MemoryErrorType::SingleBitUpset,
                    timestamp: cycle,
                    corrected: true,
                };

                self.log_error(error);
                Ok(data)
            }

            EccResult::DoubleBitError { .. } => {
                self.stats.uncorrectable_errors += 1;

                let error = MemoryError {
                    address,
                    error_type: MemoryErrorType::MultiBitUpset,
                    timestamp: cycle,
                    corrected: false,
                };

                self.log_error(error.clone());
                Err(error)
            }

            EccResult::ParityBitError { data } => {
                self.stats.parity_errors += 1;
                Ok(data)
            }
        }
    }

    /// Write with ECC encoding
    pub fn write(&mut self, data: u64) -> u128 {
        self.stats.writes += 1;

        if !self.config.enabled {
            return data as u128;
        }

        self.codec.encode(data)
    }

    fn log_error(&mut self, error: MemoryError) {
        if self.error_log.len() >= self.max_log_size {
            self.error_log.pop_front();
        }
        self.error_log.push_back(error);
    }

    /// Bandwidth overhead due to ECC
    pub fn bandwidth_overhead(&self) -> f64 {
        if self.config.enabled {
            self.codec.bandwidth_overhead() - 1.0
        } else {
            0.0
        }
    }

    /// Get error log
    pub fn error_log(&self) -> &VecDeque<MemoryError> {
        &self.error_log
    }

    /// Run scrubber for cycles
    pub fn scrub_advance(&mut self, cycles: u64, current_cycle: u64) -> Vec<u64> {
        self.scrubber.advance(cycles, current_cycle)
    }
}

// ============================================================================
// Chipkill (Advanced ECC)
// ============================================================================

/// Chipkill: Survives complete DRAM chip failure
///
/// Standard ECC can't handle chip failure (8+ bits affected).
/// Chipkill spreads each codeword across multiple chips using
/// symbol-based Reed-Solomon codes instead of bit-based Hamming.
#[derive(Debug, Clone)]
pub struct ChipkillCode {
    /// Symbols per codeword
    pub symbols: u32,
    /// Symbol width (bits)
    pub symbol_width: u32,
    /// Data symbols
    pub data_symbols: u32,
    /// Check symbols
    pub check_symbols: u32,
    /// Can correct this many symbol errors
    pub correction_capability: u32,
}

impl ChipkillCode {
    /// AMD Chipkill (SECDED + symbol correction)
    pub fn amd_chipkill() -> Self {
        Self {
            symbols: 18,
            symbol_width: 4,
            data_symbols: 16,
            check_symbols: 2,
            correction_capability: 1,
        }
    }

    /// Intel SDDC (Single Device Data Correction)
    pub fn intel_sddc() -> Self {
        Self {
            symbols: 9,
            symbol_width: 8,
            data_symbols: 8,
            check_symbols: 1,
            correction_capability: 1,
        }
    }

    /// HBM pseudo-channel redundancy
    pub fn hbm_redundancy() -> Self {
        Self {
            symbols: 8,
            symbol_width: 16,
            data_symbols: 7,
            check_symbols: 1,
            correction_capability: 1,
        }
    }

    /// Can survive single chip failure?
    pub fn survives_chip_failure(&self) -> bool {
        self.correction_capability >= 1
    }

    /// Bandwidth overhead
    pub fn bandwidth_overhead(&self) -> f64 {
        self.check_symbols as f64 / self.data_symbols as f64
    }

    /// Total bits per codeword
    pub fn codeword_bits(&self) -> u32 {
        self.symbols * self.symbol_width
    }
}

// ============================================================================
// Error Injection for Testing
// ============================================================================

/// Error injector for resilience testing
#[derive(Debug)]
pub struct ErrorInjector {
    /// Injection rate (probability per access)
    pub rate: f64,
    /// Error type weights
    pub single_bit_weight: f64,
    pub multi_bit_weight: f64,
    /// Seed for reproducibility
    seed: u64,
    /// Injected errors count
    injected: u64,
}

impl ErrorInjector {
    pub fn new(rate: f64, seed: u64) -> Self {
        Self {
            rate,
            single_bit_weight: 0.9,
            multi_bit_weight: 0.1,
            seed,
            injected: 0,
        }
    }

    /// Simple PRNG
    fn next_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Inject error into codeword
    pub fn inject(&mut self, codeword: u128) -> (u128, Option<MemoryErrorType>) {
        if self.next_rand() > self.rate {
            return (codeword, None);
        }

        self.injected += 1;

        let type_rand = self.next_rand();
        let error_type = if type_rand < self.single_bit_weight {
            MemoryErrorType::SingleBitUpset
        } else {
            MemoryErrorType::MultiBitUpset
        };

        let corrupted = match error_type {
            MemoryErrorType::SingleBitUpset => {
                let bit = (self.seed & 0x7F) as u32;
                codeword ^ (1u128 << bit.min(71))
            }
            MemoryErrorType::MultiBitUpset => {
                let bit1 = (self.seed & 0x7F) as u32;
                let bit2 = ((self.seed >> 7) & 0x7F) as u32;
                codeword ^ (1u128 << bit1.min(71)) ^ (1u128 << bit2.min(71))
            }
            _ => codeword ^ 1,
        };

        (corrupted, Some(error_type))
    }

    pub fn injected_count(&self) -> u64 {
        self.injected
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_encode_decode_no_error() {
        let codec = HammingCode::secded_64();

        let data = 0xDEADBEEFCAFEBABE_u64;
        let codeword = codec.encode(data);

        match codec.decode(codeword) {
            EccResult::NoError { data: decoded } => {
                assert_eq!(decoded, data);
            }
            _ => panic!("Expected no error"),
        }
    }

    #[test]
    fn test_hamming_various_values() {
        let codec = HammingCode::secded_64();

        let test_values = [0u64, 1, 0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0, 42];

        for &data in &test_values {
            let codeword = codec.encode(data);
            match codec.decode(codeword) {
                EccResult::NoError { data: decoded } => {
                    assert_eq!(decoded, data, "Failed for value {:#x}", data);
                }
                _ => panic!("Expected no error for value {:#x}", data),
            }
        }
    }

    #[test]
    fn test_single_bit_correction() {
        let codec = HammingCode::secded_64();

        let data = 0x123456789ABCDEF0_u64;
        let codeword = codec.encode(data);

        // Try corrupting different bit positions
        for bit in [5, 10, 20, 30, 40, 50] {
            let corrupted = codeword ^ (1u128 << bit);

            match codec.decode(corrupted) {
                EccResult::SingleBitError {
                    data: corrected, ..
                } => {
                    assert_eq!(corrected, data, "Failed to correct bit {}", bit);
                }
                other => panic!("Expected single-bit error for bit {}, got {:?}", bit, other),
            }
        }
    }

    #[test]
    fn test_double_bit_detection() {
        let codec = HammingCode::secded_64();

        let data = 0xAAAAAAAAAAAAAAAA_u64;
        let codeword = codec.encode(data);

        // Inject double-bit error
        let corrupted = codeword ^ (1u128 << 5) ^ (1u128 << 15);

        match codec.decode(corrupted) {
            EccResult::DoubleBitError { .. } => {
                // Expected
            }
            other => panic!("Expected double-bit error detection, got {:?}", other),
        }
    }

    #[test]
    fn test_bandwidth_overhead() {
        let codec = HammingCode::secded_64();
        let overhead = codec.bandwidth_overhead();

        // 72/64 = 1.125
        assert!((overhead - 1.125).abs() < 0.001);
    }

    #[test]
    fn test_error_rate_model() {
        let model = ErrorRateModel::hbm2e();

        // 80GB GPU at 70°C, sea level
        let errors_per_day = model.expected_errors_per_day(80, 70.0, 0.0);
        assert!(errors_per_day > 0.0);
        // Model produces finite values
        assert!(errors_per_day.is_finite());

        // At high altitude, should be higher
        let errors_denver = model.expected_errors_per_day(80, 70.0, 1600.0);
        assert!(errors_denver > errors_per_day);
    }

    #[test]
    fn test_cluster_error_rate() {
        let model = ErrorRateModel::hbm2e();

        // 1000 A100 GPUs (80GB each)
        let errors = model.cluster_errors_per_day(1000, 80, 65.0, 0.0);

        // At this scale, expect measurable error rate
        assert!(errors > 0.0);
    }

    #[test]
    fn test_ecc_memory_controller() {
        let config = EccConfig::default();
        let mut controller = EccMemoryController::new(80, config);

        let data = 0x1234567890ABCDEF_u64;
        let codeword = controller.write(data);

        let result = controller.read(0x1000, codeword, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), data);
        assert_eq!(controller.stats.reads, 1);
        assert_eq!(controller.stats.writes, 1);
    }

    #[test]
    fn test_ecc_controller_corrects_error() {
        let config = EccConfig::default();
        let mut controller = EccMemoryController::new(80, config);

        let data = 0xFEDCBA9876543210_u64;
        let codeword = controller.write(data);

        // Corrupt single bit
        let corrupted = codeword ^ (1u128 << 20);

        let result = controller.read(0x2000, corrupted, 100);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), data);
        assert_eq!(controller.stats.corrected_errors, 1);
    }

    #[test]
    fn test_chipkill_specs() {
        let amd = ChipkillCode::amd_chipkill();
        assert!(amd.survives_chip_failure());
        assert_eq!(amd.codeword_bits(), 72);

        let intel = ChipkillCode::intel_sddc();
        assert!(intel.survives_chip_failure());

        let hbm = ChipkillCode::hbm_redundancy();
        assert!(hbm.survives_chip_failure());
    }

    #[test]
    fn test_memory_scrubber() {
        let mut scrubber = MemoryScrubber::new(1024 * 1024, 100.0);

        assert_eq!(scrubber.progress(), 0.0);

        let addresses = scrubber.advance(1000, 1000);
        assert!(!addresses.is_empty());
        assert!(scrubber.progress() > 0.0);
    }

    #[test]
    fn test_error_injector() {
        let mut injector = ErrorInjector::new(0.5, 12345);

        let mut single_errors = 0;
        let mut multi_errors = 0;
        let mut no_errors = 0;

        for _ in 0..1000 {
            let (_, error_type) = injector.inject(0x123456789ABCDEF0);
            match error_type {
                Some(MemoryErrorType::SingleBitUpset) => single_errors += 1,
                Some(MemoryErrorType::MultiBitUpset) => multi_errors += 1,
                None => no_errors += 1,
                _ => {}
            }
        }

        // Should have roughly 50% errors with 50% rate
        assert!(single_errors > 0);
        assert!(no_errors > 0);
    }

    #[test]
    fn test_ecc_stats() {
        let mut stats = EccStats::default();
        stats.reads = 1000;
        stats.corrected_errors = 5;
        stats.uncorrectable_errors = 1;

        assert!((stats.correctable_error_rate() - 0.005).abs() < 0.0001);
        assert!((stats.uncorrectable_error_rate() - 0.001).abs() < 0.0001);
    }
}
