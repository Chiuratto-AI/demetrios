//! DRAM Microarchitecture and Performance
//!
//! SCIENTIFIC HONESTY:
//! - DRAM timing parameters vary by manufacturer and speed grade
//! - Actual performance depends on access pattern
//! - "Peak bandwidth" is achievable only with perfect access patterns
//! - HBM is more complex than this model suggests
//!
//! Sources:
//! - JEDEC DDR5/HBM2E/HBM3 specifications
//! - "A Primer on Memory Consistency and Cache Coherence" (Sorin et al.)
//! - "Demystifying GPU Memory" (Jia et al., ISPASS 2018)
//! - Micron/Samsung technical documentation

use std::collections::HashMap;

// ============================================================================
// DRAM ORGANIZATION
// ============================================================================

/// DRAM organization parameters
///
/// REALITY CHECK:
/// These numbers vary by:
/// - DRAM generation (DDR4/5, HBM2/2E/3)
/// - Speed grade
/// - Manufacturer
/// - Temperature
#[derive(Debug, Clone)]
pub struct DramOrganization {
    pub name: &'static str,
    /// Number of channels
    pub channels: u32,
    /// Banks per channel
    pub banks_per_channel: u32,
    /// Bank groups per channel (for bank group interleaving)
    pub bank_groups: u32,
    /// Row buffer size (bytes)
    pub row_buffer_size: u32,
    /// Number of rows per bank
    pub rows_per_bank: u32,
    /// Columns per row
    pub columns_per_row: u32,
    /// Device width (bits per device)
    pub device_width: u32,
    /// Devices per channel
    pub devices_per_channel: u32,
}

impl DramOrganization {
    /// HBM2E (A100)
    ///
    /// CAVEAT: These are approximate; NVIDIA doesn't publish exact details
    pub fn hbm2e() -> Self {
        Self {
            name: "HBM2E",
            channels: 8,
            banks_per_channel: 16,
            bank_groups: 4,
            row_buffer_size: 2048, // 2KB per bank
            rows_per_bank: 16384,
            columns_per_row: 256,
            device_width: 128,
            devices_per_channel: 1,
        }
    }

    /// HBM3 (H100)
    pub fn hbm3() -> Self {
        Self {
            name: "HBM3",
            channels: 16,
            banks_per_channel: 32,
            bank_groups: 8,
            row_buffer_size: 2048,
            rows_per_bank: 32768,
            columns_per_row: 256,
            device_width: 64,
            devices_per_channel: 2,
        }
    }

    /// GDDR6X (RTX 4090)
    pub fn gddr6x() -> Self {
        Self {
            name: "GDDR6X",
            channels: 12,
            banks_per_channel: 16,
            bank_groups: 4,
            row_buffer_size: 1024,
            rows_per_bank: 16384,
            columns_per_row: 128,
            device_width: 32,
            devices_per_channel: 1,
        }
    }

    /// Total banks
    pub fn total_banks(&self) -> u32 {
        self.channels * self.banks_per_channel
    }

    /// Total row buffer capacity (on-chip SRAM)
    pub fn total_row_buffer_bytes(&self) -> u64 {
        self.total_banks() as u64 * self.row_buffer_size as u64
    }

    /// Theoretical maximum parallelism
    pub fn max_parallel_accesses(&self) -> u32 {
        self.total_banks()
    }

    /// Capacity per bank in bytes
    pub fn bank_capacity(&self) -> u64 {
        self.rows_per_bank as u64 * self.row_buffer_size as u64
    }

    /// Total capacity in bytes
    pub fn total_capacity(&self) -> u64 {
        self.total_banks() as u64 * self.bank_capacity()
    }
}

// ============================================================================
// DRAM TIMING PARAMETERS
// ============================================================================

/// DRAM timing parameters
///
/// SCIENTIFIC HONESTY:
/// These timings are:
/// - Specified in clock cycles at a particular frequency
/// - Have variation across temperature
/// - Are MINIMUM values; actual may be longer
/// - Different DRAM generations have different absolute values
#[derive(Debug, Clone)]
pub struct DramTiming {
    pub name: &'static str,
    /// Clock frequency (MHz)
    pub clock_mhz: f64,
    /// tRC: Row cycle time (activate to activate, same bank)
    pub t_rc: u32,
    /// tRAS: Row active time (activate to precharge)
    pub t_ras: u32,
    /// tRP: Row precharge time (precharge to activate)
    pub t_rp: u32,
    /// tRCD: RAS to CAS delay (activate to read/write)
    pub t_rcd: u32,
    /// CL: CAS latency (read command to data)
    pub cl: u32,
    /// tWR: Write recovery time
    pub t_wr: u32,
    /// tRRD_S: Row to row delay (same bank group)
    pub t_rrd_s: u32,
    /// tRRD_L: Row to row delay (different bank group)
    pub t_rrd_l: u32,
    /// tFAW: Four activate window
    pub t_faw: u32,
    /// tREFI: Refresh interval
    pub t_refi: u32,
    /// tRFC: Refresh cycle time
    pub t_rfc: u32,
    /// Burst length
    pub burst_length: u32,
}

impl DramTiming {
    /// HBM2E at 1.6 GHz effective (A100)
    pub fn hbm2e() -> Self {
        Self {
            name: "HBM2E-1600",
            clock_mhz: 1600.0,
            t_rc: 46,
            t_ras: 32,
            t_rp: 14,
            t_rcd: 14,
            cl: 14,
            t_wr: 15,
            t_rrd_s: 4,
            t_rrd_l: 6,
            t_faw: 16,
            t_refi: 7800,
            t_rfc: 260,
            burst_length: 4,
        }
    }

    /// HBM3 at 2.0 GHz effective (H100)
    pub fn hbm3() -> Self {
        Self {
            name: "HBM3-2000",
            clock_mhz: 2000.0,
            t_rc: 40,
            t_ras: 28,
            t_rp: 12,
            t_rcd: 12,
            cl: 12,
            t_wr: 12,
            t_rrd_s: 3,
            t_rrd_l: 5,
            t_faw: 14,
            t_refi: 9750,
            t_rfc: 350,
            burst_length: 8,
        }
    }

    /// GDDR6X timing
    pub fn gddr6x() -> Self {
        Self {
            name: "GDDR6X-21000",
            clock_mhz: 21000.0 / 4.0, // Effective data rate / 4 (QDR)
            t_rc: 60,
            t_ras: 40,
            t_rp: 20,
            t_rcd: 20,
            cl: 20,
            t_wr: 18,
            t_rrd_s: 5,
            t_rrd_l: 8,
            t_faw: 20,
            t_refi: 3900,
            t_rfc: 280,
            burst_length: 16,
        }
    }

    /// Convert cycles to nanoseconds
    pub fn cycles_to_ns(&self, cycles: u32) -> f64 {
        cycles as f64 * 1000.0 / self.clock_mhz
    }

    /// Row buffer hit latency (best case)
    pub fn row_buffer_hit_latency_ns(&self) -> f64 {
        self.cycles_to_ns(self.cl)
    }

    /// Row buffer miss (different row open) latency
    pub fn row_buffer_conflict_latency_ns(&self) -> f64 {
        // Need to precharge, activate, then read
        self.cycles_to_ns(self.t_rp + self.t_rcd + self.cl)
    }

    /// Row buffer miss (no row open) latency
    pub fn row_buffer_miss_latency_ns(&self) -> f64 {
        // Just activate and read
        self.cycles_to_ns(self.t_rcd + self.cl)
    }

    /// Refresh overhead fraction
    ///
    /// UNCOMFORTABLE TRUTH: Refresh steals a significant fraction of bandwidth
    pub fn refresh_overhead(&self) -> f64 {
        self.t_rfc as f64 / self.t_refi as f64
    }
}

// ============================================================================
// DRAM ACCESS PATTERN ANALYSIS
// ============================================================================

/// DRAM access pattern analyzer
///
/// KEY INSIGHT: Access pattern determines performance more than
/// "peak bandwidth" suggests.
#[derive(Debug)]
pub struct DramAccessAnalyzer {
    org: DramOrganization,
    timing: DramTiming,
    /// Row buffer state per bank: Option<row_address>
    row_buffers: Vec<Option<u32>>,
    /// Statistics
    stats: DramStats,
}

/// DRAM access statistics
#[derive(Debug, Default, Clone)]
pub struct DramStats {
    pub total_accesses: u64,
    pub row_buffer_hits: u64,
    pub row_buffer_misses: u64,
    pub row_buffer_conflicts: u64,
    pub bank_conflicts: u64,
    pub total_cycles: u64,
}

impl DramStats {
    pub fn row_buffer_hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            return 0.0;
        }
        self.row_buffer_hits as f64 / self.total_accesses as f64
    }

    pub fn effective_latency_cycles(&self) -> f64 {
        if self.total_accesses == 0 {
            return 0.0;
        }
        self.total_cycles as f64 / self.total_accesses as f64
    }

    pub fn conflict_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            return 0.0;
        }
        self.row_buffer_conflicts as f64 / self.total_accesses as f64
    }
}

impl DramAccessAnalyzer {
    pub fn new(org: DramOrganization, timing: DramTiming) -> Self {
        let num_banks = org.total_banks() as usize;
        Self {
            org,
            timing,
            row_buffers: vec![None; num_banks],
            stats: DramStats::default(),
        }
    }

    /// Analyze a single access
    ///
    /// Address decoding (simplified):
    /// | Channel | Bank | Row | Column |
    pub fn access(&mut self, address: u64) -> AccessResult {
        self.stats.total_accesses += 1;

        // Decode address
        let (channel, bank, row, col) = self.decode_address(address);
        let bank_idx = (channel * self.org.banks_per_channel + bank) as usize;

        let (latency, access_type) = match self.row_buffers[bank_idx] {
            Some(open_row) if open_row == row => {
                // Row buffer hit
                self.stats.row_buffer_hits += 1;
                (self.timing.cl, AccessType::RowBufferHit)
            }
            Some(_) => {
                // Row buffer conflict (different row open)
                self.stats.row_buffer_conflicts += 1;
                self.row_buffers[bank_idx] = Some(row);
                let lat = self.timing.t_rp + self.timing.t_rcd + self.timing.cl;
                (lat, AccessType::RowBufferConflict)
            }
            None => {
                // Row buffer miss (no row open)
                self.stats.row_buffer_misses += 1;
                self.row_buffers[bank_idx] = Some(row);
                let lat = self.timing.t_rcd + self.timing.cl;
                (lat, AccessType::RowBufferMiss)
            }
        };

        self.stats.total_cycles += latency as u64;

        AccessResult {
            address,
            channel,
            bank,
            row,
            column: col,
            latency_cycles: latency,
            latency_ns: self.timing.cycles_to_ns(latency),
            access_type,
        }
    }

    /// Analyze a sequence of accesses
    pub fn analyze_sequence(&mut self, addresses: &[u64]) -> SequenceAnalysis {
        self.reset();

        for &addr in addresses {
            self.access(addr);
        }

        SequenceAnalysis {
            stats: self.stats.clone(),
            theoretical_bandwidth_util: self.theoretical_utilization(),
            effective_bandwidth_util: self.effective_utilization(),
            bottleneck: self.identify_bottleneck(),
        }
    }

    fn decode_address(&self, address: u64) -> (u32, u32, u32, u32) {
        // Simplified address decoding
        let col_bits = (self.org.columns_per_row as f64).log2() as u32;
        let row_bits = (self.org.rows_per_bank as f64).log2() as u32;
        let bank_bits = (self.org.banks_per_channel as f64).log2() as u32;
        let channel_bits = (self.org.channels as f64).log2() as u32;

        let col = (address as u32) & ((1 << col_bits) - 1);
        let row = ((address >> col_bits) as u32) & ((1 << row_bits) - 1);
        let bank = ((address >> (col_bits + row_bits)) as u32) & ((1 << bank_bits) - 1);
        let channel =
            ((address >> (col_bits + row_bits + bank_bits)) as u32) & ((1 << channel_bits) - 1);

        (channel, bank, row, col)
    }

    fn theoretical_utilization(&self) -> f64 {
        // Theoretical: every access is a row buffer hit
        let best_case_cycles = self.stats.total_accesses * self.timing.cl as u64;
        if self.stats.total_cycles == 0 {
            return 1.0;
        }
        best_case_cycles as f64 / self.stats.total_cycles as f64
    }

    fn effective_utilization(&self) -> f64 {
        // Account for refresh overhead
        self.theoretical_utilization() * (1.0 - self.timing.refresh_overhead())
    }

    fn identify_bottleneck(&self) -> Bottleneck {
        let hit_rate = self.stats.row_buffer_hit_rate();
        let conflict_rate = self.stats.conflict_rate();

        if hit_rate < 0.5 {
            Bottleneck::RowBufferMisses {
                hit_rate,
                recommendation: "Improve spatial locality, tile data to row size".to_string(),
            }
        } else if conflict_rate > 0.3 {
            Bottleneck::RowBufferConflicts {
                conflict_rate,
                recommendation: "Pad arrays to avoid bank conflicts".to_string(),
            }
        } else {
            Bottleneck::None
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.stats = DramStats::default();
        self.row_buffers.fill(None);
    }

    pub fn get_stats(&self) -> &DramStats {
        &self.stats
    }
}

/// Result of a single access
#[derive(Debug, Clone)]
pub struct AccessResult {
    pub address: u64,
    pub channel: u32,
    pub bank: u32,
    pub row: u32,
    pub column: u32,
    pub latency_cycles: u32,
    pub latency_ns: f64,
    pub access_type: AccessType,
}

/// Access type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    RowBufferHit,
    RowBufferMiss,
    RowBufferConflict,
}

/// Sequence analysis result
#[derive(Debug)]
pub struct SequenceAnalysis {
    pub stats: DramStats,
    pub theoretical_bandwidth_util: f64,
    pub effective_bandwidth_util: f64,
    pub bottleneck: Bottleneck,
}

/// Identified bottleneck
#[derive(Debug, Clone)]
pub enum Bottleneck {
    None,
    RowBufferMisses {
        hit_rate: f64,
        recommendation: String,
    },
    RowBufferConflicts {
        conflict_rate: f64,
        recommendation: String,
    },
    BankConflicts {
        conflict_rate: f64,
        recommendation: String,
    },
    Refresh {
        overhead: f64,
    },
}

// ============================================================================
// HONEST PERFORMANCE MODEL
// ============================================================================

/// Honest memory bandwidth model
///
/// UNCOMFORTABLE TRUTH:
/// "Peak bandwidth" (e.g., 2 TB/s for A100) is rarely achievable.
/// Real bandwidth depends on access pattern.
#[derive(Debug)]
pub struct HonestBandwidthModel {
    org: DramOrganization,
    timing: DramTiming,
    /// Peak theoretical bandwidth (GB/s)
    peak_bandwidth_gbps: f64,
}

impl HonestBandwidthModel {
    pub fn a100() -> Self {
        Self {
            org: DramOrganization::hbm2e(),
            timing: DramTiming::hbm2e(),
            peak_bandwidth_gbps: 2039.0, // 2 TB/s advertised
        }
    }

    pub fn h100() -> Self {
        Self {
            org: DramOrganization::hbm3(),
            timing: DramTiming::hbm3(),
            peak_bandwidth_gbps: 3350.0, // 3.35 TB/s advertised
        }
    }

    pub fn rtx4090() -> Self {
        Self {
            org: DramOrganization::gddr6x(),
            timing: DramTiming::gddr6x(),
            peak_bandwidth_gbps: 1008.0, // 1 TB/s advertised
        }
    }

    /// Estimate achievable bandwidth for given access pattern
    pub fn estimate_bandwidth(&self, pattern: &AccessPattern) -> BandwidthEstimate {
        let mut bandwidth = self.peak_bandwidth_gbps;
        let mut factors = Vec::new();

        // Factor 1: Row buffer efficiency
        let row_buffer_factor = 0.3 + 0.7 * pattern.spatial_locality;
        bandwidth *= row_buffer_factor;
        factors.push(BandwidthFactor {
            name: "Row buffer efficiency".to_string(),
            value: row_buffer_factor,
            notes: format!("Spatial locality: {:.0}%", pattern.spatial_locality * 100.0),
        });

        // Factor 2: Bank-level parallelism
        let blp_factor = (pattern.concurrent_banks as f64 / self.org.total_banks() as f64)
            .min(1.0)
            .max(0.1);
        bandwidth *= blp_factor;
        factors.push(BandwidthFactor {
            name: "Bank-level parallelism".to_string(),
            value: blp_factor,
            notes: format!(
                "{} of {} banks active",
                pattern.concurrent_banks,
                self.org.total_banks()
            ),
        });

        // Factor 3: Request size efficiency
        let burst_bytes = (self.org.device_width / 8
            * self.org.devices_per_channel
            * self.timing.burst_length) as f64;
        let size_factor = (pattern.average_request_size as f64 / burst_bytes).min(1.0);
        bandwidth *= size_factor;
        factors.push(BandwidthFactor {
            name: "Request size efficiency".to_string(),
            value: size_factor,
            notes: format!(
                "Avg request: {} bytes, burst: {} bytes",
                pattern.average_request_size, burst_bytes
            ),
        });

        // Factor 4: Refresh overhead
        let refresh_factor = 1.0 - self.timing.refresh_overhead();
        bandwidth *= refresh_factor;
        factors.push(BandwidthFactor {
            name: "Refresh overhead".to_string(),
            value: refresh_factor,
            notes: format!("{:.1}% overhead", self.timing.refresh_overhead() * 100.0),
        });

        // Factor 5: Read/write ratio (writes are slower)
        let rw_factor = pattern.read_fraction * 1.0 + (1.0 - pattern.read_fraction) * 0.9;
        bandwidth *= rw_factor;
        factors.push(BandwidthFactor {
            name: "Read/write mix".to_string(),
            value: rw_factor,
            notes: format!("{:.0}% reads", pattern.read_fraction * 100.0),
        });

        BandwidthEstimate {
            peak_bandwidth_gbps: self.peak_bandwidth_gbps,
            achievable_bandwidth_gbps: bandwidth,
            efficiency: bandwidth / self.peak_bandwidth_gbps,
            factors,
            honest_notes: vec![
                "These estimates assume optimal scheduling".to_string(),
                "Actual bandwidth varies with load and contention".to_string(),
                "PCIe/NVLink traffic competes for memory bandwidth".to_string(),
            ],
        }
    }

    /// Get peak bandwidth
    pub fn peak_bandwidth(&self) -> f64 {
        self.peak_bandwidth_gbps
    }
}

/// Access pattern description
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Spatial locality (0 = random, 1 = perfectly sequential)
    pub spatial_locality: f64,
    /// Number of banks accessed concurrently
    pub concurrent_banks: u32,
    /// Average request size (bytes)
    pub average_request_size: u32,
    /// Fraction of operations that are reads (vs writes)
    pub read_fraction: f64,
}

impl AccessPattern {
    /// Sequential access (best case)
    pub fn sequential() -> Self {
        Self {
            spatial_locality: 0.95,
            concurrent_banks: 96, // High parallelism for sequential streaming
            average_request_size: 256,
            read_fraction: 1.0,
        }
    }

    /// Random access (worst case)
    pub fn random() -> Self {
        Self {
            spatial_locality: 0.1,
            concurrent_banks: 4,
            average_request_size: 32,
            read_fraction: 0.5,
        }
    }

    /// Strided access
    pub fn strided(stride: u32) -> Self {
        let locality = (128.0 / stride as f64).min(1.0);
        Self {
            spatial_locality: locality,
            concurrent_banks: 16,
            average_request_size: 64,
            read_fraction: 0.5,
        }
    }

    /// Typical ML workload (GEMM)
    pub fn gemm() -> Self {
        Self {
            spatial_locality: 0.8,
            concurrent_banks: 24,
            average_request_size: 128,
            read_fraction: 0.67, // 2 reads per write
        }
    }

    /// Typical inference workload
    pub fn inference() -> Self {
        Self {
            spatial_locality: 0.85,
            concurrent_banks: 28,
            average_request_size: 128,
            read_fraction: 0.9, // Mostly reads
        }
    }
}

/// Bandwidth estimate result
#[derive(Debug)]
pub struct BandwidthEstimate {
    pub peak_bandwidth_gbps: f64,
    pub achievable_bandwidth_gbps: f64,
    pub efficiency: f64,
    pub factors: Vec<BandwidthFactor>,
    pub honest_notes: Vec<String>,
}

/// A factor affecting bandwidth
#[derive(Debug)]
pub struct BandwidthFactor {
    pub name: String,
    pub value: f64,
    pub notes: String,
}

// ============================================================================
// ROW BUFFER POLICY ANALYSIS
// ============================================================================

/// Row buffer management policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowBufferPolicy {
    /// Close row after each access
    ClosePage,
    /// Keep row open until conflict
    OpenPage,
    /// Adaptive based on access pattern
    Adaptive,
}

/// Row buffer policy analyzer
#[derive(Debug)]
pub struct RowBufferPolicyAnalyzer {
    /// Simulated results per policy
    results: HashMap<RowBufferPolicy, DramStats>,
}

impl RowBufferPolicyAnalyzer {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Analyze which policy is best for given access pattern
    pub fn analyze(
        &mut self,
        addresses: &[u64],
        org: &DramOrganization,
        timing: &DramTiming,
    ) -> PolicyRecommendation {
        // Simulate close-page
        let close_stats = self.simulate_close_page(addresses, org, timing);

        // Simulate open-page
        let open_stats = self.simulate_open_page(addresses, org, timing);

        self.results
            .insert(RowBufferPolicy::ClosePage, close_stats.clone());
        self.results
            .insert(RowBufferPolicy::OpenPage, open_stats.clone());

        // Compare
        let close_latency = close_stats.effective_latency_cycles();
        let open_latency = open_stats.effective_latency_cycles();

        if open_stats.row_buffer_hit_rate() > 0.5 {
            PolicyRecommendation {
                recommended: RowBufferPolicy::OpenPage,
                reason: format!(
                    "High row buffer hit rate ({:.0}%) benefits from open-page",
                    open_stats.row_buffer_hit_rate() * 100.0
                ),
                latency_improvement: (close_latency - open_latency) / close_latency,
            }
        } else {
            PolicyRecommendation {
                recommended: RowBufferPolicy::ClosePage,
                reason: format!(
                    "Low hit rate ({:.0}%) - close-page avoids conflict penalty",
                    open_stats.row_buffer_hit_rate() * 100.0
                ),
                latency_improvement: (open_latency - close_latency) / open_latency,
            }
        }
    }

    fn simulate_close_page(
        &self,
        addresses: &[u64],
        org: &DramOrganization,
        timing: &DramTiming,
    ) -> DramStats {
        let mut stats = DramStats::default();

        for _ in addresses {
            stats.total_accesses += 1;
            // Close-page: always pay activate + CAS
            stats.row_buffer_misses += 1;
            stats.total_cycles += (timing.t_rcd + timing.cl) as u64;
        }

        stats
    }

    fn simulate_open_page(
        &self,
        addresses: &[u64],
        org: &DramOrganization,
        timing: &DramTiming,
    ) -> DramStats {
        let mut analyzer = DramAccessAnalyzer::new(org.clone(), timing.clone());

        for &addr in addresses {
            analyzer.access(addr);
        }

        analyzer.stats.clone()
    }
}

impl Default for RowBufferPolicyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy recommendation
#[derive(Debug)]
pub struct PolicyRecommendation {
    pub recommended: RowBufferPolicy,
    pub reason: String,
    pub latency_improvement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hbm2e_organization() {
        let org = DramOrganization::hbm2e();

        assert_eq!(org.banks_per_channel, 16);
        assert!(org.total_row_buffer_bytes() > 0);
        assert_eq!(org.total_banks(), 8 * 16);
    }

    #[test]
    fn test_hbm3_organization() {
        let org = DramOrganization::hbm3();

        assert!(org.total_banks() > DramOrganization::hbm2e().total_banks());
    }

    #[test]
    fn test_row_buffer_latencies() {
        let timing = DramTiming::hbm2e();

        let hit_lat = timing.row_buffer_hit_latency_ns();
        let miss_lat = timing.row_buffer_miss_latency_ns();
        let conflict_lat = timing.row_buffer_conflict_latency_ns();

        // Hit should be fastest
        assert!(hit_lat < miss_lat);
        assert!(miss_lat < conflict_lat);

        // Conflict is ~3x hit latency for HBM2E
        assert!(conflict_lat / hit_lat > 2.0);
    }

    #[test]
    fn test_refresh_overhead() {
        let timing = DramTiming::hbm2e();
        let overhead = timing.refresh_overhead();

        // Should be around 3-5%
        assert!(overhead > 0.02);
        assert!(overhead < 0.10);
    }

    #[test]
    fn test_access_analyzer() {
        let org = DramOrganization::hbm2e();
        let timing = DramTiming::hbm2e();
        let mut analyzer = DramAccessAnalyzer::new(org, timing);

        // Sequential accesses to same row should hit
        let result1 = analyzer.access(0x1000);
        let result2 = analyzer.access(0x1004);

        assert!(matches!(result1.access_type, AccessType::RowBufferMiss));
        // Second access to same row should hit
        assert!(matches!(result2.access_type, AccessType::RowBufferHit));
    }

    #[test]
    fn test_bandwidth_estimation() {
        let model = HonestBandwidthModel::a100();

        // Sequential should achieve high bandwidth
        let seq_est = model.estimate_bandwidth(&AccessPattern::sequential());
        assert!(seq_est.efficiency > 0.6);

        // Random should be much worse
        let rand_est = model.estimate_bandwidth(&AccessPattern::random());
        assert!(rand_est.efficiency < 0.3);

        // Random should be significantly worse than sequential
        assert!(rand_est.achievable_bandwidth_gbps < seq_est.achievable_bandwidth_gbps * 0.5);
    }

    #[test]
    fn test_strided_pattern() {
        let stride_small = AccessPattern::strided(64);
        let stride_large = AccessPattern::strided(4096);

        // Larger stride = worse locality
        assert!(stride_small.spatial_locality > stride_large.spatial_locality);
    }

    #[test]
    fn test_sequence_analysis() {
        let org = DramOrganization::hbm2e();
        let timing = DramTiming::hbm2e();
        let mut analyzer = DramAccessAnalyzer::new(org, timing);

        // Sequential addresses
        let addresses: Vec<u64> = (0..100).map(|i| i * 4).collect();
        let analysis = analyzer.analyze_sequence(&addresses);

        // Should have good hit rate for sequential
        assert!(analysis.stats.row_buffer_hit_rate() > 0.5);
    }

    #[test]
    fn test_bottleneck_detection() {
        let org = DramOrganization::hbm2e();
        let timing = DramTiming::hbm2e();
        let mut analyzer = DramAccessAnalyzer::new(org, timing);

        // Random addresses (poor locality)
        let addresses: Vec<u64> = (0..100).map(|i| i * 1024 * 1024).collect();
        let analysis = analyzer.analyze_sequence(&addresses);

        // Should identify row buffer misses as bottleneck
        assert!(matches!(
            analysis.bottleneck,
            Bottleneck::RowBufferMisses { .. }
        ));
    }
}
