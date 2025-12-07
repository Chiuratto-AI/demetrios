//! Low-level interconnect protocols
//!
//! NVLink: High-speed GPU-to-GPU interconnect
//! PCIe: Standard peripheral interconnect
//!
//! Understanding these is essential for optimizing communication.

use std::time::Duration;

// ============================================================================
// NVLink Protocol
// ============================================================================

/// NVLink generation and characteristics
#[derive(Debug, Clone, Copy)]
pub struct NVLinkSpec {
    /// Generation (1-4)
    pub generation: u8,
    /// Number of links
    pub num_links: u8,
    /// Lanes per link
    pub lanes_per_link: u8,
    /// Data rate per lane (Gbps)
    pub lane_rate_gbps: f64,
    /// Encoding overhead (8b/10b, 64b/66b, etc.)
    pub encoding_efficiency: f64,
}

impl NVLinkSpec {
    /// NVLink 1.0 (Pascal P100)
    pub fn nvlink1() -> Self {
        Self {
            generation: 1,
            num_links: 4,
            lanes_per_link: 8,
            lane_rate_gbps: 20.0,
            encoding_efficiency: 128.0 / 130.0, // 128b/130b
        }
    }

    /// NVLink 2.0 (Volta V100)
    pub fn nvlink2() -> Self {
        Self {
            generation: 2,
            num_links: 6,
            lanes_per_link: 8,
            lane_rate_gbps: 25.0,
            encoding_efficiency: 128.0 / 130.0,
        }
    }

    /// NVLink 3.0 (Ampere A100)
    pub fn nvlink3() -> Self {
        Self {
            generation: 3,
            num_links: 12,
            lanes_per_link: 4, // NVLink 3 uses narrower links
            lane_rate_gbps: 50.0,
            encoding_efficiency: 128.0 / 130.0,
        }
    }

    /// NVLink 4.0 (Hopper H100)
    pub fn nvlink4() -> Self {
        Self {
            generation: 4,
            num_links: 18,
            lanes_per_link: 4,
            lane_rate_gbps: 50.0, // Same rate, more links
            encoding_efficiency: 128.0 / 130.0,
        }
    }

    /// Raw bandwidth per direction (bytes/sec)
    pub fn bandwidth_per_direction(&self) -> f64 {
        let bits_per_sec =
            self.num_links as f64 * self.lanes_per_link as f64 * self.lane_rate_gbps * 1e9;

        bits_per_sec * self.encoding_efficiency / 8.0
    }

    /// Total bidirectional bandwidth
    pub fn total_bandwidth(&self) -> f64 {
        self.bandwidth_per_direction() * 2.0
    }

    /// One-way latency (approximate)
    pub fn latency(&self) -> Duration {
        match self.generation {
            1 => Duration::from_nanos(700),
            2 => Duration::from_nanos(600),
            3 => Duration::from_nanos(500),
            4 => Duration::from_nanos(400),
            _ => Duration::from_nanos(1000),
        }
    }
}

/// NVLink packet types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NVLinkPacketType {
    /// Memory read request
    MemRd,
    /// Memory write (posted)
    MemWrPosted,
    /// Memory write (non-posted, requires completion)
    MemWrNonPosted,
    /// Atomic operation
    AtomicOp,
    /// Completion (response to non-posted request)
    Completion,
    /// Credit return (flow control)
    CreditReturn,
}

/// NVLink packet structure (simplified)
#[derive(Debug, Clone)]
pub struct NVLinkPacket {
    /// Packet type
    pub ptype: NVLinkPacketType,
    /// Virtual channel (for QoS)
    pub vc: u8,
    /// Destination address (for writes)
    pub address: Option<u64>,
    /// Payload (up to 256 bytes for data)
    pub payload: Vec<u8>,
    /// CRC for error detection
    pub crc: u32,
}

impl NVLinkPacket {
    /// Create a new packet
    pub fn new(ptype: NVLinkPacketType, vc: u8) -> Self {
        Self {
            ptype,
            vc,
            address: None,
            payload: Vec::new(),
            crc: 0,
        }
    }

    /// Packet overhead in bytes (header + CRC)
    pub fn overhead(&self) -> usize {
        match self.ptype {
            NVLinkPacketType::MemRd => 16,
            NVLinkPacketType::MemWrPosted => 16,
            NVLinkPacketType::MemWrNonPosted => 16,
            NVLinkPacketType::AtomicOp => 24,
            NVLinkPacketType::Completion => 12,
            NVLinkPacketType::CreditReturn => 8,
        }
    }

    /// Total packet size
    pub fn total_size(&self) -> usize {
        self.overhead() + self.payload.len()
    }

    /// Time to transmit at given bandwidth
    pub fn transmission_time(&self, bandwidth: f64) -> Duration {
        let bytes = self.total_size();
        let seconds = bytes as f64 / bandwidth;
        Duration::from_secs_f64(seconds)
    }
}

/// NVLink flow control using credits
#[derive(Debug)]
pub struct NVLinkFlowControl {
    /// Credits available per virtual channel
    credits: [u32; 4],
    /// Maximum credits per VC
    max_credits: u32,
    /// Pending requests awaiting credits
    pending: Vec<NVLinkPacket>,
}

impl NVLinkFlowControl {
    /// Create new flow control
    pub fn new(max_credits: u32) -> Self {
        Self {
            credits: [max_credits; 4],
            max_credits,
            pending: Vec::new(),
        }
    }

    /// Try to send packet (consumes credits)
    pub fn try_send(&mut self, packet: NVLinkPacket) -> Result<(), NVLinkPacket> {
        let vc = packet.vc as usize;
        let credits_needed = (packet.total_size() / 256 + 1) as u32;

        if self.credits[vc] >= credits_needed {
            self.credits[vc] -= credits_needed;
            Ok(())
        } else {
            self.pending.push(packet.clone());
            Err(packet)
        }
    }

    /// Return credits (from completion or credit return packet)
    pub fn return_credits(&mut self, vc: u8, count: u32) {
        self.credits[vc as usize] = (self.credits[vc as usize] + count).min(self.max_credits);

        // Try to send pending packets
        let pending = std::mem::take(&mut self.pending);
        for p in pending {
            if self.try_send(p.clone()).is_err() {
                self.pending.push(p);
            }
        }
    }

    /// Get available credits for a VC
    pub fn available_credits(&self, vc: u8) -> u32 {
        self.credits[vc as usize]
    }

    /// Get number of pending packets
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ============================================================================
// PCIe Protocol
// ============================================================================

/// PCIe generation specifications
#[derive(Debug, Clone, Copy)]
pub struct PCIeSpec {
    /// Generation (1-5)
    pub generation: u8,
    /// Number of lanes
    pub lanes: u8,
    /// Transfer rate per lane (GT/s)
    pub transfer_rate: f64,
    /// Encoding overhead
    pub encoding_efficiency: f64,
}

impl PCIeSpec {
    /// PCIe Gen 3 x16
    pub fn gen3_x16() -> Self {
        Self {
            generation: 3,
            lanes: 16,
            transfer_rate: 8.0,
            encoding_efficiency: 128.0 / 130.0,
        }
    }

    /// PCIe Gen 4 x16
    pub fn gen4_x16() -> Self {
        Self {
            generation: 4,
            lanes: 16,
            transfer_rate: 16.0,
            encoding_efficiency: 128.0 / 130.0,
        }
    }

    /// PCIe Gen 5 x16
    pub fn gen5_x16() -> Self {
        Self {
            generation: 5,
            lanes: 16,
            transfer_rate: 32.0,
            encoding_efficiency: 128.0 / 130.0,
        }
    }

    /// Bandwidth per direction (bytes/sec)
    pub fn bandwidth(&self) -> f64 {
        // GT/s * lanes * bits per transfer * encoding efficiency / 8
        self.transfer_rate * 1e9 * self.lanes as f64 * self.encoding_efficiency / 8.0
    }

    /// Round-trip latency (approximate)
    pub fn latency(&self) -> Duration {
        match self.generation {
            3 => Duration::from_nanos(700),
            4 => Duration::from_nanos(600),
            5 => Duration::from_nanos(500),
            _ => Duration::from_nanos(1000),
        }
    }
}

/// TLP (Transaction Layer Packet) types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TLPType {
    /// Memory Read Request (32-bit address)
    MemRd32,
    /// Memory Read Request (64-bit address)
    MemRd64,
    /// Memory Write Request (32-bit address)
    MemWr32,
    /// Memory Write Request (64-bit address)
    MemWr64,
    /// Completion with Data
    CplD,
    /// Completion without Data
    Cpl,
    /// Configuration Read
    CfgRd,
    /// Configuration Write
    CfgWr,
    /// Message
    Msg,
    /// Message with Data
    MsgD,
    /// AtomicOp (fetch-add, swap, CAS)
    AtomicOp,
}

/// PCIe Transaction Layer Packet
#[derive(Debug, Clone)]
pub struct TLP {
    /// TLP type
    pub tlp_type: TLPType,
    /// Traffic class
    pub tc: u8,
    /// Request ID
    pub requester_id: u16,
    /// Tag for matching completions
    pub tag: u8,
    /// Address (for memory TLPs)
    pub address: Option<u64>,
    /// Data payload (in DWORDs)
    pub data: Vec<u32>,
}

impl TLP {
    /// TLP header size in bytes
    pub fn header_size(&self) -> usize {
        match self.tlp_type {
            TLPType::MemRd32 | TLPType::MemWr32 => 12,
            TLPType::MemRd64 | TLPType::MemWr64 => 16,
            TLPType::CplD | TLPType::Cpl => 12,
            TLPType::CfgRd | TLPType::CfgWr => 12,
            TLPType::AtomicOp => 16,
            _ => 16,
        }
    }

    /// Total TLP size in bytes
    pub fn total_size(&self) -> usize {
        self.header_size() + self.data.len() * 4
    }

    /// Maximum payload size (MPS) check
    pub fn fits_in_mps(&self, mps: usize) -> bool {
        self.data.len() * 4 <= mps
    }

    /// Create memory write TLP
    pub fn mem_write(address: u64, data: Vec<u32>, tag: u8) -> Self {
        Self {
            tlp_type: if address > 0xFFFFFFFF {
                TLPType::MemWr64
            } else {
                TLPType::MemWr32
            },
            tc: 0,
            requester_id: 0,
            tag,
            address: Some(address),
            data,
        }
    }

    /// Create memory read TLP
    pub fn mem_read(address: u64, length_dw: u32, tag: u8) -> Self {
        Self {
            tlp_type: if address > 0xFFFFFFFF {
                TLPType::MemRd64
            } else {
                TLPType::MemRd32
            },
            tc: 0,
            requester_id: 0,
            tag,
            address: Some(address),
            data: vec![length_dw], // Length in DWs
        }
    }

    /// Create completion TLP
    pub fn completion(data: Vec<u32>, tag: u8) -> Self {
        Self {
            tlp_type: if data.is_empty() {
                TLPType::Cpl
            } else {
                TLPType::CplD
            },
            tc: 0,
            requester_id: 0,
            tag,
            address: None,
            data,
        }
    }
}

/// PCIe flow control
#[derive(Debug)]
pub struct PCIeFlowControl {
    /// Posted header credits
    pub ph: u32,
    /// Posted data credits
    pub pd: u32,
    /// Non-posted header credits
    pub nph: u32,
    /// Non-posted data credits
    pub npd: u32,
    /// Completion header credits
    pub cplh: u32,
    /// Completion data credits
    pub cpld: u32,
}

impl PCIeFlowControl {
    /// Create new flow control with typical values
    pub fn new() -> Self {
        Self {
            ph: 32,
            pd: 256,
            nph: 32,
            npd: 256,
            cplh: 64,
            cpld: 512,
        }
    }

    /// Check if TLP can be sent
    pub fn can_send(&self, tlp: &TLP) -> bool {
        let header_credits = 1u32;
        let data_credits = ((tlp.data.len() * 4 + 15) / 16) as u32;

        match tlp.tlp_type {
            TLPType::MemWr32 | TLPType::MemWr64 => {
                self.ph >= header_credits && self.pd >= data_credits
            }
            TLPType::MemRd32 | TLPType::MemRd64 | TLPType::AtomicOp => {
                self.nph >= header_credits && self.npd >= data_credits
            }
            TLPType::CplD | TLPType::Cpl => {
                self.cplh >= header_credits && self.cpld >= data_credits
            }
            _ => true,
        }
    }

    /// Consume credits for a TLP
    pub fn consume(&mut self, tlp: &TLP) {
        let header_credits = 1u32;
        let data_credits = ((tlp.data.len() * 4 + 15) / 16) as u32;

        match tlp.tlp_type {
            TLPType::MemWr32 | TLPType::MemWr64 => {
                self.ph = self.ph.saturating_sub(header_credits);
                self.pd = self.pd.saturating_sub(data_credits);
            }
            TLPType::MemRd32 | TLPType::MemRd64 | TLPType::AtomicOp => {
                self.nph = self.nph.saturating_sub(header_credits);
                self.npd = self.npd.saturating_sub(data_credits);
            }
            TLPType::CplD | TLPType::Cpl => {
                self.cplh = self.cplh.saturating_sub(header_credits);
                self.cpld = self.cpld.saturating_sub(data_credits);
            }
            _ => {}
        }
    }
}

impl Default for PCIeFlowControl {
    fn default() -> Self {
        Self::new()
    }
}

/// PCIe switch for multi-hop paths
#[derive(Debug)]
pub struct PCIeSwitch {
    /// Switch latency (typically 100-200 ns)
    pub latency: Duration,
    /// Internal bandwidth
    pub internal_bandwidth: f64,
    /// Number of ports
    pub num_ports: u8,
}

impl PCIeSwitch {
    /// Create a typical PCIe switch
    pub fn typical() -> Self {
        Self {
            latency: Duration::from_nanos(150),
            internal_bandwidth: 128e9, // 128 GB/s internal fabric
            num_ports: 16,
        }
    }
}

/// PCIe Root Complex to Endpoint path
#[derive(Debug)]
pub struct PCIePath {
    /// Hops in the path (switches)
    pub hops: Vec<PCIeSwitch>,
    /// Endpoint spec
    pub endpoint: PCIeSpec,
    /// Root complex overhead
    pub rc_latency: Duration,
}

impl PCIePath {
    /// Create a direct path (no switches)
    pub fn direct(endpoint: PCIeSpec) -> Self {
        Self {
            hops: Vec::new(),
            endpoint,
            rc_latency: Duration::from_nanos(100),
        }
    }

    /// Total one-way latency
    pub fn latency(&self) -> Duration {
        let switch_latency: Duration = self.hops.iter().map(|s| s.latency).sum();

        self.rc_latency + switch_latency + self.endpoint.latency()
    }

    /// Effective bandwidth (limited by bottleneck)
    pub fn bandwidth(&self) -> f64 {
        let mut min_bw = self.endpoint.bandwidth();

        for switch in &self.hops {
            min_bw = min_bw.min(switch.internal_bandwidth);
        }

        min_bw
    }
}

// ============================================================================
// GPUDirect RDMA
// ============================================================================

/// GPUDirect RDMA capabilities
#[derive(Debug, Clone)]
pub struct GPUDirectRDMA {
    /// Whether peer mapping is supported
    pub peer_mapping: bool,
    /// Whether async copy is supported
    pub async_copy: bool,
    /// Maximum transfer size
    pub max_transfer_size: usize,
    /// Requires BAR1 mapping
    pub uses_bar1: bool,
    /// Supported NIC drivers
    pub supported_nics: Vec<String>,
}

impl GPUDirectRDMA {
    /// Default capabilities
    pub fn default_capabilities() -> Self {
        Self {
            peer_mapping: true,
            async_copy: true,
            max_transfer_size: 128 * 1024 * 1024, // 128 MB
            uses_bar1: true,
            supported_nics: vec![
                "mlx5_core".to_string(), // Mellanox ConnectX
                "bnxt_re".to_string(),   // Broadcom
                "qedr".to_string(),      // Qlogic
            ],
        }
    }
}

/// RDMA operation codes
#[derive(Debug, Clone, Copy)]
pub enum RdmaOpcode {
    /// Send (requires matching receive)
    Send,
    /// RDMA Write (no receive needed)
    RdmaWrite,
    /// RDMA Write with Immediate
    RdmaWriteImm,
    /// RDMA Read
    RdmaRead,
    /// Atomic Compare-Swap
    AtomicCmpSwp,
    /// Atomic Fetch-Add
    AtomicFetchAdd,
}

/// RDMA verb for GPU memory
#[derive(Debug, Clone)]
pub enum RdmaVerb {
    /// Register GPU memory for RDMA
    RegMr {
        gpu_addr: u64,
        size: usize,
        access_flags: u32,
    },
    /// Post receive buffer
    PostRecv {
        wr_id: u64,
        lkey: u32,
        addr: u64,
        length: u32,
    },
    /// Post send (RDMA write)
    PostSend {
        wr_id: u64,
        opcode: RdmaOpcode,
        local_addr: u64,
        local_lkey: u32,
        remote_addr: u64,
        remote_rkey: u32,
        length: u32,
    },
    /// Poll completion queue
    PollCq { cq: u32, num_entries: u32 },
}

/// Model RDMA transfer performance
pub fn rdma_transfer_time(size: usize, bandwidth: f64, latency: Duration) -> Duration {
    let transfer_time = Duration::from_secs_f64(size as f64 / bandwidth);
    latency + transfer_time
}

// ============================================================================
// INTERCONNECT COMPARISON
// ============================================================================

/// Compare different interconnect options
#[derive(Debug)]
pub struct InterconnectComparison {
    /// Transfer size in bytes
    pub size: usize,
    /// NVLink time
    pub nvlink_time: Duration,
    /// PCIe time
    pub pcie_time: Duration,
    /// Network time (InfiniBand)
    pub network_time: Duration,
}

impl InterconnectComparison {
    /// Compare interconnects for a given transfer size
    pub fn compare(size: usize) -> Self {
        let nvlink = NVLinkSpec::nvlink3();
        let pcie = PCIeSpec::gen4_x16();

        // InfiniBand NDR: 400 Gbps = 50 GB/s
        let ib_bandwidth = 50e9;
        let ib_latency = Duration::from_micros(1);

        let nvlink_time = Duration::from_secs_f64(size as f64 / nvlink.bandwidth_per_direction())
            + nvlink.latency();

        let pcie_time =
            Duration::from_secs_f64(size as f64 / pcie.bandwidth()) + pcie.latency() * 2; // Round trip through CPU

        let network_time = rdma_transfer_time(size, ib_bandwidth, ib_latency);

        Self {
            size,
            nvlink_time,
            pcie_time,
            network_time,
        }
    }

    /// Get the fastest interconnect for this transfer
    pub fn fastest(&self) -> &'static str {
        let min_time = self.nvlink_time.min(self.pcie_time).min(self.network_time);

        if min_time == self.nvlink_time {
            "NVLink"
        } else if min_time == self.pcie_time {
            "PCIe"
        } else {
            "Network"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvlink_bandwidth() {
        let nvlink3 = NVLinkSpec::nvlink3();
        let bw = nvlink3.bandwidth_per_direction();

        // A100 NVLink 3: 12 links × 4 lanes × 50 Gbps × 128/130 ≈ 300 GB/s
        assert!(
            bw > 290e9 && bw < 310e9,
            "NVLink3 bandwidth: {} GB/s",
            bw / 1e9
        );
    }

    #[test]
    fn test_nvlink4_bandwidth() {
        let nvlink4 = NVLinkSpec::nvlink4();
        let bw = nvlink4.bandwidth_per_direction();

        // H100 NVLink 4: 18 links × 4 lanes × 50 Gbps × 128/130 ≈ 450 GB/s
        assert!(
            bw > 440e9 && bw < 460e9,
            "NVLink4 bandwidth: {} GB/s",
            bw / 1e9
        );
    }

    #[test]
    fn test_pcie_bandwidth() {
        let pcie4 = PCIeSpec::gen4_x16();
        let bw = pcie4.bandwidth();

        // PCIe 4.0 x16: 16 GT/s × 16 lanes × 128/130 ≈ 31.5 GB/s
        assert!(bw > 30e9 && bw < 33e9, "PCIe4 bandwidth: {} GB/s", bw / 1e9);
    }

    #[test]
    fn test_pcie5_bandwidth() {
        let pcie5 = PCIeSpec::gen5_x16();
        let bw = pcie5.bandwidth();

        // PCIe 5.0 x16: 32 GT/s × 16 lanes × 128/130 ≈ 63 GB/s
        assert!(bw > 60e9 && bw < 66e9, "PCIe5 bandwidth: {} GB/s", bw / 1e9);
    }

    #[test]
    fn test_tlp_creation() {
        let tlp = TLP::mem_write(0x1000_0000_0000, vec![1, 2, 3, 4], 0);

        assert!(matches!(tlp.tlp_type, TLPType::MemWr64));
        assert_eq!(tlp.data.len(), 4);
        assert_eq!(tlp.total_size(), 16 + 16); // Header + data
    }

    #[test]
    fn test_nvlink_packet() {
        let mut packet = NVLinkPacket::new(NVLinkPacketType::MemWrPosted, 0);
        packet.payload = vec![0u8; 256];

        assert_eq!(packet.overhead(), 16);
        assert_eq!(packet.total_size(), 16 + 256);
    }

    #[test]
    fn test_nvlink_flow_control() {
        let mut fc = NVLinkFlowControl::new(64);

        let packet = NVLinkPacket::new(NVLinkPacketType::MemWrPosted, 0);
        assert!(fc.try_send(packet).is_ok());

        assert!(fc.available_credits(0) < 64);
    }

    #[test]
    fn test_pcie_flow_control() {
        let fc = PCIeFlowControl::new();

        let tlp = TLP::mem_write(0x1000, vec![1, 2, 3, 4], 0);
        assert!(fc.can_send(&tlp));
    }

    #[test]
    fn test_interconnect_comparison() {
        let comparison = InterconnectComparison::compare(1024 * 1024); // 1 MB

        // NVLink should be fastest for intra-node
        assert!(comparison.nvlink_time < comparison.pcie_time);
        assert_eq!(comparison.fastest(), "NVLink");
    }

    #[test]
    fn test_pcie_path() {
        let path = PCIePath::direct(PCIeSpec::gen4_x16());

        assert!(path.latency() < Duration::from_micros(2));
        assert!(path.bandwidth() > 30e9);
    }

    #[test]
    fn test_rdma_transfer_time() {
        let time = rdma_transfer_time(1024 * 1024, 50e9, Duration::from_micros(1));

        // 1 MB at 50 GB/s = ~20 µs + 1 µs latency
        assert!(time > Duration::from_micros(20));
        assert!(time < Duration::from_micros(25));
    }
}
