//! Point-to-Point Communication Primitives
//!
//! The foundation of all collectives. Every all-reduce, broadcast, etc.
//! decomposes into P2P operations.

use super::memory::{MemFence, MemoryOrder, MemoryScope};
use crate::optimize::pool::DeviceId;
use crate::runtime::BufferError;
use std::sync::Arc;

/// Device pointer wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    pub fn null() -> Self {
        Self(0)
    }

    pub fn is_null(&self) -> bool {
        self.0 == 0
    }

    pub fn offset(&self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }
}

/// P2P capability between two devices
#[derive(Debug, Clone, Copy)]
pub struct P2PCapability {
    /// Source device
    pub src: DeviceId,
    /// Destination device
    pub dst: DeviceId,
    /// Direct peer access supported?
    pub peer_access: bool,
    /// Atomics supported across link?
    pub peer_atomics: bool,
    /// Native ordering (vs relaxed)?
    pub native_ordering: bool,
    /// Maximum copy size in single operation
    pub max_copy_size: usize,
    /// Optimal copy size (for bandwidth)
    pub optimal_copy_size: usize,
}

impl P2PCapability {
    /// Query capability between devices
    pub fn query(src: DeviceId, dst: DeviceId) -> Result<Self, BufferError> {
        if src == dst {
            return Ok(Self {
                src,
                dst,
                peer_access: true,
                peer_atomics: true,
                native_ordering: true,
                max_copy_size: usize::MAX,
                optimal_copy_size: 256 * 1024 * 1024, // 256 MB
            });
        }

        // Would call cudaDeviceCanAccessPeer, cudaDeviceGetP2PAttribute, etc.
        // Simulate for now
        Ok(Self {
            src,
            dst,
            peer_access: true,
            peer_atomics: false, // Usually not supported across PCIe
            native_ordering: false,
            max_copy_size: 256 * 1024 * 1024,
            optimal_copy_size: 4 * 1024 * 1024, // 4 MB for PCIe
        })
    }

    /// Enable peer access
    pub fn enable(&self) -> Result<(), BufferError> {
        if self.src == self.dst {
            return Ok(());
        }

        // Would call cudaDeviceEnablePeerAccess
        Ok(())
    }

    /// Check if direct P2P is possible
    pub fn is_direct(&self) -> bool {
        self.peer_access
    }
}

/// Copy engine types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyEngine {
    /// Use compute engine (kernel-based copy)
    Compute,
    /// Use DMA copy engine (cudaMemcpyAsync)
    Dma,
    /// Use P2P engine (direct peer access)
    P2P,
    /// Use GPUDirect RDMA
    Rdma,
}

impl CopyEngine {
    /// Get appropriate engine for given transfer
    pub fn best_for(src: DeviceId, dst: DeviceId, size: usize, cap: &P2PCapability) -> Self {
        if src == dst {
            return Self::Dma; // Intra-device copy
        }

        if cap.peer_access {
            if size >= cap.optimal_copy_size {
                Self::P2P // Large P2P transfers
            } else {
                Self::Dma // Small transfers better with DMA
            }
        } else {
            Self::Dma // Stage through host
        }
    }
}

/// Low-level copy descriptor
#[derive(Debug, Clone)]
pub struct CopyDescriptor {
    /// Source device
    pub src_device: DeviceId,
    /// Source pointer
    pub src_ptr: DevicePtr,
    /// Destination device
    pub dst_device: DeviceId,
    /// Destination pointer
    pub dst_ptr: DevicePtr,
    /// Size in bytes
    pub size: usize,
    /// Which engine to use
    pub engine: CopyEngine,
    /// Stream for async execution
    pub stream: Option<u64>,
    /// Fence before copy
    pub pre_fence: Option<MemFence>,
    /// Fence after copy
    pub post_fence: Option<MemFence>,
}

impl CopyDescriptor {
    /// Create a new copy descriptor
    pub fn new(
        src_device: DeviceId,
        src_ptr: DevicePtr,
        dst_device: DeviceId,
        dst_ptr: DevicePtr,
        size: usize,
    ) -> Self {
        Self {
            src_device,
            src_ptr,
            dst_device,
            dst_ptr,
            size,
            engine: CopyEngine::Dma,
            stream: None,
            pre_fence: None,
            post_fence: None,
        }
    }

    /// Set copy engine
    pub fn with_engine(mut self, engine: CopyEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Set stream
    pub fn with_stream(mut self, stream: u64) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Add pre-fence
    pub fn with_pre_fence(mut self, fence: MemFence) -> Self {
        self.pre_fence = Some(fence);
        self
    }

    /// Add post-fence
    pub fn with_post_fence(mut self, fence: MemFence) -> Self {
        self.post_fence = Some(fence);
        self
    }
}

/// P2P copy implementation
pub struct P2PCopier {
    capabilities: Vec<Vec<P2PCapability>>,
    num_devices: usize,
}

impl P2PCopier {
    pub fn new(devices: &[DeviceId]) -> Result<Self, BufferError> {
        let n = devices.len();
        let mut capabilities = vec![vec![]; n];

        for (i, &src) in devices.iter().enumerate() {
            for &dst in devices.iter() {
                let cap = P2PCapability::query(src, dst)?;
                capabilities[i].push(cap);
            }
        }

        Ok(Self {
            capabilities,
            num_devices: n,
        })
    }

    /// Get number of devices
    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    /// Get capability between devices
    pub fn capability(&self, src: DeviceId, dst: DeviceId) -> &P2PCapability {
        &self.capabilities[src.0 as usize][dst.0 as usize]
    }

    /// Execute copy
    pub async fn copy(&self, desc: &CopyDescriptor) -> Result<(), BufferError> {
        let cap = self.capability(desc.src_device, desc.dst_device);

        // Pre-fence if requested
        if let Some(fence) = &desc.pre_fence {
            self.execute_fence(desc.src_device, fence)?;
        }

        match desc.engine {
            CopyEngine::Compute => self.copy_compute(desc).await,
            CopyEngine::Dma => self.copy_dma(desc).await,
            CopyEngine::P2P => self.copy_p2p(desc, cap).await,
            CopyEngine::Rdma => self.copy_rdma(desc).await,
        }?;

        // Post-fence if requested
        if let Some(fence) = &desc.post_fence {
            self.execute_fence(desc.dst_device, fence)?;
        }

        Ok(())
    }

    /// Kernel-based copy (for maximum flexibility)
    async fn copy_compute(&self, _desc: &CopyDescriptor) -> Result<(), BufferError> {
        // Launch copy kernel
        // Would execute kernel here
        Ok(())
    }

    /// DMA engine copy
    async fn copy_dma(&self, _desc: &CopyDescriptor) -> Result<(), BufferError> {
        // Would call cudaMemcpyAsync or similar
        Ok(())
    }

    /// Direct P2P copy
    async fn copy_p2p(
        &self,
        desc: &CopyDescriptor,
        cap: &P2PCapability,
    ) -> Result<(), BufferError> {
        if !cap.peer_access {
            // Fall back to staged copy through host
            return self.copy_staged(desc).await;
        }

        // Would call cudaMemcpyPeerAsync
        Ok(())
    }

    /// RDMA copy (for network-attached GPUs)
    async fn copy_rdma(&self, _desc: &CopyDescriptor) -> Result<(), BufferError> {
        // Would use GPUDirect RDMA APIs
        Ok(())
    }

    /// Staged copy through host memory
    async fn copy_staged(&self, _desc: &CopyDescriptor) -> Result<(), BufferError> {
        // 1. Allocate host staging buffer
        // 2. Copy src_device -> host
        // 3. Copy host -> dst_device
        // Much slower but always works
        Ok(())
    }

    /// Execute memory fence
    fn execute_fence(&self, _device: DeviceId, _fence: &MemFence) -> Result<(), BufferError> {
        // Would execute fence on device
        Ok(())
    }

    /// Generate optimized copy kernel
    pub fn generate_copy_kernel(&self, bytes_per_thread: usize) -> String {
        let threads_per_block = 256;
        let _bytes_per_block = bytes_per_thread * threads_per_block;

        format!(
            r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry copy_kernel(
    .param .u64 dst,
    .param .u64 src,
    .param .u64 size
) {{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .v4 .u32 %rv<2>;
    .reg .pred %p<2>;

    // Calculate global thread ID
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;  // blockIdx.x * blockDim.x + threadIdx.x

    // Load parameters
    ld.param.u64 %rd0, [dst];
    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [size];

    // Calculate byte offset ({bytes_per_thread} bytes per thread)
    mul.wide.u32 %rd3, %r0, {bytes_per_thread};

    // Check bounds
    setp.ge.u64 %p0, %rd3, %rd2;
    @%p0 bra done;

    // Calculate addresses
    add.u64 %rd4, %rd1, %rd3;  // src + offset
    add.u64 %rd5, %rd0, %rd3;  // dst + offset

    // Load 16 bytes (vectorized)
    ld.global.v4.u32 %rv0, [%rd4];

    // Store 16 bytes
    st.global.v4.u32 [%rd5], %rv0;

done:
    ret;
}}
            "#,
            bytes_per_thread = bytes_per_thread
        )
    }

    /// Generate copy kernel with reduction
    pub fn generate_copy_reduce_kernel(&self, op: &str) -> String {
        format!(
            r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry copy_reduce_kernel(
    .param .u64 dst,
    .param .u64 src,
    .param .u64 size
) {{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    .reg .pred %p<2>;

    // Calculate global thread ID
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;

    // Load parameters
    ld.param.u64 %rd0, [dst];
    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [size];

    // Calculate element index
    mul.wide.u32 %rd3, %r0, 4;  // 4 bytes per float

    // Check bounds
    setp.ge.u64 %p0, %rd3, %rd2;
    @%p0 bra done;

    // Calculate addresses
    add.u64 %rd4, %rd1, %rd3;  // src
    add.u64 %rd5, %rd0, %rd3;  // dst

    // Load from both
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // Reduce
    {op}.f32 %f2, %f0, %f1;

    // Store result
    st.global.f32 [%rd5], %f2;

done:
    ret;
}}
            "#,
            op = op
        )
    }
}

/// Bidirectional exchange (send while receiving)
pub struct Exchange {
    copier: Arc<P2PCopier>,
}

impl Exchange {
    pub fn new(copier: Arc<P2PCopier>) -> Self {
        Self { copier }
    }

    /// Exchange data between two devices
    pub async fn exchange(
        &self,
        device_a: DeviceId,
        ptr_a: DevicePtr,
        device_b: DeviceId,
        ptr_b: DevicePtr,
        size: usize,
    ) -> Result<(), BufferError> {
        let copy_a_to_b = CopyDescriptor::new(device_a, ptr_a, device_b, ptr_b, size)
            .with_engine(CopyEngine::P2P)
            .with_post_fence(MemFence::new(MemoryScope::Device, MemoryOrder::Release));

        let copy_b_to_a = CopyDescriptor::new(device_b, ptr_b, device_a, ptr_a, size)
            .with_engine(CopyEngine::P2P)
            .with_post_fence(MemFence::new(MemoryScope::Device, MemoryOrder::Release));

        // Execute copies (in real implementation, these would run on different streams)
        // For now, execute sequentially - hardware would parallelize via different copy engines
        self.copier.copy(&copy_a_to_b).await?;
        self.copier.copy(&copy_b_to_a).await?;

        Ok(())
    }
}

/// Request trait for async communication
pub trait Request {
    fn is_completed(&self) -> bool;
    fn set_completed(&mut self);
    fn tag(&self) -> u32;
}

/// Send request
pub struct SendRequest {
    pub src: DeviceId,
    pub src_ptr: DevicePtr,
    pub dst: DeviceId,
    pub dst_ptr: DevicePtr,
    pub size: usize,
    pub tag: u32,
    pub completed: bool,
}

impl Request for SendRequest {
    fn is_completed(&self) -> bool {
        self.completed
    }
    fn set_completed(&mut self) {
        self.completed = true;
    }
    fn tag(&self) -> u32 {
        self.tag
    }
}

/// Receive request
pub struct RecvRequest {
    pub src: DeviceId,
    pub dst: DeviceId,
    pub dst_ptr: DevicePtr,
    pub size: usize,
    pub tag: u32,
    pub completed: bool,
}

impl Request for RecvRequest {
    fn is_completed(&self) -> bool {
        self.completed
    }
    fn set_completed(&mut self) {
        self.completed = true;
    }
    fn tag(&self) -> u32 {
        self.tag
    }
}

/// Send-receive primitive (MPI-style)
pub struct SendRecv {
    copier: Arc<P2PCopier>,
}

impl SendRecv {
    pub fn new(copier: Arc<P2PCopier>) -> Self {
        Self { copier }
    }

    /// Non-blocking send
    pub fn isend(
        &self,
        src: DeviceId,
        src_ptr: DevicePtr,
        dst: DeviceId,
        dst_ptr: DevicePtr,
        size: usize,
        tag: u32,
    ) -> SendRequest {
        SendRequest {
            src,
            src_ptr,
            dst,
            dst_ptr,
            size,
            tag,
            completed: false,
        }
    }

    /// Non-blocking receive
    pub fn irecv(
        &self,
        src: DeviceId,
        dst: DeviceId,
        dst_ptr: DevicePtr,
        size: usize,
        tag: u32,
    ) -> RecvRequest {
        RecvRequest {
            src,
            dst,
            dst_ptr,
            size,
            tag,
            completed: false,
        }
    }

    /// Wait for request to complete
    pub async fn wait<R: Request>(&self, request: &mut R) -> Result<(), BufferError> {
        // Would wait on completion flag
        request.set_completed();
        Ok(())
    }

    /// Wait for all requests
    pub async fn wait_all<R: Request>(&self, requests: &mut [R]) -> Result<(), BufferError> {
        for req in requests {
            self.wait(req).await?;
        }
        Ok(())
    }
}

/// Memory region for one-sided communication
#[derive(Debug)]
pub struct MemoryRegion {
    pub device: DeviceId,
    pub base: DevicePtr,
    pub size: usize,
    pub id: u64,
}

impl MemoryRegion {
    pub fn new(device: DeviceId, base: DevicePtr, size: usize, id: u64) -> Self {
        Self {
            device,
            base,
            size,
            id,
        }
    }

    /// Check if offset is within bounds
    pub fn contains(&self, offset: usize, size: usize) -> bool {
        offset + size <= self.size
    }
}

/// Put/Get (one-sided communication)
pub struct OneSided {
    copier: Arc<P2PCopier>,
    /// Memory regions registered for remote access
    regions: Vec<MemoryRegion>,
}

impl OneSided {
    pub fn new(copier: Arc<P2PCopier>) -> Self {
        Self {
            copier,
            regions: Vec::new(),
        }
    }

    /// Register a memory region for remote access
    pub fn register(&mut self, region: MemoryRegion) {
        self.regions.push(region);
    }

    /// Find region by ID and device
    fn find_region(&self, device: DeviceId, region_id: u64) -> Option<&MemoryRegion> {
        self.regions
            .iter()
            .find(|r| r.id == region_id && r.device == device)
    }

    /// Remote put (write to remote memory)
    pub async fn put(
        &self,
        src_device: DeviceId,
        src_ptr: DevicePtr,
        dst_device: DeviceId,
        dst_region_id: u64,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BufferError> {
        let region = self
            .find_region(dst_device, dst_region_id)
            .ok_or_else(|| BufferError::InvalidState("Region not found".into()))?;

        if !region.contains(dst_offset, size) {
            return Err(BufferError::InvalidState("Out of bounds".into()));
        }

        let dst_ptr = region.base.offset(dst_offset);

        let desc = CopyDescriptor::new(src_device, src_ptr, dst_device, dst_ptr, size)
            .with_engine(CopyEngine::P2P);

        self.copier.copy(&desc).await
    }

    /// Remote get (read from remote memory)
    pub async fn get(
        &self,
        dst_device: DeviceId,
        dst_ptr: DevicePtr,
        src_device: DeviceId,
        src_region_id: u64,
        src_offset: usize,
        size: usize,
    ) -> Result<(), BufferError> {
        let region = self
            .find_region(src_device, src_region_id)
            .ok_or_else(|| BufferError::InvalidState("Region not found".into()))?;

        if !region.contains(src_offset, size) {
            return Err(BufferError::InvalidState("Out of bounds".into()));
        }

        let src_ptr = region.base.offset(src_offset);

        let desc = CopyDescriptor::new(src_device, src_ptr, dst_device, dst_ptr, size)
            .with_engine(CopyEngine::P2P);

        self.copier.copy(&desc).await
    }

    /// Atomic fetch-and-add on remote memory
    pub async fn fetch_add(
        &self,
        _target_device: DeviceId,
        _target_region_id: u64,
        _target_offset: usize,
        _value: u64,
    ) -> Result<u64, BufferError> {
        // Would use atomic operation on remote memory
        // Requires peer atomics support
        Ok(0) // Placeholder
    }

    /// Atomic compare-and-swap on remote memory
    pub async fn compare_swap(
        &self,
        _target_device: DeviceId,
        _target_region_id: u64,
        _target_offset: usize,
        expected: u64,
        _desired: u64,
    ) -> Result<u64, BufferError> {
        // Would use CAS on remote memory
        Ok(expected) // Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2p_capability() {
        let cap = P2PCapability::query(DeviceId(0), DeviceId(0)).unwrap();
        assert!(cap.peer_access);
        assert!(cap.peer_atomics);

        let cap = P2PCapability::query(DeviceId(0), DeviceId(1)).unwrap();
        assert!(cap.peer_access);
        // Cross-device atomics usually not supported
        assert!(!cap.peer_atomics);
    }

    #[test]
    fn test_copy_descriptor() {
        let desc = CopyDescriptor::new(
            DeviceId(0),
            DevicePtr(0x1000),
            DeviceId(1),
            DevicePtr(0x2000),
            1024,
        )
        .with_engine(CopyEngine::P2P)
        .with_stream(1);

        assert_eq!(desc.src_device, DeviceId(0));
        assert_eq!(desc.dst_device, DeviceId(1));
        assert_eq!(desc.size, 1024);
        assert_eq!(desc.engine, CopyEngine::P2P);
        assert_eq!(desc.stream, Some(1));
    }

    #[test]
    fn test_p2p_copier() {
        let devices = vec![DeviceId(0), DeviceId(1)];
        let copier = P2PCopier::new(&devices).unwrap();

        assert_eq!(copier.num_devices(), 2);

        let cap = copier.capability(DeviceId(0), DeviceId(1));
        assert!(cap.peer_access);
    }

    #[test]
    fn test_memory_region() {
        let region = MemoryRegion::new(DeviceId(0), DevicePtr(0x1000), 4096, 1);

        assert!(region.contains(0, 1024));
        assert!(region.contains(3000, 1096));
        assert!(!region.contains(4000, 100));
    }

    #[test]
    fn test_send_recv_request() {
        let devices = vec![DeviceId(0), DeviceId(1)];
        let copier = Arc::new(P2PCopier::new(&devices).unwrap());
        let sr = SendRecv::new(copier);

        let mut send = sr.isend(
            DeviceId(0),
            DevicePtr(0x1000),
            DeviceId(1),
            DevicePtr(0x2000),
            1024,
            42,
        );

        assert!(!send.is_completed());
        assert_eq!(send.tag(), 42);

        send.set_completed();
        assert!(send.is_completed());
    }

    #[test]
    fn test_copy_kernel_generation() {
        let devices = vec![DeviceId(0)];
        let copier = P2PCopier::new(&devices).unwrap();

        let kernel = copier.generate_copy_kernel(16);
        assert!(kernel.contains(".entry copy_kernel"));
        assert!(kernel.contains("ld.global.v4.u32"));
        assert!(kernel.contains("st.global.v4.u32"));
    }

    #[test]
    fn test_copy_reduce_kernel_generation() {
        let devices = vec![DeviceId(0)];
        let copier = P2PCopier::new(&devices).unwrap();

        let kernel = copier.generate_copy_reduce_kernel("add");
        assert!(kernel.contains(".entry copy_reduce_kernel"));
        assert!(kernel.contains("add.f32"));
    }
}
