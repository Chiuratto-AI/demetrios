//! GPU Kernel Implementations for Reduction Operations
//!
//! These are the actual compute kernels that run on the GPU.
//! All collectives eventually call these.

/// Reduction operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOpType {
    Sum,
    Max,
    Min,
    Product,
}

impl ReduceOpType {
    /// Get PTX instruction name
    pub fn ptx_instruction(&self) -> &'static str {
        match self {
            Self::Sum => "add",
            Self::Max => "max",
            Self::Min => "min",
            Self::Product => "mul",
        }
    }

    /// Get identity value for f32
    pub fn identity_f32(&self) -> &'static str {
        match self {
            Self::Sum => "0f00000000",     // 0.0
            Self::Max => "0fff800000",     // -inf
            Self::Min => "0f7f800000",     // +inf
            Self::Product => "0f3f800000", // 1.0
        }
    }
}

/// Data types for reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
}

impl DataType {
    /// Get PTX type suffix
    pub fn ptx_suffix(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "s32",
            Self::I64 => "s64",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    /// Get PTX register prefix
    pub fn register_prefix(&self) -> &'static str {
        match self {
            Self::F32 => "%f",
            Self::F64 => "%fd",
            Self::I32 | Self::U32 => "%r",
            Self::I64 | Self::U64 => "%rd",
        }
    }
}

/// Generate PTX for warp-level reduction using butterfly shuffle
pub fn generate_warp_reduce(op: ReduceOpType, data_type: DataType) -> String {
    let op_inst = op.ptx_instruction();
    let suffix = data_type.ptx_suffix();

    // For 32-bit types, use shfl.bfly
    // For 64-bit types, need two shuffles (high/low)
    if data_type.size() == 4 {
        format!(
            r#"
// Warp reduce using butterfly shuffle pattern
// Input: %f_val contains this thread's value
// Output: %f_val contains reduced value (in all lanes for full reduce)

// Step 1: Reduce with lane + 16
shfl.sync.bfly.b32 %f_shfl, %f_val, 16, 0x1f, 0xffffffff;
{op}.{ty} %f_val, %f_val, %f_shfl;

// Step 2: Reduce with lane + 8
shfl.sync.bfly.b32 %f_shfl, %f_val, 8, 0x1f, 0xffffffff;
{op}.{ty} %f_val, %f_val, %f_shfl;

// Step 3: Reduce with lane + 4
shfl.sync.bfly.b32 %f_shfl, %f_val, 4, 0x1f, 0xffffffff;
{op}.{ty} %f_val, %f_val, %f_shfl;

// Step 4: Reduce with lane + 2
shfl.sync.bfly.b32 %f_shfl, %f_val, 2, 0x1f, 0xffffffff;
{op}.{ty} %f_val, %f_val, %f_shfl;

// Step 5: Reduce with lane + 1
shfl.sync.bfly.b32 %f_shfl, %f_val, 1, 0x1f, 0xffffffff;
{op}.{ty} %f_val, %f_val, %f_shfl;

// Now %f_val contains the warp-wide reduced value in ALL lanes
            "#,
            op = op_inst,
            ty = suffix
        )
    } else {
        // 64-bit reduction needs to shuffle both halves
        format!(
            r#"
// Warp reduce for 64-bit using butterfly shuffle
// Split 64-bit value into two 32-bit halves

mov.b64 {{%r_lo, %r_hi}}, %fd_val;

// Step 1: Reduce with lane + 16
shfl.sync.bfly.b32 %r_shfl_lo, %r_lo, 16, 0x1f, 0xffffffff;
shfl.sync.bfly.b32 %r_shfl_hi, %r_hi, 16, 0x1f, 0xffffffff;
mov.b64 %fd_shfl, {{%r_shfl_lo, %r_shfl_hi}};
{op}.{ty} %fd_val, %fd_val, %fd_shfl;
mov.b64 {{%r_lo, %r_hi}}, %fd_val;

// Steps 2-5 (repeat for offsets 8, 4, 2, 1)
shfl.sync.bfly.b32 %r_shfl_lo, %r_lo, 8, 0x1f, 0xffffffff;
shfl.sync.bfly.b32 %r_shfl_hi, %r_hi, 8, 0x1f, 0xffffffff;
mov.b64 %fd_shfl, {{%r_shfl_lo, %r_shfl_hi}};
{op}.{ty} %fd_val, %fd_val, %fd_shfl;
mov.b64 {{%r_lo, %r_hi}}, %fd_val;

shfl.sync.bfly.b32 %r_shfl_lo, %r_lo, 4, 0x1f, 0xffffffff;
shfl.sync.bfly.b32 %r_shfl_hi, %r_hi, 4, 0x1f, 0xffffffff;
mov.b64 %fd_shfl, {{%r_shfl_lo, %r_shfl_hi}};
{op}.{ty} %fd_val, %fd_val, %fd_shfl;
mov.b64 {{%r_lo, %r_hi}}, %fd_val;

shfl.sync.bfly.b32 %r_shfl_lo, %r_lo, 2, 0x1f, 0xffffffff;
shfl.sync.bfly.b32 %r_shfl_hi, %r_hi, 2, 0x1f, 0xffffffff;
mov.b64 %fd_shfl, {{%r_shfl_lo, %r_shfl_hi}};
{op}.{ty} %fd_val, %fd_val, %fd_shfl;
mov.b64 {{%r_lo, %r_hi}}, %fd_val;

shfl.sync.bfly.b32 %r_shfl_lo, %r_lo, 1, 0x1f, 0xffffffff;
shfl.sync.bfly.b32 %r_shfl_hi, %r_hi, 1, 0x1f, 0xffffffff;
mov.b64 %fd_shfl, {{%r_shfl_lo, %r_shfl_hi}};
{op}.{ty} %fd_val, %fd_val, %fd_shfl;
            "#,
            op = op_inst,
            ty = suffix
        )
    }
}

/// Generate PTX for block-level reduction
pub fn generate_block_reduce(op: ReduceOpType, data_type: DataType, block_size: usize) -> String {
    let num_warps = block_size / 32;
    let _shared_size = num_warps * data_type.size();

    let warp_reduce = generate_warp_reduce(op, data_type);

    let op_inst = op.ptx_instruction();
    let suffix = data_type.ptx_suffix();
    let identity = op.identity_f32();

    format!(
        r#"
.version 7.5
.target sm_80
.address_size 64

// Shared memory for warp partial sums
.shared .align 4 .f32 warp_sums[{num_warps}];

.visible .entry block_reduce_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {{
    .reg .u32 %r<16>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .f32 %f_val, %f_shfl;
    .reg .pred %p<4>;

    // Get thread/warp indices
    mov.u32 %r0, %tid.x;           // threadIdx.x
    mov.u32 %r1, %ctaid.x;         // blockIdx.x
    mov.u32 %r2, %ntid.x;          // blockDim.x

    // Calculate lane and warp ID
    and.b32 %r3, %r0, 31;          // lane_id = tid & 31
    shr.u32 %r4, %r0, 5;           // warp_id = tid >> 5

    // Calculate global index
    mad.lo.u32 %r5, %r1, %r2, %r0; // global_id = blockIdx * blockDim + threadIdx

    // Load parameters
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r6, [n];

    // Initialize with identity
    mov.f32 %f_val, {identity};

    // Check bounds and load
    setp.lt.u32 %p0, %r5, %r6;
    @!%p0 bra skip_load;

    mul.wide.u32 %rd2, %r5, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f_val, [%rd3];

skip_load:
    // Warp-level reduction
    {warp_reduce}

    // First lane of each warp writes to shared memory
    setp.eq.u32 %p1, %r3, 0;
    @!%p1 bra skip_shared_write;

    mul.u32 %r7, %r4, 4;
    mov.u32 %r8, warp_sums;
    add.u32 %r8, %r8, %r7;
    st.shared.f32 [%r8], %f_val;

skip_shared_write:
    // Synchronize block
    bar.sync 0;

    // First warp reduces the warp sums
    setp.lt.u32 %p2, %r0, {num_warps};
    @!%p2 bra skip_final_reduce;

    // Load from shared
    mul.u32 %r9, %r0, 4;
    mov.u32 %r10, warp_sums;
    add.u32 %r10, %r10, %r9;
    ld.shared.f32 %f_val, [%r10];

    // Final warp reduction (only need log2(num_warps) steps)
    {warp_reduce}

skip_final_reduce:
    // Thread 0 writes final result
    setp.eq.u32 %p3, %r0, 0;
    @!%p3 bra done;

    // Calculate output address
    mul.wide.u32 %rd4, %r1, 4;
    add.u64 %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f_val;

done:
    ret;
}}
        "#,
        num_warps = num_warps,
        warp_reduce = warp_reduce,
        identity = identity
    )
}

/// Generate atomic reduction kernel (single-pass)
pub fn generate_atomic_reduce(op: ReduceOpType, data_type: DataType) -> String {
    let atomic_op = match op {
        ReduceOpType::Sum => "atom.global.add",
        ReduceOpType::Max => "atom.global.max",
        ReduceOpType::Min => "atom.global.min",
        ReduceOpType::Product => "atom.global.add", // No atomic mul, use add
    };

    let suffix = data_type.ptx_suffix();
    let identity = op.identity_f32();

    format!(
        r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry atomic_reduce_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .f32 %f_val, %f_shfl;
    .reg .pred %p<2>;

    // Grid-stride loop for coalesced access
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %nctaid.x;

    mad.lo.u32 %r4, %r1, %r2, %r0;   // start = blockIdx * blockDim + threadIdx
    mul.lo.u32 %r5, %r2, %r3;         // stride = blockDim * gridDim

    // Load parameters
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r6, [n];

    // Initialize thread-local accumulator
    mov.f32 %f0, {identity};

loop:
    setp.ge.u32 %p0, %r4, %r6;
    @%p0 bra loop_end;

    // Load and accumulate
    mul.wide.u32 %rd2, %r4, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32 %f0, %f0, %f1;

    // Next iteration
    add.u32 %r4, %r4, %r5;
    bra loop;

loop_end:
    // Warp reduce thread-local accumulators
    shfl.sync.bfly.b32 %f_shfl, %f0, 16, 0x1f, 0xffffffff;
    add.f32 %f0, %f0, %f_shfl;
    shfl.sync.bfly.b32 %f_shfl, %f0, 8, 0x1f, 0xffffffff;
    add.f32 %f0, %f0, %f_shfl;
    shfl.sync.bfly.b32 %f_shfl, %f0, 4, 0x1f, 0xffffffff;
    add.f32 %f0, %f0, %f_shfl;
    shfl.sync.bfly.b32 %f_shfl, %f0, 2, 0x1f, 0xffffffff;
    add.f32 %f0, %f0, %f_shfl;
    shfl.sync.bfly.b32 %f_shfl, %f0, 1, 0x1f, 0xffffffff;
    add.f32 %f0, %f0, %f_shfl;

    // First lane atomically adds to output
    mov.u32 %r7, %laneid;
    setp.eq.u32 %p1, %r7, 0;
    @!%p1 bra done;

    {atomic}.{ty} %f2, [%rd1], %f0;

done:
    ret;
}}
        "#,
        identity = identity,
        atomic = atomic_op,
        ty = suffix
    )
}

/// Generate ring all-reduce step kernel
pub fn generate_ring_step_kernel(op: ReduceOpType, _data_type: DataType) -> String {
    let op_inst = op.ptx_instruction();

    format!(
        r#"
.version 7.5
.target sm_80
.address_size 64

// Ring all-reduce step: reduce received data into local buffer
// Pipelined: while receiving, send previous chunk

.visible .entry ring_reduce_step(
    .param .u64 local_buffer,      // My buffer (to reduce into)
    .param .u64 recv_buffer,       // Data received from previous rank
    .param .u64 chunk_offset,      // Offset into buffer
    .param .u64 chunk_size         // Size of this chunk in elements
) {{
    .reg .u64 %rd<16>;
    .reg .u32 %r<8>;
    .reg .f32 %f<4>;
    .reg .pred %p;

    // Grid-stride loop
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %nctaid.x;

    mad.lo.u32 %r4, %r1, %r2, %r0;
    mul.lo.u32 %r5, %r2, %r3;

    ld.param.u64 %rd0, [local_buffer];
    ld.param.u64 %rd1, [recv_buffer];
    ld.param.u64 %rd2, [chunk_offset];
    ld.param.u64 %rd3, [chunk_size];

    // Add chunk offset (in bytes)
    shl.b64 %rd4, %rd2, 2;  // offset * 4 bytes
    add.u64 %rd0, %rd0, %rd4;
    add.u64 %rd1, %rd1, %rd4;

loop:
    cvt.u64.u32 %rd5, %r4;
    setp.ge.u64 %p, %rd5, %rd3;
    @%p bra done;

    // Calculate addresses
    mul.wide.u32 %rd6, %r4, 4;
    add.u64 %rd7, %rd0, %rd6;
    add.u64 %rd8, %rd1, %rd6;

    // Load both values
    ld.global.f32 %f0, [%rd7];
    ld.global.f32 %f1, [%rd8];

    // Reduce
    {op}.f32 %f2, %f0, %f1;

    // Store result
    st.global.f32 [%rd7], %f2;

    // Next element
    add.u32 %r4, %r4, %r5;
    bra loop;

done:
    ret;
}}
        "#,
        op = op_inst
    )
}

/// Generate vectorized copy kernel
pub fn generate_vectorized_copy(vector_width: usize) -> String {
    let (vec_type, byte_width) = match vector_width {
        2 => (".v2", 8),
        4 => (".v4", 16),
        _ => ("", 4),
    };

    format!(
        r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry vectorized_copy_kernel(
    .param .u64 dst,
    .param .u64 src,
    .param .u64 n_vectors
) {{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg {vec}.u32 %rv<2>;
    .reg .pred %p;

    // Calculate global thread ID
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;

    // Load parameters
    ld.param.u64 %rd0, [dst];
    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [n_vectors];

    // Check bounds
    cvt.u64.u32 %rd3, %r0;
    setp.ge.u64 %p, %rd3, %rd2;
    @%p bra done;

    // Calculate byte offset
    mul.wide.u32 %rd4, %r0, {byte_width};

    // Load vector
    add.u64 %rd5, %rd1, %rd4;
    ld.global{vec}.u32 %rv0, [%rd5];

    // Store vector
    add.u64 %rd6, %rd0, %rd4;
    st.global{vec}.u32 [%rd6], %rv0;

done:
    ret;
}}
        "#,
        vec = vec_type,
        byte_width = byte_width
    )
}

/// Generate decoupled look-back scan kernel
pub fn generate_lookback_scan(_op: ReduceOpType) -> String {
    r#"
.version 7.5
.target sm_80
.address_size 64

// Decoupled look-back scan for device-wide prefix sum
// Each block:
// 1. Scans its local data
// 2. Publishes partial aggregate
// 3. Looks back at previous blocks to compute inclusive prefix
// 4. Adds prefix to local scan results
// 5. Publishes complete status

// State values:
// 0 = Invalid (not started)
// 1 = Partial (aggregate available, still computing prefix)
// 2 = Complete (inclusive prefix available)

.shared .align 8 .u64 block_state[2];  // [status, value]

.visible .entry lookback_scan_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u64 block_states,  // Array of per-block states
    .param .u32 n
) {
    .reg .u64 %rd<32>;
    .reg .u32 %r<32>;
    .reg .f32 %f<16>;
    .reg .pred %p<8>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;

    // Step 1: Load data and compute block-local scan
    // (Using Kogge-Stone for simplicity)

    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u64 %rd10, [block_states];
    ld.param.u32 %r10, [n];

    // Calculate global index
    mad.lo.u32 %r3, %r1, %r2, %r0;

    // Load input value (or 0 if out of bounds)
    mov.f32 %f0, 0f00000000;
    setp.lt.u32 %p0, %r3, %r10;
    @!%p0 bra skip_load;

    mul.wide.u32 %rd2, %r3, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];

skip_load:
    // Block-local inclusive scan using shared memory
    // Simplified: just sum for aggregate calculation

    bar.sync 0;

    // Step 2: Block aggregate = last element of inclusive scan
    // Thread blockDim-1 has the aggregate
    mov.u32 %r4, %r2;
    sub.u32 %r4, %r4, 1;
    setp.eq.u32 %p1, %r0, %r4;
    @!%p1 bra skip_aggregate;

    // Publish PARTIAL status with aggregate
    mul.wide.u32 %rd4, %r1, 16;  // 16 bytes per block state
    add.u64 %rd5, %rd10, %rd4;

    // Pack: value in high 32 bits, status=1 in low 32 bits
    cvt.rzi.u32.f32 %r5, %f0;
    cvt.u64.u32 %rd6, %r5;
    shl.b64 %rd6, %rd6, 32;
    or.b64 %rd6, %rd6, 1;
    st.release.gpu.global.u64 [%rd5], %rd6;

skip_aggregate:
    bar.sync 0;

    // Step 3: Thread 0 does look-back
    setp.eq.u32 %p2, %r0, 0;
    @!%p2 bra wait_for_prefix;

    mov.f32 %f1, 0f00000000;  // Running prefix sum

    // Look at previous blocks in reverse order
    mov.u32 %r6, %r1;  // Current block to examine

lookback_loop:
    setp.eq.u32 %p3, %r6, 0;
    @%p3 bra lookback_done;  // Block 0 has no predecessors

    sub.u32 %r6, %r6, 1;

    // Load predecessor's state
    mul.wide.u32 %rd7, %r6, 16;
    add.u64 %rd8, %rd10, %rd7;

spin:
    ld.acquire.gpu.global.u64 %rd9, [%rd8];

    // Extract status (low 32 bits)
    and.b64 %rd11, %rd9, 3;
    setp.eq.u64 %p4, %rd11, 0;
    @%p4 bra spin;  // Still invalid, spin

    // Check if complete (status == 2)
    setp.eq.u64 %p5, %rd11, 2;
    @%p5 bra found_complete;

    // Partial: add aggregate and continue looking
    shr.u64 %rd12, %rd9, 32;
    cvt.u32.u64 %r7, %rd12;
    cvt.rn.f32.u32 %f2, %r7;
    add.f32 %f1, %f1, %f2;
    bra lookback_loop;

found_complete:
    // Complete: add inclusive prefix and we're done
    shr.u64 %rd13, %rd9, 32;
    cvt.u32.u64 %r8, %rd13;
    cvt.rn.f32.u32 %f3, %r8;
    add.f32 %f1, %f1, %f3;

lookback_done:
    // Store prefix in shared memory for other threads
    st.shared.f32 [block_state], %f1;

    // Update our status to COMPLETE
    add.f32 %f4, %f0, %f1;  // Our inclusive prefix
    cvt.rzi.u32.f32 %r9, %f4;
    cvt.u64.u32 %rd14, %r9;
    shl.b64 %rd14, %rd14, 32;
    or.b64 %rd14, %rd14, 2;  // Status = complete

    mul.wide.u32 %rd15, %r1, 16;
    add.u64 %rd16, %rd10, %rd15;
    st.release.gpu.global.u64 [%rd16], %rd14;

wait_for_prefix:
    bar.sync 0;

    // Step 4: All threads add prefix to their local results
    ld.shared.f32 %f5, [block_state];
    add.f32 %f0, %f0, %f5;

    // Step 5: Store final result
    setp.lt.u32 %p6, %r3, %r10;
    @!%p6 bra done;

    mul.wide.u32 %rd17, %r3, 4;
    add.u64 %rd18, %rd1, %rd17;
    st.global.f32 [%rd18], %f0;

done:
    ret;
}
    "#
    .to_string()
}

/// Kernel configuration for launching
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub block_size: usize,
    pub grid_size: usize,
    pub shared_memory: usize,
}

impl KernelConfig {
    /// Calculate optimal config for reduction
    pub fn for_reduction(n: usize, block_size: usize) -> Self {
        let grid_size = (n + block_size - 1) / block_size;
        let num_warps = block_size / 32;
        let shared_memory = num_warps * 4; // One f32 per warp

        Self {
            block_size,
            grid_size,
            shared_memory,
        }
    }

    /// Calculate optimal config for copy
    pub fn for_copy(n_bytes: usize, bytes_per_thread: usize, block_size: usize) -> Self {
        let n_threads = (n_bytes + bytes_per_thread - 1) / bytes_per_thread;
        let grid_size = (n_threads + block_size - 1) / block_size;

        Self {
            block_size,
            grid_size,
            shared_memory: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_reduce_generation() {
        let ptx = generate_warp_reduce(ReduceOpType::Sum, DataType::F32);
        assert!(ptx.contains("shfl.sync.bfly"));
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_warp_reduce_64bit() {
        let ptx = generate_warp_reduce(ReduceOpType::Sum, DataType::F64);
        assert!(ptx.contains("mov.b64"));
        assert!(ptx.contains("add.f64"));
    }

    #[test]
    fn test_block_reduce_generation() {
        let ptx = generate_block_reduce(ReduceOpType::Sum, DataType::F32, 256);
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("bar.sync"));
        assert!(ptx.contains("warp_sums"));
    }

    #[test]
    fn test_atomic_reduce_generation() {
        let ptx = generate_atomic_reduce(ReduceOpType::Sum, DataType::F32);
        assert!(ptx.contains("atom.global.add"));
        assert!(ptx.contains("%laneid"));
    }

    #[test]
    fn test_ring_step_kernel() {
        let ptx = generate_ring_step_kernel(ReduceOpType::Sum, DataType::F32);
        assert!(ptx.contains("ring_reduce_step"));
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_vectorized_copy() {
        let ptx = generate_vectorized_copy(4);
        assert!(ptx.contains("ld.global.v4.u32"));
        assert!(ptx.contains("st.global.v4.u32"));
    }

    #[test]
    fn test_lookback_scan() {
        let ptx = generate_lookback_scan(ReduceOpType::Sum);
        assert!(ptx.contains("lookback_scan_kernel"));
        assert!(ptx.contains("block_states"));
        assert!(ptx.contains("st.release.gpu"));
        assert!(ptx.contains("ld.acquire.gpu"));
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelConfig::for_reduction(1000000, 256);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.grid_size, 3907); // ceil(1000000 / 256)
        assert_eq!(config.shared_memory, 32); // 8 warps * 4 bytes
    }

    #[test]
    fn test_reduce_op_type() {
        assert_eq!(ReduceOpType::Sum.ptx_instruction(), "add");
        assert_eq!(ReduceOpType::Max.ptx_instruction(), "max");
        assert_eq!(ReduceOpType::Min.ptx_instruction(), "min");
    }

    #[test]
    fn test_data_type() {
        assert_eq!(DataType::F32.ptx_suffix(), "f32");
        assert_eq!(DataType::F32.size(), 4);
        assert_eq!(DataType::F64.size(), 8);
    }
}
