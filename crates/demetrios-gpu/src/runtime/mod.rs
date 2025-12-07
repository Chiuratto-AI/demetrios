//! GPU Runtime
//!
//! Provides device management, memory allocation, streams, and kernel launching.

pub mod buffer;
pub mod device;
pub mod kernel;
pub mod launch;
pub mod stream;

pub use buffer::{BufferError, BufferLocation, GpuBuffer, LinearBuffer, PinnedBuffer, RawBuffer};
pub use device::{
    ComputeCapability, ComputeLimits, Device, DeviceError, DeviceManager, DeviceProperties,
    DeviceType, MemoryInfo,
};
pub use kernel::{Kernel, KernelCache, KernelError, KernelMetadata, KernelModule};
pub use launch::{
    compute_launch_config, Dim3, KernelArg, KernelLauncher, LaunchConfig, LaunchError,
    OccupancyCalculator,
};
pub use stream::{Event, EventTimer, Stream, StreamError, StreamFlags, StreamPool, StreamPriority};
