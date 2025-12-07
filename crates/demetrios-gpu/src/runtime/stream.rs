//! GPU Streams and Events
//!
//! Provides stream abstraction for concurrent kernel execution and synchronization.

use super::device::{Device, DeviceError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Stream-related errors
#[derive(Debug, Error)]
pub enum StreamError {
    #[error("Stream creation failed: {0}")]
    CreationFailed(String),

    #[error("Stream synchronization failed: {0}")]
    SyncFailed(String),

    #[error("Event error: {0}")]
    EventError(String),

    #[error("Device error: {0}")]
    Device(#[from] DeviceError),
}

/// Stream priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StreamPriority {
    High,
    #[default]
    Normal,
    Low,
}

/// Stream flags
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamFlags {
    /// Non-blocking stream (operations don't block on default stream)
    pub non_blocking: bool,
}

/// Counter for generating unique stream IDs
static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// GPU stream handle
#[derive(Debug)]
pub struct Stream {
    /// Unique stream ID
    id: u64,
    /// Device index
    device_index: usize,
    /// Priority
    priority: StreamPriority,
    /// Flags
    flags: StreamFlags,
    /// Whether this is the default stream
    is_default: bool,
    /// Pending operations count (for simulation)
    pending_ops: Arc<AtomicU64>,
}

impl Stream {
    /// Create a new stream
    pub fn new(device: &Device) -> Result<Self, StreamError> {
        Self::with_flags(device, StreamFlags::default())
    }

    /// Create a stream with specific flags
    pub fn with_flags(device: &Device, flags: StreamFlags) -> Result<Self, StreamError> {
        let id = STREAM_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        Ok(Stream {
            id,
            device_index: device.index(),
            priority: StreamPriority::Normal,
            flags,
            is_default: false,
            pending_ops: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Create a stream with priority
    pub fn with_priority(device: &Device, priority: StreamPriority) -> Result<Self, StreamError> {
        let mut stream = Self::new(device)?;
        stream.priority = priority;
        Ok(stream)
    }

    /// Create the default stream for a device
    pub fn default_stream(device: &Device) -> Self {
        Stream {
            id: 0,
            device_index: device.index(),
            priority: StreamPriority::Normal,
            flags: StreamFlags::default(),
            is_default: true,
            pending_ops: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get stream ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Get priority
    pub fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Check if this is the default stream
    pub fn is_default(&self) -> bool {
        self.is_default
    }

    /// Synchronize the stream (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<(), StreamError> {
        // For CPU simulation, just wait until pending_ops is 0
        while self.pending_ops.load(Ordering::SeqCst) > 0 {
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Check if all operations on this stream have completed
    pub fn is_complete(&self) -> bool {
        self.pending_ops.load(Ordering::SeqCst) == 0
    }

    /// Record an event on this stream
    pub fn record_event(&self) -> Result<Event, StreamError> {
        let event = Event::new(self.device_index)?;
        // In a real implementation, this would record the event on the stream
        Ok(event)
    }

    /// Wait for an event
    pub fn wait_event(&self, event: &Event) -> Result<(), StreamError> {
        // For CPU simulation, just check if event is recorded
        if !event.is_recorded() {
            return Err(StreamError::EventError("Event not recorded".to_string()));
        }
        Ok(())
    }

    /// Begin an operation (for tracking)
    pub fn begin_operation(&self) {
        self.pending_ops.fetch_add(1, Ordering::SeqCst);
    }

    /// End an operation (for tracking)
    pub fn end_operation(&self) {
        self.pending_ops.fetch_sub(1, Ordering::SeqCst);
    }

    /// Get pending operations count
    pub fn pending_operations(&self) -> u64 {
        self.pending_ops.load(Ordering::SeqCst)
    }
}

/// Counter for generating unique event IDs
static EVENT_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Event flags
#[derive(Debug, Clone, Copy, Default)]
pub struct EventFlags {
    /// Disable timing (faster but no elapsed time)
    pub disable_timing: bool,
    /// Interprocess event
    pub interprocess: bool,
}

/// GPU event for synchronization and timing
#[derive(Debug)]
pub struct Event {
    /// Unique event ID
    id: u64,
    /// Device index
    device_index: usize,
    /// Flags
    flags: EventFlags,
    /// Record time (for timing)
    record_time: Option<Instant>,
    /// Whether event has been recorded
    recorded: bool,
}

impl Event {
    /// Create a new event
    pub fn new(device_index: usize) -> Result<Self, StreamError> {
        Self::with_flags(device_index, EventFlags::default())
    }

    /// Create an event with specific flags
    pub fn with_flags(device_index: usize, flags: EventFlags) -> Result<Self, StreamError> {
        let id = EVENT_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        Ok(Event {
            id,
            device_index,
            flags,
            record_time: None,
            recorded: false,
        })
    }

    /// Get event ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Record the event (mark current point in stream)
    pub fn record(&mut self) {
        self.record_time = Some(Instant::now());
        self.recorded = true;
    }

    /// Check if event has been recorded
    pub fn is_recorded(&self) -> bool {
        self.recorded
    }

    /// Synchronize (wait for event to complete)
    pub fn synchronize(&self) -> Result<(), StreamError> {
        if !self.recorded {
            return Err(StreamError::EventError("Event not recorded".to_string()));
        }
        // For CPU simulation, event is always complete immediately
        Ok(())
    }

    /// Check if event has completed
    pub fn is_complete(&self) -> Result<bool, StreamError> {
        Ok(self.recorded)
    }

    /// Get elapsed time between this event and another
    pub fn elapsed_since(&self, start: &Event) -> Result<Duration, StreamError> {
        if self.flags.disable_timing || start.flags.disable_timing {
            return Err(StreamError::EventError(
                "Timing disabled for one or both events".to_string(),
            ));
        }

        match (start.record_time, self.record_time) {
            (Some(start_time), Some(end_time)) => {
                if end_time < start_time {
                    return Err(StreamError::EventError(
                        "End event recorded before start event".to_string(),
                    ));
                }
                Ok(end_time - start_time)
            }
            _ => Err(StreamError::EventError(
                "One or both events not recorded".to_string(),
            )),
        }
    }
}

/// Stream pool for managing multiple streams
pub struct StreamPool {
    streams: Vec<Stream>,
    device_index: usize,
}

impl StreamPool {
    /// Create a new stream pool
    pub fn new(device: &Device, count: usize) -> Result<Self, StreamError> {
        let mut streams = Vec::with_capacity(count);
        for _ in 0..count {
            streams.push(Stream::new(device)?);
        }

        Ok(StreamPool {
            streams,
            device_index: device.index(),
        })
    }

    /// Get the number of streams
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Get a stream by index
    pub fn get(&self, index: usize) -> Option<&Stream> {
        self.streams.get(index)
    }

    /// Get a stream round-robin style
    pub fn get_round_robin(&self, counter: usize) -> &Stream {
        &self.streams[counter % self.streams.len()]
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> Result<(), StreamError> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Get the least busy stream
    pub fn get_least_busy(&self) -> &Stream {
        self.streams
            .iter()
            .min_by_key(|s| s.pending_operations())
            .unwrap_or(&self.streams[0])
    }
}

/// Scoped timer using events
pub struct EventTimer {
    start: Event,
    end: Event,
}

impl EventTimer {
    /// Create a new event timer
    pub fn new(device_index: usize) -> Result<Self, StreamError> {
        Ok(EventTimer {
            start: Event::new(device_index)?,
            end: Event::new(device_index)?,
        })
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start.record();
    }

    /// Stop timing
    pub fn stop(&mut self) {
        self.end.record();
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Result<Duration, StreamError> {
        self.end.elapsed_since(&self.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        let device = Device::cpu();
        let stream = Stream::new(&device).unwrap();
        assert!(!stream.is_default());
        assert!(stream.is_complete());
    }

    #[test]
    fn test_default_stream() {
        let device = Device::cpu();
        let stream = Stream::default_stream(&device);
        assert!(stream.is_default());
    }

    #[test]
    fn test_event_timing() {
        let device = Device::cpu();
        let mut start = Event::new(device.index()).unwrap();
        let mut end = Event::new(device.index()).unwrap();

        start.record();
        std::thread::sleep(Duration::from_millis(10));
        end.record();

        let elapsed = end.elapsed_since(&start).unwrap();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_stream_pool() {
        let device = Device::cpu();
        let pool = StreamPool::new(&device, 4).unwrap();
        assert_eq!(pool.len(), 4);

        let s0 = pool.get_round_robin(0);
        let s4 = pool.get_round_robin(4);
        assert_eq!(s0.id(), s4.id());
    }

    #[test]
    fn test_event_timer() {
        let device = Device::cpu();
        let mut timer = EventTimer::new(device.index()).unwrap();

        timer.start();
        std::thread::sleep(Duration::from_millis(5));
        timer.stop();

        let elapsed = timer.elapsed().unwrap();
        assert!(elapsed >= Duration::from_millis(5));
    }
}
