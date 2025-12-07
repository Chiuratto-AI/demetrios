//! Epistemic Types Integration for GPU
//!
//! Provides confidence-tracked GPU buffers for uncertainty quantification.

use crate::runtime::{BufferError, Device, GpuBuffer};
use std::marker::PhantomData;

/// Confidence level for epistemic values
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f64);

impl Confidence {
    /// Create a new confidence level (clamped to [0, 1])
    pub fn new(value: f64) -> Self {
        Confidence(value.max(0.0).min(1.0))
    }

    /// Get the confidence value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Full confidence (1.0)
    pub fn certain() -> Self {
        Confidence(1.0)
    }

    /// No confidence (0.0)
    pub fn uncertain() -> Self {
        Confidence(0.0)
    }

    /// Default confidence for observations (0.95)
    pub fn observed() -> Self {
        Confidence(0.95)
    }

    /// Default confidence for computations (0.99)
    pub fn computed() -> Self {
        Confidence(0.99)
    }

    /// Combine two confidence levels (multiplication)
    pub fn combine(&self, other: &Confidence) -> Confidence {
        Confidence(self.0 * other.0)
    }

    /// Meet operation (minimum)
    pub fn meet(&self, other: &Confidence) -> Confidence {
        Confidence(self.0.min(other.0))
    }

    /// Join operation (maximum)
    pub fn join(&self, other: &Confidence) -> Confidence {
        Confidence(self.0.max(other.0))
    }

    /// Check if above threshold
    pub fn above_threshold(&self, threshold: f64) -> bool {
        self.0 >= threshold
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Confidence::certain()
    }
}

/// Source of epistemic knowledge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnowledgeSource {
    /// Direct observation/measurement
    Observation,
    /// Computed from other values
    Computation,
    /// From a model/simulation
    Simulation,
    /// From inference/interpolation
    Inference,
    /// From external source
    External,
    /// Prior knowledge/assumption
    Prior,
}

impl KnowledgeSource {
    /// Get default confidence for this source
    pub fn default_confidence(&self) -> Confidence {
        match self {
            KnowledgeSource::Observation => Confidence::new(0.95),
            KnowledgeSource::Computation => Confidence::new(0.99),
            KnowledgeSource::Simulation => Confidence::new(0.85),
            KnowledgeSource::Inference => Confidence::new(0.80),
            KnowledgeSource::External => Confidence::new(0.70),
            KnowledgeSource::Prior => Confidence::new(0.50),
        }
    }
}

/// Epistemic value with confidence tracking
#[derive(Debug, Clone, Copy)]
pub struct Epistemic<T> {
    value: T,
    confidence: Confidence,
    source: KnowledgeSource,
}

impl<T: Copy> Epistemic<T> {
    /// Create a new epistemic value
    pub fn new(value: T, confidence: Confidence, source: KnowledgeSource) -> Self {
        Epistemic {
            value,
            confidence,
            source,
        }
    }

    /// Create an observed value
    pub fn observed(value: T) -> Self {
        Self::new(value, Confidence::observed(), KnowledgeSource::Observation)
    }

    /// Create a computed value
    pub fn computed(value: T) -> Self {
        Self::new(value, Confidence::computed(), KnowledgeSource::Computation)
    }

    /// Create a certain value (ground truth)
    pub fn certain(value: T) -> Self {
        Self::new(value, Confidence::certain(), KnowledgeSource::Computation)
    }

    /// Get the value
    pub fn value(&self) -> T {
        self.value
    }

    /// Get the confidence
    pub fn confidence(&self) -> Confidence {
        self.confidence
    }

    /// Get the source
    pub fn source(&self) -> KnowledgeSource {
        self.source
    }

    /// Map over the value, preserving confidence
    pub fn map<U: Copy>(self, f: impl FnOnce(T) -> U) -> Epistemic<U> {
        Epistemic {
            value: f(self.value),
            confidence: self.confidence,
            source: KnowledgeSource::Computation,
        }
    }

    /// Update confidence
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Check if value meets confidence threshold
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.confidence.above_threshold(threshold)
    }
}

/// Epistemic GPU buffer with per-element confidence tracking
#[derive(Debug)]
pub struct EpistemicBuffer<T: Copy> {
    /// Value buffer
    values: GpuBuffer<T>,
    /// Confidence buffer (one per element)
    confidences: GpuBuffer<f64>,
    /// Primary source of this buffer's data
    source: KnowledgeSource,
}

impl<T: Copy + Default> EpistemicBuffer<T> {
    /// Create a new epistemic buffer
    pub fn new(len: usize, device: &Device, source: KnowledgeSource) -> Result<Self, BufferError> {
        let values = GpuBuffer::new(len, device)?;
        let confidences = GpuBuffer::new(len, device)?;

        Ok(EpistemicBuffer {
            values,
            confidences,
            source,
        })
    }

    /// Create from epistemic values
    pub fn from_epistemic(data: &[Epistemic<T>], device: &Device) -> Result<Self, BufferError> {
        let values: Vec<T> = data.iter().map(|e| e.value()).collect();
        let confidences: Vec<f64> = data.iter().map(|e| e.confidence().value()).collect();

        let source = data
            .first()
            .map(|e| e.source())
            .unwrap_or(KnowledgeSource::Computation);

        Ok(EpistemicBuffer {
            values: GpuBuffer::from_slice(&values, device)?,
            confidences: GpuBuffer::from_slice(&confidences, device)?,
            source,
        })
    }

    /// Create from raw values with uniform confidence
    pub fn from_values(
        values: &[T],
        confidence: Confidence,
        source: KnowledgeSource,
        device: &Device,
    ) -> Result<Self, BufferError> {
        let confidences = vec![confidence.value(); values.len()];

        Ok(EpistemicBuffer {
            values: GpuBuffer::from_slice(values, device)?,
            confidences: GpuBuffer::from_slice(&confidences, device)?,
            source,
        })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the knowledge source
    pub fn source(&self) -> KnowledgeSource {
        self.source
    }

    /// Upload values with uniform confidence
    pub fn upload(&mut self, values: &[T], confidence: Confidence) -> Result<(), BufferError> {
        self.values.upload(values)?;
        let confidences = vec![confidence.value(); values.len()];
        self.confidences.upload(&confidences)?;
        Ok(())
    }

    /// Upload epistemic values
    pub fn upload_epistemic(&mut self, data: &[Epistemic<T>]) -> Result<(), BufferError> {
        let values: Vec<T> = data.iter().map(|e| e.value()).collect();
        let confidences: Vec<f64> = data.iter().map(|e| e.confidence().value()).collect();

        self.values.upload(&values)?;
        self.confidences.upload(&confidences)?;
        Ok(())
    }

    /// Download as epistemic values
    pub fn download(&self) -> Result<Vec<Epistemic<T>>, BufferError> {
        let values = self.values.download()?;
        let confidences = self.confidences.download()?;

        Ok(values
            .into_iter()
            .zip(confidences)
            .map(|(v, c)| Epistemic::new(v, Confidence::new(c), self.source))
            .collect())
    }

    /// Download only values (discard confidence)
    pub fn download_values(&self) -> Result<Vec<T>, BufferError> {
        self.values.download()
    }

    /// Download only confidences
    pub fn download_confidences(&self) -> Result<Vec<f64>, BufferError> {
        self.confidences.download()
    }

    /// Get values device pointer
    pub fn values_ptr(&self) -> *const T {
        self.values.as_device_ptr()
    }

    /// Get values device pointer (mutable)
    pub fn values_ptr_mut(&mut self) -> *mut T {
        self.values.as_device_ptr_mut()
    }

    /// Get confidences device pointer
    pub fn confidences_ptr(&self) -> *const f64 {
        self.confidences.as_device_ptr()
    }

    /// Get confidences device pointer (mutable)
    pub fn confidences_ptr_mut(&mut self) -> *mut f64 {
        self.confidences.as_device_ptr_mut()
    }

    /// Compute mean confidence
    pub fn mean_confidence(&self) -> Result<f64, BufferError> {
        let confidences = self.confidences.download()?;
        if confidences.is_empty() {
            return Ok(0.0);
        }
        Ok(confidences.iter().sum::<f64>() / confidences.len() as f64)
    }

    /// Compute minimum confidence
    pub fn min_confidence(&self) -> Result<Confidence, BufferError> {
        let confidences = self.confidences.download()?;
        let min = confidences.iter().cloned().fold(1.0, f64::min);
        Ok(Confidence::new(min))
    }

    /// Filter elements below confidence threshold
    pub fn filter_reliable(
        &self,
        threshold: f64,
    ) -> Result<Vec<(usize, Epistemic<T>)>, BufferError> {
        let data = self.download()?;
        Ok(data
            .into_iter()
            .enumerate()
            .filter(|(_, e)| e.is_reliable(threshold))
            .collect())
    }
}

/// Confidence propagation for GPU operations
pub mod propagation {
    use super::*;

    /// Propagate confidence through an operation
    pub fn propagate(inputs: &[Confidence], op: ConfidenceOp) -> Confidence {
        match op {
            ConfidenceOp::Min => inputs
                .iter()
                .fold(Confidence::certain(), |acc, c| acc.meet(c)),
            ConfidenceOp::Max => inputs
                .iter()
                .fold(Confidence::uncertain(), |acc, c| acc.join(c)),
            ConfidenceOp::Product => inputs
                .iter()
                .fold(Confidence::certain(), |acc, c| acc.combine(c)),
            ConfidenceOp::Average => {
                if inputs.is_empty() {
                    Confidence::uncertain()
                } else {
                    let sum: f64 = inputs.iter().map(|c| c.value()).sum();
                    Confidence::new(sum / inputs.len() as f64)
                }
            }
        }
    }

    /// Operation type for confidence propagation
    #[derive(Debug, Clone, Copy)]
    pub enum ConfidenceOp {
        /// Minimum (conservative)
        Min,
        /// Maximum (optimistic)
        Max,
        /// Product (independent events)
        Product,
        /// Average (balanced)
        Average,
    }
}

/// Epistemic statistics
#[derive(Debug, Clone)]
pub struct EpistemicStats {
    pub count: usize,
    pub mean_confidence: f64,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub reliable_count: usize,
    pub reliability_threshold: f64,
}

impl EpistemicStats {
    /// Compute statistics for an epistemic buffer
    pub fn compute<T: Copy + Default>(
        buffer: &EpistemicBuffer<T>,
        reliability_threshold: f64,
    ) -> Result<Self, BufferError> {
        let confidences = buffer.confidences.download()?;

        if confidences.is_empty() {
            return Ok(EpistemicStats {
                count: 0,
                mean_confidence: 0.0,
                min_confidence: 0.0,
                max_confidence: 0.0,
                reliable_count: 0,
                reliability_threshold,
            });
        }

        let sum: f64 = confidences.iter().sum();
        let min = confidences.iter().cloned().fold(1.0, f64::min);
        let max = confidences.iter().cloned().fold(0.0, f64::max);
        let reliable = confidences
            .iter()
            .filter(|&&c| c >= reliability_threshold)
            .count();

        Ok(EpistemicStats {
            count: confidences.len(),
            mean_confidence: sum / confidences.len() as f64,
            min_confidence: min,
            max_confidence: max,
            reliable_count: reliable,
            reliability_threshold,
        })
    }

    /// Get reliability ratio (reliable / total)
    pub fn reliability_ratio(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.reliable_count as f64 / self.count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence() {
        let c1 = Confidence::new(0.8);
        let c2 = Confidence::new(0.9);

        assert_eq!(c1.value(), 0.8);
        assert!(c1.above_threshold(0.7));
        assert!(!c1.above_threshold(0.85));

        let combined = c1.combine(&c2);
        assert!((combined.value() - 0.72).abs() < 0.001);

        let meet = c1.meet(&c2);
        assert_eq!(meet.value(), 0.8);

        let join = c1.join(&c2);
        assert_eq!(join.value(), 0.9);
    }

    #[test]
    fn test_epistemic_value() {
        let e = Epistemic::observed(42.0f64);
        assert_eq!(e.value(), 42.0);
        assert!(e.is_reliable(0.9));
        assert_eq!(e.source(), KnowledgeSource::Observation);

        let mapped = e.map(|x| x * 2.0);
        assert_eq!(mapped.value(), 84.0);
        assert_eq!(mapped.source(), KnowledgeSource::Computation);
    }

    #[test]
    fn test_epistemic_buffer() {
        let device = Device::cpu();

        let data = vec![
            Epistemic::observed(1.0f64),
            Epistemic::observed(2.0),
            Epistemic::computed(3.0),
        ];

        let buffer = EpistemicBuffer::from_epistemic(&data, &device).unwrap();
        assert_eq!(buffer.len(), 3);

        let downloaded = buffer.download().unwrap();
        assert_eq!(downloaded[0].value(), 1.0);
        assert_eq!(downloaded[2].value(), 3.0);
    }

    #[test]
    fn test_epistemic_stats() {
        let device = Device::cpu();

        let values = vec![1.0f64, 2.0, 3.0, 4.0];
        let buffer = EpistemicBuffer::from_values(
            &values,
            Confidence::new(0.9),
            KnowledgeSource::Observation,
            &device,
        )
        .unwrap();

        let stats = EpistemicStats::compute(&buffer, 0.8).unwrap();
        assert_eq!(stats.count, 4);
        assert!((stats.mean_confidence - 0.9).abs() < 0.001);
        assert_eq!(stats.reliable_count, 4);
    }

    #[test]
    fn test_confidence_propagation() {
        use propagation::*;

        let inputs = vec![
            Confidence::new(0.8),
            Confidence::new(0.9),
            Confidence::new(0.7),
        ];

        let min = propagate(&inputs, ConfidenceOp::Min);
        assert_eq!(min.value(), 0.7);

        let max = propagate(&inputs, ConfidenceOp::Max);
        assert_eq!(max.value(), 0.9);

        let product = propagate(&inputs, ConfidenceOp::Product);
        assert!((product.value() - 0.504).abs() < 0.001);
    }
}
