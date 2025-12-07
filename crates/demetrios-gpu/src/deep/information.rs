//! Information-Theoretic Computation
//!
//! "It from Bit" - Wheeler's insight that information is fundamental.
//!
//! This module provides primitives for tracking information content,
//! entropy, complexity, and mutual information through computation.
//!
//! # Key Concepts
//!
//! ## Shannon Entropy
//! H(X) = -∑ p(x) log p(x)
//! Measures uncertainty/information content of a random variable.
//!
//! ## Kolmogorov Complexity
//! K(x) = min { |p| : U(p) = x }
//! The length of the shortest program that produces x.
//! Incomputable in general, but we can bound it.
//!
//! ## Mutual Information
//! I(X;Y) = H(X) + H(Y) - H(X,Y)
//! Measures shared information between variables.
//!
//! ## Algorithmic Information Dynamics (AID)
//! Tracks how information flows and transforms through computation,
//! enabling causal discovery without interventions.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

// ============================================================================
// ENTROPY
// ============================================================================

/// Shannon entropy in bits
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Entropy(f64);

impl Entropy {
    /// Zero entropy (complete certainty)
    pub const ZERO: Entropy = Entropy(0.0);

    /// Maximum entropy for n equiprobable outcomes
    pub fn maximum(n: usize) -> Self {
        if n <= 1 {
            Self::ZERO
        } else {
            Entropy((n as f64).log2())
        }
    }

    /// Create from bits
    pub fn bits(bits: f64) -> Self {
        assert!(bits >= 0.0, "Entropy cannot be negative");
        Entropy(bits)
    }

    /// Create from a probability distribution
    pub fn from_distribution(probs: &[f64]) -> Self {
        let h: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();
        Entropy(h)
    }

    /// Create from empirical frequencies
    pub fn from_frequencies<T: Hash + Eq>(items: &[T]) -> Self {
        if items.is_empty() {
            return Self::ZERO;
        }

        let mut counts: HashMap<&T, usize> = HashMap::new();
        for item in items {
            *counts.entry(item).or_insert(0) += 1;
        }

        let n = items.len() as f64;
        let probs: Vec<f64> = counts.values().map(|&c| c as f64 / n).collect();
        Self::from_distribution(&probs)
    }

    /// Get the value in bits
    pub fn as_bits(&self) -> f64 {
        self.0
    }

    /// Get the value in nats (natural logarithm)
    pub fn as_nats(&self) -> f64 {
        self.0 * std::f64::consts::LN_2
    }

    /// Check if this is maximum entropy for n outcomes
    pub fn is_maximum(&self, n: usize) -> bool {
        (self.0 - Self::maximum(n).0).abs() < 1e-10
    }
}

impl std::ops::Add for Entropy {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Entropy(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Entropy {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Entropy((self.0 - rhs.0).max(0.0))
    }
}

impl fmt::Display for Entropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} bits", self.0)
    }
}

// ============================================================================
// KOLMOGOROV COMPLEXITY
// ============================================================================

/// Kolmogorov complexity bounds
///
/// The true Kolmogorov complexity K(x) is incomputable,
/// but we can establish upper and lower bounds.
#[derive(Debug, Clone)]
pub struct KolmogorovComplexity {
    /// Upper bound (length of known compression)
    pub upper_bound: usize,
    /// Lower bound (theoretical minimum)
    pub lower_bound: usize,
    /// The method used to establish bounds
    pub method: ComplexityMethod,
}

/// Methods for estimating Kolmogorov complexity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityMethod {
    /// Raw size (trivial upper bound)
    RawSize,
    /// Entropy-based bound
    EntropyBound,
    /// Lempel-Ziv compression
    LempelZiv,
    /// Block Decomposition Method (BDM)
    BlockDecomposition,
    /// Coding Theorem Method (CTM)
    CodingTheorem,
}

impl KolmogorovComplexity {
    /// Trivial bound: K(x) <= |x| + O(1)
    pub fn trivial(size: usize) -> Self {
        Self {
            upper_bound: size,
            lower_bound: 0,
            method: ComplexityMethod::RawSize,
        }
    }

    /// Entropy-based bound: K(x) >= H(x)
    pub fn from_entropy(entropy: Entropy, size: usize) -> Self {
        let lower = entropy.as_bits().ceil() as usize;
        Self {
            upper_bound: size,
            lower_bound: lower,
            method: ComplexityMethod::EntropyBound,
        }
    }

    /// Estimate complexity using LZ77-style compression
    pub fn estimate_lz<T: Eq + Hash + Clone>(data: &[T]) -> Self {
        if data.is_empty() {
            return Self::trivial(0);
        }

        // Simple LZ-style complexity estimation
        // Count number of distinct substrings needed
        let mut dictionary: HashMap<Vec<T>, usize> = HashMap::new();
        let mut complexity = 0;
        let mut i = 0;

        while i < data.len() {
            let mut longest_match = 0;
            for len in 1..=(data.len() - i) {
                let substr: Vec<T> = data[i..i + len].to_vec();
                if dictionary.contains_key(&substr) {
                    longest_match = len;
                } else {
                    break;
                }
            }

            if longest_match == 0 {
                // New symbol
                dictionary.insert(vec![data[i].clone()], dictionary.len());
                complexity += 1;
                i += 1;
            } else {
                // Extend with one new symbol
                let mut new_substr: Vec<T> = data[i..i + longest_match].to_vec();
                if i + longest_match < data.len() {
                    new_substr.push(data[i + longest_match].clone());
                    dictionary.insert(new_substr, dictionary.len());
                    i += longest_match + 1;
                } else {
                    i += longest_match;
                }
                complexity += 1;
            }
        }

        // Complexity in bits: c * log(c)
        let bits = if complexity > 1 {
            (complexity as f64 * (complexity as f64).log2()).ceil() as usize
        } else {
            complexity
        };

        Self {
            upper_bound: bits,
            lower_bound: Entropy::from_frequencies(data).as_bits().ceil() as usize,
            method: ComplexityMethod::LempelZiv,
        }
    }

    /// Check if data is algorithmically random
    /// (complexity close to raw size)
    pub fn is_random(&self, size: usize, threshold: f64) -> bool {
        self.upper_bound as f64 >= size as f64 * threshold
    }

    /// Normalized complexity [0, 1]
    pub fn normalized(&self, size: usize) -> f64 {
        if size == 0 {
            0.0
        } else {
            self.upper_bound as f64 / size as f64
        }
    }
}

// ============================================================================
// MUTUAL INFORMATION
// ============================================================================

/// Mutual information between two variables
/// I(X;Y) = H(X) + H(Y) - H(X,Y)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MutualInformation(f64);

impl MutualInformation {
    /// No shared information
    pub const ZERO: MutualInformation = MutualInformation(0.0);

    /// Create from bits
    pub fn bits(bits: f64) -> Self {
        assert!(bits >= 0.0, "Mutual information cannot be negative");
        MutualInformation(bits)
    }

    /// Compute mutual information between two discrete variables
    pub fn compute<X: Hash + Eq, Y: Hash + Eq>(xs: &[X], ys: &[Y]) -> Self {
        assert_eq!(xs.len(), ys.len(), "Variables must have same length");
        if xs.is_empty() {
            return Self::ZERO;
        }

        let n = xs.len() as f64;

        // Count joint occurrences
        let mut joint: HashMap<(&X, &Y), usize> = HashMap::new();
        for (x, y) in xs.iter().zip(ys.iter()) {
            *joint.entry((x, y)).or_insert(0) += 1;
        }

        // Marginal counts
        let mut x_counts: HashMap<&X, usize> = HashMap::new();
        let mut y_counts: HashMap<&Y, usize> = HashMap::new();
        for x in xs {
            *x_counts.entry(x).or_insert(0) += 1;
        }
        for y in ys {
            *y_counts.entry(y).or_insert(0) += 1;
        }

        // I(X;Y) = ∑∑ p(x,y) log(p(x,y) / (p(x)p(y)))
        let mut mi = 0.0;
        for ((x, y), &joint_count) in &joint {
            let p_xy = joint_count as f64 / n;
            let p_x = x_counts[x] as f64 / n;
            let p_y = y_counts[y] as f64 / n;

            if p_xy > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }

        MutualInformation(mi.max(0.0))
    }

    /// Normalized mutual information [0, 1]
    pub fn normalized(&self, h_x: Entropy, h_y: Entropy) -> f64 {
        let max_h = h_x.as_bits().max(h_y.as_bits());
        if max_h > 0.0 {
            self.0 / max_h
        } else {
            0.0
        }
    }

    /// Get the value in bits
    pub fn as_bits(&self) -> f64 {
        self.0
    }
}

// ============================================================================
// INFORMATION CONTENT
// ============================================================================

/// Complete information-theoretic profile of a value
#[derive(Debug, Clone)]
pub struct InformationContent {
    /// Shannon entropy
    pub entropy: Entropy,
    /// Kolmogorov complexity bounds
    pub complexity: KolmogorovComplexity,
    /// Is this algorithmically random?
    pub is_random: bool,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

impl InformationContent {
    /// Analyze a sequence
    pub fn analyze<T: Hash + Eq + Clone>(data: &[T]) -> Self {
        let entropy = Entropy::from_frequencies(data);
        let complexity = KolmogorovComplexity::estimate_lz(data);
        let size = data.len();

        Self {
            entropy,
            is_random: complexity.is_random(size, 0.9),
            compression_ratio: if size > 0 {
                complexity.upper_bound as f64 / size as f64
            } else {
                1.0
            },
            complexity,
        }
    }

    /// Create for a uniformly random source
    pub fn random(size: usize, alphabet_size: usize) -> Self {
        let entropy = Entropy::maximum(alphabet_size);
        Self {
            entropy,
            complexity: KolmogorovComplexity::trivial(size),
            is_random: true,
            compression_ratio: 1.0,
        }
    }

    /// Create for a constant (zero entropy) source
    pub fn constant() -> Self {
        Self {
            entropy: Entropy::ZERO,
            complexity: KolmogorovComplexity::trivial(1),
            is_random: false,
            compression_ratio: 0.0,
        }
    }
}

// ============================================================================
// INFORMATION FLOW
// ============================================================================

/// Tracks how information flows through computation
#[derive(Debug, Clone)]
pub struct InformationFlow<T> {
    /// The value
    pub value: T,
    /// Information content
    pub info: InformationContent,
    /// Transformation history
    pub transforms: Vec<InformationTransform>,
}

/// A transformation applied to information
#[derive(Debug, Clone)]
pub struct InformationTransform {
    /// Name of the transformation
    pub name: String,
    /// Entropy before
    pub entropy_in: Entropy,
    /// Entropy after
    pub entropy_out: Entropy,
    /// Was information lost?
    pub lossy: bool,
    /// Information lost (if lossy)
    pub information_lost: f64,
}

impl<T: Hash + Eq + Clone> InformationFlow<T> {
    /// Create a new information flow from raw data
    pub fn new(value: T, raw_data: &[T]) -> Self {
        Self {
            value,
            info: InformationContent::analyze(raw_data),
            transforms: Vec::new(),
        }
    }

    /// Apply a transformation and track information flow
    pub fn transform<F, U>(self, name: &str, f: F, result_data: &[U]) -> InformationFlow<U>
    where
        F: FnOnce(T) -> U,
        U: Hash + Eq + Clone,
    {
        let new_value = f(self.value);
        let new_info = InformationContent::analyze(result_data);

        let entropy_in = self.info.entropy;
        let entropy_out = new_info.entropy;
        let lossy = entropy_out < entropy_in;
        let information_lost = if lossy {
            entropy_in.as_bits() - entropy_out.as_bits()
        } else {
            0.0
        };

        let transform = InformationTransform {
            name: name.to_string(),
            entropy_in,
            entropy_out,
            lossy,
            information_lost,
        };

        let mut transforms = self.transforms;
        transforms.push(transform);

        InformationFlow {
            value: new_value,
            info: new_info,
            transforms,
        }
    }

    /// Total information lost through all transformations
    pub fn total_information_lost(&self) -> f64 {
        self.transforms.iter().map(|t| t.information_lost).sum()
    }

    /// Check if any transformation was lossy
    pub fn is_lossy(&self) -> bool {
        self.transforms.iter().any(|t| t.lossy)
    }
}

// ============================================================================
// COMPRESSION BOUNDS
// ============================================================================

/// Theoretical bounds on compression
#[derive(Debug, Clone, Copy)]
pub struct CompressionBound {
    /// Minimum bits needed (entropy)
    pub minimum: f64,
    /// Achieved compression
    pub achieved: f64,
    /// Theoretical optimum (Kolmogorov)
    pub optimum: f64,
    /// Gap from optimum
    pub gap: f64,
}

impl CompressionBound {
    /// Compute compression bounds for data
    pub fn compute<T: Hash + Eq + Clone>(data: &[T]) -> Self {
        let info = InformationContent::analyze(data);
        let minimum = info.entropy.as_bits();
        let achieved = info.complexity.upper_bound as f64;
        let optimum = info.complexity.lower_bound as f64;

        Self {
            minimum,
            achieved,
            optimum,
            gap: achieved - optimum,
        }
    }

    /// How close are we to optimal compression?
    pub fn efficiency(&self) -> f64 {
        if self.achieved > 0.0 {
            self.optimum / self.achieved
        } else {
            1.0
        }
    }
}

// ============================================================================
// ALGORITHMIC RANDOMNESS
// ============================================================================

/// Tests for algorithmic randomness
#[derive(Debug, Clone)]
pub struct AlgorithmicRandomness {
    /// Normalized complexity (should be close to 1 for random)
    pub normalized_complexity: f64,
    /// Chi-squared statistic for uniformity
    pub chi_squared: f64,
    /// Is likely random?
    pub is_random: bool,
    /// Confidence level
    pub confidence: f64,
}

impl AlgorithmicRandomness {
    /// Test if a byte sequence is algorithmically random
    pub fn test_bytes(data: &[u8]) -> Self {
        if data.is_empty() {
            return Self {
                normalized_complexity: 0.0,
                chi_squared: 0.0,
                is_random: false,
                confidence: 0.0,
            };
        }

        // Kolmogorov complexity test
        let complexity = KolmogorovComplexity::estimate_lz(data);
        let normalized = complexity.normalized(data.len());

        // Chi-squared test for byte uniformity
        let mut counts = [0usize; 256];
        for &b in data {
            counts[b as usize] += 1;
        }

        let expected = data.len() as f64 / 256.0;
        let chi_squared: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();

        // Degrees of freedom = 255, critical value at 0.01 ≈ 310
        let chi_threshold = 310.0;
        let chi_random = chi_squared < chi_threshold;

        // Combined assessment
        let complexity_random = normalized > 0.9;
        let is_random = complexity_random && chi_random;

        // Confidence based on both tests
        let complexity_confidence = normalized.min(1.0);
        let chi_confidence = (1.0 - chi_squared / (chi_threshold * 2.0)).max(0.0);
        let confidence = (complexity_confidence + chi_confidence) / 2.0;

        Self {
            normalized_complexity: normalized,
            chi_squared,
            is_random,
            confidence,
        }
    }
}

// ============================================================================
// INFORMATION CHANNEL
// ============================================================================

/// An information channel with capacity constraints
#[derive(Debug, Clone)]
pub struct InformationChannel {
    /// Channel capacity in bits per use
    pub capacity: f64,
    /// Noise level (0 = noiseless, 1 = useless)
    pub noise: f64,
    /// Error probability
    pub error_rate: f64,
}

impl InformationChannel {
    /// Noiseless channel with unlimited capacity
    pub fn ideal() -> Self {
        Self {
            capacity: f64::INFINITY,
            noise: 0.0,
            error_rate: 0.0,
        }
    }

    /// Binary symmetric channel
    pub fn binary_symmetric(error_prob: f64) -> Self {
        assert!(error_prob >= 0.0 && error_prob <= 0.5);

        // C = 1 - H(p) for BSC
        let h = if error_prob > 0.0 && error_prob < 1.0 {
            -error_prob * error_prob.log2() - (1.0 - error_prob) * (1.0 - error_prob).log2()
        } else {
            0.0
        };

        Self {
            capacity: 1.0 - h,
            noise: 2.0 * error_prob,
            error_rate: error_prob,
        }
    }

    /// AWGN channel with signal-to-noise ratio
    pub fn awgn(snr_db: f64) -> Self {
        let snr = 10.0_f64.powf(snr_db / 10.0);
        let capacity = 0.5 * (1.0 + snr).log2();

        Self {
            capacity,
            noise: 1.0 / (1.0 + snr),
            error_rate: 0.0, // Depends on coding
        }
    }

    /// Maximum reliable transmission rate
    pub fn max_rate(&self) -> f64 {
        self.capacity
    }

    /// Can we reliably transmit at this rate?
    pub fn can_transmit(&self, rate: f64) -> bool {
        rate <= self.capacity
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = Entropy::from_distribution(&probs);
        assert!((h.as_bits() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_deterministic() {
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let h = Entropy::from_distribution(&probs);
        assert!(h.as_bits().abs() < 1e-10);
    }

    #[test]
    fn test_entropy_from_frequencies() {
        let data = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let h = Entropy::from_frequencies(&data);
        assert!((h.as_bits() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        let xs = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let ys = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let mi = MutualInformation::compute(&xs, &ys);
        assert!(mi.as_bits() < 0.1); // Should be close to 0
    }

    #[test]
    fn test_mutual_information_identical() {
        let xs = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let mi = MutualInformation::compute(&xs, &xs);
        let h = Entropy::from_frequencies(&xs);
        assert!((mi.as_bits() - h.as_bits()).abs() < 1e-10);
    }

    #[test]
    fn test_kolmogorov_constant() {
        let data = vec![0u8; 1000];
        let k = KolmogorovComplexity::estimate_lz(&data);
        // Constant data is highly compressible - upper bound much less than raw size
        assert!(k.upper_bound < data.len());
    }

    #[test]
    fn test_kolmogorov_pattern() {
        let data: Vec<u8> = (0..100).map(|i| (i % 10) as u8).collect();
        let k = KolmogorovComplexity::estimate_lz(&data);
        // Patterned data compresses - upper bound less than raw size in bits
        assert!(k.upper_bound < data.len() * 8);
    }

    #[test]
    fn test_binary_symmetric_channel() {
        let bsc = InformationChannel::binary_symmetric(0.0);
        assert!((bsc.capacity - 1.0).abs() < 1e-10);

        let bsc_half = InformationChannel::binary_symmetric(0.5);
        assert!(bsc_half.capacity < 0.01);
    }

    #[test]
    fn test_information_flow() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // InformationFlow::new expects value of type T and raw_data as &[T]
        // Here we track each byte as an individual element
        let flow = InformationFlow::new(data[0], &data);

        // Transform: add 10 (lossless 1-to-1 mapping)
        let transformed: Vec<u8> = data.iter().map(|x| x + 10).collect();
        let flow2 = flow.transform("add10", |d| d + 10, &transformed);

        // Adding 10 is a bijection, so entropy should be preserved
        assert!(!flow2.is_lossy());
    }
}
