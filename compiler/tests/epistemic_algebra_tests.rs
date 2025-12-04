//! Algebraic Property Tests for Epistemic Composition
//!
//! Tests the fundamental algebraic laws of the epistemic operators:
//! - Tensor associativity, commutativity, identity, monotonicity
//! - Join commutativity, idempotence, concordance boost
//! - Condition neutrality, positive evidence increases
//! - Monad laws (left/right identity, associativity)

use demetrios::epistemic::composition::tensor::tensor_identity;
use demetrios::epistemic::composition::*;

// ═══════════════════════════════════════════════════════════════
// TENSOR PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════

/// (T1) Associativity: (K₁ ⊗ K₂) ⊗ K₃ ≅ K₁ ⊗ (K₂ ⊗ K₃)
#[test]
fn test_tensor_associativity() {
    let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(1.0, 0.9);
    let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(2.0, 0.8);
    let k3: EpistemicValue<f64> = EpistemicValue::with_confidence(3.0, 0.7);

    // (K₁ ⊗ K₂) ⊗ K₃
    let left = k1.clone().tensor(k2.clone()).tensor(k3.clone());

    // K₁ ⊗ (K₂ ⊗ K₃)
    let right = k1.tensor(k2.tensor(k3));

    // Values are isomorphic: ((a,b),c) vs (a,(b,c))
    let ((a, b), c) = left.value();
    let (a2, (b2, c2)) = right.value();

    assert!((*a - *a2).abs() < 1e-10);
    assert!((*b - *b2).abs() < 1e-10);
    assert!((*c - *c2).abs() < 1e-10);

    // Confidences should be equal
    assert!((left.confidence().value() - right.confidence().value()).abs() < 1e-10);
}

/// (T2) Commutativity: K₁ ⊗ K₂ ≅ K₂ ⊗ K₁
#[test]
fn test_tensor_commutativity() {
    let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(10.0, 0.85);
    let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(20.0, 0.75);

    let result1 = k1.clone().tensor(k2.clone());
    let result2 = k2.tensor(k1);

    // Values are structurally swapped
    let (a1, b1) = result1.value();
    let (b2, a2) = result2.value();

    assert!((*a1 - *a2).abs() < 1e-10);
    assert!((*b1 - *b2).abs() < 1e-10);

    // Confidences should be equal
    assert!((result1.confidence().value() - result2.confidence().value()).abs() < 1e-10);
}

/// (T3) Identity: K ⊗ I = K (up to wrapping)
#[test]
fn test_tensor_identity() {
    let k: EpistemicValue<f64> = EpistemicValue::with_confidence(42.0, 0.8);
    let identity = tensor_identity();

    let result = k.clone().tensor(identity);

    // Value is (42.0, ())
    assert!((result.value().0 - 42.0).abs() < 1e-10);
    // Confidence preserved (times 1.0)
    assert!((result.confidence().value() - 0.8).abs() < 1e-10);
}

/// (T4) Monotonicity: ε₁ ≤ ε₂ ⟹ ε(K₁⊗K) ≤ ε(K₂⊗K)
#[test]
fn test_tensor_monotonicity() {
    let k_low: EpistemicValue<f64> = EpistemicValue::with_confidence(1.0, 0.5);
    let k_high: EpistemicValue<f64> = EpistemicValue::with_confidence(1.0, 0.9);
    let k_other: EpistemicValue<f64> = EpistemicValue::with_confidence(2.0, 0.8);

    let result_low = k_low.tensor(k_other.clone());
    let result_high = k_high.tensor(k_other);

    assert!(result_low.confidence().value() <= result_high.confidence().value());
}

/// Tensor with ontology overlap reduces confidence
#[test]
fn test_tensor_ontology_correlation() {
    let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(10.0, 0.9)
        .with_ontology(CompositionOntologyRef::new("PKPD", "clearance"));
    let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(20.0, 0.8)
        .with_ontology(CompositionOntologyRef::new("PKPD", "clearance"));

    let result = k1.tensor(k2);

    // Same ontology → γ = 0.5, so ε = 0.9 × 0.8 × 0.5 = 0.36
    assert!((result.confidence().value() - 0.36).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════
// JOIN PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════

/// (J1) Commutativity: K₁ ⊔ K₂ = K₂ ⊔ K₁
#[test]
fn test_join_commutativity() {
    let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);
    let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(5.2, 0.75);

    let result1 = k1.clone().join(k2.clone(), 0.5);
    let result2 = k2.join(k1, 0.5);

    match (result1, result2) {
        (JoinResult::Concordant(r1), JoinResult::Concordant(r2))
        | (JoinResult::Resolved { result: r1, .. }, JoinResult::Resolved { result: r2, .. }) => {
            assert!((*r1.value() - *r2.value()).abs() < 1e-10);
            assert!((r1.confidence().value() - r2.confidence().value()).abs() < 1e-10);
        }
        (JoinResult::Irreconcilable { .. }, JoinResult::Irreconcilable { .. }) => {
            // Both irreconcilable is fine
        }
        _ => panic!("Asymmetric join results"),
    }
}

/// (J2) Idempotence: K ⊔ K = K' where ε(K') ≥ ε(K)
#[test]
fn test_join_idempotence() {
    let k: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);

    let result = k.clone().join(k.clone(), 0.5);

    match result {
        JoinResult::Concordant(r) => {
            assert!((*r.value() - 5.0).abs() < 1e-10);
            // Confidence should be boosted (Dempster-Shafer)
            assert!(r.confidence().value() >= 0.8);
        }
        _ => panic!("Self-join should be concordant"),
    }
}

/// (J3) Concordance increases confidence
#[test]
fn test_join_concordance_boosts_confidence() {
    let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.7);
    let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.6);

    let result = k1.join(k2, 0.5);

    match result {
        JoinResult::Concordant(r) => {
            let max_input = 0.7_f64.max(0.6);
            // Dempster-Shafer: 1 - (1-0.7)(1-0.6) = 0.88
            assert!(r.confidence().value() > max_input);
        }
        _ => panic!("Identical values should be concordant"),
    }
}

/// (J4) High conflict results in irreconcilable
#[test]
fn test_join_irreconcilable_on_high_conflict() {
    let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);
    let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(50.0, 0.75);

    let result = k1.join(k2, 0.3);

    match result {
        JoinResult::Irreconcilable { conflict_level, .. } => {
            assert!(conflict_level.value() >= 0.3);
        }
        _ => panic!("High conflict should be irreconcilable"),
    }
}

// ═══════════════════════════════════════════════════════════════
// CONDITION PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════

/// (C1) Neutral evidence doesn't change confidence
#[test]
fn test_condition_neutral_evidence() {
    let k: EpistemicValue<bool> = EpistemicValue::with_confidence(true, 0.5);

    // Neutral likelihood = 0.5 for both H and ¬H
    let posterior = k.condition(&(), |_| 0.5, 0.5);

    // Should be unchanged
    assert!((posterior.confidence().value() - 0.5).abs() < 0.01);
}

/// (C2) Strong positive evidence increases confidence
#[test]
fn test_condition_positive_evidence_increases() {
    let k: EpistemicValue<bool> = EpistemicValue::with_confidence(true, 0.3);

    // Strong likelihood for H (0.95), weak for ¬H (0.1)
    let posterior = k.condition(&(), |_| 0.95, 0.1);

    assert!(posterior.confidence().value() > 0.3);
    // Should be around 0.803
    assert!((posterior.confidence().value() - 0.803).abs() < 0.01);
}

/// Strong negative evidence decreases confidence
#[test]
fn test_condition_negative_evidence_decreases() {
    let k: EpistemicValue<bool> = EpistemicValue::with_confidence(true, 0.7);

    // Weak likelihood for H (0.1), strong for ¬H (0.9)
    let posterior = k.condition(&(), |_| 0.1, 0.9);

    assert!(posterior.confidence().value() < 0.7);
}

/// Jeffrey conditioning with partition
#[test]
fn test_jeffrey_conditioning() {
    let k: EpistemicValue<bool> = EpistemicValue::with_confidence(true, 0.5);

    // Partition: 60% E₁ (supports H), 40% E₂ (opposes H)
    let posterior = k.condition_jeffrey(vec![
        (0.6, 0.8), // P'(E₁) = 0.6, P(H|E₁) = 0.8
        (0.4, 0.2), // P'(E₂) = 0.4, P(H|E₂) = 0.2
    ]);

    // P'(H) = 0.6×0.8 + 0.4×0.2 = 0.56
    assert!((posterior.confidence().value() - 0.56).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════════
// LIFT/EXTRACT PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════

/// (LE1) Round-trip with certainty
#[test]
fn test_lift_extract_roundtrip() {
    let v = 42.0_f64;
    let lifted: EpistemicValue<f64> = EpistemicValue::certain(v);
    let extracted = lifted.extract(ConfidenceValue::new(0.0).unwrap());

    assert!(extracted.is_some());
    assert!((*extracted.unwrap() - v).abs() < 1e-10);
}

/// (LE2) Extract fails below threshold
#[test]
fn test_extract_fails_below_threshold() {
    let k: EpistemicValue<f64> = EpistemicValue::with_confidence(42.0, 0.5);
    let threshold = ConfidenceValue::new(0.7).unwrap();
    let extracted = k.extract(threshold);

    assert!(extracted.is_none());
}

/// (LE3) Lift preserves value
#[test]
fn test_lift_preserves_value() {
    let v = 123.456_f64;
    let lifted: EpistemicValue<f64> = EpistemicValue::certain(v);

    assert!((*lifted.value() - v).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════
// MONAD LAW TESTS
// ═══════════════════════════════════════════════════════════════

/// Left identity: return a >>= f ≡ f a
#[test]
fn test_monad_left_identity() {
    let a = 5_i32;
    let f = |x: i32| EpistemicValue::with_confidence(x * 2, 0.9);

    let left = EpistemicMonad::bind(EpistemicMonad::pure(a), |x| f(x));
    let right = f(a);

    assert_eq!(*left.value(), *right.value());
}

/// Right identity: m >>= return ≡ m
#[test]
fn test_monad_right_identity() {
    let m: EpistemicValue<i32> = EpistemicValue::with_confidence(10, 0.8);

    let result = EpistemicMonad::bind(m.clone(), EpistemicMonad::pure);

    assert_eq!(*result.value(), *m.value());
}

/// Associativity: (m >>= f) >>= g ≡ m >>= (λx. f x >>= g)
#[test]
fn test_monad_associativity() {
    let m: EpistemicValue<i32> = EpistemicValue::with_confidence(5, 0.9);
    let f = |x: i32| EpistemicValue::with_confidence(x + 1, 0.8);
    let g = |x: i32| EpistemicValue::with_confidence(x * 2, 0.7);

    let left = EpistemicMonad::bind(EpistemicMonad::bind(m.clone(), |x| f(x)), |x| g(x));
    let right = EpistemicMonad::bind(m, |x| EpistemicMonad::bind(f(x), |y| g(y)));

    assert_eq!(*left.value(), *right.value());
    // (5+1)*2 = 12
    assert_eq!(*left.value(), 12);
}

// ═══════════════════════════════════════════════════════════════
// CONFIDENCE COMBINATION TESTS
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_multiplicative_combination() {
    let c1 = ConfidenceValue::new(0.8).unwrap();
    let c2 = ConfidenceValue::new(0.9).unwrap();

    let result = combine_confidence(c1, c2, &CombinationStrategy::Multiplicative);
    assert!((result.value() - 0.72).abs() < 1e-10);
}

#[test]
fn test_dempster_shafer_combination() {
    let c1 = ConfidenceValue::new(0.8).unwrap();
    let c2 = ConfidenceValue::new(0.9).unwrap();

    let result = combine_confidence(c1, c2, &CombinationStrategy::DempsterShafer);
    // 1 - (0.2)(0.1) = 0.98
    assert!((result.value() - 0.98).abs() < 1e-10);
}

#[test]
fn test_penalized_average_combination() {
    let c1 = ConfidenceValue::new(0.8).unwrap();
    let c2 = ConfidenceValue::new(0.6).unwrap();

    let result = combine_confidence(
        c1,
        c2,
        &CombinationStrategy::PenalizedAverage { conflict: 0.2 },
    );
    // avg = 0.7, penalty factor = 0.8 → 0.56
    assert!((result.value() - 0.56).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════

/// PKPD workflow: combine measurements, fuse estimates, condition on evidence
#[test]
fn test_pkpd_workflow() {
    // Source 1: Literature clearance
    let literature: EpistemicValue<f64> =
        EpistemicValue::from_source(5.2, 0.85, SourceInfo::from_doi("10.1234/clin.pharm"));

    // Source 2: Model prediction
    let model: EpistemicValue<f64> =
        EpistemicValue::from_source(5.4, 0.78, SourceInfo::from_inference("NONMEM"));

    // JOIN: Same phenomenon, should be concordant
    let prior = literature.join(model, 0.3);
    assert!(prior.is_success());

    let prior_value = prior.unwrap();

    // Confidence should be boosted (concordant sources)
    assert!(prior_value.confidence().value() > 0.85);

    // Condition on TDM measurement
    let posterior = prior_value.condition(
        &"TDM_001",
        |predicted| {
            let measured = 4.9_f64;
            let error = (predicted - measured).abs();
            let sigma: f64 = 0.5;
            (-error.powi(2) / (2.0 * sigma.powi(2))).exp()
        },
        0.1,
    );

    // Posterior should be valid
    assert!(posterior.confidence().value() > 0.0);
    assert!(posterior.confidence().value() <= 1.0);
}

/// Diagnostic reasoning: tensor symptoms, condition on tests
#[test]
fn test_diagnostic_workflow() {
    // Prior probability of disease
    let prior: EpistemicValue<bool> = EpistemicValue::with_confidence(true, 0.15);

    // Positive test result with sensitivity=0.95, specificity=0.90
    let after_test = prior.condition(&"test_positive", |_| 0.95, 0.10);

    // Prior was 0.15, should increase substantially
    let after_test_conf = after_test.confidence().value();
    assert!(after_test_conf > 0.5);

    // Second confirmatory test
    let after_second = after_test.condition(&"confirm_positive", |_| 0.92, 0.08);

    // Should increase further
    assert!(after_second.confidence().value() > after_test_conf);
}
