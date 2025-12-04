//! Temporal Logic Property Tests
//!
//! Tests the algebraic laws and properties of the temporal epistemic system:
//! - Decay laws
//! - AT laws
//! - Composition laws with time
//! - Version laws

use chrono::{Duration, Utc};
use demetrios::epistemic::composition::EpistemicValue;
use demetrios::temporal::composition::TemporalComposition;
use demetrios::temporal::decay::{DecayFunction, TimeUnit};
use demetrios::temporal::knowledge::TemporalKnowledge;
use demetrios::temporal::operators::TemporalEvent;
use demetrios::temporal::types::{Temporal, Version};
use demetrios::temporal::versioning::VersionedKnowledge;

// ═══════════════════════════════════════════════════════════════
// DECAY LAWS
// ═══════════════════════════════════════════════════════════════

/// (D1) D(0) = 1 - No delay, no decay
#[test]
fn test_decay_law_zero() {
    let decay = DecayFunction::exponential(1.0, TimeUnit::Years);
    assert!((decay.evaluate(Duration::zero()) - 1.0).abs() < 1e-10);

    let linear = DecayFunction::linear(Duration::days(100));
    assert!((linear.evaluate(Duration::zero()) - 1.0).abs() < 1e-10);

    let sigmoid = DecayFunction::sigmoid(Duration::days(30), 0.1);
    assert!(sigmoid.evaluate(Duration::zero()) > 0.9);
}

/// (D2) D(∞) → 0 - Eventual decay to zero
#[test]
fn test_decay_law_infinity() {
    let decay = DecayFunction::exponential(1.0, TimeUnit::Years);
    let far_future = Duration::days(36500); // 100 years
    assert!(decay.evaluate(far_future) < 0.0001);

    let linear = DecayFunction::linear(Duration::days(100));
    assert_eq!(linear.evaluate(Duration::days(200)), 0.0);
}

/// (D3) D is monotonically decreasing
#[test]
fn test_decay_law_monotonic() {
    let decay = DecayFunction::exponential(0.5, TimeUnit::Years);

    let t1 = Duration::days(100);
    let t2 = Duration::days(200);
    let t3 = Duration::days(300);

    let d1 = decay.evaluate(t1);
    let d2 = decay.evaluate(t2);
    let d3 = decay.evaluate(t3);

    assert!(d1 >= d2);
    assert!(d2 >= d3);
}

// ═══════════════════════════════════════════════════════════════
// AT LAWS
// ═══════════════════════════════════════════════════════════════

/// (AT1) K@t₀ = K - At creation, no decay
#[test]
fn test_at_law_creation() {
    let k: TemporalKnowledge<f64> =
        TemporalKnowledge::instant(EpistemicValue::with_confidence(5.0, 0.95));

    let at_now = k.now();
    assert!((at_now.core.confidence().value() - 0.95).abs() < 0.01);
}

/// (AT2) ε(K@t₁) ≥ ε(K@t₂) if t₁ ≤ t₂ - Earlier has higher confidence
#[test]
fn test_at_law_monotonic() {
    let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
        EpistemicValue::with_confidence(5.0, 0.95),
        1.0,
        TimeUnit::Years,
    );

    let now = Utc::now();
    let t1 = now + Duration::days(100);
    let t2 = now + Duration::days(200);
    let t3 = now + Duration::days(300);

    let e1 = k.at(t1).core.confidence().value();
    let e2 = k.at(t2).core.confidence().value();
    let e3 = k.at(t3).core.confidence().value();

    assert!(e1 >= e2);
    assert!(e2 >= e3);
}

// ═══════════════════════════════════════════════════════════════
// TEMPORAL COMPOSITION LAWS
// ═══════════════════════════════════════════════════════════════

/// (TC1) (K₁@t) ⊗ (K₂@t) = (K₁⊗K₂)@t - Tensor commutes with at
#[test]
fn test_tensor_commutes_with_at() {
    let k1: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
        EpistemicValue::with_confidence(5.0, 0.9),
        0.5,
        TimeUnit::Years,
    );
    let k2: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
        EpistemicValue::with_confidence(10.0, 0.8),
        0.5,
        TimeUnit::Years,
    );

    let t = Utc::now() + Duration::days(180);

    // (K₁@t) ⊗ (K₂@t)
    let left = k1.at(t).tensor_temporal(k2.at(t));

    // (K₁⊗K₂)@t
    let right = k1.tensor_temporal(k2).at(t);

    // Values should be the same
    assert_eq!(left.value(), right.value());

    // Confidences should be close (minor differences due to computation order)
    assert!((left.core.confidence().value() - right.core.confidence().value()).abs() < 0.05);
}

/// (TC2) join(K₁,K₂).temporal = newer(t₁,t₂) - Join preserves more recent
#[test]
fn test_join_preserves_newer() {
    let now = Utc::now();

    let k_old: TemporalKnowledge<f64> = TemporalKnowledge {
        core: EpistemicValue::with_confidence(5.0, 0.8),
        temporal: Temporal::Instant(now - Duration::days(30)),
        history: None,
    };

    let k_new: TemporalKnowledge<f64> = TemporalKnowledge {
        core: EpistemicValue::with_confidence(5.1, 0.8),
        temporal: Temporal::Instant(now),
        history: None,
    };

    let result = k_old.join_temporal(k_new, 0.3);

    if let Some(r) = result.result() {
        // Result should have newer temporal
        assert!(r.temporal.effective_instant() >= now - Duration::seconds(1));
    }
}

// ═══════════════════════════════════════════════════════════════
// VERSION LAWS
// ═══════════════════════════════════════════════════════════════

/// (V1) superseded(v) ⟹ ε(v) = 0 - Superseded has zero confidence
#[test]
fn test_version_law_superseded() {
    let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
        EpistemicValue::with_confidence(100.0, 0.90),
        "FDA",
        "Initial",
    );

    let v2 = v1.major_release(
        EpistemicValue::with_confidence(150.0, 0.95),
        "FDA",
        "Update",
        "Trial data",
    );

    // v1 should be superseded
    let v1_in_history = v2.at_version(&Version::new(1, 0, 0)).unwrap();
    assert!(v1_in_history.temporal.is_superseded());
}

/// (V2) extends(v₂,v₁) ⟹ ε(v₂) ≥ ε(v₁) - Extension doesn't reduce confidence
#[test]
fn test_version_law_extends() {
    let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
        EpistemicValue::with_confidence(100.0, 0.85),
        "FDA",
        "Initial",
    );

    let v1_1 = v1.minor_release(
        EpistemicValue::with_confidence(100.0, 0.90),
        "FDA",
        "Extended",
        vec!["new_feature".to_string()],
    );

    assert!(v1_1.confidence().value() >= 0.85);
}

/// (V3) refines(v₂,v₁) ⟹ val(v₂) ≈ val(v₁) - Refinement preserves value
#[test]
fn test_version_law_refines() {
    let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
        EpistemicValue::with_confidence(100.0, 0.85),
        "FDA",
        "Initial",
    );

    let v1_0_1 = v1.patch_release(
        EpistemicValue::with_confidence(100.0, 0.87),
        "FDA",
        "Refined",
        vec!["precision_improvement".to_string()],
    );

    // Value should be the same
    assert_eq!(*v1_0_1.value(), *v1.value());
}

// ═══════════════════════════════════════════════════════════════
// TEMPORAL OPERATOR TESTS
// ═══════════════════════════════════════════════════════════════

/// HISTORICALLY operator
#[test]
fn test_historically_operator() {
    let mut k: TemporalKnowledge<f64> =
        TemporalKnowledge::with_history(EpistemicValue::with_confidence(5.0, 0.9), Temporal::now());

    // Add consistent history
    if let Some(ref mut history) = k.history {
        history.push(TemporalKnowledge::new(
            EpistemicValue::with_confidence(5.0, 0.85),
            Temporal::Instant(Utc::now() - Duration::days(30)),
        ));
        history.push(TemporalKnowledge::new(
            EpistemicValue::with_confidence(5.0, 0.80),
            Temporal::Instant(Utc::now() - Duration::days(60)),
        ));
    }

    let assessment = k.historically();
    assert!(assessment.was_always_true);
    assert_eq!(assessment.history_depth, 2);
    assert!((assessment.historical_confidence.value() - 0.80).abs() < 0.01);
}

/// SINCE operator
#[test]
fn test_since_operator() {
    let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
        EpistemicValue::with_confidence(5.0, 0.95),
        0.1,
        TimeUnit::Years,
    );

    let event: TemporalEvent<&str> =
        TemporalEvent::past("treatment_started", Duration::days(30), "Treatment started");

    let assessment = k.since(&event);
    assert!(assessment.was_true_at_trigger);
    assert!(assessment.maintained_since);
}

/// EVENTUALLY operator
#[test]
fn test_eventually_operator() {
    let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
        EpistemicValue::with_confidence(5.0, 1.0),
        1.0,
        TimeUnit::Years,
    );

    // Project 1 year ahead
    let prediction = k.eventually(Duration::days(365));

    // Confidence should decay significantly
    assert!(prediction.projected_confidence.value() < 0.5);

    // Uncertainty should be non-zero
    assert!(prediction.uncertainty > 0.0);
}

/// ALWAYS operator
#[test]
fn test_always_operator() {
    let k: TemporalKnowledge<f64> =
        TemporalKnowledge::timeless(EpistemicValue::with_confidence(299792458.0, 1.0));

    let constraint = k.always(None);

    assert!(!constraint.requires_revalidation);
    assert!(constraint.is_satisfied());
}

/// UNTIL operator
#[test]
fn test_until_operator() {
    let k: TemporalKnowledge<f64> =
        TemporalKnowledge::instant(EpistemicValue::with_confidence(5.0, 0.95));

    let mut monitor = k.until(|v| *v > 100.0, Duration::seconds(1));

    // Should still be waiting (value is 5.0, not > 100)
    let status = monitor.check();
    assert!(status.is_waiting());
}

// ═══════════════════════════════════════════════════════════════
// DECAY FUNCTION TESTS
// ═══════════════════════════════════════════════════════════════

/// Exponential decay half-life
#[test]
fn test_exponential_half_life() {
    let decay = DecayFunction::from_half_life(Duration::days(30), TimeUnit::Days);

    // At half-life, decay should be 0.5
    let at_half = decay.evaluate(Duration::days(30));
    assert!((at_half - 0.5).abs() < 0.05);

    // At 2x half-life, decay should be 0.25
    let at_double = decay.evaluate(Duration::days(60));
    assert!((at_double - 0.25).abs() < 0.05);
}

/// Step decay behavior
#[test]
fn test_step_decay_behavior() {
    let decay = DecayFunction::step(Duration::days(30));

    // Before threshold: full confidence
    assert_eq!(decay.evaluate(Duration::days(15)), 1.0);
    assert_eq!(decay.evaluate(Duration::days(29)), 1.0);

    // At threshold: still valid
    assert_eq!(decay.evaluate(Duration::days(30)), 1.0);

    // After threshold: zero
    assert_eq!(decay.evaluate(Duration::days(31)), 0.0);
}

/// Linear decay behavior
#[test]
fn test_linear_decay_behavior() {
    let decay = DecayFunction::linear(Duration::days(100));

    // At t=0: 1.0
    assert!((decay.evaluate(Duration::zero()) - 1.0).abs() < 1e-10);

    // At t=50: 0.5
    assert!((decay.evaluate(Duration::days(50)) - 0.5).abs() < 0.01);

    // At t=100: 0.0
    assert!((decay.evaluate(Duration::days(100)) - 0.0).abs() < 0.01);
}

/// Sigmoid decay behavior
#[test]
fn test_sigmoid_decay_behavior() {
    let decay = DecayFunction::sigmoid(Duration::days(30), 0.2);

    // At t=0: close to 1.0
    assert!(decay.evaluate(Duration::zero()) > 0.9);

    // At midpoint: 0.5
    assert!((decay.evaluate(Duration::days(30)) - 0.5).abs() < 0.01);

    // After midpoint: close to 0.0
    assert!(decay.evaluate(Duration::days(100)) < 0.1);
}

/// Product of decay functions
#[test]
fn test_decay_product() {
    let d1 = DecayFunction::exponential(0.5, TimeUnit::Years);
    let d2 = DecayFunction::exponential(0.3, TimeUnit::Years);
    let product = DecayFunction::product(&d1, &d2);

    let one_year = Duration::days(365);
    let expected = (-0.5_f64).exp() * (-0.3_f64).exp();
    assert!((product.evaluate(one_year) - expected).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════

/// Drug clearance evolution over time
#[test]
fn test_pkpd_clearance_evolution() {
    // Initial measurement (simulated as 2 years ago)
    let now = Utc::now();
    let k_2022: TemporalKnowledge<f64> = TemporalKnowledge {
        core: EpistemicValue::with_confidence(5.2, 0.95),
        temporal: Temporal::Decaying {
            created: now - Duration::days(730),
            decay_fn: DecayFunction::exponential(0.3, TimeUnit::Years),
        },
        history: None,
    };

    // Current confidence should be reduced
    let current_conf = k_2022.current_confidence().value();
    assert!(current_conf < 0.60); // 0.95 * e^(-0.3*2) ≈ 0.52

    // New measurement
    let k_2024: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
        EpistemicValue::with_confidence(4.8, 0.95),
        0.3,
        TimeUnit::Years,
    );

    // Join with recency bonus
    let result = k_2022.join_temporal(k_2024, 0.3);

    assert!(result.is_success());
    if let Some(r) = result.result() {
        // Value should be closer to newer measurement
        assert!(*r.value() < 5.0);
        // Recency bonus should be significant
        assert!(result.recency_bonus().unwrap() > 1.5);
    }
}

/// Clinical guideline versioning
#[test]
fn test_clinical_guideline_versioning() {
    // Initial guideline
    let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
        EpistemicValue::with_confidence(100.0, 0.90), // 100mg dose
        "FDA",
        "Initial dose recommendation based on Phase III",
    );

    // Minor update (added indication)
    let v1_1 = v1.minor_release(
        EpistemicValue::with_confidence(100.0, 0.92),
        "FDA",
        "Added renal adjustment guidance",
        vec!["renal_dosing".to_string()],
    );

    // Major update (dose change based on new trial)
    let v2 = v1_1.major_release(
        EpistemicValue::with_confidence(150.0, 0.95),
        "FDA",
        "Updated dose based on CONFIRM-2 trial",
        "Higher dose shows improved efficacy",
    );

    // Check version history
    assert_eq!(v2.all_versions().len(), 3);

    // v1 should be superseded
    assert!(
        v2.at_version(&Version::new(1, 0, 0))
            .unwrap()
            .temporal
            .is_superseded()
    );

    // Current version should be valid
    assert!(v2.is_valid());
    assert_eq!(*v2.value(), 150.0);
}

/// Multi-source knowledge integration with temporal awareness
#[test]
fn test_multi_source_temporal_integration() {
    let now = Utc::now();

    // Literature (6 months old)
    let literature: TemporalKnowledge<f64> = TemporalKnowledge {
        core: EpistemicValue::with_confidence(5.2, 0.85),
        temporal: Temporal::Decaying {
            created: now - Duration::days(180),
            decay_fn: demetrios::temporal::decay::presets::scientific_literature(),
        },
        history: None,
    };

    // Model prediction (current)
    let model: TemporalKnowledge<f64> =
        TemporalKnowledge::instant(EpistemicValue::with_confidence(5.4, 0.78));

    // Recent measurement (1 week old)
    let measurement: TemporalKnowledge<f64> = TemporalKnowledge {
        core: EpistemicValue::with_confidence(5.1, 0.90),
        temporal: Temporal::Decaying {
            created: now - Duration::days(7),
            decay_fn: demetrios::temporal::decay::presets::patient_data(),
        },
        history: None,
    };

    // Join literature with model
    let prior = literature.join_temporal(model, 0.3);
    assert!(prior.is_success());

    // Join with measurement (should get recency bonus)
    let result = prior
        .result()
        .unwrap()
        .clone()
        .join_temporal(measurement, 0.3);
    assert!(result.is_success());

    let final_result = result.result().unwrap();
    // Final value should be close to measurement due to recency and high confidence
    assert!((*final_result.value() - 5.1).abs() < 0.2);
}
