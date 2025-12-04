//! Comprehensive tests for Day 36: Linear Epistemic Types
//!
//! Tests cover:
//! - Modality lattice and operations
//! - Linear type construction and duality
//! - Context management and splitting
//! - Exponentials (!, ?)
//! - Session types and protocols
//! - Subtyping rules
//! - Consumption tracking

use demetrios::dependent::types::OntologyType;
use demetrios::linear::session_types::SessionChecker;
use demetrios::linear::*;
use demetrios::types::Type;

// ============================================================================
// MODALITY TESTS
// ============================================================================

mod modality_tests {
    use super::*;

    #[test]
    fn test_modality_combine_linear_dominates() {
        // Linear combined with anything is Linear
        assert_eq!(Modality::Linear.combine(Modality::Linear), Modality::Linear);
        assert_eq!(Modality::Linear.combine(Modality::Affine), Modality::Linear);
        assert_eq!(
            Modality::Linear.combine(Modality::Relevant),
            Modality::Linear
        );
        assert_eq!(
            Modality::Linear.combine(Modality::Unrestricted),
            Modality::Linear
        );
    }

    #[test]
    fn test_modality_combine_affine_relevant() {
        // Affine + Relevant = Linear (can't satisfy both)
        assert_eq!(
            Modality::Affine.combine(Modality::Relevant),
            Modality::Linear
        );
        assert_eq!(
            Modality::Relevant.combine(Modality::Affine),
            Modality::Linear
        );
    }

    #[test]
    fn test_modality_combine_unrestricted() {
        // Unrestricted combined with X is X
        assert_eq!(
            Modality::Unrestricted.combine(Modality::Linear),
            Modality::Linear
        );
        assert_eq!(
            Modality::Unrestricted.combine(Modality::Affine),
            Modality::Affine
        );
        assert_eq!(
            Modality::Unrestricted.combine(Modality::Relevant),
            Modality::Relevant
        );
        assert_eq!(
            Modality::Unrestricted.combine(Modality::Unrestricted),
            Modality::Unrestricted
        );
    }

    #[test]
    fn test_modality_subtype_lattice() {
        // Linear is bottom
        assert!(Modality::Linear.is_subtype_of(Modality::Linear));
        assert!(Modality::Linear.is_subtype_of(Modality::Affine));
        assert!(Modality::Linear.is_subtype_of(Modality::Relevant));
        assert!(Modality::Linear.is_subtype_of(Modality::Unrestricted));

        // Unrestricted is top
        assert!(Modality::Affine.is_subtype_of(Modality::Unrestricted));
        assert!(Modality::Relevant.is_subtype_of(Modality::Unrestricted));

        // Affine and Relevant are incomparable
        assert!(!Modality::Affine.is_subtype_of(Modality::Relevant));
        assert!(!Modality::Relevant.is_subtype_of(Modality::Affine));
    }

    #[test]
    fn test_modality_structural_rules() {
        // Weakening
        assert!(!Modality::Linear.allows_weakening());
        assert!(Modality::Affine.allows_weakening());
        assert!(!Modality::Relevant.allows_weakening());
        assert!(Modality::Unrestricted.allows_weakening());

        // Contraction
        assert!(!Modality::Linear.allows_contraction());
        assert!(!Modality::Affine.allows_contraction());
        assert!(Modality::Relevant.allows_contraction());
        assert!(Modality::Unrestricted.allows_contraction());
    }

    #[test]
    fn test_modality_must_use() {
        assert!(Modality::Linear.must_use());
        assert!(!Modality::Affine.must_use());
        assert!(Modality::Relevant.must_use());
        assert!(!Modality::Unrestricted.must_use());
    }

    #[test]
    fn test_modality_parse() {
        assert_eq!(Modality::parse("linear"), Some(Modality::Linear));
        assert_eq!(Modality::parse("affine"), Some(Modality::Affine));
        assert_eq!(Modality::parse("relevant"), Some(Modality::Relevant));
        assert_eq!(
            Modality::parse("unrestricted"),
            Some(Modality::Unrestricted)
        );
        assert_eq!(Modality::parse("unknown"), None);
    }
}

// ============================================================================
// LINEAR TYPE TESTS
// ============================================================================

mod linear_type_tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let a = LinearType::One;
        let b = LinearType::Top;
        let tensor = LinearType::tensor(a, b);

        match tensor {
            LinearType::Tensor(_, _) => {}
            _ => panic!("Expected Tensor"),
        }
    }

    #[test]
    fn test_lollipop_creation() {
        let a = LinearType::One;
        let b = LinearType::Top;
        let lolli = LinearType::lollipop(a, b);

        match lolli {
            LinearType::Lollipop(_, _) => {}
            _ => panic!("Expected Lollipop"),
        }
    }

    #[test]
    fn test_dual_multiplicatives() {
        // dual(A ⊗ B) = dual(A) ⅋ dual(B)
        let tensor = LinearType::tensor(LinearType::One, LinearType::Top);
        let dual = tensor.dual();

        match dual {
            LinearType::Par(_, _) => {}
            _ => panic!("Expected Par, got {:?}", dual),
        }

        // dual(A ⅋ B) = dual(A) ⊗ dual(B)
        let par = LinearType::par(LinearType::One, LinearType::Top);
        let dual = par.dual();

        match dual {
            LinearType::Tensor(_, _) => {}
            _ => panic!("Expected Tensor, got {:?}", dual),
        }
    }

    #[test]
    fn test_dual_additives() {
        // dual(A & B) = dual(A) ⊕ dual(B)
        let with = LinearType::with(LinearType::One, LinearType::Top);
        let dual = with.dual();

        match dual {
            LinearType::Plus(_, _) => {}
            _ => panic!("Expected Plus, got {:?}", dual),
        }

        // dual(A ⊕ B) = dual(A) & dual(B)
        let plus = LinearType::plus(LinearType::One, LinearType::Top);
        let dual = plus.dual();

        match dual {
            LinearType::With(_, _) => {}
            _ => panic!("Expected With, got {:?}", dual),
        }
    }

    #[test]
    fn test_dual_exponentials() {
        // dual(!A) = ?dual(A)
        let bang = LinearType::bang(LinearType::One);
        let dual = bang.dual();

        match dual {
            LinearType::Quest(_) => {}
            _ => panic!("Expected Quest, got {:?}", dual),
        }

        // dual(?A) = !dual(A)
        let quest = LinearType::quest(LinearType::One);
        let dual = quest.dual();

        match dual {
            LinearType::Bang(_) => {}
            _ => panic!("Expected Bang, got {:?}", dual),
        }
    }

    #[test]
    fn test_dual_units() {
        assert!(matches!(LinearType::One.dual(), LinearType::Bottom));
        assert!(matches!(LinearType::Bottom.dual(), LinearType::One));
        assert!(matches!(LinearType::Top.dual(), LinearType::Zero));
        assert!(matches!(LinearType::Zero.dual(), LinearType::Top));
    }

    #[test]
    fn test_dual_involution() {
        // Test with types that are structurally involutive
        // Note: Lollipop is A⊥ ⅋ B, so (A ⊸ B)⊥ = A ⊗ B⊥, and dual again gives
        // a Par not a Lollipop. We test simpler structural duals here.
        let typ = LinearType::tensor(
            LinearType::par(LinearType::One, LinearType::Top),
            LinearType::bang(LinearType::One),
        );

        let dual1 = typ.dual();
        let dual2 = dual1.dual();

        assert!(typ.definitionally_equal(&dual2));
    }

    #[test]
    fn test_modality_inference() {
        // Tensor combines modalities
        let linear = LinearType::linear_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let affine = LinearType::affine_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let tensor = LinearType::tensor(linear, affine);
        assert_eq!(tensor.modality(), Modality::Linear);

        // Bang is always unrestricted
        let bang = LinearType::bang(LinearType::One);
        assert_eq!(bang.modality(), Modality::Unrestricted);

        // Quest is always affine
        let quest = LinearType::quest(LinearType::One);
        assert_eq!(quest.modality(), Modality::Affine);
    }

    #[test]
    fn test_free_vars() {
        let typ = LinearType::lollipop(
            LinearType::Var("A".to_string()),
            LinearType::tensor(
                LinearType::Var("B".to_string()),
                LinearType::Var("A".to_string()),
            ),
        );

        let vars = typ.free_vars();
        assert!(vars.contains("A"));
        assert!(vars.contains("B"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_substitute() {
        let typ = LinearType::Var("X".to_string());
        let result = typ.substitute("X", &LinearType::One);
        assert!(matches!(result, LinearType::One));
    }
}

// ============================================================================
// CONTEXT TESTS
// ============================================================================

mod context_tests {
    use super::*;

    #[test]
    fn test_context_linear_binding() {
        let mut ctx = LinearContext::new();
        ctx.add_linear("x", LinearType::One);

        let binding = ctx.lookup("x").unwrap();
        assert_eq!(binding.modality, Modality::Linear);
        assert_eq!(binding.usage, UsageCount::Zero);
    }

    #[test]
    fn test_context_use_var() {
        let mut ctx = LinearContext::new();
        ctx.add_linear("x", LinearType::One);

        // Use once
        ctx.use_var("x").unwrap();
        let binding = ctx.lookup("x").unwrap();
        assert_eq!(binding.usage, UsageCount::One);

        // Try to use again - should fail for linear
        assert!(ctx.use_var("x").is_err());
    }

    #[test]
    fn test_context_unrestricted_reuse() {
        let mut ctx = LinearContext::new();
        ctx.add_unrestricted("x", LinearType::One);

        // Can use multiple times
        ctx.use_var("x").unwrap();
        ctx.use_var("x").unwrap();
        ctx.use_var("x").unwrap();

        assert!(ctx.check_exhausted().is_ok());
    }

    #[test]
    fn test_context_split_unrestricted() {
        let mut ctx = LinearContext::new();
        ctx.add_unrestricted("x", LinearType::One);

        let mut split = ctx.split();
        let (left, right) = split.complete().unwrap();

        // Unrestricted should be in both
        assert!(left.lookup("x").is_some());
        assert!(right.lookup("x").is_some());
    }

    #[test]
    fn test_context_split_linear() {
        let mut ctx = LinearContext::new();
        ctx.add_linear("x", LinearType::One);
        ctx.add_linear("y", LinearType::One);

        let mut split = ctx.split();
        split.assign_left("x");
        split.assign_right("y");
        let (left, right) = split.complete().unwrap();

        assert!(left.lookup("x").is_some());
        assert!(left.lookup("y").is_none());
        assert!(right.lookup("x").is_none());
        assert!(right.lookup("y").is_some());
    }

    #[test]
    fn test_context_exhaustion_linear_unused() {
        let mut ctx = LinearContext::new();
        ctx.add_linear("x", LinearType::One);

        // Linear not used - should fail
        assert!(ctx.check_exhausted().is_err());
    }

    #[test]
    fn test_context_exhaustion_affine_unused() {
        let mut ctx = LinearContext::new();
        ctx.add_affine("x", LinearType::One);

        // Affine not used - should be OK
        assert!(ctx.check_exhausted().is_ok());
    }

    #[test]
    fn test_context_exhaustion_relevant_unused() {
        let mut ctx = LinearContext::new();
        ctx.add_relevant("x", LinearType::One);

        // Relevant not used - should fail
        assert!(ctx.check_exhausted().is_err());
    }
}

// ============================================================================
// EXPONENTIAL TESTS
// ============================================================================

mod exponential_tests {
    use super::*;
    use demetrios::linear::exponentials::*;

    #[test]
    fn test_bang_dereliction() {
        let inner = LinearType::One;
        let bang = BangType::new(inner.clone());
        let derel = bang.dereliction();
        assert!(matches!(derel, LinearType::One));
    }

    #[test]
    fn test_bang_contraction() {
        let inner = LinearType::One;
        let bang = BangType::new(inner);
        let contracted = bang.contraction();

        match contracted {
            LinearType::Tensor(a, b) => {
                assert!(matches!(&*a, LinearType::Bang(_)));
                assert!(matches!(&*b, LinearType::Bang(_)));
            }
            _ => panic!("Expected tensor of bangs"),
        }
    }

    #[test]
    fn test_bang_weakening() {
        let inner = LinearType::One;
        let bang = BangType::new(inner);
        let weak = bang.weakening();
        assert!(matches!(weak, LinearType::One));
    }

    #[test]
    fn test_seely_isomorphism() {
        let a = BangType::new(LinearType::One);
        let b = BangType::new(LinearType::Top);

        // Forward: !A ⊗ !B → !(A & B)
        let combined = seely_forward(&a, &b);
        match &*combined.inner {
            LinearType::With(_, _) => {}
            _ => panic!("Expected With type"),
        }

        // Backward: !(A & B) → !A ⊗ !B
        let (a2, b2) = seely_backward(&combined).unwrap();
        assert!(matches!(&*a2.inner, LinearType::One));
        assert!(matches!(&*b2.inner, LinearType::Top));
    }

    #[test]
    fn test_comonoid_contract_n() {
        let bang = BangType::new(LinearType::One);
        let comonoid = BangComonoid::new(bang);

        // n=0 gives 1
        let zero = comonoid.contract_n(0);
        assert!(matches!(zero, LinearType::One));

        // n=1 gives !A
        let one = comonoid.contract_n(1);
        assert!(matches!(one, LinearType::Bang(_)));

        // n=2 gives !A ⊗ !A
        let two = comonoid.contract_n(2);
        assert!(matches!(two, LinearType::Tensor(_, _)));
    }

    #[test]
    fn test_can_promote() {
        // Unrestricted can be promoted
        let unrest = LinearType::unrestricted_knowledge(Type::Bool, 0.9, OntologyType::Any);
        assert!(BangType::can_promote(&unrest));

        // Linear cannot be promoted
        let linear = LinearType::linear_knowledge(Type::Bool, 0.9, OntologyType::Any);
        assert!(!BangType::can_promote(&linear));

        // Bang can always be promoted
        assert!(BangType::can_promote(&LinearType::bang(LinearType::One)));
    }

    #[test]
    fn test_exponential_coercion() {
        let bang = LinearType::bang(LinearType::One);

        // Dereliction
        let result = ExponentialCoercion::Dereliction.apply(&bang);
        assert!(matches!(result, Some(LinearType::One)));

        // Digging
        let result = ExponentialCoercion::Digging.apply(&bang);
        assert!(matches!(result, Some(LinearType::Bang(_))));

        // Contraction
        let result = ExponentialCoercion::Contraction.apply(&bang);
        assert!(matches!(result, Some(LinearType::Tensor(_, _))));

        // Weakening
        let result = ExponentialCoercion::Weakening.apply(&bang);
        assert!(matches!(result, Some(LinearType::One)));
    }
}

// ============================================================================
// SESSION TYPE TESTS
// ============================================================================

mod session_tests {
    use super::*;

    #[test]
    fn test_session_dual_send_recv() {
        let send = SessionType::send(LinearType::One, SessionType::End);
        let dual = send.dual();

        match dual {
            SessionType::Recv { continuation, .. } => {
                assert!(continuation.is_end());
            }
            _ => panic!("Expected Recv"),
        }
    }

    #[test]
    fn test_session_dual_offer_choose() {
        let offer = SessionType::offer_binary(SessionType::End, SessionType::End);
        let dual = offer.dual();

        match dual {
            SessionType::Choose { .. } => {}
            _ => panic!("Expected Choose"),
        }
    }

    #[test]
    fn test_session_dual_involution() {
        let session = SessionType::send(
            LinearType::One,
            SessionType::recv(LinearType::Top, SessionType::End),
        );

        let dual1 = session.dual();
        let dual2 = dual1.dual();

        assert!(session.definitionally_equal(&dual2));
    }

    #[test]
    fn test_query_response_protocol() {
        let query = LinearType::One;
        let response = LinearType::Top;
        let protocol = SessionType::query_response(query.clone(), response.clone());

        let mut checker = SessionChecker::new(protocol);

        // Send query
        checker.send(&query).unwrap();

        // Receive response
        let received = checker.recv().unwrap();
        assert!(received.definitionally_equal(&response));

        // Close
        checker.close().unwrap();
    }

    #[test]
    fn test_session_checker_wrong_order() {
        let protocol = SessionType::recv(LinearType::One, SessionType::End);
        let mut checker = SessionChecker::new(protocol);

        // Try to send when should receive
        let result = checker.send(&LinearType::One);
        assert!(result.is_err());
    }

    #[test]
    fn test_recursive_session() {
        let stream = SessionType::stream(LinearType::One);

        // Unfold once
        let unfolded = stream.unfold();

        // Should be Offer
        match unfolded {
            SessionType::Offer { branches } => {
                assert_eq!(branches.len(), 2);
            }
            _ => panic!("Expected Offer after unfold"),
        }
    }

    #[test]
    fn test_is_dual() {
        let client = SessionType::send(LinearType::One, SessionType::End);
        let server = SessionType::recv(LinearType::One, SessionType::End);

        assert!(client.is_dual_of(&server));
        assert!(server.is_dual_of(&client));
    }
}

// ============================================================================
// SUBTYPING TESTS
// ============================================================================

mod subtyping_tests {
    use super::*;
    use demetrios::linear::subtyping::*;

    #[test]
    fn test_reflexivity() {
        let checker = LinearSubtypeChecker::new();
        let typ = LinearType::One;
        assert!(checker.is_subtype(&typ, &typ).is_ok());
    }

    #[test]
    fn test_modality_subtyping() {
        let checker = LinearSubtypeChecker::new();

        let linear = LinearType::linear_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let affine = LinearType::affine_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let relevant = LinearType::relevant_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let unrestricted = LinearType::unrestricted_knowledge(Type::Bool, 0.9, OntologyType::Any);

        // Linear <: everything
        assert!(checker.is_subtype(&linear, &affine).is_ok());
        assert!(checker.is_subtype(&linear, &relevant).is_ok());
        assert!(checker.is_subtype(&linear, &unrestricted).is_ok());

        // Everything <: Unrestricted
        assert!(checker.is_subtype(&affine, &unrestricted).is_ok());
        assert!(checker.is_subtype(&relevant, &unrestricted).is_ok());

        // Affine and Relevant incomparable
        assert!(checker.is_subtype(&affine, &relevant).is_err());
        assert!(checker.is_subtype(&relevant, &affine).is_err());
    }

    #[test]
    fn test_confidence_subtyping() {
        let checker = LinearSubtypeChecker::new();

        let high = LinearType::linear_knowledge(Type::Bool, 0.95, OntologyType::Any);
        let low = LinearType::linear_knowledge(Type::Bool, 0.80, OntologyType::Any);

        // High <: Low (more confident is more specific)
        assert!(checker.is_subtype(&high, &low).is_ok());

        // NOT: Low <: High
        assert!(checker.is_subtype(&low, &high).is_err());
    }

    #[test]
    fn test_tensor_covariant() {
        let checker = LinearSubtypeChecker::new();

        let high = LinearType::linear_knowledge(Type::Bool, 0.95, OntologyType::Any);
        let low = LinearType::linear_knowledge(Type::Bool, 0.80, OntologyType::Any);

        let tensor_high = LinearType::tensor(high.clone(), LinearType::One);
        let tensor_low = LinearType::tensor(low.clone(), LinearType::One);

        // Covariant: high <: low ⟹ tensor(high, 1) <: tensor(low, 1)
        assert!(checker.is_subtype(&tensor_high, &tensor_low).is_ok());
    }

    #[test]
    fn test_lollipop_contravariant() {
        let checker = LinearSubtypeChecker::new();

        let high = LinearType::linear_knowledge(Type::Bool, 0.95, OntologyType::Any);
        let low = LinearType::linear_knowledge(Type::Bool, 0.80, OntologyType::Any);

        let f_low = LinearType::lollipop(low.clone(), LinearType::One);
        let f_high = LinearType::lollipop(high.clone(), LinearType::One);

        // Contravariant in domain: low -> 1 <: high -> 1
        assert!(checker.is_subtype(&f_low, &f_high).is_ok());
    }

    #[test]
    fn test_bang_dereliction_subtype() {
        let checker = LinearSubtypeChecker::new();

        let inner = LinearType::One;
        let bang = LinearType::bang(inner.clone());

        // !A <: A
        assert!(checker.is_subtype(&bang, &inner).is_ok());
    }

    #[test]
    fn test_quest_intro_subtype() {
        let checker = LinearSubtypeChecker::new();

        let inner = LinearType::One;
        let quest = LinearType::quest(inner.clone());

        // A <: ?A
        assert!(checker.is_subtype(&inner, &quest).is_ok());
    }

    #[test]
    fn test_top_is_supertype() {
        let checker = LinearSubtypeChecker::new();

        assert!(
            checker
                .is_subtype(&LinearType::One, &LinearType::Top)
                .is_ok()
        );
        assert!(
            checker
                .is_subtype(&LinearType::Zero, &LinearType::Top)
                .is_ok()
        );
    }

    #[test]
    fn test_zero_is_subtype() {
        let checker = LinearSubtypeChecker::new();

        assert!(
            checker
                .is_subtype(&LinearType::Zero, &LinearType::One)
                .is_ok()
        );
        assert!(
            checker
                .is_subtype(&LinearType::Zero, &LinearType::Top)
                .is_ok()
        );
    }

    #[test]
    fn test_gradual_subtyping() {
        let checker = LinearSubtypeChecker::with_gradual(true);

        // With gradual, Unknown matches anything
        assert!(
            checker
                .is_subtype(&LinearType::Unknown, &LinearType::One)
                .is_ok()
        );
        assert!(
            checker
                .is_subtype(&LinearType::One, &LinearType::Unknown)
                .is_ok()
        );
    }

    #[test]
    fn test_variance_compose() {
        assert_eq!(
            Variance::Covariant.compose(Variance::Covariant),
            Variance::Covariant
        );
        assert_eq!(
            Variance::Contravariant.compose(Variance::Contravariant),
            Variance::Covariant
        );
        assert_eq!(
            Variance::Covariant.compose(Variance::Contravariant),
            Variance::Contravariant
        );
        assert_eq!(
            Variance::Invariant.compose(Variance::Covariant),
            Variance::Invariant
        );
    }

    #[test]
    fn test_coercion_compute() {
        let inner = LinearType::One;
        let bang = LinearType::bang(inner.clone());

        let coercion = LinearCoercion::compute(&bang, &inner).unwrap();
        assert!(matches!(coercion, LinearCoercion::Dereliction));
    }
}

// ============================================================================
// CONSUMPTION TRACKING TESTS
// ============================================================================

mod consumption_tests {
    use super::*;

    #[test]
    fn test_linear_consumption() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("x");

        // Can consume once
        assert!(tracker.consume("x").is_ok());

        // Cannot consume again
        assert!(tracker.consume("x").is_err());

        // Final state is valid
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_linear_not_consumed() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("x");

        // Not consuming is an error
        assert!(tracker.check_all_final().is_err());
    }

    #[test]
    fn test_affine_can_be_unused() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_affine("x");

        // Not using is OK for affine
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_affine_cannot_reuse() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_affine("x");

        tracker.consume("x").unwrap();

        // Cannot use again
        assert!(tracker.consume("x").is_err());
    }

    #[test]
    fn test_relevant_must_use() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_relevant("x");

        // Not using is error
        assert!(tracker.check_all_final().is_err());

        // Using makes it valid
        tracker.consume("x").unwrap();
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_relevant_can_reuse() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_relevant("x");

        tracker.consume("x").unwrap();
        tracker.consume("x").unwrap();

        // Multiple use is fine
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_unrestricted_any_usage() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_unrestricted("x");

        // Can use many times
        tracker.consume("x").unwrap();
        tracker.consume("x").unwrap();
        tracker.consume("x").unwrap();

        assert!(tracker.check_all_final().is_ok());

        // Or not at all
        let mut tracker2 = ConsumptionTracker::new();
        tracker2.add_unrestricted("y");
        assert!(tracker2.check_all_final().is_ok());
    }

    #[test]
    fn test_split_tracker() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("x");
        tracker.add_linear("y");
        tracker.add_unrestricted("z");

        let (left, right) = tracker.split(&["x"]).unwrap();

        // x in left, y in right, z in both
        assert!(left.is_available("x"));
        assert!(!left.is_available("y"));
        assert!(left.is_available("z"));

        assert!(!right.is_available("x"));
        assert!(right.is_available("y"));
        assert!(right.is_available("z"));
    }

    #[test]
    fn test_must_consume() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("x");
        tracker.add_affine("y");
        tracker.add_relevant("z");
        tracker.add_unrestricted("w");

        let must = tracker.must_consume();
        assert!(must.contains(&"x"));
        assert!(!must.contains(&"y"));
        assert!(must.contains(&"z"));
        assert!(!must.contains(&"w"));
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_quantum_measurement_pattern() {
        // Quantum measurement is linear - can only measure once
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("quantum_state");

        // Measure (consume)
        tracker.consume("quantum_state").unwrap();

        // Cannot measure again
        assert!(tracker.consume("quantum_state").is_err());

        // Valid final state
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_credential_pattern() {
        // Credential is affine - can expire unused
        let mut tracker = ConsumptionTracker::new();
        tracker.add_affine("credential");

        // Can choose not to use
        assert!(tracker.check_all_final().is_ok());

        // Or can use once
        let mut tracker2 = ConsumptionTracker::new();
        tracker2.add_affine("credential");
        tracker2.consume("credential").unwrap();
        assert!(tracker2.check_all_final().is_ok());
    }

    #[test]
    fn test_mandatory_evidence_pattern() {
        // Regulatory evidence is relevant - must use
        let mut tracker = ConsumptionTracker::new();
        tracker.add_relevant("clinical_data");

        // Must use at least once
        assert!(tracker.check_all_final().is_err());

        tracker.consume("clinical_data").unwrap();
        assert!(tracker.check_all_final().is_ok());

        // Can use multiple times for different analyses
        tracker.consume("clinical_data").unwrap();
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_published_knowledge_pattern() {
        // Published knowledge is unrestricted - use freely
        let mut tracker = ConsumptionTracker::new();
        tracker.add_unrestricted("published_paper");

        // Can cite many times
        for _ in 0..10 {
            tracker.consume("published_paper").unwrap();
        }

        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_session_query_response() {
        // Build a query-response session
        let query_type = LinearType::linear_knowledge(
            Type::Named {
                name: "Query".to_string(),
                args: vec![],
            },
            1.0,
            OntologyType::Any,
        );
        let response_type = LinearType::linear_knowledge(
            Type::Named {
                name: "Response".to_string(),
                args: vec![],
            },
            0.95,
            OntologyType::Any,
        );

        let protocol = SessionType::query_response(query_type.clone(), response_type.clone());
        let mut checker = SessionChecker::new(protocol);

        // Execute protocol
        checker.send(&query_type).unwrap();
        let response = checker.recv().unwrap();
        assert!(response.definitionally_equal(&response_type));
        checker.close().unwrap();
    }

    #[test]
    fn test_client_server_duality() {
        // Client protocol
        let client = SessionType::send(
            LinearType::One,
            SessionType::recv(LinearType::Top, SessionType::End),
        );

        // Server should be dual
        let server = client.dual();

        // Verify duality
        match &server {
            SessionType::Recv { continuation, .. } => match &**continuation {
                SessionType::Send { continuation, .. } => {
                    assert!(continuation.is_end());
                }
                _ => panic!("Expected Send"),
            },
            _ => panic!("Expected Recv"),
        }

        assert!(client.is_dual_of(&server));
    }
}
