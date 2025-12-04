//! Comprehensive tests for Day 35: Dependent Epistemic Types
//!
//! Tests cover:
//! - ConfidenceType evaluation and bounds
//! - OntologyType containment
//! - Predicate logic and normalization
//! - Proof terms and construction
//! - Subtyping rules
//! - Proof search algorithm
//! - Gradual typing fallback
//! - Constraint-based inference

use demetrios::dependent::*;
use demetrios::types::Type;
use std::collections::HashSet;
use std::time::Duration;

// ============================================================================
// CONFIDENCE TYPE TESTS
// ============================================================================

mod confidence_type_tests {
    use super::*;

    #[test]
    fn test_literal_evaluation() {
        let conf = ConfidenceType::literal(0.95);
        let ctx = TypeContext::new();
        assert!((conf.evaluate(&ctx).unwrap() - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_literal_clamping() {
        let over = ConfidenceType::literal(1.5);
        let under = ConfidenceType::literal(-0.5);
        let ctx = TypeContext::new();
        assert!((over.evaluate(&ctx).unwrap() - 1.0).abs() < 1e-10);
        assert!((under.evaluate(&ctx).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_product_evaluation() {
        let c1 = ConfidenceType::literal(0.9);
        let c2 = ConfidenceType::literal(0.8);
        let product = ConfidenceType::product(c1, c2);
        let ctx = TypeContext::new();
        assert!((product.evaluate(&ctx).unwrap() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_dempster_shafer_evaluation() {
        let c1 = ConfidenceType::literal(0.6);
        let c2 = ConfidenceType::literal(0.7);
        let ds = ConfidenceType::dempster_shafer(c1, c2);
        let ctx = TypeContext::new();
        // 1 - (1-0.6)*(1-0.7) = 1 - 0.4*0.3 = 0.88
        assert!((ds.evaluate(&ctx).unwrap() - 0.88).abs() < 1e-10);
    }

    #[test]
    fn test_decay_evaluation() {
        let base = ConfidenceType::literal(1.0);
        let decay = ConfidenceType::decay(base, 0.1, Duration::from_secs(10));
        let ctx = TypeContext::new();
        // e^(-0.1 * 10) = e^(-1) ≈ 0.3679
        let result = decay.evaluate(&ctx).unwrap();
        assert!((result - 0.3679).abs() < 0.01);
    }

    #[test]
    fn test_min_max_evaluation() {
        let c1 = ConfidenceType::literal(0.8);
        let c2 = ConfidenceType::literal(0.9);
        let min = ConfidenceType::min(c1.clone(), c2.clone());
        let max = ConfidenceType::max(c1, c2);
        let ctx = TypeContext::new();
        assert!((min.evaluate(&ctx).unwrap() - 0.8).abs() < 1e-10);
        assert!((max.evaluate(&ctx).unwrap() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_variable_binding() {
        let var = ConfidenceType::var("ε");
        let mut ctx = TypeContext::new();
        assert!(var.evaluate(&ctx).is_none());

        ctx.bind_confidence("ε", ConfidenceType::literal(0.85));
        assert!((var.evaluate(&ctx).unwrap() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_nested_product_with_variable() {
        let mut ctx = TypeContext::new();
        ctx.bind_confidence("α", ConfidenceType::literal(0.9));
        ctx.bind_confidence("β", ConfidenceType::literal(0.8));

        let product = ConfidenceType::product(ConfidenceType::var("α"), ConfidenceType::var("β"));
        assert!((product.evaluate(&ctx).unwrap() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_lower_bound() {
        let ctx = TypeContext::new();

        // Literal has exact bound
        let lit = ConfidenceType::literal(0.9);
        assert!((lit.lower_bound(&ctx).unwrap() - 0.9).abs() < 1e-10);

        // Product of literals
        let product =
            ConfidenceType::product(ConfidenceType::literal(0.8), ConfidenceType::literal(0.9));
        assert!((product.lower_bound(&ctx).unwrap() - 0.72).abs() < 1e-10);

        // Unknown has 0 lower bound
        assert!((ConfidenceType::Unknown.lower_bound(&ctx).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_upper_bound() {
        let ctx = TypeContext::new();

        // Unknown has 1.0 upper bound
        assert!((ConfidenceType::Unknown.upper_bound(&ctx).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_vars() {
        let expr = ConfidenceType::product(
            ConfidenceType::var("α"),
            ConfidenceType::product(ConfidenceType::var("β"), ConfidenceType::literal(0.5)),
        );
        let vars = expr.free_vars();
        assert!(vars.contains("α"));
        assert!(vars.contains("β"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_substitution() {
        let expr = ConfidenceType::product(ConfidenceType::var("ε"), ConfidenceType::literal(0.8));
        let subst = expr.substitute("ε", &ConfidenceType::literal(0.9));
        let ctx = TypeContext::new();
        assert!((subst.evaluate(&ctx).unwrap() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_definitional_equality() {
        let c1 = ConfidenceType::literal(0.95);
        let c2 = ConfidenceType::literal(0.95);
        let c3 = ConfidenceType::literal(0.90);

        assert!(c1.definitionally_equal(&c2));
        assert!(!c1.definitionally_equal(&c3));

        let v1 = ConfidenceType::var("ε");
        let v2 = ConfidenceType::var("ε");
        let v3 = ConfidenceType::var("δ");

        assert!(v1.definitionally_equal(&v2));
        assert!(!v1.definitionally_equal(&v3));
    }
}

// ============================================================================
// ONTOLOGY TYPE TESTS
// ============================================================================

mod ontology_type_tests {
    use super::*;

    #[test]
    fn test_concrete_ontology() {
        let ont = OntologyType::concrete("PKPD");
        assert!(matches!(ont, OntologyType::Concrete { ontology, .. } if ontology == "PKPD"));
    }

    #[test]
    fn test_ontology_containment() {
        let pkpd = OntologyType::concrete("PKPD");
        let chebi = OntologyType::concrete("ChEBI");
        let union = OntologyType::union(pkpd.clone(), chebi.clone());

        // Union contains both
        assert!(union.contains(&pkpd));
        assert!(union.contains(&chebi));

        // Any contains everything
        assert!(OntologyType::Any.contains(&pkpd));

        // Everything contains None
        assert!(pkpd.contains(&OntologyType::None));
    }

    #[test]
    fn test_ontology_to_set() {
        let pkpd = OntologyType::concrete("PKPD");
        let chebi = OntologyType::concrete("ChEBI");
        let union = OntologyType::union(pkpd, chebi);

        let set = union.to_set();
        assert!(set.contains("PKPD"));
        assert!(set.contains("ChEBI"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_ontology_intersection() {
        let pkpd = OntologyType::concrete("PKPD");
        let chebi = OntologyType::concrete("ChEBI");

        // Intersection of different singletons is empty
        let intersection = OntologyType::intersection(pkpd.clone(), chebi.clone());
        assert!(intersection.to_set().is_empty());

        // Intersection with self
        let self_int = OntologyType::intersection(pkpd.clone(), pkpd.clone());
        assert!(self_int.to_set().contains("PKPD"));
    }
}

// ============================================================================
// PREDICATE TESTS
// ============================================================================

mod predicate_tests {
    use super::*;

    #[test]
    fn test_trivial_predicates() {
        assert!(Predicate::true_().is_trivially_true());
        assert!(Predicate::false_().is_trivially_false());
    }

    #[test]
    fn test_confidence_predicate_evaluation() {
        let ctx = TypeContext::new();

        let geq =
            ConfidencePredicate::Geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        assert_eq!(geq.evaluate(&ctx), Some(true));

        let leq =
            ConfidencePredicate::Leq(ConfidenceType::literal(0.80), ConfidenceType::literal(0.90));
        assert_eq!(leq.evaluate(&ctx), Some(true));

        let eq =
            ConfidencePredicate::Eq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.95));
        assert_eq!(eq.evaluate(&ctx), Some(true));
    }

    #[test]
    fn test_predicate_normalization_double_negation() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let double_neg = Predicate::not(Predicate::not(p.clone()));
        let normalized = double_neg.normalize();
        assert_eq!(normalized, p);
    }

    #[test]
    fn test_predicate_normalization_and_true() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let with_true = Predicate::and(p.clone(), Predicate::true_());
        let normalized = with_true.normalize();
        assert_eq!(normalized, p);
    }

    #[test]
    fn test_predicate_normalization_and_false() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let with_false = Predicate::and(p, Predicate::false_());
        let normalized = with_false.normalize();
        assert!(normalized.is_trivially_false());
    }

    #[test]
    fn test_predicate_normalization_or_true() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let with_true = Predicate::or(p, Predicate::true_());
        let normalized = with_true.normalize();
        assert!(normalized.is_trivially_true());
    }

    #[test]
    fn test_predicate_free_vars() {
        let p = Predicate::and(
            Predicate::confidence_geq(ConfidenceType::var("α"), ConfidenceType::literal(0.9)),
            Predicate::confidence_geq(ConfidenceType::var("β"), ConfidenceType::var("α")),
        );
        let vars = p.free_vars();
        assert!(vars.contains("α"));
        assert!(vars.contains("β"));
    }

    #[test]
    fn test_predicate_substitution() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let substituted = p.substitute_confidence("ε", &ConfidenceType::literal(0.97));

        if let PredicateKind::Confidence(ConfidencePredicate::Geq(lhs, _)) = &substituted.kind {
            assert!(matches!(lhs, ConfidenceType::Literal(v) if (*v - 0.97).abs() < 0.001));
        } else {
            panic!("Expected confidence predicate");
        }
    }

    #[test]
    fn test_ontology_predicate() {
        let pkpd = OntologyType::concrete("PKPD");
        let chebi = OntologyType::concrete("ChEBI");
        let union = OntologyType::union(pkpd.clone(), chebi.clone());

        let pred = OntologyPredicate::Superset(union, pkpd);
        let ctx = TypeContext::new();
        assert_eq!(pred.evaluate(&ctx), Some(true));
    }

    #[test]
    fn test_causal_predicate_display() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let pred = CausalPredicate::Identifiable {
            graph,
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        };
        let s = format!("{}", pred);
        assert!(s.contains("identifiable"));
    }
}

// ============================================================================
// PROOF TESTS
// ============================================================================

mod proof_tests {
    use super::*;

    #[test]
    fn test_literal_cmp_proof() {
        let proof = Proof::literal_cmp(0.95, 0.90);
        assert!(proof.is_some());
        let p = proof.unwrap();
        assert!(p.is_static());
    }

    #[test]
    fn test_literal_cmp_fails() {
        let proof = Proof::literal_cmp(0.80, 0.90);
        assert!(proof.is_none());
    }

    #[test]
    fn test_refl_proof() {
        let conf = ConfidenceType::literal(0.95);
        let proof = Proof::refl(conf);
        assert!(proof.is_static());
    }

    #[test]
    fn test_runtime_check_not_static() {
        let pred =
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let proof = Proof::runtime_check(pred);
        assert!(!proof.is_static());
    }

    #[test]
    fn test_and_intro_proof() {
        let p1 = Proof::literal_cmp(0.95, 0.90).unwrap();
        let p2 = Proof::literal_cmp(0.85, 0.80).unwrap();
        let and_proof = Proof::and_intro(p1, p2);
        assert!(and_proof.is_static());
    }

    #[test]
    fn test_arith_derivation_product() {
        let deriv = ArithDerivation::product(0.9, 0.8, 0.7);
        assert_eq!(deriv.steps.len(), 2);
    }

    #[test]
    fn test_arith_derivation_ds() {
        let deriv = ArithDerivation::dempster_shafer(0.6, 0.7, 0.85);
        assert_eq!(deriv.steps.len(), 2);
    }

    #[test]
    fn test_arith_derivation_decay() {
        let deriv = ArithDerivation::decay(1.0, 0.1, 5.0, 0.5);
        assert_eq!(deriv.steps.len(), 2);
    }

    #[test]
    fn test_proof_description() {
        let proof = Proof::literal_cmp(0.95, 0.90).unwrap();
        let desc = proof.describe();
        assert!(desc.contains("literal comparison"));
    }

    #[test]
    fn test_backdoor_proof_simple() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let proof = Proof::backdoor(graph, "X".to_string(), "Y".to_string(), HashSet::new());
        // Simple X → Y should work with empty adjustment
        assert!(proof.is_some());
    }
}

// ============================================================================
// SUBTYPING TESTS
// ============================================================================

mod subtyping_tests {
    use super::*;

    #[test]
    fn test_reflexivity() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let ty = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let result = checker.check(&ty, &ty);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_confidence_covariance() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let high = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let low = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.90),
            OntologyType::concrete("PKPD"),
        );

        // High confidence is subtype of low
        let result = checker.check(&high, &low);
        assert!(result.is_subtype());

        // Low confidence is NOT subtype of high
        let result2 = checker.check(&low, &high);
        assert!(result2.is_not_subtype());
    }

    #[test]
    fn test_ontology_covariance() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let pkpd_chebi = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::union(
                OntologyType::concrete("PKPD"),
                OntologyType::concrete("ChEBI"),
            ),
        );

        let pkpd_only = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let result = checker.check(&pkpd_chebi, &pkpd_only);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_knowledge_hierarchy() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let causal = EpistemicType::causal_knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
            graph,
        );

        let knowledge = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.90),
            OntologyType::concrete("PKPD"),
        );

        // CausalKnowledge <: Knowledge
        let result = checker.check(&causal, &knowledge);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_refinement_weakening() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let base = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let refined = EpistemicType::refinement(
            base.clone(),
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.99)),
        );

        // Refined is subtype of base
        let result = checker.check(&refined, &base);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_gradual_unknown() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx).with_gradual(true);

        let known = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let result = checker.check(&EpistemicType::Unknown, &known);
        assert!(result.is_subtype());

        let result2 = checker.check(&known, &EpistemicType::Unknown);
        assert!(result2.is_subtype());
    }

    #[test]
    fn test_confidence_variable_bounds() {
        let mut ctx = TypeContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.97));

        let checker = SubtypeChecker::new(&ctx);

        let with_var = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::var("ε"),
            OntologyType::concrete("PKPD"),
        );

        let with_literal = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let result = checker.check(&with_var, &with_literal);
        assert!(result.is_subtype());
    }
}

// ============================================================================
// PROOF SEARCH TESTS
// ============================================================================

mod proof_search_tests {
    use super::*;

    #[test]
    fn test_trivial_true() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);
        let result = searcher.search(&Predicate::true_());
        assert!(result.is_proven());
    }

    #[test]
    fn test_trivial_false() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);
        let result = searcher.search(&Predicate::false_());
        assert!(result.is_disproven());
    }

    #[test]
    fn test_literal_confidence_search() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let pred =
            Predicate::confidence_geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_literal_confidence_fails() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let pred =
            Predicate::confidence_geq(ConfidenceType::literal(0.80), ConfidenceType::literal(0.90));
        let result = searcher.search(&pred);
        assert!(result.is_disproven());
    }

    #[test]
    fn test_conjunction_search() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let p1 =
            Predicate::confidence_geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        let p2 =
            Predicate::confidence_geq(ConfidenceType::literal(0.85), ConfidenceType::literal(0.80));
        let pred = Predicate::and(p1, p2);

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_disjunction_search() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let p1 =
            Predicate::confidence_geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        let p2 =
            Predicate::confidence_geq(ConfidenceType::literal(0.70), ConfidenceType::literal(0.90)); // False
        let pred = Predicate::or(p1, p2);

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_variable_with_binding() {
        let mut ctx = TypeContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.97));

        let mut searcher = ProofSearcher::new(&ctx);

        let pred =
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_product_bound_search() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // 0.9 * 0.9 = 0.81 ≥ 0.80
        let pred = Predicate::confidence_geq(
            ConfidenceType::product(ConfidenceType::literal(0.9), ConfidenceType::literal(0.9)),
            ConfidenceType::literal(0.80),
        );
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_ds_bound_search() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // 0.6 ⊕ 0.7 = 0.88 ≥ 0.85
        let pred = Predicate::confidence_geq(
            ConfidenceType::dempster_shafer(
                ConfidenceType::literal(0.6),
                ConfidenceType::literal(0.7),
            ),
            ConfidenceType::literal(0.85),
        );
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_gradual_fallback() {
        let ctx = TypeContext::new();
        let config = ProofSearchConfig {
            allow_gradual: true,
            ..Default::default()
        };
        let mut searcher = ProofSearcher::with_config(&ctx, config);

        let pred = Predicate::confidence_geq(
            ConfidenceType::var("unknown"),
            ConfidenceType::literal(0.95),
        );
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_causal_backdoor_search() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let pred = Predicate::causal(CausalPredicate::Identifiable {
            graph,
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_assumption_in_context() {
        let mut ctx = TypeContext::new();
        let pred =
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        ctx.assume(pred.clone());

        let mut searcher = ProofSearcher::new(&ctx);
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }
}

// ============================================================================
// GRADUAL TYPING TESTS
// ============================================================================

mod gradual_tests {
    use super::*;

    #[test]
    fn test_gradual_mode() {
        assert!(!GradualMode::Strict.allows_runtime_checks());
        assert!(GradualMode::Permissive.allows_runtime_checks());
        assert!(GradualMode::Dynamic.allows_runtime_checks());
    }

    #[test]
    fn test_gradual_config() {
        let strict = GradualConfig::strict();
        assert!(strict.mode.requires_static());

        let permissive = GradualConfig::permissive();
        assert!(permissive.warn_on_runtime);

        let dynamic = GradualConfig::dynamic();
        assert!(dynamic.mode.is_dynamic());
    }

    #[test]
    fn test_runtime_check_code_gen() {
        let pred =
            Predicate::confidence_geq(ConfidenceType::var("k"), ConfidenceType::literal(0.95));
        let check = confidence_check("k", 0.95, pred);
        let code = check.generate_check_code();
        assert!(code.contains("confidence()"));
        assert!(code.contains("0.95"));
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("src/main.d", 42, 10).with_length(5);
        let s = format!("{}", loc);
        assert!(s.contains("src/main.d"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_gradual_warning() {
        let pred =
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let warning = GradualWarning::new(pred, "Variable unbound")
            .with_location(SourceLocation::new("test.d", 10, 5))
            .with_suggestion("Add type annotation");

        let formatted = warning.format();
        assert!(formatted.contains("warning"));
        assert!(formatted.contains("unbound"));
        assert!(formatted.contains("test.d"));
    }

    #[test]
    fn test_gradual_diagnostics() {
        let mut diag = GradualDiagnostics::new();

        let pred = Predicate::true_();
        diag.add_check(confidence_check("k1", 0.90, pred.clone()));
        diag.add_check(confidence_check("k2", 0.95, pred.clone()));
        diag.add_warning(GradualWarning::new(pred, "test"));

        assert_eq!(diag.total_checks(), 2);
        assert_eq!(diag.total_warnings(), 1);
        assert!(diag.has_runtime_checks());

        let summary = diag.summary();
        assert!(summary.contains("Runtime checks: 2"));
    }

    #[test]
    fn test_gradual_annotation_parse() {
        assert_eq!(
            GradualAnnotation::parse("@static_proof"),
            Some(GradualAnnotation::StaticProof)
        );
        assert_eq!(
            GradualAnnotation::parse("allow_runtime"),
            Some(GradualAnnotation::AllowRuntime)
        );
        assert_eq!(
            GradualAnnotation::parse("@trusted"),
            Some(GradualAnnotation::Trusted)
        );
        assert!(GradualAnnotation::parse("unknown").is_none());
    }
}

// ============================================================================
// INFERENCE TESTS
// ============================================================================

mod inference_tests {
    use super::*;

    #[test]
    fn test_fresh_var() {
        let mut ctx = InferenceContext::new();
        let v1 = ctx.fresh_var("T", TypeVarKind::Epistemic);
        let v2 = ctx.fresh_var("T", TypeVarKind::Epistemic);
        assert_ne!(v1.id, v2.id);
    }

    #[test]
    fn test_confidence_constraint_solving() {
        let mut ctx = InferenceContext::new();
        let c1 = ctx.fresh_confidence("ε");
        ctx.confidence_geq(c1.clone(), ConfidenceType::literal(0.95), "test");

        let solver = ConstraintSolver::new(ctx).with_gradual(true);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_var_instantiation() {
        let mut ctx = InferenceContext::new();
        let var = ctx.fresh_var("T", TypeVarKind::Epistemic);

        let ty = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        ctx.add_constraint(Constraint::new(
            ConstraintKind::Instantiate(var.clone(), ty.clone()),
            "test",
        ));

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve().unwrap();

        assert!(result.substitution().contains_key(&var.to_string()));
    }

    #[test]
    fn test_confidence_binding_inference() {
        let mut ctx = InferenceContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.97));

        ctx.confidence_geq(
            ConfidenceType::var("ε"),
            ConfidenceType::literal(0.95),
            "test",
        );

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_subtype_constraint() {
        let mut ctx = InferenceContext::new();

        let high = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.97),
            OntologyType::concrete("PKPD"),
        );

        let low = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        ctx.subtype(high, low, "test");

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_predicate_constraint() {
        let mut ctx = InferenceContext::new();

        ctx.must_prove(
            Predicate::confidence_geq(ConfidenceType::literal(0.97), ConfidenceType::literal(0.95)),
            "test",
        );

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_gradual_allows_unknown() {
        let mut ctx = InferenceContext::new().with_gradual(true);

        ctx.confidence_geq(
            ConfidenceType::var("unknown"),
            ConfidenceType::literal(0.95),
            "test",
        );

        let solver = ConstraintSolver::new(ctx).with_gradual(true);
        let result = solver.solve();
        assert!(result.is_ok());
    }
}

// ============================================================================
// CAUSAL GRAPH TYPE TESTS
// ============================================================================

mod causal_graph_tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let mut graph = CausalGraphType::new();
        graph.add_node("X");
        graph.add_node("Y");
        graph.add_edge("X", "Y");

        assert!(graph.nodes.contains("X"));
        assert!(graph.nodes.contains("Y"));
        assert!(graph.edges.contains(&("X".to_string(), "Y".to_string())));
    }

    #[test]
    fn test_graph_descendants() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "M");
        graph.add_edge("M", "Y");
        graph.add_edge("X", "Y");

        let desc = graph.descendants("X");
        assert!(desc.contains("M"));
        assert!(desc.contains("Y"));
        assert!(!desc.contains("X"));
    }

    #[test]
    fn test_graph_ancestors() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "M");
        graph.add_edge("M", "Y");
        graph.add_edge("U", "Y");

        let anc = graph.ancestors("Y");
        assert!(anc.contains("M"));
        assert!(anc.contains("X"));
        assert!(anc.contains("U"));
    }

    #[test]
    fn test_graph_surgery_incoming() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("U", "X");
        graph.add_edge("X", "Y");
        graph.add_edge("U", "Y");

        // G_X̄: remove incoming to X
        let g_x_bar = graph.remove_incoming("X");
        assert!(!g_x_bar.edges.contains(&("U".to_string(), "X".to_string())));
        assert!(g_x_bar.edges.contains(&("X".to_string(), "Y".to_string())));
    }

    #[test]
    fn test_graph_surgery_outgoing() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("U", "X");
        graph.add_edge("X", "Y");
        graph.add_edge("U", "Y");

        // G_X_: remove outgoing from X
        let g_x_under = graph.remove_outgoing("X");
        assert!(
            g_x_under
                .edges
                .contains(&("U".to_string(), "X".to_string()))
        );
        assert!(
            !g_x_under
                .edges
                .contains(&("X".to_string(), "Y".to_string()))
        );
    }

    #[test]
    fn test_directed_path() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "M");
        graph.add_edge("M", "Y");

        assert!(graph.has_directed_path("X", "Y"));
        assert!(graph.has_directed_path("X", "M"));
        assert!(!graph.has_directed_path("Y", "X"));
    }

    #[test]
    fn test_bidirected_edges() {
        let mut graph = CausalGraphType::new();
        graph.add_bidirected("X", "Y");

        // Should be stored in canonical order
        assert!(
            graph
                .bidirected
                .contains(&("X".to_string(), "Y".to_string()))
                || graph
                    .bidirected
                    .contains(&("Y".to_string(), "X".to_string()))
        );
    }
}

// ============================================================================
// EPISTEMIC TYPE TESTS
// ============================================================================

mod epistemic_type_tests {
    use super::*;

    #[test]
    fn test_knowledge_type_creation() {
        let ty = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        assert!(ty.confidence().is_some());
        assert!(ty.ontology().is_some());
    }

    #[test]
    fn test_causal_knowledge_type() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let ty = EpistemicType::causal_knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
            graph,
        );

        assert!(ty.causal_graph().is_some());
    }

    #[test]
    fn test_refinement_type() {
        let base = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let refined = EpistemicType::refinement(
            base,
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.99)),
        );

        assert!(matches!(refined, EpistemicType::Refinement { .. }));
    }

    #[test]
    fn test_pi_type() {
        let body = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::var("ε"),
            OntologyType::concrete("PKPD"),
        );

        let pi = EpistemicType::pi("ε", Type::F64, body);
        assert!(matches!(pi, EpistemicType::Pi { .. }));
    }

    #[test]
    fn test_sigma_type() {
        let snd = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::var("ε"),
            OntologyType::concrete("PKPD"),
        );

        let sigma = EpistemicType::sigma("ε", Type::F64, snd);
        assert!(matches!(sigma, EpistemicType::Sigma { .. }));
    }

    #[test]
    fn test_knowledge_hierarchy_subtype() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let structural = EpistemicType::StructuralKnowledge {
            inner: std::sync::Arc::new(Type::F64),
            confidence: ConfidenceType::literal(0.95),
            ontology: OntologyType::concrete("PKPD"),
            provenance: ProvenanceType::Unknown,
            temporal: TemporalType::Unknown,
            graph: graph.clone(),
            has_equations: true,
        };

        let causal = EpistemicType::causal_knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
            graph,
        );

        let knowledge = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        assert!(structural.is_knowledge_hierarchy_subtype(&causal));
        assert!(structural.is_knowledge_hierarchy_subtype(&knowledge));
        assert!(causal.is_knowledge_hierarchy_subtype(&knowledge));
        assert!(!knowledge.is_knowledge_hierarchy_subtype(&causal));
    }

    #[test]
    fn test_epistemic_type_display() {
        let ty = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let s = format!("{}", ty);
        assert!(s.contains("Knowledge"));
        assert!(s.contains("0.95"));
        assert!(s.contains("PKPD"));
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

mod integration_tests {
    use super::*;

    /// Test the complete flow: type → predicate → proof search → subtyping
    #[test]
    fn test_safe_extraction_scenario() {
        // Scenario: We have Knowledge[Drug, ε=0.97, δ=PKPD]
        // We want to extract with threshold 0.95
        // This should succeed because 0.97 ≥ 0.95

        let mut ctx = TypeContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.97));

        // Create the knowledge type
        let knowledge_type = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::var("ε"),
            OntologyType::concrete("PKPD"),
        );

        // Create the requirement type (threshold 0.95)
        let required_type = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        // Check subtyping
        let checker = SubtypeChecker::new(&ctx);
        let result = checker.check(&knowledge_type, &required_type);

        assert!(result.is_subtype());
    }

    /// Test causal identifiability with backdoor criterion
    #[test]
    fn test_causal_identifiability_scenario() {
        // Scenario: Drug → Concentration → Effect with confounder U
        // U → Drug, U → Effect (common cause)
        // Backdoor: adjust for U

        let mut graph = CausalGraphType::new();
        graph.add_edge("Drug", "Concentration");
        graph.add_edge("Concentration", "Effect");
        graph.add_edge("U", "Drug");
        graph.add_edge("U", "Effect");
        graph.set_treatment("Drug");
        graph.set_outcome("Effect");

        // Check identifiability via backdoor
        let mut adjustment = HashSet::new();
        adjustment.insert("U".to_string());

        let is_backdoor = CausalPredicate::check_backdoor(&graph, "Drug", "Effect", &adjustment);

        // U blocks the backdoor path Drug ← U → Effect
        assert!(is_backdoor);
    }

    /// Test temporal decay with proof
    #[test]
    fn test_temporal_decay_scenario() {
        // Scenario: Knowledge created at t=0 with ε=0.95
        // Decay formula: ε' = ε * e^(-λ * t) where t is in seconds
        // For λ = 0.1/year = 0.1 / (365*24*60*60) per second ≈ 3.17e-9/sec
        // After 2 years: ε' = 0.95 * e^(-0.1 * 2) ≈ 0.78

        // Lambda is per-second since elapsed.as_secs_f64() is used in evaluate
        let seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0;
        let lambda_per_second = 0.1 / seconds_per_year;
        let elapsed_seconds = 2.0 * seconds_per_year;

        let decayed = ConfidenceType::decay(
            ConfidenceType::literal(0.95),
            lambda_per_second,
            Duration::from_secs_f64(elapsed_seconds),
        );

        let ctx = TypeContext::new();
        let result = decayed.evaluate(&ctx).unwrap();

        // e^(-0.2) ≈ 0.819, so 0.95 * 0.819 ≈ 0.778
        assert!(result >= 0.75, "Expected result >= 0.75, got {}", result);
        assert!(result < 0.85, "Expected result < 0.85, got {}", result);
    }

    /// Test composition with Dempster-Shafer
    #[test]
    fn test_ds_composition_scenario() {
        // Scenario: Two independent sources with ε₁=0.7, ε₂=0.8
        // Combined via DS: 1 - (1-0.7)(1-0.8) = 1 - 0.3*0.2 = 0.94
        // Check if combined ≥ 0.90

        let ds = ConfidenceType::dempster_shafer(
            ConfidenceType::literal(0.7),
            ConfidenceType::literal(0.8),
        );

        let ctx = TypeContext::new();
        let result = ds.evaluate(&ctx).unwrap();

        assert!((result - 0.94).abs() < 0.01);

        // Proof search should find this
        let pred = Predicate::confidence_geq(ds, ConfidenceType::literal(0.90));
        let mut searcher = ProofSearcher::new(&ctx);
        let proof_result = searcher.search(&pred);

        assert!(proof_result.is_proven());
    }

    /// Test inference with multiple constraints
    #[test]
    fn test_multi_constraint_inference() {
        let mut ctx = InferenceContext::new();

        // Create several type variables
        let alpha = ctx.fresh_confidence("α");
        let beta = ctx.fresh_confidence("β");

        // Add constraints: α ≥ 0.90, β ≥ α, β ≥ 0.95
        ctx.confidence_geq(alpha.clone(), ConfidenceType::literal(0.90), "minimum α");
        ctx.confidence_geq(beta.clone(), alpha.clone(), "β ≥ α");
        ctx.confidence_geq(beta.clone(), ConfidenceType::literal(0.95), "β threshold");

        // With gradual typing, this should succeed
        let solver = ConstraintSolver::new(ctx).with_gradual(true);
        let result = solver.solve();

        assert!(result.is_ok());
    }
}
