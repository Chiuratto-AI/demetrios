//! Integration tests for the refinement type system
//!
//! Tests the predicate language, constraint generation, subtype checking,
//! qualifiers, and inference without requiring Z3.

use demetrios::refinement::qualifiers::{QualifierCategory, QualifierSet};
use demetrios::refinement::subtype::SubtypeResult;
use demetrios::refinement::*;
use demetrios::types::Type;

// ============================================================================
// Predicate Tests
// ============================================================================

mod predicate_tests {
    use super::*;

    #[test]
    fn test_trivial_refinement() {
        let ty = RefinementType::trivial(Type::I64);
        assert!(!ty.is_refined());
        assert_eq!(ty.predicate, Predicate::True);
        assert_eq!(ty.var, "v");
    }

    #[test]
    fn test_positive_refinement() {
        let ty = RefinementType::positive(Type::I64);
        assert!(ty.is_refined());

        // Should have form: v > 0
        let display = format!("{}", ty);
        assert!(display.contains("v"));
        assert!(display.contains(">"));
    }

    #[test]
    fn test_non_negative_refinement() {
        let ty = RefinementType::non_negative(Type::I64);
        assert!(ty.is_refined());

        // Should have form: v >= 0
        let display = format!("{}", ty);
        assert!(display.contains("v"));
        assert!(display.contains(">="));
    }

    #[test]
    fn test_bounded_refinement() {
        let ty = RefinementType::bounded(Type::F64, 0.0, 100.0);
        assert!(ty.is_refined());

        // Should have form: 0 <= v <= 100
        let vars = ty.free_vars();
        assert!(vars.is_empty()); // v is bound
    }

    #[test]
    fn test_predicate_and_simplification() {
        // true && P = P
        let p = Predicate::gt(Term::var("x"), Term::int(0));
        let result = Predicate::and([Predicate::True, p.clone()]);
        assert_eq!(result, p);

        // P && false = false
        let result = Predicate::and([p.clone(), Predicate::False]);
        assert_eq!(result, Predicate::False);

        // true && true = true
        let result = Predicate::and([Predicate::True, Predicate::True]);
        assert_eq!(result, Predicate::True);
    }

    #[test]
    fn test_predicate_or_simplification() {
        // false || P = P
        let p = Predicate::gt(Term::var("x"), Term::int(0));
        let result = Predicate::or([Predicate::False, p.clone()]);
        assert_eq!(result, p);

        // P || true = true
        let result = Predicate::or([p.clone(), Predicate::True]);
        assert_eq!(result, Predicate::True);

        // false || false = false
        let result = Predicate::or([Predicate::False, Predicate::False]);
        assert_eq!(result, Predicate::False);
    }

    #[test]
    fn test_predicate_implies_simplification() {
        // true => P = P
        let p = Predicate::gt(Term::var("x"), Term::int(0));
        let result = Predicate::implies(Predicate::True, p.clone());
        assert_eq!(result, p);

        // false => P = true
        let result = Predicate::implies(Predicate::False, p.clone());
        assert_eq!(result, Predicate::True);

        // P => true = true
        let result = Predicate::implies(p.clone(), Predicate::True);
        assert_eq!(result, Predicate::True);
    }

    #[test]
    fn test_term_substitution() {
        let term = Term::add(Term::var("x"), Term::int(1));
        let result = term.substitute("x", &Term::int(5));

        match result {
            Term::BinOp(BinOp::Add, lhs, rhs) => {
                assert_eq!(*lhs, Term::Int(5));
                assert_eq!(*rhs, Term::Int(1));
            }
            _ => panic!("Expected BinOp"),
        }
    }

    #[test]
    fn test_predicate_substitution() {
        let pred = Predicate::gt(Term::var("x"), Term::int(0));
        let result = pred.substitute("x", &Term::var("y"));

        assert_eq!(result, Predicate::gt(Term::var("y"), Term::int(0)));
    }

    #[test]
    fn test_refinement_type_substitution() {
        let ty = RefinementType::positive(Type::I64);
        let subst = ty.substitute("v", &Term::var("x"));

        // The refinement variable should still be "v"
        assert_eq!(subst.var, "v");
    }

    #[test]
    fn test_free_vars() {
        let pred = Predicate::and([
            Predicate::gt(Term::var("x"), Term::int(0)),
            Predicate::lt(Term::var("y"), Term::var("z")),
        ]);

        let vars = pred.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_compare_op_negate() {
        assert_eq!(CompareOp::Eq.negate(), CompareOp::Ne);
        assert_eq!(CompareOp::Ne.negate(), CompareOp::Eq);
        assert_eq!(CompareOp::Lt.negate(), CompareOp::Ge);
        assert_eq!(CompareOp::Le.negate(), CompareOp::Gt);
        assert_eq!(CompareOp::Gt.negate(), CompareOp::Le);
        assert_eq!(CompareOp::Ge.negate(), CompareOp::Lt);
    }

    #[test]
    fn test_compare_op_flip() {
        assert_eq!(CompareOp::Eq.flip(), CompareOp::Eq);
        assert_eq!(CompareOp::Lt.flip(), CompareOp::Gt);
        assert_eq!(CompareOp::Le.flip(), CompareOp::Ge);
        assert_eq!(CompareOp::Gt.flip(), CompareOp::Lt);
        assert_eq!(CompareOp::Ge.flip(), CompareOp::Le);
    }

    #[test]
    fn test_term_is_constant() {
        assert!(Term::int(42).is_constant());
        assert!(Term::float(3.14).is_constant());
        assert!(Term::Bool(true).is_constant());
        assert!(!Term::var("x").is_constant());

        let expr = Term::add(Term::int(1), Term::int(2));
        assert!(expr.is_constant());

        let expr = Term::add(Term::var("x"), Term::int(2));
        assert!(!expr.is_constant());
    }
}

// ============================================================================
// Medical Refinement Tests
// ============================================================================

mod medical_tests {
    use super::*;
    use demetrios::refinement::predicate::medical;

    #[test]
    fn test_positive_dose() {
        let ty = medical::positive(Type::F64);
        assert!(ty.is_refined());
    }

    #[test]
    fn test_safe_dose() {
        let ty = medical::safe_dose(Type::F64, 1000.0);
        assert!(ty.is_refined());
        assert_eq!(ty.var, "dose");
    }

    #[test]
    fn test_valid_crcl() {
        let ty = medical::valid_crcl(Type::F64);
        assert!(ty.is_refined());
        assert_eq!(ty.var, "crcl");
    }

    #[test]
    fn test_valid_age() {
        let ty = medical::valid_age(Type::F64);
        assert!(ty.is_refined());
        // Should be bounded 0 to 150
    }

    #[test]
    fn test_valid_weight() {
        let ty = medical::valid_weight(Type::F64);
        assert!(ty.is_refined());
        assert_eq!(ty.var, "weight");
    }

    #[test]
    fn test_valid_serum_creatinine() {
        let ty = medical::valid_serum_creatinine(Type::F64);
        assert!(ty.is_refined());
        // Should be bounded 0.1 to 20
    }

    #[test]
    fn test_therapeutic_range() {
        // Vancomycin trough: 10-20 mg/L
        let ty = medical::therapeutic_range(Type::F64, 10.0, 20.0);
        assert!(ty.is_refined());
    }

    #[test]
    fn test_probability() {
        let ty = medical::probability(Type::F64);
        assert!(ty.is_refined());
        // Should be bounded 0 to 1
    }

    #[test]
    fn test_adjustment_factor() {
        let ty = medical::adjustment_factor(Type::F64);
        assert!(ty.is_refined());
        assert_eq!(ty.var, "factor");
    }

    #[test]
    fn test_valid_heart_rate() {
        let ty = medical::valid_heart_rate(Type::F64);
        assert!(ty.is_refined());
    }

    #[test]
    fn test_valid_blood_pressure() {
        let sys = medical::valid_systolic_bp(Type::F64);
        let dia = medical::valid_diastolic_bp(Type::F64);

        assert!(sys.is_refined());
        assert!(dia.is_refined());
    }

    #[test]
    fn test_valid_temperature() {
        let ty = medical::valid_temperature(Type::F64);
        assert!(ty.is_refined());
    }
}

// ============================================================================
// Array Refinement Tests
// ============================================================================

mod array_tests {
    use super::*;
    use demetrios::refinement::predicate::array;

    #[test]
    fn test_non_empty() {
        let pred = array::non_empty();
        assert!(pred.free_vars().contains("arr"));
    }

    #[test]
    fn test_min_length() {
        let pred = array::min_length(5);
        assert!(pred.free_vars().contains("arr"));
    }

    #[test]
    fn test_exact_length() {
        let pred = array::exact_length(10);
        assert!(pred.free_vars().contains("arr"));
    }

    #[test]
    fn test_valid_index() {
        let pred = array::valid_index("arr", "i");
        let vars = pred.free_vars();
        assert!(vars.contains("arr"));
        assert!(vars.contains("i"));
    }

    #[test]
    fn test_bounded_index() {
        let pred = array::bounded_index("i", 100);
        assert!(pred.free_vars().contains("i"));
    }
}

// ============================================================================
// Constraint Generation Tests
// ============================================================================

mod constraint_tests {
    use super::*;

    #[test]
    fn test_constraint_generator_new() {
        let cg = ConstraintGenerator::new();
        assert!(cg.is_empty());
    }

    #[test]
    fn test_push_pop_binding() {
        let mut cg = ConstraintGenerator::new();

        cg.push_binding("x", RefinementType::positive(Type::I64));
        assert!(cg.lookup_binding("x").is_some());

        cg.pop_binding();
        assert!(cg.lookup_binding("x").is_none());
    }

    #[test]
    fn test_fresh_variable() {
        let mut cg = ConstraintGenerator::new();

        let v1 = cg.fresh("temp");
        let v2 = cg.fresh("temp");

        assert_ne!(v1, v2);
        assert!(v1.starts_with("temp_"));
        assert!(v2.starts_with("temp_"));
    }

    #[test]
    fn test_add_subtype_constraint() {
        let mut cg = ConstraintGenerator::new();

        let pos = RefinementType::positive(Type::I64);
        let non_neg = RefinementType::non_negative(Type::I64);

        cg.add_subtype(&pos, &non_neg, Span::dummy());

        assert_eq!(cg.len(), 1);
    }

    #[test]
    fn test_trivial_subtype_not_generated() {
        let mut cg = ConstraintGenerator::new();

        let trivial = RefinementType::trivial(Type::I64);
        cg.add_subtype(&trivial, &trivial, Span::dummy());

        assert!(cg.is_empty());
    }

    #[test]
    fn test_bounds_check_constraint() {
        let mut cg = ConstraintGenerator::new();

        cg.add_bounds_check(Term::var("i"), Term::var("len"), Span::dummy());

        assert_eq!(cg.len(), 1);

        let constraint = &cg.constraints()[0];
        assert!(matches!(
            constraint.reason,
            ConstraintReason::BoundsCheck { .. }
        ));
    }

    #[test]
    fn test_division_check_constraint() {
        let mut cg = ConstraintGenerator::new();

        cg.add_division_check(Term::var("x"), Span::dummy());

        assert_eq!(cg.len(), 1);

        let constraint = &cg.constraints()[0];
        assert!(matches!(
            constraint.reason,
            ConstraintReason::DivisionCheck { .. }
        ));
    }

    #[test]
    fn test_path_conditions() {
        let mut cg = ConstraintGenerator::new();

        let cond = Predicate::gt(Term::var("x"), Term::int(0));
        cg.push_path_condition(cond);

        let env = cg.current_env();
        assert!(!env.is_empty());

        cg.pop_path_condition();
        let env = cg.current_env();
        assert!(env.is_empty());
    }

    #[test]
    fn test_constraint_as_implication() {
        let mut cg = ConstraintGenerator::new();

        let pos = RefinementType::positive(Type::I64);
        cg.push_binding("x", pos);

        cg.add_assertion(
            Predicate::ge(Term::var("x"), Term::int(0)),
            "x is non-negative",
            Span::dummy(),
        );

        let constraint = &cg.constraints()[0];
        let impl_pred = constraint.as_implication();

        assert!(matches!(impl_pred, Predicate::Implies(_, _)));
    }
}

// ============================================================================
// Subtype Checker Tests
// ============================================================================

mod subtype_tests {
    use super::*;

    #[test]
    fn test_subtype_checker_new() {
        let checker = SubtypeChecker::new();
        assert_eq!(checker.pending_constraints(), 0);
    }

    #[test]
    fn test_trivial_subtype() {
        let mut checker = SubtypeChecker::new();

        let trivial = RefinementType::trivial(Type::I64);
        assert!(checker.is_subtype(&trivial, &trivial, Span::dummy()));

        let result = checker.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_base_type_mismatch() {
        let mut checker = SubtypeChecker::new();

        let int_pos = RefinementType::positive(Type::I64);
        let float_pos = RefinementType::positive(Type::F64);

        assert!(!checker.is_subtype(&int_pos, &float_pos, Span::dummy()));
    }

    #[test]
    fn test_check_value_positive() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);
        checker.check_value(&Term::int(42), &pos, Span::dummy());

        let result = checker.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_check_value_negative() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);
        checker.check_value(&Term::int(-5), &pos, Span::dummy());

        let result = checker.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_value_zero() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);
        checker.check_value(&Term::int(0), &pos, Span::dummy());

        let result = checker.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_assumptions() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);
        checker.assume("x", pos);

        // After assumption, constraint count unchanged
        assert_eq!(checker.pending_constraints(), 0);

        checker.unassume();
    }

    #[test]
    fn test_branch_conditions() {
        let mut checker = SubtypeChecker::new();

        checker.enter_branch(Predicate::gt(Term::var("x"), Term::int(0)));
        checker.exit_branch();

        // No constraints added by just entering/exiting branches
        assert_eq!(checker.pending_constraints(), 0);
    }

    #[test]
    fn test_subtype_result_counts() {
        let result = SubtypeResult {
            valid: true,
            constraints: vec![],
            results: vec![
                VerifyResult::Valid,
                VerifyResult::Valid,
                VerifyResult::Invalid {
                    constraint_idx: 2,
                    counterexample: None,
                },
                VerifyResult::Unknown {
                    constraint_idx: 3,
                    reason: "timeout".to_string(),
                },
            ],
            errors: vec![],
        };

        assert_eq!(result.num_valid(), 2);
        assert_eq!(result.num_invalid(), 1);
        assert_eq!(result.num_unknown(), 1);
    }
}

// ============================================================================
// Qualifier Tests
// ============================================================================

mod qualifier_tests {
    use super::*;

    #[test]
    fn test_standard_qualifiers_exist() {
        let qualifiers = standard_qualifiers();
        assert!(qualifiers.len() >= 15);

        // Check for essential qualifiers
        let names: Vec<_> = qualifiers.iter().map(|q| q.name.as_str()).collect();
        assert!(names.contains(&"Pos"));
        assert!(names.contains(&"NonNeg"));
        assert!(names.contains(&"Zero"));
        assert!(names.contains(&"EqVar"));
        assert!(names.contains(&"LtVar"));
    }

    #[test]
    fn test_medical_qualifiers_exist() {
        let qualifiers = medical_qualifiers();
        assert!(qualifiers.len() >= 10);

        // All should be medical category
        for q in &qualifiers {
            assert_eq!(q.category, QualifierCategory::Medical);
        }
    }

    #[test]
    fn test_qualifier_instantiation() {
        let q = Qualifier::basic(
            "LtVar",
            vec!["v", "x"],
            Predicate::lt(Term::var("v"), Term::var("x")),
        );

        let result = q.instantiate(&[Term::var("y"), Term::var("z")]);
        assert!(result.is_some());

        let pred = result.unwrap();
        assert_eq!(pred, Predicate::lt(Term::var("y"), Term::var("z")));
    }

    #[test]
    fn test_qualifier_wrong_arity() {
        let q = Qualifier::basic(
            "LtVar",
            vec!["v", "x"],
            Predicate::lt(Term::var("v"), Term::var("x")),
        );

        let result = q.instantiate(&[Term::var("y")]);
        assert!(result.is_none());
    }

    #[test]
    fn test_qualifier_set_standard() {
        let set = QualifierSet::standard();
        assert!(!set.is_empty());
        assert!(set.len() >= 15);
    }

    #[test]
    fn test_qualifier_set_with_medical() {
        let set = QualifierSet::with_medical();
        let standard = QualifierSet::standard();

        assert!(set.len() > standard.len());
    }

    #[test]
    fn test_qualifier_set_by_category() {
        let set = QualifierSet::with_medical();

        let basic = set.by_category(QualifierCategory::Basic);
        let medical = set.by_category(QualifierCategory::Medical);

        assert!(!basic.is_empty());
        assert!(!medical.is_empty());
    }

    #[test]
    fn test_single_param_qualifier() {
        let q = Qualifier::basic(
            "Pos",
            vec!["v"],
            Predicate::gt(Term::var("v"), Term::int(0)),
        );

        let v = Term::var("x");
        let vars = vec![Term::var("a"), Term::var("b")];

        let instantiations = q.all_instantiations(&v, &vars);

        assert_eq!(instantiations.len(), 1);
        assert_eq!(
            instantiations[0],
            Predicate::gt(Term::var("x"), Term::int(0))
        );
    }
}

// ============================================================================
// Inference Tests
// ============================================================================

mod inference_tests {
    use super::*;

    #[test]
    fn test_inference_new() {
        let infer = RefinementInference::new();
        let stats = infer.stats();

        assert_eq!(stats.num_variables, 0);
        assert!(stats.num_qualifiers > 0);
    }

    #[test]
    fn test_inference_with_medical() {
        let infer = RefinementInference::with_medical();
        let standard = RefinementInference::new();

        assert!(infer.stats().num_qualifiers > standard.stats().num_qualifiers);
    }

    #[test]
    fn test_infer_int_literal() {
        let mut infer = RefinementInference::new();
        let ty = infer.infer_literal(infer::LiteralValue::Int(42));

        assert!(ty.is_refined());
        assert_eq!(ty.base, Type::I64);
    }

    #[test]
    fn test_infer_float_literal() {
        let mut infer = RefinementInference::new();
        let ty = infer.infer_literal(infer::LiteralValue::Float(3.14));

        assert!(ty.is_refined());
        assert_eq!(ty.base, Type::F64);
    }

    #[test]
    fn test_infer_bool_literal() {
        let mut infer = RefinementInference::new();
        let ty = infer.infer_literal(infer::LiteralValue::Bool(true));

        assert!(ty.is_refined());
        assert_eq!(ty.base, Type::Bool);
    }

    #[test]
    fn test_add_binding() {
        let mut infer = RefinementInference::new();
        infer.add_binding("x", Type::I64);

        let ty = infer.lookup("x");
        assert!(ty.is_some());
        assert!(!ty.unwrap().is_refined());
    }

    #[test]
    fn test_add_refined_binding() {
        let mut infer = RefinementInference::new();
        let pos = RefinementType::positive(Type::I64);
        infer.add_refined_binding("x", pos);

        let ty = infer.lookup("x").unwrap();
        assert!(ty.is_refined());
    }

    #[test]
    fn test_solve() {
        let mut infer = RefinementInference::new();

        let pos = RefinementType::positive(Type::I64);
        infer.add_refined_binding("x", pos);

        let result = infer.solve();

        assert!(result.has_refinements());
        assert!(result.get("x").is_some());
    }

    #[test]
    fn test_patterns_array_index() {
        let ty = infer::patterns::array_index("i", Term::int(10));

        assert!(ty.is_refined());
        assert_eq!(ty.var, "i");
    }

    #[test]
    fn test_patterns_loop_counter() {
        let ty = infer::patterns::loop_counter("i", Term::var("n"));

        assert!(ty.is_refined());
        assert_eq!(ty.var, "i");
    }

    #[test]
    fn test_patterns_safe_divisor() {
        let ty = infer::patterns::safe_divisor("x", Type::I64);

        assert!(ty.is_refined());
    }

    #[test]
    fn test_patterns_dose_result() {
        let ty = infer::patterns::dose_result(1000.0);

        assert!(ty.is_refined());
        assert_eq!(ty.var, "dose");
    }
}

// ============================================================================
// Simple Checker Tests
// ============================================================================

mod simple_checker_tests {
    use super::*;
    use demetrios::refinement::solver::SimpleChecker;

    #[test]
    fn test_check_true() {
        assert_eq!(SimpleChecker::check(&Predicate::True), Some(true));
    }

    #[test]
    fn test_check_false() {
        assert_eq!(SimpleChecker::check(&Predicate::False), Some(false));
    }

    #[test]
    fn test_check_int_comparison() {
        let pred = Predicate::lt(Term::int(5), Term::int(10));
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        let pred = Predicate::gt(Term::int(5), Term::int(10));
        assert_eq!(SimpleChecker::check(&pred), Some(false));

        let pred = Predicate::eq(Term::int(5), Term::int(5));
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        let pred = Predicate::ne(Term::int(5), Term::int(10));
        assert_eq!(SimpleChecker::check(&pred), Some(true));
    }

    #[test]
    fn test_check_and() {
        let pred = Predicate::and([
            Predicate::lt(Term::int(5), Term::int(10)),
            Predicate::gt(Term::int(5), Term::int(0)),
        ]);
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        let pred = Predicate::and([
            Predicate::lt(Term::int(5), Term::int(10)),
            Predicate::lt(Term::int(5), Term::int(0)),
        ]);
        assert_eq!(SimpleChecker::check(&pred), Some(false));
    }

    #[test]
    fn test_check_or() {
        let pred = Predicate::or([
            Predicate::lt(Term::int(5), Term::int(10)),
            Predicate::lt(Term::int(5), Term::int(0)),
        ]);
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        let pred = Predicate::or([
            Predicate::gt(Term::int(5), Term::int(10)),
            Predicate::lt(Term::int(5), Term::int(0)),
        ]);
        assert_eq!(SimpleChecker::check(&pred), Some(false));
    }

    #[test]
    fn test_check_implies() {
        // false => anything = true
        let pred = Predicate::implies(Predicate::False, Predicate::False);
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        // true => true = true
        let pred = Predicate::implies(Predicate::True, Predicate::True);
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        // true => false = false
        let pred = Predicate::implies(Predicate::True, Predicate::False);
        assert_eq!(SimpleChecker::check(&pred), Some(false));
    }

    #[test]
    fn test_check_not() {
        let pred = Predicate::not(Predicate::True);
        assert_eq!(SimpleChecker::check(&pred), Some(false));

        let pred = Predicate::not(Predicate::False);
        assert_eq!(SimpleChecker::check(&pred), Some(true));
    }

    #[test]
    fn test_check_with_variables() {
        // Can't determine result with variables
        let pred = Predicate::gt(Term::var("x"), Term::int(0));
        assert_eq!(SimpleChecker::check(&pred), None);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_positive_subtype_non_negative() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);
        let non_neg = RefinementType::non_negative(Type::I64);

        // x > 0 implies x >= 0
        assert!(checker.is_subtype(&pos, &non_neg, Span::dummy()));
    }

    #[test]
    fn test_bounded_subtype_non_negative() {
        let mut checker = SubtypeChecker::new();

        // 0 <= x <= 100 implies x >= 0
        let bounded = RefinementType::bounded(Type::F64, 0.0, 100.0);
        let non_neg = RefinementType::non_negative(Type::F64);

        assert!(checker.is_subtype(&bounded, &non_neg, Span::dummy()));
    }

    #[test]
    fn test_medical_dose_workflow() {
        let mut checker = SubtypeChecker::new();

        // Simulate dose calculation
        // Input: weight > 0
        let pos_weight = medical::positive(Type::F64);
        checker.assume("weight", pos_weight);

        // Input: dose_per_kg in safe range
        let safe_dose_per_kg = medical::safe_dose(Type::F64, 20.0);
        checker.assume("dose_per_kg", safe_dose_per_kg);

        // The product should be positive
        // This would be checked by the SMT solver
        assert_eq!(checker.pending_constraints(), 0);
    }

    #[test]
    fn test_array_bounds_workflow() {
        let mut checker = SubtypeChecker::new();

        // Array length is positive
        checker.assume("len", RefinementType::positive(Type::I64));

        // Index is bounded: 0 <= i < len
        let idx_type = RefinementType::refined(
            Type::I64,
            "i",
            Predicate::and([
                Predicate::ge(Term::var("i"), Term::int(0)),
                Predicate::lt(Term::var("i"), Term::var("len")),
            ]),
        );
        checker.assume("i", idx_type);

        // Check bounds
        checker.check_bounds(Term::var("i"), Term::var("len"), Span::dummy());

        assert_eq!(checker.pending_constraints(), 1);
    }

    #[test]
    fn test_function_signature_verification() {
        // Define a function: fn positive_only(x: {v | v > 0}) -> {v | v >= 0}
        let params = vec![("x".to_string(), RefinementType::positive(Type::I64))];
        let return_type = RefinementType::non_negative(Type::I64);

        // Body returns x directly (which is positive)
        let body_type = RefinementType::positive(Type::I64);

        let result =
            subtype::check_function_signature(&params, &return_type, &body_type, "positive_only");

        // Should have one constraint: positive <: non_negative
        assert_eq!(result.num_constraints(), 1);
    }
}
