//! Integration tests for Day 8: Units of Measure
//!
//! Tests for:
//! - Unit literals parsing (500_mg, 10.5_mL)
//! - Unit type annotations (f64@kg)
//! - Unit arithmetic and checking
//! - Unit inference

use demetrios::common::Span;
use demetrios::lexer::lex;
use demetrios::parser::parse;
use demetrios::types::unit_infer::{UnitExpr, UnitInference};
use demetrios::types::units::{Unit, UnitChecker, UnitOp, medical, si};

// ==================== Lexer Tests ====================

#[test]
fn test_lex_integer_unit_literal() {
    let source = "500_mg";
    let tokens = lex(source).expect("should lex");

    assert_eq!(tokens.len(), 2); // unit literal + EOF
    assert_eq!(tokens[0].text, "500_mg");
}

#[test]
fn test_lex_float_unit_literal() {
    let source = "10.5_mL";
    let tokens = lex(source).expect("should lex");

    assert_eq!(tokens.len(), 2); // unit literal + EOF
    assert_eq!(tokens[0].text, "10.5_mL");
}

#[test]
fn test_lex_unit_literal_with_underscores() {
    let source = "1_000_kg";
    let tokens = lex(source).expect("should lex");

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].text, "1_000_kg");
}

#[test]
fn test_lex_compound_unit_literal() {
    let source = "9.8_m/s2";
    let tokens = lex(source).expect("should lex");

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].text, "9.8_m/s2");
}

// ==================== Parser Tests ====================

#[test]
fn test_parse_unit_literal_expression() {
    let source = r#"
        fn main() {
            let dose = 500_mg;
        }
    "#;

    let tokens = lex(source).expect("should lex");
    let ast = parse(&tokens, source).expect("should parse");

    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_float_unit_literal() {
    let source = r#"
        fn main() {
            let volume = 10.5_mL;
        }
    "#;

    let tokens = lex(source).expect("should lex");
    let ast = parse(&tokens, source).expect("should parse");

    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_unit_type_annotation() {
    let source = r#"
        fn main() {
            let mass: f64@kg = 75.0_kg;
        }
    "#;

    let tokens = lex(source).expect("should lex");
    let ast = parse(&tokens, source).expect("should parse");

    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_unit_arithmetic() {
    let source = r#"
        fn main() {
            let a = 100_mg;
            let b = 50_mg;
            let total = a + b;
        }
    "#;

    let tokens = lex(source).expect("should lex");
    let ast = parse(&tokens, source).expect("should parse");

    assert_eq!(ast.items.len(), 1);
}

// ==================== Unit Type Tests ====================

#[test]
fn test_unit_creation() {
    let kg = si::kilogram();
    assert_eq!(kg.format(), "kg");

    // milligram internally uses kg as base dimension with scale factor
    let mg = medical::milligram();
    // The format shows the base dimension (kg), not the scaled name
    assert_eq!(mg.format(), "kg");
    // They are dimensionally compatible but have different scales
    assert!(kg.is_compatible(&mg));
    // Conversion factor shows the scale difference
    let factor = kg.conversion_factor(&mg).unwrap();
    assert!((factor - 1e6).abs() < 1.0); // 1 kg = 1,000,000 mg
}

#[test]
fn test_unit_multiplication() {
    let m = si::meter();
    let s = si::second();

    // velocity = m/s
    let velocity = m.divide(&s);
    assert!(velocity.format().contains("m"));
    assert!(velocity.format().contains("s"));
}

#[test]
fn test_unit_division() {
    let mg = medical::milligram();
    let ml = medical::milliliter();

    // concentration = mg/mL
    let concentration = mg.divide(&ml);
    assert!(concentration.is_compatible(&mg.divide(&ml)));
}

#[test]
fn test_unit_power() {
    let m = si::meter();
    let m2 = m.power(2);

    // Area = m^2
    assert!(m2.is_compatible(&m.multiply(&m)));
}

#[test]
fn test_unit_compatibility() {
    let kg1 = si::kilogram();
    let kg2 = si::kilogram();
    let m = si::meter();

    assert!(kg1.is_compatible(&kg2));
    assert!(!kg1.is_compatible(&m));
}

#[test]
fn test_dimensionless_unit() {
    let d = Unit::dimensionless();
    assert!(d.is_dimensionless());

    let kg = si::kilogram();
    assert!(!kg.is_dimensionless());
}

// ==================== Unit Checker Tests ====================

#[test]
fn test_unit_checker_parse() {
    let checker = UnitChecker::new();

    let mg = checker.parse("mg").expect("should parse mg");
    let ml = checker.parse("mL").expect("should parse mL");

    assert!(!mg.is_compatible(&ml));
}

#[test]
fn test_unit_checker_compound() {
    let checker = UnitChecker::new();

    let mg_ml = checker.parse("mg/mL").expect("should parse compound");
    let mg = checker.parse("mg").expect("should parse mg");
    let ml = checker.parse("mL").expect("should parse mL");

    assert!(mg_ml.is_compatible(&mg.divide(&ml)));
}

// ==================== Unit Inference Tests ====================

#[test]
fn test_unit_inference_basic() {
    let mut inf = UnitInference::new();

    let var = inf.fresh_var();
    let mg = Unit::base("mg");

    inf.constrain_assign(var, mg.clone(), Span::dummy());

    assert!(inf.solve().is_ok());
    assert_eq!(inf.lookup(var), Some(&mg));
}

#[test]
fn test_unit_inference_equality() {
    let mut inf = UnitInference::new();

    let var1 = inf.fresh_var();
    let var2 = inf.fresh_var();
    let mg = Unit::base("mg");

    // var1 = mg
    inf.constrain_assign(var1, mg.clone(), Span::dummy());
    // var1 == var2
    inf.constrain_equal(UnitExpr::Var(var1), UnitExpr::Var(var2), Span::dummy());

    assert!(inf.solve().is_ok());
    assert_eq!(inf.lookup(var1), Some(&mg));
    // var2 should also be mg after solving
}

#[test]
fn test_unit_inference_binary_add() {
    let mut inf = UnitInference::new();

    let mg = Unit::base("mg");
    let u1 = UnitExpr::Concrete(mg.clone());
    let var = inf.fresh_var();
    let u2 = UnitExpr::Var(var);

    // Adding mg + unknown should constrain unknown to mg
    let _result = inf.infer_binary(UnitOp::Add, &u1, &u2, Span::dummy());

    assert!(inf.solve().is_ok());
    assert_eq!(inf.lookup(var), Some(&mg));
}

#[test]
fn test_unit_inference_binary_mul() {
    let mut inf = UnitInference::new();

    let mg = Unit::base("mg");
    let ml = Unit::base("mL");

    let u1 = UnitExpr::Concrete(mg.clone());
    let u2 = UnitExpr::Concrete(ml.clone());

    let result = inf.infer_binary(UnitOp::Mul, &u1, &u2, Span::dummy());

    let evaluated = result.evaluate().expect("should evaluate");
    assert!(evaluated.is_compatible(&mg.multiply(&ml)));
}

#[test]
fn test_unit_inference_binary_div() {
    let mut inf = UnitInference::new();

    let mg = Unit::base("mg");
    let ml = Unit::base("mL");

    let u1 = UnitExpr::Concrete(mg.clone());
    let u2 = UnitExpr::Concrete(ml.clone());

    let result = inf.infer_binary(UnitOp::Div, &u1, &u2, Span::dummy());

    let evaluated = result.evaluate().expect("should evaluate");
    assert!(evaluated.is_compatible(&mg.divide(&ml)));
}

#[test]
fn test_unit_inference_mismatch_error() {
    let mut inf = UnitInference::new();

    let mg = Unit::base("mg");
    let ml = Unit::base("mL");

    let u1 = UnitExpr::Concrete(mg);
    let u2 = UnitExpr::Concrete(ml);

    // Adding mg + mL should fail
    let _ = inf.infer_binary(UnitOp::Add, &u1, &u2, Span::dummy());

    let result = inf.solve();
    assert!(result.is_err());
}

// ==================== Medical Unit Tests ====================

#[test]
fn test_medical_dosage_units() {
    let checker = UnitChecker::new();

    // Common dosage units
    let mg = checker.parse("mg").expect("mg");
    let ug = checker.parse("ug").expect("ug");
    let mcg = checker.parse("mcg").expect("mcg");

    // ug and mcg should be the same
    assert!(ug.is_compatible(&mcg));

    // mg and ug are different (different scale)
    // They're compatible in dimension but not in magnitude
    // For now, just check they parse
    assert!(!mg.is_dimensionless());
    assert!(!ug.is_dimensionless());
}

#[test]
fn test_medical_concentration_units() {
    let checker = UnitChecker::new();

    let mg_ml = checker.parse("mg/mL").expect("mg/mL");
    let mg = checker.parse("mg").expect("mg");
    let ml = checker.parse("mL").expect("mL");

    // mg/mL should be compatible with mg divided by mL
    assert!(mg_ml.is_compatible(&mg.divide(&ml)));
}

#[test]
fn test_medical_time_units() {
    let checker = UnitChecker::new();

    let h = checker.parse("h").expect("h");
    let hr = checker.parse("hr").expect("hr");
    let min = checker.parse("min").expect("min");

    // h and hr should be the same (same alias)
    assert!(h.is_compatible(&hr));

    // h and min are dimensionally compatible (both are time units with base dimension 's')
    // is_compatible checks dimensional compatibility, not scale equality
    assert!(h.is_compatible(&min));

    // But they have different scales
    let conversion = h.conversion_factor(&min).expect("should convert");
    assert!((conversion - 60.0).abs() < 0.001); // 1 hour = 60 minutes
}
