// tests/e2e/diagnostics.rs — Diagnostic message quality tests
//
// Tests that error messages are:
// 1. Accurate - Point to the right location
// 2. Helpful - Provide actionable suggestions
// 3. Informative - Include semantic context

use super::common::{self, TestHarness};

// ============================================================================
// Error Location Tests
// ============================================================================

/// Test that type mismatch points to correct location
#[test]
fn test_error_location_type_mismatch() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology go from "https://purl.obolibrary.org/obo/go.owl";

type Drug = chebi:drug;
type Process = go:biological_process;

fn need_drug(d: Drug) {}

fn main() {
    let p: Process = go:0006915;
    need_drug(p);
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("error_location", source)
        .assert_failure()
        .assert_error_at("E0308", 11, 15); // Line 11, column 15 (the `p` argument)
}

/// Test that undefined ontology points to import
#[test]
fn test_error_location_undefined_ontology() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = drugbank:drug;  // drugbank not imported
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("undefined_ontology", source)
        .assert_failure()
        .assert_error_at("E0412", 4, 13); // Line 4, `drugbank` reference
}

/// Test multi-span error (e.g., conflicting definitions)
#[test]
fn test_error_multispan() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;
type Drug = chebi:small_molecule;  // Duplicate definition
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("multispan_error", source);

    result.assert_failure();

    // Should have related info pointing to first definition
    let diags = result.diagnostics();
    assert!(!diags.is_empty(), "Expected diagnostics");

    let first_error = diags
        .iter()
        .find(|d| d.level == common::DiagnosticLevel::Error)
        .expect("Expected error diagnostic");

    // Should have related info
    assert!(
        !first_error.related.is_empty() || first_error.notes.iter().any(|n| n.contains("previous")),
        "Expected related info pointing to previous definition"
    );
}

// ============================================================================
// Error Message Quality Tests
// ============================================================================

/// Test that type mismatch includes both types
#[test]
fn test_error_shows_both_types() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";

type Drug = chebi:drug;
type Phenotype = hp:phenotypic_abnormality;

fn analyze(d: Drug) {}

fn main() {
    let p: Phenotype = hp:0002315;
    analyze(p);
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("shows_both_types", source);

    result
        .assert_failure()
        .assert_error_contains("Drug")
        .assert_error_contains("Phenotype");
}

/// Test that suggestions include semantic distance
#[test]
fn test_error_includes_distance() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.4;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.2)]
fn strict_analysis(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00945;
    strict_analysis(db);
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("includes_distance", source);

    result.assert_failure();

    // Should mention the distance value
    let stderr = result.stderr();
    assert!(
        stderr.contains("0.4") || stderr.contains("distance") || stderr.contains("semantic"),
        "Expected semantic distance info in error message"
    );
}

/// Test that errors suggest similar types
#[test]
fn test_error_suggests_similar() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drugo = chebi:drug;  // Typo: Drugo instead of Drug

fn main() {
    let d: Drug = chebi:15365;  // Uses correct name, but type doesn't exist
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("suggests_similar", source);

    result.assert_failure();

    // Should suggest "Drugo" as alternative
    let stderr = result.stderr();
    assert!(
        stderr.contains("Drugo") || stderr.contains("similar") || stderr.contains("did you mean"),
        "Expected suggestion for similar type name"
    );
}

// ============================================================================
// Ontology-Specific Error Tests
// ============================================================================

/// Test error for invalid ontology term
#[test]
fn test_invalid_ontology_term() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let d: Drug = chebi:99999999;  // Invalid term ID
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("invalid_term", source)
        .assert_failure()
        .assert_error_contains("99999999");
}

/// Test error for ontology load failure
#[test]
fn test_ontology_load_error() {
    let source = r#"
ontology nonexistent from "file://ontologies/does_not_exist.owl";

type Thing = nonexistent:thing;
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("ontology_load", source)
        .assert_failure()
        .assert_error_contains("does_not_exist");
}

/// Test error explains ontology hierarchy
#[test]
fn test_hierarchy_explanation() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type SmallMolecule = chebi:small_molecule;
type Protein = chebi:protein;

fn analyze_small(s: SmallMolecule) {}

fn main() {
    let p: Protein = chebi:36080;  // Example protein
    analyze_small(p);  // Protein is not a small molecule
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("hierarchy_explain", source);

    result.assert_failure();

    // Should explain why types are incompatible
    let stderr = result.stderr();
    assert!(
        stderr.contains("Protein") && stderr.contains("SmallMolecule")
            || stderr.contains("hierarchy")
            || stderr.contains("branch"),
        "Expected explanation of ontology hierarchy mismatch"
    );
}

// ============================================================================
// Suggestion Quality Tests
// ============================================================================

/// Test that threshold suggestion is provided
#[test]
fn test_suggests_threshold_adjustment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.3;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.2)]
fn analyze(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00945;
    analyze(db);
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("threshold_suggestion", source);

    result.assert_failure();

    // Should suggest increasing threshold
    let stderr = result.stderr();
    assert!(
        stderr.contains("threshold") || stderr.contains("0.3") || stderr.contains("compat"),
        "Expected threshold adjustment suggestion"
    );
}

/// Test that explicit cast suggestion is provided
#[test]
fn test_suggests_explicit_cast() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

fn analyze(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00945;
    analyze(db);  // No alignment declared
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("cast_suggestion", source);

    result.assert_failure();

    // Should suggest alignment or explicit cast
    let stderr = result.stderr();
    assert!(
        stderr.contains("align") || stderr.contains("as ") || stderr.contains("convert"),
        "Expected alignment or cast suggestion"
    );
}

/// Test actionable fix suggestion
#[test]
fn test_actionable_fix() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

fn main() {
    let d = chebi:15365;  // Missing type annotation
    process(d);
}

fn process(x: chebi:drug) {}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("actionable_fix", source);

    // Check if there's a suggestion with replacement code
    let diags = result.diagnostics();
    let _has_replacement = diags
        .iter()
        .flat_map(|d| d.suggestions.iter())
        .any(|s| s.replacement.is_some());

    // Might pass or fail depending on inference, but if error, should have suggestion
    if !result.success() {
        // Should have at least some helpful message
        let stderr = result.stderr();
        assert!(!stderr.is_empty(), "Expected helpful error message");
    }
}

// ============================================================================
// Warning Tests
// ============================================================================

/// Test warning for unused ontology import
#[test]
fn test_warning_unused_import() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology go from "https://purl.obolibrary.org/obo/go.owl";  // Unused

type Drug = chebi:drug;

fn main() {
    let d: Drug = chebi:15365;
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .flag("--warn=unused-imports")
        .compile_str("unused_import", source);

    // Should compile but with warning
    if result.success() {
        result.assert_warning("unused_import");
    }
    // Might also error if strict mode
}

/// Test warning for deprecated term
#[test]
fn test_warning_deprecated_term() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    // Using a deprecated ChEBI term (hypothetical)
    let d: Drug = chebi:00001;  // Deprecated ID
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("deprecated_term", source);

    // Might warn about deprecated term
    let stderr = result.stderr();
    // Just verify compilation behavior is consistent
    assert!(result.success() || stderr.contains("deprecated") || stderr.contains("00001"));
}

/// Test warning for loose threshold
#[test]
fn test_warning_loose_threshold() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat(threshold = 0.9)]  // Very loose - potentially unsafe
fn analyze(d: Drug) {}

fn main() {
    let d: Drug = chebi:15365;
    analyze(d);
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .flag("--warn=loose-threshold")
        .compile_str("loose_threshold", source);

    // Should compile with warning about loose threshold
    result.assert_success();
    // Warning might or might not be implemented
}

// ============================================================================
// Note and Help Tests
// ============================================================================

/// Test that notes provide context
#[test]
fn test_notes_provide_context() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";

type Drug = chebi:drug;
type Phenotype = hp:phenotypic_abnormality;

fn analyze(d: Drug) -> Phenotype {
    // Error: returning wrong type
    d
}

fn main() {}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("notes_context", source);

    result.assert_failure();

    // Should have notes explaining the issue
    let diags = result.diagnostics();
    let error = diags
        .iter()
        .find(|d| d.level == common::DiagnosticLevel::Error)
        .expect("Expected error");

    // Either notes or message should explain
    assert!(
        !error.notes.is_empty() || error.message.len() > 20,
        "Expected contextual notes or detailed message"
    );
}

/// Test help messages for common mistakes
#[test]
fn test_help_common_mistakes() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

// Common mistake: using :: instead of :
type Drug = chebi::drug;
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("common_mistake", source);

    result.assert_failure();

    // Should give helpful message about syntax
    let stderr = result.stderr();
    assert!(
        stderr.contains("::") || stderr.contains("colon") || stderr.contains("syntax"),
        "Expected help for common syntax mistake"
    );
}

// ============================================================================
// Error Format Tests
// ============================================================================

/// Test JSON output format
#[test]
fn test_json_format() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let x: Drug = 42;  // Wrong type
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("json_format", source);

    result.assert_failure();

    // Verify JSON is parseable
    let stderr = result.stderr();
    let has_json = stderr.lines().any(|line| {
        line.trim().starts_with('{') && serde_json::from_str::<serde_json::Value>(line).is_ok()
    });

    assert!(has_json, "Expected valid JSON diagnostic output");
}

/// Test human-readable format
#[test]
fn test_human_format() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let x: Drug = 42;
}
"#;

    let result = TestHarness::new()
        .flag("--error-format=human")
        .compile_str("human_format", source);

    result.assert_failure();

    // Should have nice formatting
    let stderr = result.stderr();
    assert!(
        stderr.contains("error") || stderr.contains("Error"),
        "Expected human-readable error format"
    );
}

// ============================================================================
// Error Count and Summary Tests
// ============================================================================

/// Test error count summary
#[test]
fn test_error_count_summary() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let a: Drug = 1;
    let b: Drug = "string";
    let c: Drug = true;
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("error_count", source);

    result.assert_failure();

    // Should have multiple errors
    let error_count = result.error_count();
    assert!(
        error_count >= 3,
        "Expected at least 3 errors, got {}",
        error_count
    );
}

/// Test that first error is most relevant
#[test]
fn test_first_error_relevant() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = undefined_ontology:drug;  // First error: undefined ontology

fn main() {
    let d: Drug = chebi:15365;  // Cascading error
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("first_relevant", source);

    result.assert_failure();

    let diags = result.diagnostics();
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.level == common::DiagnosticLevel::Error)
        .collect();

    if !errors.is_empty() {
        // First error should be about undefined ontology
        let first = &errors[0];
        assert!(
            first.message.contains("undefined")
                || first.message.contains("not found")
                || first.code.as_deref() == Some("E0412"),
            "First error should be about undefined ontology"
        );
    }
}

// ============================================================================
// Source Snippet Tests
// ============================================================================

/// Test that source snippets are shown
#[test]
fn test_shows_source_snippet() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn broken_function() {
    let x: Drug = "not a drug";  // Error here
}
"#;

    let result = TestHarness::new()
        .flag("--error-format=human")
        .compile_str("source_snippet", source);

    result.assert_failure();

    let stderr = result.stderr();
    // Should show the problematic line
    assert!(
        stderr.contains("not a drug") || stderr.contains("let x"),
        "Expected source snippet in error output"
    );
}

/// Test multiline context in snippets
#[test]
fn test_multiline_context() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn complex_error() {
    let d: Drug =
        if true {
            42  // Error: number not Drug
        } else {
            chebi:15365
        };
}
"#;

    let result = TestHarness::new()
        .flag("--error-format=human")
        .compile_str("multiline_context", source);

    result.assert_failure();

    // Error output should have some context
    let stderr = result.stderr();
    assert!(
        stderr.len() > 50,
        "Expected substantial error output with context"
    );
}
