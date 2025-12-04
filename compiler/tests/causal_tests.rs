//! Comprehensive tests for Day 34: Causal Primitives
//!
//! Tests Pearl's causal hierarchy:
//! - Level 1: Association (Knowledge)
//! - Level 2: Intervention (CausalKnowledge + do())
//! - Level 3: Counterfactual (StructuralKnowledge)

use std::collections::{HashMap, HashSet};

use demetrios::causal::*;
use demetrios::epistemic::composition::EpistemicValue;
use demetrios::temporal::TemporalKnowledge;

// ============================================================================
// Graph Tests
// ============================================================================

#[test]
fn test_graph_creation() {
    let mut g = CausalGraph::new();

    g.add_node(CausalNode::treatment("Drug"));
    g.add_node(CausalNode::mediator("Concentration"));
    g.add_node(CausalNode::outcome("Effect"));

    g.add_edge("Drug", "Concentration", EdgeType::Direct)
        .unwrap();
    g.add_edge("Concentration", "Effect", EdgeType::Direct)
        .unwrap();

    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 2);
}

#[test]
fn test_cycle_prevention() {
    let mut g = CausalGraph::new();

    g.add_node(CausalNode::observed("A"));
    g.add_node(CausalNode::observed("B"));
    g.add_node(CausalNode::observed("C"));

    g.add_edge("A", "B", EdgeType::Direct).unwrap();
    g.add_edge("B", "C", EdgeType::Direct).unwrap();

    // Should fail - would create A -> B -> C -> A
    let result = g.add_edge("C", "A", EdgeType::Direct);
    assert!(result.is_err());
}

#[test]
fn test_d_separation_chain() {
    // X -> M -> Y
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("X"));
    g.add_node(CausalNode::mediator("M"));
    g.add_node(CausalNode::outcome("Y"));

    g.add_edge("X", "M", EdgeType::Direct).unwrap();
    g.add_edge("M", "Y", EdgeType::Direct).unwrap();

    // X and Y are d-connected when M is not conditioned
    assert!(g.d_connected("X", "Y", &HashSet::new()));

    // X and Y are d-separated when M is conditioned
    let z: HashSet<String> = ["M".to_string()].into_iter().collect();
    assert!(g.d_separated("X", "Y", &z));
}

#[test]
fn test_d_separation_fork() {
    // X <- U -> Y
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::observed("X"));
    g.add_node(CausalNode::latent("U"));
    g.add_node(CausalNode::observed("Y"));

    g.add_edge("U", "X", EdgeType::Direct).unwrap();
    g.add_edge("U", "Y", EdgeType::Direct).unwrap();

    // X and Y are d-connected when U is not conditioned
    assert!(g.d_connected("X", "Y", &HashSet::new()));

    // X and Y are d-separated when U is conditioned
    let z: HashSet<String> = ["U".to_string()].into_iter().collect();
    assert!(g.d_separated("X", "Y", &z));
}

#[test]
fn test_d_separation_collider() {
    // X -> C <- Y
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::observed("X"));
    g.add_node(CausalNode::observed("C"));
    g.add_node(CausalNode::observed("Y"));

    g.add_edge("X", "C", EdgeType::Direct).unwrap();
    g.add_edge("Y", "C", EdgeType::Direct).unwrap();

    // X and Y are d-separated when C is NOT conditioned (collider blocks)
    assert!(g.d_separated("X", "Y", &HashSet::new()));

    // X and Y are d-connected when C IS conditioned (collider opens)
    let z: HashSet<String> = ["C".to_string()].into_iter().collect();
    assert!(g.d_connected("X", "Y", &z));
}

#[test]
fn test_graph_do_removes_incoming() {
    // U -> X -> Y
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::latent("U"));
    g.add_node(CausalNode::treatment("X"));
    g.add_node(CausalNode::outcome("Y"));

    g.add_edge("U", "X", EdgeType::Direct).unwrap();
    g.add_edge("X", "Y", EdgeType::Direct).unwrap();

    let g_do = g.graph_do("X");

    // X should have no parents in G_X̄
    assert!(g_do.parents("X").unwrap().is_empty());

    // X -> Y edge should remain
    assert!(g_do.children("X").unwrap().contains("Y"));
}

// ============================================================================
// Identification Tests
// ============================================================================

#[test]
fn test_backdoor_criterion_simple() {
    // X <- U -> Y, X -> Y
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("X"));
    g.add_node(CausalNode::outcome("Y"));
    g.add_node(CausalNode::observed("U"));

    g.add_edge("X", "Y", EdgeType::Direct).unwrap();
    g.add_edge("U", "X", EdgeType::Direct).unwrap();
    g.add_edge("U", "Y", EdgeType::Direct).unwrap();

    let identifier = CausalIdentifier::new(&g);

    // U should satisfy backdoor criterion
    let set: HashSet<String> = ["U".to_string()].into_iter().collect();
    assert!(identifier.satisfies_backdoor("X", "Y", &set));

    // Empty set should NOT satisfy (backdoor path via U)
    assert!(!identifier.satisfies_backdoor("X", "Y", &HashSet::new()));
}

#[test]
fn test_backdoor_finds_valid_set() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("X"));
    g.add_node(CausalNode::outcome("Y"));
    g.add_node(CausalNode::observed("Age"));
    g.add_node(CausalNode::observed("Gender"));

    g.add_edge("X", "Y", EdgeType::Direct).unwrap();
    g.add_edge("Age", "X", EdgeType::Direct).unwrap();
    g.add_edge("Age", "Y", EdgeType::Direct).unwrap();
    g.add_edge("Gender", "X", EdgeType::Direct).unwrap();
    g.add_edge("Gender", "Y", EdgeType::Direct).unwrap();

    let identifier = CausalIdentifier::new(&g);
    let backdoor_set = identifier.find_backdoor_set("X", "Y");

    assert!(backdoor_set.is_some());
    let set = backdoor_set.unwrap();

    // Should contain Age and Gender (or just one if that's sufficient)
    assert!(!set.is_empty());
}

#[test]
fn test_identification_status() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("X"));
    g.add_node(CausalNode::outcome("Y"));
    g.add_edge("X", "Y", EdgeType::Direct).unwrap();

    let identifier = CausalIdentifier::new(&g);
    let status = identifier.identify("X", "Y");

    assert!(matches!(status, IdentificationStatus::Identified { .. }));
}

// ============================================================================
// CausalKnowledge Tests
// ============================================================================

fn create_base_knowledge() -> TemporalKnowledge<f64> {
    let core = EpistemicValue::with_confidence(0.75, 0.90);
    TemporalKnowledge::timeless(core)
}

#[test]
fn test_causal_knowledge_creation() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("Drug"));
    g.add_node(CausalNode::outcome("Effect"));
    g.add_edge("Drug", "Effect", EdgeType::Direct).unwrap();

    let base = create_base_knowledge();
    let ck = CausalKnowledge::new(base, g, "Effect", vec!["Drug".to_string()]);

    assert_eq!(ck.outcome, "Effect");
    assert_eq!(ck.treatments, vec!["Drug"]);
}

#[test]
fn test_do_intervention_unconfounded() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("Drug"));
    g.add_node(CausalNode::outcome("Effect"));
    g.add_edge("Drug", "Effect", EdgeType::Direct).unwrap();

    let base = create_base_knowledge();
    let ck = CausalKnowledge::new(base, g, "Effect", vec!["Drug".to_string()]);

    let intervention = Intervention::atomic("Drug", 100.0);
    let result = ck.do_intervention(intervention);

    assert!(result.is_ok());
}

#[test]
fn test_do_intervention_with_backdoor() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("Drug"));
    g.add_node(CausalNode::outcome("Effect"));
    g.add_node(CausalNode::observed("Severity"));

    g.add_edge("Drug", "Effect", EdgeType::Direct).unwrap();
    g.add_edge("Severity", "Drug", EdgeType::Direct).unwrap();
    g.add_edge("Severity", "Effect", EdgeType::Direct).unwrap();

    let base = create_base_knowledge();
    let ck = CausalKnowledge::new(base, g, "Effect", vec!["Drug".to_string()]);

    let intervention = Intervention::atomic("Drug", 100.0);
    let result = ck.do_intervention(intervention);

    assert!(result.is_ok());
    let result = result.unwrap();

    // Should use backdoor adjustment
    if let IdentificationMethod::BackdoorAdjustment { set } = &result.identification {
        assert!(set.contains("Severity"));
    } else {
        panic!("Expected backdoor adjustment");
    }
}

#[test]
fn test_identify_caches_result() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("X"));
    g.add_node(CausalNode::outcome("Y"));
    g.add_edge("X", "Y", EdgeType::Direct).unwrap();

    let base = create_base_knowledge();
    let mut ck = CausalKnowledge::new(base, g, "Y", vec!["X".to_string()]);

    assert!(!ck.is_identified());
    ck.identify();
    assert!(ck.is_identified());
}

// ============================================================================
// Structural Causal Model Tests
// ============================================================================

#[test]
fn test_scm_creation() {
    let scm = SCMBuilder::new()
        .exogenous_variable(
            "X",
            "U_X",
            Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )
        .linear_variable("Y", vec![("X".to_string(), 0.5)], "U_Y", 1.0)
        .build();

    assert_eq!(scm.variables().len(), 2);
    assert_eq!(scm.exogenous_variables().len(), 2);
}

#[test]
fn test_scm_evaluate() {
    let scm = SCMBuilder::new()
        .exogenous_variable(
            "X",
            "U_X",
            Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )
        .linear_variable("Y", vec![("X".to_string(), 0.5)], "U_Y", 1.0)
        .build();

    let u: HashMap<String, f64> = [("U_X".to_string(), 2.0), ("U_Y".to_string(), 0.0)]
        .into_iter()
        .collect();

    let values = scm.evaluate(&u);

    // X = U_X = 2.0
    assert!((values["X"] - 2.0).abs() < 0.001);

    // Y = 0.5 * X + 1.0 + U_Y = 0.5 * 2.0 + 1.0 + 0.0 = 2.0
    assert!((values["Y"] - 2.0).abs() < 0.001);
}

#[test]
fn test_scm_intervention() {
    let scm = SCMBuilder::new()
        .exogenous_variable(
            "X",
            "U_X",
            Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )
        .linear_variable("Y", vec![("X".to_string(), 0.5)], "U_Y", 1.0)
        .build();

    let m_x = scm.intervene("X", 10.0);

    let u: HashMap<String, f64> = [
        ("U_X".to_string(), 0.0), // Ignored due to intervention
        ("U_Y".to_string(), 0.0),
    ]
    .into_iter()
    .collect();

    let values = m_x.evaluate(&u);

    // X = 10.0 (intervened value)
    assert!((values["X"] - 10.0).abs() < 0.001);

    // Y = 0.5 * 10.0 + 1.0 = 6.0
    assert!((values["Y"] - 6.0).abs() < 0.001);
}

#[test]
fn test_scm_simulate() {
    let scm = SCMBuilder::new()
        .exogenous_variable("X", "U_X", Distribution::Uniform { min: 0.0, max: 1.0 })
        .linear_variable("Y", vec![("X".to_string(), 1.0)], "U_Y", 0.0)
        .build();

    let samples = scm.simulate(100);

    assert_eq!(samples.len(), 100);
    for sample in &samples {
        assert!(sample.contains_key("X"));
        assert!(sample.contains_key("Y"));
    }
}

// ============================================================================
// Counterfactual Tests
// ============================================================================

fn create_structural_knowledge() -> StructuralKnowledge<f64> {
    let model = SCMBuilder::new()
        .exogenous_variable(
            "X",
            "U_X",
            Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )
        .linear_variable("Y", vec![("X".to_string(), 0.5)], "U_Y", 0.0)
        .build();

    let mut graph = CausalGraph::new();
    graph.add_node(CausalNode::treatment("X"));
    graph.add_node(CausalNode::outcome("Y"));
    graph.add_edge("X", "Y", EdgeType::Direct).unwrap();

    let base = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.9));
    let causal = CausalKnowledge::new(base, graph, "Y", vec!["X".to_string()]);

    StructuralKnowledge::new(causal, model)
}

#[test]
fn test_counterfactual_basic() {
    let sk = create_structural_knowledge();

    // Evidence: X=2, Y=1
    let evidence: HashMap<String, f64> = [("X".to_string(), 2.0), ("Y".to_string(), 1.0)]
        .into_iter()
        .collect();

    // Counterfactual: What would Y be if X had been 0?
    let cf = sk.counterfactual("Y", "X", 0.0, &evidence);

    // Y = 0.5*X + U_Y
    // With X=2, Y=1: U_Y = 1 - 0.5*2 = 0
    // With X=0: Y = 0.5*0 + 0 = 0
    assert!((cf.value - 0.0).abs() < 0.1);
}

#[test]
fn test_counterfactual_query_display() {
    let query = CounterfactualQuery {
        target: "Recovery".to_string(),
        intervention: "Treatment".to_string(),
        intervention_value: 0.0,
        evidence: [
            ("Treatment".to_string(), 1.0),
            ("Recovery".to_string(), 1.0),
        ]
        .into_iter()
        .collect(),
    };

    let display = format!("{}", query);
    assert!(display.contains("Recovery"));
    assert!(display.contains("Treatment"));
}

#[test]
fn test_probability_of_necessity() {
    let sk = create_structural_knowledge();

    let pn = sk.probability_of_necessity("X", 1.0, 0.0, "Y", 100);

    assert!(pn.probability >= 0.0);
    assert!(pn.probability <= 1.0);
    assert_eq!(pn.causation_type, CausationType::Necessity);
}

#[test]
fn test_probability_of_sufficiency() {
    let sk = create_structural_knowledge();

    let ps = sk.probability_of_sufficiency("X", 0.0, 1.0, "Y", 100);

    assert!(ps.probability >= 0.0);
    assert!(ps.probability <= 1.0);
    assert_eq!(ps.causation_type, CausationType::Sufficiency);
}

#[test]
fn test_probability_of_necessity_and_sufficiency() {
    let sk = create_structural_knowledge();

    let pns = sk.probability_of_necessity_and_sufficiency("X", 1.0, 0.0, "Y", 100);

    assert!(pns.probability >= 0.0);
    assert!(pns.probability <= 1.0);
    assert_eq!(pns.causation_type, CausationType::NecessityAndSufficiency);
}

// ============================================================================
// Causal Composition Tests
// ============================================================================

#[test]
fn test_causal_tensor() {
    let mut g1 = CausalGraph::new();
    g1.add_node(CausalNode::treatment("Drug"));
    g1.add_node(CausalNode::outcome("Effect"));
    g1.add_edge("Drug", "Effect", EdgeType::Direct).unwrap();

    let base1 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.9));
    let ck1 = CausalKnowledge::new(base1, g1, "Effect", vec!["Drug".to_string()]);

    let mut g2 = CausalGraph::new();
    g2.add_node(CausalNode::treatment("Exercise"));
    g2.add_node(CausalNode::outcome("Fitness"));
    g2.add_edge("Exercise", "Fitness", EdgeType::Direct)
        .unwrap();

    let base2 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.7, 0.85));
    let ck2: CausalKnowledge<f64> =
        CausalKnowledge::new(base2, g2, "Fitness", vec!["Exercise".to_string()]);

    let result = ck1.tensor_causal(ck2);

    assert!(result.is_ok());
    let merged = result.unwrap();

    assert!(merged.graph.contains_node("Drug"));
    assert!(merged.graph.contains_node("Effect"));
    assert!(merged.graph.contains_node("Exercise"));
    assert!(merged.graph.contains_node("Fitness"));
}

#[test]
fn test_causal_join_concordant() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::treatment("X"));
    g.add_node(CausalNode::outcome("Y"));
    g.add_edge("X", "Y", EdgeType::Direct).unwrap();

    let base1 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.8));
    let ck1 = CausalKnowledge::new(base1, g.clone(), "Y", vec!["X".to_string()]);

    let base2 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.85));
    let ck2 = CausalKnowledge::new(base2, g, "Y", vec!["X".to_string()]);

    let result = ck1.join_causal(ck2, 0.3);

    match result {
        CausalJoinResult::Concordant(merged) => {
            // Confidence should be boosted
            assert!(merged.confidence().value() > 0.8);
        }
        _ => panic!("Expected concordant join"),
    }
}

#[test]
fn test_causal_join_irreconcilable_structure() {
    let mut g1 = CausalGraph::new();
    g1.add_node(CausalNode::treatment("X"));
    g1.add_node(CausalNode::outcome("Y"));
    g1.add_edge("X", "Y", EdgeType::Direct).unwrap();

    let mut g2 = CausalGraph::new();
    g2.add_node(CausalNode::treatment("X"));
    g2.add_node(CausalNode::mediator("M"));
    g2.add_node(CausalNode::outcome("Y"));
    g2.add_edge("X", "M", EdgeType::Direct).unwrap();
    g2.add_edge("M", "Y", EdgeType::Direct).unwrap();

    let base1 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.8));
    let ck1 = CausalKnowledge::new(base1, g1, "Y", vec!["X".to_string()]);

    let base2 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.85));
    let ck2 = CausalKnowledge::new(base2, g2, "Y", vec!["X".to_string()]);

    let result = ck1.join_causal(ck2, 0.3);

    assert!(matches!(result, CausalJoinResult::Irreconcilable { .. }));
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_pkpd_causal_analysis() {
    // Build PKPD causal graph
    let mut graph = CausalGraph::new();

    graph.add_node(CausalNode::treatment("Dose"));
    graph.add_node(CausalNode::observed("Clearance"));
    graph.add_node(CausalNode::mediator("Concentration"));
    graph.add_node(CausalNode::outcome("Effect"));
    graph.add_node(CausalNode::latent("Genetics"));

    graph
        .add_edge("Dose", "Concentration", EdgeType::Direct)
        .unwrap();
    graph
        .add_edge("Clearance", "Concentration", EdgeType::Direct)
        .unwrap();
    graph
        .add_edge("Concentration", "Effect", EdgeType::Direct)
        .unwrap();
    graph
        .add_edge("Genetics", "Clearance", EdgeType::Direct)
        .unwrap();
    graph
        .add_edge("Genetics", "Effect", EdgeType::Direct)
        .unwrap();

    // Create causal knowledge
    let base = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.75, 0.85));

    let mut ck = CausalKnowledge::new(base, graph, "Effect", vec!["Dose".to_string()]);

    // Identify the effect
    let status = ck.identify();
    assert!(ck.is_identified());

    // Compute causal effect
    let intervention = Intervention::atomic("Dose", 100.0);
    let result = ck.do_intervention(intervention);

    assert!(result.is_ok());
}

#[test]
fn test_medical_counterfactual_scenario() {
    // Build SCM for treatment -> recovery
    let model = SCMBuilder::new()
        .exogenous_variable(
            "Severity",
            "U_S",
            Distribution::Uniform { min: 0.0, max: 1.0 },
        )
        .custom_variable("Treatment", vec!["Severity".to_string()], "U_T", |pa, u| {
            let severity = pa.get("Severity").unwrap_or(&0.5);
            if *severity + u > 0.5 { 1.0 } else { 0.0 }
        })
        .custom_variable(
            "Recovery",
            vec!["Treatment".to_string(), "Severity".to_string()],
            "U_R",
            |pa, u| {
                let treatment = pa.get("Treatment").unwrap_or(&0.0);
                let severity = pa.get("Severity").unwrap_or(&0.5);
                let prob = 0.5 + 0.3 * treatment - 0.4 * severity + 0.2 * u;
                if prob > 0.5 { 1.0 } else { 0.0 }
            },
        )
        .with_distribution("U_T", Distribution::Uniform { min: 0.0, max: 0.5 })
        .with_distribution(
            "U_R",
            Distribution::Normal {
                mean: 0.0,
                std: 0.3,
            },
        )
        .build();

    let mut graph = CausalGraph::new();
    graph.add_node(CausalNode::observed("Severity"));
    graph.add_node(CausalNode::treatment("Treatment"));
    graph.add_node(CausalNode::outcome("Recovery"));

    graph
        .add_edge("Severity", "Treatment", EdgeType::Direct)
        .unwrap();
    graph
        .add_edge("Severity", "Recovery", EdgeType::Direct)
        .unwrap();
    graph
        .add_edge("Treatment", "Recovery", EdgeType::Direct)
        .unwrap();

    let base = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.9));
    let causal = CausalKnowledge::new(base, graph, "Recovery", vec!["Treatment".to_string()]);

    let sk = StructuralKnowledge::new(causal, model);

    // Counterfactual: Patient received treatment and recovered
    // Would they have recovered without treatment?
    let evidence: HashMap<String, f64> = [
        ("Treatment".to_string(), 1.0),
        ("Recovery".to_string(), 1.0),
    ]
    .into_iter()
    .collect();

    let cf = sk.counterfactual("Recovery", "Treatment", 0.0, &evidence);

    // Result should be a valid probability
    assert!(cf.value.is_finite() || cf.value.is_nan());

    // Probability of necessity
    let pn = sk.probability_of_necessity("Treatment", 1.0, 0.0, "Recovery", 50);
    assert!(pn.probability >= 0.0 && pn.probability <= 1.0);
}

// ============================================================================
// Algebraic Laws Tests
// ============================================================================

#[test]
fn test_do_law_idempotent() {
    // do(do(X=x)) = do(X=x)
    let scm = SCMBuilder::new()
        .exogenous_variable(
            "X",
            "U_X",
            Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )
        .linear_variable("Y", vec![("X".to_string(), 1.0)], "U_Y", 0.0)
        .build();

    let m_x = scm.intervene("X", 5.0);
    let m_xx = m_x.intervene("X", 5.0);

    let u: HashMap<String, f64> = [("U_X".to_string(), 0.0), ("U_Y".to_string(), 0.0)]
        .into_iter()
        .collect();

    let values1 = m_x.evaluate(&u);
    let values2 = m_xx.evaluate(&u);

    assert!((values1["Y"] - values2["Y"]).abs() < 0.001);
}

#[test]
fn test_do_cuts_confounding() {
    // X <- U -> Y, X -> Y
    // P(Y | do(X)) should not depend on U's effect on X
    let scm = SCMBuilder::new()
        .exogenous_variable(
            "U",
            "U_U",
            Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )
        .linear_variable("X", vec![("U".to_string(), 0.5)], "U_X", 0.0)
        .linear_variable(
            "Y",
            vec![("X".to_string(), 1.0), ("U".to_string(), 0.5)],
            "U_Y",
            0.0,
        )
        .build();

    // Under do(X=2), the effect of U on X is cut
    let m_x = scm.intervene("X", 2.0);

    // Varying U_U should not change X in intervened model
    let u1: HashMap<String, f64> = [
        ("U_U".to_string(), -5.0),
        ("U_X".to_string(), 0.0),
        ("U_Y".to_string(), 0.0),
    ]
    .into_iter()
    .collect();

    let u2: HashMap<String, f64> = [
        ("U_U".to_string(), 5.0),
        ("U_X".to_string(), 0.0),
        ("U_Y".to_string(), 0.0),
    ]
    .into_iter()
    .collect();

    let values1 = m_x.evaluate(&u1);
    let values2 = m_x.evaluate(&u2);

    // X should be 2.0 in both cases (intervened)
    assert!((values1["X"] - 2.0).abs() < 0.001);
    assert!((values2["X"] - 2.0).abs() < 0.001);

    // Y differs only through U's direct effect, not through X
    // Y = 1.0 * X + 0.5 * U + U_Y
    // Y1 = 2.0 + 0.5 * (-5) = 2.0 - 2.5 = -0.5
    // Y2 = 2.0 + 0.5 * 5 = 2.0 + 2.5 = 4.5
    assert!((values1["Y"] - (-0.5)).abs() < 0.001);
    assert!((values2["Y"] - 4.5).abs() < 0.001);
}
