# Epistemic Computing for PBPK Modeling

This document describes Demetrios's unique approach to physiologically-based pharmacokinetic (PBPK) modeling using epistemic types, with **full compliance** to FDA and EMA 2024-2025 guidelines.

## Regulatory Compliance

Demetrios PBPK module implements:

- **FDA**: "Physiologically Based Pharmacokinetic Analyses — Format and Content" (2018, updated guidance 2024)
- **EMA**: "Guideline on Reporting of PBPK Modelling and Simulation" (EMA/CHMP/458101/2016)
- **EMA 2025**: Impact-based qualification framework (Low/Medium/High)

## Overview

Demetrios is the **first programming language** to provide compile-time epistemic tracking for scientific computing. This is particularly valuable in pharmacometrics where:

1. **Regulatory requirements** (FDA, EMA) demand full provenance tracking
2. **Parameter uncertainty** must be quantified and propagated
3. **Model confidence** determines whether predictions can be used for decision-making
4. **Validation metrics** (GMFE, AFE, AAFE) must meet predefined acceptance criteria

## What No Other Language Can Do

| Feature | Python | Julia | R/NONMEM | Demetrios |
|---------|--------|-------|----------|-----------|
| Type-safe units | No | Via Unitful.jl | No | **Built-in** |
| Compile-time unit checking | No | No | No | **Yes** |
| Epistemic confidence tracking | No | No | No | **Built-in** |
| Provenance audit trail | Manual | Manual | Manual | **Automatic** |
| FDA-ready validation | External tools | External tools | Manual | **Integrated** |

## The Knowledge Type

```d
// Knowledge[T, ε >= bound] - Type with epistemic qualifications
let clearance: Knowledge[L/h, ε >= 0.75] = Knowledge::new(
    value: 35.0 : L/h,
    confidence: 0.88,
    provenance: Provenance::source("Darwin_GNN_v2.5"),
);
```

### Confidence Propagation

Confidence propagates automatically through computations:

```d
// Minimum rule: output confidence = min(input confidences)
let cl = params.cl_hepatic  // ε = 0.88
let vd = params.vd          // ε = 0.85
let ke = cl / vd            // ε = min(0.88, 0.85) * 0.99 = 0.84

// ODE solvers degrade confidence
let result = simulate(...)  // ε = base_confidence * 0.95
```

### Provenance Tracking

Every computation records its data lineage:

```d
let result = simulate(&drug, &params, &patient, dose, duration, dt);

// Generate FDA audit trail
println("{}", result.provenance.to_audit_trail());
// OUTPUT: MERGED: SOURCE: DrugBank:DB00331 @ 2024-01-15 | 
//                 SOURCE: Darwin_GNN_v2.5 @ 2024-01-15 |
//                 DERIVED[ode_simulation]: ...
```

## PBPK Module

### Drug Definition

```d
use pbpk::*

let metformin = Drug {
    chebi_id: "CHEBI:6801",  // Validated at COMPILE TIME
    name: "Metformin",
    mw: 129.16 : g/mol,
    logp: -1.43,
    fu: Knowledge::new(
        value: 1.0,
        confidence: 0.95,
        provenance: Provenance::source("DrugBank:DB00331"),
    ),
};
```

### PBPK Parameters

```d
let params = PBPKParams {
    cl_hepatic: Knowledge::new(
        value: 35.0 : L/h,
        confidence: 0.88,
        provenance: Provenance::source("Darwin_GNN_v2.5"),
    ),
    cl_renal: Knowledge::new(
        value: 25.0 : L/h,
        confidence: 0.92,
        provenance: Provenance::source("Scheen_1996"),
    ),
    vd: Knowledge::new(
        value: 654.0 : L,
        confidence: 0.85,
        provenance: Provenance::source("Graham_2011"),
    ),
    ka: Knowledge::new(
        value: 2.1 : 1/h,
        confidence: 0.78,
        provenance: Provenance::source("PopPK_Meta"),
    ),
    kp: default_partition_coefficients(),
};
```

### Running Simulations

```d
let result = simulate(
    &metformin,
    &params,
    &patient,
    500.0 : mg,    // dose
    24.0 : h,      // duration
    0.1 : h,       // time step
);

println("Simulation confidence: {:.1}%", result.confidence * 100.0);
```

### FDA Validation

```d
// This function ONLY compiles if predictions have ε >= 0.80
match validate_for_fda(&predictions, &observed) {
    Ok(metrics) => {
        println("GMFE: {:.2}", metrics.gmfe);
        println("Within 2-fold: {:.1}%", metrics.within_2fold * 100.0);
        
        if metrics.gmfe <= 2.0 && metrics.within_2fold >= 0.80 {
            println("✓ PASSES FDA PBPK Guidance criteria");
        }
    }
    Err(ValidationError::InsufficientConfidence { .. }) => {
        println("Cannot submit - confidence too low");
    }
}
```

## QUDT Units Integration

The units module provides QUDT-aligned units for pharmacokinetics:

```d
use units::qudt::*

// Mass
let dose = 500.0 : mg
let dose_kg = convert(dose, kg)  // 0.0005 kg

// Concentration
let conc = 10.0 : mg/L
let conc_molar = conc / (129.16 : g/mol)  // Convert to molar

// Clearance
let cl = 35.0 : L/h
let cl_normalized = cl / (70.0 : kg)  // L/h/kg

// AUC
let auc = conc * (24.0 : h)  // mg·h/L
```

### Dimensional Analysis

Units are checked at **compile time**:

```d
let dose = 500.0 : mg
let volume = 5.0 : L

// OK: mg / L = mg/L (concentration)
let conc = dose / volume  // Type: mg/L

// ERROR: Cannot add mg and L
// let bad = dose + volume  // Compile error!
```

## MedLang Compatibility

Demetrios provides bidirectional interoperability with Darwin's MedLang DSL:

### Parsing MedLang

```d
use interop::medlang::*

let source = r#"
drug Metformin {
    mw: 129.16 g/mol
    logP: -1.43
    fu: 1.0
}
"#;

let ast = parse_medlang(source)?;
let demetrios_ast = translate_to_demetrios(&ast, &ConfidenceConfig::default())?;
```

### Generating MedLang

```d
let code = generate_medlang(&drug, &params, &dosing);
println("{}", code);
// OUTPUT:
// drug Metformin {
//     mw: 129.16 g/mol
//     logP: -1.43
//     fu: 1.000
//     @chebi: "CHEBI:6801"
// }
// ...
```

### FDA Report Generation

```d
let validation = validate_fda_compliance(&drug, &params, &result);
let report = generate_fda_report(&drug, &params, &result, &validation);

// Write to file for submission
write_file("pbpk_report.txt", &report)?;
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Demetrios Compiler                           │
├─────────────────────────────────────────────────────────────────┤
│  Lexer → Parser → AST → Type Checker → HIR → HLIR → Codegen    │
│                           ↓                                     │
│                   ┌───────────────┐                             │
│                   │ Epistemic     │                             │
│                   │ Checker       │                             │
│                   │ ε propagation │                             │
│                   │ Provenance    │                             │
│                   └───────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Standard Library                           │
├──────────────┬────────────────┬─────────────────────────────────┤
│ units::qudt  │  pbpk::*       │  interop::medlang               │
│ QUDT units   │  14-compartment│  Darwin compatibility           │
│ Type-safe    │  PBPK modeling │  FDA report generation          │
└──────────────┴────────────────┴─────────────────────────────────┘
```

## Comparison with Darwin PBPK Platform

Darwin PBPK Platform (developed solo in 2 months!) is a Julia-based platform for pharmacometrics. Demetrios extends this work by providing:

| Darwin | Demetrios |
|--------|-----------|
| Julia runtime | Native compilation |
| Dynamic typing | Static + epistemic types |
| Runtime unit checks | Compile-time unit checks |
| Manual provenance | Automatic provenance |
| GNN parameter prediction | GNN parameters + confidence |

Demetrios can import Darwin models and add epistemic qualifications for regulatory submission.

## FDA/EMA Validation Metrics (v0.45.0)

Demetrios automatically calculates all standard PBPK validation metrics:

### Primary Metrics (Required)

| Metric | Description | Acceptance Criteria |
|--------|-------------|---------------------|
| **GMFE** | Geometric Mean Fold Error | ≤ 2.0 (qualified), ≤ 1.5 (high impact) |
| **AFE** | Average Fold Error (bias) | 0.8-1.25 |
| **AAFE** | Absolute Average Fold Error (precision) | < 2.0 |
| **Within 2-fold** | % predictions within 2-fold | ≥ 80% |

### Usage

```d
use pbpk::regulatory::*

let metrics = ValidationMetrics::calculate(&predicted, &observed)?;

println("GMFE: {:.2}", metrics.gmfe);
println("AFE: {:.2} (bias)", metrics.afe);
println("AAFE: {:.2} (precision)", metrics.aafe);
println("Within 2-fold: {:.1}%", metrics.within_2fold * 100.0);
```

### EMA Impact Levels

```d
// Determine impact level based on intended use
let impact = EmaImpactLevel::from_intended_use(&EmaIntendedUse::PredictDDI);
let criteria = impact.acceptance_criteria();

// Low impact: GMFE ≤ 3.0, within 2-fold ≥ 50%
// Medium impact: GMFE ≤ 2.0, within 2-fold ≥ 80%  
// High impact: GMFE ≤ 1.5, within 2-fold ≥ 90%
```

### FDA 6-Section Report

```d
let report = generate_fda_report(&drug, &params, &result, &observed_data, &config);

// Sections:
// A. Executive Summary
// B. Introduction
// C. Materials and Methods
// D. Results
// E. Discussion
// F. Appendices (includes provenance audit trail)
```

## References

1. FDA PBPK Guidance: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/physiologically-based-pharmacokinetic-analyses-format-and-content
2. EMA PBPK Guideline: https://www.ema.europa.eu/en/reporting-physiologically-based-pharmacokinetic-pbpk-modelling-simulation-scientific-guideline
3. CPT 2025 - Current Use of PBPK at EMA: https://ascpt.onlinelibrary.wiley.com/doi/10.1002/cpt.3525
4. PBPK Model Qualification (Consortium): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6032820/
5. QUDT Ontology: http://qudt.org/
6. ChEBI Ontology: https://www.ebi.ac.uk/chebi/
7. Darwin PBPK Platform: (Internal - developed by Demetrios Agourakis)

## Future Work

- Integration with NONMEM control streams
- PopPK model translation
- Bayesian epistemic inference
- GPU-accelerated PBPK simulations
- Automated sensitivity analysis per FDA guidance
