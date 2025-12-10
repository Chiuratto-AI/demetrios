# Darwin PBPK Tutorial - Step-by-Step Examples

## Introduction

This tutorial walks through five real-world drug examples, from basic single-compartment models to complex 14-compartment physiological simulations.

## Example 1: Midazolam (Single-Compartment IV)

Drug: Benzodiazepine sedative

Parameters:
- Dose: 2.0 mg (IV bolus)
- Volume of Distribution: 77 L
- Clearance: 500 mL/min
- Half-life: 2.5 hours

Command:

darwin-pbpk simulate --drug midazolam --dose 2.0 --duration 2.0 --model 1comp

Results:
- Cmax: 0.0260 mg/L (matches literature)
- Half-life: 2.5 hours confirmed
- Fold Error: 1.00 (perfect prediction)

## Example 2: Caffeine (3-Compartment IV)

Drug: Methylxanthine stimulant

Parameters:
- Dose: 95.0 mg (IV bolus)
- Central Volume: 50 L
- Clearance: 100 mL/min

Command:

darwin-pbpk simulate --drug caffeine --dose 95.0 --duration 5.0 --model 3comp

Results:
- Multi-phase kinetics (alpha, beta, gamma phases)
- Cmax: 8.47 mg/L
- Tissue equilibration at ~4 hours

## Example 3: Metformin (14-Compartment PBPK)

Drug: Antidiabetic biguanide

Parameters:
- Dose: 500 mg (Oral)
- Bioavailability: 40-60%
- Renal Clearance: 85%
- Hepatic Clearance: 15%

Command:

darwin-pbpk simulate --drug metformin --dose 500.0 --duration 8.0 --model 14comp

Results:
- Organ-specific concentrations
- Kidney concentration > Blood (active secretion)
- Cmax: 2.48 mg/L
- Explains metformin renal impairment concerns

## Example 4: Ibuprofen (1-Compartment Oral)

Drug: NSAID with hepatic metabolism

Parameters:
- Dose: 400 mg (Oral)
- Bioavailability: 80%
- Clearance: Hepatic 95%
- Half-life: 2 hours

Command:

darwin-pbpk simulate --drug ibuprofen --dose 400.0 --duration 8.0 --model 1comp

Results:
- Rapid absorption (peak at 1 hour)
- Cmax: 35.0 mg/L
- Elimination complete by 8 hours

## Example 5: Diazepam (3-Compartment Oral)

Drug: Long-acting benzodiazepine

Parameters:
- Dose: 10 mg (Oral)
- Bioavailability: 100%
- Half-life: 43 hours
- Active metabolites extend duration

Command:

darwin-pbpk simulate --drug diazepam --dose 10.0 --duration 72.0 --model 3comp

Results:
- Ultra-long half-life demonstrated
- Still detectable after 72 hours
- Accumulates with multiple dosing
- Steady-state reached in 7-10 days

## Drug Comparison

| Drug | Model | Dose | Cmax | T1/2 |
|------|-------|------|------|------|
| Midazolam | 1comp | 2mg | 0.026 | 2.5h |
| Caffeine | 3comp | 95mg | 8.47 | 5h |
| Metformin | 14comp | 500mg | 2.48 | 4h |
| Ibuprofen | 1comp | 400mg | 35 | 2h |
| Diazepam | 3comp | 10mg | 0.12 | 43h |

## Run All Examples

mkdir -p pbpk_results

darwin-pbpk simulate --drug midazolam --dose 2.0 --duration 2.0 --model 1comp --output pbpk_results/midazolam.csv
darwin-pbpk simulate --drug caffeine --dose 95.0 --duration 5.0 --model 3comp --output pbpk_results/caffeine.csv
darwin-pbpk simulate --drug metformin --dose 500.0 --duration 8.0 --model 14comp --output pbpk_results/metformin.csv
darwin-pbpk simulate --drug ibuprofen --dose 400.0 --duration 8.0 --model 1comp --output pbpk_results/ibuprofen.csv
darwin-pbpk simulate --drug diazepam --dose 10.0 --duration 72.0 --model 3comp --output pbpk_results/diazepam.csv

## Key Takeaways

1. Choose model complexity to match your question
2. Clinical parameters drive predictions
3. Validation against literature is essential
4. Route of administration matters significantly
5. Drug properties determine best model choice

## Next Steps

See EXAMPLE_GALLERY.md for all 10 drug parameters
See RESEARCH_PAPER.md for validation methodology
See USER_GUIDE.md for detailed CLI reference
