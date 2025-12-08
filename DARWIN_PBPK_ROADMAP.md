# Darwin PBPK Platform - Demetrios Migration Roadmap

**Goal**: Port Darwin PBPK from Julia to Demetrios with 80%+ validation accuracy
**Status**: 70% validation (7/10 drugs within 2-fold)
**Target**: 80%+ (8/10 drugs within 2-fold, GMFE < 1.5)

## PHASE 1: Compiler Enhancements

### 1.1 While Loop Implementation
- [ ] Trace while loop from parser to codegen
- [ ] Implement LLVM IR emission for while loops
- [ ] Test with ODE time-stepping loop

### 1.2 Array Operations  
- [ ] Implement array indexing in codegen
- [ ] Add array bounds checking
- [ ] Test with compartment state arrays

### 1.3 Unit System Improvements
- [ ] Add compound unit inference
- [ ] Improve error messages for unit mismatches

## PHASE 2: Clinical Data Collection

### 2.1 Literature Data Extraction
- [ ] Midazolam PK from FDA NDA 021466
- [ ] Caffeine PK from Arnaud 2011
- [ ] Metformin PK from Tucker 1981

### 2.2 Drug Parameter Database
- [ ] Create drug_parameters.d with 20+ drugs
- [ ] Include LogP, pKa, fu, Rb, MW

## PHASE 3: PBPK Calibration

### 3.1 Mechanistic Improvements
- [ ] Implement Poulin-Theil Kp method
- [ ] Add Berezhkovskiy correction

### 3.2 Transporter Models
- [ ] Add P-gp efflux
- [ ] Add OATP1B1 hepatic uptake
- [ ] Add OCT2 renal secretion

### 3.3 Validation Campaign (10 drugs)
- [ ] Midazolam, Caffeine, Metformin
- [ ] Ibuprofen, Diazepam, Omeprazole
- [ ] Warfarin, Digoxin, Atorvastatin, Morphine

## PHASE 4: Julia Backend

- [ ] Create codegen/julia.rs module
- [ ] Generate ODEProblem from PBPK model
- [ ] Benchmark LLVM vs Julia

## PHASE 5: CLI and Documentation

- [ ] Create darwin-pbpk CLI binary
- [ ] Write Demetrios PBPK tutorial
- [ ] Create validation report template

## Current Status: Phase 1 - Verify while loop codegen
