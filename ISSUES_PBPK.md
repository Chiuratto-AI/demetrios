# Demetrios Compiler - Issues from PBPK Development

## Date: 2024-12-08
## Compiler Version: 0.52.0

---

## PARSER ERRORS

### 1. Slash in Types Not Supported
- Error: Expected identifier, found Slash
- Example: crcl: mL/min fails
- Workaround: Use f64 instead
- Priority: MEDIUM

### 2. Unicode Characters Not Supported  
- Error: Unexpected character epsilon
- Workaround: Use epsilon instead of e
- Priority: LOW

### 3. Binary Operators in If-Else Branches
- Error: Expected Comma, found Star/Plus
- Example: if x { a * b } else { c } fails
- Workaround: Use intermediate variables
- Priority: HIGH

### 4. Never Type (!) Not Parsed
- Error: Expected type, found Bang
- Priority: LOW

---

## TYPE CHECKER ISSUES

### 5. Unit Compatibility Not Enforced
- Issue: dose@mg + time@h compiles (should fail)
- Priority: HIGH

### 6. Unit Inference Not Implemented
- Issue: dose@mg / volume@L should infer mg/L
- Priority: MEDIUM

---

## MISSING FEATURES FOR PBPK

### 7. No Print/IO in Interpreter - HIGH
### 8. No Array Literals with Size - MEDIUM  
### 9. No Refinement Types - LOW
### 10. No Epistemic Types - LOW

---

## WHAT WORKS

- Struct definitions
- Function definitions  
- f64@unit annotation syntax
- Nested if statements
- Basic arithmetic (outside if branches)
- Struct instantiation

---

Filed by: Claude Code
Repository: /mnt/e/workspace/demetrios
