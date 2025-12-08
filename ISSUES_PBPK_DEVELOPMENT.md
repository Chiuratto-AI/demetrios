# Demetrios Compiler - Issues & Improvements from PBPK Development

**Date**: 2024-12-08
**Context**: Developing Darwin PBPK 14-Compartment Model in Demetrios
**Compiler Version**: 0.52.0

---

## PARSER ERRORS

### 1. Slash in Types Not Supported
**File**: 
**Error**: 
**Code that fails**:

**Workaround**: Use  instead of compound unit types
**Suggestion**: Support  syntax in type positions

---

### 2. Unicode Characters Not Supported
**Error**: 
**Code that fails**:

**Workaround**: Use  instead of 
**Suggestion**: Support Unicode identifiers (Greek letters common in science)

---

### 3. Multiplication in If-Else Branches
**Error**: 
**Code that fails**:

**Workaround**: Split into separate statements:

**Suggestion**: Fix binary operator parsing in if-else expressions

---

### 4. Addition in If-Else Branches
**Error**: 
**Code that fails**:

**Workaround**: Same as above - use intermediate variables
**Suggestion**: Same fix needed for all binary operators

---

### 5. Bang Type (Never) Not Parsed
**File**: 
**Error**: 
**Code that fails**:

**Suggestion**: Implement  (never/bottom type) parsing

---

## TYPE CHECKER ISSUES

### 6. Unit Compatibility Not Enforced
**Issue**: Adding incompatible units compiles without error
**Code that should fail**:

**Current behavior**: Compiles successfully
**Expected behavior**: Type error - incompatible units
**Suggestion**: Integrate UnitChecker into type checking pass

---

### 7. Unit Inference Not Implemented
**Issue**: Result of unit operations not inferred
**Code**:

**Suggestion**: Implement unit inference for arithmetic operations

---

## MISSING FEATURES FOR PBPK

### 8. No Array Literals with Size
**Needed for**:

**Suggestion**: Implement fixed-size array literals

---

### 9. No Refinement Type Enforcement
**Needed for**:

**Current**: Syntax not supported
**Suggestion**: Add refinement types with SMT verification

---

### 10. No Epistemic Types
**Needed for**:

**Current**: Generic syntax parses but semantics not implemented
**Suggestion**: Implement epistemic type tracking with confidence propagation

---

### 11. No Print/IO in Interpreter
**Issue**: Cannot output results when running
**Code**:

**Suggestion**: Implement  or  built-in

---

### 12. No Floating Point Literals with Units
**Needed for**:

**Suggestion**: Support unit suffixes on literals

---

## STDLIB IMPROVEMENTS

### 13. QUDT Module Not Loadable
**File**: 
**Issue**: Contains syntax not yet supported (, complex expressions)
**Suggestion**: Update stdlib to match current parser capabilities

---

### 14. PBPK Module Incomplete
**File**: 
**Issue**: Uses advanced features not yet implemented
**Suggestion**: Create minimal working PBPK stdlib

---

## PRIORITY RECOMMENDATIONS

### High Priority (Blocks PBPK Development)
1. Fix binary operators in if-else expressions (#3, #4)
2. Implement unit compatibility checking (#6)
3. Add print/IO capability (#11)

### Medium Priority (Improves Usability)
4. Support compound unit types in parser (#1)
5. Implement unit inference (#7)
6. Add array literals (#8)

### Lower Priority (Nice to Have)
7. Unicode identifier support (#2)
8. Never type (#5)
9. Refinement types (#9)
10. Epistemic types (#10)

---

## WORKING SYNTAX EXAMPLES

### What Works Now:


---

**Filed by**: Claude Code during Darwin PBPK development session
**Repository**: /mnt/e/workspace/demetrios
