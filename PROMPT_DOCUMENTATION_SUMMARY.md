# Prompt Documentation Summary

## Overview

Two comprehensive prompt documentation files have been created for inclusion in the paper appendix:

1. **APPENDIX_PROMPTS.md** - English version (detailed)
2. **APPENDIX_PROMPTS_CN.md** - Chinese version (concise)

---

## Document Structure

### 1. Claim Decomposition (1 prompt)
- Decomposes complex claims into 2-4 atomic, verifiable sub-claims
- Defines 6 sub-claim types: Entity, Event, Time, Quantity, Qualifier, Relationship
- Output: JSON format with verification_type and rationale

### 2. Pro Agent Query Generation (2 prompts)
- **First Round**: Generate 2-3 initial supporting queries
- **Follow-up Rounds**: Generate 1 targeted query to counter opponent
- Focus on authority, precision, and diversity

### 3. Con Agent Query Generation (2 prompts)
- **First Round**: Generate 2-3 initial refuting queries
- **Follow-up Rounds**: Generate 1 targeted query to counter opponent
- Focus on identifying vulnerabilities and questionable aspects

### 4. Argument Extraction (1 prompt)
- Extracts specific arguments from evidence documents
- Addresses sub-claims with confidence scoring (0.3-1.0)
- Includes published time extraction
- Output: JSON with point_text, supporting_snippet, confidence

### 5. Attack Relation Detection (1 prompt)
- Identifies conflicts between arguments for argumentation framework
- 4 conflict types: contradictory_conclusion, contradictory_evidence, logical_opposition, direct_refutation
- Output: CONFLICT pairs or NO_CONFLICTS

### 6. Judge Verdict Generation (1 prompt)
- Generates final verdict: Supported / Refuted / Not Enough Evidence
- 5-step judgment protocol
- 3 critical evaluation rules: Qualifier Contradictions, Self-Consistency, Burden of Proof
- Enables web search for evidence gap filling
- Output: JSON with justification, verdict, key_evidence

---

## Total Prompts: 8

| Module | Prompts | Purpose |
|--------|---------|---------|
| Claim Decomposition | 1 | Break down complex claims |
| Pro Agent | 2 | Generate supporting queries |
| Con Agent | 2 | Generate refuting queries |
| Argument Extraction | 1 | Extract structured arguments |
| Attack Detection | 1 | Build argumentation framework |
| Judge | 1 | Generate final verdict |

---

## Key Features

### 1. Language Consistency
- All prompts maintain the language of input claim (Chinese/English)
- Bilingual examples provided in query generation prompts

### 2. Structured Output
- Most prompts require JSON format for reliable parsing
- Clear output specifications with examples

### 3. Quality Control
- Specific quality standards in each prompt
- Confidence scoring for argument extraction
- Validation rules for output

### 4. Argumentation Framework Integration
- Attack relation detection based on abstract argumentation theory
- Grounded extension computation for accepted arguments
- Bidirectional attack support (authority + timeliness)

### 5. Web Search Integration
- Judge prompt explicitly enables web search
- Used to fill evidence gaps in uncovered sub-claims
- Ensures comprehensive verification

---

## Usage in Paper

### Recommended Placement
Place in **Appendix A: System Prompts** or **Supplementary Materials**

### Citation Format
```
For implementation details of our prompts, see Appendix A.
The claim decomposition prompt (Appendix A.1) breaks down complex 
claims into 2-4 atomic sub-claims...
```

### Key Points to Highlight in Main Text
1. **Multi-round adversarial search** with adaptive query generation
2. **Sub-claim based argument extraction** for fine-grained verification
3. **Abstract argumentation framework** for conflict resolution
4. **5-step judgment protocol** with critical evaluation rules
5. **Web search integration** for evidence gap filling

---

## File Locations

```
ArgCheck_v1/
├── APPENDIX_PROMPTS.md          # English version (detailed)
├── APPENDIX_PROMPTS_CN.md       # Chinese version (concise)
└── PROMPT_DOCUMENTATION_SUMMARY.md  # This file
```

---

## Maintenance Notes

- All prompts are extracted from actual implementation code
- Prompts are version-controlled with the codebase
- Any updates to prompts should be reflected in both documentation files
- Current version: 1.0 (2026-01-20)

---

## Reproducibility

The documented prompts enable:
1. **Full system replication** - All critical prompts are provided
2. **Ablation studies** - Individual prompts can be modified/removed
3. **Comparison studies** - Prompts can be compared with other systems
4. **Extension research** - Prompts can be adapted for related tasks


