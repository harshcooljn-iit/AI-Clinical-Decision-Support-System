# Model Evaluation — MedGemma-4B vs. MedGemma-27B Ablation Study

This directory contains the evaluation pipeline for comparing **MedGemma-4B** and **MedGemma-27B** model variants used in the AI-Clinical-Decision-Support-System's multi-agent RAG framework.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (first-time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 3. Run the evaluation
python evaluate_models.py \
    --fourb  medgemma_4b_results.csv \
    --twentysevenb  medgemma_27b_results.csv \
    --output-dir  ./eval_output
```

---

## Input Format

Both CSV files **must** contain the following columns:

| Column                | Description                                                         |
|-----------------------|---------------------------------------------------------------------|
| `Patient_ID`          | Unique patient identifier                                           |
| `EHR_Snippet`         | Raw Electronic Health Record text for the patient                   |
| `Selected_Guidelines` | Comma-separated list of clinical guidelines matched to the patient   |
| `Extracted_Variables` | Python-dict string of extracted clinical variables (key: question, value: answer) |
| `Final_Output`        | The model's complete generated output (reasoning + prescription)     |

---

## Metrics Computed

### 1. Variable Extraction Accuracy
Measures how completely and correctly the model extracts clinical variables from the EHR.

- **UNKNOWN Rate** — Fraction of variables the model could not resolve (left as `UNKNOWN`). Lower is better.
- **Cross-model Agreement** — Cohen's kappa over shared variable keys between 4B and 27B outputs.
- **Deterministic Extraction** — Counts of Yes/No/numeric answers vs. UNKNOWN answers.

### 2. Clinical Reasoning Quality
Assesses the depth and rigour of the model's clinical reasoning section.

- **Reasoning Steps** — Number of distinct logical steps (bullet/numbered items).
- **Guideline References** — Count of explicit guideline citations in the reasoning.
- **Logical Connectors** — Use of causal/contrastive language (*therefore*, *however*, *given*, etc.).
- **Patient-specific References** — Mentions of the patient's specific condition in the reasoning.
- **Composite Quality Score** (0–1) — Weighted combination of the above.

### 3. Completeness / Coverage
Measures how thoroughly the output addresses all relevant clinical items.

- **Guideline Coverage** — Fraction of matched guidelines that receive explicit treatment.
- **Variable Coverage** — Fraction of definitive (non-UNKNOWN) variables referenced in the output.
- **Composite Completeness Score** (0–1) — Average of guideline and variable coverage.

### 4. Hallucination Rate (Heuristic)
Lower-bound estimate of clinically unsupported generation.

- **Hallucinated Drug Names** — Drug names in the output not found in the AIIMS guideline pharmacopoeia.
- **Implausible Dosages** — Numeric dosage values exceeding clinically reasonable ranges.
- **Contradiction Indicators** — Co-occurrence of contradictory recommendations in the same output.
- **Composite Hallucination Score** (0–1) — Weighted combination; 0 = no hallucination detected.

### 5. Formatting Compliance
Checks adherence to the expected clinical decision-support output format.

- **Required Sections** — Presence of *Clinical Reasoning*, *Final Prescription*, *Abbreviations*.
- **Structural Elements** — Bold headers, bullet/numbered lists, emoji markers.
- **Section Order** — Reasoning appears before prescription.
- **Compliance Score** (0–1) — Fraction of checks passed.

### 6. Repetition / Degradation Rate
Quantifies degenerate or infinite-loop generation patterns.

- **Line Repetition Ratio** — 1 − (unique lines / total lines).
- **N-gram Repetition Ratio** — 1 − (unique n-grams / total n-grams), n = 3.
- **Max Consecutive Duplicates** — Longest run of identical consecutive lines.
- **Truncation Detection** — Whether the output ends mid-sentence.
- **Degradation Score** (0–1) — Weighted combination; 0 = no degradation.

### 7. Output Length Statistics
Descriptive statistics of generated output length (characters, words, sentences).

---

## Output Files

| File                        | Description                                           |
|-----------------------------|-------------------------------------------------------|
| `metrics_summary.csv`       | Per-patient metric table (100 rows: 50 × 2 models)   |
| `aggregate_comparison.csv`  | Side-by-side mean comparison per metric               |
| `metrics_summary.json`      | Machine-readable full results (includes variable agreement & Cohen's kappa) |
| `evaluation_report.txt`     | Human-readable text report                            |

---

## Interpreting Results

| Metric                     | Direction | What It Tells You                                    |
|----------------------------|-----------|------------------------------------------------------|
| UNKNOWN Rate               | ↓ Lower   | Better clinical variable extraction from EHR         |
| Reasoning Quality          | ↑ Higher  | More structured, guideline-aware clinical reasoning  |
| Completeness Score         | ↑ Higher  | More guideline items and variables addressed          |
| Hallucination Score        | ↓ Lower   | Fewer unsupported / fabricated clinical claims       |
| Formatting Compliance      | ↑ Higher  | Better adherence to the expected output template     |
| Degradation Score          | ↓ Lower   | Less repetitive / degenerate generation              |

---

## Extending the Evaluation

- **Adding ground-truth labels:** Create a `ground_truth.csv` with expected variable values and prescription elements, then add an exact-match / F1 evaluation layer.
- **LLM-as-judge:** Use an external LLM (e.g., GPT-4) to score each output on clinical accuracy and safety.
- **Expert review:** Export `metrics_summary.csv` and add clinician-annotated columns for real-world validation.

---

## Citation

If you use this evaluation script in your research, please cite:

> H. Jain, "AI-Clinical-Decision-Support-System: A Multi-Agent RAG Framework for Guideline-Adherent Clinical Decision Support," B.Tech.+M.Tech. Thesis, IIT Kharagpur, 2025.

---

## License

This evaluation script is released under the same license as the parent repository (AI-Clinical-Decision-Support-System).
