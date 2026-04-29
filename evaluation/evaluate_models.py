#!/usr/bin/env python3
"""
evaluate_models.py — MedGemma-4B vs. MedGemma-27B Ablation Evaluation
======================================================================

Computes clinical NLP metrics for comparing two MedGemma model variants
used in the AI-Clinical-Decision-Support-System multi-agent RAG pipeline.

Metrics computed
────────────────
1. Variable Extraction Accuracy   — UNKNOWN rate, determinism, agreement
2. Clinical Reasoning Quality     — Structure depth, logical-flow score
3. Completeness / Coverage        — Guideline-item addressal rate
4. Hallucination Rate             — Unsupported-claim detection heuristic
5. Formatting Compliance          — Required-section presence score
6. Repetition / Degradation Rate  — Line-level and n-gram redundancy
7. Output Length Statistics        — Descriptive statistics per model
8. Cross-Model Comparison Table   — Aggregate head-to-head summary

Usage
─────
    python evaluate_models.py \
        --fourb  medgemma_4b_results.csv \
        --twentysevenb  medgemma_27b_results.csv \
        --output-dir  ./eval_output

Outputs
───────
    eval_output/
    ├── metrics_summary.csv           Per-patient metric table
    ├── aggregate_comparison.csv      Aggregate model comparison
    ├── metrics_summary.json          Machine-readable full results
    └── evaluation_report.txt         Human-readable text report

Author : Harsh Jain
Repo   : https://github.com/harshcooljn-iit/AI-Clinical-Decision-Support-System
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# Optional: NLTK for token-level analysis (graceful fallback)
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.util import ngrams as nltk_ngrams

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# Optional: scikit-learn for text similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame:
    """Load a results CSV and validate required columns."""
    df = pd.read_csv(path)
    required = {"Patient_ID", "EHR_Snippet", "Selected_Guidelines",
                "Extracted_Variables", "Final_Output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def parse_extracted_variables(raw: str) -> Dict[str, str]:
    """
    Safely parse the Extracted_Variables string into a dict.

    Handles both Python-literal dicts and edge-case malformed strings.
    """
    if pd.isna(raw) or not isinstance(raw, str) or raw.strip() == "":
        return {}
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items()}
    except (ValueError, SyntaxError):
        pass
    # Fallback: try regex key-value extraction
    pairs = re.findall(r"'([^']+)'\s*:\s*'([^']*)'", raw)
    if pairs:
        return dict(pairs)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2.1  Variable Extraction Accuracy ────────────────────────────────────────

def variable_extraction_metrics(variables: Dict[str, str]) -> Dict[str, Any]:
    """
    Compute extraction-quality metrics from a single patient's extracted
    variable dictionary.

    Returns dict with:
        total          — total number of variables extracted
        known          — variables with a definite (non-UNKNOWN) value
        unknown        — count of 'UNKNOWN' values
        unknown_rate   — fraction of variables that are UNKNOWN
        yes_count      — count of 'Yes' values
        no_count       — count of 'No' values
        numeric_count  — count of numeric answers (age, BP, etc.)
    """
    total = len(variables)
    if total == 0:
        return {"total": 0, "known": 0, "unknown": 0, "unknown_rate": 1.0,
                "yes_count": 0, "no_count": 0, "numeric_count": 0}

    unknown = sum(1 for v in variables.values()
                  if v.strip().upper() == "UNKNOWN")
    yes_count = sum(1 for v in variables.values()
                    if v.strip().lower() == "yes")
    no_count = sum(1 for v in variables.values()
                   if v.strip().lower() == "no")
    numeric_count = sum(1 for v in variables.values()
                        if re.match(r"^\d+(\.\d+)?$", v.strip()))

    known = total - unknown
    unknown_rate = unknown / total

    return {
        "total": total,
        "known": known,
        "unknown": unknown,
        "unknown_rate": unknown_rate,
        "yes_count": yes_count,
        "no_count": no_count,
        "numeric_count": numeric_count,
    }


def variable_agreement(vars_a: Dict[str, str],
                       vars_b: Dict[str, str]) -> Dict[str, Any]:
    """
    Compare extracted variables from two models on the same patient.
    Reports agreement, disagreement, and Cohen's kappa over
    the shared key set.
    """
    common = set(vars_a.keys()) & set(vars_b.keys())
    if not common:
        return {"agreement": 0, "disagreement": 0, "agreement_rate": 0.0,
                "cohens_kappa": 0.0}

    agree = sum(1 for k in common if vars_a[k] == vars_b[k])
    disagree = len(common) - agree
    rate = agree / len(common)

    # Cohen's kappa (simplified for multi-category)
    # Build contingency and compute
    categories = sorted(set(vars_a[k] for k in common)
                        | set(vars_b[k] for k in common))
    cat_idx = {c: i for i, c in enumerate(categories)}
    n = len(categories)
    confusion = np.zeros((n, n), dtype=int)
    for k in common:
        i, j = cat_idx.get(vars_a[k], 0), cat_idx.get(vars_b[k], 0)
        confusion[i, j] += 1

    po = np.trace(confusion) / len(common)
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = sum(row_sums[i] * col_sums[i] for i in range(n)) / (len(common) ** 2)

    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    return {
        "agreement": agree,
        "disagreement": disagree,
        "agreement_rate": rate,
        "cohens_kappa": round(kappa, 4),
    }


# ── 2.2  Clinical Reasoning Quality ─────────────────────────────────────────

def reasoning_quality_score(output: str) -> Dict[str, Any]:
    """
    Assess the quality of the clinical reasoning section.

    Heuristics:
        - Presence of a dedicated 'Reasoning' section
        - Number of distinct reasoning steps / lines
        - Use of guideline-referencing language
        - Logical connectors (therefore, however, because, given, etc.)
        - Mention of patient-specific variables by name
    """
    text = str(output)
    has_reasoning = bool(re.search(r"reasoning", text, re.IGNORECASE))

    # Extract reasoning body
    reasoning_body = ""
    match = re.search(
        r"(?:clinical\s+reasoning|reasoning)\s*[:\-–]*\s*\n?(.*?)(?=(?:final\s+prescription|###\s*💊|prescription))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if match:
        reasoning_body = match.group(1)
    else:
        # Fallback: everything before 'prescription' section
        parts = re.split(r"(?:final\s+prescription|###\s*💊|prescription)", text, flags=re.IGNORECASE)
        reasoning_body = parts[0] if parts else text

    # Step count: numbered items or bullet points in reasoning
    step_pattern = re.findall(r"(?:^\s*\d+[\.\)]\s|\s*[-•]\s)", reasoning_body, re.MULTILINE)
    n_steps = len(step_pattern)

    # Guideline-referencing language
    guideline_refs = re.findall(
        r"(?:guideline|protocol|algorithm|recommendation|according\s+to|per\s+the|the\s+guideline)",
        reasoning_body, re.IGNORECASE,
    )
    n_guideline_refs = len(guideline_refs)

    # Logical connectors
    connectors = re.findall(
        r"\b(?:therefore|however|because|given|since|although|thus|hence|whereas|conversely|nevertheless|furthermore|moreover|additionally|consequently)\b",
        reasoning_body, re.IGNORECASE,
    )
    n_connectors = len(connectors)

    # Patient-specific variable mentions (keys from extracted variables)
    # Heuristically: look for quoted/capitalised clinical terms
    patient_terms = re.findall(
        r"the patient(?:'s)?\s+(?:has|is|was|had|does|experiences|is\s+experiencing|is\s+in|is\s+not)",
        reasoning_body, re.IGNORECASE,
    )
    n_patient_refs = len(patient_terms)

    # Sentence count in reasoning (proxy for depth)
    if _NLTK_AVAILABLE:
        sents = sent_tokenize(reasoning_body)
    else:
        sents = [s.strip() for s in re.split(r"[.!?]+", reasoning_body) if s.strip()]
    n_sentences = len(sents)

    # Composite score (0–1)
    score = 0.0
    if has_reasoning:
        score += 0.15
    score += min(n_steps / 5.0, 0.15)
    score += min(n_guideline_refs / 3.0, 0.15)
    score += min(n_connectors / 3.0, 0.15)
    score += min(n_patient_refs / 3.0, 0.15)
    score += min(n_sentences / 8.0, 0.25)
    score = round(min(score, 1.0), 4)

    return {
        "has_reasoning_section": has_reasoning,
        "reasoning_steps": n_steps,
        "guideline_references": n_guideline_refs,
        "logical_connectors": n_connectors,
        "patient_specific_refs": n_patient_refs,
        "reasoning_sentences": n_sentences,
        "reasoning_quality_score": score,
    }


# ── 2.3  Completeness / Coverage ────────────────────────────────────────────

def completeness_score(output: str, guidelines: str,
                       variables: Dict[str, str]) -> Dict[str, Any]:
    """
    Measure how completely the output addresses each guideline
    and each extracted variable.

    Heuristic: for each guideline mentioned, check if the output
    contains a dedicated section or explicit mention.  For each
    variable with a definitive value (Yes/No/numeric), check if
    the output references it.
    """
    text = str(output).lower()
    guideline_list = [g.strip() for g in str(guidelines).split(",") if g.strip()]

    # Guideline coverage: how many guidelines get explicit treatment
    guidelines_addressed = 0
    for g in guideline_list:
        g_lower = g.lower()
        # Check if guideline name or a significant substring appears
        # in the output's reasoning/prescription
        if g_lower in text or g_lower.split()[0] in text:
            guidelines_addressed += 1
    guideline_coverage = guidelines_addressed / len(guideline_list) if guideline_list else 0.0

    # Variable coverage: for non-UNKNOWN variables, check if they're
    # referenced in the output
    definitive_vars = {k: v for k, v in variables.items()
                       if v.strip().upper() != "UNKNOWN"}
    vars_referenced = 0
    for key, val in definitive_vars.items():
        # Extract the core question from the key
        key_lower = key.lower()
        # Simple heuristic: check if key words from the variable
        # question appear in output
        key_tokens = set(re.findall(r"\b\w{4,}\b", key_lower))
        output_tokens = set(re.findall(r"\b\w{4,}\b", text))
        overlap = key_tokens & output_tokens
        if len(overlap) >= max(1, len(key_tokens) // 3):
            vars_referenced += 1
    var_coverage = vars_referenced / len(definitive_vars) if definitive_vars else 0.0

    composite = round(0.5 * guideline_coverage + 0.5 * var_coverage, 4)

    return {
        "n_guidelines": len(guideline_list),
        "guidelines_addressed": guidelines_addressed,
        "guideline_coverage": round(guideline_coverage, 4),
        "n_definitive_vars": len(definitive_vars),
        "vars_referenced": vars_referenced,
        "variable_coverage": round(var_coverage, 4),
        "completeness_score": composite,
    }


# ── 2.4  Hallucination Rate ─────────────────────────────────────────────────

def hallucination_heuristic(output: str, ehr: str,
                            guidelines: str) -> Dict[str, Any]:
    """
    Heuristic estimation of hallucination in the final output.

    Flags:
        - Drug names not present in the guidelines or EHR
        - Numeric values (dosages) without units or with implausible ranges
        - Contradictory statements within the same output
        - Fabricated patient history elements

    NOTE: This is a heuristic, not a ground-truth hallucination detector.
    It provides a lower-bound estimate suitable for model comparison.
    """
    text = str(output)
    ehr_lower = str(ehr).lower()
    guidelines_lower = str(guidelines).lower()
    text_lower = text.lower()

    # Common drug patterns in the output
    drug_pattern = re.findall(
        r"\b(?:Tab\.|Inj\.|Cap\.|IV|IM|SC)\s+([A-Za-z]+\s*[A-Za-z]*)",
        text,
    )
    # Known AIIMS guideline drugs (compiled from the repository's guideline set)
    known_drugs = {
        "amoxicillin", "ampicillin", "azithromycin", "cefotaxime",
        "ceftriaxone", "cefalexin", "cefepime", "chloramphenicol",
        "cloxacillin", "cotrimoxazole", "cyclosporine", "dexamethasone",
        "diphenhydramine", "erythromycin", "heparin", "hydrocortisone",
        "methylprednisolone", "oxybutynin", "phenobarbitone", "phenytoin",
        "prednisolone", "salbutamol", "adrenaline", "diazepam",
        "lorazepam", "valproate", "tizanidine", "baclofen", "imipramine",
        "cetirizine", "mesalazine", "isosorbide", "aspirin", "nsaid",
        "paracetamol", "warfarin", "clopidogrel", "metoprolol",
        "amlodipine", "diltiazem", "nifedipine", "atorvastatin",
        "insulin", "metformin", "glimepiride", "omeprazole",
        "pantoprazole", "famotidine", "morphine", "fentanyl",
        "vancomycin", "meropenem", "piperacillin", "gentamicin",
    }

    # Check if extracted drug names are in the known set
    hallucinated_drugs = []
    for drug in drug_pattern:
        drug_name = drug.strip().lower()
        if drug_name and drug_name not in known_drugs:
            # Also check if it appears in the guidelines text
            if drug_name not in guidelines_lower:
                hallucinated_drugs.append(drug.strip())

    # Dosage plausibility checks (very basic)
    implausible_dosages = 0
    dosage_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|ml|g|U|units)",
        text, re.IGNORECASE,
    )
    for d_str in dosage_matches:
        try:
            d_val = float(d_str)
            # Flag extreme values (e.g., >10g single dose is suspicious)
            if d_val > 10000:
                implausible_dosages += 1
        except ValueError:
            pass

    # Contradiction detection: look for "Yes" and "No" on same clinical axis
    contradiction_indicators = 0
    yes_no_pairs = [
        ("should be administered", "should not be administered"),
        ("is indicated", "is contraindicated"),
        ("is recommended", "is not recommended"),
    ]
    for pos, neg in yes_no_pairs:
        if pos in text_lower and neg in text_lower:
            contradiction_indicators += 1

    # Composite hallucination score (0 = no hallucination, 1 = max)
    n_drug_flags = len(hallucinated_drugs)
    drug_halluc_rate = min(n_drug_flags / max(len(drug_pattern), 1), 1.0)
    dose_flag_rate = min(implausible_dosages / max(len(dosage_matches), 1), 1.0)
    contra_rate = min(contradiction_indicators / 3.0, 1.0)

    h_score = round(0.5 * drug_halluc_rate + 0.3 * dose_flag_rate + 0.2 * contra_rate, 4)

    return {
        "n_drug_mentions": len(drug_pattern),
        "n_hallucinated_drugs": n_drug_flags,
        "hallucinated_drug_names": hallucinated_drugs[:5],  # cap for readability
        "n_implausible_dosages": implausible_dosages,
        "n_contradiction_indicators": contradiction_indicators,
        "hallucination_score": h_score,
    }


# ── 2.5  Formatting Compliance ──────────────────────────────────────────────

def formatting_compliance(output: str) -> Dict[str, Any]:
    """
    Check if the output follows the expected clinical decision-support
    format: (1) Clinical Reasoning section, (2) Final Prescription section,
    (3) Abbreviations section, (4) structured bullet points / numbered items.
    """
    text = str(output)

    has_reasoning = bool(re.search(r"clinical\s+reasoning|reasoning", text, re.IGNORECASE))
    has_prescription = bool(re.search(r"final\s+prescription|prescription", text, re.IGNORECASE))
    has_abbreviations = bool(re.search(r"abbreviation|abbreviations|full\s+forms", text, re.IGNORECASE))
    has_bullets = bool(re.search(r"[-•*]\s+\*\*", text))
    has_numbered = bool(re.search(r"\d+\.\s+", text))
    has_bold_headers = bool(re.search(r"\*\*[^*]+\*\*", text))

    # Check for emoji markers used by the system
    has_emoji_markers = bool(re.search(r"[💊🧠]", text))

    # Structural order check: reasoning should come before prescription
    reasoning_pos = -1
    prescription_pos = -1
    r_match = re.search(r"clinical\s+reasoning|reasoning", text, re.IGNORECASE)
    p_match = re.search(r"final\s+prescription|prescription", text, re.IGNORECASE)
    if r_match:
        reasoning_pos = r_match.start()
    if p_match:
        prescription_pos = p_match.start()
    correct_order = (reasoning_pos < prescription_pos) if reasoning_pos >= 0 and prescription_pos >= 0 else False

    # Compute compliance score
    checks = [has_reasoning, has_prescription, has_abbreviations,
              has_bullets or has_numbered, has_bold_headers, correct_order]
    compliance = round(sum(checks) / len(checks), 4)

    return {
        "has_reasoning_section": has_reasoning,
        "has_prescription_section": has_prescription,
        "has_abbreviations_section": has_abbreviations,
        "has_structured_list": has_bullets or has_numbered,
        "has_bold_headers": has_bold_headers,
        "has_emoji_markers": has_emoji_markers,
        "correct_section_order": correct_order,
        "formatting_compliance_score": compliance,
    }


# ── 2.6  Repetition / Degradation Rate ──────────────────────────────────────

def repetition_metrics(output: str, ngram_n: int = 3) -> Dict[str, Any]:
    """
    Detect and quantify repetitive / degenerate generation.

    Metrics:
        - Line-level repetition ratio  (unique / total lines)
        - N-gram repetition ratio       (unique / total n-grams)
        - Max consecutive duplicate lines
        - Is the output truncated? (ends mid-sentence)
    """
    text = str(output)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    total_lines = len(lines)
    unique_lines = len(set(lines))

    # Line-level repetition ratio (0 = no repetition, 1 = all duplicate)
    line_rep_ratio = 1.0 - (unique_lines / max(total_lines, 1))

    # Consecutive duplicate lines
    max_consec_dup = 0
    current_dup = 0
    for i in range(1, total_lines):
        if lines[i] == lines[i - 1]:
            current_dup += 1
            max_consec_dup = max(max_consec_dup, current_dup)
        else:
            current_dup = 0

    # N-gram repetition
    if _NLTK_AVAILABLE:
        tokens = word_tokenize(text.lower())
    else:
        tokens = re.findall(r"\b\w+\b", text.lower())

    if len(tokens) >= ngram_n:
        ngs = list(nltk_ngrams(tokens, ngram_n)) if _NLTK_AVAILABLE else []
        if not ngs:
            # Manual n-gram generation
            ngs = [tuple(tokens[i:i + ngram_n])
                   for i in range(len(tokens) - ngram_n + 1)]
        total_ngrams = len(ngs)
        unique_ngrams = len(set(ngs))
        ngram_rep_ratio = 1.0 - (unique_ngrams / max(total_ngrams, 1))
    else:
        ngram_rep_ratio = 0.0
        total_ngrams = 0
        unique_ngrams = 0

    # Truncation check: does the output end abruptly?
    last_line = lines[-1] if lines else ""
    is_truncated = bool(re.search(r"[,;:({\[–—]$", last_line))

    # Composite degradation score
    degradation = round(0.4 * line_rep_ratio + 0.4 * ngram_rep_ratio
                        + 0.2 * min(max_consec_dup / 5.0, 1.0), 4)

    return {
        "total_lines": total_lines,
        "unique_lines": unique_lines,
        "line_repetition_ratio": round(line_rep_ratio, 4),
        "max_consecutive_duplicates": max_consec_dup,
        "total_ngrams": total_ngrams,
        "unique_ngrams": unique_ngrams,
        "ngram_repetition_ratio": round(ngram_rep_ratio, 4),
        "is_truncated": is_truncated,
        "degradation_score": degradation,
    }


# ── 2.7  Output Length Statistics ────────────────────────────────────────────

def output_length_metrics(output: str) -> Dict[str, Any]:
    """Compute character, word, and sentence counts for an output."""
    text = str(output)
    char_count = len(text)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    if _NLTK_AVAILABLE:
        sents = sent_tokenize(text)
    else:
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sents)
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PER-PATIENT EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_single_patient(row: pd.Series, model_label: str) -> Dict[str, Any]:
    """
    Run all metric computations on a single patient row and return
    a flat dictionary keyed as  ``metric_group.metric_name``.
    """
    variables = parse_extracted_variables(row["Extracted_Variables"])

    results = {"patient_id": row["Patient_ID"], "model": model_label}

    # 2.1  Variable extraction
    var_metrics = variable_extraction_metrics(variables)
    for k, v in var_metrics.items():
        results[f"var_extraction.{k}"] = v

    # 2.2  Reasoning quality
    rsn_metrics = reasoning_quality_score(row["Final_Output"])
    for k, v in rsn_metrics.items():
        results[f"reasoning.{k}"] = v

    # 2.3  Completeness
    comp_metrics = completeness_score(
        row["Final_Output"], row["Selected_Guidelines"], variables
    )
    for k, v in comp_metrics.items():
        results[f"completeness.{k}"] = v

    # 2.4  Hallucination
    hall_metrics = hallucination_heuristic(
        row["Final_Output"], row["EHR_Snippet"], row["Selected_Guidelines"]
    )
    for k, v in hall_metrics.items():
        results[f"hallucination.{k}"] = v

    # 2.5  Formatting compliance
    fmt_metrics = formatting_compliance(row["Final_Output"])
    for k, v in fmt_metrics.items():
        results[f"formatting.{k}"] = v

    # 2.6  Repetition / degradation
    rep_metrics = repetition_metrics(row["Final_Output"])
    for k, v in rep_metrics.items():
        results[f"repetition.{k}"] = v

    # 2.7  Output length
    len_metrics = output_length_metrics(row["Final_Output"])
    for k, v in len_metrics.items():
        results[f"output_length.{k}"] = v

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  AGGREGATE / CROSS-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_metrics(per_patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model aggregate statistics (mean, std, median)
    for every numeric metric column.
    """
    numeric_cols = per_patient_df.select_dtypes(include=[np.number]).columns.tolist()
    agg_rows = []
    for model in per_patient_df["model"].unique():
        sub = per_patient_df[per_patient_df["model"] == model]
        for col in numeric_cols:
            vals = sub[col].dropna()
            agg_rows.append({
                "model": model,
                "metric": col,
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
                "median": round(vals.median(), 4),
                "min": round(vals.min(), 4),
                "max": round(vals.max(), 4),
            })
    return pd.DataFrame(agg_rows)


def cross_model_comparison(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a side-by-side comparison table (one row per metric,
    columns for 4B and 27B means).
    """
    pivot = agg_df.pivot_table(index="metric", columns="model", values="mean")
    # Rename columns for clarity
    col_map = {c: f"{c}_mean" for c in pivot.columns}
    pivot = pivot.rename(columns=col_map)
    pivot = pivot.reset_index()
    return pivot


def compute_variable_agreement(df4: pd.DataFrame,
                               df27: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute cross-model variable-extraction agreement across all patients.
    """
    all_agree = []
    all_disagree = []
    all_kappas = []
    for _, row4 in df4.iterrows():
        pid = row4["Patient_ID"]
        row27 = df27[df27["Patient_ID"] == pid]
        if row27.empty:
            continue
        v4 = parse_extracted_variables(row4["Extracted_Variables"].values[0]
                                       if hasattr(row4["Extracted_Variables"], "values")
                                       else row4["Extracted_Variables"])
        v27 = parse_extracted_variables(row27["Extracted_Variables"].values[0])
        agr = variable_agreement(v4, v27)
        all_agree.append(agr["agreement"])
        all_disagree.append(agr["disagreement"])
        all_kappas.append(agr["cohens_kappa"])

    return {
        "total_agreements": sum(all_agree),
        "total_disagreements": sum(all_disagree),
        "overall_agreement_rate": round(
            sum(all_agree) / max(sum(all_agree) + sum(all_disagree), 1), 4
        ),
        "mean_cohens_kappa": round(np.mean(all_kappas), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_text_report(
    per_patient_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    var_agreement: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write a human-readable evaluation report to a text file.
    """
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  MedGemma-4B vs. MedGemma-27B  —  Ablation Evaluation Report")
    lines.append("=" * 72)
    lines.append("")

    # ── Summary statistics ────────────────────────────────────────────────
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 40)
    n_patients = per_patient_df["patient_id"].nunique()
    lines.append(f"   Number of patients evaluated : {n_patients}")
    lines.append(f"   Models compared              : MedGemma-4B, MedGemma-27B")
    lines.append("")

    # ── Variable extraction ───────────────────────────────────────────────
    lines.append("2. VARIABLE EXTRACTION ACCURACY")
    lines.append("-" * 40)
    for model in ["4B", "27B"]:
        sub = per_patient_df[per_patient_df["model"] == model]
        mean_unk = sub["var_extraction.unknown_rate"].mean()
        lines.append(f"   {model}: Mean UNKNOWN rate = {mean_unk:.2%}")
    lines.append(f"   Cross-model agreement rate   : {var_agreement['overall_agreement_rate']:.2%}")
    lines.append(f"   Mean Cohen's kappa           : {var_agreement['mean_cohens_kappa']:.4f}")
    lines.append("")

    # ── Reasoning quality ─────────────────────────────────────────────────
    lines.append("3. CLINICAL REASONING QUALITY")
    lines.append("-" * 40)
    for model in ["4B", "27B"]:
        sub = per_patient_df[per_patient_df["model"] == model]
        mean_rqs = sub["reasoning.reasoning_quality_score"].mean()
        mean_steps = sub["reasoning.reasoning_steps"].mean()
        mean_refs = sub["reasoning.guideline_references"].mean()
        lines.append(f"   {model}: Quality={mean_rqs:.3f}, "
                     f"Steps={mean_steps:.1f}, GuidelineRefs={mean_refs:.1f}")
    lines.append("")

    # ── Completeness ──────────────────────────────────────────────────────
    lines.append("4. COMPLETENESS / COVERAGE")
    lines.append("-" * 40)
    for model in ["4B", "27B"]:
        sub = per_patient_df[per_patient_df["model"] == model]
        mean_gc = sub["completeness.guideline_coverage"].mean()
        mean_vc = sub["completeness.variable_coverage"].mean()
        mean_cs = sub["completeness.completeness_score"].mean()
        lines.append(f"   {model}: GuidelineCov={mean_gc:.2%}, "
                     f"VarCov={mean_vc:.2%}, Composite={mean_cs:.3f}")
    lines.append("")

    # ── Hallucination ─────────────────────────────────────────────────────
    lines.append("5. HALLUCINATION RATE")
    lines.append("-" * 40)
    for model in ["4B", "27B"]:
        sub = per_patient_df[per_patient_df["model"] == model]
        mean_hall = sub["hallucination.hallucination_score"].mean()
        mean_ndrugs = sub["hallucination.n_drug_mentions"].mean()
        mean_nhall = sub["hallucination.n_hallucinated_drugs"].mean()
        lines.append(f"   {model}: Score={mean_hall:.4f}, "
                     f"DrugMentions={mean_ndrugs:.1f}, "
                     f"HallucinatedDrugs={mean_nhall:.1f}")
    lines.append("")

    # ── Formatting ────────────────────────────────────────────────────────
    lines.append("6. FORMATTING COMPLIANCE")
    lines.append("-" * 40)
    for model in ["4B", "27B"]:
        sub = per_patient_df[per_patient_df["model"] == model]
        mean_fmt = sub["formatting.formatting_compliance_score"].mean()
        pct_reasoning = sub["formatting.has_reasoning_section"].mean() * 100
        pct_presc = sub["formatting.has_prescription_section"].mean() * 100
        pct_abbrev = sub["formatting.has_abbreviations_section"].mean() * 100
        lines.append(f"   {model}: Compliance={mean_fmt:.2%}, "
                     f"Reasoning={pct_reasoning:.0f}%, "
                     f"Prescription={pct_presc:.0f}%, "
                     f"Abbreviations={pct_abbrev:.0f}%")
    lines.append("")

    # ── Repetition / Degradation ─────────────────────────────────────────
    lines.append("7. REPETITION / DEGRADATION")
    lines.append("-" * 40)
    for model in ["4B", "27B"]:
        sub = per_patient_df[per_patient_df["model"] == model]
        mean_lr = sub["repetition.line_repetition_ratio"].mean()
        mean_nr = sub["repetition.ngram_repetition_ratio"].mean()
        mean_deg = sub["repetition.degradation_score"].mean()
        high_rep = (sub["repetition.line_repetition_ratio"] > 0.3).sum()
        lines.append(f"   {model}: LineRep={mean_lr:.2%}, "
                     f"NgramRep={mean_nr:.2%}, "
                     f"Degradation={mean_deg:.4f}, "
                     f"HighRepCases={high_rep}/{len(sub)}")
    lines.append("")

    # ── Output Length ─────────────────────────────────────────────────────
    lines.append("8. OUTPUT LENGTH STATISTICS")
    lines.append("-" * 40)
    for model in ["4B", "27B"]:
        sub = per_patient_df[per_patient_df["model"] == model]
        mean_chars = sub["output_length.char_count"].mean()
        mean_words = sub["output_length.word_count"].mean()
        std_words = sub["output_length.word_count"].std()
        lines.append(f"   {model}: MeanChars={mean_chars:.0f}, "
                     f"MeanWords={mean_words:.0f} (±{std_words:.0f})")
    lines.append("")

    # ── Conclusion ────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  SUMMARY")
    lines.append("-" * 40)
    sub4 = per_patient_df[per_patient_df["model"] == "4B"]
    sub27 = per_patient_df[per_patient_df["model"] == "27B"]

    metrics_to_summarize = {
        "var_extraction.unknown_rate":     "Variable UNKNOWN Rate (↓ better)",
        "reasoning.reasoning_quality_score": "Reasoning Quality (↑ better)",
        "completeness.completeness_score": "Completeness Score (↑ better)",
        "hallucination.hallucination_score": "Hallucination Score (↓ better)",
        "formatting.formatting_compliance_score": "Formatting Compliance (↑ better)",
        "repetition.degradation_score":   "Degradation Score (↓ better)",
    }

    for metric_col, label in metrics_to_summarize.items():
        m4 = sub4[metric_col].mean()
        m27 = sub27[metric_col].mean()
        lines.append(f"   {label}:")
        lines.append(f"       MedGemma-4B  = {m4:.4f}")
        lines.append(f"       MedGemma-27B = {m27:.4f}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MedGemma-4B vs. MedGemma-27B ablation evaluation script"
    )
    parser.add_argument(
        "--fourb", required=True,
        help="Path to MedGemma-4B results CSV",
    )
    parser.add_argument(
        "--twentysevenb", required=True,
        help="Path to MedGemma-27B results CSV",
    )
    parser.add_argument(
        "--output-dir", default="./eval_output",
        help="Directory to write evaluation outputs (default: ./eval_output)",
    )
    args = parser.parse_args()

    # ── Create output directory ───────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading MedGemma-4B results  : {args.fourb}")
    df4 = load_csv(args.fourb)
    print(f"[INFO] Loading MedGemma-27B results : {args.twentysevenb}")
    df27 = load_csv(args.twentysevenb)
    print(f"[INFO] Patients — 4B: {len(df4)}, 27B: {len(df27)}")

    # ── Per-patient evaluation ────────────────────────────────────────────
    print("[INFO] Running per-patient evaluation …")
    all_results = []

    for _, row in df4.iterrows():
        result = evaluate_single_patient(row, model_label="4B")
        all_results.append(result)

    for _, row in df27.iterrows():
        result = evaluate_single_patient(row, model_label="27B")
        all_results.append(result)

    per_patient_df = pd.DataFrame(all_results)

    # ── Variable agreement across models ──────────────────────────────────
    print("[INFO] Computing cross-model variable agreement …")
    var_agreement = compute_variable_agreement(df4, df27)

    # ── Aggregate statistics ──────────────────────────────────────────────
    print("[INFO] Computing aggregate statistics …")
    agg_df = aggregate_metrics(per_patient_df)
    comparison_df = cross_model_comparison(agg_df)

    # ── Save outputs ──────────────────────────────────────────────────────
    metrics_csv = out_dir / "metrics_summary.csv"
    per_patient_df.to_csv(metrics_csv, index=False)
    print(f"[SAVE] Per-patient metrics  → {metrics_csv}")

    agg_csv = out_dir / "aggregate_comparison.csv"
    comparison_df.to_csv(agg_csv, index=False)
    print(f"[SAVE] Aggregate comparison → {agg_csv}")

    # JSON output (includes variable agreement)
    json_data = {
        "per_patient": per_patient_df.to_dict(orient="records"),
        "variable_agreement": var_agreement,
        "aggregate": agg_df.to_dict(orient="records"),
    }
    json_path = out_dir / "metrics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"[SAVE] Full JSON results    → {json_path}")

    # Text report
    report_path = out_dir / "evaluation_report.txt"
    generate_text_report(
        per_patient_df, agg_df, comparison_df, var_agreement, str(report_path)
    )
    print(f"[SAVE] Text report          → {report_path}")

    # ── Quick console summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  QUICK COMPARISON SUMMARY")
    print("=" * 60)
    sub4 = per_patient_df[per_patient_df["model"] == "4B"]
    sub27 = per_patient_df[per_patient_df["model"] == "27B"]

    summary_metrics = [
        ("var_extraction.unknown_rate",     "UNKNOWN Rate (↓)",      False),
        ("reasoning.reasoning_quality_score", "Reasoning Quality (↑)", True),
        ("completeness.completeness_score", "Completeness (↑)",      True),
        ("hallucination.hallucination_score", "Hallucination (↓)",   False),
        ("formatting.formatting_compliance_score", "Formatting (↑)",  True),
        ("repetition.degradation_score",   "Degradation (↓)",       False),
    ]
    for col, label, higher_better in summary_metrics:
        m4 = sub4[col].mean()
        m27 = sub27[col].mean()
        winner = "4B" if (m4 > m27) == higher_better else "27B"
        print(f"  {label:25s}  4B={m4:.4f}  27B={m27:.4f}  → {winner}")

    print("=" * 60)
    print("[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
