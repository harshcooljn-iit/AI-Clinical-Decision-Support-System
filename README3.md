<div align="center">

<br/>

<img src="https://img.shields.io/badge/MedGemma-27B-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="MedGemma 27B"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/AIIMS_Rishikesh-Guidelines-0066CC?style=for-the-badge" alt="AIIMS Guidelines"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>

<br/><br/>

# 🩺 AI Clinical Decision Support System

**An end-to-end autonomous diagnostic pipeline that translates 300+ static clinical guidelines into a real-time, physician-in-the-loop prescription engine.**

_Powered by MedGemma-27B · Built on AIIMS Rishikesh Standard Treatment Guidelines_

<br/>

[![▶ Watch Demo](https://img.shields.io/badge/▶%20Watch%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/13hyQv1MVS8)
&nbsp;
[![📄 AIIMS Guidelines](https://img.shields.io/badge/📄%20AIIMS%20Guidelines-PDF-blue?style=for-the-badge)](https://aiimsrishikesh.edu.in/documents/standard-treatment-guidelines.pdf)

<br/>

[![Demo Preview](https://img.youtube.com/vi/13hyQv1MVS8/maxresdefault.jpg)](https://youtu.be/13hyQv1MVS8)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
  - [Phase 1: Offline Database Ingestion](#phase-1-offline-database-ingestion)
  - [Phase 2: Online Real-Time Inference](#phase-2-online-real-time-inference)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Benchmarking](#-benchmarking)
- [Future Work](#-future-work)

---

## 🔬 Overview

This system bridges the gap between static clinical documentation and dynamic, context-aware medical decision-making. It ingests the **430+ page AIIMS Rishikesh Standard Treatment Guidelines** and converts them into an interactive diagnostic engine that a physician can query using a raw patient EHR.

The architecture is deliberately **decoupled**: a heavy-inference FastAPI backend handles all LLM computation on GPU, while a lightweight Streamlit frontend provides a clean, responsive physician interface with a mandatory **Human-in-the-Loop (HITL)** verification gate before any prescription is generated.

> **Design Philosophy:** Each component is labeled as an _Agent_ (1–4). This isn't merely cosmetic — it reflects a modular design where every stage is a replaceable, upgradeable unit. Swap out an LLM call for a fine-tuned specialist model or a fully autonomous agent without touching the surrounding pipeline.

---

## ✨ Key Features

| Feature                               | Description                                                                                                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 🔍 **Semantic EHR Search**            | Uses `S-PubMedBert-MS-MARCO` medical embeddings to match unstructured patient records to the top 5 relevant clinical guidelines, with AI-generated rationale for every match.  |
| 🔀 **Comorbidity Handling**           | Merges multiple clinical checklists simultaneously and evaluates overlapping treatment algorithms to detect drug contraindications across conditions.                          |
| 🧑‍⚕️ **Human-in-the-Loop Verification** | A mandatory physician review stage displays all extracted clinical variables. Doctors can verify, edit, or fill missing fields before the system generates any recommendation. |
| ⚡ **Streaming Clinical Reasoning**   | Agent 4 streams a structured prescription and step-by-step clinical reasoning directly to the UI in real time — no waiting for a full response.                                |
| 🧬 **Pre-computed Checklists**        | All 300+ disease checklists are pre-generated offline (Agent 2), eliminating bottlenecks during live inference.                                                                |
| 📐 **Strict Schema Validation**       | Agent 3 uses dynamic **Pydantic** schemas to validate extracted clinical variables before they ever reach the recommendation engine.                                           |

---

## 🏗️ System Architecture

The pipeline runs in two distinct phases: an **offline ingestion phase** (run once) and an **online inference phase** (run per patient).

```
╔══════════════════════════════════════════════════════════════════╗
║                   PHASE 1 · OFFLINE INGESTION                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  AIIMS PDF (430+ pages)                                          ║
║       │                                                          ║
║       ▼                                                          ║
║  llama-parse  ──────────────►  Markdown (preserves tables)       ║
║       │                                                          ║
║       ▼                                                          ║
║  fuzzy_extract.py  ─────────►  300+ Disease Markdown Files       ║
║       │                                                          ║
║       ├──► Agent 1 (MedGemma-27B)  ──►  IF-ELSE Logic .txt Files ║
║       │                                                          ║
║       └──► Agent 2 (MedGemma-27B)  ──►  Clinical Checklists JSON ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                  PHASE 2 · ONLINE INFERENCE                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Physician inputs EHR text                                       ║
║       │                                                          ║
║       ▼                                                          ║
║  Dense Semantic Search  ───────►  Top 5 Relevant Guidelines      ║
║       │                                                          ║
║       ▼                                                          ║
║  Agent 3 ──► Fetch checklists + Extract variables from EHR       ║
║       │      (Pydantic schema validation)                        ║
║       ▼                                                          ║
║  🧑‍⚕️ PHYSICIAN VERIFICATION  ◄── Edit / Confirm / Fill gaps       ║
║       │                                                          ║
║       ▼                                                          ║
║  Agent 4 ──► Evaluate IF-ELSE logic ──► Stream Final Prescription║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Phase 1: Offline Database Ingestion

> _Run once to build the knowledge base. No GPU required for fuzzy extraction._

1. **Document Parsing** — The AIIMS PDF is converted to Markdown via `llama-parse` (external service). This critical step preserves structured content like dosage tables and diagnostic flowcharts that plain-text extraction destroys.

2. **Fuzzy Extraction** — `fuzzy_extract.py` uses regex and fuzzy string matching to segment the master Markdown document into **300+ individual disease files**, one per condition.

3. **Logic Translation · Agent 1** — MedGemma-27B converts each disease Markdown file into a strict, machine-evaluable **IF-ELSE logic text file**. This is the rule engine that Agent 4 will query at runtime.

4. **Checklist Pre-generation · Agent 2** — MedGemma-27B pre-generates the required **clinical question checklists (JSON)** for every disease. Pre-computing these offline is what makes online inference fast.

### Phase 2: Online Real-Time Inference

> _Called live for every patient. Runs on GPU._

1. **Semantic Search** — The physician pastes a patient's EHR. Dense vector search returns the **top 5 most relevant guidelines** from the database.

2. **Extraction · Agent 3** — Fetches pre-generated checklists for the selected diseases and extracts answers directly from the EHR text using **dynamic Pydantic schema validation**.

3. **Human Verification** — The Streamlit UI presents all extracted variables to the physician. This is the critical safety gate: doctors review, correct, or fill any missing information before proceeding.

4. **Recommendation · Agent 4** — Evaluates the confirmed variables against the combined IF-ELSE logic to **stream a final, safe prescription** alongside step-by-step clinical reasoning.

---

## 📂 Repository Structure

```
ai-clinical-decision-support-system/
│
├── offline phase/
│   ├── data/
│   │   ├── clinical_checklists_db/      # Pre-computed JSON question files (one per disease)
│   │   ├── disease_algorithms_db/       # IF-ELSE logic text files (Agent 1 output)
│   │   ├── disease_markdown_files/      # Segmented Markdown per disease
│   │   └── standard-treatment-guidelines.pdf  # Source AIIMS document
│   │
│   └── scripts/
│       ├── fuzzy_extract.py             # Document segmentation into disease files
│       ├── generate_questions.py        # Clinical checklist generation (Agent 2)
│       └── process_guidelines.py        # Markdown → IF-ELSE logic processing (Agent 1)
│
├── online phase/
│   ├── backend/
│   │   └── backend.py                   # FastAPI server — loads embeddings + MedGemma into VRAM
│   │
│   ├── benchmark data/
│   │   ├── mimic3.csv                   # MIMIC-III sample cases for evaluation
│   │   └── benchmark.py                 # Automated comparison: 27B vs 4B models
│   │
│   └── frontend/
│       └── frontend.py                  # Streamlit UI — HITL verification + streaming output
│
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation & Setup

### Prerequisites

| Requirement        | Details                                                   |
| ------------------ | --------------------------------------------------------- |
| **Python**         | 3.10 or higher                                            |
| **GPU VRAM**       | ~60 GB free (e.g., A100 or dual V100s) for MedGemma-27B   |
| **Lighter option** | Swap model ID to `google/medgemma-4b-it` for smaller GPUs |

> **Note:** The backend (GPU inference) and the frontend (Streamlit UI) are designed to run on **separate nodes**. The frontend communicates with the backend over HTTP — ideal for cloud GPU instances paired with a local client.

---

### Step 1 · Clone the Repository

```bash
git clone https://github.com/harshcooljn-iit/AI-Clinical-Decision-Support-System.git
cd ai-clinical-decision-support-system
```

### Step 2 · Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 · Start the Backend (GPU Node)

This loads the embedding model and MedGemma-27B weights into VRAM. Allow a few minutes for model loading on first run.

```bash
cd "online phase/backend"
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### Step 4 · Start the Frontend (Local / Client Node)

In a separate terminal, launch the Streamlit UI. By default it connects to `localhost:8000` — update the backend URL in `frontend.py` if running on a remote GPU node.

```bash
cd "online phase/frontend"
streamlit run frontend.py
```

> The UI will be accessible at **http://localhost:8501**

---

### (Optional) Re-run Offline Ingestion

To rebuild the disease database from scratch (e.g., after updating the guidelines PDF):

```bash
# 1. Segment the master document into per-disease Markdown files
python "offline phase/scripts/fuzzy_extract.py"

# 2. Generate IF-ELSE logic for each disease (Agent 1)
python "offline phase/scripts/process_guidelines.py"

# 3. Pre-compute clinical checklists (Agent 2)
python "offline phase/scripts/generate_questions.py"
```

---

## 📊 Benchmarking

The `online phase/benchmark data/` directory contains tools to evaluate system performance against real-world clinical data.

**Dataset:** [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) — a large, de-identified database of ICU patient records.

**Metrics evaluated:**

- 📋 **Formatting Adherence** — Does the output follow the structured prescription format?
- 🧠 **Hallucination Rate** — Does the model invent drugs, dosages, or conditions not in the guidelines?
- 🛡️ **Clinical Safety Score** — Are contraindications and drug interactions correctly flagged?

```bash
# Run the full benchmark comparison (27B vs 4B)
python "online phase/benchmark data/benchmark.py"
```

Output is saved as CSV files for downstream analysis and visualization.

---

## 🔮 Future Work

### Transition to True Multi-Agent Systems

The current architecture uses a single LLM across all agents. Future iterations will specialize each agent independently:

- **Agent 3 (Extraction)** → Fine-tuned clinical NER model, optimized purely for entity extraction from EHRs.
- **Agent 4 (Recommendation)** → A Reasoning Agent with access to live medical knowledge graphs (e.g., SNOMED CT, RxNorm).
- **Self-Critique Agent** → A dedicated "reviewer" that checks Agent 4's prescription against the source guidelines before the physician ever sees it.
- **Autonomous Tool Use** → Agents that proactively query external databases (PubMed, UpToDate) when internal AIIMS guidelines are insufficient for complex or rare cases.
- **Multi-Agent Collaboration** → Specialized departmental agents (e.g., a Cardiology Agent and a Pharmacology Agent) that debate and co-refine treatment plans for multi-morbidity patients.

### Prompt & Pipeline Optimization

- **DSPy Integration** — Transition Agent 4 from static f-string prompts to programmatic prompt compilation via [DSPy](https://github.com/stanfordnlp/dspy), mathematically optimizing for instruction-following and clinical accuracy.
- **Dynamic Model Routing** — A router that dispatches simple extraction tasks to fast small models (e.g., Llama-3-8B) while reserving complex reasoning for MedGemma-27B.

### Expanded Knowledge Base

- **Additional Guidelines** — Integrate WHO and Mayo Clinic protocols into the offline ingestion pipeline alongside AIIMS.
- **Multilingual Support** — Extend to regional-language guidelines for broader Indian hospital deployment.

---

<div align="center">

**Built with ❤️ for better clinical outcomes**

_If you find this useful, consider starring ⭐ the repository._

</div>
