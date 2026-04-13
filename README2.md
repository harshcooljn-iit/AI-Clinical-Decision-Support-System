# 🩺 AI Clinical Decision Support System

<p align="center">
  <b>An AI-powered clinical reasoning engine for real-time diagnosis and evidence-based treatment planning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/>
  <img src="https://img.shields.io/badge/Backend-FastAPI-green"/>
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red"/>
  <img src="https://img.shields.io/badge/LLM-MedGemma--27B-purple"/>
  <img src="https://img.shields.io/badge/Status-Research--Prototype-orange"/>
</p>

---

## 📌 Overview

This project is an **end-to-end AI clinical decision support system** that assists physicians in:

- Diagnosing conditions from unstructured EHRs
- Mapping cases to standardized clinical guidelines
- Generating **safe, explainable treatment plans**

It transforms **300+ static clinical guidelines (AIIMS Rishikesh)** into an **interactive diagnostic engine** powered by large language models.

> ⚡ **Key Idea:** Convert static medical knowledge → structured logic → real-time clinical reasoning

🔗 [AIIMS Clinical Guidelines](https://aiimsrishikesh.edu.in/documents/standard-treatment-guidelines.pdf)

---

## 🎥 Demo

<p align="center">
  <a href="https://youtu.be/13hyQv1MVS8">
    <img src="https://img.youtube.com/vi/13hyQv1MVS8/maxresdefault.jpg" width="800"/>
  </a>
</p>

---

## ✨ Key Features

### 🔍 Semantic Clinical Retrieval

- Uses **PubMedBERT-based embeddings** (`S-PubMedBert-MS-MARCO`)
- Matches unstructured EHRs to relevant guidelines
- Provides **AI-generated justification** for retrieval

### 🧠 Multi-Disease Reasoning

- Handles **comorbidities**
- Merges multiple treatment algorithms
- Detects **drug conflicts & contraindications**

### 👨‍⚕️ Human-in-the-Loop (HITL)

- Physician validates extracted variables before reasoning
- Ensures **clinical safety and trust**

### ⚡ Streaming Reasoning Engine

- Real-time generation of:
  - Structured prescriptions
  - Step-by-step reasoning

- Improves interpretability and auditability

---

## 🏗️ System Architecture

> Modular pipeline designed for future transition into **fully autonomous agent systems**

```
                ┌────────────────────────────┐
                │   AIIMS Guidelines (PDF)   │
                └────────────┬───────────────┘
                             │
                (Offline Processing Pipeline)
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
 Document Parsing     Logic Translation     Checklist Generation
 (llama-parse)        (Agent 1 - LLM)       (Agent 2 - LLM)
        │                    │                    │
        └──────────────→ Structured Knowledge Base
                             │
                             ▼
                (Online Inference Pipeline)
                             │
        ┌──────────────┬──────────────┬──────────────┐
        │              │              │              │
     Search         Extract        Verify        Recommend
   (Embedding)     (Agent 3)        (HITL)       (Agent 4)
        │              │              │              │
        └──────────────┴──────→ Final Prescription
```

---

## ⚙️ Pipeline Breakdown

### 🧱 Phase 1: Offline Knowledge Construction

- **Document Parsing:** External `llama-parse` service converts PDF → Markdown
- **Segmentation:** Regex + fuzzy matching splits into disease-specific files
- **Agent 1:** Converts text → IF-ELSE clinical logic
- **Agent 2:** Generates structured clinical checklists (JSON)

---

### ⚡ Phase 2: Real-Time Inference

1. **Search:** Retrieve top-k relevant diseases from EHR
2. **Extract (Agent 3):** Populate structured variables
3. **Human Verification:** Physician validates inputs
4. **Recommend (Agent 4):** Generate safe, explainable prescriptions

---

## 📂 Repository Structure

```
offline phase/
├── data/
│   ├── clinical_checklists_db/
│   ├── disease_algorithms_db/
│   ├── disease_markdown_files/
│   └── raw_guidelines/
├── scripts/
│   ├── fuzzy_extract.py
│   ├── generate_questions.py
│   └── process_guidelines.py

online phase/
├── backend/
│   └── backend.py
├── frontend/
│   └── frontend.py
├── benchmark data/
│   ├── mimic3.csv
│   └── benchmark.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- GPU with **~60GB VRAM** (A100 recommended)

> 💡 Use `medgemma-4b-it` for lower-resource environments

---

### Installation

```bash
git clone https://github.com/harshcooljn-iit/AI-Clinical-Decision-Support-System.git
cd ai-clinical-decision-support-system
pip install -r requirements.txt
```

---

### Run Backend

```bash
cd "online phase/backend"
uvicorn backend:app --host 0.0.0.0 --port 8000
```

---

### Run Frontend

```bash
cd "online phase/frontend"
streamlit run frontend.py
```

---

## 📊 Benchmarking

- Dataset: **MIMIC-III**
- Evaluates:
  - Hallucination rate
  - Formatting correctness
  - Clinical safety

```bash
python benchmark.py
```

---

## 🔮 Future Work

### 🤖 Agentic AI Systems

- Transition to frameworks like **LangGraph / CrewAI**
- Introduce:
  - Self-correcting agents
  - Tool-using agents (PubMed, UpToDate)
  - Multi-agent collaboration

### 🧠 Prompt Optimization

- Replace static prompts with **DSPy**
- Goal: **maximize accuracy via programmatic optimization**

### 📚 Expanded Medical Knowledge

- Integrate:
  - Mayo Clinic
  - WHO
  - Additional clinical datasets

### ⚡ Dynamic Model Routing

- Small models → extraction
- Large models → reasoning
- Optimize **latency vs accuracy tradeoff**

---

## ⚠️ Disclaimer

> This system is a **research prototype** and is **not intended for clinical deployment**.
> All outputs must be reviewed by qualified medical professionals.

---

## 🤝 Contributing

Contributions, ideas, and discussions are welcome!
Feel free to open issues or submit pull requests.

---

## 📜 License

This project is intended for **research and educational purposes**.
