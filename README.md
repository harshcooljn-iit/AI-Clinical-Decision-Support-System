---
# 🩺 AI Clinical Decision Support System

An end-to-end, multi-agent AI pipeline designed to assist physicians in diagnosing and formulating evidence-based treatment plans in real-time.

This system translates over 300 static clinical guidelines from AIIMS Rishikesh into an interactive, autonomous diagnostic engine. It features a decoupled architecture with a heavy-inference FastAPI backend (powered by MedGemma-27B) and a lightweight, Human-in-the-Loop Streamlit frontend.
---

## 🎥 Demo

<p align="center">
  <a href="https://youtu.be/13hyQv1MVS8">
    <img src="https://img.youtube.com/vi/13hyQv1MVS8/maxresdefault.jpg" width="800" alt="Demo Video"/>
  </a>
</p>

---

## ✨ Key Features

- **Explainable Semantic Search:** Uses specialized medical embeddings (`S-PubMedBert-MS-MARCO`) to match unstructured Electronic Health Records (EHR) to relevant guidelines, providing an AI-generated rationale for every match.
- **Comorbidity Handling:** Capable of merging multiple clinical checklists and evaluating overlapping treatment algorithms simultaneously to check for drug contraindications.
- **Human-in-the-Loop (HITL) Verification:** Pauses the AI pipeline mid-inference, allowing the physician to verify, edit, or fill in missing clinical variables extracted by the AI before generating a final prescription.
- **Streaming Clinical Reasoning:** Streams a structured, bulleted medical prescription alongside step-by-step clinical reasoning directly to the UI.

---

## 🏗️ System Architecture

The pipeline is divided into two distinct phases:

### Phase 1: Offline Database Ingestion

1. **Document Parsing:** Converts the 430+ page AIIMS Rishikesh PDF into Markdown using `llama-parse` to preserve dosage tables and flowcharts.
2. **Fuzzy Extraction:** A custom Python script (`fuzzy_extract.py`) uses regex and fuzzy string matching to segment the master document into 300+ individual disease Markdown files.
3. **Logic Translation (Agent 1):** MedGemma-27B converts the Markdown files into strict IF-ELSE logic text files.
4. **Pre-computing Checklists (Agent 2):** MedGemma-27B pre-generates the required clinical question checklists (JSON) for every disease to optimize online inference speed.

### Phase 2: Online Real-Time Inference

1. **Search:** The physician inputs patient EHR. The system performs a dense semantic search to return the top 5 relevant guidelines.
2. **Extract (Agent 3):** Fetches the pre-generated checklists for selected diseases and extracts answers directly from the EHR using dynamic Pydantic schema validation.
3. **Human Verification:** The Streamlit UI prompts the physician to confirm the extracted variables.
4. **Recommend (Agent 4):** Evaluates the confirmed variables against the combined IF-ELSE logic algorithms to stream a final, safe prescription.

---

## 📂 Repository Structure

Based on the project tree, the repository is organized as follows:

```text
├── offline phase/
│   ├── data/
│   │   ├── clinical_checklists_db/      # Pre-computed JSON question files
│   │   ├── disease_algorithms_db/       # Extracted IF-ELSE logic text files
│   │   ├── disease_markdown_files/      # Segmented markdown files per disease
│   │   └── standard-treatment...        # Original AIIMS guidelines
│   └── scripts/
│       ├── fuzzy_extract.py             # Script for document segmentation
│       ├── generate_questions.py        # Generates clinical checklist questions
│       └── process_guidelines.py        # Processes guideline markdown/logic assets
├── online phase/
│   ├── backend/
│   │   └── backend.py                   # FastAPI server (GPU Inference)
│   ├── benchmark data/
│   │   ├── mimic3.csv                   # Sample dataset for testing
│   │   └── benchmark.py                 # Automated testing script (27B vs 4B)
│   └── frontend/
│       └── frontend.py                  # Streamlit UI
├── requirements.txt                     # Python dependencies
└── .gitignore
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- **Hardware:** Running the MedGemma-27B model requires a GPU with approximately **~60GB of free VRAM** (e.g., A100 or dual V100s). For smaller GPUs, you can swap the model ID in `backend.py` to `google/medgemma-4b-it`.

### 1. Clone the Repository

```bash
git clone https://github.com/harshcooljn-iit/AI-Clinical-Decision-Support-System.git
cd ai-clinical-decision-support-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Backend Server (GPU Node)

Navigate to the backend directory and spin up the FastAPI server. This will load the embedding model and the MedGemma weights into VRAM.

```bash
cd "online phase/backend"
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### 4. Start the Frontend UI (Local/Client Node)

In a new terminal window, navigate to the frontend directory and launch Streamlit.

```bash
cd "online phase/frontend"
streamlit run frontend.py
```

_The UI will be accessible at `http://localhost:8501`._

---

## 📊 Benchmarking

The `online phase/benchmark data/` directory contains tools to evaluate the system against real-world data. We use the MIMIC-III dataset to systematically compare the formatting adherence, hallucination rates, and clinical safety of the 27B model versus the 4B model. Run `python benchmark.py` to generate the comparison CSVs.

---

## 🔮 Future Work

- **Prompt Optimization:** Transitioning Agent 4 from static f-strings to programmatic prompt compilation using the **DSPy** framework to mathematically maximize instruction-following and accuracy.
- **Expanded Database:** Integrating additional medical guidelines beyond AIIMS (e.g., Mayo Clinic, WHO) into the offline ingestion pipeline.
- **Model Routing:** Implementing a dynamic router that sends simple extraction tasks to smaller, faster models (e.g., Llama-3-8B) while reserving complex comorbidity reasoning for the 27B parameter model.
