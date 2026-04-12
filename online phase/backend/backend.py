import os
import glob
import json
import re
import torch
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, util

# Isolate to a single GPU (which becomes logical device 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

app = FastAPI(title="AI Clinical Decision Support System")

print("Loading Models onto GPU...")
embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
model_id = "google/medgemma-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def _build_max_memory():
    """Build a conservative per-device memory map from currently free VRAM."""
    max_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_bytes, _ = torch.cuda.mem_get_info(i)
            # Keep a safety headroom to avoid fragmentation/OOM during generation.
            free_gib = int(free_bytes / (1024 ** 3))
            budget_gib = max(4, free_gib - 4)
            max_memory[i] = f"{budget_gib}GiB"

    # Allow offloading any overflow to host RAM.
    max_memory["cpu"] = "256GiB"
    return max_memory


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map={"":0},
    # max_memory=_build_max_memory(),
    # offload_folder="./model_offload",
    # low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)

# Index Database
db_files = glob.glob("offline phase/data/disease_algorithms_db/*.txt")
db_disease_names = [os.path.basename(f).replace(".txt", "").replace("_", " ").title() for f in db_files]
db_embeddings = embedder.encode(db_disease_names, convert_to_tensor=True) if db_disease_names else None

# --- PYDANTIC DATA MODELS ---
class SearchRequest(BaseModel):
    ehr_text: str

class ExtractRequest(BaseModel):
    ehr_text: str
    disease_names: list[str] # Updated to accept a list of multiple diseases

class RecommendRequest(BaseModel):
    disease_names: list[str]
    extracted_answers: dict
    auto_assume: bool = False  # NEW FLAG FOR AUTOMATION (This is for benchmarking the auto-assumption feature. The LLM will be instructed to make safe assumptions for any "UNKNOWN" variables when this flag is True.)

# --- API ENDPOINTS ---

@app.post("/search")
def search_guidelines(req: SearchRequest):
    if db_embeddings is None:
        raise HTTPException(status_code=500, detail="Database is empty.")
    
    # 1. Perform Semantic Search
    query_embedding = embedder.encode(req.ehr_text, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, db_embeddings, top_k=5)[0]
    
    results = []
    
    # 2. Loop through the top hits to generate rationales (Agent 1.5)
    for hit in hits:
        disease_name = db_disease_names[hit['corpus_id']]
        # Convert cosine similarity tensor to a readable percentage (e.g., 0.854 -> 85)
        score_pct = int(round(hit['score'] * 100))
        
        # Micro-Prompt for Agent 1.5
        prompt = f"""You are a medical justification AI.
EHR: {req.ehr_text}
MATCHED DISEASE: {disease_name}
INSTRUCTION: Explain exactly why this disease matches the EHR. Keep it under 15 words. Focus on matching symptoms."""

        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
        
        # Fast generation lock
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        rationale = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Clean up memory after each micro-call
        torch.cuda.empty_cache()
        
        results.append({
            "name": disease_name,
            "score": score_pct,
            "rationale": rationale
        })
        
    return {"top_matches": results}

@app.post("/extract")
def extract_variables(req: ExtractRequest):
    all_questions = {}
    
    # 1. Merge checklists from all selected diseases
    for disease in req.disease_names:
        disease_filename = disease.replace(" ", "_").lower()
        checklist_path = f"offline phase/data/clinical_checklists_db/{disease_filename}_questions.json"
        
        if os.path.exists(checklist_path):
            with open(checklist_path, "r") as f:
                questions_data = json.load(f)
                questions = questions_data.get("questions", questions_data) if isinstance(questions_data, dict) else questions_data
                # Use dict keys to automatically deduplicate overlapping questions
                for q in questions:
                    all_questions[q] = "UNKNOWN"
        
    prompt = f"""You are an expert Clinical Data Extractor.
Read the Patient Notes and extract the answers to the provided Combined Medical Checklist.

PATIENT NOTES: 
{req.ehr_text}

COMBINED CHECKLIST: 
{json.dumps(list(all_questions.keys()), indent=2)}

INSTRUCTIONS:
1. Answer ONLY based on the text. If missing, output "UNKNOWN".
2. Keep answers extremely short (e.g., "Yes", "28", "P. falciparum").
3. Output a strict JSON object where the keys are the exact questions."""

    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
    
    # Prevent PyTorch from calculating memory-heavy training gradients
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=800, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
    raw_json = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Clean up GPU memory so the next API call has room to breathe
    torch.cuda.empty_cache()
    
    match = re.search(r'\{.*\}', raw_json, re.DOTALL)
    clean_json_str = match.group(0) if match else raw_json.strip()
    
    try:
        return {"extracted_answers": json.loads(clean_json_str)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed to output JSON: {raw_json}")

@app.post("/recommend")
def recommend_treatment(req: RecommendRequest):
    combined_algorithms = ""
    
    # 1. Concatenate all relevant treatment algorithms
    for disease in req.disease_names:
        disease_filename = disease.replace(" ", "_").lower()
        algorithm_path = f"offline phase/data/disease_algorithms_db/{disease_filename}.txt"
        if os.path.exists(algorithm_path):
            with open(algorithm_path, "r") as f:
                combined_algorithms += f"\n--- GUIDELINE FOR {disease.upper()} ---\n"
                combined_algorithms += f.read() + "\n"
        
    # Inject the automation instruction if the flag is True
    assumption_instruction = ""
    if req.auto_assume:
        assumption_instruction = "\n8. AUTOMATED MODE: Some patient variables are 'UNKNOWN'. You MUST assume safe, typical clinical values for these missing variables to generate a complete prescription. Explicitly list your assumptions under the Clinical Reasoning section."

    prompt = f"""You are an expert Medical Officer. 
Determine the final treatment plan using ONLY the Combined Algorithms and the Patient Variables.

COMBINED ALGORITHMS: 
{combined_algorithms}

VARIABLES: 
{json.dumps(req.extracted_answers, indent=2)}

INSTRUCTIONS:
1. Evaluate all guidelines simultaneously. First, think step-by-step about the patient's condition.
2. CONFLICT RESOLUTION: If Guideline A suggests a drug that is contraindicated by Guideline B (or the patient's variables), prioritize safety and suggest an alternative.
3. Then, provide the final prescription separated under the exact heading: "### 💊 FINAL PRESCRIPTION".
4. Stop generating immediately after the prescription is complete.
5. Do not repeat yourself.
6. At the end of the Final Prescription mention the full forms of any abbreviations used in the prescription.
7. FORMATTING RULE: The FINAL PRESCRIPTION section MUST be formatted as a clear, bulleted list.{assumption_instruction}"""
    
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
    
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    terminators = [t for t in terminators if t is not None]
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs, streamer=streamer, max_new_tokens=4096, do_sample=False, 
        eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id
    )
    
    # Wrap the generation thread in an inference_mode lock
    def generate_with_no_grad():
        with torch.inference_mode():
            model.generate(**generation_kwargs)
        torch.cuda.empty_cache() # Clean up when done
            
    thread = Thread(target=generate_with_no_grad)
    thread.start()
    
    def token_generator():
        for new_text in streamer:
            yield new_text.replace("<unused94>thought", "\n### 🧠 Clinical Reasoning\n")
            
    return StreamingResponse(token_generator(), media_type="text/plain")

# uvicorn backend_new:app --host 0.0.0.0 --port 8000