import os
import glob
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

# --- 1. CONFIGURATION ---
# login(token="hf_YOUR_TOKEN_HERE") # Uncomment and add your token if needed

INPUT_DIR = "disease_algorithms_db"
OUTPUT_DIR = "clinical_checklists_db"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. LOAD MEDGEMMA 27B (Full Precision) ---
print("Loading MedGemma 27B in native bfloat16 (No Quantization)...")
model_id = "google/medgemma-27b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# device_map="auto" will gracefully split the ~54GB across your two H100s
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000, 
    do_sample=False, # Greedy decoding for maximum logical consistency
    return_full_text=False
)
print("✅ 27B Model loaded successfully!\n")

# --- 3. THE PROCESSING LOOP ---
txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
print(f"Found {len(txt_files)} disease algorithms to process.\n")

for file_path in txt_files:
    filename = os.path.basename(file_path)
    disease_name = filename.replace(".txt", "").replace("_", " ").title()
    
    # Resume Check
    out_file_path = os.path.join(OUTPUT_DIR, f"{disease_name.replace(' ', '_').lower()}_questions.json")
    if os.path.exists(out_file_path):
        print(f"⏭️ Skipping {disease_name} - Checklist already exists.")
        continue
        
    print(f"⚙️ Agent 2 is generating the clinical checklist for: {disease_name}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        algorithm_text = f.read()

    # --- 4. THE HIGH-PERFORMANCE PROMPT ---
    # --- 4. THE HIGH-PERFORMANCE PROMPT ---
    prompt_agent_2 = f"""You are the Chief Clinical Information Officer designing an intake form for an electronic health record (EHR) system.

Your task is to read the provided medical treatment algorithm and extract the absolute minimum number of questions a doctor must answer to navigate the logic tree.

TREATMENT ALGORITHM:
{algorithm_text}

CRITICAL RULES FOR QUESTION GENERATION:
1. VARIABLE AGGREGATION: NEVER ask separate Yes/No questions for continuous variables like age, weight, or duration. 
   - BAD: "Is the patient 1-4 years old?"
   - GOOD: "What is the patient's exact age in years?"
2. CATEGORICAL CONSOLIDATION: If the algorithm branches based on different test results, combine them.
   - BAD: "Is it P. vivax?" / "Is it P. falciparum?"
   - GOOD: "What is the specific species identified in the test result?"
3. AVOID OBVIOUS QUESTIONS: Do not ask if the patient has the disease. Assume the diagnosis is already suspected or confirmed.
4. CONTRAINDICATIONS: You MUST ask explicit Yes/No questions about any contraindications mentioned (e.g., Pregnancy, allergies, G6PD deficiency).

OUTPUT FORMAT:
Output ONLY a valid JSON object containing a single key named "questions", which holds an array of strings. Do not include markdown formatting, explanations, or introductory text.
Example:
{{
  "questions": [
    "What is the patient's exact age in years?",
    "Is the patient pregnant? (Yes/No)",
    "What is the result of the diagnostic test?"
  ]
}}
"""

    try:
        # Run Inference
        messages = [{"role": "user", "content": prompt_agent_2}]
        output = llm_pipe(messages)
        
        raw_text = output[0]["generated_text"].strip()
        
        # Clean the output (LLMs love to add ```json at the start)
        raw_text = re.sub(r'^```json\s*', '', raw_text)
        raw_text = re.sub(r'\s*```$', '', raw_text)
        
        # Validate JSON
        json_data = json.loads(raw_text)
        
        # Extract the list of questions from the object
        checklist = json_data.get("questions", [])
        
        # Save File
        with open(out_file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4) # Saves the full { "questions": [...] } object
            
        print(f"✅ Saved {len(checklist)} questions to: {out_file_path}\n")   
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON for {disease_name}. The LLM output was malformed.")
        # Save raw output for debugging
        error_path = os.path.join(OUTPUT_DIR, f"ERROR_{disease_name.replace(' ', '_').lower()}.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
    except Exception as e:
        print(f"❌ System Error processing {disease_name}: {e}\n")

print("🎉 Online Phase: Checklist Generation Complete!")