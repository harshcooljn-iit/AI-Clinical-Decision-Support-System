import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login

# --- 1. AUTHENTICATION & CONFIGURATION ---
# Replace with your actual Hugging Face token
login(token="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")

INPUT_DIR = "/home/anurag/nas_anurag/harsh/btp2_new/disease_markdown_files"
OUTPUT_DIR = "/home/anurag/nas_anurag/harsh/btp2_new/disease_algorithms_db"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. LOAD THE MODEL (MedGemma 27B) ---
print("Loading MedGemma 27B Model...")
model_id = "google/medgemma-27b-it"

# If you are on Kaggle (T4 GPUs), keep this 4-bit config. 
# If you are on your H100, you can delete this config and just use torch_dtype=torch.bfloat16 for maximum speed!
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,       
    bnb_4bit_quant_type="nf4"             
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" 
)

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096, # Massively increased to handle long tables
    do_sample=False,
    return_full_text=False 
)
print("✅ Model loaded successfully!\n")

# --- 3. THE PROCESSING LOOP ---
md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
print(f"Found {len(md_files)} disease files to process.\n")
for file_path in md_files:
    filename = os.path.basename(file_path)
    disease_name = filename.replace(".md", "").replace("_", " ").title()
    # --- RESUME CHECK: Skip if already processed ---
    out_file_path = os.path.join(OUTPUT_DIR, f"{disease_name.replace(' ', '_').lower()}.txt")
    if os.path.exists(out_file_path):
        print(f"⏭️ Skipping {disease_name} - Already processed.")
        continue
    
    print(f"⚙️ Agent 1 is writing the IF/ELSE algorithm for: {disease_name}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # --- 4. THE PROMPT ---
    prompt_agent_1 = f"""You are a master clinical decision support engineer and data extractor.

TASK: Convert the following standard treatment guideline into a highly structured, text-based algorithm using IF/THEN/ELSE logic.

FORMAT REQUIREMENTS:
1. DISEASE: {disease_name}
2. DIAGNOSTICS: Briefly list how it is diagnosed.
3. TREATMENT ALGORITHM: Translate the flowcharts and text into clear IF/ELSE blocks.
4. TABULAR DOSAGES (CRITICAL): If you encounter an HTML or Markdown table containing age, weight, or severity-based dosages, you MUST NOT summarize it. You must create a specific, nested IF/ELSE condition for EVERY SINGLE ROW in the table. 
5. CONTRAINDICATIONS: Use IF/THEN statements to list who CANNOT receive certain drugs.
6. ABBREVIATIONS: List unique medical abbreviations found in the text alongside their full meaning (e.g., HIV: Human Immunodeficiency Virus). DO NOT list standard measurements (e.g., mg, kg, ml, dl). LIMIT listing to a maximum of 10 items. DO NOT REPEAT items.

CRITICAL RULES:
- NEVER summarize tables. Extract the exact Day 1, Day 2, etc., dosages for every single age/weight bracket mentioned.
- TABULAR DATA CONVERSION: If the source contains a table, you MUST convert every single row into a text-based IF/ELSE block.
- Be highly specific with drug names, mg values, and pill fractions (e.g., "1/2 tab").
- Actively scan the bottom of tables and flowcharts for abbreviation legends.
- Do not repeat abbreviations.
- Do not output anything other than the structured text.
- Do not make mistakes

RAW GUIDELINE TEXT:
{md_content}
"""

    try:
        # --- 5. RUN INFERENCE ---
        messages = [{"role": "user", "content": prompt_agent_1}]
        output = llm_pipe(messages)
        
        algorithm_text = str(output[0]["generated_text"]).strip()
        
        # --- 6. SAVE FILE ---
        with open(out_file_path, "w", encoding="utf-8") as f:
            f.write(algorithm_text)
            
        print(f"✅ Successfully saved: {out_file_path}\n")
        
    except Exception as e:
        print(f"❌ Error processing {disease_name}: {e}\n")

print("🎉 Offline Processing Complete!")