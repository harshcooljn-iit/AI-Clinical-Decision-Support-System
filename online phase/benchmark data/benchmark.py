import pandas as pd
import requests
import time

API_URL = "http://localhost:8000"
NUM_SAMPLES = 50
INPUT_CSV = "mimic3.csv"
OUTPUT_CSV = "medgemma_27b_results.csv" # Change this to 4b when testing the smaller model!

print(f"Loading {INPUT_CSV} and sampling {NUM_SAMPLES} records...")
# Adjust 'TEXT' if your MIMIC column has a different name (e.g., 'text' or 'discharge_summary')
df = pd.read_csv(INPUT_CSV)
sample_df = df.sample(n=NUM_SAMPLES, random_state=42) 

results = []

for index, row in sample_df.iterrows():
    ehr_text = str(row['text'])[:2000] # Truncate to avoid massive token limits just in case
    print(f"\n--- Processing Patient Index {index} ---")
    
    # 1. Search Guidelines
    search_res = requests.post(f"{API_URL}/search", json={"ehr_text": ehr_text})
    if search_res.status_code != 200:
        print("Search failed, skipping...")
        continue
        
    top_matches = search_res.json()["top_matches"]
    # Automatically select the top 2 matches for comorbidity testing
    selected_diseases = [top_matches[0]['name'], top_matches[1]['name']] if len(top_matches) > 1 else [top_matches[0]['name']]
    print(f"Selected Guidelines: {selected_diseases}")
    
    # 2. Extract Variables
    extract_res = requests.post(f"{API_URL}/extract", json={"ehr_text": ehr_text, "disease_names": selected_diseases})
    if extract_res.status_code != 200:
        # print("Extraction failed, skipping...", extract_res.json())
        # continue
        extracted_answers = str(extract_res.json())
    else:  
        extracted_answers = extract_res.json()["extracted_answers"]
    
    # 3. Recommend Treatment (With auto_assume=True)
    recommend_res = requests.post(
        f"{API_URL}/recommend", 
        json={
            "disease_names": selected_diseases, 
            "extracted_answers": extracted_answers,
            "auto_assume": True # Triggers our backend modification!
        },
        stream=True
    )
    
    # Accumulate the streaming response into a single string
    final_prescription = ""
    for chunk in recommend_res.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            final_prescription += chunk
            
    # Save the run
    results.append({
        "Patient_ID": index,
        "EHR_Snippet": ehr_text[:500] + "...", 
        "Selected_Guidelines": ", ".join(selected_diseases),
        "Extracted_Variables": str(extracted_answers),
        "Final_Output": final_prescription
    })
    
    time.sleep(1) # Brief pause to let GPU memory clear

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")

print("\nBenchmarking complete! Results saved to", OUTPUT_CSV)
