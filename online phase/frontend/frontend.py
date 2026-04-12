import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Clinical Decision Support", layout="wide")

# --- INITIALIZE SESSION MEMORY ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'input'
if 'ehr_text' not in st.session_state:
    st.session_state.ehr_text = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_diseases' not in st.session_state:
    st.session_state.selected_diseases = [] # Now a list for multiple selections
if 'extracted_answers' not in st.session_state:
    st.session_state.extracted_answers = {}

st.title("🩺 AI Clinical Decision Support")

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# --- STAGE 1: Input ---
if st.session_state.stage == 'input':
    ehr_input = st.text_area("Paste patient history and symptoms here:", height=200)
    if st.button("Search Guidelines", type="primary"):
        with st.spinner("Searching database..."):
            response = requests.post(f"{API_URL}/search", json={"ehr_text": ehr_input})
            if response.status_code == 200:
                st.session_state.search_results = response.json()["top_matches"]
                st.session_state.ehr_text = ehr_input
                st.session_state.stage = 'select'
                st.rerun()
            else:
                st.error(f"Backend Error: {response.text}")

# --- STAGE 2: Select Guidelines (Comorbidities Enabled) ---
elif st.session_state.stage == 'select':
    st.subheader("Select Applicable Guidelines")
    st.markdown("We found the following matches based on the EHR. **Select all that apply for comorbidities:**")
    
    with st.form("guideline_selection_form"):
        selected_diseases = []
        
        # Loop through the structured JSON from the backend
        for match in st.session_state.search_results:
            is_checked = st.checkbox(f"**{match['name']}** ({match['score']}% Match)", key=match['name'])
            st.caption(f"✨ *Rationale: {match['rationale']}*")
            st.write("---") # Visual divider
            
            if is_checked:
                selected_diseases.append(match['name'])
                
        # Submit button
        if st.form_submit_button("Extract Clinical Variables", type="primary"):
            if len(selected_diseases) == 0:
                st.warning("Please select at least one guideline to continue.")
            else:
                # Save the LIST of selected disease names
                st.session_state.selected_diseases = selected_diseases
                st.session_state.stage = 'extract'
                st.rerun()

# --- STAGE 3: Extract & Human Fallback ---
elif st.session_state.stage == 'extract':
    if not st.session_state.extracted_answers:
        with st.spinner("Extracting clinical variables..."):
            response = requests.post(
                f"{API_URL}/extract", 
                json={
                    "ehr_text": st.session_state.ehr_text, 
                    "disease_names": st.session_state.selected_diseases # Passing the list here
                }
            )
            if response.status_code == 200:
                st.session_state.extracted_answers = response.json()["extracted_answers"]
            else:
                st.error(f"Extraction failed: {response.text}")
                st.stop()
                
    st.subheader("Verify Clinical Variables")
    st.markdown("⚠️ **Physician Review Required:** Please verify the data extracted by the AI. You can edit any incorrect values or fill in missing information below.")
    
    with st.form("verification_form"):
        updated_answers = {}
        
        # Loop through every merged question the backend sent us
        for question, llm_answer in st.session_state.extracted_answers.items():
            display_val = "" if str(llm_answer).strip().upper() in ["UNKNOWN", "", "N/A"] else str(llm_answer)
            updated_answers[question] = st.text_input(question, value=display_val)
            
        if st.form_submit_button("Confirm & Generate Prescription", type="primary"):
            st.session_state.extracted_answers = updated_answers
            st.session_state.stage = 'recommend'
            st.rerun()

# --- STAGE 4: Stream Recommendation ---
elif st.session_state.stage == 'recommend':
    st.subheader("Final Recommendation")
    
    col1, col2 = st.columns([8, 2])
    with col2:
        st.button("🔄 New Patient", on_click=reset_app, key="new_patient_btn")
        
    if 'final_rx' not in st.session_state:
        with requests.post(
            f"{API_URL}/recommend", 
            json={
                "disease_names": st.session_state.selected_diseases, # Passing the list here
                "extracted_answers": st.session_state.extracted_answers
            },
            stream=True
        ) as response:
            
            if response.status_code != 200:
                st.error(f"Backend Error {response.status_code}: {response.text}")
                st.stop()
                
            def stream_parser():
                print("\n\n--- INCOMING STREAM START ---")
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk: 
                        print(chunk, end="", flush=True) 
                        yield chunk
                print("\n--- INCOMING STREAM END ---\n")
                        
            st.session_state.final_rx = st.write_stream(stream_parser())
    else:
        st.markdown(st.session_state.final_rx)
