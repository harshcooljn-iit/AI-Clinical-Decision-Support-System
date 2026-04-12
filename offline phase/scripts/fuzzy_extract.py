import os
import re
import difflib

# Same topics data logic
topics_data = """
fever, Fever of unknown origin, Acute rheumatic fever, Anaemia, Typhoid fever, Malaria, Dengue, Chikungunya, Tuberculosis and RNTCP, Epilepsy, Status epilepticus, Urinary tract infection,
Cardiopulmonary resuscitation, Anaphylaxis, Acute airway obstruction, Stridor, Shock, Fluid and electrolyte imbalance, Septicemia, Organophosphorus poisoning, Kerosene and petrol poisoning, Datura poisoning, Abdominal injury, Foreign body in ear, Chemical injuries of the eye, Corneal and conjunctival foreign bodies, Traumatic hyphema, Animal bites- Dog bite, Snake bite, Insect and Arachnid bites, Scorpion bite,
Infective endocarditis, Acute pericarditis, Cardiomyopathy, Hypertension, Angina pectoris, Myocardial infarction, Congestive heart failure, Arrhythmias, Ventricular tachycardia, Atrial fibrillation, bradyarrhythmias,
Migraine, Tension headache, Cluster headache, Stroke, Facial palsy, Parkinson’s disease, Dementia, Guillain- Barre Syndrome, Acute bacterial meningitis, Viral encephalitis, Neurocysticercosis, Paraplegia and quadriplegia, Vertigo,
Nephrotic syndrome, Chronic kidney disease, Acute renal failure, Hypokalemia, Hyperkalemia,
Hypothyroidism, Hyperthyroidism, Hypocalcemia, Hypercalcemia, Diabetes mellitus, Diabetes ketoacidosis,
Aphthous ulcers, Dyspepsia, Peptic ulcer disease, Vomiting, Irritable bowel syndrome, Acute gastroenteritis, Ulcerative colitis, Gastroesophageal reflux disorder, Upper gastrointestinal bleeding, Acute pancreatitis, Chronic pancreatitis, Constipation, Amoebic liver abscess, Pyogenic liver abscess,
Tetanus, Leptospirosis, Cholera, Influenza infection, HIV and AIDS, Opportunistic infections, Amoebiasis, Giardiasis, Hookworm infestation, Roundworm infestation, Enterobiasis, Filariasis,
Ear wax, Keratosis obturans, Furuncle, Otomycosis, Malignant otitis externa, Acute suppurative otitis media, Chronic suppurative otitis media, Serous otitis media, Acute sinusitis, Epistaxis, Allergic rhinitis, Common cold, Furunculosis of nose, Acute tonsillitis, Chronic tonsillitis, Acute parotitis, Bacterial parotitis,
Stye, Chalazion, Viral conjunctivitis, Allergic conjunctivitis, Atopic conjunctivitis, Non gonococcal conjunctivitis, Gonococcal conjunctivitis, Trachoma, Corneal Ulcer, Bacterial keratitis, Fungal keratitis, Viral keratitis, Glaucoma, Primary open angle glaucoma, Primary angle closure glaucoma, Lens induced glaucoma, Anterior Uveitis, Orbital cellulitis, Endophthalmitis, Optic neuritis, Diabetic retinopathy, Retinal detachment, Vitamin A deficiency, Senile cataract, Refractive errors, Strabismus, Computer vision syndrome,
Bacterial skin infections, Leprosy, Cutaneous tuberculosis, Scabies, Pediculosis, Myiasis, Pityrosporum infections, Pityriasis, Tinea corporis, Capitis, Onychomycosis, Candidiasis, Diaper dermatitis, Herpes labialis, Chickenpox, Herpes zoster, Molluscum contagiosum, Wart, Acne vulgaris, Miliaria, Eczema, Contact dermatitis, Pityriasis alba, Vitiligo, Albinism, Melasma, Urticaria, Psoriasis, Lichen planus, Alopecia areata, Pemphigus vulgaris, Cutaneous reactions to drugs, Herpes genitalis, Syphilis, Chancroid, Lymphogranuloma venerum, Urethral discharge, Dermatological emergencies,
Normal Pregnancy, Nausea and vomiting in pregnancy, First trimester bleeding, Anaemia in pregnancy, Pregnancy induced hypertension, Pregnancy with diabetes, Pregnancy with heart disease, Antepartum haemorrhage, Premature rupture of membranes, Preterm labour, Normal labour, Fetal distress in labour, Meconium stained liquor, Induction and augmentation of labour, Postpartum haemorrhage, Prevention of maternal to child transmission, Vaginal discharge, Pelvic inflammatory disease, Pre-Menstrual syndrome, Dysfunctional uterine bleeding , Menopause, Postmenopausal bleeding, Screening guidelines for early detection of cancer, Contraception,
Schizophrenia and other psychotic disorders, Major depressive disorders, Bipolar mood disorder, Insomnia, Generalized anxiety disorders, Panic disorder, Obsessive compulsive disorder, Phobic disorder, Post-traumatic stress disorder, Conversion disorder, Dissociative disorder, Alcohol use disorder, Opioid use disorder, Cannabis use disorder, Autistic disorder, Attention deficit hyperactivity disorder, Mental retardation, Suicidal patient, Psychotropic drugs in special population, Management of wandering mentally ill patient, Personality disorder,
Osteoarthritis, Rheumatoid arthritis, Cervical and lumbar spondylosis, Sprains, Acute pyogenic osteomyelitis, Acute septic arthritis,
Pre-operative assessment and preparations, Post-operative care, Surgical site infection, Wound care, Sterilization and disinfection, Suturing of primary wound, Acute appendicitis, Hernia, Hydrocele, Cholelithiasis, Hydatid disease of liver, Obstructive jaundice, Intestinal obstruction, Typhoid perforation, Small bowel tuberculosis, Anal fissure, Fistula in ano, Hemorrhoids, Diabetic foot, Varicose veins, Deep vein thrombosis, Cervical lymphadenopathy, Thyroid nodule, Benign breast diseases, Breast abscess, Carcinoma breast, Renal stones, Urinary retention,
Community acquired pneumonia, Hospital acquired pneumonia, Bronchial asthma, Chronic obstructive pulmonary disease, Chronic obstructive pulmonary diseases, Bronchiectasis,
Newborn care, Low birth weight baby, Neonatal jaundice, Management of common clinical problems in new born, Immunization schedule, Fluid and electrolyte therapy in children, Iron deficiency anemia, Megaloblastic anemia, Protein energy malnutrition, Nutritional rickets, Pica, Primary nocturnal enuresis, Wheezy child, Acute bronchiolitis, Pneumonia, Thrush (oral candidiasis), Constipation, Acute diarrhea, Acute viral hepatitis, Chicken pox(varicella), Measles, Mumps, Acute flaccid paralysis, Pertusis, Cardiac failure, Diabetes mellitus, Hypothyroidism, Urinary tract infection, Post streptococcal acute glomerulonephritis, Nephrotic syndrome, Febrile seizure, Acute pyogenic meningitis, Tubercular meningitis, Status epilepticus, Hypovolemic shock, Diphtheria, Severe malaria, Hypertensive encephalopathy, Acute severe asthma,
Tooth avulsion, Toothache, Tooth/maxillary fracture, Dental caries, Dental abscess, Adult type periodontitis, Juvenile periodontitis, Inflammatory gingival enlargement, Dental fluorosis, Trigeminal neuralgia, Oral submucous fibrosis, Cyst of tumor of jaw, Oral hygiene, Antibiotic prophylaxis in dental procedure
"""

diseases_list = [d.strip() for d in topics_data.split(',') if d.strip()]
diseases = set(diseases_list)

def normalize(text):
    text = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
    return text

normalized_to_original = {normalize(d): d for d in diseases}
all_normalized = list(normalized_to_original.keys())

out_dir = "disease_markdown_files"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Hardcode logic for common broad categories/aliases
def manual_alias_check(header):
    header = header.lower()
    if 'tinea cruris and corporis' in header: return 'Tinea corporis'
    if 'tinea capitis' in header: return 'Capitis'
    if 'tooth fracture' in header or 'maxillary bone fracture' in header: return 'Tooth/maxillary fracture'
    if 'good oral hygiene' in header: return 'Oral hygiene'
    return None

def get_best_match(header_text):
    alias = manual_alias_check(header_text)
    if alias: return alias
    
    head_no_paren = re.sub(r'\([^)]*\)', '', header_text)
    norm_head = normalize(header_text)
    norm_head_no_paren = normalize(head_no_paren)
    
    if not norm_head: return None

    # Try exact normalized match
    if norm_head in normalized_to_original:
        return normalized_to_original[norm_head]
    if norm_head_no_paren in normalized_to_original:
        return normalized_to_original[norm_head_no_paren]
    
    # Try reverse substring match (header without parenthesis is substring of disease index name)
    for norm_dis in all_normalized:
        if len(norm_head_no_paren) > 6 and norm_head_no_paren in norm_dis:
            return normalized_to_original[norm_dis]
        if len(norm_head) > 6 and norm_head in norm_dis:
            return normalized_to_original[norm_dis]
            
    # Try difflib match
    matches = difflib.get_close_matches(norm_head_no_paren, all_normalized, n=1, cutoff=0.8)
    if matches:
        return normalized_to_original[matches[0]]
        
    matches = difflib.get_close_matches(norm_head, all_normalized, n=1, cutoff=0.8)
    if matches:
        return normalized_to_original[matches[0]]
        
    return None

with open('standard-treatment-guidelines.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_disease = None
content_buffer = []

for line in lines:
    matched_disease = None
    if line.strip().startswith('#'):
        text = re.sub(r'^#+\s*', '', line).strip()
        matched_disease = get_best_match(text)
    
    if matched_disease:
        # Save previous disease content if there is any
        if current_disease:
            filename = current_disease.replace(' ', '_').replace('/', '_').replace('’', '').replace('(', '').replace(')', '').replace('-', '_').lower() + '.md'
            with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as out_f:
                out_f.writelines(content_buffer)
        
        # Start new disease
        current_disease = matched_disease
        content_buffer = [line]
    else:
        # If we found at least one disease, append lines to its buffer
        if current_disease is not None:
            content_buffer.append(line)

# Save the very last one
if current_disease and content_buffer:
    filename = current_disease.replace(' ', '_').replace('/', '_').replace('’', '').replace('(', '').replace(')', '').replace('-', '_').lower() + '.md'
    with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as out_f:
        out_f.writelines(content_buffer)

print(f"Finished extracting into semantic disease files using fuzzy matching.")
