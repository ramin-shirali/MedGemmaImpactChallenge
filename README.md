# MedGemma Agent Framework

A modular medical AI agent framework with 36 tools across 8 categories.

## Setup

```bash
uv sync
```

## Usage

```bash
# List all tools
uv run medgemma --list-tools

# Run a specific tool
uv run medgemma --tool <tool_name> --input '<json>'

# Interactive mode
uv run medgemma

# REST API (port 8000)
uv run medgemma-api

# Web UI (port 7860)
uv run medgemma-ui
```

---

## Tool Examples

### Medical Knowledge

**medical_calculator** - BMI, GFR, Wells score, CHADS2, MELD, etc.
```bash
uv run medgemma --tool medical_calculator --input '{"calculator": "bmi", "parameters": {"weight_kg": 70, "height_cm": 175}}'
```

**terminology_explainer** - Explain medical terms in plain language
```bash
uv run medgemma --tool terminology_explainer --input '{"term": "myocardial infarction"}'
```

**drug_interaction** - Check drug-drug interactions
```bash
uv run medgemma --tool drug_interaction --input '{"medications": ["warfarin", "aspirin", "ibuprofen"]}'
```

**icd_cpt_lookup** - Search ICD-10 and CPT codes
```bash
uv run medgemma --tool icd_cpt_lookup --input '{"query": "diabetes", "code_system": "icd10"}'
```

**guidelines_rag** - Search clinical practice guidelines
```bash
uv run medgemma --tool guidelines_rag --input '{"query": "hypertension treatment", "specialty": "cardiology"}'
```

**pubmed_search** - Search PubMed for medical literature
```bash
uv run medgemma --tool pubmed_search --input '{"query": "COVID-19 treatment outcomes", "max_results": 5}'
```

---

### Clinical Decision Support

**triage_classifier** - Emergency severity index classification
```bash
uv run medgemma --tool triage_classifier --input '{"chief_complaint": "chest pain", "symptoms": ["shortness of breath", "sweating"], "vital_signs": {"heart_rate": 110, "blood_pressure_systolic": 90}}'
```

**risk_assessment** - Cardiovascular, falls, readmission risk scores
```bash
uv run medgemma --tool risk_assessment --input '{"assessment_type": "cardiovascular", "patient_data": {"age": 65, "gender": "male", "smoker": true, "diabetic": true, "systolic_bp": 140, "total_cholesterol": 240}}'
```

**differential_diagnosis** - Generate ranked differential diagnoses
```bash
uv run medgemma --tool differential_diagnosis --input '{"symptoms": ["chest pain", "shortness of breath", "sweating"], "patient_info": {"age": 55, "gender": "male"}, "vital_signs": {"heart_rate": 100, "blood_pressure": "150/90"}}'
```

**treatment_recommender** - Evidence-based treatment recommendations
```bash
uv run medgemma --tool treatment_recommender --input '{"diagnosis": "type 2 diabetes", "patient_info": {"age": 50, "allergies": ["penicillin"]}, "current_medications": ["lisinopril"]}'
```

---

### Document Processing

**lab_report_parser** - Parse laboratory reports
```bash
uv run medgemma --tool lab_report_parser --input '{"text": "WBC: 12.5 (H) [4.5-11.0], Hemoglobin: 13.5 [12.0-16.0], Glucose: 250 (H) [70-100], Creatinine: 1.2 [0.7-1.3]"}'
```

**radiology_report_parser** - Parse radiology reports
```bash
uv run medgemma --tool radiology_report_parser --input '{"text": "CHEST X-RAY: Findings: Bilateral lower lobe infiltrates. No pleural effusion. Heart size normal. Impression: Findings consistent with pneumonia."}'
```

**pathology_report_parser** - Parse pathology reports
```bash
uv run medgemma --tool pathology_report_parser --input '{"text": "Diagnosis: Invasive ductal carcinoma, Grade 2. Tumor size: 2.1 cm. Margins: Negative. ER: Positive (95%), PR: Positive (80%), HER2: Negative."}'
```

**clinical_notes_parser** - Parse SOAP and clinical notes
```bash
uv run medgemma --tool clinical_notes_parser --input '{"text": "S: Patient reports persistent cough for 2 weeks. O: Lungs with bilateral wheezes. A: Acute bronchitis. P: Prescribed albuterol inhaler, follow up in 1 week."}'
```

**discharge_summary_parser** - Parse discharge summaries
```bash
uv run medgemma --tool discharge_summary_parser --input '{"text": "Discharge Diagnosis: Acute MI. Procedures: Cardiac catheterization with stent placement. Medications: Aspirin 81mg, Plavix 75mg, Metoprolol 25mg. Follow-up: Cardiology in 2 weeks."}'
```

**prescription_parser** - Parse prescriptions
```bash
uv run medgemma --tool prescription_parser --input '{"text": "Metformin 500mg, take one tablet twice daily with meals. Lisinopril 10mg, take one tablet daily. Atorvastatin 20mg, take one tablet at bedtime."}'
```

**insurance_claims_parser** - Parse insurance claims and EOBs
```bash
uv run medgemma --tool insurance_claims_parser --input '{"text": "CPT: 99213 - Office visit, established patient. ICD-10: E11.9 - Type 2 diabetes. Billed: $150.00. Allowed: $95.00. Patient responsibility: $20.00"}'
```

---

### Medical Imaging

**dicom_handler** - Load and process DICOM files
```bash
uv run medgemma --tool dicom_handler --input '{"file_path": "/path/to/image.dcm", "extract_pixels": true}'
```

**xray_analyzer** - Analyze X-ray images
```bash
uv run medgemma --tool xray_analyzer --input '{"image_data": "<base64_encoded_image>", "body_region": "chest", "clinical_context": "cough and fever"}'
```

**ct_analyzer** - Analyze CT scans
```bash
uv run medgemma --tool ct_analyzer --input '{"image_data": "<base64_encoded_image>", "body_region": "chest", "clinical_context": "rule out pulmonary embolism"}'
```

**mri_analyzer** - Analyze MRI scans
```bash
uv run medgemma --tool mri_analyzer --input '{"image_data": "<base64_encoded_image>", "body_region": "brain", "sequence_type": "T2_FLAIR"}'
```

**ultrasound_analyzer** - Analyze ultrasound images
```bash
uv run medgemma --tool ultrasound_analyzer --input '{"image_data": "<base64_encoded_image>", "exam_type": "thyroid"}'
```

**fundus_analyzer** - Analyze retinal fundus images
```bash
uv run medgemma --tool fundus_analyzer --input '{"image_data": "<base64_encoded_image>", "clinical_context": "diabetic patient annual screening"}'
```

**dermoscopy_analyzer** - Analyze skin lesion images
```bash
uv run medgemma --tool dermoscopy_analyzer --input '{"image_data": "<base64_encoded_image>", "lesion_location": "back", "patient_history": "new mole noticed 3 months ago"}'
```

**histopath_analyzer** - Analyze histopathology slides
```bash
uv run medgemma --tool histopath_analyzer --input '{"image_data": "<base64_encoded_image>", "tissue_type": "breast", "stain_type": "H&E"}'
```

---

### Integration

**fhir_adapter** - Parse and create FHIR R4 resources
```bash
uv run medgemma --tool fhir_adapter --input '{"operation": "parse", "resource_type": "Patient", "data": {"resourceType": "Patient", "name": [{"family": "Smith", "given": ["John"]}], "birthDate": "1970-01-15"}}'
```

**hl7_parser** - Parse HL7 v2.x messages
```bash
uv run medgemma --tool hl7_parser --input '{"message": "MSH|^~\\&|LAB|HOSP|EMR|HOSP|202401151030||ORU^R01|123456|P|2.5\rPID|1||12345^^^HOSP||Smith^John||19700115|M"}'
```

**patient_timeline** - Create unified patient timeline
```bash
uv run medgemma --tool patient_timeline --input '{"patient_id": "12345", "events": [{"date": "2024-01-15", "type": "encounter", "description": "Annual physical"}, {"date": "2024-01-20", "type": "lab", "description": "CBC results"}]}'
```

---

### Safety & Compliance

**safety_checker** - Validate outputs for medical safety
```bash
uv run medgemma --tool safety_checker --input '{"content": "Take 1000mg of acetaminophen every 4 hours for pain relief.", "content_type": "medication_advice"}'
```

**hallucination_detector** - Detect potential hallucinations in AI output
```bash
uv run medgemma --tool hallucination_detector --input '{"content": "Studies show that 87% of patients recover within 3 days using this treatment.", "content_type": "statistic"}'
```

**uncertainty_quantifier** - Quantify prediction uncertainty
```bash
uv run medgemma --tool uncertainty_quantifier --input '{"prediction": "The findings suggest pneumonia", "confidence_score": 0.75, "prediction_type": "diagnosis"}'
```

**audit_logger** - Log actions for HIPAA compliance
```bash
uv run medgemma --tool audit_logger --input '{"action": "view_record", "user_id": "dr_smith", "patient_id": "12345", "resource_type": "lab_results", "details": "Viewed CBC results"}'
```

---

### Utilities

**entity_extractor** - Extract medical entities from text
```bash
uv run medgemma --tool entity_extractor --input '{"text": "Patient has diabetes and hypertension. Currently taking metformin 500mg and lisinopril 10mg daily."}'
```

**medical_summarizer** - Summarize medical documents
```bash
uv run medgemma --tool medical_summarizer --input '{"text": "Patient is a 65-year-old male presenting with chest pain. History includes hypertension and diabetes. Assessment: Possible angina. Plan: ECG, troponin levels, cardiology consult.", "summary_type": "brief"}'
```

**patient_translator** - Translate medical jargon to plain language
```bash
uv run medgemma --tool patient_translator --input '{"text": "Patient presents with dyspnea and bilateral lower extremity edema consistent with CHF exacerbation."}'
```

**report_generator** - Generate structured medical reports
```bash
uv run medgemma --tool report_generator --input '{"report_type": "consultation", "patient_info": {"name": "John Smith", "age": 65}, "findings": ["Elevated blood pressure", "Mild edema"], "impressions": ["Hypertension, uncontrolled"], "recommendations": ["Increase lisinopril to 20mg", "Low sodium diet"]}'
```

---

## Interactive Mode Commands

```
/tools              - List all available tools
/tool <name>        - Get detailed info about a tool
/run <tool> <json>  - Run a tool with JSON input
/history            - Show conversation history
/clear              - Clear history
/quit               - Exit
```
