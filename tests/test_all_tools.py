"""
Test all MedGemma tools with example inputs.
Run with: uv run pytest tests/test_all_tools.py -v
"""

import pytest
from core.registry import ToolRegistry

# Initialize registry once for all tests
@pytest.fixture(scope="module")
def registry():
    reg = ToolRegistry()
    reg.auto_discover()
    return reg


# ============== Medical Knowledge ==============

@pytest.mark.asyncio
async def test_medical_calculator(registry):
    tool = await registry.get_tool("medical_calculator")
    input_data = {"calculator": "bmi", "parameters": {"weight_kg": 70, "height_cm": 175}}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success
    assert result.result is not None


@pytest.mark.asyncio
async def test_terminology_explainer(registry):
    tool = await registry.get_tool("terminology_explainer")
    input_data = {"term": "myocardial infarction"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_drug_interaction(registry):
    tool = await registry.get_tool("drug_interaction")
    input_data = {"drugs": ["warfarin", "aspirin", "ibuprofen"]}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_icd_cpt_lookup(registry):
    tool = await registry.get_tool("icd_cpt_lookup")
    input_data = {"query": "diabetes", "code_type": "ICD-10"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_guidelines_rag(registry):
    tool = await registry.get_tool("guidelines_rag")
    input_data = {"query": "hypertension treatment"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_pubmed_search(registry):
    tool = await registry.get_tool("pubmed_search")
    input_data = {"query": "COVID-19 treatment", "max_results": 5}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


# ============== Clinical Decision Support ==============

@pytest.mark.asyncio
async def test_triage_classifier(registry):
    tool = await registry.get_tool("triage_classifier")
    input_data = {
        "chief_complaint": "chest pain",
        "symptoms": ["shortness of breath", "sweating"],
        "vital_signs": {"heart_rate": 110, "blood_pressure_systolic": 90}
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_risk_assessment(registry):
    tool = await registry.get_tool("risk_assessment")
    input_data = {
        "assessment_type": "cardiovascular",
        "patient_data": {
            "age": 65, "gender": "male", "smoker": True,
            "diabetic": True, "systolic_bp": 140, "total_cholesterol": 240
        }
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_differential_diagnosis(registry):
    tool = await registry.get_tool("differential_diagnosis")
    input_data = {
        "chief_complaint": "chest pain",
        "symptoms": ["shortness of breath", "sweating"],
        "patient_age": 55,
        "patient_sex": "male"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_treatment_recommender(registry):
    tool = await registry.get_tool("treatment_recommender")
    input_data = {
        "diagnosis": "type 2 diabetes",
        "patient_age": 50
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


# ============== Document Processing ==============

@pytest.mark.asyncio
async def test_lab_report_parser(registry):
    tool = await registry.get_tool("lab_report_parser")
    input_data = {
        "document_text": "WBC: 12.5 (H) [4.5-11.0], Hemoglobin: 13.5 [12.0-16.0], Glucose: 250 (H) [70-100]"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_radiology_report_parser(registry):
    tool = await registry.get_tool("radiology_report_parser")
    input_data = {
        "document_text": "CHEST X-RAY: Findings: Bilateral infiltrates. Impression: Pneumonia."
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_pathology_report_parser(registry):
    tool = await registry.get_tool("pathology_report_parser")
    input_data = {
        "document_text": "Diagnosis: Invasive ductal carcinoma, Grade 2. Tumor size: 2.1 cm. Margins: Negative."
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_clinical_notes_parser(registry):
    tool = await registry.get_tool("clinical_notes_parser")
    input_data = {
        "document_text": "S: Cough for 2 weeks. O: Bilateral wheezes. A: Acute bronchitis. P: Albuterol inhaler."
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_discharge_summary_parser(registry):
    tool = await registry.get_tool("discharge_summary_parser")
    input_data = {
        "document_text": "Discharge Diagnosis: Acute MI. Medications: Aspirin 81mg, Plavix 75mg. Follow-up: Cardiology 2 weeks."
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_prescription_parser(registry):
    tool = await registry.get_tool("prescription_parser")
    input_data = {
        "document_text": "Metformin 500mg twice daily. Lisinopril 10mg daily. Atorvastatin 20mg at bedtime."
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_insurance_claims_parser(registry):
    tool = await registry.get_tool("insurance_claims_parser")
    input_data = {
        "document_text": "CPT: 99213 Office visit. ICD-10: E11.9 Type 2 diabetes. Billed: $150.00"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


# ============== Medical Imaging ==============
# Note: Image tools require actual image data - using minimal 1x1 PNG for testing

# Minimal valid 1x1 white PNG in base64
DUMMY_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.mark.asyncio
async def test_dicom_handler(registry):
    tool = await registry.get_tool("dicom_handler")
    input_data = {"dicom_path": "/nonexistent/test.dcm"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    # Should handle gracefully (may fail but not crash)
    assert result is not None


@pytest.mark.asyncio
async def test_xray_analyzer(registry):
    tool = await registry.get_tool("xray_analyzer")
    input_data = {"image_base64": DUMMY_IMAGE_BASE64, "body_region": "chest"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result is not None


@pytest.mark.asyncio
async def test_ct_analyzer(registry):
    tool = await registry.get_tool("ct_analyzer")
    input_data = {"image_base64": DUMMY_IMAGE_BASE64, "body_region": "chest"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result is not None


@pytest.mark.asyncio
async def test_mri_analyzer(registry):
    tool = await registry.get_tool("mri_analyzer")
    input_data = {"image_base64": DUMMY_IMAGE_BASE64, "body_region": "brain"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result is not None


@pytest.mark.asyncio
async def test_ultrasound_analyzer(registry):
    tool = await registry.get_tool("ultrasound_analyzer")
    input_data = {"image_base64": DUMMY_IMAGE_BASE64, "exam_type": "abdominal"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result is not None


@pytest.mark.asyncio
async def test_fundus_analyzer(registry):
    tool = await registry.get_tool("fundus_analyzer")
    input_data = {"image_base64": DUMMY_IMAGE_BASE64, "image_type": "color"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result is not None


@pytest.mark.asyncio
async def test_dermoscopy_analyzer(registry):
    tool = await registry.get_tool("dermoscopy_analyzer")
    input_data = {"image_base64": DUMMY_IMAGE_BASE64, "location": "back"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result is not None


@pytest.mark.asyncio
async def test_histopath_analyzer(registry):
    tool = await registry.get_tool("histopath_analyzer")
    input_data = {"image_base64": DUMMY_IMAGE_BASE64, "tissue_type": "breast", "stain": "H&E"}
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result is not None


# ============== Integration ==============

@pytest.mark.asyncio
async def test_fhir_adapter(registry):
    tool = await registry.get_tool("fhir_adapter")
    input_data = {
        "operation": "validate",
        "resource_type": "Patient",
        "resource_data": {
            "resourceType": "Patient",
            "name": [{"family": "Smith", "given": ["John"]}]
        }
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_hl7_parser(registry):
    tool = await registry.get_tool("hl7_parser")
    input_data = {
        "message": "MSH|^~\\&|LAB|HOSP|EMR|HOSP|202401151030||ORU^R01|123456|P|2.5"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_patient_timeline(registry):
    tool = await registry.get_tool("patient_timeline")
    input_data = {
        "patient_id": "12345",
        "events": [
            {"date": "2024-01-15", "type": "encounter", "description": "Annual physical"},
            {"date": "2024-01-20", "type": "lab", "description": "CBC results"}
        ]
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


# ============== Safety & Compliance ==============

@pytest.mark.asyncio
async def test_safety_checker(registry):
    tool = await registry.get_tool("safety_checker")
    input_data = {
        "content": "Take acetaminophen 500mg every 6 hours for pain.",
        "content_type": "medication_advice"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_hallucination_detector(registry):
    tool = await registry.get_tool("hallucination_detector")
    input_data = {
        "content": "Studies show that 87% of patients recover within 3 days.",
        "content_type": "statistic"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_uncertainty_quantifier(registry):
    tool = await registry.get_tool("uncertainty_quantifier")
    input_data = {
        "prediction": "The findings suggest pneumonia",
        "confidence_score": 0.75,
        "prediction_type": "diagnosis"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_audit_logger(registry):
    tool = await registry.get_tool("audit_logger")
    input_data = {
        "event_type": "access",
        "action": "view_record",
        "user_id": "dr_smith",
        "patient_id": "12345",
        "resource_type": "lab_results"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


# ============== Utilities ==============

@pytest.mark.asyncio
async def test_entity_extractor(registry):
    tool = await registry.get_tool("entity_extractor")
    input_data = {
        "text": "Patient has diabetes and hypertension. Taking metformin 500mg and lisinopril 10mg."
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_medical_summarizer(registry):
    tool = await registry.get_tool("medical_summarizer")
    input_data = {
        "text": "Patient is a 65-year-old male with chest pain. Assessment: Possible angina. Plan: ECG, troponin.",
        "summary_type": "brief"
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_patient_translator(registry):
    tool = await registry.get_tool("patient_translator")
    input_data = {
        "text": "Patient presents with dyspnea and bilateral lower extremity edema."
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success


@pytest.mark.asyncio
async def test_report_generator(registry):
    tool = await registry.get_tool("report_generator")
    input_data = {
        "report_type": "consultation",
        "patient_info": {"name": "John Smith", "age": 65},
        "findings": ["Elevated blood pressure"],
        "impressions": ["Hypertension"],
        "recommendations": ["Increase medication"]
    }
    validated = tool.validate_input(input_data)
    result = await tool.execute(validated)
    assert result.success
