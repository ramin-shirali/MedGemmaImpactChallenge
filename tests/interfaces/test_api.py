"""
Tests for the MedGemma REST API.
Run with: uv run pytest tests/interfaces/test_api.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware


@pytest.fixture(scope="module")
def mock_registry():
    """Create a mock registry with tools."""
    from core.registry import ToolRegistry
    registry = ToolRegistry()
    registry.auto_discover()
    return registry


@pytest.fixture(scope="module")
def test_app(mock_registry):
    """Create a test FastAPI app without lifespan."""
    # Import the module
    import interfaces.api as api_module

    # Set up registry (avoid full agent initialization)
    api_module.registry = mock_registry
    api_module.agent = None

    # Create a new app without lifespan for testing
    app = FastAPI(title="Test MedGemma API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import route functions directly and add them
    from interfaces.api import (
        health_check, list_tools, get_tool, execute_tool,
        process_query, process_query_async, get_task_status,
        HealthResponse, ToolInfo, ToolExecutionResponse,
        QueryResponse, AsyncTaskResponse, TaskStatusResponse,
        ToolExecutionRequest, QueryRequest
    )

    app.get("/health", response_model=HealthResponse)(health_check)
    app.get("/tools", response_model=list[ToolInfo])(list_tools)
    app.get("/tools/{tool_name}", response_model=ToolInfo)(get_tool)
    app.post("/tools/{tool_name}/execute", response_model=ToolExecutionResponse)(execute_tool)
    app.post("/query", response_model=QueryResponse)(process_query)
    app.post("/query/async", response_model=AsyncTaskResponse)(process_query_async)
    app.get("/tasks/{task_id}", response_model=TaskStatusResponse)(get_task_status)

    return app


@pytest.fixture(scope="module")
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


# ============== Health Endpoint ==============

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "tools_available" in data
    assert data["tools_available"] > 0


# ============== Tools Endpoints ==============

def test_list_tools(client):
    """Test listing all tools."""
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    # Check tool structure
    tool = data[0]
    assert "name" in tool
    assert "description" in tool
    assert "version" in tool
    assert "category" in tool
    assert "input_schema" in tool
    assert "output_schema" in tool


def test_get_specific_tool(client):
    """Test getting info about a specific tool."""
    response = client.get("/tools/medical_calculator")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "medical_calculator"
    assert "description" in data
    assert "input_schema" in data


def test_get_nonexistent_tool(client):
    """Test getting a tool that doesn't exist."""
    response = client.get("/tools/nonexistent_tool")
    assert response.status_code == 404


# ============== Tool Execution Endpoints ==============

def test_execute_medical_calculator(client):
    """Test executing the medical calculator tool."""
    response = client.post(
        "/tools/medical_calculator/execute",
        json={
            "tool_name": "medical_calculator",
            "input_data": {
                "calculator": "bmi",
                "parameters": {"weight_kg": 70, "height_cm": 175}
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["tool_name"] == "medical_calculator"
    assert "result" in data
    assert "processing_time_ms" in data


def test_execute_terminology_explainer(client):
    """Test executing the terminology explainer tool."""
    response = client.post(
        "/tools/terminology_explainer/execute",
        json={
            "tool_name": "terminology_explainer",
            "input_data": {"term": "myocardial infarction"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_drug_interaction(client):
    """Test executing the drug interaction tool."""
    response = client.post(
        "/tools/drug_interaction/execute",
        json={
            "tool_name": "drug_interaction",
            "input_data": {"drugs": ["warfarin", "aspirin"]}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_lab_report_parser(client):
    """Test executing the lab report parser tool."""
    response = client.post(
        "/tools/lab_report_parser/execute",
        json={
            "tool_name": "lab_report_parser",
            "input_data": {
                "document_text": "WBC: 12.5 (H) [4.5-11.0], Hemoglobin: 13.5 [12.0-16.0]"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_triage_classifier(client):
    """Test executing the triage classifier tool."""
    response = client.post(
        "/tools/triage_classifier/execute",
        json={
            "tool_name": "triage_classifier",
            "input_data": {
                "chief_complaint": "chest pain",
                "symptoms": ["shortness of breath"],
                "vital_signs": {"heart_rate": 110}
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_fhir_adapter(client):
    """Test executing the FHIR adapter tool."""
    response = client.post(
        "/tools/fhir_adapter/execute",
        json={
            "tool_name": "fhir_adapter",
            "input_data": {
                "operation": "validate",
                "resource_type": "Patient",
                "resource_data": {
                    "resourceType": "Patient",
                    "name": [{"family": "Smith", "given": ["John"]}]
                }
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_safety_checker(client):
    """Test executing the safety checker tool."""
    response = client.post(
        "/tools/safety_checker/execute",
        json={
            "tool_name": "safety_checker",
            "input_data": {
                "content": "Take acetaminophen 500mg every 6 hours.",
                "content_type": "medication_advice"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_entity_extractor(client):
    """Test executing the entity extractor tool."""
    response = client.post(
        "/tools/entity_extractor/execute",
        json={
            "tool_name": "entity_extractor",
            "input_data": {
                "text": "Patient has diabetes and hypertension."
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_nonexistent_tool(client):
    """Test executing a tool that doesn't exist."""
    response = client.post(
        "/tools/nonexistent_tool/execute",
        json={
            "tool_name": "nonexistent_tool",
            "input_data": {}
        }
    )
    assert response.status_code == 404


def test_execute_with_invalid_input(client):
    """Test executing a tool with invalid input."""
    response = client.post(
        "/tools/medical_calculator/execute",
        json={
            "tool_name": "medical_calculator",
            "input_data": {"invalid": "data"}
        }
    )
    assert response.status_code == 400


# ============== Query Endpoint (requires agent) ==============

def test_query_without_agent(client):
    """Test that query endpoint returns 503 when agent not initialized."""
    response = client.post(
        "/query",
        json={"query": "What is diabetes?"}
    )
    assert response.status_code == 503


def test_async_query_without_agent(client):
    """Test async query returns task ID even without agent."""
    response = client.post(
        "/query/async",
        json={"query": "What is diabetes?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"


# ============== Task Status Endpoint ==============

def test_get_task_status_not_found(client):
    """Test getting status of nonexistent task."""
    response = client.get("/tasks/nonexistent-task-id")
    assert response.status_code == 404


# ============== Additional Tool Tests ==============

def test_execute_risk_assessment(client):
    """Test executing the risk assessment tool."""
    response = client.post(
        "/tools/risk_assessment/execute",
        json={
            "tool_name": "risk_assessment",
            "input_data": {
                "assessment_type": "cardiovascular",
                "patient_data": {
                    "age": 65, "gender": "male", "smoker": True,
                    "diabetic": True, "systolic_bp": 140, "total_cholesterol": 240
                }
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_differential_diagnosis(client):
    """Test executing the differential diagnosis tool."""
    response = client.post(
        "/tools/differential_diagnosis/execute",
        json={
            "tool_name": "differential_diagnosis",
            "input_data": {
                "chief_complaint": "chest pain",
                "symptoms": ["shortness of breath"],
                "patient_age": 55,
                "patient_sex": "male"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_icd_cpt_lookup(client):
    """Test executing the ICD/CPT lookup tool."""
    response = client.post(
        "/tools/icd_cpt_lookup/execute",
        json={
            "tool_name": "icd_cpt_lookup",
            "input_data": {
                "query": "diabetes",
                "code_type": "ICD-10"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_hl7_parser(client):
    """Test executing the HL7 parser tool."""
    response = client.post(
        "/tools/hl7_parser/execute",
        json={
            "tool_name": "hl7_parser",
            "input_data": {
                "message": "MSH|^~\\&|LAB|HOSP|EMR|HOSP|202401151030||ORU^R01|123456|P|2.5"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_execute_hallucination_detector(client):
    """Test executing the hallucination detector tool."""
    response = client.post(
        "/tools/hallucination_detector/execute",
        json={
            "tool_name": "hallucination_detector",
            "input_data": {
                "content": "Studies show that 87% of patients recover.",
                "content_type": "statistic"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
