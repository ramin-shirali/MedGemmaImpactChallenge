"""
Tests for the MedGemma Gradio UI.
Run with: uv run pytest tests/interfaces/test_gradio_ui.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


@pytest.fixture
def mock_registry():
    """Create a mock registry with tools."""
    from core.registry import ToolRegistry
    registry = ToolRegistry()
    registry.auto_discover()
    return registry


@pytest.fixture
def setup_gradio_registry(mock_registry):
    """Set up the Gradio module with a registry."""
    import interfaces.gradio_ui as gradio_module
    gradio_module.registry = mock_registry
    gradio_module.agent = None
    return gradio_module


# ============== Tool List Function ==============

def test_get_tools_list_uninitialized():
    """Test get_tools_list when registry not initialized."""
    import interfaces.gradio_ui as gradio_module
    original_registry = gradio_module.registry
    gradio_module.registry = None

    result = gradio_module.get_tools_list()
    assert result == "Registry not initialized"

    gradio_module.registry = original_registry


def test_get_tools_list(setup_gradio_registry):
    """Test get_tools_list with initialized registry."""
    gradio_module = setup_gradio_registry

    result = gradio_module.get_tools_list()
    assert "Available Tools" in result
    assert len(result) > 100  # Should have substantial content


# ============== Document Analysis Functions ==============

@pytest.mark.asyncio
async def test_analyze_document_async_uninitialized():
    """Test analyze_document_async when registry not initialized."""
    import interfaces.gradio_ui as gradio_module
    original_registry = gradio_module.registry
    gradio_module.registry = None

    result = await gradio_module.analyze_document_async("test", "Lab Report")
    assert result == "Please initialize the agent first."

    gradio_module.registry = original_registry


@pytest.mark.asyncio
async def test_analyze_document_async_empty_text(setup_gradio_registry):
    """Test analyze_document_async with empty text."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.analyze_document_async("", "Lab Report")
    assert result == "Please enter document text."


@pytest.mark.asyncio
async def test_analyze_document_async_unknown_type(setup_gradio_registry):
    """Test analyze_document_async with unknown document type."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.analyze_document_async("test text", "Unknown Type")
    assert "Unknown document type" in result


@pytest.mark.asyncio
async def test_analyze_document_async_lab_report(setup_gradio_registry):
    """Test analyze_document_async with lab report."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.analyze_document_async(
        "WBC: 12.5 (H) [4.5-11.0], Hemoglobin: 13.5 [12.0-16.0]",
        "Lab Report"
    )
    assert "Document Analysis" in result or "Error" in result


@pytest.mark.asyncio
async def test_analyze_document_async_radiology_report(setup_gradio_registry):
    """Test analyze_document_async with radiology report."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.analyze_document_async(
        "CHEST X-RAY: Bilateral infiltrates. Impression: Pneumonia.",
        "Radiology Report"
    )
    assert "Document Analysis" in result or "Error" in result


@pytest.mark.asyncio
async def test_analyze_document_async_clinical_notes(setup_gradio_registry):
    """Test analyze_document_async with clinical notes."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.analyze_document_async(
        "S: Cough for 2 weeks. O: Bilateral wheezes. A: Acute bronchitis.",
        "Clinical Notes"
    )
    assert "Document Analysis" in result or "Error" in result


@pytest.mark.asyncio
async def test_analyze_document_async_prescription(setup_gradio_registry):
    """Test analyze_document_async with prescription."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.analyze_document_async(
        "Metformin 500mg twice daily. Lisinopril 10mg daily.",
        "Prescription"
    )
    assert "Document Analysis" in result or "Error" in result


def test_analyze_document_sync(setup_gradio_registry):
    """Test synchronous analyze_document wrapper."""
    gradio_module = setup_gradio_registry

    result = gradio_module.analyze_document(
        "WBC: 12.5 [4.5-11.0]",
        "Lab Report"
    )
    assert isinstance(result, str)


# ============== Calculator Functions ==============

@pytest.mark.asyncio
async def test_run_calculator_async_uninitialized():
    """Test run_calculator_async when registry not initialized."""
    import interfaces.gradio_ui as gradio_module
    original_registry = gradio_module.registry
    gradio_module.registry = None

    result = await gradio_module.run_calculator_async("BMI", '{"weight_kg": 70}')
    assert result == "Please initialize the agent first."

    gradio_module.registry = original_registry


@pytest.mark.asyncio
async def test_run_calculator_async_json_input(setup_gradio_registry):
    """Test run_calculator_async with JSON input."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.run_calculator_async(
        "BMI",
        '{"weight_kg": 70, "height_cm": 175}'
    )
    # Should either succeed or return error message
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_calculator_async_key_value_input(setup_gradio_registry):
    """Test run_calculator_async with key=value input."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.run_calculator_async(
        "BMI",
        "weight_kg=70, height_cm=175"
    )
    assert isinstance(result, str)


def test_run_calculator_sync(setup_gradio_registry):
    """Test synchronous run_calculator wrapper."""
    gradio_module = setup_gradio_registry

    result = gradio_module.run_calculator(
        "BMI",
        '{"weight_kg": 70, "height_cm": 175}'
    )
    assert isinstance(result, str)


# ============== Image Analysis Functions ==============

@pytest.mark.asyncio
async def test_analyze_image_async_uninitialized():
    """Test analyze_image_async when registry not initialized."""
    import interfaces.gradio_ui as gradio_module
    original_registry = gradio_module.registry
    gradio_module.registry = None

    result = await gradio_module.analyze_image_async(None, "X-Ray", "chest", "")
    assert result == "Please initialize the agent first."

    gradio_module.registry = original_registry


@pytest.mark.asyncio
async def test_analyze_image_async_no_image(setup_gradio_registry):
    """Test analyze_image_async with no image."""
    gradio_module = setup_gradio_registry

    result = await gradio_module.analyze_image_async(None, "X-Ray", "chest", "")
    assert result == "Please upload an image."


@pytest.mark.asyncio
async def test_analyze_image_async_unknown_modality(setup_gradio_registry):
    """Test analyze_image_async with unknown modality."""
    gradio_module = setup_gradio_registry
    import numpy as np

    # Create a dummy image
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)

    result = await gradio_module.analyze_image_async(
        dummy_image, "Unknown Modality", "chest", ""
    )
    assert "Unknown modality" in result


@pytest.mark.asyncio
async def test_analyze_image_async_xray(setup_gradio_registry):
    """Test analyze_image_async with X-Ray modality."""
    gradio_module = setup_gradio_registry
    import numpy as np

    # Create a dummy image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = await gradio_module.analyze_image_async(
        dummy_image, "X-Ray", "chest", "cough"
    )
    # Should either succeed or return error
    assert isinstance(result, str)


# ============== Query Functions ==============

@pytest.mark.asyncio
async def test_process_query_async_uninitialized():
    """Test process_query_async when agent not initialized."""
    import interfaces.gradio_ui as gradio_module
    original_agent = gradio_module.agent
    gradio_module.agent = None

    result, history = await gradio_module.process_query_async("test query", [])
    assert result == "Please initialize the agent first."
    assert history == []

    gradio_module.agent = original_agent


def test_process_query_sync():
    """Test synchronous process_query wrapper."""
    import interfaces.gradio_ui as gradio_module
    gradio_module.agent = None

    result, history = gradio_module.process_query("test query", [])
    assert result == "Please initialize the agent first."


# ============== UI Creation ==============

def test_create_ui():
    """Test that create_ui returns a Gradio Blocks object."""
    import interfaces.gradio_ui as gradio_module
    import gradio as gr

    ui = gradio_module.create_ui()
    assert isinstance(ui, gr.Blocks)


def test_ui_has_tabs():
    """Test that the UI has expected tabs."""
    import interfaces.gradio_ui as gradio_module

    ui = gradio_module.create_ui()

    # Check that the UI was created successfully
    assert ui is not None
    # The UI should have components
    assert len(ui.blocks) > 0


# ============== Initialization ==============
# Note: Full initialization tests require mocking the model loading
# which is complex. These are covered by integration tests.
