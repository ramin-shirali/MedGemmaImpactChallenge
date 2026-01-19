"""
MedGemma Agent Framework - Gradio UI

Interactive web interface for the medical AI agent.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# Global instances
agent = None
registry = None


async def initialize_agent():
    """Initialize the agent and registry."""
    global agent, registry

    from core.agent import MedGemmaAgent
    from core.registry import ToolRegistry
    from core.config import MedGemmaConfig

    config = MedGemmaConfig()

    registry = ToolRegistry()
    registry.auto_discover()

    agent = MedGemmaAgent(config=config)
    await agent.initialize()

    return f"Initialized with {len(registry.list_tools())} tools"


def sync_initialize():
    """Synchronous wrapper for initialization."""
    return asyncio.run(initialize_agent())


async def process_query_async(query: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Process a query asynchronously."""
    if not agent:
        return "Please initialize the agent first.", history

    try:
        response = await agent.process(query)
        history.append((query, response))
        return "", history
    except Exception as e:
        return f"Error: {e}", history


def process_query(query: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Process a natural language query."""
    return asyncio.run(process_query_async(query, history))


async def analyze_image_async(
    image,
    modality: str,
    body_region: str,
    clinical_context: str
) -> str:
    """Analyze a medical image asynchronously."""
    if not registry:
        return "Please initialize the agent first."

    try:
        # Map modality to tool
        tool_map = {
            "X-Ray": "xray_analyzer",
            "CT Scan": "ct_analyzer",
            "MRI": "mri_analyzer",
            "Fundus": "fundus_analyzer",
            "Dermoscopy": "dermoscopy_analyzer",
            "Ultrasound": "ultrasound_analyzer",
            "Histopathology": "histopath_analyzer"
        }

        tool_name = tool_map.get(modality)
        if not tool_name:
            return f"Unknown modality: {modality}"

        tool = await registry.get_tool(tool_name)
        if not tool:
            return f"Tool {tool_name} not available"

        # Convert image to base64
        import io
        from PIL import Image

        if image is None:
            return "Please upload an image."

        buffered = io.BytesIO()
        if isinstance(image, str):
            # Image path
            img = Image.open(image)
        else:
            img = Image.fromarray(image)

        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Prepare input
        input_data = {
            "image_data": image_base64,
            "image_format": "image/png"
        }

        if body_region:
            input_data["body_region"] = body_region.lower()
        if clinical_context:
            input_data["clinical_context"] = clinical_context

        validated_input = tool.validate_input(input_data)
        result = await tool.execute(validated_input)

        # Format result
        if result.success:
            output = f"## Analysis Results\n\n"
            output += f"**Modality:** {modality}\n"
            output += f"**Confidence:** {result.confidence:.1%}\n\n"

            result_dict = result.model_dump()
            for key, value in result_dict.items():
                if key not in ["success", "status", "data", "confidence", "errors"]:
                    if isinstance(value, list):
                        output += f"**{key.replace('_', ' ').title()}:**\n"
                        for item in value:
                            if isinstance(item, dict):
                                output += f"  - {json.dumps(item)}\n"
                            else:
                                output += f"  - {item}\n"
                    elif isinstance(value, dict):
                        output += f"**{key.replace('_', ' ').title()}:**\n"
                        for k, v in value.items():
                            output += f"  - {k}: {v}\n"
                    elif value:
                        output += f"**{key.replace('_', ' ').title()}:** {value}\n"

            return output
        else:
            return f"Analysis failed: {result.errors}"

    except Exception as e:
        return f"Error: {e}"


def analyze_image(image, modality: str, body_region: str, clinical_context: str) -> str:
    """Analyze a medical image."""
    return asyncio.run(analyze_image_async(image, modality, body_region, clinical_context))


async def analyze_document_async(document_text: str, document_type: str) -> str:
    """Analyze a medical document asynchronously."""
    if not registry:
        return "Please initialize the agent first."

    if not document_text.strip():
        return "Please enter document text."

    try:
        # Map document type to tool
        tool_map = {
            "Lab Report": "lab_report_parser",
            "Radiology Report": "radiology_report_parser",
            "Pathology Report": "pathology_report_parser",
            "Discharge Summary": "discharge_summary_parser",
            "Clinical Notes": "clinical_notes_parser",
            "Prescription": "prescription_parser"
        }

        tool_name = tool_map.get(document_type)
        if not tool_name:
            return f"Unknown document type: {document_type}"

        tool = await registry.get_tool(tool_name)
        if not tool:
            return f"Tool {tool_name} not available"

        input_data = {"text": document_text}
        validated_input = tool.validate_input(input_data)
        result = await tool.execute(validated_input)

        if result.success:
            output = f"## Document Analysis\n\n"
            output += f"**Document Type:** {document_type}\n"
            output += f"**Confidence:** {result.confidence:.1%}\n\n"

            result_dict = result.model_dump()
            for key, value in result_dict.items():
                if key not in ["success", "status", "data", "confidence", "errors"]:
                    if isinstance(value, list) and value:
                        output += f"**{key.replace('_', ' ').title()}:**\n"
                        for item in value[:10]:  # Limit display
                            if isinstance(item, dict):
                                output += f"  - {json.dumps(item, indent=2)}\n"
                            else:
                                output += f"  - {item}\n"
                    elif isinstance(value, dict) and value:
                        output += f"**{key.replace('_', ' ').title()}:**\n"
                        for k, v in list(value.items())[:10]:
                            output += f"  - {k}: {v}\n"
                    elif value:
                        output += f"**{key.replace('_', ' ').title()}:** {value}\n"

            return output
        else:
            return f"Analysis failed: {result.errors}"

    except Exception as e:
        return f"Error: {e}"


def analyze_document(document_text: str, document_type: str) -> str:
    """Analyze a medical document."""
    return asyncio.run(analyze_document_async(document_text, document_type))


async def run_calculator_async(calculator: str, inputs: str) -> str:
    """Run a medical calculator asynchronously."""
    if not registry:
        return "Please initialize the agent first."

    try:
        tool = await registry.get_tool("medical_calculator")
        if not tool:
            return "Medical calculator tool not available"

        # Parse inputs
        try:
            values = json.loads(inputs)
        except json.JSONDecodeError:
            # Try parsing as key=value pairs
            values = {}
            for pair in inputs.split(","):
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    try:
                        values[key.strip()] = float(val.strip())
                    except ValueError:
                        values[key.strip()] = val.strip()

        input_data = {
            "calculator_name": calculator.lower().replace(" ", "_").replace("-", "_"),
            "values": values
        }

        validated_input = tool.validate_input(input_data)
        result = await tool.execute(validated_input)

        if result.success:
            output = f"## Calculator Result\n\n"
            output += f"**Calculator:** {calculator}\n"
            output += f"**Result:** {result.result}\n"
            if hasattr(result, 'interpretation') and result.interpretation:
                output += f"**Interpretation:** {result.interpretation}\n"
            if hasattr(result, 'formula') and result.formula:
                output += f"**Formula:** {result.formula}\n"
            return output
        else:
            return f"Calculation failed: {result.errors}"

    except Exception as e:
        return f"Error: {e}"


def run_calculator(calculator: str, inputs: str) -> str:
    """Run a medical calculator."""
    return asyncio.run(run_calculator_async(calculator, inputs))


def get_tools_list() -> str:
    """Get formatted list of available tools."""
    if not registry:
        return "Registry not initialized"

    tool_names = registry.list_tools()
    output = "## Available Tools\n\n"

    categories = {}
    for tool_name in tool_names:
        try:
            info = registry.get_tool_info(tool_name)
            cat = info.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(info)
        except Exception:
            pass

    for category, cat_tools in sorted(categories.items()):
        output += f"### {category.title()}\n"
        for tool in cat_tools:
            desc = tool.get('description', '')[:60]
            output += f"- **{tool.get('name', 'unknown')}**: {desc}...\n"
        output += "\n"

    return output


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(title="MedGemma Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # MedGemma Agent Framework

        A modular medical AI assistant powered by MedGemma. This demo provides access to various
        medical analysis tools including image analysis, document parsing, and clinical calculators.

        **Disclaimer:** This is for demonstration purposes only. Not for clinical use.
        """)

        with gr.Row():
            init_btn = gr.Button("Initialize Agent", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)

        init_btn.click(fn=sync_initialize, outputs=init_status)

        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a medical question...",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                submit_btn.click(
                    fn=process_query,
                    inputs=[query_input, chatbot],
                    outputs=[query_input, chatbot]
                )

                query_input.submit(
                    fn=process_query,
                    inputs=[query_input, chatbot],
                    outputs=[query_input, chatbot]
                )

            # Image Analysis Tab
            with gr.TabItem("Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Medical Image")
                        modality_dropdown = gr.Dropdown(
                            choices=["X-Ray", "CT Scan", "MRI", "Fundus", "Dermoscopy", "Ultrasound", "Histopathology"],
                            label="Image Modality",
                            value="X-Ray"
                        )
                        body_region_input = gr.Textbox(
                            label="Body Region (optional)",
                            placeholder="e.g., chest, brain, abdomen"
                        )
                        clinical_context_input = gr.Textbox(
                            label="Clinical Context (optional)",
                            placeholder="e.g., cough, fever, trauma"
                        )
                        analyze_image_btn = gr.Button("Analyze Image", variant="primary")

                    with gr.Column():
                        image_output = gr.Markdown(label="Analysis Results")

                analyze_image_btn.click(
                    fn=analyze_image,
                    inputs=[image_input, modality_dropdown, body_region_input, clinical_context_input],
                    outputs=image_output
                )

            # Document Analysis Tab
            with gr.TabItem("Document Analysis"):
                with gr.Row():
                    with gr.Column():
                        document_input = gr.Textbox(
                            label="Document Text",
                            placeholder="Paste medical document text here...",
                            lines=10
                        )
                        doc_type_dropdown = gr.Dropdown(
                            choices=["Lab Report", "Radiology Report", "Pathology Report",
                                    "Discharge Summary", "Clinical Notes", "Prescription"],
                            label="Document Type",
                            value="Lab Report"
                        )
                        analyze_doc_btn = gr.Button("Analyze Document", variant="primary")

                    with gr.Column():
                        doc_output = gr.Markdown(label="Analysis Results")

                analyze_doc_btn.click(
                    fn=analyze_document,
                    inputs=[document_input, doc_type_dropdown],
                    outputs=doc_output
                )

            # Medical Calculators Tab
            with gr.TabItem("Calculators"):
                with gr.Row():
                    with gr.Column():
                        calc_dropdown = gr.Dropdown(
                            choices=["BMI", "GFR (CKD-EPI)", "MELD Score", "CHADS2-VASc",
                                    "Wells Score DVT", "CURB-65", "APACHE II"],
                            label="Calculator",
                            value="BMI"
                        )
                        calc_inputs = gr.Textbox(
                            label="Input Values (JSON or key=value pairs)",
                            placeholder='{"weight_kg": 70, "height_m": 1.75} or weight_kg=70, height_m=1.75',
                            lines=3
                        )
                        calc_btn = gr.Button("Calculate", variant="primary")

                    with gr.Column():
                        calc_output = gr.Markdown(label="Result")

                calc_btn.click(
                    fn=run_calculator,
                    inputs=[calc_dropdown, calc_inputs],
                    outputs=calc_output
                )

            # Tools Tab
            with gr.TabItem("Available Tools"):
                tools_display = gr.Markdown()
                refresh_btn = gr.Button("Refresh Tool List")
                refresh_btn.click(fn=get_tools_list, outputs=tools_display)

        gr.Markdown("""
        ---
        **Note:** This interface is for demonstration and educational purposes only.
        All outputs should be reviewed by qualified healthcare professionals.
        """)

    return demo


def launch(share: bool = False, server_port: int = 7860):
    """Launch the Gradio interface."""
    demo = create_ui()
    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch()
