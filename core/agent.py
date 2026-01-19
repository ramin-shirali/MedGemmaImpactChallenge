"""
MedGemma Agent Framework - Main Agent Orchestrator

This module provides the main agent that orchestrates all components:
- Manages conversation flow
- Routes requests to appropriate tools
- Integrates with MedGemma model
- Handles multi-modal inputs (images, documents, text)
- Coordinates planning and execution

Usage:
    from core.agent import MedGemmaAgent

    # Create and initialize agent
    agent = MedGemmaAgent()
    await agent.initialize()

    # Process a query
    response = await agent.process("Analyze this CT scan", images=["scan.dcm"])

    # Chat interface
    response = await agent.chat("What are the findings?")

    # Cleanup
    await agent.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from core.config import get_config, MedGemmaConfig, MEDICAL_DISCLAIMER
from core.memory import ConversationMemory, PatientContext, MessageRole
from core.planner import TaskPlanner, Plan, Task
from core.registry import ToolRegistry, get_registry


logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Response from the agent."""

    message: str = Field(description="Main response message")
    findings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Structured findings if applicable"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Recommendations if applicable"
    )
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Results from tool executions"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )
    disclaimer: Optional[str] = Field(
        default=None,
        description="Medical disclaimer if applicable"
    )
    plan_executed: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Execution plan details"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


class FileInfo(BaseModel):
    """Information about an uploaded file."""

    path: str
    type: str  # dicom, image, pdf, text, etc.
    name: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MedGemmaAgent:
    """
    Main agent orchestrator for the MedGemma framework.

    Coordinates between the model, tools, memory, and planner to provide
    a unified interface for medical AI assistance.
    """

    def __init__(
        self,
        config: Optional[MedGemmaConfig] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the agent.

        Args:
            config: Optional configuration
            system_prompt: Optional custom system prompt
        """
        self._config = config or get_config()
        self._registry: Optional[ToolRegistry] = None
        self._planner: Optional[TaskPlanner] = None
        self._memory: Optional[ConversationMemory] = None
        self._model = None
        self._processor = None
        self._initialized = False

        # Default system prompt
        self._system_prompt = system_prompt or self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are MedGemma, an advanced medical AI assistant powered by Google's MedGemma model.

Your capabilities include:
- Analyzing medical images (CT, MRI, X-ray, pathology, dermoscopy, fundus, ultrasound)
- Processing medical documents (lab reports, radiology reports, clinical notes, prescriptions)
- Providing medical knowledge (drug interactions, clinical guidelines, medical terminology)
- Supporting clinical decision-making (differential diagnosis, risk assessment, triage)

Guidelines:
1. Always prioritize patient safety
2. Provide evidence-based information
3. Acknowledge uncertainty and limitations
4. Recommend professional consultation when appropriate
5. Maintain patient privacy and confidentiality
6. Use clear, professional medical terminology with explanations

Remember: You are an AI assistant. Your analysis should support, not replace, clinical judgment.
"""

    async def initialize(self) -> None:
        """
        Initialize all agent components.

        Loads the model, sets up the registry, and prepares for processing.
        """
        if self._initialized:
            return

        logger.info("Initializing MedGemma Agent...")

        # Initialize registry
        self._registry = get_registry(self._config)

        # Initialize planner
        self._planner = TaskPlanner(self._registry, self._config)

        # Initialize memory
        self._memory = ConversationMemory(
            config=self._config,
            system_prompt=self._system_prompt
        )

        # Load MedGemma model
        await self._load_model()

        # Ensure directories exist
        self._config.ensure_directories()

        self._initialized = True
        logger.info("MedGemma Agent initialized successfully")

    async def _load_model(self) -> None:
        """Load the MedGemma model."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch

            model_config = self._config.model
            device = model_config.get_device()

            logger.info(f"Loading model: {model_config.model_id}")
            logger.info(f"Device: {device}")

            # Set up model kwargs
            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": model_config.trust_remote_code,
            }

            # Handle dtype
            if model_config.torch_dtype != "auto":
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                if model_config.torch_dtype in dtype_map:
                    model_kwargs["torch_dtype"] = dtype_map[model_config.torch_dtype]

            # Handle quantization
            if model_config.quantization == "int8":
                model_kwargs["load_in_8bit"] = True
            elif model_config.quantization == "int4":
                model_kwargs["load_in_4bit"] = True

            # Handle flash attention
            if model_config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Get HuggingFace token if available
            token = self._config.api_keys.huggingface_token

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                model_config.model_id,
                trust_remote_code=model_config.trust_remote_code,
                token=token
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                device_map=device if device != "cpu" else None,
                token=token,
                **model_kwargs
            )

            if device == "cpu":
                self._model = self._model.to(device)

            logger.info("Model loaded successfully")

        except ImportError as e:
            logger.warning(f"Could not load transformers: {e}")
            logger.warning("Running in tool-only mode without model")
            self._model = None
            self._processor = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None
            self._processor = None

    async def process(
        self,
        query: str,
        images: Optional[List[Union[str, Path]]] = None,
        documents: Optional[List[Union[str, Path]]] = None,
        patient_context: Optional[PatientContext] = None,
        use_planning: bool = True
    ) -> AgentResponse:
        """
        Process a user query with optional files.

        Args:
            query: User's question or request
            images: Optional list of image file paths
            documents: Optional list of document file paths
            patient_context: Optional patient context
            use_planning: Whether to use the planner for multi-step tasks

        Returns:
            Agent response with findings and recommendations
        """
        if not self._initialized:
            await self.initialize()

        # Update memory with user message
        self._memory.add_user_message(query)

        # Update patient context if provided
        if patient_context:
            self._memory.patient_context = patient_context

        # Prepare file information
        files = self._prepare_files(images, documents)

        try:
            if use_planning and self._should_use_planning(query, files):
                # Complex query - use planner
                response = await self._process_with_planning(query, files)
            else:
                # Simple query - direct processing
                response = await self._process_direct(query, files)

            # Add medical disclaimer if configured
            if self._config.safety.require_disclaimer:
                response.disclaimer = MEDICAL_DISCLAIMER.strip()

            # Update memory with assistant response
            self._memory.add_assistant_message(response.message)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = AgentResponse(
                message=f"I encountered an error while processing your request: {str(e)}",
                metadata={"error": str(e)}
            )
            self._memory.add_assistant_message(error_response.message)
            return error_response

    def _prepare_files(
        self,
        images: Optional[List[Union[str, Path]]],
        documents: Optional[List[Union[str, Path]]]
    ) -> List[FileInfo]:
        """Prepare file information for processing."""
        files = []

        if images:
            for img_path in images:
                path = Path(img_path)
                file_type = self._detect_image_type(path)
                files.append(FileInfo(
                    path=str(path),
                    type=file_type,
                    name=path.name,
                    size_bytes=path.stat().st_size if path.exists() else None
                ))

        if documents:
            for doc_path in documents:
                path = Path(doc_path)
                file_type = self._detect_document_type(path)
                files.append(FileInfo(
                    path=str(path),
                    type=file_type,
                    name=path.name,
                    size_bytes=path.stat().st_size if path.exists() else None
                ))

        return files

    def _detect_image_type(self, path: Path) -> str:
        """Detect the type of medical image."""
        suffix = path.suffix.lower()
        if suffix in [".dcm", ".dicom"]:
            return "dicom"
        elif suffix in [".nii", ".nii.gz"]:
            return "nifti"
        elif suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            return "image"
        return "unknown"

    def _detect_document_type(self, path: Path) -> str:
        """Detect the type of document."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        elif suffix in [".txt", ".text"]:
            return "text"
        elif suffix in [".doc", ".docx"]:
            return "word"
        elif suffix == ".hl7":
            return "hl7"
        elif suffix == ".json":
            return "fhir"
        return "document"

    def _should_use_planning(self, query: str, files: List[FileInfo]) -> bool:
        """Determine if the query requires planning."""
        # Use planning for:
        # - Multiple files
        # - Complex queries (multiple questions, comparisons)
        # - Queries requesting comprehensive analysis

        if len(files) > 1:
            return True

        complex_indicators = [
            "compare", "analyze all", "comprehensive", "review all",
            "differential", "and also", "then check", "followed by"
        ]

        return any(ind in query.lower() for ind in complex_indicators)

    async def _process_with_planning(
        self,
        query: str,
        files: List[FileInfo]
    ) -> AgentResponse:
        """Process query using the planner for multi-step execution."""
        # Create execution plan
        plan = await self._planner.create_plan(
            query=query,
            context=self._memory,
            patient_context=self._memory.patient_context,
            available_files=[f.model_dump() for f in files]
        )

        # Execute the plan
        executed_plan = await self._planner.execute_plan(plan)

        # Generate response from plan results
        response = await self._generate_response_from_plan(query, executed_plan)

        return response

    async def _process_direct(
        self,
        query: str,
        files: List[FileInfo]
    ) -> AgentResponse:
        """Process query directly without planning."""
        # If we have a model, use it
        if self._model is not None and self._processor is not None:
            response_text = await self._generate_with_model(query, files)
            return AgentResponse(message=response_text)

        # Otherwise, try to route to appropriate tool
        tool_results = []

        for file_info in files:
            tool_name = self._route_to_tool(file_info)
            if tool_name and self._registry.is_registered(tool_name):
                result = await self._registry.execute(
                    tool_name,
                    {"query": query, "file_path": file_info.path}
                )
                tool_results.append({
                    "tool": tool_name,
                    "file": file_info.name,
                    "result": result.model_dump()
                })

                # Record in memory
                self._memory.add_tool_result(
                    tool_name,
                    {"query": query, "file_path": file_info.path},
                    result.model_dump(),
                    success=result.success
                )

        if tool_results:
            response_text = self._format_tool_results(query, tool_results)
            return AgentResponse(
                message=response_text,
                tool_results=tool_results
            )

        # No files, no model - return basic response
        return AgentResponse(
            message="I apologize, but I'm currently operating without the MedGemma model "
                    "and no relevant tools could be applied to your query. "
                    "Please provide medical images or documents for analysis, "
                    "or ensure the model is properly loaded."
        )

    def _route_to_tool(self, file_info: FileInfo) -> Optional[str]:
        """Route a file to the appropriate analysis tool."""
        type_to_tool = {
            "dicom": "dicom_handler",
            "ct": "ct_analyzer",
            "mri": "mri_analyzer",
            "xray": "xray_analyzer",
            "image": "xray_analyzer",  # Default for generic images
            "pdf": "clinical_notes_parser",
            "text": "clinical_notes_parser",
            "hl7": "hl7_parser",
            "fhir": "fhir_adapter",
        }
        return type_to_tool.get(file_info.type)

    async def _generate_with_model(
        self,
        query: str,
        files: List[FileInfo]
    ) -> str:
        """Generate response using the MedGemma model."""
        import torch
        from PIL import Image

        # Prepare conversation context
        context = self._memory.get_context_for_model()

        # Load images if provided
        images = []
        for file_info in files:
            if file_info.type in ["image", "dicom"]:
                try:
                    if file_info.type == "dicom":
                        # Would need pydicom to load DICOM
                        # For now, skip DICOM images
                        continue
                    else:
                        img = Image.open(file_info.path)
                        images.append(img)
                except Exception as e:
                    logger.warning(f"Could not load image {file_info.path}: {e}")

        # Prepare inputs
        messages = context + [{"role": "user", "content": query}]

        if images:
            inputs = self._processor(
                text=str(messages),
                images=images,
                return_tensors="pt"
            )
        else:
            inputs = self._processor(
                text=str(messages),
                return_tensors="pt"
            )

        # Move to device
        device = self._config.model.get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._config.model.max_length,
                temperature=self._config.model.temperature,
                top_p=self._config.model.top_p,
                top_k=self._config.model.top_k,
                do_sample=True
            )

        # Decode
        response = self._processor.decode(outputs[0], skip_special_tokens=True)

        # Extract just the new response (after the prompt)
        # This is a simplified extraction - would need refinement
        if query in response:
            response = response.split(query)[-1].strip()

        return response

    async def _generate_response_from_plan(
        self,
        query: str,
        plan: Plan
    ) -> AgentResponse:
        """Generate a response from executed plan results."""
        # Collect all findings and recommendations
        all_findings = []
        all_recommendations = []
        tool_results = []

        for task in plan.tasks:
            if task.output:
                tool_results.append({
                    "task": task.name,
                    "tool": task.tool_name,
                    "status": task.status.value,
                    "output": task.output
                })

                data = task.output.get("data", {})
                if isinstance(data, dict):
                    if data.get("findings"):
                        all_findings.extend(data["findings"])
                    if data.get("recommendations"):
                        all_recommendations.extend(data["recommendations"])

        # Calculate average confidence
        confidences = [
            task.output.get("confidence")
            for task in plan.tasks
            if task.output and task.output.get("confidence")
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Format message
        message_parts = [f"Based on my analysis of your query: \"{query}\"\n"]

        if all_findings:
            message_parts.append("\n**Key Findings:**")
            for i, finding in enumerate(all_findings[:10], 1):  # Limit to 10
                if isinstance(finding, dict):
                    desc = finding.get("description", str(finding))
                else:
                    desc = str(finding)
                message_parts.append(f"{i}. {desc}")

        if all_recommendations:
            message_parts.append("\n**Recommendations:**")
            for i, rec in enumerate(all_recommendations[:5], 1):  # Limit to 5
                message_parts.append(f"{i}. {rec}")

        if not all_findings and not all_recommendations:
            message_parts.append(
                "\nAnalysis complete. No significant findings or specific recommendations "
                "could be generated from the available data."
            )

        return AgentResponse(
            message="\n".join(message_parts),
            findings=all_findings if all_findings else None,
            recommendations=all_recommendations if all_recommendations else None,
            tool_results=tool_results if tool_results else None,
            confidence=avg_confidence,
            plan_executed={
                "id": plan.id,
                "tasks_total": len(plan.tasks),
                "tasks_completed": sum(
                    1 for t in plan.tasks if t.status.value == "completed"
                ),
                "progress": plan.get_progress()
            }
        )

    def _format_tool_results(
        self,
        query: str,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """Format tool results into a readable response."""
        parts = [f"Here are the results for your query: \"{query}\"\n"]

        for result in tool_results:
            tool_name = result["tool"].replace("_", " ").title()
            parts.append(f"\n**{tool_name}** (File: {result.get('file', 'N/A')}):")

            output = result.get("result", {})
            if output.get("success"):
                data = output.get("data", {})
                if isinstance(data, dict):
                    if data.get("summary"):
                        parts.append(f"Summary: {data['summary']}")
                    if data.get("findings"):
                        parts.append("Findings:")
                        for finding in data["findings"][:5]:
                            parts.append(f"  - {finding}")
                else:
                    parts.append(f"Result: {data}")
            else:
                errors = output.get("errors", ["Unknown error"])
                parts.append(f"Error: {', '.join(errors)}")

        return "\n".join(parts)

    async def chat(self, message: str) -> AgentResponse:
        """
        Simple chat interface.

        Args:
            message: User message

        Returns:
            Agent response
        """
        return await self.process(message)

    async def stream_chat(
        self,
        message: str
    ) -> AsyncIterator[str]:
        """
        Stream a chat response.

        Args:
            message: User message

        Yields:
            Response tokens
        """
        # For now, just return the full response
        # In production, would implement actual streaming
        response = await self.chat(message)
        yield response.message

    def set_patient_context(self, context: PatientContext) -> None:
        """Set the patient context for the session."""
        if self._memory:
            self._memory.patient_context = context

    def get_patient_context(self) -> Optional[PatientContext]:
        """Get the current patient context."""
        return self._memory.patient_context if self._memory else None

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        if self._memory:
            self._memory.clear(keep_system=True)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        if self._memory:
            return self._memory.get_context_for_model()
        return []

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        if self._registry:
            return self._registry.list_tools()
        return []

    async def execute_tool(
        self,
        tool_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a specific tool directly.

        Args:
            tool_name: Name of the tool
            input_data: Input data for the tool

        Returns:
            Tool output
        """
        if not self._initialized:
            await self.initialize()

        result = await self._registry.execute(tool_name, input_data)
        return result.model_dump()

    async def shutdown(self) -> None:
        """Shutdown the agent and release resources."""
        logger.info("Shutting down MedGemma Agent...")

        if self._registry:
            await self._registry.shutdown()

        # Clear model from memory
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._initialized = False
        logger.info("MedGemma Agent shutdown complete")

    def __repr__(self) -> str:
        return (
            f"<MedGemmaAgent(initialized={self._initialized}, "
            f"tools={len(self.get_available_tools())})>"
        )


# Convenience function for quick usage
async def create_agent(
    config: Optional[MedGemmaConfig] = None
) -> MedGemmaAgent:
    """
    Create and initialize a MedGemma agent.

    Args:
        config: Optional configuration

    Returns:
        Initialized agent
    """
    agent = MedGemmaAgent(config)
    await agent.initialize()
    return agent
