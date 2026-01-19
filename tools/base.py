"""
MedGemma Agent Framework - Base Tool Interface

This module defines the base interface that all tools in the framework must implement.
It provides:
- Abstract base class for tools
- Standard input/output schemas using Pydantic
- Validation and error handling utilities
- Tool metadata and documentation structure

Every tool module in the framework inherits from BaseTool and implements:
- Input/output schemas (Pydantic models)
- execute() method for tool logic
- Metadata (name, description, version)

Usage:
    from tools.base import BaseTool, ToolInput, ToolOutput

    class MyToolInput(ToolInput):
        query: str

    class MyToolOutput(ToolOutput):
        result: str

    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"

        async def execute(self, input: MyToolInput) -> MyToolOutput:
            # Implementation
            return MyToolOutput(success=True, data={"result": "done"})
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, field_validator


class ToolCategory(str, Enum):
    """Categories of tools in the framework."""
    IMAGING = "imaging"
    DOCUMENTS = "documents"
    KNOWLEDGE = "knowledge"
    CLINICAL = "clinical"
    INTEGRATION = "integration"
    SAFETY = "safety"
    UTILITIES = "utilities"


class ToolStatus(str, Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ConfidenceLevel(str, Enum):
    """Qualitative confidence levels."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ToolInput(BaseModel):
    """
    Base class for tool input schemas.

    All tool-specific input classes should inherit from this.
    Override with specific fields for each tool.
    """

    class Config:
        extra = "forbid"  # Reject unknown fields

    def to_prompt(self) -> str:
        """Convert input to a prompt string for the model."""
        return str(self.model_dump())


class ToolOutput(BaseModel):
    """
    Base class for tool output schemas.

    All tool-specific output classes should inherit from this.
    Provides standard fields for success status, data, confidence, and errors.
    """

    success: bool = Field(
        description="Whether the tool execution was successful"
    )
    status: ToolStatus = Field(
        default=ToolStatus.SUCCESS,
        description="Detailed status of execution"
    )
    data: Any = Field(
        default=None,
        description="The main output data from the tool"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0) for the output"
    )
    confidence_level: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Qualitative confidence level"
    )
    errors: Optional[List[str]] = Field(
        default=None,
        description="List of error messages if any"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="List of warning messages if any"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the execution"
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Execution time in milliseconds"
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Model used for inference if applicable"
    )

    @classmethod
    def from_error(
        cls,
        error: str,
        status: ToolStatus = ToolStatus.FAILURE
    ) -> "ToolOutput":
        """Create an error output."""
        return cls(
            success=False,
            status=status,
            data=None,
            errors=[error]
        )

    @classmethod
    def from_errors(
        cls,
        errors: List[str],
        status: ToolStatus = ToolStatus.FAILURE
    ) -> "ToolOutput":
        """Create an output with multiple errors."""
        return cls(
            success=False,
            status=status,
            data=None,
            errors=errors
        )

    def get_confidence_level(self) -> ConfidenceLevel:
        """Get qualitative confidence level from numeric confidence."""
        if self.confidence_level:
            return self.confidence_level
        if self.confidence is None:
            return ConfidenceLevel.MEDIUM

        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        if self.errors is None:
            self.errors = []
        self.errors.append(error)
        self.success = False


# Type variables for generic typing
TInput = TypeVar("TInput", bound=ToolInput)
TOutput = TypeVar("TOutput", bound=ToolOutput)


class ToolMetadata(BaseModel):
    """Metadata about a tool for documentation and discovery."""

    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="Human-readable description")
    version: str = Field(default="1.0.0", description="Tool version")
    category: ToolCategory = Field(description="Tool category")
    author: Optional[str] = Field(default=None, description="Tool author")
    requires_model: bool = Field(
        default=True,
        description="Whether tool requires MedGemma model"
    )
    requires_gpu: bool = Field(
        default=False,
        description="Whether tool requires GPU"
    )
    input_types: List[str] = Field(
        default_factory=list,
        description="Accepted input types (e.g., 'image', 'text', 'dicom')"
    )
    output_types: List[str] = Field(
        default_factory=list,
        description="Output types produced"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for discovery"
    )


class BaseTool(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for all tools in the MedGemma Agent Framework.

    Tools must implement:
    - name: Unique identifier for the tool
    - description: Human-readable description
    - execute(): Main tool logic
    - get_input_schema(): JSON schema for inputs
    - get_output_schema(): JSON schema for outputs

    Optional overrides:
    - version: Tool version (default "1.0.0")
    - category: Tool category
    - validate_input(): Custom input validation
    - setup(): Async initialization
    - teardown(): Async cleanup
    """

    # Class-level attributes (override in subclasses)
    name: ClassVar[str] = "base_tool"
    description: ClassVar[str] = "Base tool - override in subclass"
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.UTILITIES

    # Type hints for input/output classes (override in subclasses)
    input_class: ClassVar[Type[ToolInput]] = ToolInput
    output_class: ClassVar[Type[ToolOutput]] = ToolOutput

    def __init__(self) -> None:
        """Initialize the tool."""
        self._initialized = False
        self._model = None
        self._config = None

    async def setup(self) -> None:
        """
        Async initialization - load models, resources, etc.

        Override in subclass if needed.
        Called once before first execute().
        """
        self._initialized = True

    async def teardown(self) -> None:
        """
        Async cleanup - release resources.

        Override in subclass if needed.
        Called when tool is unregistered or on shutdown.
        """
        self._initialized = False

    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata for documentation and discovery."""
        return ToolMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            category=self.category,
            requires_model=True,
            requires_gpu=False,
            input_types=[],
            output_types=[],
            tags=[],
        )

    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for inputs."""
        return self.input_class.model_json_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        """Return JSON schema for outputs."""
        return self.output_class.model_json_schema()

    def validate_input(self, input_data: Dict[str, Any]) -> TInput:
        """
        Validate and parse input data.

        Args:
            input_data: Raw input dictionary

        Returns:
            Validated input object

        Raises:
            ValueError: If validation fails
        """
        try:
            return self.input_class.model_validate(input_data)
        except Exception as e:
            raise ValueError(f"Input validation failed: {e}")

    @abstractmethod
    async def execute(self, input: TInput) -> TOutput:
        """
        Execute the tool with validated input.

        Args:
            input: Validated input object

        Returns:
            Tool output object

        This is the main method to override in tool implementations.
        """
        pass

    async def run(
        self,
        input_data: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> TOutput:
        """
        Run the tool with automatic validation, timing, and error handling.

        Args:
            input_data: Raw input dictionary
            timeout: Optional timeout in seconds

        Returns:
            Tool output with execution metadata
        """
        start_time = time.time()

        try:
            # Ensure setup has been called
            if not self._initialized:
                await self.setup()

            # Validate input
            validated_input = self.validate_input(input_data)

            # Execute with optional timeout
            if timeout:
                output = await asyncio.wait_for(
                    self.execute(validated_input),
                    timeout=timeout
                )
            else:
                output = await self.execute(validated_input)

            # Add execution metadata
            execution_time_ms = (time.time() - start_time) * 1000
            if isinstance(output, ToolOutput):
                output.execution_time_ms = execution_time_ms

            return output

        except asyncio.TimeoutError:
            return self.output_class.from_error(
                f"Tool execution timed out after {timeout}s",
                status=ToolStatus.TIMEOUT
            )
        except ValueError as e:
            return self.output_class.from_error(str(e))
        except Exception as e:
            return self.output_class.from_error(
                f"Tool execution failed: {type(e).__name__}: {e}"
            )

    def to_function_spec(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI-style function specification.

        Useful for integration with LLM function calling.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_input_schema(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"


class ImageInput(ToolInput):
    """Common input schema for image-based tools."""

    image_path: Optional[str] = Field(
        default=None,
        description="Path to image file"
    )
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw image bytes"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL to image"
    )

    @field_validator("image_path", "image_bytes", "image_base64", "image_url")
    @classmethod
    def at_least_one_image_source(cls, v, info):
        """Ensure at least one image source is provided."""
        # This validation happens per-field, so we check in model_validator
        return v

    def model_post_init(self, __context):
        """Validate that at least one image source is provided."""
        sources = [
            self.image_path,
            self.image_bytes,
            self.image_base64,
            self.image_url
        ]
        if not any(sources):
            raise ValueError("At least one image source must be provided")


class TextInput(ToolInput):
    """Common input schema for text-based tools."""

    text: str = Field(
        description="Input text to process"
    )
    language: str = Field(
        default="en",
        description="Language code (ISO 639-1)"
    )


class DocumentInput(ToolInput):
    """Common input schema for document-based tools."""

    document_path: Optional[str] = Field(
        default=None,
        description="Path to document file"
    )
    document_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw document bytes"
    )
    document_text: Optional[str] = Field(
        default=None,
        description="Extracted document text"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Document type (pdf, docx, etc.)"
    )

    def model_post_init(self, __context):
        """Validate that at least one document source is provided."""
        sources = [
            self.document_path,
            self.document_bytes,
            self.document_text
        ]
        if not any(sources):
            raise ValueError("At least one document source must be provided")


class Finding(BaseModel):
    """A medical finding from analysis."""

    description: str = Field(description="Description of the finding")
    location: Optional[str] = Field(
        default=None,
        description="Anatomical location"
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity level (mild, moderate, severe)"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    evidence: Optional[List[str]] = Field(
        default=None,
        description="Supporting evidence"
    )
    icd_codes: Optional[List[str]] = Field(
        default=None,
        description="Related ICD-10 codes"
    )


class AnalysisOutput(ToolOutput):
    """Common output schema for analysis tools."""

    findings: List[Finding] = Field(
        default_factory=list,
        description="List of findings"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary of analysis"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Recommendations based on findings"
    )
    differential_diagnoses: Optional[List[str]] = Field(
        default=None,
        description="Possible diagnoses"
    )


def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """
    Decorator to register a tool class with the global registry.

    Usage:
        @register_tool
        class MyTool(BaseTool):
            ...
    """
    # Import here to avoid circular imports
    from core.registry import get_registry
    registry = get_registry()
    registry.register(cls)
    return cls
