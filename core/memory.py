"""
MedGemma Agent Framework - Memory and Context Management

This module handles conversation history and patient context:
- Conversation memory with sliding window
- Patient context (demographics, history, current session)
- Session state management
- Context serialization/deserialization
- Memory summarization for long conversations

Usage:
    from core.memory import ConversationMemory, PatientContext

    # Create conversation memory
    memory = ConversationMemory(max_turns=50)
    memory.add_user_message("What does this CT scan show?")
    memory.add_assistant_message("The CT scan shows...")
    memory.add_tool_result("ct_analyzer", {"findings": [...]})

    # Create patient context
    patient = PatientContext(
        patient_id="P123",
        demographics={"age": 45, "sex": "M"},
        chief_complaint="Chest pain"
    )

    # Get formatted context for model
    context = memory.get_context_for_model()
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from core.config import get_config, MedGemmaConfig


class MessageRole(str, Enum):
    """Role of a message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    # For tool messages
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None


class ToolCall(BaseModel):
    """Record of a tool invocation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PatientDemographics(BaseModel):
    """Patient demographic information."""

    age: Optional[int] = Field(default=None, ge=0, le=150)
    sex: Optional[str] = Field(default=None)
    gender: Optional[str] = Field(default=None)
    ethnicity: Optional[str] = Field(default=None)
    weight_kg: Optional[float] = Field(default=None, ge=0)
    height_cm: Optional[float] = Field(default=None, ge=0)
    bmi: Optional[float] = Field(default=None, ge=0)


class MedicalHistory(BaseModel):
    """Patient medical history."""

    conditions: List[str] = Field(default_factory=list)
    surgeries: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    social_history: Optional[Dict[str, Any]] = None
    immunizations: List[str] = Field(default_factory=list)


class VitalSigns(BaseModel):
    """Patient vital signs."""

    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    respiratory_rate: Optional[int] = None
    temperature_celsius: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    pain_score: Optional[int] = Field(default=None, ge=0, le=10)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PatientContext(BaseModel):
    """
    Patient context for the current session.

    Contains all relevant patient information that tools may need.
    Can be incrementally built as information is gathered.
    """

    # Identifiers
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Demographics
    demographics: PatientDemographics = Field(
        default_factory=PatientDemographics
    )

    # Medical history
    medical_history: MedicalHistory = Field(
        default_factory=MedicalHistory
    )

    # Current encounter
    chief_complaint: Optional[str] = None
    history_of_present_illness: Optional[str] = None
    vital_signs: Optional[VitalSigns] = None

    # Working diagnosis and plan
    working_diagnoses: List[str] = Field(default_factory=list)
    differential_diagnoses: List[str] = Field(default_factory=list)
    active_problems: List[str] = Field(default_factory=list)

    # Session data
    uploaded_files: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_results: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def update(self, **kwargs) -> None:
        """Update context fields and timestamp."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()

    def add_file(
        self,
        file_path: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an uploaded file."""
        self.uploaded_files.append({
            "path": file_path,
            "type": file_type,
            "metadata": metadata or {},
            "uploaded_at": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()

    def add_analysis_result(
        self,
        tool_name: str,
        result: Dict[str, Any]
    ) -> None:
        """Record an analysis result."""
        self.analysis_results.append({
            "tool": tool_name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()

    def add_entities(self, entity_type: str, entities: List[str]) -> None:
        """Add extracted entities."""
        if entity_type not in self.extracted_entities:
            self.extracted_entities[entity_type] = []
        self.extracted_entities[entity_type].extend(entities)
        self.updated_at = datetime.utcnow()

    def get_summary(self) -> str:
        """Get a text summary of the patient context."""
        parts = []

        if self.demographics.age or self.demographics.sex:
            demo = []
            if self.demographics.age:
                demo.append(f"{self.demographics.age}yo")
            if self.demographics.sex:
                demo.append(self.demographics.sex)
            parts.append(" ".join(demo))

        if self.chief_complaint:
            parts.append(f"CC: {self.chief_complaint}")

        if self.medical_history.conditions:
            parts.append(f"PMH: {', '.join(self.medical_history.conditions[:5])}")

        if self.medical_history.medications:
            parts.append(f"Meds: {', '.join(self.medical_history.medications[:5])}")

        if self.medical_history.allergies:
            parts.append(f"Allergies: {', '.join(self.medical_history.allergies)}")

        if self.working_diagnoses:
            parts.append(f"Working Dx: {', '.join(self.working_diagnoses[:3])}")

        return " | ".join(parts) if parts else "No patient context available"

    def to_prompt_context(self) -> str:
        """Format patient context for inclusion in model prompt."""
        lines = ["## Patient Context"]

        if self.demographics.age or self.demographics.sex:
            demo_parts = []
            if self.demographics.age:
                demo_parts.append(f"Age: {self.demographics.age}")
            if self.demographics.sex:
                demo_parts.append(f"Sex: {self.demographics.sex}")
            lines.append(f"Demographics: {', '.join(demo_parts)}")

        if self.chief_complaint:
            lines.append(f"Chief Complaint: {self.chief_complaint}")

        if self.history_of_present_illness:
            lines.append(f"HPI: {self.history_of_present_illness}")

        if self.vital_signs:
            vitals = []
            if self.vital_signs.blood_pressure_systolic:
                vitals.append(
                    f"BP: {self.vital_signs.blood_pressure_systolic}/"
                    f"{self.vital_signs.blood_pressure_diastolic}"
                )
            if self.vital_signs.heart_rate:
                vitals.append(f"HR: {self.vital_signs.heart_rate}")
            if self.vital_signs.temperature_celsius:
                vitals.append(f"Temp: {self.vital_signs.temperature_celsius}°C")
            if self.vital_signs.oxygen_saturation:
                vitals.append(f"SpO2: {self.vital_signs.oxygen_saturation}%")
            if vitals:
                lines.append(f"Vitals: {', '.join(vitals)}")

        if self.medical_history.conditions:
            lines.append(f"Medical History: {', '.join(self.medical_history.conditions)}")

        if self.medical_history.medications:
            lines.append(f"Medications: {', '.join(self.medical_history.medications)}")

        if self.medical_history.allergies:
            lines.append(f"Allergies: {', '.join(self.medical_history.allergies)}")

        if self.working_diagnoses:
            lines.append(f"Working Diagnoses: {', '.join(self.working_diagnoses)}")

        if self.differential_diagnoses:
            lines.append(f"Differential: {', '.join(self.differential_diagnoses)}")

        return "\n".join(lines)


class ConversationMemory:
    """
    Manages conversation history for the agent.

    Features:
    - Sliding window for long conversations
    - Message summarization
    - Tool call tracking
    - Context formatting for model
    """

    def __init__(
        self,
        config: Optional[MedGemmaConfig] = None,
        max_turns: int = 50,
        max_tokens_estimate: int = 8000,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize conversation memory.

        Args:
            config: Optional configuration
            max_turns: Maximum conversation turns to keep
            max_tokens_estimate: Approximate token limit for context
            system_prompt: System prompt for the conversation
        """
        self._config = config or get_config()
        self._max_turns = max_turns
        self._max_tokens_estimate = max_tokens_estimate
        self._messages: List[Message] = []
        self._tool_calls: List[ToolCall] = []
        self._patient_context: Optional[PatientContext] = None
        self._summary: Optional[str] = None
        self._session_id = str(uuid.uuid4())

        # Add system prompt if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM, system_prompt)

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def patient_context(self) -> Optional[PatientContext]:
        """Get the patient context."""
        return self._patient_context

    @patient_context.setter
    def patient_context(self, context: PatientContext) -> None:
        """Set the patient context."""
        self._patient_context = context

    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to the conversation.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata

        Returns:
            The created message
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata
        )
        self._messages.append(message)
        self._maybe_truncate()
        return message

    def add_user_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a user message."""
        return self.add_message(MessageRole.USER, content, metadata)

    def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add an assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, metadata)

    def add_tool_result(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> ToolCall:
        """
        Record a tool call and add to conversation.

        Args:
            tool_name: Name of the tool
            input_data: Input to the tool
            output_data: Output from the tool
            success: Whether the call succeeded
            error: Error message if failed
            execution_time_ms: Execution time

        Returns:
            The created ToolCall
        """
        tool_call = ToolCall(
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            success=success,
            error=error,
            execution_time_ms=execution_time_ms
        )
        self._tool_calls.append(tool_call)

        # Add as message
        if success:
            content = f"Tool '{tool_name}' executed successfully."
            if output_data.get("summary"):
                content += f"\nSummary: {output_data['summary']}"
        else:
            content = f"Tool '{tool_name}' failed: {error}"

        message = Message(
            role=MessageRole.TOOL,
            content=content,
            tool_name=tool_name,
            tool_input=input_data,
            tool_output=output_data
        )
        self._messages.append(message)

        return tool_call

    def get_messages(
        self,
        include_system: bool = True,
        include_tools: bool = True,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get conversation messages.

        Args:
            include_system: Include system messages
            include_tools: Include tool messages
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        messages = self._messages

        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if not include_tools:
            messages = [m for m in messages if m.role != MessageRole.TOOL]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_tool_calls(self, tool_name: Optional[str] = None) -> List[ToolCall]:
        """
        Get tool calls, optionally filtered by tool name.

        Args:
            tool_name: Optional tool name to filter by

        Returns:
            List of tool calls
        """
        if tool_name:
            return [tc for tc in self._tool_calls if tc.tool_name == tool_name]
        return self._tool_calls

    def get_context_for_model(
        self,
        include_patient_context: bool = True,
        include_summary: bool = True,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for the model.

        Args:
            include_patient_context: Include patient context
            include_summary: Include conversation summary if available
            max_messages: Maximum messages to include

        Returns:
            List of message dicts with 'role' and 'content'
        """
        context = []

        # Add system message with patient context
        system_content_parts = []

        # Get original system message if exists
        system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        if system_messages:
            system_content_parts.append(system_messages[0].content)

        # Add patient context
        if include_patient_context and self._patient_context:
            system_content_parts.append(self._patient_context.to_prompt_context())

        # Add conversation summary
        if include_summary and self._summary:
            system_content_parts.append(f"## Previous Conversation Summary\n{self._summary}")

        if system_content_parts:
            context.append({
                "role": "system",
                "content": "\n\n".join(system_content_parts)
            })

        # Add conversation messages
        messages = [m for m in self._messages if m.role != MessageRole.SYSTEM]

        if max_messages:
            messages = messages[-max_messages:]

        for message in messages:
            role = message.role.value
            if role == "tool":
                # Format tool messages as assistant messages for most models
                role = "assistant"

            context.append({
                "role": role,
                "content": message.content
            })

        return context

    def _maybe_truncate(self) -> None:
        """Truncate conversation if it exceeds limits."""
        # Keep system messages
        system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in self._messages if m.role != MessageRole.SYSTEM]

        if len(other_messages) > self._max_turns:
            # Summarize older messages
            messages_to_summarize = other_messages[:-self._max_turns]
            self._update_summary(messages_to_summarize)

            # Keep recent messages
            self._messages = system_messages + other_messages[-self._max_turns:]

    def _update_summary(self, messages: List[Message]) -> None:
        """Update the conversation summary with older messages."""
        # Simple summary - in production, use model to summarize
        summary_parts = []

        if self._summary:
            summary_parts.append(self._summary)

        for msg in messages:
            if msg.role == MessageRole.USER:
                summary_parts.append(f"User asked about: {msg.content[:100]}...")
            elif msg.role == MessageRole.ASSISTANT:
                summary_parts.append(f"Assistant responded about: {msg.content[:100]}...")
            elif msg.role == MessageRole.TOOL:
                summary_parts.append(f"Tool '{msg.tool_name}' was used")

        self._summary = "\n".join(summary_parts[-10:])  # Keep last 10 summary items

    def clear(self, keep_system: bool = True) -> None:
        """
        Clear conversation history.

        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            self._messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        else:
            self._messages = []

        self._tool_calls = []
        self._summary = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "session_id": self._session_id,
            "messages": [m.model_dump() for m in self._messages],
            "tool_calls": [tc.model_dump() for tc in self._tool_calls],
            "patient_context": (
                self._patient_context.model_dump()
                if self._patient_context else None
            ),
            "summary": self._summary
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config: Optional[MedGemmaConfig] = None
    ) -> "ConversationMemory":
        """Deserialize memory from dictionary."""
        memory = cls(config=config)
        memory._session_id = data.get("session_id", str(uuid.uuid4()))

        for msg_data in data.get("messages", []):
            memory._messages.append(Message.model_validate(msg_data))

        for tc_data in data.get("tool_calls", []):
            memory._tool_calls.append(ToolCall.model_validate(tc_data))

        if data.get("patient_context"):
            memory._patient_context = PatientContext.model_validate(
                data["patient_context"]
            )

        memory._summary = data.get("summary")

        return memory

    def save(self, file_path: str) -> None:
        """Save memory to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(
        cls,
        file_path: str,
        config: Optional[MedGemmaConfig] = None
    ) -> "ConversationMemory":
        """Load memory from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, config)

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"<ConversationMemory(session_id='{self._session_id}', "
            f"messages={len(self._messages)}, tool_calls={len(self._tool_calls)})>"
        )
