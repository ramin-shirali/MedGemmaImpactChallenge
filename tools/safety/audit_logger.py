"""
MedGemma Agent Framework - Audit Logger Tool

HIPAA-compliant audit logging for all agent actions.
"""

from __future__ import annotations

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus
from core.config import get_config


class AuditEvent(BaseModel):
    """Audit event record."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    event_id: str = ""
    event_type: str  # access, query, modification, export
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    patient_id_hash: Optional[str] = None  # Hashed for privacy
    details: Optional[Dict[str, Any]] = None
    outcome: str = "success"  # success, failure, error
    ip_address: Optional[str] = None


class AuditLoggerInput(ToolInput):
    """Input for audit logger."""
    event_type: str = Field(description="Event type: access, query, modification, export")
    action: str = Field(description="Action performed")
    resource_type: Optional[str] = Field(default=None, description="Type of resource accessed")
    resource_id: Optional[str] = Field(default=None, description="Resource identifier")
    patient_id: Optional[str] = Field(default=None, description="Patient ID (will be hashed)")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    outcome: str = Field(default="success", description="Outcome: success, failure, error")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class AuditLoggerOutput(ToolOutput):
    """Output for audit logger."""
    event_id: str = ""
    logged: bool = False
    event: Optional[AuditEvent] = None


class AuditLoggerTool(BaseTool[AuditLoggerInput, AuditLoggerOutput]):
    """HIPAA-compliant audit logging."""

    name: ClassVar[str] = "audit_logger"
    description: ClassVar[str] = "Log all agent actions for HIPAA compliance and audit trails."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.SAFETY

    input_class: ClassVar[Type[AuditLoggerInput]] = AuditLoggerInput
    output_class: ClassVar[Type[AuditLoggerOutput]] = AuditLoggerOutput

    def __init__(self):
        super().__init__()
        self._logger = None
        self._log_file = None

    async def setup(self) -> None:
        """Initialize audit logger."""
        config = get_config()

        # Create audit log directory
        log_dir = config.logging.log_dir / "audit"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Set up file handler
        self._log_file = log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"

        # Set up logger
        self._logger = logging.getLogger("medgemma.audit")
        self._logger.setLevel(logging.INFO)

        # File handler for JSON lines
        if not self._logger.handlers:
            fh = logging.FileHandler(self._log_file)
            fh.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(fh)

        await super().setup()

    async def execute(self, input: AuditLoggerInput) -> AuditLoggerOutput:
        try:
            # Generate event ID
            event_id = self._generate_event_id()

            # Hash patient ID for privacy
            patient_id_hash = None
            if input.patient_id:
                patient_id_hash = self._hash_phi(input.patient_id)

            # Redact PHI from details
            safe_details = self._redact_phi(input.details) if input.details else None

            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                event_type=input.event_type,
                user_id=input.user_id,
                session_id=input.session_id,
                action=input.action,
                resource_type=input.resource_type,
                resource_id=input.resource_id,
                patient_id_hash=patient_id_hash,
                details=safe_details,
                outcome=input.outcome
            )

            # Log the event
            self._log_event(event)

            return AuditLoggerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"event_id": event_id},
                event_id=event_id,
                logged=True,
                event=event,
                confidence=1.0
            )

        except Exception as e:
            return AuditLoggerOutput.from_error(str(e))

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return f"AUD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def _hash_phi(self, value: str) -> str:
        """Hash PHI for storage."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def _redact_phi(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Redact potential PHI from details."""
        phi_fields = [
            'patient_name', 'name', 'dob', 'date_of_birth', 'ssn',
            'social_security', 'address', 'phone', 'email', 'mrn'
        ]

        redacted = {}
        for key, value in details.items():
            if any(phi in key.lower() for phi in phi_fields):
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_phi(value)
            else:
                redacted[key] = value

        return redacted

    def _log_event(self, event: AuditEvent) -> None:
        """Write event to audit log."""
        if self._logger:
            self._logger.info(event.model_dump_json())


async def log_audit_event(
    event_type: str,
    action: str,
    **kwargs
) -> str:
    """Convenience function to log audit events."""
    logger = AuditLoggerTool()
    await logger.setup()
    result = await logger.execute(AuditLoggerInput(
        event_type=event_type,
        action=action,
        **kwargs
    ))
    return result.event_id
