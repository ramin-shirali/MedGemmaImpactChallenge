"""
MedGemma Agent Framework - Patient Timeline Tool

Creates unified patient timeline from multiple data sources.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class TimelineEvent(BaseModel):
    """A single event in patient timeline."""
    date: str
    event_type: str  # encounter, diagnosis, medication, lab, procedure, imaging
    title: str
    description: Optional[str] = None
    provider: Optional[str] = None
    location: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class PatientTimelineInput(ToolInput):
    """Input for patient timeline."""
    patient_id: str = Field(description="Patient identifier")
    events: List[Dict[str, Any]] = Field(description="Raw events to process")
    start_date: Optional[str] = Field(default=None, description="Filter start date")
    end_date: Optional[str] = Field(default=None, description="Filter end date")
    event_types: Optional[List[str]] = Field(default=None, description="Filter by event types")


class PatientTimelineOutput(ToolOutput):
    """Output for patient timeline."""
    patient_id: str = ""
    timeline: List[TimelineEvent] = Field(default_factory=list)
    event_count: int = 0
    date_range: Optional[Dict[str, str]] = None
    summary: Optional[str] = None


class PatientTimelineTool(BaseTool[PatientTimelineInput, PatientTimelineOutput]):
    """Create unified patient timeline."""

    name: ClassVar[str] = "patient_timeline"
    description: ClassVar[str] = "Create unified patient timeline from encounters, diagnoses, medications, labs, and procedures."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.INTEGRATION

    input_class: ClassVar[Type[PatientTimelineInput]] = PatientTimelineInput
    output_class: ClassVar[Type[PatientTimelineOutput]] = PatientTimelineOutput

    async def execute(self, input: PatientTimelineInput) -> PatientTimelineOutput:
        try:
            # Process events
            timeline = []
            for event_data in input.events:
                event = self._process_event(event_data)
                if event:
                    timeline.append(event)

            # Apply filters
            if input.event_types:
                timeline = [e for e in timeline if e.event_type in input.event_types]

            if input.start_date:
                timeline = [e for e in timeline if e.date >= input.start_date]

            if input.end_date:
                timeline = [e for e in timeline if e.date <= input.end_date]

            # Sort by date
            timeline.sort(key=lambda x: x.date, reverse=True)

            # Calculate date range
            date_range = None
            if timeline:
                dates = [e.date for e in timeline]
                date_range = {"start": min(dates), "end": max(dates)}

            # Generate summary
            summary = self._generate_summary(timeline)

            return PatientTimelineOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"event_count": len(timeline)},
                patient_id=input.patient_id,
                timeline=timeline,
                event_count=len(timeline),
                date_range=date_range,
                summary=summary,
                confidence=0.9
            )

        except Exception as e:
            return PatientTimelineOutput.from_error(str(e))

    def _process_event(self, event_data: Dict) -> Optional[TimelineEvent]:
        """Process raw event data into TimelineEvent."""
        # Determine event type and extract relevant info
        event_type = event_data.get("type", event_data.get("event_type", "unknown"))
        date = event_data.get("date", event_data.get("effectiveDateTime", event_data.get("recordedDate", "")))

        if not date:
            return None

        # Normalize date format
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")
        elif 'T' in str(date):
            date = str(date).split('T')[0]

        title = event_data.get("title", event_data.get("display", event_data.get("description", "Event")))
        description = event_data.get("description", event_data.get("text", ""))

        return TimelineEvent(
            date=date,
            event_type=self._normalize_event_type(event_type),
            title=title,
            description=description,
            provider=event_data.get("provider"),
            location=event_data.get("location"),
            details=event_data.get("details"),
            source=event_data.get("source")
        )

    def _normalize_event_type(self, event_type: str) -> str:
        """Normalize event type string."""
        type_map = {
            "Encounter": "encounter",
            "Condition": "diagnosis",
            "MedicationRequest": "medication",
            "Observation": "lab",
            "Procedure": "procedure",
            "DiagnosticReport": "imaging",
            "ImagingStudy": "imaging",
        }
        return type_map.get(event_type, event_type.lower())

    def _generate_summary(self, timeline: List[TimelineEvent]) -> str:
        """Generate timeline summary."""
        if not timeline:
            return "No events found."

        # Count by type
        type_counts: Dict[str, int] = {}
        for event in timeline:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1

        parts = [f"Timeline contains {len(timeline)} events:"]
        for event_type, count in sorted(type_counts.items()):
            parts.append(f"  - {event_type}: {count}")

        # Recent events
        recent = timeline[:3]
        if recent:
            parts.append("\nRecent events:")
            for event in recent:
                parts.append(f"  - {event.date}: {event.title}")

        return "\n".join(parts)
