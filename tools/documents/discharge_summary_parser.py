"""
MedGemma Agent Framework - Discharge Summary Parser Tool

Parses hospital discharge summaries to extract diagnoses, medications,
and follow-up instructions.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, DocumentInput, ToolCategory, ToolOutput, ToolStatus


class DischargeSummaryParserInput(DocumentInput):
    """Input for discharge summary parser."""
    extract_reconciliation: bool = Field(default=True, description="Extract medication reconciliation")


class DischargeSummaryParserOutput(ToolOutput):
    """Output for discharge summary parser."""
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    length_of_stay: Optional[int] = None
    admitting_diagnosis: Optional[str] = None
    discharge_diagnoses: List[str] = Field(default_factory=list)
    procedures_performed: List[str] = Field(default_factory=list)
    discharge_medications: List[Dict[str, str]] = Field(default_factory=list)
    follow_up_appointments: List[str] = Field(default_factory=list)
    discharge_instructions: Optional[str] = None
    discharge_condition: Optional[str] = None
    disposition: Optional[str] = None


class DischargeSummaryParserTool(BaseTool[DischargeSummaryParserInput, DischargeSummaryParserOutput]):
    """Parse discharge summaries for key information."""

    name: ClassVar[str] = "discharge_summary_parser"
    description: ClassVar[str] = "Parse discharge summaries to extract diagnoses, medications, and follow-up care."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENTS

    input_class: ClassVar[Type[DischargeSummaryParserInput]] = DischargeSummaryParserInput
    output_class: ClassVar[Type[DischargeSummaryParserOutput]] = DischargeSummaryParserOutput

    async def execute(self, input: DischargeSummaryParserInput) -> DischargeSummaryParserOutput:
        try:
            text = input.document_text or ""
            if input.document_path:
                with open(input.document_path, 'r') as f:
                    text = f.read()

            return DischargeSummaryParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"parsed": True},
                admission_date=self._extract_date(text, "admission"),
                discharge_date=self._extract_date(text, "discharge"),
                discharge_diagnoses=self._extract_diagnoses(text),
                procedures_performed=self._extract_procedures(text),
                discharge_medications=self._extract_medications(text),
                follow_up_appointments=self._extract_followup(text),
                discharge_instructions=self._extract_section(text, "INSTRUCTIONS"),
                discharge_condition=self._extract_condition(text),
                disposition=self._extract_disposition(text),
                confidence=0.8
            )
        except Exception as e:
            return DischargeSummaryParserOutput.from_error(str(e))

    def _extract_date(self, text: str, date_type: str) -> Optional[str]:
        import re
        pattern = rf'{date_type}\s*date[:\s]*(\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_diagnoses(self, text: str) -> List[str]:
        import re
        diagnoses = []
        match = re.search(r'DISCHARGE DIAGNOS[IE]S?:?\s*(.+?)(?=\n[A-Z]{2,}:|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            items = re.findall(r'\d+\.\s*(.+?)(?=\n|$)', match.group(1))
            diagnoses.extend(items)
        return diagnoses[:10]

    def _extract_procedures(self, text: str) -> List[str]:
        import re
        procedures = []
        match = re.search(r'PROCEDURES?:?\s*(.+?)(?=\n[A-Z]{2,}:|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            items = re.findall(r'(?:\d+\.)?\s*(.+?)(?=\n|$)', match.group(1))
            procedures.extend([p.strip() for p in items if p.strip()])
        return procedures[:10]

    def _extract_medications(self, text: str) -> List[Dict[str, str]]:
        import re
        medications = []
        match = re.search(r'DISCHARGE MEDICATION[S]?:?\s*(.+?)(?=\n[A-Z]{2,}:|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            lines = match.group(1).split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 3:
                    medications.append({"name": line.split()[0] if line.split() else line, "instructions": line})
        return medications[:20]

    def _extract_followup(self, text: str) -> List[str]:
        import re
        followup = []
        match = re.search(r'FOLLOW[- ]?UP:?\s*(.+?)(?=\n[A-Z]{2,}:|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            items = match.group(1).split('\n')
            followup.extend([item.strip() for item in items if item.strip()])
        return followup[:5]

    def _extract_section(self, text: str, section: str) -> Optional[str]:
        import re
        match = re.search(rf'{section}:?\s*(.+?)(?=\n[A-Z]{{2,}}:|$)', text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_condition(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'CONDITION\s*(?:AT|ON)\s*DISCHARGE:?\s*(\w+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_disposition(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'DISPOSITION:?\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None
