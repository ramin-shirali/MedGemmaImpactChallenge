"""
MedGemma Agent Framework - Prescription Parser Tool

Parses prescriptions and medication lists to extract structured drug information.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, DocumentInput, ToolCategory, ToolOutput, ToolStatus


class PrescriptionParserInput(DocumentInput):
    """Input for prescription parser."""
    validate_doses: bool = Field(default=True, description="Validate dose ranges")


class Medication(BaseModel):
    """Parsed medication details."""
    name: str
    dose: Optional[str] = None
    unit: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    duration: Optional[str] = None
    quantity: Optional[int] = None
    refills: Optional[int] = None
    instructions: Optional[str] = None
    rxnorm_code: Optional[str] = None


class PrescriptionParserOutput(ToolOutput):
    """Output for prescription parser."""
    medications: List[Medication] = Field(default_factory=list)
    prescriber: Optional[str] = None
    prescription_date: Optional[str] = None
    pharmacy: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class PrescriptionParserTool(BaseTool[PrescriptionParserInput, PrescriptionParserOutput]):
    """Parse prescriptions for medication details."""

    name: ClassVar[str] = "prescription_parser"
    description: ClassVar[str] = "Parse prescriptions to extract medication names, doses, frequencies, and instructions."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENTS

    input_class: ClassVar[Type[PrescriptionParserInput]] = PrescriptionParserInput
    output_class: ClassVar[Type[PrescriptionParserOutput]] = PrescriptionParserOutput

    async def execute(self, input: PrescriptionParserInput) -> PrescriptionParserOutput:
        try:
            text = input.document_text or ""
            if input.document_path:
                with open(input.document_path, 'r') as f:
                    text = f.read()

            medications = self._parse_medications(text)
            warnings = self._validate_medications(medications) if input.validate_doses else []

            return PrescriptionParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"medication_count": len(medications)},
                medications=medications,
                prescriber=self._extract_prescriber(text),
                prescription_date=self._extract_date(text),
                warnings=warnings,
                confidence=0.85
            )
        except Exception as e:
            return PrescriptionParserOutput.from_error(str(e))

    def _parse_medications(self, text: str) -> List[Medication]:
        import re
        medications = []

        # Pattern for medication lines
        patterns = [
            r'([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\s*(?:,?\s*([\w\s]+?))?(?:,?\s*#(\d+))?(?:,?\s*(?:refills?:?\s*)?(\d+))?',
            r'(\w+)\s+(\d+)\s*(mg|mcg)\s+(po|iv|sq|im)\s+(daily|bid|tid|qid|prn|qhs)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                med = Medication(
                    name=match[0].strip(),
                    dose=match[1] if len(match) > 1 else None,
                    unit=match[2] if len(match) > 2 else None,
                    frequency=match[3] if len(match) > 3 else None,
                    quantity=int(match[4]) if len(match) > 4 and match[4] else None,
                    refills=int(match[5]) if len(match) > 5 and match[5] else None,
                )
                medications.append(med)

        # Deduplicate by name
        seen = set()
        unique = []
        for med in medications:
            if med.name.lower() not in seen:
                seen.add(med.name.lower())
                unique.append(med)

        return unique

    def _validate_medications(self, medications: List[Medication]) -> List[str]:
        warnings = []
        # Common high-alert medications
        high_alert = ['warfarin', 'insulin', 'heparin', 'methotrexate', 'opioid']

        for med in medications:
            for alert in high_alert:
                if alert in med.name.lower():
                    warnings.append(f"High-alert medication: {med.name}")
                    break

        return warnings

    def _extract_prescriber(self, text: str) -> Optional[str]:
        import re
        patterns = [r'(?:Dr\.|MD|DO|NP|PA)[\s,]+([A-Z][a-z]+\s+[A-Z][a-z]+)', r'Prescriber:\s*(.+?)(?:\n|$)']
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'(?:Date|Prescribed):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
        return match.group(1) if match else None
