"""
MedGemma Agent Framework - Clinical Notes Parser Tool

Parses clinical notes (SOAP notes, progress notes, H&P) to extract
structured medical information.

Usage:
    parser = ClinicalNotesParserTool()
    result = await parser.run({
        "document_text": "CHIEF COMPLAINT: Chest pain...",
        "note_type": "H&P"
    })
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import (
    BaseTool,
    DocumentInput,
    ToolCategory,
    ToolOutput,
    ToolStatus,
)


class ClinicalNotesParserInput(DocumentInput):
    """Input schema for clinical notes parser."""

    note_type: Optional[str] = Field(
        default=None,
        description="Note type (SOAP, H&P, progress, consult, procedure)"
    )
    extract_entities: bool = Field(
        default=True,
        description="Extract medical entities"
    )
    extract_medications: bool = Field(
        default=True,
        description="Extract medication mentions"
    )


class SOAPNote(BaseModel):
    """SOAP note structure."""

    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None


class HPNote(BaseModel):
    """History and Physical structure."""

    chief_complaint: Optional[str] = None
    hpi: Optional[str] = None
    past_medical_history: Optional[List[str]] = None
    past_surgical_history: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    review_of_systems: Optional[Dict[str, str]] = None
    physical_exam: Optional[Dict[str, str]] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None


class ClinicalNotesParserOutput(ToolOutput):
    """Output schema for clinical notes parser."""

    note_type_detected: Optional[str] = None
    soap_note: Optional[SOAPNote] = None
    hp_note: Optional[HPNote] = None
    diagnoses: List[str] = Field(default_factory=list)
    problems: List[str] = Field(default_factory=list)
    medications_mentioned: List[str] = Field(default_factory=list)
    procedures_mentioned: List[str] = Field(default_factory=list)
    extracted_entities: Optional[Dict[str, List[str]]] = None
    summary: Optional[str] = None


class ClinicalNotesParserTool(BaseTool[ClinicalNotesParserInput, ClinicalNotesParserOutput]):
    """
    Tool for parsing clinical notes.

    Supports various note types:
    - SOAP notes
    - History and Physical (H&P)
    - Progress notes
    - Consultation notes
    """

    name: ClassVar[str] = "clinical_notes_parser"
    description: ClassVar[str] = (
        "Parse clinical notes (SOAP, H&P, progress notes) to extract "
        "structured medical information."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENTS

    input_class: ClassVar[Type[ClinicalNotesParserInput]] = ClinicalNotesParserInput
    output_class: ClassVar[Type[ClinicalNotesParserOutput]] = ClinicalNotesParserOutput

    async def execute(self, input: ClinicalNotesParserInput) -> ClinicalNotesParserOutput:
        """Execute clinical notes parsing."""
        try:
            text = await self._get_text(input)
            if not text:
                return ClinicalNotesParserOutput.from_error("No text provided")

            # Detect note type
            note_type = input.note_type or self._detect_note_type(text)

            # Parse based on type
            soap_note = None
            hp_note = None

            if note_type in ["SOAP", "progress"]:
                soap_note = self._parse_soap(text)
            else:
                hp_note = self._parse_hp(text)

            # Extract entities
            diagnoses = self._extract_diagnoses(text)
            problems = self._extract_problems(text)
            medications = self._extract_medications(text) if input.extract_medications else []
            procedures = self._extract_procedures(text)

            entities = None
            if input.extract_entities:
                entities = {
                    "diagnoses": diagnoses,
                    "medications": medications,
                    "procedures": procedures,
                    "anatomy": self._extract_anatomy(text),
                }

            return ClinicalNotesParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"note_type": note_type},
                note_type_detected=note_type,
                soap_note=soap_note,
                hp_note=hp_note,
                diagnoses=diagnoses,
                problems=problems,
                medications_mentioned=medications,
                procedures_mentioned=procedures,
                extracted_entities=entities,
                summary=self._generate_summary(soap_note, hp_note, diagnoses),
                confidence=0.8
            )

        except Exception as e:
            return ClinicalNotesParserOutput.from_error(f"Parsing failed: {str(e)}")

    async def _get_text(self, input: ClinicalNotesParserInput) -> Optional[str]:
        """Get document text."""
        if input.document_text:
            return input.document_text
        if input.document_path:
            with open(input.document_path, 'r') as f:
                return f.read()
        return None

    def _detect_note_type(self, text: str) -> str:
        """Detect note type."""
        text_upper = text.upper()
        if "SUBJECTIVE" in text_upper and "OBJECTIVE" in text_upper:
            return "SOAP"
        if "CHIEF COMPLAINT" in text_upper and "HISTORY OF PRESENT ILLNESS" in text_upper:
            return "H&P"
        if "PROGRESS NOTE" in text_upper:
            return "progress"
        if "CONSULT" in text_upper:
            return "consult"
        return "H&P"

    def _parse_soap(self, text: str) -> SOAPNote:
        """Parse SOAP note."""
        sections = {"subjective": None, "objective": None, "assessment": None, "plan": None}
        markers = [("SUBJECTIVE", "subjective"), ("OBJECTIVE", "objective"),
                   ("ASSESSMENT", "assessment"), ("PLAN", "plan")]

        text_upper = text.upper()
        for marker, key in markers:
            if marker in text_upper:
                start = text_upper.index(marker) + len(marker)
                end = len(text)
                for other_marker, _ in markers:
                    if other_marker != marker:
                        idx = text_upper.find(other_marker, start)
                        if idx > 0 and idx < end:
                            end = idx
                sections[key] = text[start:end].strip().lstrip(":").strip()

        return SOAPNote(**sections)

    def _parse_hp(self, text: str) -> HPNote:
        """Parse H&P note."""
        hp = HPNote()

        # Extract sections
        section_map = {
            "CHIEF COMPLAINT": "chief_complaint",
            "HISTORY OF PRESENT ILLNESS": "hpi",
            "HPI": "hpi",
            "PAST MEDICAL HISTORY": "past_medical_history",
            "PMH": "past_medical_history",
            "PAST SURGICAL HISTORY": "past_surgical_history",
            "PSH": "past_surgical_history",
            "MEDICATIONS": "medications",
            "ALLERGIES": "allergies",
            "FAMILY HISTORY": "family_history",
            "SOCIAL HISTORY": "social_history",
            "REVIEW OF SYSTEMS": "review_of_systems",
            "ROS": "review_of_systems",
            "PHYSICAL EXAM": "physical_exam",
            "PE": "physical_exam",
            "ASSESSMENT": "assessment",
            "PLAN": "plan",
        }

        text_upper = text.upper()
        for marker, field in section_map.items():
            if marker in text_upper:
                start = text_upper.index(marker) + len(marker)
                end = self._find_next_section(text_upper, start, list(section_map.keys()))
                content = text[start:end].strip().lstrip(":").strip()

                if field in ["past_medical_history", "past_surgical_history", "medications", "allergies"]:
                    # Parse as list
                    items = [item.strip() for item in content.replace("\n", ",").split(",") if item.strip()]
                    setattr(hp, field, items)
                else:
                    setattr(hp, field, content)

        return hp

    def _find_next_section(self, text: str, start: int, markers: List[str]) -> int:
        """Find the start of the next section."""
        end = len(text)
        for marker in markers:
            idx = text.find(marker, start + 5)
            if idx > 0 and idx < end:
                end = idx
        return end

    def _extract_diagnoses(self, text: str) -> List[str]:
        """Extract diagnoses from text."""
        import re
        diagnoses = []

        # Look for assessment/diagnosis section
        patterns = [
            r"(?:ASSESSMENT|DIAGNOSIS|DIAGNOSES|IMPRESSION):\s*\n?([\s\S]*?)(?=\n[A-Z]{2,}:|$)",
            r"\d+\.\s+([A-Z][^.\n]+(?:disease|syndrome|disorder|infection|itis|osis|emia))",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    # Split numbered items
                    items = re.split(r'\d+\.', match)
                    for item in items:
                        item = item.strip()
                        if item and len(item) > 3 and len(item) < 100:
                            diagnoses.append(item)

        return diagnoses[:10]

    def _extract_problems(self, text: str) -> List[str]:
        """Extract problem list."""
        import re
        problems = []

        if "PROBLEM" in text.upper():
            idx = text.upper().index("PROBLEM")
            end = self._find_next_section(text.upper(), idx, ["ASSESSMENT", "PLAN", "MEDICATIONS"])
            section = text[idx:end]
            items = re.findall(r'\d+\.\s*([^\n]+)', section)
            problems.extend(items)

        return problems[:10]

    def _extract_medications(self, text: str) -> List[str]:
        """Extract medication mentions."""
        import re
        medications = []

        # Common medication patterns
        med_pattern = r'([A-Z][a-z]+(?:in|ol|ide|ate|one|ine|pam|lam)\s*\d*\s*(?:mg|mcg|g)?)'
        matches = re.findall(med_pattern, text)
        medications.extend(matches)

        # Look for medications section
        if "MEDICATION" in text.upper():
            idx = text.upper().index("MEDICATION")
            end = self._find_next_section(text.upper(), idx, ["ALLERGIES", "ASSESSMENT", "PLAN"])
            section = text[idx:end]
            lines = section.split("\n")
            for line in lines[1:]:
                line = line.strip()
                if line and len(line) > 3:
                    medications.append(line.split()[0])

        return list(set(medications))[:15]

    def _extract_procedures(self, text: str) -> List[str]:
        """Extract procedure mentions."""
        import re
        procedures = []
        procedure_terms = [
            "surgery", "biopsy", "endoscopy", "colonoscopy", "catheterization",
            "resection", "replacement", "repair", "fusion", "excision"
        ]

        for term in procedure_terms:
            pattern = rf'\b\w*\s*{term}\b'
            matches = re.findall(pattern, text, re.IGNORECASE)
            procedures.extend(matches)

        return list(set(procedures))[:10]

    def _extract_anatomy(self, text: str) -> List[str]:
        """Extract anatomical mentions."""
        anatomy_terms = [
            "heart", "lung", "liver", "kidney", "brain", "spine", "chest",
            "abdomen", "pelvis", "head", "neck", "extremity", "joint"
        ]
        found = []
        text_lower = text.lower()
        for term in anatomy_terms:
            if term in text_lower:
                found.append(term)
        return found

    def _generate_summary(
        self,
        soap: Optional[SOAPNote],
        hp: Optional[HPNote],
        diagnoses: List[str]
    ) -> str:
        """Generate note summary."""
        parts = []

        if hp and hp.chief_complaint:
            parts.append(f"Chief complaint: {hp.chief_complaint}")
        if soap and soap.subjective:
            parts.append(f"Subjective: {soap.subjective[:100]}...")

        if diagnoses:
            parts.append(f"Diagnoses: {', '.join(diagnoses[:3])}")

        if hp and hp.plan:
            parts.append(f"Plan: {hp.plan[:100]}...")
        if soap and soap.plan:
            parts.append(f"Plan: {soap.plan[:100]}...")

        return "\n".join(parts) if parts else "Clinical note parsed successfully."
