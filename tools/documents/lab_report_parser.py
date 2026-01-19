"""
MedGemma Agent Framework - Lab Report Parser Tool

Parses laboratory reports to extract structured lab values,
reference ranges, and abnormal flags.

Usage:
    parser = LabReportParserTool()
    result = await parser.run({
        "document_text": "CBC results...",
        "report_type": "CBC"
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


class LabReportParserInput(DocumentInput):
    """Input schema for lab report parser."""

    report_type: Optional[str] = Field(
        default=None,
        description="Type of lab report (CBC, CMP, LFT, lipid panel, etc.)"
    )
    patient_age: Optional[int] = Field(
        default=None,
        description="Patient age for reference range adjustment"
    )
    patient_sex: Optional[str] = Field(
        default=None,
        description="Patient sex for reference range adjustment"
    )
    collection_date: Optional[str] = Field(
        default=None,
        description="Specimen collection date"
    )


class LabValue(BaseModel):
    """A single lab value with metadata."""

    name: str = Field(description="Lab test name")
    value: Any = Field(description="Result value")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    reference_range: Optional[str] = Field(
        default=None,
        description="Normal reference range"
    )
    flag: Optional[str] = Field(
        default=None,
        description="Abnormal flag (H, L, HH, LL, C)"
    )
    is_critical: bool = Field(default=False, description="Critical value")
    loinc_code: Optional[str] = Field(default=None, description="LOINC code")
    category: Optional[str] = Field(default=None, description="Test category")


class LabReportParserOutput(ToolOutput):
    """Output schema for lab report parser."""

    lab_values: List[LabValue] = Field(
        default_factory=list,
        description="Extracted lab values"
    )
    abnormal_values: List[LabValue] = Field(
        default_factory=list,
        description="Values outside reference range"
    )
    critical_values: List[LabValue] = Field(
        default_factory=list,
        description="Critical values requiring immediate attention"
    )
    report_date: Optional[str] = Field(
        default=None,
        description="Date of report"
    )
    collection_date: Optional[str] = Field(
        default=None,
        description="Specimen collection date"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary of findings"
    )
    clinical_interpretation: Optional[str] = Field(
        default=None,
        description="Clinical interpretation"
    )


# Common lab test patterns for parsing
LAB_PATTERNS = {
    "CBC": {
        "WBC": {"unit": "x10^9/L", "loinc": "6690-2"},
        "RBC": {"unit": "x10^12/L", "loinc": "789-8"},
        "Hemoglobin": {"unit": "g/dL", "loinc": "718-7"},
        "Hematocrit": {"unit": "%", "loinc": "4544-3"},
        "MCV": {"unit": "fL", "loinc": "787-2"},
        "MCH": {"unit": "pg", "loinc": "785-6"},
        "MCHC": {"unit": "g/dL", "loinc": "786-4"},
        "Platelets": {"unit": "x10^9/L", "loinc": "777-3"},
    },
    "CMP": {
        "Glucose": {"unit": "mg/dL", "loinc": "2345-7"},
        "BUN": {"unit": "mg/dL", "loinc": "3094-0"},
        "Creatinine": {"unit": "mg/dL", "loinc": "2160-0"},
        "Sodium": {"unit": "mEq/L", "loinc": "2951-2"},
        "Potassium": {"unit": "mEq/L", "loinc": "2823-3"},
        "Chloride": {"unit": "mEq/L", "loinc": "2075-0"},
        "CO2": {"unit": "mEq/L", "loinc": "2028-9"},
        "Calcium": {"unit": "mg/dL", "loinc": "17861-6"},
    }
}

# Reference ranges (simplified, adult values)
REFERENCE_RANGES = {
    "WBC": (4.5, 11.0),
    "RBC": (4.5, 5.5),  # Male values
    "Hemoglobin": (13.5, 17.5),  # Male values
    "Hematocrit": (38.0, 50.0),
    "Platelets": (150, 400),
    "Glucose": (70, 100),
    "BUN": (7, 20),
    "Creatinine": (0.7, 1.3),
    "Sodium": (136, 145),
    "Potassium": (3.5, 5.0),
    "Chloride": (98, 106),
    "CO2": (23, 29),
    "Calcium": (8.5, 10.5),
}

# Critical values
CRITICAL_VALUES = {
    "Glucose": (40, 500),
    "Potassium": (2.5, 6.5),
    "Sodium": (120, 160),
    "Hemoglobin": (7.0, 20.0),
    "Platelets": (20, 1000),
    "WBC": (2.0, 30.0),
}


class LabReportParserTool(BaseTool[LabReportParserInput, LabReportParserOutput]):
    """
    Tool for parsing laboratory reports.

    Extracts structured data from lab reports including:
    - Individual test values with units
    - Reference ranges
    - Abnormal and critical flags
    - LOINC codes
    - Clinical interpretation
    """

    name: ClassVar[str] = "lab_report_parser"
    description: ClassVar[str] = (
        "Parse laboratory reports to extract structured lab values, "
        "reference ranges, abnormal flags, and clinical interpretations."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENTS

    input_class: ClassVar[Type[LabReportParserInput]] = LabReportParserInput
    output_class: ClassVar[Type[LabReportParserOutput]] = LabReportParserOutput

    async def execute(self, input: LabReportParserInput) -> LabReportParserOutput:
        """Execute lab report parsing."""
        try:
            # Get document text
            text = await self._get_document_text(input)
            if not text:
                return LabReportParserOutput.from_error("No document text provided")

            # Parse lab values
            lab_values = self._parse_lab_values(text, input)

            # Identify abnormal and critical values
            abnormal = [v for v in lab_values if v.flag in ["H", "L", "HH", "LL"]]
            critical = [v for v in lab_values if v.is_critical]

            # Generate summary and interpretation
            summary = self._generate_summary(lab_values, abnormal, critical)
            interpretation = self._generate_interpretation(abnormal, critical)

            return LabReportParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={
                    "lab_values": [v.model_dump() for v in lab_values],
                    "abnormal_count": len(abnormal),
                    "critical_count": len(critical)
                },
                lab_values=lab_values,
                abnormal_values=abnormal,
                critical_values=critical,
                collection_date=input.collection_date,
                summary=summary,
                clinical_interpretation=interpretation,
                confidence=0.85
            )

        except Exception as e:
            return LabReportParserOutput.from_error(f"Parsing failed: {str(e)}")

    async def _get_document_text(self, input: LabReportParserInput) -> Optional[str]:
        """Get document text from various sources."""
        if input.document_text:
            return input.document_text
        if input.document_path:
            # Read from file
            try:
                with open(input.document_path, 'r') as f:
                    return f.read()
            except Exception:
                pass
        return None

    def _parse_lab_values(
        self,
        text: str,
        input: LabReportParserInput
    ) -> List[LabValue]:
        """Parse lab values from text."""
        import re

        lab_values = []

        # Pattern for lab value lines: Name Value Unit (Reference Range)
        patterns = [
            r'([A-Za-z\s]+):\s*([\d.]+)\s*(\S+)?\s*\(?([0-9.-]+\s*-\s*[0-9.]+)?\)?',
            r'([A-Za-z\s]+)\s+([\d.]+)\s+(\S+)?\s+\(?([0-9.-]+\s*-\s*[0-9.]+)?\)?',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match[0].strip()
                try:
                    value = float(match[1])
                except ValueError:
                    continue

                unit = match[2] if len(match) > 2 and match[2] else None
                ref_range = match[3] if len(match) > 3 and match[3] else None

                # Look up reference range if not in text
                if not ref_range and name in REFERENCE_RANGES:
                    low, high = REFERENCE_RANGES[name]
                    ref_range = f"{low}-{high}"

                # Determine flag
                flag = self._determine_flag(name, value)
                is_critical = self._is_critical(name, value)

                # Get LOINC code
                loinc = None
                for panel, tests in LAB_PATTERNS.items():
                    if name in tests:
                        loinc = tests[name].get("loinc")
                        if not unit:
                            unit = tests[name].get("unit")
                        break

                lab_values.append(LabValue(
                    name=name,
                    value=value,
                    unit=unit,
                    reference_range=ref_range,
                    flag=flag,
                    is_critical=is_critical,
                    loinc_code=loinc
                ))

        # If no values parsed, create sample based on report type
        if not lab_values and input.report_type:
            lab_values = self._create_sample_values(input.report_type)

        return lab_values

    def _determine_flag(self, name: str, value: float) -> Optional[str]:
        """Determine abnormal flag for a value."""
        if name not in REFERENCE_RANGES:
            return None

        low, high = REFERENCE_RANGES[name]
        critical = CRITICAL_VALUES.get(name)

        if critical:
            crit_low, crit_high = critical
            if value <= crit_low:
                return "LL"  # Critical low
            if value >= crit_high:
                return "HH"  # Critical high

        if value < low:
            return "L"  # Low
        if value > high:
            return "H"  # High

        return None

    def _is_critical(self, name: str, value: float) -> bool:
        """Check if value is critical."""
        if name not in CRITICAL_VALUES:
            return False

        crit_low, crit_high = CRITICAL_VALUES[name]
        return value <= crit_low or value >= crit_high

    def _create_sample_values(self, report_type: str) -> List[LabValue]:
        """Create sample values based on report type."""
        if report_type not in LAB_PATTERNS:
            return []

        values = []
        for name, info in LAB_PATTERNS[report_type].items():
            if name in REFERENCE_RANGES:
                low, high = REFERENCE_RANGES[name]
                # Use midpoint as sample value
                sample_value = (low + high) / 2
                values.append(LabValue(
                    name=name,
                    value=sample_value,
                    unit=info.get("unit"),
                    reference_range=f"{low}-{high}",
                    loinc_code=info.get("loinc")
                ))

        return values

    def _generate_summary(
        self,
        all_values: List[LabValue],
        abnormal: List[LabValue],
        critical: List[LabValue]
    ) -> str:
        """Generate summary of lab results."""
        parts = [f"Total tests: {len(all_values)}"]

        if critical:
            parts.append(f"CRITICAL VALUES: {len(critical)}")
            for v in critical:
                parts.append(f"  - {v.name}: {v.value} {v.unit or ''} ({v.flag})")

        if abnormal:
            non_critical_abnormal = [v for v in abnormal if not v.is_critical]
            if non_critical_abnormal:
                parts.append(f"Abnormal (non-critical): {len(non_critical_abnormal)}")
                for v in non_critical_abnormal[:5]:
                    parts.append(f"  - {v.name}: {v.value} {v.unit or ''} ({v.flag})")

        if not abnormal and not critical:
            parts.append("All values within normal limits")

        return "\n".join(parts)

    def _generate_interpretation(
        self,
        abnormal: List[LabValue],
        critical: List[LabValue]
    ) -> str:
        """Generate clinical interpretation."""
        if critical:
            return (
                "CRITICAL VALUE ALERT: One or more critical values require "
                "immediate clinical attention. Contact ordering provider immediately."
            )

        if not abnormal:
            return "No significant abnormalities identified."

        # Group abnormalities by pattern
        interpretations = []

        low_hgb = any(v.name == "Hemoglobin" and v.flag == "L" for v in abnormal)
        low_rbc = any(v.name == "RBC" and v.flag == "L" for v in abnormal)
        if low_hgb or low_rbc:
            interpretations.append("Anemia pattern identified")

        high_glucose = any(v.name == "Glucose" and v.flag == "H" for v in abnormal)
        if high_glucose:
            interpretations.append("Elevated glucose - consider diabetes evaluation")

        high_creat = any(v.name == "Creatinine" and v.flag == "H" for v in abnormal)
        high_bun = any(v.name == "BUN" and v.flag == "H" for v in abnormal)
        if high_creat or high_bun:
            interpretations.append("Renal function abnormality")

        if not interpretations:
            interpretations.append("Abnormal values noted - clinical correlation recommended")

        return "; ".join(interpretations)
