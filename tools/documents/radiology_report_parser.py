"""
MedGemma Agent Framework - Radiology Report Parser Tool

Parses radiology reports to extract structured findings,
impressions, and recommendations.

Usage:
    parser = RadiologyReportParserTool()
    result = await parser.run({
        "document_text": "CT CHEST WITHOUT CONTRAST...",
        "modality": "CT"
    })
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import (
    BaseTool,
    DocumentInput,
    Finding,
    ToolCategory,
    ToolOutput,
    ToolStatus,
)


class RadiologyReportParserInput(DocumentInput):
    """Input schema for radiology report parser."""

    modality: Optional[str] = Field(
        default=None,
        description="Imaging modality (CT, MRI, XR, US, NM, PET)"
    )
    body_region: Optional[str] = Field(
        default=None,
        description="Body region examined"
    )
    extract_measurements: bool = Field(
        default=True,
        description="Extract measurements from report"
    )
    extract_comparisons: bool = Field(
        default=True,
        description="Extract comparison statements"
    )


class RadiologyFinding(Finding):
    """Radiology-specific finding."""

    organ_system: Optional[str] = Field(
        default=None,
        description="Organ system"
    )
    measurement: Optional[str] = Field(
        default=None,
        description="Measurement if applicable"
    )
    change_from_prior: Optional[str] = Field(
        default=None,
        description="Change from prior study"
    )
    fleischner_category: Optional[str] = Field(
        default=None,
        description="Fleischner category for nodules"
    )
    li_rads_category: Optional[str] = Field(
        default=None,
        description="LI-RADS category for liver lesions"
    )
    lung_rads_category: Optional[str] = Field(
        default=None,
        description="Lung-RADS category"
    )


class RadiologyReportParserOutput(ToolOutput):
    """Output schema for radiology report parser."""

    findings: List[RadiologyFinding] = Field(
        default_factory=list,
        description="Extracted findings"
    )
    impression: Optional[str] = Field(
        default=None,
        description="Report impression"
    )
    technique: Optional[str] = Field(
        default=None,
        description="Imaging technique"
    )
    comparison: Optional[str] = Field(
        default=None,
        description="Comparison studies"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Extracted recommendations"
    )
    critical_findings: Optional[List[str]] = Field(
        default=None,
        description="Critical/urgent findings"
    )
    measurements: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key measurements"
    )
    modality_detected: Optional[str] = Field(
        default=None,
        description="Detected imaging modality"
    )
    body_region_detected: Optional[str] = Field(
        default=None,
        description="Detected body region"
    )


class RadiologyReportParserTool(BaseTool[RadiologyReportParserInput, RadiologyReportParserOutput]):
    """
    Tool for parsing radiology reports.

    Extracts structured information including:
    - Findings with anatomical localization
    - Impressions
    - Recommendations
    - Measurements
    - Comparison with priors
    - Standardized categories (Lung-RADS, LI-RADS, etc.)
    """

    name: ClassVar[str] = "radiology_report_parser"
    description: ClassVar[str] = (
        "Parse radiology reports to extract structured findings, "
        "impressions, measurements, and recommendations."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENTS

    input_class: ClassVar[Type[RadiologyReportParserInput]] = RadiologyReportParserInput
    output_class: ClassVar[Type[RadiologyReportParserOutput]] = RadiologyReportParserOutput

    async def execute(self, input: RadiologyReportParserInput) -> RadiologyReportParserOutput:
        """Execute radiology report parsing."""
        try:
            text = await self._get_document_text(input)
            if not text:
                return RadiologyReportParserOutput.from_error("No document text")

            # Detect modality and body region
            modality = input.modality or self._detect_modality(text)
            body_region = input.body_region or self._detect_body_region(text)

            # Extract sections
            technique = self._extract_technique(text)
            comparison = self._extract_comparison(text)
            findings = self._extract_findings(text)
            impression = self._extract_impression(text)
            recommendations = self._extract_recommendations(text)

            # Extract measurements
            measurements = {}
            if input.extract_measurements:
                measurements = self._extract_measurements(text)

            # Identify critical findings
            critical = self._identify_critical_findings(findings, impression)

            return RadiologyReportParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={
                    "findings_count": len(findings),
                    "has_critical": len(critical) > 0
                },
                findings=findings,
                impression=impression,
                technique=technique,
                comparison=comparison,
                recommendations=recommendations,
                critical_findings=critical if critical else None,
                measurements=measurements if measurements else None,
                modality_detected=modality,
                body_region_detected=body_region,
                confidence=0.85
            )

        except Exception as e:
            return RadiologyReportParserOutput.from_error(f"Parsing failed: {str(e)}")

    async def _get_document_text(self, input: RadiologyReportParserInput) -> Optional[str]:
        """Get document text."""
        if input.document_text:
            return input.document_text
        if input.document_path:
            try:
                with open(input.document_path, 'r') as f:
                    return f.read()
            except Exception:
                pass
        return None

    def _detect_modality(self, text: str) -> Optional[str]:
        """Detect imaging modality from text."""
        text_upper = text.upper()
        if "CT " in text_upper or "COMPUTED TOMOGRAPHY" in text_upper:
            return "CT"
        if "MRI " in text_upper or "MR " in text_upper or "MAGNETIC RESONANCE" in text_upper:
            return "MRI"
        if "X-RAY" in text_upper or "RADIOGRAPH" in text_upper or " XR " in text_upper:
            return "XR"
        if "ULTRASOUND" in text_upper or " US " in text_upper:
            return "US"
        if "PET" in text_upper:
            return "PET"
        if "NUCLEAR" in text_upper:
            return "NM"
        return None

    def _detect_body_region(self, text: str) -> Optional[str]:
        """Detect body region from text."""
        text_upper = text.upper()
        regions = [
            ("CHEST", "chest"),
            ("ABDOMEN", "abdomen"),
            ("PELVIS", "pelvis"),
            ("HEAD", "head"),
            ("BRAIN", "brain"),
            ("SPINE", "spine"),
            ("CERVICAL", "c-spine"),
            ("THORACIC", "t-spine"),
            ("LUMBAR", "l-spine"),
            ("EXTREMITY", "extremity"),
            ("KNEE", "knee"),
            ("SHOULDER", "shoulder"),
        ]
        for pattern, region in regions:
            if pattern in text_upper:
                return region
        return None

    def _extract_technique(self, text: str) -> Optional[str]:
        """Extract technique section."""
        patterns = ["TECHNIQUE:", "EXAMINATION:", "PROCEDURE:"]
        for pattern in patterns:
            if pattern in text.upper():
                idx = text.upper().index(pattern)
                end = self._find_section_end(text, idx + len(pattern))
                return text[idx:end].strip()
        return None

    def _extract_comparison(self, text: str) -> Optional[str]:
        """Extract comparison statement."""
        import re
        patterns = [
            r"(?:COMPARISON|COMPARED TO|PRIOR):\s*([^\n]+)",
            r"compared to (?:prior |previous )?([\w\s]+(?:dated|from)?\s*[\d/]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_findings(self, text: str) -> List[RadiologyFinding]:
        """Extract findings from report."""
        findings = []

        # Find FINDINGS section
        findings_text = ""
        if "FINDINGS:" in text.upper():
            idx = text.upper().index("FINDINGS:")
            end = self._find_section_end(text, idx)
            findings_text = text[idx:end]
        elif "IMPRESSION:" in text.upper():
            # Use everything before impression
            idx = text.upper().index("IMPRESSION:")
            findings_text = text[:idx]

        if not findings_text:
            findings_text = text

        # Parse findings by organ system
        organ_patterns = [
            "LUNGS", "HEART", "MEDIASTINUM", "PLEURA", "BONES",
            "LIVER", "GALLBLADDER", "PANCREAS", "SPLEEN", "KIDNEYS",
            "BOWEL", "BLADDER", "BRAIN", "VENTRICLES"
        ]

        current_organ = None
        for line in findings_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if this is an organ header
            for organ in organ_patterns:
                if line.upper().startswith(organ):
                    current_organ = organ.title()
                    if ":" in line:
                        desc = line.split(":", 1)[1].strip()
                        if desc:
                            findings.append(RadiologyFinding(
                                description=desc,
                                location=current_organ,
                                organ_system=current_organ,
                                severity=self._assess_severity(desc),
                                confidence=0.8
                            ))
                    break
            else:
                # Regular finding line
                if line.startswith("-") or line.startswith("•"):
                    desc = line.lstrip("-•").strip()
                    findings.append(RadiologyFinding(
                        description=desc,
                        location=current_organ,
                        organ_system=current_organ,
                        severity=self._assess_severity(desc),
                        confidence=0.8
                    ))

        return findings

    def _extract_impression(self, text: str) -> Optional[str]:
        """Extract impression section."""
        patterns = ["IMPRESSION:", "CONCLUSION:", "SUMMARY:"]
        for pattern in patterns:
            if pattern in text.upper():
                idx = text.upper().index(pattern)
                end = self._find_section_end(text, idx + len(pattern))
                content = text[idx + len(pattern):end].strip()
                return content
        return None

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from report."""
        import re
        recommendations = []

        # Look for recommendation patterns
        patterns = [
            r"(?:recommend|suggest|consider|advised|follow[- ]?up)(?:ed|ing)?\s+(.+?)(?:\.|$)",
            r"(?:further|additional)\s+(?:evaluation|imaging|workup)\s+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            recommendations.extend(matches)

        # Look for RECOMMENDATION section
        if "RECOMMENDATION" in text.upper():
            idx = text.upper().index("RECOMMENDATION")
            end = self._find_section_end(text, idx)
            rec_text = text[idx:end]
            for line in rec_text.split("\n")[1:]:
                line = line.strip()
                if line and len(line) > 10:
                    recommendations.append(line)

        return recommendations[:5] if recommendations else None

    def _extract_measurements(self, text: str) -> Dict[str, str]:
        """Extract measurements from report."""
        import re
        measurements = {}

        # Common measurement patterns
        patterns = [
            r"(\w+(?:\s+\w+)?)\s+(?:measuring|measures)\s+([\d.]+\s*x\s*[\d.]+(?:\s*x\s*[\d.]+)?\s*(?:cm|mm))",
            r"([\d.]+\s*x\s*[\d.]+(?:\s*x\s*[\d.]+)?\s*(?:cm|mm))\s+(\w+(?:\s+\w+)?)",
            r"(\w+):\s*([\d.]+\s*(?:cm|mm))",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    key = match[0].strip()
                    value = match[1].strip()
                    if len(key) < 50:
                        measurements[key] = value

        return measurements

    def _identify_critical_findings(
        self,
        findings: List[RadiologyFinding],
        impression: Optional[str]
    ) -> List[str]:
        """Identify critical findings."""
        critical = []
        critical_terms = [
            "pneumothorax", "pulmonary embolism", "aortic dissection",
            "acute stroke", "hemorrhage", "free air", "bowel obstruction",
            "critical", "emergent", "urgent"
        ]

        # Check findings
        for finding in findings:
            if any(term in finding.description.lower() for term in critical_terms):
                critical.append(finding.description)

        # Check impression
        if impression:
            for term in critical_terms:
                if term in impression.lower():
                    critical.append(f"Impression mentions: {term}")

        return list(set(critical))

    def _assess_severity(self, description: str) -> str:
        """Assess severity of a finding."""
        desc_lower = description.lower()
        if any(term in desc_lower for term in ["normal", "unremarkable", "no ", "negative"]):
            return "normal"
        if any(term in desc_lower for term in ["mild", "minor", "small"]):
            return "mild"
        if any(term in desc_lower for term in ["moderate"]):
            return "moderate"
        if any(term in desc_lower for term in ["severe", "large", "critical", "urgent"]):
            return "severe"
        return "abnormal"

    def _find_section_end(self, text: str, start: int) -> int:
        """Find end of a section."""
        headers = ["FINDINGS:", "IMPRESSION:", "TECHNIQUE:", "COMPARISON:",
                   "RECOMMENDATION:", "CONCLUSION:"]
        end = len(text)
        for header in headers:
            idx = text.upper().find(header, start + 10)
            if idx > 0 and idx < end:
                end = idx
        return end
