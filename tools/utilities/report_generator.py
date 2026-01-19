"""
MedGemma Agent Framework - Report Generator Tool

Generates structured medical reports from analysis results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class ReportSection(ToolInput):
    """A section of a medical report."""
    title: str
    content: str
    priority: int = 0  # Higher = more important


class ReportGeneratorInput(ToolInput):
    """Input for report generator."""
    report_type: str = Field(description="Report type: consultation, imaging, lab, procedure, summary")
    patient_info: Optional[Dict[str, Any]] = Field(default=None, description="Patient demographics")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Report sections")
    findings: Optional[List[str]] = Field(default=None, description="Key findings to include")
    impressions: Optional[List[str]] = Field(default=None, description="Clinical impressions")
    recommendations: Optional[List[str]] = Field(default=None, description="Recommendations")
    format: str = Field(default="structured", description="Output format: structured, narrative, brief")
    include_timestamp: bool = Field(default=True, description="Include generation timestamp")
    include_disclaimer: bool = Field(default=True, description="Include AI disclaimer")


class ReportGeneratorOutput(ToolOutput):
    """Output for report generator."""
    report: str = ""
    report_type: str = ""
    word_count: int = 0
    sections_included: List[str] = Field(default_factory=list)
    generated_at: str = ""


class ReportGeneratorTool(BaseTool[ReportGeneratorInput, ReportGeneratorOutput]):
    """Generate structured medical reports."""

    name: ClassVar[str] = "report_generator"
    description: ClassVar[str] = "Generate structured medical reports from analysis results and clinical data."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.UTILITIES

    input_class: ClassVar[Type[ReportGeneratorInput]] = ReportGeneratorInput
    output_class: ClassVar[Type[ReportGeneratorOutput]] = ReportGeneratorOutput

    # Report templates
    REPORT_TEMPLATES = {
        "consultation": {
            "header": "CONSULTATION REPORT",
            "sections": ["reason_for_consultation", "history", "examination", "assessment", "plan"],
        },
        "imaging": {
            "header": "IMAGING REPORT",
            "sections": ["examination", "technique", "findings", "impression"],
        },
        "lab": {
            "header": "LABORATORY REPORT",
            "sections": ["tests_performed", "results", "interpretation", "recommendations"],
        },
        "procedure": {
            "header": "PROCEDURE REPORT",
            "sections": ["procedure", "indication", "technique", "findings", "complications", "disposition"],
        },
        "summary": {
            "header": "CLINICAL SUMMARY",
            "sections": ["overview", "key_findings", "assessment", "plan"],
        },
    }

    AI_DISCLAIMER = """
---
DISCLAIMER: This report was generated with AI assistance. All findings and
recommendations should be reviewed and verified by a qualified healthcare
professional before clinical use. This tool is intended for informational
purposes only and does not constitute medical advice.
---
"""

    async def execute(self, input: ReportGeneratorInput) -> ReportGeneratorOutput:
        try:
            timestamp = datetime.now().isoformat()

            # Get template for report type
            template = self.REPORT_TEMPLATES.get(
                input.report_type,
                self.REPORT_TEMPLATES["summary"]
            )

            # Build report based on format
            if input.format == "narrative":
                report = self._generate_narrative_report(input, template)
            elif input.format == "brief":
                report = self._generate_brief_report(input, template)
            else:
                report = self._generate_structured_report(input, template)

            # Add timestamp
            if input.include_timestamp:
                report = f"Generated: {timestamp}\n\n{report}"

            # Add disclaimer
            if input.include_disclaimer:
                report = f"{report}\n{self.AI_DISCLAIMER}"

            # Count words
            word_count = len(report.split())

            # Track sections included
            sections_included = []
            for section in input.sections:
                if isinstance(section, dict) and "title" in section:
                    sections_included.append(section["title"])

            return ReportGeneratorOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"word_count": word_count},
                report=report,
                report_type=input.report_type,
                word_count=word_count,
                sections_included=sections_included,
                generated_at=timestamp,
                confidence=0.9
            )

        except Exception as e:
            return ReportGeneratorOutput.from_error(str(e))

    def _generate_structured_report(self, input: ReportGeneratorInput, template: Dict) -> str:
        """Generate a structured report with clear sections."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append(template["header"].center(60))
        lines.append("=" * 60)
        lines.append("")

        # Patient info
        if input.patient_info:
            lines.append("PATIENT INFORMATION")
            lines.append("-" * 40)
            for key, value in input.patient_info.items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")
            lines.append("")

        # Custom sections
        if input.sections:
            for section in input.sections:
                if isinstance(section, dict):
                    title = section.get("title", "Section")
                    content = section.get("content", "")
                    lines.append(title.upper())
                    lines.append("-" * 40)
                    lines.append(content)
                    lines.append("")

        # Findings
        if input.findings:
            lines.append("FINDINGS")
            lines.append("-" * 40)
            for i, finding in enumerate(input.findings, 1):
                lines.append(f"  {i}. {finding}")
            lines.append("")

        # Impressions
        if input.impressions:
            lines.append("IMPRESSION")
            lines.append("-" * 40)
            for impression in input.impressions:
                lines.append(f"  • {impression}")
            lines.append("")

        # Recommendations
        if input.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(input.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_narrative_report(self, input: ReportGeneratorInput, template: Dict) -> str:
        """Generate a narrative-style report."""
        paragraphs = []

        paragraphs.append(f"**{template['header']}**\n")

        # Patient info as opening
        if input.patient_info:
            patient_desc = []
            if "age" in input.patient_info:
                patient_desc.append(f"{input.patient_info['age']}-year-old")
            if "gender" in input.patient_info:
                patient_desc.append(input.patient_info["gender"])
            if "name" in input.patient_info:
                patient_desc.append(f"({input.patient_info['name']})")
            if patient_desc:
                paragraphs.append(f"Patient: {' '.join(patient_desc)}\n")

        # Combine sections into narrative
        if input.sections:
            for section in input.sections:
                if isinstance(section, dict):
                    title = section.get("title", "")
                    content = section.get("content", "")
                    if title and content:
                        paragraphs.append(f"**{title}**: {content}\n")

        # Findings as paragraph
        if input.findings:
            findings_text = "Findings include: " + "; ".join(input.findings) + "."
            paragraphs.append(findings_text + "\n")

        # Impressions
        if input.impressions:
            impressions_text = "Impression: " + " ".join(input.impressions)
            paragraphs.append(impressions_text + "\n")

        # Recommendations
        if input.recommendations:
            rec_text = "Recommendations: " + " ".join(
                f"({i+1}) {rec}" for i, rec in enumerate(input.recommendations)
            )
            paragraphs.append(rec_text)

        return "\n".join(paragraphs)

    def _generate_brief_report(self, input: ReportGeneratorInput, template: Dict) -> str:
        """Generate a brief summary report."""
        lines = []

        lines.append(f"[{template['header']}]")

        # One-line patient info
        if input.patient_info:
            info_parts = [f"{k}: {v}" for k, v in list(input.patient_info.items())[:3]]
            lines.append(f"Patient: {', '.join(info_parts)}")

        # Key findings only
        if input.findings:
            lines.append(f"Findings: {'; '.join(input.findings[:3])}")

        # Primary impression
        if input.impressions:
            lines.append(f"Impression: {input.impressions[0]}")

        # Top recommendations
        if input.recommendations:
            lines.append(f"Recommendations: {'; '.join(input.recommendations[:2])}")

        return "\n".join(lines)
