"""
MedGemma Agent Framework - Medical Summarizer Tool

Summarizes long medical documents while preserving key clinical information.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class MedicalSummarizerInput(ToolInput):
    """Input for medical summarizer."""
    text: str = Field(description="Text to summarize")
    summary_type: str = Field(default="brief", description="Summary type: brief, detailed, structured")
    max_length: int = Field(default=500, description="Maximum summary length in characters")
    preserve_sections: Optional[List[str]] = Field(default=None, description="Sections to preserve")


class MedicalSummarizerOutput(ToolOutput):
    """Output for medical summarizer."""
    summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    critical_values: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    original_length: int = 0
    summary_length: int = 0
    compression_ratio: float = 0.0


class MedicalSummarizerTool(BaseTool[MedicalSummarizerInput, MedicalSummarizerOutput]):
    """Summarize medical documents."""

    name: ClassVar[str] = "medical_summarizer"
    description: ClassVar[str] = "Summarize medical documents while preserving clinically important information."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.UTILITIES

    input_class: ClassVar[Type[MedicalSummarizerInput]] = MedicalSummarizerInput
    output_class: ClassVar[Type[MedicalSummarizerOutput]] = MedicalSummarizerOutput

    # Key sections to extract
    KEY_SECTIONS = [
        "chief complaint", "diagnosis", "impression", "assessment",
        "plan", "recommendations", "findings", "conclusion"
    ]

    # Critical terms to highlight
    CRITICAL_TERMS = [
        "critical", "urgent", "emergent", "stat", "abnormal",
        "significant", "concerning", "elevated", "decreased"
    ]

    async def execute(self, input: MedicalSummarizerInput) -> MedicalSummarizerOutput:
        try:
            text = input.text
            original_length = len(text)

            # Extract key sections
            sections = self._extract_sections(text)

            # Identify key findings
            key_findings = self._extract_key_findings(text)

            # Identify critical values
            critical_values = self._extract_critical_values(text)

            # Generate summary based on type
            if input.summary_type == "structured":
                summary = self._generate_structured_summary(sections, key_findings)
            elif input.summary_type == "detailed":
                summary = self._generate_detailed_summary(text, sections)
            else:
                summary = self._generate_brief_summary(text, sections)

            # Truncate to max length
            if len(summary) > input.max_length:
                summary = summary[:input.max_length-3] + "..."

            # Extract action items
            action_items = self._extract_action_items(text)

            summary_length = len(summary)
            compression_ratio = summary_length / original_length if original_length > 0 else 0

            return MedicalSummarizerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"compression_ratio": compression_ratio},
                summary=summary,
                key_findings=key_findings,
                critical_values=critical_values,
                action_items=action_items,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=round(compression_ratio, 2),
                confidence=0.85
            )

        except Exception as e:
            return MedicalSummarizerOutput.from_error(str(e))

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from text."""
        sections = {}
        text_lower = text.lower()

        for section_name in self.KEY_SECTIONS:
            # Look for section header
            pattern_idx = text_lower.find(section_name)
            if pattern_idx >= 0:
                # Find end of section (next section or end)
                start = pattern_idx + len(section_name)
                # Skip colon and whitespace
                while start < len(text) and text[start] in ':\n\t ':
                    start += 1

                end = len(text)
                for other_section in self.KEY_SECTIONS:
                    if other_section != section_name:
                        other_idx = text_lower.find(other_section, start)
                        if other_idx > 0 and other_idx < end:
                            end = other_idx

                content = text[start:end].strip()
                if content:
                    sections[section_name] = content[:500]  # Limit each section

        return sections

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key clinical findings."""
        findings = []
        text_lower = text.lower()

        # Look for patterns indicating findings
        import re
        finding_patterns = [
            r'(?:shows?|reveals?|demonstrates?|indicates?)\s+(.+?)(?:\.|$)',
            r'(?:positive for|negative for)\s+(.+?)(?:\.|$)',
            r'(?:evidence of|no evidence of)\s+(.+?)(?:\.|$)',
        ]

        for pattern in finding_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches[:5]:  # Limit findings
                finding = match.strip()
                if len(finding) > 10 and len(finding) < 200:
                    findings.append(finding)

        return findings[:10]

    def _extract_critical_values(self, text: str) -> List[str]:
        """Extract critical or abnormal values."""
        critical = []
        text_lower = text.lower()

        # Look for critical terms near values
        import re
        for term in self.CRITICAL_TERMS:
            pattern = rf'{term}\s*[:\-]?\s*(.{{0,100}})'
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match.strip():
                    critical.append(f"{term}: {match.strip()[:100]}")

        return critical[:10]

    def _extract_action_items(self, text: str) -> List[str]:
        """Extract recommended actions."""
        actions = []
        text_lower = text.lower()

        action_patterns = [
            r'(?:recommend|suggest|advise|should|needs?)\s+(.+?)(?:\.|$)',
            r'(?:follow[- ]?up|return|schedule)\s+(.+?)(?:\.|$)',
            r'(?:order|obtain|check)\s+(.+?)(?:\.|$)',
        ]

        import re
        for pattern in action_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                action = match.strip()
                if len(action) > 5 and len(action) < 200:
                    actions.append(action)

        return actions[:5]

    def _generate_brief_summary(self, text: str, sections: Dict[str, str]) -> str:
        """Generate brief summary."""
        parts = []

        # Priority sections for brief summary
        priority = ["impression", "assessment", "diagnosis", "plan"]

        for section in priority:
            if section in sections:
                content = sections[section]
                # Take first sentence or first 100 chars
                first_sentence = content.split('.')[0]
                if len(first_sentence) > 100:
                    first_sentence = first_sentence[:100] + "..."
                parts.append(f"{section.title()}: {first_sentence}")

        if not parts:
            # Fallback: first 300 chars
            return text[:300] + "..." if len(text) > 300 else text

        return " | ".join(parts)

    def _generate_detailed_summary(self, text: str, sections: Dict[str, str]) -> str:
        """Generate detailed summary."""
        parts = []

        for section, content in sections.items():
            parts.append(f"**{section.title()}**: {content[:200]}")

        return "\n\n".join(parts) if parts else text[:500]

    def _generate_structured_summary(self, sections: Dict[str, str], findings: List[str]) -> str:
        """Generate structured summary."""
        parts = ["## Summary"]

        if "diagnosis" in sections or "assessment" in sections:
            dx = sections.get("diagnosis", sections.get("assessment", ""))
            parts.append(f"**Diagnosis/Assessment**: {dx[:200]}")

        if findings:
            parts.append("**Key Findings**:")
            for f in findings[:5]:
                parts.append(f"  - {f}")

        if "plan" in sections:
            parts.append(f"**Plan**: {sections['plan'][:200]}")

        return "\n".join(parts)
