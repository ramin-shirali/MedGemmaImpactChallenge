"""
MedGemma Agent Framework - Hallucination Detector Tool

Detects potential hallucinations in AI-generated medical content.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class HallucinationDetectorInput(ToolInput):
    """Input for hallucination detector."""
    content: str = Field(description="Content to check for hallucinations")
    content_type: str = Field(default="general", description="Content type: diagnosis, medication, procedure, statistic")
    source_context: Optional[str] = Field(default=None, description="Source material for verification")
    check_citations: bool = Field(default=True, description="Verify any cited sources")


class HallucinationDetectorOutput(ToolOutput):
    """Output for hallucination detector."""
    hallucination_risk: str = "low"  # low, medium, high
    flagged_claims: List[Dict[str, Any]] = Field(default_factory=list)
    unverifiable_statements: List[str] = Field(default_factory=list)
    suspicious_citations: List[str] = Field(default_factory=list)
    verification_suggestions: List[str] = Field(default_factory=list)


class HallucinationDetectorTool(BaseTool[HallucinationDetectorInput, HallucinationDetectorOutput]):
    """Detect potential hallucinations in medical content."""

    name: ClassVar[str] = "hallucination_detector"
    description: ClassVar[str] = "Detect potential hallucinations, unverifiable claims, and suspicious citations in AI-generated medical content."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.SAFETY

    input_class: ClassVar[Type[HallucinationDetectorInput]] = HallucinationDetectorInput
    output_class: ClassVar[Type[HallucinationDetectorOutput]] = HallucinationDetectorOutput

    # Known valid medical sources
    VALID_SOURCES = [
        "pubmed", "medline", "cochrane", "uptodate", "nejm", "lancet",
        "jama", "bmj", "annals of internal medicine", "fda", "cdc",
        "who", "nih", "mayo clinic", "cleveland clinic"
    ]

    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r"studies show that \d+% of",  # Specific percentages without citation
        r"research proves",  # Absolute claims
        r"according to experts",  # Vague attribution
        r"scientists discovered",  # Vague claims
        r"recent studies",  # Without specific citation
        r"it is well known",  # Unverified common knowledge claims
    ]

    async def execute(self, input: HallucinationDetectorInput) -> HallucinationDetectorOutput:
        try:
            import re

            content = input.content
            content_lower = content.lower()

            flagged_claims = []
            unverifiable = []
            suspicious_citations = []
            suggestions = []

            # Check for suspicious patterns
            for pattern in self.SUSPICIOUS_PATTERNS:
                matches = re.findall(pattern, content_lower)
                for match in matches:
                    unverifiable.append(match)
                    suggestions.append(f"Verify claim: '{match}' with specific citation")

            # Check for specific statistics without citations
            stat_pattern = r'(\d+(?:\.\d+)?%|\d+(?:,\d+)?\s*(?:patients|people|cases|deaths))'
            stats = re.findall(stat_pattern, content)
            for stat in stats:
                # Check if there's a citation nearby
                stat_idx = content.find(stat)
                context = content[max(0, stat_idx-50):min(len(content), stat_idx+100)]
                has_citation = any(src in context.lower() for src in self.VALID_SOURCES)
                if not has_citation and "et al" not in context.lower():
                    flagged_claims.append({
                        "claim": stat,
                        "type": "unverified_statistic",
                        "reason": "Specific statistic without apparent citation"
                    })

            # Check for medication dosages
            dose_pattern = r'(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?))'
            doses = re.findall(dose_pattern, content_lower)
            if doses and input.content_type == "medication":
                suggestions.append("Verify all medication dosages against official sources")
                for dose in doses[:3]:
                    flagged_claims.append({
                        "claim": dose,
                        "type": "medication_dose",
                        "reason": "Medication dosage should be verified"
                    })

            # Check citations if requested
            if input.check_citations:
                # Look for citation patterns
                citation_patterns = [
                    r'\(([^)]+\d{4})\)',  # (Author 2024)
                    r'\[(\d+)\]',  # [1]
                    r'doi:\s*(\S+)',  # DOI
                ]
                for pattern in citation_patterns:
                    citations = re.findall(pattern, content)
                    for citation in citations:
                        # Check if it looks valid
                        if not any(src in citation.lower() for src in self.VALID_SOURCES):
                            if not re.match(r'10\.\d{4,}', citation):  # Not a DOI
                                suspicious_citations.append(citation)

            # Determine overall risk level
            total_issues = len(flagged_claims) + len(unverifiable) + len(suspicious_citations)
            if total_issues == 0:
                risk_level = "low"
            elif total_issues <= 3:
                risk_level = "medium"
            else:
                risk_level = "high"

            return HallucinationDetectorOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"issues_count": total_issues},
                hallucination_risk=risk_level,
                flagged_claims=flagged_claims,
                unverifiable_statements=unverifiable,
                suspicious_citations=suspicious_citations,
                verification_suggestions=suggestions,
                confidence=0.75
            )

        except Exception as e:
            return HallucinationDetectorOutput.from_error(str(e))
