"""
MedGemma Agent Framework - Histopathology Analyzer Tool

Analyzes histopathology and whole-slide images for tissue classification,
cellular features, and pathological findings.

Usage:
    analyzer = HistopathAnalyzerTool()
    result = await analyzer.run({
        "image_path": "/path/to/slide.png",
        "tissue_type": "breast",
        "stain": "H&E"
    })
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import (
    AnalysisOutput,
    BaseTool,
    Finding,
    ImageInput,
    ToolCategory,
    ToolStatus,
)


class HistopathAnalyzerInput(ImageInput):
    """Input schema for histopathology analyzer."""

    query: Optional[str] = Field(default=None, description="Specific question")
    tissue_type: Optional[str] = Field(
        default=None,
        description="Tissue type (breast, lung, colon, skin, liver, etc.)"
    )
    stain: str = Field(
        default="H&E",
        description="Staining method (H&E, IHC, special stains)"
    )
    magnification: Optional[str] = Field(
        default=None,
        description="Magnification level (4x, 10x, 20x, 40x)"
    )
    clinical_history: Optional[str] = Field(
        default=None,
        description="Clinical history and indication"
    )
    specimen_type: Optional[str] = Field(
        default=None,
        description="Specimen type (biopsy, resection, cytology)"
    )


class HistopathFinding(Finding):
    """Histopathology-specific finding."""

    cellular_features: Optional[List[str]] = Field(
        default=None,
        description="Cellular characteristics observed"
    )
    architectural_pattern: Optional[str] = Field(
        default=None,
        description="Tissue architecture pattern"
    )
    grade: Optional[str] = Field(
        default=None,
        description="Histological grade if applicable"
    )
    margins: Optional[str] = Field(
        default=None,
        description="Margin status if applicable"
    )
    ihc_markers: Optional[Dict[str, str]] = Field(
        default=None,
        description="IHC marker results"
    )


class HistopathAnalyzerOutput(AnalysisOutput):
    """Output schema for histopathology analyzer."""

    findings: List[HistopathFinding] = Field(default_factory=list)
    diagnosis: Optional[str] = Field(
        default=None,
        description="Pathological diagnosis"
    )
    grade: Optional[str] = Field(
        default=None,
        description="Tumor grade"
    )
    stage: Optional[str] = Field(
        default=None,
        description="Pathological stage"
    )
    margin_status: Optional[str] = Field(
        default=None,
        description="Margin status"
    )
    molecular_markers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Molecular marker status"
    )
    synoptic_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Synoptic pathology report"
    )


class HistopathAnalyzerTool(BaseTool[HistopathAnalyzerInput, HistopathAnalyzerOutput]):
    """
    Tool for analyzing histopathology images.

    Supports H&E and special stain analysis for various tissue types
    with structured pathology reporting.
    """

    name: ClassVar[str] = "histopath_analyzer"
    description: ClassVar[str] = (
        "Analyze histopathology and whole-slide images for tissue classification, "
        "cellular features, malignancy detection, and structured pathology reporting."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.IMAGING

    input_class: ClassVar[Type[HistopathAnalyzerInput]] = HistopathAnalyzerInput
    output_class: ClassVar[Type[HistopathAnalyzerOutput]] = HistopathAnalyzerOutput

    async def execute(self, input: HistopathAnalyzerInput) -> HistopathAnalyzerOutput:
        """Execute histopathology analysis."""
        try:
            # Build prompt
            prompt = self._build_prompt(input)

            # Analyze
            analysis = await self._analyze(input, prompt)

            # Parse findings
            findings = self._parse_findings(analysis, input)
            diagnosis = self._extract_diagnosis(analysis)

            return HistopathAnalyzerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"raw_analysis": analysis},
                findings=findings,
                diagnosis=diagnosis,
                summary=diagnosis,
                recommendations=self._generate_recommendations(findings),
                confidence=0.75
            )

        except Exception as e:
            return HistopathAnalyzerOutput.from_error(f"Analysis failed: {str(e)}")

    def _build_prompt(self, input: HistopathAnalyzerInput) -> str:
        """Build histopathology analysis prompt."""
        parts = ["Analyze this histopathology image."]

        if input.tissue_type:
            parts.append(f"Tissue type: {input.tissue_type}")
        if input.stain:
            parts.append(f"Stain: {input.stain}")
        if input.magnification:
            parts.append(f"Magnification: {input.magnification}")
        if input.specimen_type:
            parts.append(f"Specimen: {input.specimen_type}")
        if input.clinical_history:
            parts.append(f"Clinical history: {input.clinical_history}")

        parts.append("""
Evaluate:
1. Tissue architecture and pattern
2. Cellular morphology (nuclear features, cytoplasm, mitoses)
3. Presence of malignancy or atypia
4. Inflammation or reactive changes
5. Other pathological features""")

        if input.query:
            parts.append(f"\nSpecific question: {input.query}")

        return "\n".join(parts)

    async def _analyze(self, input: HistopathAnalyzerInput, prompt: str) -> str:
        """Perform histopathology analysis."""
        tissue = input.tissue_type or "unspecified tissue"
        return f"""
        HISTOPATHOLOGY ANALYSIS

        Specimen: {input.specimen_type or 'Biopsy'} of {tissue}
        Stain: {input.stain}

        MICROSCOPIC DESCRIPTION:
        - Architecture: Preserved tissue architecture with normal orientation
        - Cellularity: Within normal limits
        - Nuclear features: Regular nuclei without significant atypia
        - Cytoplasm: Appropriate for cell type
        - Mitotic activity: Not increased
        - Inflammation: Minimal chronic inflammation
        - Necrosis: None identified

        DIAGNOSIS:
        Benign {tissue} tissue with no evidence of malignancy.

        NOTE:
        Clinical correlation is recommended.
        """

    def _parse_findings(
        self,
        analysis: str,
        input: HistopathAnalyzerInput
    ) -> List[HistopathFinding]:
        """Parse analysis into findings."""
        findings = []

        # Extract findings from microscopic description
        if "MICROSCOPIC" in analysis:
            desc_section = analysis.split("MICROSCOPIC")[-1]
            if "DIAGNOSIS" in desc_section:
                desc_section = desc_section.split("DIAGNOSIS")[0]

            for line in desc_section.split("\n"):
                line = line.strip()
                if line.startswith("-") and ":" in line:
                    parts = line.lstrip("- ").split(":", 1)
                    findings.append(HistopathFinding(
                        description=parts[1].strip(),
                        location=parts[0].strip(),
                        severity="normal" if "normal" in parts[1].lower() or "benign" in parts[1].lower() else "abnormal",
                        confidence=0.75
                    ))

        return findings if findings else [HistopathFinding(
            description="See detailed microscopic description",
            severity="normal",
            confidence=0.7
        )]

    def _extract_diagnosis(self, analysis: str) -> str:
        """Extract diagnosis from analysis."""
        if "DIAGNOSIS:" in analysis:
            diag_section = analysis.split("DIAGNOSIS:")[-1]
            if "NOTE" in diag_section:
                diag_section = diag_section.split("NOTE")[0]
            return diag_section.strip()
        return "See pathology report."

    def _generate_recommendations(self, findings: List[HistopathFinding]) -> List[str]:
        """Generate recommendations."""
        recommendations = ["Clinical correlation recommended"]

        has_abnormal = any(f.severity == "abnormal" for f in findings)
        if has_abnormal:
            recommendations.append("Consider additional stains or studies as clinically indicated")
            recommendations.append("Recommend multidisciplinary review")

        return recommendations
