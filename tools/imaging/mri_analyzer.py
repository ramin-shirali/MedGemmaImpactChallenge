"""
MedGemma Agent Framework - MRI Analyzer Tool

Analyzes MRI scans across various sequences and body regions.

Usage:
    analyzer = MRIAnalyzerTool()
    result = await analyzer.run({
        "image_path": "/path/to/mri_series/",
        "body_region": "brain",
        "sequences": ["T1", "T2", "FLAIR", "DWI"]
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


class MRIAnalyzerInput(ImageInput):
    """Input schema for MRI analyzer."""

    query: Optional[str] = Field(default=None, description="Specific question")
    body_region: str = Field(
        default="brain",
        description="Body region (brain, spine, knee, shoulder, abdomen, pelvis)"
    )
    sequences: Optional[List[str]] = Field(
        default=None,
        description="MRI sequences (T1, T2, FLAIR, DWI, ADC, etc.)"
    )
    contrast: Optional[str] = Field(
        default=None,
        description="Contrast administration (with gadolinium, without)"
    )
    clinical_indication: Optional[str] = Field(
        default=None,
        description="Clinical indication"
    )
    field_strength: Optional[str] = Field(
        default=None,
        description="Field strength (1.5T, 3T)"
    )


class MRIFinding(Finding):
    """MRI-specific finding."""

    signal_characteristics: Optional[Dict[str, str]] = Field(
        default=None,
        description="Signal on each sequence (T1, T2, FLAIR, etc.)"
    )
    enhancement_pattern: Optional[str] = Field(
        default=None,
        description="Gadolinium enhancement pattern"
    )
    diffusion_restriction: Optional[bool] = Field(
        default=None,
        description="Whether there is restricted diffusion"
    )
    dimensions_mm: Optional[List[float]] = Field(
        default=None,
        description="Size measurements"
    )


class MRIAnalyzerOutput(AnalysisOutput):
    """Output schema for MRI analyzer."""

    findings: List[MRIFinding] = Field(default_factory=list)
    impression: Optional[str] = None
    technique: Optional[str] = None
    sequences_reviewed: Optional[List[str]] = None
    critical_findings: Optional[List[str]] = None


class MRIAnalyzerTool(BaseTool[MRIAnalyzerInput, MRIAnalyzerOutput]):
    """
    Tool for analyzing MRI scans.

    Supports brain, spine, musculoskeletal, and body MRI analysis
    with multi-sequence interpretation.
    """

    name: ClassVar[str] = "mri_analyzer"
    description: ClassVar[str] = (
        "Analyze MRI scans (brain, spine, MSK, body) with multi-sequence "
        "interpretation for pathology detection and reporting."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.IMAGING

    input_class: ClassVar[Type[MRIAnalyzerInput]] = MRIAnalyzerInput
    output_class: ClassVar[Type[MRIAnalyzerOutput]] = MRIAnalyzerOutput

    async def execute(self, input: MRIAnalyzerInput) -> MRIAnalyzerOutput:
        """Execute MRI analysis."""
        try:
            # Build analysis prompt
            prompt = self._build_prompt(input)

            # Analyze (placeholder)
            analysis = await self._analyze(input, prompt)

            # Parse findings
            findings = self._parse_findings(analysis, input)

            return MRIAnalyzerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"raw_analysis": analysis},
                findings=findings,
                impression=self._extract_impression(analysis),
                technique=self._build_technique(input),
                sequences_reviewed=input.sequences,
                summary=self._extract_impression(analysis),
                confidence=0.8
            )

        except Exception as e:
            return MRIAnalyzerOutput.from_error(f"MRI analysis failed: {str(e)}")

    def _build_prompt(self, input: MRIAnalyzerInput) -> str:
        """Build MRI analysis prompt."""
        prompts = {
            "brain": """Analyze brain MRI:
1. Gray-white matter differentiation
2. Ventricles and sulci
3. Signal abnormalities (T1, T2, FLAIR, DWI)
4. Enhancement patterns if contrast given
5. Mass lesions
6. Extra-axial spaces
7. Posterior fossa
8. Sella and cavernous sinuses""",

            "spine": """Analyze spine MRI:
1. Vertebral body alignment and height
2. Disc morphology and hydration
3. Spinal cord signal
4. Neural foramina
5. Ligaments
6. Paraspinal soft tissues""",

            "knee": """Analyze knee MRI:
1. Menisci (medial and lateral)
2. Cruciate ligaments (ACL, PCL)
3. Collateral ligaments
4. Articular cartilage
5. Bone marrow
6. Extensor mechanism
7. Joint effusion"""
        }

        base = prompts.get(input.body_region, prompts["brain"])

        if input.clinical_indication:
            base += f"\n\nClinical indication: {input.clinical_indication}"
        if input.query:
            base += f"\n\nSpecific question: {input.query}"

        return base

    async def _analyze(self, input: MRIAnalyzerInput, prompt: str) -> str:
        """Perform MRI analysis."""
        # Placeholder response
        return f"""
        TECHNIQUE: MRI of the {input.body_region} with standard sequences.

        FINDINGS:
        Brain parenchyma: Normal signal intensity. No mass, hemorrhage, or infarct.
        Ventricles: Normal size and configuration.
        Extra-axial spaces: Normal.
        Posterior fossa: Unremarkable.

        IMPRESSION:
        Normal MRI of the {input.body_region}.
        """

    def _parse_findings(
        self,
        analysis: str,
        input: MRIAnalyzerInput
    ) -> List[MRIFinding]:
        """Parse analysis into findings."""
        findings = []
        lines = analysis.split("\n")

        for line in lines:
            line = line.strip()
            if ":" in line and not line.endswith(":") and not line.startswith(("TECHNIQUE", "IMPRESSION")):
                parts = line.split(":", 1)
                findings.append(MRIFinding(
                    description=parts[1].strip(),
                    location=parts[0].strip(),
                    severity="normal" if "normal" in parts[1].lower() else "abnormal",
                    confidence=0.8
                ))

        return findings if findings else [MRIFinding(
            description="See detailed report",
            location="General",
            severity="normal",
            confidence=0.7
        )]

    def _extract_impression(self, analysis: str) -> str:
        """Extract impression."""
        if "IMPRESSION:" in analysis:
            return analysis.split("IMPRESSION:")[-1].strip()
        return "See findings."

    def _build_technique(self, input: MRIAnalyzerInput) -> str:
        """Build technique description."""
        parts = [f"MRI of the {input.body_region}"]
        if input.field_strength:
            parts.append(f"at {input.field_strength}")
        if input.contrast:
            parts.append(f"{input.contrast}")
        if input.sequences:
            parts.append(f"Sequences: {', '.join(input.sequences)}")
        return ". ".join(parts)
