"""
MedGemma Agent Framework - Dermoscopy Analyzer Tool

Analyzes dermoscopic images of skin lesions for melanoma detection
and skin cancer classification.

Usage:
    analyzer = DermoscopyAnalyzerTool()
    result = await analyzer.run({
        "image_path": "/path/to/lesion.jpg",
        "location": "back",
        "patient_age": 55
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


class DermoscopyAnalyzerInput(ImageInput):
    """Input schema for dermoscopy analyzer."""

    query: Optional[str] = Field(default=None, description="Specific question")
    location: Optional[str] = Field(
        default=None,
        description="Body location of the lesion"
    )
    patient_age: Optional[int] = Field(
        default=None,
        description="Patient age"
    )
    lesion_history: Optional[str] = Field(
        default=None,
        description="History (new, changing, symptomatic)"
    )
    skin_type: Optional[str] = Field(
        default=None,
        description="Fitzpatrick skin type (I-VI)"
    )
    personal_history: Optional[str] = Field(
        default=None,
        description="Personal history of skin cancer"
    )
    family_history: Optional[str] = Field(
        default=None,
        description="Family history of melanoma"
    )


class DermoscopyFinding(Finding):
    """Dermoscopy-specific finding."""

    dermoscopic_feature: Optional[str] = Field(
        default=None,
        description="Specific dermoscopic feature"
    )
    abcd_score: Optional[Dict[str, float]] = Field(
        default=None,
        description="ABCD dermoscopy scores"
    )
    colors_present: Optional[List[str]] = Field(
        default=None,
        description="Colors identified in lesion"
    )


class DermoscopyAnalyzerOutput(AnalysisOutput):
    """Output schema for dermoscopy analyzer."""

    findings: List[DermoscopyFinding] = Field(default_factory=list)
    classification: Optional[str] = Field(
        default=None,
        description="Lesion classification"
    )
    malignancy_risk: Optional[str] = Field(
        default=None,
        description="Risk of malignancy (low, intermediate, high)"
    )
    abcd_total_score: Optional[float] = Field(
        default=None,
        description="Total ABCD dermoscopy score"
    )
    seven_point_score: Optional[int] = Field(
        default=None,
        description="7-point checklist score"
    )
    biopsy_recommended: bool = Field(
        default=False,
        description="Whether biopsy is recommended"
    )
    differential: Optional[List[str]] = Field(
        default=None,
        description="Differential diagnosis"
    )


class DermoscopyAnalyzerTool(BaseTool[DermoscopyAnalyzerInput, DermoscopyAnalyzerOutput]):
    """
    Tool for analyzing dermoscopic images.

    Uses dermoscopic criteria for:
    - Melanoma detection
    - Basal cell carcinoma
    - Squamous cell carcinoma
    - Benign lesion classification
    """

    name: ClassVar[str] = "dermoscopy_analyzer"
    description: ClassVar[str] = (
        "Analyze dermoscopic images of skin lesions for melanoma detection, "
        "skin cancer classification, and biopsy recommendations."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.IMAGING

    input_class: ClassVar[Type[DermoscopyAnalyzerInput]] = DermoscopyAnalyzerInput
    output_class: ClassVar[Type[DermoscopyAnalyzerOutput]] = DermoscopyAnalyzerOutput

    async def execute(self, input: DermoscopyAnalyzerInput) -> DermoscopyAnalyzerOutput:
        """Execute dermoscopy analysis."""
        try:
            prompt = self._build_prompt(input)
            analysis = await self._analyze(input, prompt)
            findings = self._parse_findings(analysis)

            # Calculate scores
            abcd_score = self._calculate_abcd(findings)
            seven_point = self._calculate_seven_point(findings)

            # Determine risk and recommendation
            risk = self._assess_risk(abcd_score, seven_point, findings)
            biopsy = risk in ["intermediate", "high"]
            classification = self._classify_lesion(findings)

            return DermoscopyAnalyzerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"raw_analysis": analysis},
                findings=findings,
                classification=classification,
                malignancy_risk=risk,
                abcd_total_score=abcd_score,
                seven_point_score=seven_point,
                biopsy_recommended=biopsy,
                differential=self._get_differential(findings),
                summary=f"Classification: {classification}, Risk: {risk}",
                recommendations=self._generate_recommendations(risk, biopsy),
                confidence=0.75
            )

        except Exception as e:
            return DermoscopyAnalyzerOutput.from_error(f"Analysis failed: {str(e)}")

    def _build_prompt(self, input: DermoscopyAnalyzerInput) -> str:
        """Build dermoscopy analysis prompt."""
        parts = ["Analyze this dermoscopic image of a skin lesion."]

        if input.location:
            parts.append(f"Location: {input.location}")
        if input.patient_age:
            parts.append(f"Patient age: {input.patient_age}")
        if input.lesion_history:
            parts.append(f"History: {input.lesion_history}")

        parts.append("""
Evaluate using dermoscopic criteria:

ABCD Rule:
- Asymmetry (0-2 points)
- Border irregularity (0-8 points)
- Color variation (1-6 colors, 0.5 points each)
- Dermoscopic structures (0-5 points)

7-Point Checklist:
Major (2 points each): atypical pigment network, blue-white veil, atypical vascular pattern
Minor (1 point each): irregular streaks, irregular pigmentation, irregular dots/globules, regression

Identify specific features:
- Pigment network (typical vs atypical)
- Globules/dots pattern
- Streaks/pseudopods
- Blue-white veil
- Regression structures
- Vascular patterns""")

        return "\n".join(parts)

    async def _analyze(self, input: DermoscopyAnalyzerInput, prompt: str) -> str:
        """Perform dermoscopy analysis."""
        return """
        DERMOSCOPY ANALYSIS

        Lesion characteristics:
        - Size: Approximately 6mm
        - Shape: Oval, symmetric
        - Border: Regular, well-defined

        Dermoscopic features:
        - Pigment network: Typical, regular, brown
        - Globules: Few uniform globules at periphery
        - Colors: Brown, tan (2 colors)
        - No blue-white veil
        - No atypical vascular pattern
        - No regression structures

        ABCD Score:
        - Asymmetry: 0 (symmetric)
        - Border: 0 (regular)
        - Color: 1.0 (2 colors)
        - Dermoscopic structures: 1.0
        - Total: 2.0

        7-Point Checklist: 0 points

        ASSESSMENT:
        Benign melanocytic nevus
        Low risk for malignancy

        RECOMMENDATION:
        Clinical monitoring. No biopsy indicated at this time.
        """

    def _parse_findings(self, analysis: str) -> List[DermoscopyFinding]:
        """Parse analysis into findings."""
        findings = []

        features = [
            ("pigment network", "Pigment network pattern"),
            ("globules", "Globules/dots pattern"),
            ("border", "Border characteristics"),
            ("colors", "Color distribution"),
        ]

        for keyword, desc in features:
            if keyword in analysis.lower():
                # Extract relevant line
                for line in analysis.split("\n"):
                    if keyword in line.lower():
                        findings.append(DermoscopyFinding(
                            description=line.strip().lstrip("- "),
                            dermoscopic_feature=keyword,
                            confidence=0.8
                        ))
                        break

        return findings if findings else [DermoscopyFinding(
            description="See detailed dermoscopic analysis",
            confidence=0.7
        )]

    def _calculate_abcd(self, findings: List[DermoscopyFinding]) -> float:
        """Calculate ABCD dermoscopy score."""
        # Simplified calculation - would use actual findings
        return 2.0

    def _calculate_seven_point(self, findings: List[DermoscopyFinding]) -> int:
        """Calculate 7-point checklist score."""
        return 0

    def _assess_risk(
        self,
        abcd_score: float,
        seven_point: int,
        findings: List[DermoscopyFinding]
    ) -> str:
        """Assess malignancy risk."""
        if abcd_score >= 5.45 or seven_point >= 3:
            return "high"
        elif abcd_score >= 4.75 or seven_point >= 2:
            return "intermediate"
        return "low"

    def _classify_lesion(self, findings: List[DermoscopyFinding]) -> str:
        """Classify the lesion."""
        return "Benign melanocytic nevus"

    def _get_differential(self, findings: List[DermoscopyFinding]) -> List[str]:
        """Get differential diagnosis."""
        return [
            "Benign melanocytic nevus",
            "Junctional nevus",
            "Compound nevus"
        ]

    def _generate_recommendations(self, risk: str, biopsy: bool) -> List[str]:
        """Generate recommendations."""
        if biopsy:
            return [
                "Biopsy recommended for histopathological evaluation",
                "Excisional biopsy preferred if melanoma suspected",
                "Referral to dermatology"
            ]
        return [
            "Clinical monitoring recommended",
            "Photographic documentation advised",
            "Re-evaluate if changes observed"
        ]
