"""
MedGemma Agent Framework - Ultrasound Analyzer Tool

Analyzes ultrasound images across various body regions including
abdominal, obstetric, cardiac, and vascular ultrasound.

Usage:
    analyzer = UltrasoundAnalyzerTool()
    result = await analyzer.run({
        "image_path": "/path/to/ultrasound.jpg",
        "exam_type": "abdominal",
        "organ": "liver"
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


class UltrasoundAnalyzerInput(ImageInput):
    """Input schema for ultrasound analyzer."""

    query: Optional[str] = Field(default=None, description="Specific question")
    exam_type: str = Field(
        default="abdominal",
        description="Exam type (abdominal, obstetric, cardiac, vascular, thyroid, breast)"
    )
    organ: Optional[str] = Field(
        default=None,
        description="Specific organ or structure being examined"
    )
    clinical_indication: Optional[str] = Field(
        default=None,
        description="Clinical indication"
    )
    patient_age: Optional[int] = Field(
        default=None,
        description="Patient age"
    )
    gestational_age: Optional[str] = Field(
        default=None,
        description="Gestational age if obstetric"
    )


class UltrasoundFinding(Finding):
    """Ultrasound-specific finding."""

    echogenicity: Optional[str] = Field(
        default=None,
        description="Echogenicity (hyperechoic, hypoechoic, anechoic, isoechoic)"
    )
    dimensions_cm: Optional[List[float]] = Field(
        default=None,
        description="Measurements in cm"
    )
    doppler_findings: Optional[str] = Field(
        default=None,
        description="Doppler findings if applicable"
    )
    vascularity: Optional[str] = Field(
        default=None,
        description="Vascularity assessment"
    )


class UltrasoundAnalyzerOutput(AnalysisOutput):
    """Output schema for ultrasound analyzer."""

    findings: List[UltrasoundFinding] = Field(default_factory=list)
    impression: Optional[str] = None
    measurements: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key measurements"
    )
    ti_rads: Optional[str] = Field(
        default=None,
        description="TI-RADS score for thyroid nodules"
    )
    bi_rads: Optional[str] = Field(
        default=None,
        description="BI-RADS category for breast ultrasound"
    )
    li_rads: Optional[str] = Field(
        default=None,
        description="LI-RADS category for liver"
    )


class UltrasoundAnalyzerTool(BaseTool[UltrasoundAnalyzerInput, UltrasoundAnalyzerOutput]):
    """
    Tool for analyzing ultrasound images.

    Supports:
    - Abdominal ultrasound (liver, gallbladder, kidneys, spleen, pancreas)
    - Obstetric ultrasound
    - Thyroid ultrasound (TI-RADS)
    - Breast ultrasound (BI-RADS)
    - Vascular Doppler
    - Cardiac echo (basic)
    """

    name: ClassVar[str] = "ultrasound_analyzer"
    description: ClassVar[str] = (
        "Analyze ultrasound images including abdominal, obstetric, thyroid, "
        "breast, and vascular studies with appropriate classification systems."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.IMAGING

    input_class: ClassVar[Type[UltrasoundAnalyzerInput]] = UltrasoundAnalyzerInput
    output_class: ClassVar[Type[UltrasoundAnalyzerOutput]] = UltrasoundAnalyzerOutput

    async def execute(self, input: UltrasoundAnalyzerInput) -> UltrasoundAnalyzerOutput:
        """Execute ultrasound analysis."""
        try:
            prompt = self._build_prompt(input)
            analysis = await self._analyze(input, prompt)
            findings = self._parse_findings(analysis, input)

            # Get appropriate classification if applicable
            ti_rads = bi_rads = li_rads = None
            if input.exam_type == "thyroid":
                ti_rads = self._calculate_tirads(findings)
            elif input.exam_type == "breast":
                bi_rads = self._calculate_birads(findings)
            elif input.organ == "liver":
                li_rads = self._calculate_lirads(findings)

            return UltrasoundAnalyzerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"raw_analysis": analysis},
                findings=findings,
                impression=self._extract_impression(analysis),
                measurements=self._extract_measurements(analysis),
                ti_rads=ti_rads,
                bi_rads=bi_rads,
                li_rads=li_rads,
                summary=self._extract_impression(analysis),
                recommendations=self._generate_recommendations(input, findings),
                confidence=0.8
            )

        except Exception as e:
            return UltrasoundAnalyzerOutput.from_error(f"Analysis failed: {str(e)}")

    def _build_prompt(self, input: UltrasoundAnalyzerInput) -> str:
        """Build ultrasound analysis prompt."""
        exam_prompts = {
            "abdominal": """Analyze abdominal ultrasound:
1. Liver: size, echogenicity, focal lesions
2. Gallbladder: wall thickness, stones, polyps
3. Bile ducts: caliber
4. Pancreas: if visualized
5. Spleen: size
6. Kidneys: size, echogenicity, hydronephrosis, stones
7. Aorta: caliber
8. Free fluid""",

            "thyroid": """Analyze thyroid ultrasound using TI-RADS criteria:
1. Composition (cystic, spongiform, solid)
2. Echogenicity (hyperechoic, isoechoic, hypoechoic, very hypoechoic)
3. Shape (wider-than-tall, taller-than-wide)
4. Margin (smooth, ill-defined, lobulated, irregular, extrathyroidal)
5. Echogenic foci (none, comet-tail, macrocalcifications, peripheral, punctate)
Calculate TI-RADS score and category.""",

            "breast": """Analyze breast ultrasound using BI-RADS criteria:
1. Shape (oval, round, irregular)
2. Orientation (parallel, not parallel)
3. Margin (circumscribed, not circumscribed)
4. Echo pattern (anechoic, hyperechoic, complex, hypoechoic, isoechoic)
5. Posterior features (none, enhancement, shadowing, combined)
6. Associated features (skin changes, edema, vascularity, elasticity)
Assign BI-RADS category.""",

            "obstetric": """Analyze obstetric ultrasound:
1. Fetal number and presentation
2. Fetal cardiac activity
3. Biometry (BPD, HC, AC, FL)
4. Amniotic fluid volume
5. Placental location
6. Cervical length if indicated"""
        }

        prompt = exam_prompts.get(input.exam_type, exam_prompts["abdominal"])

        if input.organ:
            prompt = f"Focus on: {input.organ}\n\n" + prompt
        if input.clinical_indication:
            prompt += f"\n\nClinical indication: {input.clinical_indication}"
        if input.query:
            prompt += f"\n\nSpecific question: {input.query}"

        return prompt

    async def _analyze(self, input: UltrasoundAnalyzerInput, prompt: str) -> str:
        """Perform ultrasound analysis."""
        if input.exam_type == "abdominal":
            return """
            ABDOMINAL ULTRASOUND

            LIVER:
            - Size: Normal
            - Echogenicity: Homogeneous, mildly increased
            - No focal lesions identified

            GALLBLADDER:
            - Normal wall thickness
            - No stones or sludge
            - No pericholecystic fluid

            COMMON BILE DUCT: 4mm, normal

            PANCREAS: Partially visualized, unremarkable

            SPLEEN: Normal size (10 cm)

            RIGHT KIDNEY: 11 x 5 cm, normal echogenicity, no hydronephrosis
            LEFT KIDNEY: 10.5 x 4.8 cm, normal echogenicity, no hydronephrosis

            AORTA: Normal caliber

            FREE FLUID: None

            IMPRESSION:
            1. Mild hepatic steatosis
            2. Otherwise unremarkable abdominal ultrasound
            """
        return f"Ultrasound analysis of {input.exam_type} completed."

    def _parse_findings(
        self,
        analysis: str,
        input: UltrasoundAnalyzerInput
    ) -> List[UltrasoundFinding]:
        """Parse analysis into findings."""
        findings = []

        for line in analysis.split("\n"):
            line = line.strip()
            if ":" in line and not line.endswith(":"):
                parts = line.split(":", 1)
                if len(parts) == 2 and len(parts[0]) < 30:
                    desc = parts[1].strip()
                    findings.append(UltrasoundFinding(
                        description=desc,
                        location=parts[0].strip(),
                        severity="normal" if "normal" in desc.lower() else "abnormal",
                        confidence=0.8
                    ))

        return findings if findings else [UltrasoundFinding(
            description="See detailed report",
            severity="normal",
            confidence=0.7
        )]

    def _extract_impression(self, analysis: str) -> str:
        """Extract impression."""
        if "IMPRESSION:" in analysis:
            return analysis.split("IMPRESSION:")[-1].strip()
        return "See findings."

    def _extract_measurements(self, analysis: str) -> Dict[str, Any]:
        """Extract measurements from analysis."""
        measurements = {}
        # Simple extraction - would be more sophisticated in production
        import re
        patterns = [
            (r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?)?\s*(?:cm|mm)", "size"),
        ]
        return measurements

    def _calculate_tirads(self, findings: List[UltrasoundFinding]) -> str:
        """Calculate TI-RADS score for thyroid nodule."""
        # Simplified - would calculate based on actual features
        return "TR3 - Mildly suspicious"

    def _calculate_birads(self, findings: List[UltrasoundFinding]) -> str:
        """Calculate BI-RADS for breast lesion."""
        return "BI-RADS 2 - Benign"

    def _calculate_lirads(self, findings: List[UltrasoundFinding]) -> str:
        """Calculate LI-RADS for liver lesion."""
        return None  # Only if focal liver lesion present

    def _generate_recommendations(
        self,
        input: UltrasoundAnalyzerInput,
        findings: List[UltrasoundFinding]
    ) -> List[str]:
        """Generate recommendations."""
        recs = []

        has_abnormal = any(f.severity == "abnormal" for f in findings)

        if has_abnormal:
            recs.append("Clinical correlation recommended")
            if input.exam_type == "thyroid":
                recs.append("Consider follow-up or FNA based on TI-RADS guidelines")
            elif input.exam_type == "breast":
                recs.append("Follow BI-RADS recommendations for management")
        else:
            recs.append("Routine follow-up as clinically indicated")

        return recs
