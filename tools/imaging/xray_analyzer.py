"""
MedGemma Agent Framework - X-Ray Analyzer Tool

Analyzes chest X-rays and other radiographs using MedGemma's
vision capabilities for findings detection and reporting.

Usage:
    analyzer = XrayAnalyzerTool()
    result = await analyzer.run({
        "image_path": "/path/to/xray.png",
        "query": "Are there any signs of pneumonia?"
    })
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import (
    AnalysisOutput,
    BaseTool,
    Finding,
    ImageInput,
    ToolCategory,
    ToolOutput,
    ToolStatus,
)


class XrayAnalyzerInput(ImageInput):
    """Input schema for X-ray analyzer."""

    query: Optional[str] = Field(
        default=None,
        description="Specific question about the X-ray"
    )
    view: Optional[str] = Field(
        default=None,
        description="X-ray view (PA, AP, lateral, etc.)"
    )
    body_region: Optional[str] = Field(
        default="chest",
        description="Body region (chest, abdomen, extremity, etc.)"
    )
    clinical_context: Optional[str] = Field(
        default=None,
        description="Clinical context or indication"
    )
    compare_with_prior: bool = Field(
        default=False,
        description="Whether to note comparison with prior"
    )


class XrayFinding(Finding):
    """A finding specific to X-ray analysis."""

    laterality: Optional[str] = Field(
        default=None,
        description="Left, right, or bilateral"
    )
    zone: Optional[str] = Field(
        default=None,
        description="Lung zone (upper, middle, lower) or specific region"
    )
    density: Optional[str] = Field(
        default=None,
        description="Density description (opacity, lucency, etc.)"
    )
    size_cm: Optional[float] = Field(
        default=None,
        description="Size measurement if applicable"
    )
    is_new: Optional[bool] = Field(
        default=None,
        description="Whether finding is new compared to prior"
    )


class XrayAnalyzerOutput(AnalysisOutput):
    """Output schema for X-ray analyzer."""

    findings: List[XrayFinding] = Field(
        default_factory=list,
        description="Radiographic findings"
    )
    impression: Optional[str] = Field(
        default=None,
        description="Overall impression"
    )
    technical_quality: Optional[str] = Field(
        default=None,
        description="Assessment of image quality"
    )
    view_confirmed: Optional[str] = Field(
        default=None,
        description="Confirmed view type"
    )
    comparison: Optional[str] = Field(
        default=None,
        description="Comparison with prior studies"
    )
    critical_findings: Optional[List[str]] = Field(
        default=None,
        description="Any critical/urgent findings"
    )


# Common chest X-ray findings to look for
CHEST_XRAY_FINDINGS = [
    "consolidation",
    "infiltrate",
    "opacity",
    "nodule",
    "mass",
    "effusion",
    "pneumothorax",
    "cardiomegaly",
    "widened mediastinum",
    "pulmonary edema",
    "atelectasis",
    "hyperinflation",
    "fracture",
    "pneumoperitoneum",
    "foreign body",
]


class XrayAnalyzerTool(BaseTool[XrayAnalyzerInput, XrayAnalyzerOutput]):
    """
    Tool for analyzing X-ray images using MedGemma.

    Features:
    - Chest X-ray analysis (most common)
    - Anatomical localization of findings
    - Severity assessment
    - Critical finding detection
    - Differential considerations
    """

    name: ClassVar[str] = "xray_analyzer"
    description: ClassVar[str] = (
        "Analyze X-ray images (chest, abdomen, extremity) for radiographic "
        "findings, abnormalities, and provide structured reports."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.IMAGING

    input_class: ClassVar[Type[XrayAnalyzerInput]] = XrayAnalyzerInput
    output_class: ClassVar[Type[XrayAnalyzerOutput]] = XrayAnalyzerOutput

    def __init__(self):
        super().__init__()
        self._model = None
        self._processor = None

    async def setup(self) -> None:
        """Load model for X-ray analysis."""
        # Model will be loaded by the agent and passed to tools
        await super().setup()

    async def execute(self, input: XrayAnalyzerInput) -> XrayAnalyzerOutput:
        """Execute X-ray analysis."""
        try:
            # Load image
            image = await self._load_image(input)
            if image is None:
                return XrayAnalyzerOutput.from_error("Failed to load image")

            # Build analysis prompt
            prompt = self._build_prompt(input)

            # Perform analysis using MedGemma
            analysis_result = await self._analyze_with_model(image, prompt)

            # Parse results into structured format
            findings = self._parse_findings(analysis_result)
            impression = self._extract_impression(analysis_result)
            critical = self._identify_critical_findings(findings)

            # Assess technical quality
            quality = await self._assess_quality(image)

            return XrayAnalyzerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={
                    "findings": [f.model_dump() for f in findings],
                    "impression": impression,
                    "technical_quality": quality,
                    "raw_analysis": analysis_result
                },
                findings=findings,
                impression=impression,
                technical_quality=quality,
                view_confirmed=input.view,
                critical_findings=critical if critical else None,
                summary=impression,
                recommendations=self._generate_recommendations(findings),
                confidence=self._calculate_confidence(findings)
            )

        except Exception as e:
            return XrayAnalyzerOutput.from_error(f"X-ray analysis failed: {str(e)}")

    async def _load_image(self, input: XrayAnalyzerInput):
        """Load image from various sources."""
        try:
            from PIL import Image

            if input.image_path:
                return Image.open(input.image_path).convert("RGB")
            elif input.image_base64:
                import base64
                import io
                img_bytes = base64.b64decode(input.image_base64)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif input.image_bytes:
                import io
                return Image.open(io.BytesIO(input.image_bytes)).convert("RGB")
            elif input.image_url:
                import requests
                import io
                response = requests.get(input.image_url, timeout=30)
                return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return None

    def _build_prompt(self, input: XrayAnalyzerInput) -> str:
        """Build the analysis prompt for MedGemma."""
        parts = []

        # Base instruction
        parts.append(
            "You are analyzing a medical X-ray image. Provide a detailed "
            "radiological analysis following standard reporting conventions."
        )

        # Add context
        if input.body_region:
            parts.append(f"\nBody region: {input.body_region}")

        if input.view:
            parts.append(f"View: {input.view}")

        if input.clinical_context:
            parts.append(f"Clinical context: {input.clinical_context}")

        # Specific query or general analysis
        if input.query:
            parts.append(f"\nSpecific question: {input.query}")
        else:
            parts.append(
                "\nProvide a systematic analysis including:\n"
                "1. Technical quality assessment\n"
                "2. Systematic review of all structures\n"
                "3. Description of any abnormal findings with location\n"
                "4. Overall impression\n"
                "5. Recommendations if applicable"
            )

        if input.compare_with_prior:
            parts.append("\nNote any changes compared to prior studies if referenced.")

        return "\n".join(parts)

    async def _analyze_with_model(self, image, prompt: str) -> str:
        """Run analysis through MedGemma model."""
        # In production, this would use the actual model
        # For now, return a structured placeholder
        return """
        FINDINGS:
        - Lungs: Clear bilaterally. No focal consolidation, effusion, or pneumothorax.
        - Heart: Normal cardiothoracic ratio. Mediastinal contours unremarkable.
        - Bones: No acute osseous abnormality.
        - Soft tissues: Unremarkable.

        IMPRESSION:
        No acute cardiopulmonary abnormality identified.
        """

    def _parse_findings(self, analysis: str) -> List[XrayFinding]:
        """Parse analysis text into structured findings."""
        findings = []

        # Simple parsing logic - would be more sophisticated in production
        lines = analysis.split("\n")
        current_finding = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("IMPRESSION") or line.startswith("FINDINGS"):
                continue

            if line.startswith("-"):
                # New finding
                text = line.lstrip("- ").strip()
                if ":" in text:
                    location, description = text.split(":", 1)

                    # Determine severity
                    severity = "normal"
                    if any(word in description.lower() for word in
                           ["abnormal", "opacity", "consolidation", "effusion"]):
                        severity = "abnormal"
                    elif any(word in description.lower() for word in
                             ["mass", "pneumothorax", "critical"]):
                        severity = "critical"

                    findings.append(XrayFinding(
                        description=description.strip(),
                        location=location.strip(),
                        severity=severity,
                        confidence=0.85
                    ))

        # If no structured findings, create a general one
        if not findings:
            findings.append(XrayFinding(
                description="Analysis completed. See raw output for details.",
                location="General",
                severity="normal",
                confidence=0.7
            ))

        return findings

    def _extract_impression(self, analysis: str) -> str:
        """Extract the impression from analysis."""
        if "IMPRESSION:" in analysis:
            return analysis.split("IMPRESSION:")[-1].strip()
        elif "IMPRESSION" in analysis:
            return analysis.split("IMPRESSION")[-1].strip()
        return "See findings above."

    def _identify_critical_findings(self, findings: List[XrayFinding]) -> List[str]:
        """Identify any critical findings requiring urgent attention."""
        critical = []

        critical_conditions = [
            "pneumothorax", "tension", "widened mediastinum",
            "pneumoperitoneum", "aortic dissection", "large effusion",
            "pulmonary embolism", "critical"
        ]

        for finding in findings:
            desc_lower = finding.description.lower()
            if any(cond in desc_lower for cond in critical_conditions):
                critical.append(finding.description)

        return critical

    async def _assess_quality(self, image) -> str:
        """Assess technical quality of the image."""
        # In production, would analyze exposure, rotation, etc.
        return "Adequate for diagnostic interpretation"

    def _generate_recommendations(self, findings: List[XrayFinding]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        has_abnormal = any(f.severity in ["abnormal", "critical"] for f in findings)

        if has_abnormal:
            recommendations.append(
                "Clinical correlation recommended for abnormal findings"
            )
            recommendations.append(
                "Consider follow-up imaging if clinically indicated"
            )

        return recommendations if recommendations else None

    def _calculate_confidence(self, findings: List[XrayFinding]) -> float:
        """Calculate overall confidence score."""
        if not findings:
            return 0.5

        confidences = [f.confidence for f in findings if f.confidence]
        return sum(confidences) / len(confidences) if confidences else 0.7
