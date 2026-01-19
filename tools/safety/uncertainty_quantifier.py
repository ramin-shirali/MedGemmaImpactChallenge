"""
MedGemma Agent Framework - Uncertainty Quantifier Tool

Quantifies and communicates uncertainty in AI outputs.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class UncertaintyQuantifierInput(ToolInput):
    """Input for uncertainty quantifier."""
    prediction: str = Field(description="The prediction or claim to assess")
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1, description="Model confidence if available")
    evidence_strength: Optional[str] = Field(default=None, description="Strength of supporting evidence")
    prediction_type: str = Field(default="general", description="Type: diagnosis, prognosis, treatment, test_result")
    alternatives_considered: Optional[List[str]] = Field(default=None, description="Alternative possibilities")


class UncertaintyQuantifierOutput(ToolOutput):
    """Output for uncertainty quantifier."""
    uncertainty_level: str = "moderate"  # very_low, low, moderate, high, very_high
    confidence_percentage: float = 50.0
    uncertainty_sources: List[str] = Field(default_factory=list)
    calibrated_statement: str = ""
    communication_guidance: str = ""
    should_defer_to_expert: bool = False


class UncertaintyQuantifierTool(BaseTool[UncertaintyQuantifierInput, UncertaintyQuantifierOutput]):
    """Quantify and communicate uncertainty in predictions."""

    name: ClassVar[str] = "uncertainty_quantifier"
    description: ClassVar[str] = "Quantify uncertainty in AI predictions and generate appropriately hedged statements."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.SAFETY

    input_class: ClassVar[Type[UncertaintyQuantifierInput]] = UncertaintyQuantifierInput
    output_class: ClassVar[Type[UncertaintyQuantifierOutput]] = UncertaintyQuantifierOutput

    # Uncertainty language mapping
    UNCERTAINTY_LANGUAGE = {
        "very_low": ["almost certainly", "very likely", "highly probable"],
        "low": ["likely", "probably", "appears to be"],
        "moderate": ["possibly", "may be", "could be", "suggests"],
        "high": ["uncertain", "might be", "cannot rule out"],
        "very_high": ["highly uncertain", "speculative", "limited confidence"],
    }

    # Evidence strength mapping
    EVIDENCE_STRENGTH_SCORE = {
        "strong": 0.9,
        "moderate": 0.7,
        "weak": 0.5,
        "very_weak": 0.3,
        "none": 0.1,
    }

    async def execute(self, input: UncertaintyQuantifierInput) -> UncertaintyQuantifierOutput:
        try:
            # Calculate base confidence
            confidence = self._calculate_confidence(input)

            # Identify uncertainty sources
            sources = self._identify_uncertainty_sources(input)

            # Determine uncertainty level
            uncertainty_level = self._categorize_uncertainty(confidence)

            # Generate calibrated statement
            calibrated = self._generate_calibrated_statement(input.prediction, uncertainty_level)

            # Generate communication guidance
            guidance = self._generate_guidance(uncertainty_level, input.prediction_type)

            # Determine if expert referral needed
            should_defer = confidence < 0.5 or uncertainty_level in ["high", "very_high"]
            if input.prediction_type in ["diagnosis", "treatment"] and confidence < 0.7:
                should_defer = True

            return UncertaintyQuantifierOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"confidence": confidence},
                uncertainty_level=uncertainty_level,
                confidence_percentage=round(confidence * 100, 1),
                uncertainty_sources=sources,
                calibrated_statement=calibrated,
                communication_guidance=guidance,
                should_defer_to_expert=should_defer,
                confidence=confidence
            )

        except Exception as e:
            return UncertaintyQuantifierOutput.from_error(str(e))

    def _calculate_confidence(self, input: UncertaintyQuantifierInput) -> float:
        """Calculate overall confidence score."""
        scores = []

        # Model confidence if available
        if input.confidence_score is not None:
            scores.append(input.confidence_score)

        # Evidence strength
        if input.evidence_strength:
            evidence_score = self.EVIDENCE_STRENGTH_SCORE.get(input.evidence_strength.lower(), 0.5)
            scores.append(evidence_score)

        # Alternatives penalty
        if input.alternatives_considered:
            num_alternatives = len(input.alternatives_considered)
            alternative_penalty = max(0.5, 1 - (num_alternatives * 0.1))
            scores.append(alternative_penalty)

        # Prediction type baseline
        type_baselines = {
            "diagnosis": 0.6,
            "prognosis": 0.5,
            "treatment": 0.65,
            "test_result": 0.8,
            "general": 0.7,
        }
        scores.append(type_baselines.get(input.prediction_type, 0.6))

        return sum(scores) / len(scores) if scores else 0.5

    def _identify_uncertainty_sources(self, input: UncertaintyQuantifierInput) -> List[str]:
        """Identify sources of uncertainty."""
        sources = []

        if input.confidence_score and input.confidence_score < 0.7:
            sources.append("Model confidence below threshold")

        if input.evidence_strength in ["weak", "very_weak", "none"]:
            sources.append(f"Limited supporting evidence ({input.evidence_strength})")

        if input.alternatives_considered and len(input.alternatives_considered) > 2:
            sources.append(f"Multiple alternative possibilities ({len(input.alternatives_considered)})")

        if input.prediction_type in ["diagnosis", "prognosis"]:
            sources.append("Inherent medical uncertainty in predictions")

        if not sources:
            sources.append("Standard AI model limitations")

        return sources

    def _categorize_uncertainty(self, confidence: float) -> str:
        """Categorize confidence into uncertainty level."""
        if confidence >= 0.9:
            return "very_low"
        elif confidence >= 0.75:
            return "low"
        elif confidence >= 0.5:
            return "moderate"
        elif confidence >= 0.3:
            return "high"
        else:
            return "very_high"

    def _generate_calibrated_statement(self, prediction: str, uncertainty_level: str) -> str:
        """Generate appropriately hedged statement."""
        qualifiers = self.UNCERTAINTY_LANGUAGE[uncertainty_level]
        qualifier = qualifiers[0]

        # Construct calibrated statement
        if prediction.startswith("The ") or prediction.startswith("This "):
            # Transform to add qualifier
            words = prediction.split()
            words.insert(1, qualifier)
            return " ".join(words)
        else:
            return f"This {qualifier} {prediction.lower()}"

    def _generate_guidance(self, uncertainty_level: str, prediction_type: str) -> str:
        """Generate communication guidance."""
        guidance_parts = []

        # Uncertainty level guidance
        if uncertainty_level in ["high", "very_high"]:
            guidance_parts.append("Emphasize significant uncertainty when communicating this assessment.")
            guidance_parts.append("Strongly recommend professional medical consultation.")
        elif uncertainty_level == "moderate":
            guidance_parts.append("Include appropriate hedging language.")
            guidance_parts.append("Suggest verification with healthcare provider.")
        else:
            guidance_parts.append("Confidence is reasonable but always recommend clinical correlation.")

        # Type-specific guidance
        if prediction_type == "diagnosis":
            guidance_parts.append("This is a differential consideration, not a confirmed diagnosis.")
        elif prediction_type == "treatment":
            guidance_parts.append("Treatment decisions require individualized medical assessment.")
        elif prediction_type == "prognosis":
            guidance_parts.append("Prognostic estimates have inherent variability.")

        return " ".join(guidance_parts)
