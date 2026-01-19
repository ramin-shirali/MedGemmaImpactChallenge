"""
MedGemma Agent Framework - Safety Checker Tool

Validates outputs for safety and blocks dangerous content.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class SafetyCheckerInput(ToolInput):
    """Input for safety checker."""
    content: str = Field(description="Content to check")
    content_type: str = Field(default="general", description="Content type: general, diagnosis, treatment, medication")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class SafetyCheckerOutput(ToolOutput):
    """Output for safety checker."""
    is_safe: bool = True
    risk_level: str = "low"  # low, medium, high, critical
    issues_found: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    filtered_content: Optional[str] = None
    blocked: bool = False


class SafetyCheckerTool(BaseTool[SafetyCheckerInput, SafetyCheckerOutput]):
    """Check content for safety issues and dangerous advice."""

    name: ClassVar[str] = "safety_checker"
    description: ClassVar[str] = "Validate AI outputs for safety, block dangerous medical advice, and ensure appropriate disclaimers."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.SAFETY

    input_class: ClassVar[Type[SafetyCheckerInput]] = SafetyCheckerInput
    output_class: ClassVar[Type[SafetyCheckerOutput]] = SafetyCheckerOutput

    # Dangerous patterns to block
    CRITICAL_PATTERNS = [
        "stop taking your medication",
        "don't see a doctor",
        "ignore these symptoms",
        "this is definitely",
        "you definitely have",
        "100% certain",
        "guaranteed cure",
        "no need for professional",
        "self-treat",
        "don't go to the hospital",
        "stop your treatment",
    ]

    # Warning patterns
    WARNING_PATTERNS = [
        "might be cancer",
        "probably fatal",
        "you will die",
        "terminal",
        "no hope",
        "untreatable",
        "specific dose",
        "take this medication",
        "prescribe",
    ]

    # Required disclaimers for certain content types
    REQUIRED_DISCLAIMERS = {
        "diagnosis": "This is not a diagnosis. Please consult a healthcare provider.",
        "treatment": "Treatment recommendations should be discussed with your doctor.",
        "medication": "Do not start or stop medications without consulting a healthcare provider.",
    }

    async def execute(self, input: SafetyCheckerInput) -> SafetyCheckerOutput:
        try:
            content_lower = input.content.lower()
            issues = []
            recommendations = []
            risk_level = "low"
            blocked = False

            # Check for critical patterns
            for pattern in self.CRITICAL_PATTERNS:
                if pattern in content_lower:
                    issues.append(f"Dangerous advice detected: '{pattern}'")
                    risk_level = "critical"
                    blocked = True

            # Check for warning patterns
            for pattern in self.WARNING_PATTERNS:
                if pattern in content_lower:
                    issues.append(f"Potentially concerning content: '{pattern}'")
                    if risk_level != "critical":
                        risk_level = "high"

            # Check for missing disclaimers
            if input.content_type in self.REQUIRED_DISCLAIMERS:
                disclaimer = self.REQUIRED_DISCLAIMERS[input.content_type]
                if disclaimer.lower() not in content_lower:
                    issues.append(f"Missing required disclaimer for {input.content_type}")
                    recommendations.append(f"Add disclaimer: {disclaimer}")
                    if risk_level == "low":
                        risk_level = "medium"

            # Check for absolute statements
            absolute_patterns = ["always", "never", "must", "definitely", "certainly"]
            for pattern in absolute_patterns:
                if pattern in content_lower and any(med in content_lower for med in ["medication", "drug", "treatment"]):
                    issues.append(f"Absolute statement in medical context: '{pattern}'")
                    recommendations.append("Use qualifying language like 'may', 'might', 'could'")

            # Generate filtered content if issues found
            filtered_content = None
            if issues and not blocked:
                filtered_content = self._add_safety_wrapper(input.content, input.content_type)

            is_safe = len(issues) == 0

            return SafetyCheckerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"issues_count": len(issues)},
                is_safe=is_safe,
                risk_level=risk_level,
                issues_found=issues,
                recommendations=recommendations,
                filtered_content=filtered_content,
                blocked=blocked,
                confidence=0.9
            )

        except Exception as e:
            return SafetyCheckerOutput.from_error(str(e))

    def _add_safety_wrapper(self, content: str, content_type: str) -> str:
        """Add safety disclaimers to content."""
        disclaimer = self.REQUIRED_DISCLAIMERS.get(content_type, "")

        prefix = "**IMPORTANT NOTICE**: This information is for educational purposes only. "
        suffix = f"\n\n*{disclaimer}*" if disclaimer else ""

        return f"{prefix}\n\n{content}{suffix}"
