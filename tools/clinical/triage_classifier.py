"""
MedGemma Agent Framework - Triage Classifier Tool

Classifies patient urgency using Emergency Severity Index (ESI) criteria.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class TriageInput(ToolInput):
    """Input for triage classification."""
    chief_complaint: str = Field(description="Chief complaint")
    symptoms: Optional[List[str]] = Field(default=None, description="Presenting symptoms")
    vital_signs: Optional[Dict[str, Any]] = Field(default=None, description="Vital signs")
    mental_status: Optional[str] = Field(default="alert", description="Mental status (alert, confused, unresponsive)")
    pain_score: Optional[int] = Field(default=None, ge=0, le=10, description="Pain score 0-10")
    age: Optional[int] = Field(default=None, description="Patient age")
    medical_history: Optional[List[str]] = Field(default=None, description="Relevant medical history")


class TriageOutput(ToolOutput):
    """Output for triage classification."""
    esi_level: int = Field(ge=1, le=5, description="ESI level 1-5")
    esi_description: str = ""
    urgency: str = ""
    rationale: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    recommended_resources: int = Field(default=0, description="Expected resources needed")
    time_to_provider: str = ""


class TriageClassifierTool(BaseTool[TriageInput, TriageOutput]):
    """Classify patient urgency using ESI triage system."""

    name: ClassVar[str] = "triage_classifier"
    description: ClassVar[str] = "Classify patient urgency using Emergency Severity Index (ESI) levels 1-5."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.CLINICAL

    input_class: ClassVar[Type[TriageInput]] = TriageInput
    output_class: ClassVar[Type[TriageOutput]] = TriageOutput

    # ESI level descriptions
    ESI_LEVELS = {
        1: ("Resuscitation", "Immediate", "Requires immediate life-saving intervention"),
        2: ("Emergent", "Within 10 minutes", "High risk, confused/lethargic, severe pain"),
        3: ("Urgent", "Within 30 minutes", "Stable, multiple resources expected"),
        4: ("Less Urgent", "Within 60 minutes", "Stable, one resource expected"),
        5: ("Non-Urgent", "Within 120 minutes", "Stable, no resources expected"),
    }

    # Critical symptoms requiring ESI-1
    ESI1_CRITERIA = [
        "cardiac arrest", "respiratory arrest", "unresponsive", "intubated",
        "severe respiratory distress", "pulseless", "apneic"
    ]

    # High-risk symptoms for ESI-2
    ESI2_CRITERIA = [
        "chest pain", "stroke symptoms", "severe abdominal pain", "suicidal",
        "altered mental status", "high-risk mechanism", "severe bleeding"
    ]

    async def execute(self, input: TriageInput) -> TriageOutput:
        try:
            rationale = []
            red_flags = []

            # Check for ESI-1 criteria
            complaint_lower = input.chief_complaint.lower()
            symptoms_lower = [s.lower() for s in (input.symptoms or [])]
            all_symptoms = complaint_lower + " " + " ".join(symptoms_lower)

            # ESI Level 1: Immediate life-saving intervention needed
            if any(crit in all_symptoms for crit in self.ESI1_CRITERIA):
                rationale.append("Critical presentation requiring immediate intervention")
                return self._build_output(1, rationale, red_flags, ["Critical condition detected"])

            if input.mental_status and input.mental_status.lower() == "unresponsive":
                rationale.append("Patient unresponsive")
                return self._build_output(1, rationale, red_flags, ["Unresponsive patient"])

            # Check vital signs for instability
            if input.vital_signs:
                vitals_flags = self._check_vitals(input.vital_signs, input.age)
                if vitals_flags.get("critical"):
                    rationale.extend(vitals_flags["reasons"])
                    return self._build_output(1, rationale, red_flags, vitals_flags["reasons"])
                if vitals_flags.get("abnormal"):
                    red_flags.extend(vitals_flags["reasons"])

            # ESI Level 2: High-risk situation
            if any(crit in all_symptoms for crit in self.ESI2_CRITERIA):
                rationale.append("High-risk presentation")
                return self._build_output(2, rationale, red_flags)

            if input.mental_status and input.mental_status.lower() in ["confused", "lethargic"]:
                rationale.append("Altered mental status")
                return self._build_output(2, rationale, red_flags)

            if input.pain_score and input.pain_score >= 7:
                rationale.append(f"Severe pain (score: {input.pain_score}/10)")
                return self._build_output(2, rationale, red_flags)

            # ESI Levels 3-5: Based on expected resources
            resources = self._estimate_resources(input)
            rationale.append(f"Expected resources: {resources}")

            if resources >= 2:
                return self._build_output(3, rationale, red_flags, resources=resources)
            elif resources == 1:
                return self._build_output(4, rationale, red_flags, resources=resources)
            else:
                return self._build_output(5, rationale, red_flags, resources=resources)

        except Exception as e:
            return TriageOutput.from_error(str(e))

    def _build_output(
        self,
        level: int,
        rationale: List[str],
        red_flags: List[str],
        additional_flags: List[str] = None,
        resources: int = 0
    ) -> TriageOutput:
        """Build triage output."""
        description, time, explanation = self.ESI_LEVELS[level]
        red_flags.extend(additional_flags or [])

        urgency_map = {1: "Critical", 2: "Emergency", 3: "Urgent", 4: "Semi-Urgent", 5: "Non-Urgent"}

        return TriageOutput(
            success=True,
            status=ToolStatus.SUCCESS,
            data={"esi_level": level},
            esi_level=level,
            esi_description=f"ESI-{level}: {description}",
            urgency=urgency_map[level],
            rationale=rationale + [explanation],
            red_flags=red_flags,
            recommended_resources=resources if level >= 3 else level,
            time_to_provider=time,
            confidence=0.85 if level <= 2 else 0.8
        )

    def _check_vitals(self, vitals: Dict[str, Any], age: Optional[int]) -> Dict[str, Any]:
        """Check vital signs for abnormalities."""
        result = {"critical": False, "abnormal": False, "reasons": []}

        hr = vitals.get("heart_rate")
        sbp = vitals.get("systolic_bp") or vitals.get("sbp")
        rr = vitals.get("respiratory_rate") or vitals.get("rr")
        spo2 = vitals.get("oxygen_saturation") or vitals.get("spo2")
        temp = vitals.get("temperature")

        # Critical vital signs
        if sbp and sbp < 90:
            result["critical"] = True
            result["reasons"].append(f"Hypotension (SBP {sbp})")
        if spo2 and spo2 < 90:
            result["critical"] = True
            result["reasons"].append(f"Severe hypoxia (SpO2 {spo2}%)")
        if rr and rr < 8:
            result["critical"] = True
            result["reasons"].append(f"Bradypnea (RR {rr})")

        # Abnormal vital signs
        if hr and (hr < 50 or hr > 120):
            result["abnormal"] = True
            result["reasons"].append(f"Abnormal heart rate ({hr})")
        if rr and rr > 24:
            result["abnormal"] = True
            result["reasons"].append(f"Tachypnea (RR {rr})")
        if temp and (temp < 35 or temp > 39):
            result["abnormal"] = True
            result["reasons"].append(f"Abnormal temperature ({temp}°C)")

        return result

    def _estimate_resources(self, input: TriageInput) -> int:
        """Estimate number of resources needed."""
        resources = 0
        complaint_lower = input.chief_complaint.lower()

        # Lab work indicators
        lab_indicators = ["infection", "fever", "weakness", "fatigue", "diabetes", "bleeding"]
        if any(ind in complaint_lower for ind in lab_indicators):
            resources += 1

        # Imaging indicators
        imaging_indicators = ["pain", "injury", "fall", "trauma", "swelling"]
        if any(ind in complaint_lower for ind in imaging_indicators):
            resources += 1

        # IV/medication indicators
        iv_indicators = ["dehydration", "vomiting", "pain", "infection"]
        if any(ind in complaint_lower for ind in iv_indicators):
            resources += 1

        return min(resources, 3)
