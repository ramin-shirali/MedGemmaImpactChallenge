"""
MedGemma Agent Framework - Risk Assessment Tool

Calculates disease-specific risk scores and prognosis.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class RiskAssessmentInput(ToolInput):
    """Input for risk assessment."""
    assessment_type: str = Field(description="Type of risk assessment (cardiovascular, falls, readmission, etc.)")
    patient_data: Dict[str, Any] = Field(description="Patient data for risk calculation")


class RiskAssessmentOutput(ToolOutput):
    """Output for risk assessment."""
    risk_score: Optional[float] = None
    risk_category: str = ""
    risk_percentage: Optional[float] = None
    interpretation: str = ""
    risk_factors_present: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    score_name: str = ""


class RiskAssessmentTool(BaseTool[RiskAssessmentInput, RiskAssessmentOutput]):
    """Calculate disease-specific risk scores."""

    name: ClassVar[str] = "risk_assessment"
    description: ClassVar[str] = "Calculate cardiovascular, falls, readmission, and other disease-specific risk scores."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.CLINICAL

    input_class: ClassVar[Type[RiskAssessmentInput]] = RiskAssessmentInput
    output_class: ClassVar[Type[RiskAssessmentOutput]] = RiskAssessmentOutput

    ASSESSMENTS = {
        'cardiovascular': '_assess_cv_risk',
        'ascvd': '_assess_cv_risk',
        'framingham': '_assess_cv_risk',
        'falls': '_assess_falls_risk',
        'morse': '_assess_falls_risk',
        'readmission': '_assess_readmission_risk',
        'hospital': '_assess_readmission_risk',
        'pressure_ulcer': '_assess_pressure_ulcer_risk',
        'braden': '_assess_pressure_ulcer_risk',
        'bleeding': '_assess_bleeding_risk',
        'hasbled': '_assess_bleeding_risk',
    }

    async def execute(self, input: RiskAssessmentInput) -> RiskAssessmentOutput:
        try:
            assessment_key = input.assessment_type.lower().replace(' ', '_').replace('-', '_')

            method_name = None
            for key, method in self.ASSESSMENTS.items():
                if key in assessment_key:
                    method_name = method
                    break

            if not method_name:
                available = list(set(self.ASSESSMENTS.keys()))
                return RiskAssessmentOutput.from_error(f"Unknown assessment. Available: {available}")

            method = getattr(self, method_name)
            return method(input.patient_data)

        except Exception as e:
            return RiskAssessmentOutput.from_error(str(e))

    def _assess_cv_risk(self, data: Dict) -> RiskAssessmentOutput:
        """10-year ASCVD risk assessment."""
        factors = []
        score = 0

        age = data.get('age', 50)
        sex = data.get('sex', 'male').lower()
        total_chol = data.get('total_cholesterol', 200)
        hdl = data.get('hdl', 50)
        sbp = data.get('systolic_bp', 120)
        diabetes = data.get('diabetes', False)
        smoker = data.get('smoker', False)
        on_bp_meds = data.get('on_bp_medication', False)

        # Simplified risk calculation
        if age >= 40:
            score += (age - 40) * 0.5
            factors.append(f"Age {age}")
        if sex == 'male':
            score += 5
        if total_chol > 200:
            score += (total_chol - 200) * 0.02
            factors.append(f"Total cholesterol {total_chol}")
        if hdl < 40:
            score += 5
            factors.append(f"Low HDL {hdl}")
        if sbp > 120:
            score += (sbp - 120) * 0.1
            factors.append(f"Elevated BP {sbp}")
        if diabetes:
            score += 10
            factors.append("Diabetes")
        if smoker:
            score += 8
            factors.append("Current smoker")

        risk_pct = min(score, 50)

        if risk_pct < 5:
            category = "Low"
            recommendations = ["Maintain healthy lifestyle", "Rescreen in 5 years"]
        elif risk_pct < 7.5:
            category = "Borderline"
            recommendations = ["Lifestyle modifications", "Consider statin if risk enhancers present"]
        elif risk_pct < 20:
            category = "Intermediate"
            recommendations = ["Moderate-intensity statin recommended", "Lifestyle modifications"]
        else:
            category = "High"
            recommendations = ["High-intensity statin recommended", "Aggressive risk factor management"]

        return RiskAssessmentOutput(
            success=True, status=ToolStatus.SUCCESS,
            data={"ascvd_risk": risk_pct},
            risk_score=round(score, 1),
            risk_category=category,
            risk_percentage=round(risk_pct, 1),
            interpretation=f"{round(risk_pct, 1)}% 10-year ASCVD risk ({category})",
            risk_factors_present=factors,
            recommendations=recommendations,
            score_name="ASCVD Risk Score",
            confidence=0.8
        )

    def _assess_falls_risk(self, data: Dict) -> RiskAssessmentOutput:
        """Morse Fall Scale assessment."""
        score = 0
        factors = []

        if data.get('fall_history'):
            score += 25
            factors.append("History of falls")
        if data.get('secondary_diagnosis'):
            score += 15
            factors.append("Secondary diagnosis")
        if data.get('ambulatory_aid'):
            aid = data['ambulatory_aid']
            if aid == 'furniture':
                score += 30
            elif aid in ['cane', 'walker', 'crutches']:
                score += 15
            factors.append(f"Uses {aid}")
        if data.get('iv_access'):
            score += 20
            factors.append("IV/Heparin lock")
        if data.get('impaired_gait'):
            score += 20
            factors.append("Impaired gait")
        if data.get('impaired_mental_status'):
            score += 15
            factors.append("Impaired mental status")

        if score <= 24:
            category = "Low"
            recommendations = ["Standard fall precautions"]
        elif score <= 44:
            category = "Moderate"
            recommendations = ["Implement fall precautions", "Fall risk signage", "Non-skid footwear"]
        else:
            category = "High"
            recommendations = ["High fall risk interventions", "Bed/chair alarm", "Close supervision", "Fall risk bracelet"]

        return RiskAssessmentOutput(
            success=True, status=ToolStatus.SUCCESS,
            data={"morse_score": score},
            risk_score=score,
            risk_category=category,
            interpretation=f"Morse Fall Score: {score} ({category} risk)",
            risk_factors_present=factors,
            recommendations=recommendations,
            score_name="Morse Fall Scale",
            confidence=0.9
        )

    def _assess_readmission_risk(self, data: Dict) -> RiskAssessmentOutput:
        """Hospital readmission risk (HOSPITAL score)."""
        score = 0
        factors = []

        if data.get('hemoglobin', 14) < 12:
            score += 1
            factors.append("Low hemoglobin")
        if data.get('discharge_from_oncology'):
            score += 2
            factors.append("Oncology service")
        if data.get('sodium', 140) < 135:
            score += 1
            factors.append("Low sodium")
        if data.get('procedure_performed'):
            score += 1
            factors.append("Procedure during admission")
        if data.get('index_admission_type') == 'emergency':
            score += 1
            factors.append("Emergency admission")
        if data.get('admissions_past_year', 0) > 0:
            score += min(data['admissions_past_year'], 5)
            factors.append(f"{data['admissions_past_year']} admissions in past year")
        if data.get('length_of_stay', 1) >= 5:
            score += 2
            factors.append("Length of stay ≥5 days")

        if score <= 4:
            category = "Low"
            risk_pct = 5.8
        elif score <= 6:
            category = "Intermediate"
            risk_pct = 11.9
        else:
            category = "High"
            risk_pct = 23.8

        recommendations = []
        if category in ["Intermediate", "High"]:
            recommendations = [
                "Schedule follow-up within 7 days",
                "Medication reconciliation",
                "Patient education on warning signs",
                "Consider transitional care program"
            ]

        return RiskAssessmentOutput(
            success=True, status=ToolStatus.SUCCESS,
            data={"hospital_score": score},
            risk_score=score,
            risk_category=category,
            risk_percentage=risk_pct,
            interpretation=f"HOSPITAL Score: {score} ({risk_pct}% 30-day readmission risk)",
            risk_factors_present=factors,
            recommendations=recommendations,
            score_name="HOSPITAL Score",
            confidence=0.85
        )

    def _assess_pressure_ulcer_risk(self, data: Dict) -> RiskAssessmentOutput:
        """Braden Scale for pressure ulcer risk."""
        # Each subscale 1-4, total 6-23
        sensory = data.get('sensory_perception', 4)
        moisture = data.get('moisture', 4)
        activity = data.get('activity', 4)
        mobility = data.get('mobility', 4)
        nutrition = data.get('nutrition', 4)
        friction = data.get('friction_shear', 3)

        score = sensory + moisture + activity + mobility + nutrition + friction
        factors = []

        if sensory <= 2:
            factors.append("Impaired sensory perception")
        if moisture <= 2:
            factors.append("Excessive moisture")
        if activity <= 2:
            factors.append("Limited activity")
        if mobility <= 2:
            factors.append("Limited mobility")
        if nutrition <= 2:
            factors.append("Inadequate nutrition")

        if score <= 9:
            category = "Very High"
        elif score <= 12:
            category = "High"
        elif score <= 14:
            category = "Moderate"
        elif score <= 18:
            category = "Mild"
        else:
            category = "No Risk"

        recommendations = []
        if category in ["Very High", "High"]:
            recommendations = [
                "Pressure redistribution surface",
                "Reposition every 2 hours",
                "Skin inspection every shift",
                "Nutrition consult"
            ]

        return RiskAssessmentOutput(
            success=True, status=ToolStatus.SUCCESS,
            data={"braden_score": score},
            risk_score=score,
            risk_category=category,
            interpretation=f"Braden Score: {score} ({category} risk)",
            risk_factors_present=factors,
            recommendations=recommendations,
            score_name="Braden Scale",
            confidence=0.9
        )

    def _assess_bleeding_risk(self, data: Dict) -> RiskAssessmentOutput:
        """HAS-BLED bleeding risk score."""
        score = 0
        factors = []

        if data.get('hypertension'):
            score += 1
            factors.append("Hypertension")
        if data.get('renal_disease') or data.get('liver_disease'):
            if data.get('renal_disease'):
                score += 1
                factors.append("Renal disease")
            if data.get('liver_disease'):
                score += 1
                factors.append("Liver disease")
        if data.get('stroke_history'):
            score += 1
            factors.append("Stroke history")
        if data.get('bleeding_history'):
            score += 1
            factors.append("Prior bleeding")
        if data.get('labile_inr'):
            score += 1
            factors.append("Labile INR")
        if data.get('age', 0) > 65:
            score += 1
            factors.append("Age > 65")
        if data.get('antiplatelet') or data.get('nsaid'):
            score += 1
            factors.append("Antiplatelet/NSAID use")
        if data.get('alcohol'):
            score += 1
            factors.append("Alcohol use")

        if score <= 1:
            category = "Low"
            risk_pct = 1.0
        elif score == 2:
            category = "Moderate"
            risk_pct = 1.9
        else:
            category = "High"
            risk_pct = 4.0 + (score - 3) * 1.5

        return RiskAssessmentOutput(
            success=True, status=ToolStatus.SUCCESS,
            data={"hasbled_score": score},
            risk_score=score,
            risk_category=category,
            risk_percentage=risk_pct,
            interpretation=f"HAS-BLED Score: {score} ({category} bleeding risk)",
            risk_factors_present=factors,
            recommendations=["Modify reversible risk factors", "Regular monitoring on anticoagulation"],
            score_name="HAS-BLED Score",
            confidence=0.85
        )
