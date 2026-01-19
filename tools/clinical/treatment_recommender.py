"""
MedGemma Agent Framework - Treatment Recommender Tool

Provides evidence-based treatment recommendations.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class TreatmentRecommenderInput(ToolInput):
    """Input for treatment recommender."""
    diagnosis: str = Field(description="Primary diagnosis")
    patient_age: Optional[int] = Field(default=None, description="Patient age")
    patient_sex: Optional[str] = Field(default=None, description="Patient sex")
    comorbidities: Optional[List[str]] = Field(default=None, description="Comorbidities")
    allergies: Optional[List[str]] = Field(default=None, description="Drug allergies")
    current_medications: Optional[List[str]] = Field(default=None, description="Current medications")
    renal_function: Optional[str] = Field(default=None, description="Renal function (eGFR or stage)")
    liver_function: Optional[str] = Field(default=None, description="Liver function")
    pregnancy: Optional[bool] = Field(default=False, description="Pregnancy status")


class Medication(BaseModel):
    """Medication recommendation."""
    name: str
    dose: str
    frequency: str
    duration: Optional[str] = None
    route: str = "oral"
    notes: Optional[str] = None


class TreatmentPlan(BaseModel):
    """Treatment plan with medications and non-pharmacological interventions."""
    medications: List[Medication] = Field(default_factory=list)
    non_pharmacological: List[str] = Field(default_factory=list)
    monitoring: List[str] = Field(default_factory=list)
    follow_up: Optional[str] = None
    referrals: List[str] = Field(default_factory=list)


class TreatmentRecommenderOutput(ToolOutput):
    """Output for treatment recommender."""
    treatment_plan: Optional[TreatmentPlan] = None
    first_line: List[str] = Field(default_factory=list)
    alternatives: List[str] = Field(default_factory=list)
    contraindicated: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    guideline_source: Optional[str] = None


class TreatmentRecommenderTool(BaseTool[TreatmentRecommenderInput, TreatmentRecommenderOutput]):
    """Recommend evidence-based treatments."""

    name: ClassVar[str] = "treatment_recommender"
    description: ClassVar[str] = "Provide evidence-based treatment recommendations considering patient factors and guidelines."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.CLINICAL

    input_class: ClassVar[Type[TreatmentRecommenderInput]] = TreatmentRecommenderInput
    output_class: ClassVar[Type[TreatmentRecommenderOutput]] = TreatmentRecommenderOutput

    # Treatment database
    TREATMENTS = {
        "hypertension": {
            "first_line": [
                Medication(name="Lisinopril", dose="10mg", frequency="daily", notes="Titrate to 40mg"),
                Medication(name="Amlodipine", dose="5mg", frequency="daily", notes="Titrate to 10mg"),
                Medication(name="Hydrochlorothiazide", dose="12.5-25mg", frequency="daily"),
            ],
            "alternatives": ["Losartan", "Chlorthalidone", "Metoprolol"],
            "non_pharm": ["DASH diet", "Sodium restriction <2g/day", "Exercise 150 min/week", "Weight loss if overweight"],
            "monitoring": ["Blood pressure", "Renal function", "Electrolytes"],
            "guideline": "ACC/AHA 2017",
        },
        "type 2 diabetes": {
            "first_line": [
                Medication(name="Metformin", dose="500mg", frequency="twice daily", notes="Titrate to 1000mg BID"),
            ],
            "alternatives": ["GLP-1 agonist", "SGLT2 inhibitor", "DPP-4 inhibitor"],
            "non_pharm": ["Medical nutrition therapy", "Exercise 150 min/week", "Weight management", "Diabetes education"],
            "monitoring": ["HbA1c q3 months", "Fasting glucose", "Renal function annually", "Annual eye exam"],
            "guideline": "ADA Standards of Care 2024",
        },
        "community acquired pneumonia": {
            "first_line": [
                Medication(name="Amoxicillin", dose="1g", frequency="three times daily", duration="5-7 days"),
            ],
            "alternatives": ["Doxycycline 100mg BID", "Azithromycin 500mg day 1, then 250mg x4 days"],
            "inpatient": [
                Medication(name="Ceftriaxone", dose="1g", frequency="daily", route="IV"),
                Medication(name="Azithromycin", dose="500mg", frequency="daily"),
            ],
            "non_pharm": ["Hydration", "Rest", "Smoking cessation"],
            "monitoring": ["Temperature", "Respiratory status", "Follow-up CXR if not improving"],
            "guideline": "IDSA/ATS 2019",
        },
        "heart failure": {
            "first_line": [
                Medication(name="Lisinopril", dose="5mg", frequency="daily", notes="Titrate to 40mg"),
                Medication(name="Carvedilol", dose="3.125mg", frequency="twice daily", notes="Titrate to 25mg BID"),
                Medication(name="Spironolactone", dose="25mg", frequency="daily"),
                Medication(name="Dapagliflozin", dose="10mg", frequency="daily", notes="SGLT2i for HFrEF"),
            ],
            "alternatives": ["Sacubitril/valsartan (replace ACEi)", "Metoprolol succinate", "Eplerenone"],
            "non_pharm": ["Sodium restriction <2g/day", "Fluid restriction 1.5-2L/day", "Daily weights", "Exercise training"],
            "monitoring": ["Daily weights", "Renal function", "Potassium", "Blood pressure"],
            "guideline": "ACC/AHA HF Guidelines 2022",
        },
        "atrial fibrillation": {
            "first_line": [
                Medication(name="Apixaban", dose="5mg", frequency="twice daily", notes="2.5mg BID if age ≥80, weight ≤60kg, or Cr ≥1.5"),
            ],
            "rate_control": [
                Medication(name="Metoprolol", dose="25-100mg", frequency="twice daily"),
            ],
            "alternatives": ["Rivaroxaban", "Dabigatran", "Warfarin (if mechanical valve)"],
            "non_pharm": ["Treat underlying cause", "Cardioversion if appropriate", "Catheter ablation consideration"],
            "monitoring": ["Heart rate", "Renal function", "Signs of bleeding"],
            "guideline": "ACC/AHA/HRS AF Guidelines",
        },
    }

    async def execute(self, input: TreatmentRecommenderInput) -> TreatmentRecommenderOutput:
        try:
            dx_lower = input.diagnosis.lower()
            warnings = []
            contraindicated = []

            # Find matching treatment
            matched_treatment = None
            for key in self.TREATMENTS:
                if key in dx_lower or dx_lower in key:
                    matched_treatment = self.TREATMENTS[key]
                    break

            if not matched_treatment:
                return TreatmentRecommenderOutput(
                    success=True,
                    status=ToolStatus.SUCCESS,
                    data={"diagnosis": input.diagnosis},
                    warnings=["No specific guidelines in database. Recommend clinical judgment and specialty consultation."],
                    confidence=0.5
                )

            # Check contraindications
            medications = list(matched_treatment["first_line"])
            alternatives = matched_treatment.get("alternatives", [])

            if input.allergies:
                for med in medications[:]:
                    if any(allergy.lower() in med.name.lower() for allergy in input.allergies):
                        contraindicated.append(f"{med.name} - patient allergy")
                        medications.remove(med)

            if input.pregnancy:
                # Remove pregnancy contraindicated meds
                pregnancy_avoid = ["lisinopril", "losartan", "warfarin", "methotrexate", "statins"]
                for med in medications[:]:
                    if any(avoid in med.name.lower() for avoid in pregnancy_avoid):
                        contraindicated.append(f"{med.name} - contraindicated in pregnancy")
                        medications.remove(med)
                        warnings.append(f"Pregnancy: {med.name} contraindicated")

            if input.renal_function:
                if "stage 4" in input.renal_function.lower() or "stage 5" in input.renal_function.lower():
                    for med in medications[:]:
                        if "metformin" in med.name.lower():
                            contraindicated.append("Metformin - contraindicated in CKD 4-5")
                            medications.remove(med)
                    warnings.append("Adjust doses for renal impairment")

            # Build treatment plan
            plan = TreatmentPlan(
                medications=medications,
                non_pharmacological=matched_treatment.get("non_pharm", []),
                monitoring=matched_treatment.get("monitoring", []),
                follow_up="2-4 weeks for reassessment"
            )

            return TreatmentRecommenderOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"medication_count": len(medications)},
                treatment_plan=plan,
                first_line=[m.name for m in medications],
                alternatives=alternatives,
                contraindicated=contraindicated,
                warnings=warnings,
                guideline_source=matched_treatment.get("guideline"),
                confidence=0.8
            )

        except Exception as e:
            return TreatmentRecommenderOutput.from_error(str(e))
