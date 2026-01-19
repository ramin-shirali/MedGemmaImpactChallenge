"""
MedGemma Agent Framework - Differential Diagnosis Tool

Generates differential diagnoses based on symptoms and clinical data.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class DifferentialDiagnosisInput(ToolInput):
    """Input for differential diagnosis."""
    chief_complaint: str = Field(description="Chief complaint")
    symptoms: List[str] = Field(description="List of symptoms")
    duration: Optional[str] = Field(default=None, description="Duration of symptoms")
    patient_age: Optional[int] = Field(default=None, description="Patient age")
    patient_sex: Optional[str] = Field(default=None, description="Patient sex")
    medical_history: Optional[List[str]] = Field(default=None, description="Past medical history")
    medications: Optional[List[str]] = Field(default=None, description="Current medications")
    vital_signs: Optional[Dict[str, Any]] = Field(default=None, description="Vital signs")
    exam_findings: Optional[List[str]] = Field(default=None, description="Physical exam findings")
    lab_results: Optional[Dict[str, Any]] = Field(default=None, description="Lab results")


class Diagnosis(BaseModel):
    """A differential diagnosis with supporting information."""
    diagnosis: str
    probability: str  # high, moderate, low
    supporting_features: List[str]
    against_features: List[str] = Field(default_factory=list)
    workup_recommended: List[str]
    icd10_code: Optional[str] = None


class DifferentialDiagnosisOutput(ToolOutput):
    """Output for differential diagnosis."""
    differentials: List[Diagnosis] = Field(default_factory=list)
    most_likely: Optional[str] = None
    cannot_miss: List[str] = Field(default_factory=list)
    recommended_workup: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)


# Symptom-diagnosis knowledge base
DIAGNOSIS_PATTERNS = {
    "chest pain": {
        "must_consider": ["acute coronary syndrome", "pulmonary embolism", "aortic dissection", "tension pneumothorax"],
        "common": ["musculoskeletal", "GERD", "costochondritis", "anxiety"],
        "by_feature": {
            "exertional": ["angina", "stable CAD"],
            "pleuritic": ["pulmonary embolism", "pericarditis", "pneumonia", "pleuritis"],
            "tearing": ["aortic dissection"],
            "positional": ["pericarditis", "GERD", "musculoskeletal"],
            "reproducible": ["musculoskeletal", "costochondritis"],
        }
    },
    "shortness of breath": {
        "must_consider": ["pulmonary embolism", "acute coronary syndrome", "pneumothorax"],
        "common": ["asthma", "COPD exacerbation", "pneumonia", "heart failure", "anxiety"],
        "by_feature": {
            "sudden": ["pulmonary embolism", "pneumothorax", "anaphylaxis"],
            "progressive": ["heart failure", "COPD", "interstitial lung disease"],
            "with wheezing": ["asthma", "COPD", "heart failure"],
            "orthopnea": ["heart failure"],
        }
    },
    "abdominal pain": {
        "must_consider": ["appendicitis", "ectopic pregnancy", "aortic aneurysm", "mesenteric ischemia"],
        "common": ["gastritis", "GERD", "constipation", "IBS", "gastroenteritis"],
        "by_feature": {
            "right lower quadrant": ["appendicitis", "ovarian pathology", "ectopic pregnancy"],
            "right upper quadrant": ["cholecystitis", "hepatitis", "pneumonia"],
            "epigastric": ["gastritis", "peptic ulcer", "pancreatitis", "GERD"],
            "left lower quadrant": ["diverticulitis", "ovarian pathology", "constipation"],
            "diffuse": ["gastroenteritis", "SBO", "peritonitis"],
        }
    },
    "headache": {
        "must_consider": ["subarachnoid hemorrhage", "meningitis", "temporal arteritis", "brain tumor"],
        "common": ["tension headache", "migraine", "sinusitis", "medication overuse"],
        "by_feature": {
            "thunderclap": ["subarachnoid hemorrhage", "cerebral venous thrombosis"],
            "worst headache": ["subarachnoid hemorrhage"],
            "fever": ["meningitis", "encephalitis", "sinusitis"],
            "visual changes": ["migraine with aura", "temporal arteritis", "increased ICP"],
            "positional": ["low CSF pressure", "increased ICP"],
        }
    },
    "fever": {
        "must_consider": ["sepsis", "meningitis", "endocarditis"],
        "common": ["viral URI", "UTI", "pneumonia", "skin/soft tissue infection"],
        "by_feature": {
            "with rash": ["viral exanthem", "meningococcemia", "drug reaction"],
            "with cough": ["pneumonia", "bronchitis", "COVID-19", "influenza"],
            "with dysuria": ["UTI", "pyelonephritis"],
            "with confusion": ["meningitis", "sepsis", "UTI in elderly"],
        }
    },
}


class DifferentialDiagnosisTool(BaseTool[DifferentialDiagnosisInput, DifferentialDiagnosisOutput]):
    """Generate differential diagnoses based on clinical presentation."""

    name: ClassVar[str] = "differential_diagnosis"
    description: ClassVar[str] = "Generate ranked differential diagnoses with supporting evidence and recommended workup."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.CLINICAL

    input_class: ClassVar[Type[DifferentialDiagnosisInput]] = DifferentialDiagnosisInput
    output_class: ClassVar[Type[DifferentialDiagnosisOutput]] = DifferentialDiagnosisOutput

    async def execute(self, input: DifferentialDiagnosisInput) -> DifferentialDiagnosisOutput:
        try:
            # Generate differentials based on chief complaint and symptoms
            differentials = self._generate_differentials(input)
            cannot_miss = self._identify_cannot_miss(input)
            red_flags = self._identify_red_flags(input)
            workup = self._recommend_workup(differentials, input)

            most_likely = differentials[0].diagnosis if differentials else None

            return DifferentialDiagnosisOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"differential_count": len(differentials)},
                differentials=differentials,
                most_likely=most_likely,
                cannot_miss=cannot_miss,
                recommended_workup=workup,
                red_flags=red_flags,
                confidence=0.75
            )

        except Exception as e:
            return DifferentialDiagnosisOutput.from_error(str(e))

    def _generate_differentials(self, input: DifferentialDiagnosisInput) -> List[Diagnosis]:
        """Generate ranked differential diagnoses."""
        differentials = []
        complaint_lower = input.chief_complaint.lower()
        symptoms_lower = [s.lower() for s in input.symptoms]

        # Find matching pattern
        matched_pattern = None
        for pattern_key in DIAGNOSIS_PATTERNS:
            if pattern_key in complaint_lower:
                matched_pattern = DIAGNOSIS_PATTERNS[pattern_key]
                break

        if not matched_pattern:
            # Generic differential
            return [Diagnosis(
                diagnosis="Further evaluation needed",
                probability="moderate",
                supporting_features=input.symptoms,
                workup_recommended=["Complete history and physical", "Basic labs"]
            )]

        # Add must-consider diagnoses
        for dx in matched_pattern.get("must_consider", []):
            supporting = self._find_supporting_features(dx, symptoms_lower, input)
            differentials.append(Diagnosis(
                diagnosis=dx,
                probability="moderate" if supporting else "low",
                supporting_features=supporting,
                workup_recommended=self._get_workup_for_diagnosis(dx)
            ))

        # Add common diagnoses
        for dx in matched_pattern.get("common", []):
            supporting = self._find_supporting_features(dx, symptoms_lower, input)
            differentials.append(Diagnosis(
                diagnosis=dx,
                probability="high" if len(supporting) >= 2 else "moderate",
                supporting_features=supporting,
                workup_recommended=self._get_workup_for_diagnosis(dx)
            ))

        # Check feature-based refinement
        for feature, diagnoses in matched_pattern.get("by_feature", {}).items():
            if any(feature in s for s in symptoms_lower):
                for dx in diagnoses:
                    # Upgrade probability if already in list
                    for d in differentials:
                        if d.diagnosis.lower() == dx.lower():
                            d.probability = "high"
                            d.supporting_features.append(f"Feature: {feature}")
                            break

        # Sort by probability
        priority = {"high": 0, "moderate": 1, "low": 2}
        differentials.sort(key=lambda x: priority.get(x.probability, 2))

        return differentials[:10]

    def _find_supporting_features(
        self,
        diagnosis: str,
        symptoms: List[str],
        input: DifferentialDiagnosisInput
    ) -> List[str]:
        """Find symptoms that support a diagnosis."""
        supporting = []
        dx_lower = diagnosis.lower()

        # Age-based support
        if input.patient_age:
            if input.patient_age > 65 and "coronary" in dx_lower:
                supporting.append(f"Age {input.patient_age}")
            if input.patient_age > 50 and "arteritis" in dx_lower:
                supporting.append(f"Age > 50")

        # Symptom matching
        symptom_dx_map = {
            "acute coronary syndrome": ["chest pain", "diaphoresis", "nausea", "jaw pain", "arm pain"],
            "pulmonary embolism": ["dyspnea", "pleuritic", "leg swelling", "hemoptysis"],
            "pneumonia": ["cough", "fever", "productive", "chills"],
            "heart failure": ["edema", "orthopnea", "pnd", "dyspnea"],
        }

        if dx_lower in symptom_dx_map:
            for symptom in symptom_dx_map[dx_lower]:
                if any(symptom in s for s in symptoms):
                    supporting.append(symptom.title())

        return supporting[:5]

    def _identify_cannot_miss(self, input: DifferentialDiagnosisInput) -> List[str]:
        """Identify 'cannot miss' diagnoses."""
        cannot_miss = []
        complaint_lower = input.chief_complaint.lower()

        critical_map = {
            "chest pain": ["Acute MI", "Pulmonary embolism", "Aortic dissection"],
            "shortness of breath": ["Pulmonary embolism", "Tension pneumothorax", "Anaphylaxis"],
            "headache": ["Subarachnoid hemorrhage", "Meningitis", "Intracranial mass"],
            "abdominal pain": ["Ruptured AAA", "Ectopic pregnancy", "Appendicitis"],
            "fever": ["Sepsis", "Meningitis", "Necrotizing fasciitis"],
        }

        for key, diagnoses in critical_map.items():
            if key in complaint_lower:
                cannot_miss.extend(diagnoses)

        return list(set(cannot_miss))

    def _identify_red_flags(self, input: DifferentialDiagnosisInput) -> List[str]:
        """Identify red flag symptoms."""
        red_flags = []
        all_symptoms = " ".join([input.chief_complaint] + input.symptoms).lower()

        flag_patterns = [
            ("worst headache", "Thunderclap headache - consider SAH"),
            ("sudden onset", "Sudden onset - consider vascular emergency"),
            ("tearing", "Tearing pain - consider aortic dissection"),
            ("syncope", "Syncope - consider cardiac/neurological emergency"),
            ("unilateral weakness", "Focal weakness - consider stroke"),
            ("fever + rash", "Fever with rash - consider meningococcemia"),
            ("immunocompromised", "Immunocompromised - broader differential"),
        ]

        for pattern, flag in flag_patterns:
            if pattern in all_symptoms:
                red_flags.append(flag)

        # Vital sign red flags
        if input.vital_signs:
            if input.vital_signs.get("systolic_bp", 120) < 90:
                red_flags.append("Hypotension")
            if input.vital_signs.get("heart_rate", 80) > 120:
                red_flags.append("Tachycardia")
            if input.vital_signs.get("oxygen_saturation", 98) < 92:
                red_flags.append("Hypoxia")

        return red_flags

    def _recommend_workup(
        self,
        differentials: List[Diagnosis],
        input: DifferentialDiagnosisInput
    ) -> List[str]:
        """Recommend workup based on differentials."""
        workup = set()

        for d in differentials[:5]:
            workup.update(d.workup_recommended)

        # Add based on chief complaint
        complaint_lower = input.chief_complaint.lower()
        if "chest pain" in complaint_lower:
            workup.update(["ECG", "Troponin", "Chest X-ray"])
        if "shortness of breath" in complaint_lower:
            workup.update(["Chest X-ray", "BMP", "CBC", "Pulse oximetry"])
        if "abdominal pain" in complaint_lower:
            workup.update(["CBC", "CMP", "Lipase", "Urinalysis"])
        if "headache" in complaint_lower:
            workup.update(["Neurological exam", "Consider CT head"])

        return list(workup)[:10]

    def _get_workup_for_diagnosis(self, diagnosis: str) -> List[str]:
        """Get recommended workup for specific diagnosis."""
        workup_map = {
            "acute coronary syndrome": ["ECG", "Troponin", "CXR", "BMP"],
            "pulmonary embolism": ["D-dimer", "CT-PA", "ECG", "ABG"],
            "aortic dissection": ["CT angiography", "CXR", "Type and screen"],
            "pneumonia": ["CXR", "CBC", "BMP", "Procalcitonin"],
            "heart failure": ["BNP", "CXR", "Echo", "BMP"],
            "meningitis": ["LP", "Blood cultures", "CBC", "CT head"],
            "appendicitis": ["CT abdomen", "CBC", "CMP", "Urinalysis"],
        }
        return workup_map.get(diagnosis.lower(), ["Clinical correlation"])
