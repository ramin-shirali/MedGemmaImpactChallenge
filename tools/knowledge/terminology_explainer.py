"""
MedGemma Agent Framework - Medical Terminology Explainer Tool

Explains medical terminology in plain language using MedGemma.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class TerminologyExplainerInput(ToolInput):
    """Input for terminology explainer."""
    term: str = Field(description="Medical term to explain")
    context: Optional[str] = Field(default=None, description="Clinical context")
    audience: str = Field(default="patient", description="Target audience (patient, medical_student, clinician)")


class TerminologyExplainerOutput(ToolOutput):
    """Output for terminology explainer."""
    term: str = ""
    definition: Optional[str] = None
    pronunciation: Optional[str] = None
    etymology: Optional[str] = None
    plain_language: Optional[str] = None
    related_terms: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


# Medical terminology database
MEDICAL_TERMS = {
    "hypertension": {
        "definition": "Persistently elevated blood pressure in the arteries",
        "pronunciation": "hy-per-TEN-shun",
        "etymology": "Greek: hyper (over) + Latin: tensio (tension)",
        "plain_language": "High blood pressure - when the force of blood pushing against your artery walls is consistently too high",
        "related": ["blood pressure", "systolic", "diastolic", "antihypertensive"],
        "examples": ["Stage 1 hypertension: 130-139/80-89 mmHg", "Stage 2 hypertension: ≥140/90 mmHg"]
    },
    "myocardial infarction": {
        "definition": "Death of heart muscle tissue due to prolonged ischemia",
        "pronunciation": "my-oh-KAR-dee-al in-FARK-shun",
        "etymology": "Greek: myo (muscle) + kardia (heart) + Latin: infarctus (stuffed in)",
        "plain_language": "Heart attack - when blood flow to part of your heart muscle is blocked, causing tissue damage",
        "related": ["coronary artery disease", "troponin", "STEMI", "NSTEMI"],
        "examples": ["STEMI: ST-elevation myocardial infarction", "NSTEMI: Non-ST-elevation myocardial infarction"]
    },
    "pneumonia": {
        "definition": "Infection that inflames the air sacs in one or both lungs",
        "pronunciation": "noo-MOH-nyuh",
        "etymology": "Greek: pneumon (lung) + -ia (condition)",
        "plain_language": "A lung infection that fills the air sacs with fluid or pus, making it hard to breathe",
        "related": ["bronchitis", "consolidation", "infiltrate", "respiratory failure"],
        "examples": ["Community-acquired pneumonia", "Hospital-acquired pneumonia", "Aspiration pneumonia"]
    },
    "diabetes mellitus": {
        "definition": "A group of metabolic diseases characterized by high blood glucose levels",
        "pronunciation": "dy-uh-BEE-teez meh-LY-tus",
        "etymology": "Greek: diabetes (siphon) + Latin: mellitus (honey-sweet)",
        "plain_language": "A condition where your body can't properly control blood sugar levels, either due to lack of insulin or resistance to it",
        "related": ["hyperglycemia", "insulin", "HbA1c", "glucose"],
        "examples": ["Type 1 diabetes", "Type 2 diabetes", "Gestational diabetes"]
    },
    "tachycardia": {
        "definition": "Heart rate exceeding 100 beats per minute at rest",
        "pronunciation": "tak-ih-KAR-dee-uh",
        "etymology": "Greek: tachys (fast) + kardia (heart)",
        "plain_language": "A fast heartbeat - when your heart beats more than 100 times per minute while resting",
        "related": ["arrhythmia", "palpitations", "bradycardia", "heart rate"],
        "examples": ["Sinus tachycardia", "Atrial fibrillation", "Ventricular tachycardia"]
    },
    "dyspnea": {
        "definition": "Subjective sensation of difficulty breathing or breathlessness",
        "pronunciation": "DISP-nee-uh",
        "etymology": "Greek: dys (difficult) + pnoia (breathing)",
        "plain_language": "Shortness of breath - feeling like you can't get enough air",
        "related": ["respiratory distress", "orthopnea", "tachypnea"],
        "examples": ["Dyspnea on exertion", "Paroxysmal nocturnal dyspnea"]
    },
    "edema": {
        "definition": "Abnormal accumulation of fluid in interstitial spaces",
        "pronunciation": "eh-DEE-muh",
        "etymology": "Greek: oidema (swelling)",
        "plain_language": "Swelling caused by fluid trapped in your body's tissues",
        "related": ["swelling", "peripheral edema", "pulmonary edema", "anasarca"],
        "examples": ["Pitting edema", "Non-pitting edema", "Pedal edema"]
    },
}


class TerminologyExplainerTool(BaseTool[TerminologyExplainerInput, TerminologyExplainerOutput]):
    """Explain medical terminology in plain language."""

    name: ClassVar[str] = "terminology_explainer"
    description: ClassVar[str] = "Explain medical terms in plain language suitable for patients or different audiences."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.KNOWLEDGE

    input_class: ClassVar[Type[TerminologyExplainerInput]] = TerminologyExplainerInput
    output_class: ClassVar[Type[TerminologyExplainerOutput]] = TerminologyExplainerOutput

    async def execute(self, input: TerminologyExplainerInput) -> TerminologyExplainerOutput:
        try:
            term_lower = input.term.lower()

            # Check database
            if term_lower in MEDICAL_TERMS:
                data = MEDICAL_TERMS[term_lower]
                return TerminologyExplainerOutput(
                    success=True,
                    status=ToolStatus.SUCCESS,
                    data=data,
                    term=input.term,
                    definition=data["definition"],
                    pronunciation=data["pronunciation"],
                    etymology=data["etymology"],
                    plain_language=data["plain_language"],
                    related_terms=data["related"],
                    examples=data["examples"],
                    confidence=0.95
                )

            # Fall back to generated explanation
            explanation = await self._generate_explanation(input.term, input.audience)

            return TerminologyExplainerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"generated": True},
                term=input.term,
                definition=explanation,
                plain_language=explanation,
                confidence=0.7
            )

        except Exception as e:
            return TerminologyExplainerOutput.from_error(str(e))

    async def _generate_explanation(self, term: str, audience: str) -> str:
        """Generate explanation using model (placeholder)."""
        # Would use MedGemma in production
        return f"'{term}' is a medical term. Please consult a medical dictionary for precise definition."
