"""
MedGemma Agent Framework - Patient Translator Tool

Translates medical jargon into patient-friendly language.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class PatientTranslatorInput(ToolInput):
    """Input for patient translator."""
    text: str = Field(description="Medical text to translate")
    reading_level: str = Field(default="general", description="Target reading level: simple, general, advanced")
    include_definitions: bool = Field(default=True, description="Include definitions of medical terms")


class PatientTranslatorOutput(ToolOutput):
    """Output for patient translator."""
    translated_text: str = ""
    terms_explained: Dict[str, str] = Field(default_factory=dict)
    reading_level_achieved: str = ""


class PatientTranslatorTool(BaseTool[PatientTranslatorInput, PatientTranslatorOutput]):
    """Translate medical jargon to patient-friendly language."""

    name: ClassVar[str] = "patient_translator"
    description: ClassVar[str] = "Translate medical terminology into patient-friendly plain language."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.UTILITIES

    input_class: ClassVar[Type[PatientTranslatorInput]] = PatientTranslatorInput
    output_class: ClassVar[Type[PatientTranslatorOutput]] = PatientTranslatorOutput

    # Medical term translations
    TRANSLATIONS = {
        # Conditions
        "hypertension": "high blood pressure",
        "hypotension": "low blood pressure",
        "tachycardia": "fast heart rate",
        "bradycardia": "slow heart rate",
        "dyspnea": "shortness of breath",
        "edema": "swelling",
        "myocardial infarction": "heart attack",
        "cerebrovascular accident": "stroke",
        "pneumonia": "lung infection",
        "diabetes mellitus": "diabetes (high blood sugar)",
        "hyperlipidemia": "high cholesterol",
        "renal insufficiency": "kidney problems",
        "hepatic dysfunction": "liver problems",
        "anemia": "low blood count",
        "arrhythmia": "irregular heartbeat",

        # Symptoms
        "nausea": "feeling sick to your stomach",
        "emesis": "vomiting",
        "syncope": "fainting",
        "vertigo": "dizziness",
        "malaise": "feeling unwell",
        "fatigue": "tiredness",
        "diaphoresis": "sweating",
        "pruritus": "itching",
        "erythema": "redness",

        # Anatomy
        "cardiac": "heart",
        "pulmonary": "lung",
        "hepatic": "liver",
        "renal": "kidney",
        "cerebral": "brain",
        "thoracic": "chest",
        "abdominal": "belly/stomach area",
        "extremities": "arms and legs",

        # Tests/Procedures
        "echocardiogram": "heart ultrasound",
        "electrocardiogram": "heart rhythm test (ECG/EKG)",
        "colonoscopy": "camera exam of your colon",
        "endoscopy": "camera exam of your digestive system",
        "biopsy": "taking a small tissue sample",
        "MRI": "detailed imaging scan using magnets",
        "CT scan": "detailed X-ray imaging",
        "ultrasound": "imaging using sound waves",

        # Instructions
        "prn": "as needed",
        "bid": "twice a day",
        "tid": "three times a day",
        "qid": "four times a day",
        "qhs": "at bedtime",
        "po": "by mouth",
        "stat": "right away/immediately",
        "npo": "nothing to eat or drink",

        # Results
        "benign": "not cancer/not harmful",
        "malignant": "cancerous",
        "negative": "did not find what we were looking for (usually good)",
        "positive": "found what we were looking for",
        "elevated": "higher than normal",
        "decreased": "lower than normal",
        "within normal limits": "normal",
        "unremarkable": "normal/nothing concerning",
    }

    async def execute(self, input: PatientTranslatorInput) -> PatientTranslatorOutput:
        try:
            text = input.text
            translated = text
            terms_explained = {}

            # Apply translations
            for medical_term, plain_term in self.TRANSLATIONS.items():
                if medical_term.lower() in translated.lower():
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(medical_term), re.IGNORECASE)

                    if input.include_definitions:
                        # Add explanation in parentheses
                        replacement = f"{medical_term} ({plain_term})"
                    else:
                        replacement = plain_term

                    translated = pattern.sub(replacement, translated)
                    terms_explained[medical_term] = plain_term

            # Simplify based on reading level
            if input.reading_level == "simple":
                translated = self._simplify_text(translated)

            return PatientTranslatorOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"terms_translated": len(terms_explained)},
                translated_text=translated,
                terms_explained=terms_explained,
                reading_level_achieved=input.reading_level,
                confidence=0.85
            )

        except Exception as e:
            return PatientTranslatorOutput.from_error(str(e))

    def _simplify_text(self, text: str) -> str:
        """Further simplify text for basic reading level."""
        # Break long sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        simplified = []

        for sentence in sentences:
            # If sentence is too long, try to break it
            if len(sentence) > 100:
                # Try to split at conjunctions
                parts = re.split(r'\s+(?:and|but|or|which|that|because)\s+', sentence)
                for part in parts:
                    if part.strip():
                        simplified.append(part.strip())
            else:
                simplified.append(sentence)

        return ' '.join(simplified)
