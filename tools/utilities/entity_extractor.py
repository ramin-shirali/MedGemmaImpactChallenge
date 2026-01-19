"""
MedGemma Agent Framework - Medical Entity Extractor Tool

Extracts medical entities (conditions, medications, procedures, etc.) from text.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class MedicalEntity(BaseModel):
    """An extracted medical entity."""
    text: str
    entity_type: str  # condition, medication, procedure, anatomy, lab_test, symptom
    start_pos: int
    end_pos: int
    normalized_form: Optional[str] = None
    code: Optional[str] = None
    code_system: Optional[str] = None
    confidence: float = 0.8


class EntityExtractorInput(ToolInput):
    """Input for entity extractor."""
    text: str = Field(description="Text to extract entities from")
    entity_types: Optional[List[str]] = Field(default=None, description="Filter by entity types")
    include_negated: bool = Field(default=False, description="Include negated entities")


class EntityExtractorOutput(ToolOutput):
    """Output for entity extractor."""
    entities: List[MedicalEntity] = Field(default_factory=list)
    entity_counts: Dict[str, int] = Field(default_factory=dict)
    negated_entities: List[MedicalEntity] = Field(default_factory=list)


class EntityExtractorTool(BaseTool[EntityExtractorInput, EntityExtractorOutput]):
    """Extract medical entities from clinical text."""

    name: ClassVar[str] = "entity_extractor"
    description: ClassVar[str] = "Extract medical entities (conditions, medications, procedures, anatomy) from clinical text."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.UTILITIES

    input_class: ClassVar[Type[EntityExtractorInput]] = EntityExtractorInput
    output_class: ClassVar[Type[EntityExtractorOutput]] = EntityExtractorOutput

    # Entity patterns
    MEDICATION_PATTERNS = [
        r'\b([A-Z][a-z]+(?:in|ol|ide|ate|one|ine|pam|lam|cin|xin|mab|nib))\b',
        r'\b(aspirin|tylenol|advil|motrin|ibuprofen|acetaminophen)\b',
    ]

    CONDITION_PATTERNS = [
        r'\b(\w+(?:itis|osis|emia|pathy|algia|trophy|plasia))\b',
        r'\b(diabetes|hypertension|cancer|pneumonia|infection|failure)\b',
        r'\b(disease|syndrome|disorder)\b',
    ]

    ANATOMY_PATTERNS = [
        r'\b(heart|lung|liver|kidney|brain|spine|chest|abdomen|head|neck)\b',
        r'\b(artery|vein|nerve|muscle|bone|joint|tendon)\b',
        r'\b(left|right|bilateral)\s+(lung|kidney|eye|ear|arm|leg)\b',
    ]

    PROCEDURE_PATTERNS = [
        r'\b(\w+(?:ectomy|otomy|ostomy|plasty|scopy|graphy))\b',
        r'\b(biopsy|surgery|transplant|catheterization)\b',
    ]

    LAB_PATTERNS = [
        r'\b(CBC|CMP|BMP|LFT|lipid panel|urinalysis)\b',
        r'\b(hemoglobin|glucose|creatinine|sodium|potassium)\b',
        r'\b(troponin|BNP|proBNP|d-dimer|INR|PT|PTT)\b',
    ]

    SYMPTOM_PATTERNS = [
        r'\b(pain|fever|cough|nausea|vomiting|diarrhea|fatigue)\b',
        r'\b(shortness of breath|chest pain|headache|dizziness)\b',
        r'\b(swelling|rash|bleeding|weakness|numbness)\b',
    ]

    NEGATION_PATTERNS = [
        r'no\s+', r'denies\s+', r'without\s+', r'negative\s+for\s+',
        r'ruled\s+out', r'not\s+', r'absence\s+of\s+'
    ]

    async def execute(self, input: EntityExtractorInput) -> EntityExtractorOutput:
        try:
            text = input.text
            entities = []
            negated = []

            # Extract each entity type
            entity_patterns = {
                'medication': self.MEDICATION_PATTERNS,
                'condition': self.CONDITION_PATTERNS,
                'anatomy': self.ANATOMY_PATTERNS,
                'procedure': self.PROCEDURE_PATTERNS,
                'lab_test': self.LAB_PATTERNS,
                'symptom': self.SYMPTOM_PATTERNS,
            }

            for entity_type, patterns in entity_patterns.items():
                if input.entity_types and entity_type not in input.entity_types:
                    continue

                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        entity_text = match.group(0)
                        start = match.start()
                        end = match.end()

                        # Check for negation
                        is_negated = self._check_negation(text, start)

                        entity = MedicalEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            start_pos=start,
                            end_pos=end,
                            confidence=0.75
                        )

                        if is_negated:
                            negated.append(entity)
                        else:
                            entities.append(entity)

            # Deduplicate entities
            entities = self._deduplicate(entities)
            negated = self._deduplicate(negated)

            # Count by type
            counts: Dict[str, int] = {}
            for entity in entities:
                counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1

            # Include negated if requested
            if input.include_negated:
                entities.extend(negated)

            return EntityExtractorOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"entity_count": len(entities)},
                entities=entities,
                entity_counts=counts,
                negated_entities=negated,
                confidence=0.8
            )

        except Exception as e:
            return EntityExtractorOutput.from_error(str(e))

    def _check_negation(self, text: str, entity_start: int) -> bool:
        """Check if entity is negated."""
        # Look at 30 characters before entity
        context_start = max(0, entity_start - 30)
        context = text[context_start:entity_start].lower()

        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, context):
                return True
        return False

    def _deduplicate(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicate entities."""
        seen = set()
        unique = []
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique
