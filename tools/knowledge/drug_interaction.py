"""
MedGemma Agent Framework - Drug Interaction Checker Tool

Checks for drug-drug interactions and contraindications.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class DrugInteractionInput(ToolInput):
    """Input for drug interaction checker."""
    drugs: List[str] = Field(description="List of drug names to check")
    patient_conditions: Optional[List[str]] = Field(default=None, description="Patient conditions for contraindication check")


class Interaction(BaseModel):
    """Drug interaction details."""
    drug1: str
    drug2: str
    severity: str  # mild, moderate, severe, contraindicated
    description: str
    mechanism: Optional[str] = None
    management: Optional[str] = None


class DrugInteractionOutput(ToolOutput):
    """Output for drug interaction checker."""
    interactions: List[Interaction] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    drugs_checked: List[str] = Field(default_factory=list)


# Sample drug interaction database
DRUG_INTERACTIONS = [
    {"drug1": "warfarin", "drug2": "aspirin", "severity": "severe",
     "description": "Increased bleeding risk", "mechanism": "Additive anticoagulant effects",
     "management": "Avoid combination or monitor closely"},
    {"drug1": "warfarin", "drug2": "ibuprofen", "severity": "severe",
     "description": "Increased bleeding risk and INR elevation",
     "mechanism": "NSAID inhibits platelet function and displaces warfarin from protein binding",
     "management": "Use acetaminophen instead if possible"},
    {"drug1": "metformin", "drug2": "contrast", "severity": "severe",
     "description": "Risk of lactic acidosis",
     "mechanism": "Contrast-induced nephropathy impairs metformin clearance",
     "management": "Hold metformin 48 hours before and after contrast"},
    {"drug1": "lisinopril", "drug2": "potassium", "severity": "moderate",
     "description": "Risk of hyperkalemia",
     "mechanism": "ACE inhibitors reduce aldosterone, decreasing potassium excretion",
     "management": "Monitor potassium levels"},
    {"drug1": "lisinopril", "drug2": "spironolactone", "severity": "moderate",
     "description": "Risk of hyperkalemia",
     "mechanism": "Both drugs increase serum potassium",
     "management": "Monitor potassium closely"},
    {"drug1": "simvastatin", "drug2": "clarithromycin", "severity": "severe",
     "description": "Increased risk of myopathy/rhabdomyolysis",
     "mechanism": "Clarithromycin inhibits CYP3A4 metabolism of simvastatin",
     "management": "Avoid combination; use azithromycin or hold statin"},
    {"drug1": "clopidogrel", "drug2": "omeprazole", "severity": "moderate",
     "description": "Reduced clopidogrel efficacy",
     "mechanism": "Omeprazole inhibits CYP2C19 activation of clopidogrel",
     "management": "Use pantoprazole instead"},
    {"drug1": "methotrexate", "drug2": "trimethoprim", "severity": "severe",
     "description": "Increased methotrexate toxicity",
     "mechanism": "Both are folate antagonists; trimethoprim decreases methotrexate clearance",
     "management": "Avoid combination"},
    {"drug1": "ssri", "drug2": "tramadol", "severity": "severe",
     "description": "Risk of serotonin syndrome",
     "mechanism": "Both increase serotonergic activity",
     "management": "Avoid combination or use with extreme caution"},
    {"drug1": "ciprofloxacin", "drug2": "tizanidine", "severity": "contraindicated",
     "description": "Markedly increased tizanidine levels",
     "mechanism": "Ciprofloxacin strongly inhibits CYP1A2",
     "management": "Combination is contraindicated"},
]


class DrugInteractionTool(BaseTool[DrugInteractionInput, DrugInteractionOutput]):
    """Check for drug-drug interactions."""

    name: ClassVar[str] = "drug_interaction"
    description: ClassVar[str] = "Check for drug-drug interactions, contraindications, and provide management recommendations."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.KNOWLEDGE

    input_class: ClassVar[Type[DrugInteractionInput]] = DrugInteractionInput
    output_class: ClassVar[Type[DrugInteractionOutput]] = DrugInteractionOutput

    async def execute(self, input: DrugInteractionInput) -> DrugInteractionOutput:
        try:
            interactions = []
            warnings = []
            drugs_lower = [d.lower() for d in input.drugs]

            # Check all drug pairs
            for i, drug1 in enumerate(drugs_lower):
                for drug2 in drugs_lower[i+1:]:
                    interaction = self._check_interaction(drug1, drug2)
                    if interaction:
                        interactions.append(interaction)

            # Check contraindications with conditions
            contraindications = []
            if input.patient_conditions:
                contraindications = self._check_contraindications(drugs_lower, input.patient_conditions)

            # Generate warnings for severe interactions
            for inter in interactions:
                if inter.severity in ["severe", "contraindicated"]:
                    warnings.append(f"ALERT: {inter.drug1} + {inter.drug2}: {inter.description}")

            return DrugInteractionOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"interaction_count": len(interactions)},
                interactions=interactions,
                contraindications=contraindications,
                warnings=warnings,
                drugs_checked=input.drugs,
                confidence=0.85
            )

        except Exception as e:
            return DrugInteractionOutput.from_error(str(e))

    def _check_interaction(self, drug1: str, drug2: str) -> Optional[Interaction]:
        """Check for interaction between two drugs."""
        for inter in DRUG_INTERACTIONS:
            d1, d2 = inter["drug1"].lower(), inter["drug2"].lower()
            # Check both orderings and partial matches
            if (d1 in drug1 or drug1 in d1) and (d2 in drug2 or drug2 in d2):
                return Interaction(**inter)
            if (d1 in drug2 or drug2 in d1) and (d2 in drug1 or drug1 in d2):
                return Interaction(
                    drug1=inter["drug2"], drug2=inter["drug1"],
                    severity=inter["severity"], description=inter["description"],
                    mechanism=inter.get("mechanism"), management=inter.get("management")
                )
        return None

    def _check_contraindications(self, drugs: List[str], conditions: List[str]) -> List[str]:
        """Check drug-condition contraindications."""
        contraindications = []
        conditions_lower = [c.lower() for c in conditions]

        # Sample contraindication rules
        rules = [
            (["metformin"], ["renal failure", "kidney failure", "ckd stage 4", "ckd stage 5"], "Metformin contraindicated in severe renal impairment"),
            (["nsaid", "ibuprofen", "naproxen"], ["gi bleed", "peptic ulcer"], "NSAIDs contraindicated with GI bleeding history"),
            (["ace inhibitor", "lisinopril", "enalapril"], ["angioedema"], "ACE inhibitors contraindicated with angioedema history"),
            (["beta blocker", "metoprolol", "atenolol"], ["asthma", "severe copd"], "Beta blockers may worsen bronchospasm"),
        ]

        for drug_patterns, condition_patterns, message in rules:
            drug_match = any(any(dp in d for dp in drug_patterns) for d in drugs)
            condition_match = any(any(cp in c for cp in condition_patterns) for c in conditions_lower)
            if drug_match and condition_match:
                contraindications.append(message)

        return contraindications
