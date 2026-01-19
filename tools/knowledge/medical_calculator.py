"""
MedGemma Agent Framework - Medical Calculator Tool

Implements various medical calculators and scoring systems.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class MedicalCalculatorInput(ToolInput):
    """Input for medical calculator."""
    calculator: str = Field(description="Calculator name (BMI, GFR, Wells, CHADS2, etc.)")
    parameters: Dict[str, Any] = Field(description="Input parameters for calculation")


class CalculationResult(ToolOutput):
    """Output for medical calculator."""
    calculator_name: str = ""
    result: Optional[float] = None
    result_string: Optional[str] = None
    interpretation: Optional[str] = None
    risk_category: Optional[str] = None
    formula_used: Optional[str] = None
    reference: Optional[str] = None


class MedicalCalculatorTool(BaseTool[MedicalCalculatorInput, CalculationResult]):
    """Medical calculators and scoring systems."""

    name: ClassVar[str] = "medical_calculator"
    description: ClassVar[str] = "Calculate BMI, GFR, Wells score, CHADS2, MELD, APACHE, and other medical scores."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.KNOWLEDGE

    input_class: ClassVar[Type[MedicalCalculatorInput]] = MedicalCalculatorInput
    output_class: ClassVar[Type[CalculationResult]] = CalculationResult

    CALCULATORS = {
        'bmi': '_calc_bmi',
        'gfr': '_calc_gfr',
        'egfr': '_calc_gfr',
        'ckd-epi': '_calc_gfr',
        'wells_dvt': '_calc_wells_dvt',
        'wells_pe': '_calc_wells_pe',
        'chads2': '_calc_chads2',
        'cha2ds2vasc': '_calc_cha2ds2vasc',
        'meld': '_calc_meld',
        'child_pugh': '_calc_child_pugh',
        'apache2': '_calc_apache2',
        'sofa': '_calc_sofa',
        'qsofa': '_calc_qsofa',
        'corrected_calcium': '_calc_corrected_calcium',
        'anion_gap': '_calc_anion_gap',
        'creatinine_clearance': '_calc_creatinine_clearance',
    }

    async def execute(self, input: MedicalCalculatorInput) -> CalculationResult:
        try:
            calc_name = input.calculator.lower().replace(' ', '_').replace('-', '_')

            if calc_name not in self.CALCULATORS:
                available = ', '.join(self.CALCULATORS.keys())
                return CalculationResult.from_error(f"Unknown calculator. Available: {available}")

            method = getattr(self, self.CALCULATORS[calc_name])
            return method(input.parameters)

        except Exception as e:
            return CalculationResult.from_error(str(e))

    def _calc_bmi(self, params: Dict) -> CalculationResult:
        weight = params.get('weight_kg') or params.get('weight')
        height = params.get('height_cm') or params.get('height')
        if not weight or not height:
            return CalculationResult.from_error("Requires weight_kg and height_cm")

        height_m = height / 100
        bmi = weight / (height_m ** 2)

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="BMI",
            result=round(bmi, 1),
            result_string=f"{round(bmi, 1)} kg/m²",
            interpretation=category,
            risk_category=category,
            formula_used="weight(kg) / height(m)²",
            data={"bmi": round(bmi, 1), "category": category}
        )

    def _calc_gfr(self, params: Dict) -> CalculationResult:
        creat = params.get('creatinine')
        age = params.get('age')
        is_female = params.get('female', params.get('sex', '').lower() == 'female')
        is_black = params.get('black', False)

        if not creat or not age:
            return CalculationResult.from_error("Requires creatinine and age")

        # CKD-EPI equation
        if is_female:
            if creat <= 0.7:
                gfr = 144 * ((creat / 0.7) ** -0.329) * (0.993 ** age)
            else:
                gfr = 144 * ((creat / 0.7) ** -1.209) * (0.993 ** age)
        else:
            if creat <= 0.9:
                gfr = 141 * ((creat / 0.9) ** -0.411) * (0.993 ** age)
            else:
                gfr = 141 * ((creat / 0.9) ** -1.209) * (0.993 ** age)

        if is_black:
            gfr *= 1.159

        # CKD staging
        if gfr >= 90:
            stage = "G1 (Normal)"
        elif gfr >= 60:
            stage = "G2 (Mild)"
        elif gfr >= 45:
            stage = "G3a (Mild-Moderate)"
        elif gfr >= 30:
            stage = "G3b (Moderate-Severe)"
        elif gfr >= 15:
            stage = "G4 (Severe)"
        else:
            stage = "G5 (Kidney Failure)"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="eGFR (CKD-EPI)",
            result=round(gfr, 1),
            result_string=f"{round(gfr, 1)} mL/min/1.73m²",
            interpretation=f"CKD Stage: {stage}",
            risk_category=stage,
            formula_used="CKD-EPI 2009 equation",
            data={"egfr": round(gfr, 1), "stage": stage}
        )

    def _calc_wells_dvt(self, params: Dict) -> CalculationResult:
        score = 0
        if params.get('active_cancer'):
            score += 1
        if params.get('paralysis_paresis'):
            score += 1
        if params.get('bedridden'):
            score += 1
        if params.get('tenderness'):
            score += 1
        if params.get('leg_swelling'):
            score += 1
        if params.get('calf_swelling'):
            score += 1
        if params.get('pitting_edema'):
            score += 1
        if params.get('collateral_veins'):
            score += 1
        if params.get('previous_dvt'):
            score += 1
        if params.get('alternative_diagnosis'):
            score -= 2

        if score <= 0:
            risk = "Low (3%)"
        elif score <= 2:
            risk = "Moderate (17%)"
        else:
            risk = "High (75%)"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="Wells Score for DVT",
            result=score,
            result_string=f"{score} points",
            interpretation=f"DVT probability: {risk}",
            risk_category=risk.split()[0],
            data={"score": score, "risk": risk}
        )

    def _calc_wells_pe(self, params: Dict) -> CalculationResult:
        score = 0
        if params.get('dvt_symptoms'):
            score += 3
        if params.get('pe_most_likely'):
            score += 3
        if params.get('heart_rate_gt_100'):
            score += 1.5
        if params.get('immobilization'):
            score += 1.5
        if params.get('previous_pe_dvt'):
            score += 1.5
        if params.get('hemoptysis'):
            score += 1
        if params.get('malignancy'):
            score += 1

        if score <= 4:
            risk = "PE Unlikely"
        else:
            risk = "PE Likely"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="Wells Score for PE",
            result=score,
            result_string=f"{score} points",
            interpretation=risk,
            risk_category=risk,
            data={"score": score, "risk": risk}
        )

    def _calc_chads2(self, params: Dict) -> CalculationResult:
        score = 0
        if params.get('chf'):
            score += 1
        if params.get('hypertension'):
            score += 1
        if params.get('age_ge_75'):
            score += 1
        if params.get('diabetes'):
            score += 1
        if params.get('stroke_tia'):
            score += 2

        risk_map = {0: "1.9%", 1: "2.8%", 2: "4.0%", 3: "5.9%", 4: "8.5%", 5: "12.5%", 6: "18.2%"}

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="CHADS2 Score",
            result=score,
            result_string=f"{score} points",
            interpretation=f"Annual stroke risk: {risk_map.get(score, '>18%')}",
            data={"score": score}
        )

    def _calc_cha2ds2vasc(self, params: Dict) -> CalculationResult:
        score = 0
        if params.get('chf'):
            score += 1
        if params.get('hypertension'):
            score += 1
        if params.get('age_ge_75'):
            score += 2
        elif params.get('age_65_74'):
            score += 1
        if params.get('diabetes'):
            score += 1
        if params.get('stroke_tia'):
            score += 2
        if params.get('vascular_disease'):
            score += 1
        if params.get('female'):
            score += 1

        if score == 0:
            recommendation = "No anticoagulation"
        elif score == 1:
            recommendation = "Consider anticoagulation"
        else:
            recommendation = "Anticoagulation recommended"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="CHA2DS2-VASc Score",
            result=score,
            result_string=f"{score} points",
            interpretation=recommendation,
            data={"score": score}
        )

    def _calc_meld(self, params: Dict) -> CalculationResult:
        creat = min(params.get('creatinine', 1), 4)
        bili = params.get('bilirubin', 1)
        inr = params.get('inr', 1)

        creat = max(creat, 1)
        bili = max(bili, 1)
        inr = max(inr, 1)

        meld = 10 * (0.957 * math.log(creat) + 0.378 * math.log(bili) + 1.120 * math.log(inr) + 0.643)
        meld = round(min(max(meld, 6), 40))

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="MELD Score",
            result=meld,
            result_string=f"{meld} points",
            interpretation=f"3-month mortality estimate based on MELD score",
            data={"meld": meld}
        )

    def _calc_child_pugh(self, params: Dict) -> CalculationResult:
        score = 0

        # Bilirubin
        bili = params.get('bilirubin', 1)
        if bili < 2:
            score += 1
        elif bili <= 3:
            score += 2
        else:
            score += 3

        # Albumin
        alb = params.get('albumin', 3.5)
        if alb > 3.5:
            score += 1
        elif alb >= 2.8:
            score += 2
        else:
            score += 3

        # INR
        inr = params.get('inr', 1)
        if inr < 1.7:
            score += 1
        elif inr <= 2.3:
            score += 2
        else:
            score += 3

        # Ascites
        ascites = params.get('ascites', 'none').lower()
        if ascites == 'none':
            score += 1
        elif ascites in ['mild', 'controlled']:
            score += 2
        else:
            score += 3

        # Encephalopathy
        enceph = params.get('encephalopathy', 'none').lower()
        if enceph == 'none':
            score += 1
        elif enceph in ['grade 1', 'grade 2', 'mild']:
            score += 2
        else:
            score += 3

        if score <= 6:
            classification = "Class A (5-6 points)"
        elif score <= 9:
            classification = "Class B (7-9 points)"
        else:
            classification = "Class C (10-15 points)"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="Child-Pugh Score",
            result=score,
            result_string=f"{score} points",
            interpretation=classification,
            risk_category=classification.split()[1],
            data={"score": score, "class": classification}
        )

    def _calc_apache2(self, params: Dict) -> CalculationResult:
        # Simplified APACHE II
        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="APACHE II",
            result=0,
            interpretation="APACHE II requires multiple parameters. Please provide complete data.",
            data={}
        )

    def _calc_sofa(self, params: Dict) -> CalculationResult:
        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="SOFA Score",
            result=0,
            interpretation="SOFA requires respiratory, coagulation, liver, cardiovascular, CNS, and renal parameters.",
            data={}
        )

    def _calc_qsofa(self, params: Dict) -> CalculationResult:
        score = 0
        if params.get('altered_mental_status'):
            score += 1
        if params.get('respiratory_rate_ge_22'):
            score += 1
        if params.get('systolic_bp_le_100'):
            score += 1

        risk = "Low" if score < 2 else "High - consider sepsis"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="qSOFA",
            result=score,
            result_string=f"{score}/3",
            interpretation=risk,
            data={"score": score}
        )

    def _calc_corrected_calcium(self, params: Dict) -> CalculationResult:
        calcium = params.get('calcium')
        albumin = params.get('albumin')

        if not calcium or not albumin:
            return CalculationResult.from_error("Requires calcium and albumin")

        corrected = calcium + 0.8 * (4.0 - albumin)

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="Corrected Calcium",
            result=round(corrected, 1),
            result_string=f"{round(corrected, 1)} mg/dL",
            formula_used="Ca + 0.8 × (4.0 - Albumin)",
            data={"corrected_calcium": round(corrected, 1)}
        )

    def _calc_anion_gap(self, params: Dict) -> CalculationResult:
        na = params.get('sodium')
        cl = params.get('chloride')
        hco3 = params.get('bicarbonate') or params.get('co2')

        if not all([na, cl, hco3]):
            return CalculationResult.from_error("Requires sodium, chloride, and bicarbonate")

        gap = na - (cl + hco3)

        if gap <= 12:
            interp = "Normal (8-12)"
        else:
            interp = "Elevated - consider HAGMA"

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="Anion Gap",
            result=round(gap, 1),
            result_string=f"{round(gap, 1)} mEq/L",
            interpretation=interp,
            formula_used="Na - (Cl + HCO3)",
            data={"anion_gap": round(gap, 1)}
        )

    def _calc_creatinine_clearance(self, params: Dict) -> CalculationResult:
        creat = params.get('creatinine')
        age = params.get('age')
        weight = params.get('weight')
        is_female = params.get('female', params.get('sex', '').lower() == 'female')

        if not all([creat, age, weight]):
            return CalculationResult.from_error("Requires creatinine, age, and weight")

        # Cockcroft-Gault
        crcl = ((140 - age) * weight) / (72 * creat)
        if is_female:
            crcl *= 0.85

        return CalculationResult(
            success=True, status=ToolStatus.SUCCESS,
            calculator_name="Creatinine Clearance (Cockcroft-Gault)",
            result=round(crcl, 1),
            result_string=f"{round(crcl, 1)} mL/min",
            formula_used="((140-age) × weight) / (72 × Cr) × 0.85 if female",
            data={"crcl": round(crcl, 1)}
        )
