"""
MedGemma Agent Framework - ICD/CPT Code Lookup Tool

Provides lookup and search for ICD-10 and CPT codes.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class ICDCPTLookupInput(ToolInput):
    """Input for ICD/CPT lookup."""
    query: str = Field(description="Code or description to search")
    code_type: Optional[str] = Field(default=None, description="Filter by type (ICD-10, CPT)")
    limit: int = Field(default=10, description="Maximum results to return")


class CodeResult(ToolOutput):
    """Output for code lookup."""
    codes: List[Dict[str, str]] = Field(default_factory=list)
    query: str = ""
    code_type: Optional[str] = None
    total_found: int = 0


# Sample ICD-10 codes database
ICD10_CODES = {
    "A00-B99": "Certain infectious and parasitic diseases",
    "C00-D49": "Neoplasms",
    "D50-D89": "Diseases of the blood",
    "E00-E89": "Endocrine, nutritional and metabolic diseases",
    "I10": "Essential (primary) hypertension",
    "I21.0": "ST elevation myocardial infarction involving left main coronary artery",
    "I21.1": "ST elevation myocardial infarction involving left anterior descending coronary artery",
    "I21.9": "Acute myocardial infarction, unspecified",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "I50.9": "Heart failure, unspecified",
    "J06.9": "Acute upper respiratory infection, unspecified",
    "J18.9": "Pneumonia, unspecified organism",
    "J44.1": "Chronic obstructive pulmonary disease with acute exacerbation",
    "J45.20": "Mild intermittent asthma, uncomplicated",
    "K21.0": "Gastro-esophageal reflux disease with esophagitis",
    "M54.5": "Low back pain",
    "N18.3": "Chronic kidney disease, stage 3",
    "N18.4": "Chronic kidney disease, stage 4",
    "N18.5": "Chronic kidney disease, stage 5",
    "R07.9": "Chest pain, unspecified",
    "R10.9": "Unspecified abdominal pain",
    "Z87.891": "Personal history of nicotine dependence",
}

# Sample CPT codes database
CPT_CODES = {
    "99201": "Office visit, new patient, minimal",
    "99202": "Office visit, new patient, low",
    "99203": "Office visit, new patient, moderate",
    "99204": "Office visit, new patient, high",
    "99205": "Office visit, new patient, comprehensive",
    "99211": "Office visit, established patient, minimal",
    "99212": "Office visit, established patient, low",
    "99213": "Office visit, established patient, moderate",
    "99214": "Office visit, established patient, high",
    "99215": "Office visit, established patient, comprehensive",
    "71046": "Radiologic examination, chest; 2 views",
    "71250": "CT thorax; without contrast",
    "71260": "CT thorax; with contrast",
    "71270": "CT thorax; without contrast, followed by contrast",
    "93000": "Electrocardiogram, routine ECG with interpretation",
    "93010": "Electrocardiogram, interpretation and report only",
    "80053": "Comprehensive metabolic panel",
    "80061": "Lipid panel",
    "85025": "Complete blood count with differential",
    "36415": "Collection of venous blood by venipuncture",
    "43239": "Upper GI endoscopy with biopsy",
    "45378": "Colonoscopy, diagnostic",
    "45380": "Colonoscopy with biopsy",
}


class ICDCPTLookupTool(BaseTool[ICDCPTLookupInput, CodeResult]):
    """Lookup ICD-10 and CPT codes."""

    name: ClassVar[str] = "icd_cpt_lookup"
    description: ClassVar[str] = "Search and lookup ICD-10 diagnosis codes and CPT procedure codes."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.KNOWLEDGE

    input_class: ClassVar[Type[ICDCPTLookupInput]] = ICDCPTLookupInput
    output_class: ClassVar[Type[CodeResult]] = CodeResult

    async def execute(self, input: ICDCPTLookupInput) -> CodeResult:
        try:
            results = []
            query_lower = input.query.lower()

            # Search ICD-10 codes
            if not input.code_type or input.code_type.upper() in ['ICD', 'ICD-10', 'ICD10']:
                for code, desc in ICD10_CODES.items():
                    if query_lower in code.lower() or query_lower in desc.lower():
                        results.append({
                            "code": code,
                            "description": desc,
                            "type": "ICD-10"
                        })

            # Search CPT codes
            if not input.code_type or input.code_type.upper() == 'CPT':
                for code, desc in CPT_CODES.items():
                    if query_lower in code.lower() or query_lower in desc.lower():
                        results.append({
                            "code": code,
                            "description": desc,
                            "type": "CPT"
                        })

            # Limit results
            results = results[:input.limit]

            return CodeResult(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"results": results},
                codes=results,
                query=input.query,
                code_type=input.code_type,
                total_found=len(results),
                confidence=0.9
            )

        except Exception as e:
            return CodeResult.from_error(str(e))
