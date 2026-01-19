"""
MedGemma Agent Framework - Insurance Claims Parser Tool

Parses insurance claims and EOBs to extract billing codes, amounts, and coverage.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, DocumentInput, ToolCategory, ToolOutput, ToolStatus


class InsuranceClaimsParserInput(DocumentInput):
    """Input for insurance claims parser."""
    validate_codes: bool = Field(default=True, description="Validate ICD/CPT codes")


class ClaimLine(BaseModel):
    """A single claim line item."""
    service_date: Optional[str] = None
    cpt_code: Optional[str] = None
    icd_codes: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    billed_amount: Optional[float] = None
    allowed_amount: Optional[float] = None
    paid_amount: Optional[float] = None
    patient_responsibility: Optional[float] = None
    status: Optional[str] = None


class InsuranceClaimsParserOutput(ToolOutput):
    """Output for insurance claims parser."""
    claim_number: Optional[str] = None
    claim_date: Optional[str] = None
    patient_name: Optional[str] = None
    provider_name: Optional[str] = None
    claim_lines: List[ClaimLine] = Field(default_factory=list)
    total_billed: Optional[float] = None
    total_allowed: Optional[float] = None
    total_paid: Optional[float] = None
    total_patient_responsibility: Optional[float] = None
    claim_status: Optional[str] = None
    denial_reasons: List[str] = Field(default_factory=list)


class InsuranceClaimsParserTool(BaseTool[InsuranceClaimsParserInput, InsuranceClaimsParserOutput]):
    """Parse insurance claims for billing information."""

    name: ClassVar[str] = "insurance_claims_parser"
    description: ClassVar[str] = "Parse insurance claims and EOBs to extract codes, amounts, and coverage details."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENTS

    input_class: ClassVar[Type[InsuranceClaimsParserInput]] = InsuranceClaimsParserInput
    output_class: ClassVar[Type[InsuranceClaimsParserOutput]] = InsuranceClaimsParserOutput

    async def execute(self, input: InsuranceClaimsParserInput) -> InsuranceClaimsParserOutput:
        try:
            text = input.document_text or ""
            if input.document_path:
                with open(input.document_path, 'r') as f:
                    text = f.read()

            claim_lines = self._parse_claim_lines(text)
            totals = self._calculate_totals(claim_lines)

            return InsuranceClaimsParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"line_count": len(claim_lines)},
                claim_number=self._extract_claim_number(text),
                claim_date=self._extract_date(text),
                claim_lines=claim_lines,
                total_billed=totals.get('billed'),
                total_allowed=totals.get('allowed'),
                total_paid=totals.get('paid'),
                total_patient_responsibility=totals.get('patient'),
                claim_status=self._extract_status(text),
                denial_reasons=self._extract_denials(text),
                confidence=0.8
            )
        except Exception as e:
            return InsuranceClaimsParserOutput.from_error(str(e))

    def _parse_claim_lines(self, text: str) -> List[ClaimLine]:
        import re
        lines = []

        # Pattern for CPT codes
        cpt_pattern = r'(\d{5})\s+([^$\n]+?)\s+\$?([\d,]+\.?\d*)'
        matches = re.findall(cpt_pattern, text)

        for match in matches:
            lines.append(ClaimLine(
                cpt_code=match[0],
                description=match[1].strip(),
                billed_amount=float(match[2].replace(',', '')) if match[2] else None
            ))

        # Extract ICD codes separately
        icd_pattern = r'([A-Z]\d{2}\.?\d*)'
        icd_codes = re.findall(icd_pattern, text)

        # Associate with first line if any
        if lines and icd_codes:
            lines[0].icd_codes = icd_codes[:5]

        return lines

    def _calculate_totals(self, lines: List[ClaimLine]) -> Dict[str, float]:
        totals = {'billed': 0, 'allowed': 0, 'paid': 0, 'patient': 0}
        for line in lines:
            if line.billed_amount:
                totals['billed'] += line.billed_amount
            if line.allowed_amount:
                totals['allowed'] += line.allowed_amount
            if line.paid_amount:
                totals['paid'] += line.paid_amount
            if line.patient_responsibility:
                totals['patient'] += line.patient_responsibility
        return totals

    def _extract_claim_number(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'(?:Claim|Reference)\s*#?:?\s*([A-Z0-9-]+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_date(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'(?:Claim|Service)\s*Date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_status(self, text: str) -> Optional[str]:
        statuses = ['PAID', 'DENIED', 'PENDING', 'PROCESSED', 'APPEALED']
        text_upper = text.upper()
        for status in statuses:
            if status in text_upper:
                return status
        return None

    def _extract_denials(self, text: str) -> List[str]:
        import re
        denials = []
        patterns = [
            r'(?:Denial|Reject)\s*(?:Reason|Code):?\s*(.+?)(?:\n|$)',
            r'(?:CO|PR|OA)-?\d+:\s*(.+?)(?:\n|$)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            denials.extend(matches)
        return denials[:5]
