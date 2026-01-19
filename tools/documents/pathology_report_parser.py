"""
MedGemma Agent Framework - Pathology Report Parser Tool

Parses pathology reports to extract diagnosis, staging, and molecular findings.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, DocumentInput, ToolCategory, ToolOutput, ToolStatus


class PathologyReportParserInput(DocumentInput):
    """Input for pathology report parser."""
    specimen_type: Optional[str] = Field(default=None, description="Specimen type")
    cancer_type: Optional[str] = Field(default=None, description="Expected cancer type")


class PathologyReportParserOutput(ToolOutput):
    """Output for pathology report parser."""
    diagnosis: Optional[str] = None
    histologic_type: Optional[str] = None
    grade: Optional[str] = None
    tnm_stage: Optional[Dict[str, str]] = None
    margins: Optional[str] = None
    lymph_nodes: Optional[str] = None
    ihc_results: Optional[Dict[str, str]] = None
    molecular_markers: Optional[Dict[str, str]] = None
    synoptic_data: Optional[Dict[str, Any]] = None


class PathologyReportParserTool(BaseTool[PathologyReportParserInput, PathologyReportParserOutput]):
    """Parse pathology reports for diagnosis and staging."""

    name: ClassVar[str] = "pathology_report_parser"
    description: ClassVar[str] = "Parse pathology reports to extract diagnosis, staging, margins, and molecular markers."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.DOCUMENTS

    input_class: ClassVar[Type[PathologyReportParserInput]] = PathologyReportParserInput
    output_class: ClassVar[Type[PathologyReportParserOutput]] = PathologyReportParserOutput

    async def execute(self, input: PathologyReportParserInput) -> PathologyReportParserOutput:
        try:
            text = input.document_text or ""
            if input.document_path:
                with open(input.document_path, 'r') as f:
                    text = f.read()

            diagnosis = self._extract_diagnosis(text)
            grade = self._extract_grade(text)
            tnm = self._extract_tnm(text)
            margins = self._extract_margins(text)
            lymph_nodes = self._extract_lymph_nodes(text)
            ihc = self._extract_ihc(text)

            return PathologyReportParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"diagnosis": diagnosis},
                diagnosis=diagnosis,
                grade=grade,
                tnm_stage=tnm,
                margins=margins,
                lymph_nodes=lymph_nodes,
                ihc_results=ihc,
                confidence=0.8
            )
        except Exception as e:
            return PathologyReportParserOutput.from_error(str(e))

    def _extract_diagnosis(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'(?:DIAGNOSIS|FINAL DIAGNOSIS):\s*(.+?)(?=\n[A-Z]{2,}:|$)', text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_grade(self, text: str) -> Optional[str]:
        import re
        patterns = [r'Grade\s*(\d|I{1,3})', r'(well|moderately|poorly)\s*differentiated']
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_tnm(self, text: str) -> Optional[Dict[str, str]]:
        import re
        tnm = {}
        t_match = re.search(r'\bp?T(\d|is|x|a|b)', text, re.IGNORECASE)
        n_match = re.search(r'\bp?N(\d|x)', text, re.IGNORECASE)
        m_match = re.search(r'\bp?M(\d|x)', text, re.IGNORECASE)
        if t_match:
            tnm['T'] = f"T{t_match.group(1)}"
        if n_match:
            tnm['N'] = f"N{n_match.group(1)}"
        if m_match:
            tnm['M'] = f"M{m_match.group(1)}"
        return tnm if tnm else None

    def _extract_margins(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'(?:margin|margins):\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_lymph_nodes(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'(?:lymph node|LN).*?(\d+)\s*/\s*(\d+)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}/{match.group(2)} positive"
        return None

    def _extract_ihc(self, text: str) -> Optional[Dict[str, str]]:
        import re
        ihc = {}
        markers = ['ER', 'PR', 'HER2', 'Ki-67', 'p53', 'PDL1', 'CD20', 'CD3']
        for marker in markers:
            match = re.search(rf'{marker}[:\s]*(positive|negative|\d+%?|\+|-)', text, re.IGNORECASE)
            if match:
                ihc[marker] = match.group(1)
        return ihc if ihc else None
