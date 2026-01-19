"""
MedGemma Agent Framework - Clinical Guidelines RAG Tool

RAG system for retrieving and querying clinical guidelines.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class GuidelinesRAGInput(ToolInput):
    """Input for guidelines RAG."""
    query: str = Field(description="Clinical question or topic")
    specialty: Optional[str] = Field(default=None, description="Medical specialty filter")
    guideline_source: Optional[str] = Field(default=None, description="Source filter (AHA, ACC, IDSA, etc.)")
    max_results: int = Field(default=5, description="Maximum results to return")


class GuidelineResult(ToolOutput):
    """Output for guidelines RAG."""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    query: str = ""
    sources_searched: List[str] = Field(default_factory=list)
    summary: Optional[str] = None


# Sample clinical guidelines database
GUIDELINES_DB = [
    {
        "title": "ACC/AHA Guideline for the Management of Heart Failure",
        "source": "ACC/AHA",
        "year": 2022,
        "specialty": "cardiology",
        "topics": ["heart failure", "hfref", "hfpef", "cardiomyopathy"],
        "key_recommendations": [
            "GDMT includes ACEi/ARB/ARNI, beta-blockers, MRA, and SGLT2i for HFrEF",
            "Target LVEF < 40% for HFrEF diagnosis",
            "SGLT2 inhibitors recommended for HFrEF (Class I)",
            "ICD recommended for primary prevention if LVEF ≤ 35%"
        ],
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063"
    },
    {
        "title": "ACC/AHA Guideline for Blood Pressure Management",
        "source": "ACC/AHA",
        "year": 2017,
        "specialty": "cardiology",
        "topics": ["hypertension", "blood pressure", "antihypertensive"],
        "key_recommendations": [
            "BP < 130/80 mmHg target for most adults",
            "Stage 1 HTN: 130-139/80-89 mmHg",
            "Stage 2 HTN: ≥ 140/90 mmHg",
            "First-line agents: thiazides, ACEi, ARB, CCB"
        ],
        "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065"
    },
    {
        "title": "ADA Standards of Care in Diabetes",
        "source": "ADA",
        "year": 2024,
        "specialty": "endocrinology",
        "topics": ["diabetes", "glucose", "hba1c", "insulin"],
        "key_recommendations": [
            "HbA1c target < 7% for most adults",
            "Metformin remains first-line therapy",
            "GLP-1 RA or SGLT2i for patients with ASCVD or CKD",
            "Annual screening for diabetic complications"
        ],
        "url": "https://diabetesjournals.org/care"
    },
    {
        "title": "IDSA Guidelines for Community-Acquired Pneumonia",
        "source": "IDSA/ATS",
        "year": 2019,
        "specialty": "infectious disease",
        "topics": ["pneumonia", "cap", "respiratory infection", "antibiotics"],
        "key_recommendations": [
            "Outpatient: amoxicillin or doxycycline or macrolide",
            "Inpatient non-severe: beta-lactam + macrolide OR respiratory fluoroquinolone",
            "Severe: beta-lactam + macrolide OR beta-lactam + fluoroquinolone",
            "Procalcitonin can guide antibiotic duration"
        ],
        "url": "https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/"
    },
    {
        "title": "ACEP Clinical Policy: Chest Pain",
        "source": "ACEP",
        "year": 2018,
        "specialty": "emergency medicine",
        "topics": ["chest pain", "acs", "acute coronary syndrome", "emergency"],
        "key_recommendations": [
            "Risk stratify using validated tools (HEART, TIMI)",
            "High-sensitivity troponin for early rule-out",
            "HEART score ≤ 3 may be safely discharged",
            "Consider 0/1-hour or 0/3-hour troponin protocols"
        ],
        "url": "https://www.acep.org/patient-care/clinical-policies/"
    },
]


class GuidelinesRAGTool(BaseTool[GuidelinesRAGInput, GuidelineResult]):
    """Query clinical guidelines using RAG."""

    name: ClassVar[str] = "guidelines_rag"
    description: ClassVar[str] = "Search and retrieve clinical practice guidelines from major medical societies."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.KNOWLEDGE

    input_class: ClassVar[Type[GuidelinesRAGInput]] = GuidelinesRAGInput
    output_class: ClassVar[Type[GuidelineResult]] = GuidelineResult

    async def execute(self, input: GuidelinesRAGInput) -> GuidelineResult:
        try:
            results = self._search_guidelines(
                query=input.query,
                specialty=input.specialty,
                source=input.guideline_source,
                max_results=input.max_results
            )

            summary = self._generate_summary(input.query, results)

            return GuidelineResult(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"result_count": len(results)},
                results=results,
                query=input.query,
                sources_searched=list(set(r["source"] for r in results)),
                summary=summary,
                confidence=0.85
            )

        except Exception as e:
            return GuidelineResult.from_error(str(e))

    def _search_guidelines(
        self,
        query: str,
        specialty: Optional[str],
        source: Optional[str],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search guidelines database."""
        query_lower = query.lower()
        results = []

        for guideline in GUIDELINES_DB:
            # Apply filters
            if specialty and specialty.lower() != guideline["specialty"]:
                continue
            if source and source.upper() not in guideline["source"]:
                continue

            # Check relevance
            relevance = 0
            for topic in guideline["topics"]:
                if topic in query_lower:
                    relevance += 2
            if any(word in guideline["title"].lower() for word in query_lower.split()):
                relevance += 1

            if relevance > 0:
                results.append({
                    **guideline,
                    "relevance_score": relevance
                })

        # Sort by relevance and limit
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]

    def _generate_summary(self, query: str, results: List[Dict]) -> str:
        """Generate summary of guideline recommendations."""
        if not results:
            return f"No guidelines found for '{query}'. Consider broadening your search."

        summary_parts = [f"Guidelines relevant to '{query}':\n"]

        for r in results[:3]:
            summary_parts.append(f"\n{r['title']} ({r['source']}, {r['year']}):")
            for rec in r["key_recommendations"][:2]:
                summary_parts.append(f"  • {rec}")

        return "\n".join(summary_parts)
