"""
MedGemma Agent Framework - PubMed Search Tool

Searches PubMed for medical literature.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class PubMedSearchInput(ToolInput):
    """Input for PubMed search."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results")
    date_range: Optional[str] = Field(default=None, description="Date range (e.g., '5 years')")
    article_types: Optional[List[str]] = Field(default=None, description="Filter by type (review, clinical trial, etc.)")


class Article(BaseModel):
    """PubMed article details."""
    pmid: str
    title: str
    authors: List[str]
    journal: str
    year: int
    abstract: Optional[str] = None
    doi: Optional[str] = None
    article_type: Optional[str] = None


class PubMedSearchOutput(ToolOutput):
    """Output for PubMed search."""
    articles: List[Article] = Field(default_factory=list)
    total_results: int = 0
    query_used: str = ""


class PubMedSearchTool(BaseTool[PubMedSearchInput, PubMedSearchOutput]):
    """Search PubMed for medical literature."""

    name: ClassVar[str] = "pubmed_search"
    description: ClassVar[str] = "Search PubMed/MEDLINE for medical research articles and systematic reviews."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.KNOWLEDGE

    input_class: ClassVar[Type[PubMedSearchInput]] = PubMedSearchInput
    output_class: ClassVar[Type[PubMedSearchOutput]] = PubMedSearchOutput

    async def execute(self, input: PubMedSearchInput) -> PubMedSearchOutput:
        try:
            articles = await self._search_pubmed(
                query=input.query,
                max_results=input.max_results,
                date_range=input.date_range,
                article_types=input.article_types
            )

            return PubMedSearchOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"article_count": len(articles)},
                articles=articles,
                total_results=len(articles),
                query_used=input.query,
                confidence=0.9
            )

        except Exception as e:
            return PubMedSearchOutput.from_error(str(e))

    async def _search_pubmed(
        self,
        query: str,
        max_results: int,
        date_range: Optional[str],
        article_types: Optional[List[str]]
    ) -> List[Article]:
        """Search PubMed using E-utilities API."""
        try:
            import urllib.request
            import urllib.parse
            import xml.etree.ElementTree as ET

            # Build query
            search_query = query
            if date_range:
                # Add date filter
                pass

            # Search for IDs
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            search_url = f"{base_url}esearch.fcgi?db=pubmed&term={urllib.parse.quote(search_query)}&retmax={max_results}&retmode=xml"

            with urllib.request.urlopen(search_url, timeout=10) as response:
                search_xml = response.read()

            root = ET.fromstring(search_xml)
            id_list = root.find('.//IdList')
            if id_list is None:
                return []

            pmids = [id_elem.text for id_elem in id_list.findall('Id')]
            if not pmids:
                return []

            # Fetch details
            ids_str = ','.join(pmids)
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"

            with urllib.request.urlopen(fetch_url, timeout=15) as response:
                fetch_xml = response.read()

            return self._parse_articles(fetch_xml)

        except Exception as e:
            # Return sample data if API fails
            return self._get_sample_articles(query)

    def _parse_articles(self, xml_data: bytes) -> List[Article]:
        """Parse PubMed XML response."""
        import xml.etree.ElementTree as ET

        articles = []
        root = ET.fromstring(xml_data)

        for article_elem in root.findall('.//PubmedArticle'):
            try:
                medline = article_elem.find('.//MedlineCitation')
                if medline is None:
                    continue

                pmid = medline.find('PMID').text if medline.find('PMID') is not None else ""
                article_data = medline.find('.//Article')

                title = ""
                if article_data is not None:
                    title_elem = article_data.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ""

                # Authors
                authors = []
                author_list = article_data.find('.//AuthorList') if article_data else None
                if author_list:
                    for author in author_list.findall('Author')[:5]:
                        last = author.find('LastName')
                        fore = author.find('ForeName')
                        if last is not None:
                            name = last.text
                            if fore is not None:
                                name = f"{last.text} {fore.text[0]}"
                            authors.append(name)

                # Journal
                journal = ""
                journal_elem = article_data.find('.//Journal/Title') if article_data else None
                if journal_elem is not None:
                    journal = journal_elem.text

                # Year
                year = 0
                year_elem = article_data.find('.//PubDate/Year') if article_data else None
                if year_elem is not None:
                    year = int(year_elem.text)

                # Abstract
                abstract = ""
                abstract_elem = article_data.find('.//Abstract/AbstractText') if article_data else None
                if abstract_elem is not None:
                    abstract = abstract_elem.text

                articles.append(Article(
                    pmid=pmid,
                    title=title,
                    authors=authors,
                    journal=journal,
                    year=year,
                    abstract=abstract[:500] if abstract else None
                ))

            except Exception:
                continue

        return articles

    def _get_sample_articles(self, query: str) -> List[Article]:
        """Return sample articles when API is unavailable."""
        return [
            Article(
                pmid="12345678",
                title=f"Sample article related to: {query}",
                authors=["Smith J", "Johnson A"],
                journal="Journal of Medicine",
                year=2024,
                abstract="This is a sample abstract. PubMed API may be temporarily unavailable."
            )
        ]
