"""
MedGemma Agent Framework - Task Planner

This module handles task decomposition and planning:
- Breaks complex queries into executable steps
- Determines which tools to use for each step
- Manages dependencies between steps
- Handles plan execution and replanning

Usage:
    from core.planner import TaskPlanner, Task

    planner = TaskPlanner(registry)

    # Create a plan from a user query
    plan = await planner.create_plan(
        "Analyze this CT scan and compare with the previous MRI"
    )

    # Execute the plan
    results = await planner.execute_plan(plan)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from pydantic import BaseModel, Field

from core.config import get_config, MedicAItionConfig
from tools.base import ToolCategory, ToolOutput

if TYPE_CHECKING:
    from core.registry import ToolRegistry
    from core.memory import ConversationMemory, PatientContext


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority level for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Task(BaseModel):
    """
    A single task in an execution plan.

    Represents one step that uses a specific tool with given inputs.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Human-readable task name")
    description: str = Field(description="Task description")
    tool_name: str = Field(description="Name of the tool to use")
    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for the tool"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete first"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status"
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority"
    )
    output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task output after completion"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if task failed"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=2)

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self, output: Dict[str, Any]) -> None:
        """Mark task as completed with output."""
        self.status = TaskStatus.COMPLETED
        self.output = output
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries


class Plan(BaseModel):
    """
    An execution plan consisting of multiple tasks.

    Manages the overall workflow and task dependencies.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Plan name")
    description: str = Field(description="Plan description")
    tasks: List[Task] = Field(default_factory=list)
    original_query: str = Field(description="Original user query")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    final_output: Optional[Dict[str, Any]] = None

    def add_task(self, task: Task) -> None:
        """Add a task to the plan."""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute."""
        completed = self._get_completed_task_ids()
        return [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING and task.is_ready(completed)
        ]

    def _get_completed_task_ids(self) -> Set[str]:
        """Get IDs of all completed tasks."""
        return {
            task.id for task in self.tasks
            if task.status == TaskStatus.COMPLETED
        }

    def is_complete(self) -> bool:
        """Check if all tasks are complete or failed."""
        return all(
            task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
            for task in self.tasks
        )

    def get_progress(self) -> Dict[str, int]:
        """Get plan execution progress."""
        status_counts = {status: 0 for status in TaskStatus}
        for task in self.tasks:
            status_counts[task.status] += 1
        return {
            "total": len(self.tasks),
            "completed": status_counts[TaskStatus.COMPLETED],
            "failed": status_counts[TaskStatus.FAILED],
            "running": status_counts[TaskStatus.RUNNING],
            "pending": status_counts[TaskStatus.PENDING],
        }

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of the plan."""
        lines = ["graph TD"]

        for task in self.tasks:
            # Node
            label = f"{task.name}\\n({task.tool_name})"
            status_style = {
                TaskStatus.COMPLETED: ":::completed",
                TaskStatus.FAILED: ":::failed",
                TaskStatus.RUNNING: ":::running",
            }.get(task.status, "")

            lines.append(f'    {task.id}["{label}"]{status_style}')

            # Edges
            for dep_id in task.dependencies:
                lines.append(f"    {dep_id} --> {task.id}")

        # Styles
        lines.extend([
            "    classDef completed fill:#90EE90",
            "    classDef failed fill:#FFB6C1",
            "    classDef running fill:#87CEEB",
        ])

        return "\n".join(lines)


class TaskPlanner:
    """
    Plans and orchestrates task execution.

    Takes natural language queries and creates executable plans
    that utilize the available tools.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        config: Optional[MedicAItionConfig] = None
    ):
        """
        Initialize the planner.

        Args:
            registry: Tool registry for discovering available tools
            config: Optional configuration
        """
        self._registry = registry
        self._config = config or get_config()
        self._model = None

    async def create_plan(
        self,
        query: str,
        context: Optional["ConversationMemory"] = None,
        patient_context: Optional["PatientContext"] = None,
        available_files: Optional[List[Dict[str, Any]]] = None
    ) -> Plan:
        """
        Create an execution plan from a user query.

        This method analyzes the query and available context to determine
        which tools need to be called and in what order.

        Args:
            query: User's natural language query
            context: Conversation context
            patient_context: Patient context
            available_files: List of available files with metadata

        Returns:
            Execution plan
        """
        # Get available tools
        available_tools = self._registry.list_tools()
        tool_info = [
            self._registry.get_tool_info(name)
            for name in available_tools
        ]

        # Analyze the query to determine required tasks
        tasks = await self._analyze_query(
            query=query,
            tool_info=tool_info,
            available_files=available_files or []
        )

        # Create the plan
        plan = Plan(
            name=f"Plan for: {query[:50]}...",
            description=f"Execution plan for user query",
            original_query=query
        )

        for task in tasks:
            plan.add_task(task)

        logger.info(f"Created plan with {len(tasks)} tasks")
        return plan

    async def _analyze_query(
        self,
        query: str,
        tool_info: List[Dict[str, Any]],
        available_files: List[Dict[str, Any]]
    ) -> List[Task]:
        """
        Analyze query and determine required tasks.

        This is a simplified heuristic-based implementation.
        In production, this would use the LLM for planning.
        """
        tasks = []
        query_lower = query.lower()

        # Detect file types mentioned or available
        has_image = any(
            f.get("type") in ["dicom", "image", "ct", "mri", "xray"]
            for f in available_files
        )
        has_document = any(
            f.get("type") in ["pdf", "text", "report"]
            for f in available_files
        )

        # Keywords to tool mapping
        keyword_tool_map = {
            # Imaging
            ("ct", "ct scan", "computed tomography"): "ct_analyzer",
            ("mri", "magnetic resonance"): "mri_analyzer",
            ("xray", "x-ray", "radiograph", "chest film"): "xray_analyzer",
            ("pathology", "histology", "biopsy", "slide"): "histopath_analyzer",
            ("fundus", "retina", "retinal"): "fundus_analyzer",
            ("dermoscopy", "skin lesion", "mole"): "dermoscopy_analyzer",
            ("ultrasound", "sonogram", "echo"): "ultrasound_analyzer",

            # Documents
            ("lab", "laboratory", "blood test"): "lab_report_parser",
            ("radiology report",): "radiology_report_parser",
            ("pathology report",): "pathology_report_parser",
            ("discharge", "discharge summary"): "discharge_summary_parser",
            ("clinical notes", "soap"): "clinical_notes_parser",
            ("prescription", "medication list"): "prescription_parser",
            ("insurance", "claim"): "insurance_claims_parser",

            # Knowledge
            ("calculate", "score", "bmi", "gfr", "wells"): "medical_calculator",
            ("icd", "cpt", "code"): "icd_cpt_lookup",
            ("explain", "meaning", "definition"): "terminology_explainer",
            ("drug interaction", "contraindication"): "drug_interaction",
            ("guideline", "protocol"): "guidelines_rag",
            ("research", "pubmed", "literature"): "pubmed_search",

            # Clinical
            ("triage", "urgency", "emergency"): "triage_classifier",
            ("risk", "prognosis"): "risk_assessment",
            ("differential", "diagnosis", "diagnose"): "differential_diagnosis",
            ("treatment", "therapy", "manage"): "treatment_recommender",

            # Utilities
            ("summarize", "summary"): "medical_summarizer",
            ("extract", "entities", "ner"): "entity_extractor",
            ("plain language", "patient friendly"): "patient_translator",
            ("report", "generate report"): "report_generator",
        }

        # Find matching tools
        matched_tools = []
        for keywords, tool_name in keyword_tool_map.items():
            if any(kw in query_lower for kw in keywords):
                if tool_name in self._registry.list_tools():
                    matched_tools.append(tool_name)

        # If no specific tools matched but files are available
        if not matched_tools:
            if has_image:
                # Default image analysis
                for file_info in available_files:
                    file_type = file_info.get("type", "").lower()
                    if "ct" in file_type:
                        matched_tools.append("ct_analyzer")
                    elif "mri" in file_type:
                        matched_tools.append("mri_analyzer")
                    elif file_type in ["dicom", "xray", "image"]:
                        matched_tools.append("xray_analyzer")

            if has_document:
                # Default document parsing
                matched_tools.append("clinical_notes_parser")

        # If still no tools, add general analysis
        if not matched_tools:
            if "differential_diagnosis" in self._registry.list_tools():
                matched_tools.append("differential_diagnosis")

        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in matched_tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)

        # Create tasks from matched tools
        previous_task_id = None
        for i, tool_name in enumerate(unique_tools):
            task = Task(
                name=f"Step {i + 1}: {tool_name.replace('_', ' ').title()}",
                description=f"Execute {tool_name} tool",
                tool_name=tool_name,
                input_data=self._prepare_tool_input(
                    tool_name, query, available_files
                ),
                dependencies=[previous_task_id] if previous_task_id else []
            )
            tasks.append(task)
            previous_task_id = task.id

        # Add summarization task at the end if multiple tools
        if len(tasks) > 1 and "medical_summarizer" in self._registry.list_tools():
            summary_task = Task(
                name="Final Summary",
                description="Summarize all findings",
                tool_name="medical_summarizer",
                input_data={"text": "{{aggregated_results}}"},
                dependencies=[t.id for t in tasks]
            )
            tasks.append(summary_task)

        return tasks

    def _prepare_tool_input(
        self,
        tool_name: str,
        query: str,
        available_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare input data for a specific tool."""
        input_data: Dict[str, Any] = {"query": query}

        # Add relevant files based on tool type
        imaging_tools = {
            "ct_analyzer", "mri_analyzer", "xray_analyzer",
            "histopath_analyzer", "fundus_analyzer",
            "dermoscopy_analyzer", "ultrasound_analyzer"
        }
        document_tools = {
            "lab_report_parser", "radiology_report_parser",
            "pathology_report_parser", "discharge_summary_parser",
            "clinical_notes_parser", "prescription_parser",
            "insurance_claims_parser"
        }

        for file_info in available_files:
            file_type = file_info.get("type", "")

            if tool_name in imaging_tools:
                if file_type in ["dicom", "image", "ct", "mri", "xray"]:
                    input_data["image_path"] = file_info.get("path")
                    break

            elif tool_name in document_tools:
                if file_type in ["pdf", "text", "document"]:
                    input_data["document_path"] = file_info.get("path")
                    break

        return input_data

    async def execute_plan(
        self,
        plan: Plan,
        max_concurrent: int = 4
    ) -> Plan:
        """
        Execute a plan, running tasks as their dependencies are met.

        Args:
            plan: The plan to execute
            max_concurrent: Maximum concurrent tasks

        Returns:
            The executed plan with results
        """
        plan.status = TaskStatus.RUNNING
        plan.started_at = datetime.utcnow()

        try:
            while not plan.is_complete():
                # Get tasks that are ready to run
                ready_tasks = plan.get_ready_tasks()

                if not ready_tasks:
                    # Check if we're stuck (no ready tasks but not complete)
                    pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
                    if pending:
                        logger.error("Plan stuck: pending tasks with unmet dependencies")
                        for task in pending:
                            task.mark_failed("Dependencies could not be met")
                    break

                # Limit concurrent tasks
                tasks_to_run = ready_tasks[:max_concurrent]

                # Execute tasks concurrently
                await asyncio.gather(*[
                    self._execute_task(task, plan)
                    for task in tasks_to_run
                ])

            # Mark plan complete
            plan.status = TaskStatus.COMPLETED
            plan.completed_at = datetime.utcnow()

            # Aggregate final output
            plan.final_output = self._aggregate_outputs(plan)

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            plan.status = TaskStatus.FAILED

        return plan

    async def _execute_task(self, task: Task, plan: Plan) -> None:
        """Execute a single task."""
        task.mark_running()
        logger.info(f"Executing task: {task.name} ({task.tool_name})")

        try:
            # Resolve any template variables in input
            resolved_input = self._resolve_input_templates(task.input_data, plan)

            # Execute the tool
            result = await self._registry.execute(
                task.tool_name,
                resolved_input,
                timeout=self._config.resources.tool_timeout_seconds
            )

            if result.success:
                task.mark_completed(result.model_dump())
                logger.info(f"Task completed: {task.name}")
            else:
                if task.can_retry():
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    logger.warning(f"Task failed, will retry: {task.name}")
                else:
                    task.mark_failed(str(result.errors))
                    logger.error(f"Task failed: {task.name}")

        except Exception as e:
            task.mark_failed(str(e))
            logger.error(f"Task execution error: {task.name}: {e}")

    def _resolve_input_templates(
        self,
        input_data: Dict[str, Any],
        plan: Plan
    ) -> Dict[str, Any]:
        """Resolve template variables in task input."""
        resolved = {}

        for key, value in input_data.items():
            if isinstance(value, str) and "{{" in value:
                # Handle aggregated results template
                if "aggregated_results" in value:
                    aggregated = self._aggregate_outputs(plan)
                    resolved[key] = str(aggregated)
                else:
                    # Handle task output references like {{task_id.output.field}}
                    resolved[key] = self._resolve_template(value, plan)
            else:
                resolved[key] = value

        return resolved

    def _resolve_template(self, template: str, plan: Plan) -> str:
        """Resolve a single template string."""
        import re

        pattern = r"\{\{([^}]+)\}\}"
        matches = re.findall(pattern, template)

        result = template
        for match in matches:
            parts = match.split(".")
            if len(parts) >= 2:
                task_id = parts[0]
                task = plan.get_task(task_id)
                if task and task.output:
                    value = task.output
                    for part in parts[1:]:
                        if isinstance(value, dict):
                            value = value.get(part, "")
                        else:
                            break
                    result = result.replace(f"{{{{{match}}}}}", str(value))

        return result

    def _aggregate_outputs(self, plan: Plan) -> Dict[str, Any]:
        """Aggregate outputs from all completed tasks."""
        aggregated = {
            "tasks": [],
            "findings": [],
            "recommendations": [],
            "errors": []
        }

        for task in plan.tasks:
            task_summary = {
                "name": task.name,
                "tool": task.tool_name,
                "status": task.status.value
            }

            if task.status == TaskStatus.COMPLETED and task.output:
                task_summary["output"] = task.output

                # Extract common fields
                if task.output.get("data"):
                    data = task.output["data"]
                    if isinstance(data, dict):
                        if data.get("findings"):
                            aggregated["findings"].extend(data["findings"])
                        if data.get("recommendations"):
                            aggregated["recommendations"].extend(data["recommendations"])

            elif task.status == TaskStatus.FAILED:
                aggregated["errors"].append({
                    "task": task.name,
                    "error": task.error
                })

            aggregated["tasks"].append(task_summary)

        return aggregated

    async def replan(
        self,
        plan: Plan,
        feedback: str
    ) -> Plan:
        """
        Create a new plan based on feedback about the current plan.

        Args:
            plan: Current plan
            feedback: User feedback or error information

        Returns:
            New revised plan
        """
        # Create new plan incorporating feedback
        revised_query = f"{plan.original_query}\n\nAdditional context: {feedback}"

        return await self.create_plan(
            query=revised_query,
            available_files=[]  # Would need to preserve file context
        )
