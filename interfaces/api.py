"""
MedGemma Agent Framework - REST API

FastAPI-based REST API for the medical AI agent.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.registry import ToolNotFoundError

# Global instances
agent = None
registry = None
task_results: Dict[str, Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    """Request for natural language query."""
    query: str = Field(..., description="Natural language query")
    patient_context: Optional[Dict[str, Any]] = Field(default=None, description="Patient context")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")


class QueryResponse(BaseModel):
    """Response from natural language query."""
    response: str
    session_id: str
    tools_used: List[str] = []
    confidence: Optional[float] = None
    processing_time_ms: float


class ToolExecutionRequest(BaseModel):
    """Request for direct tool execution."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    input_data: Dict[str, Any] = Field(..., description="Tool input data")


class ToolExecutionResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    tool_name: str
    result: Dict[str, Any]
    confidence: Optional[float] = None
    processing_time_ms: float


class ToolInfo(BaseModel):
    """Information about a tool."""
    name: str
    description: str
    version: str
    category: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class AsyncTaskResponse(BaseModel):
    """Response for async task submission."""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Response for task status check."""
    task_id: str
    status: str  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    tools_available: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global agent, registry

    from core.agent import MedGemmaAgent
    from core.registry import ToolRegistry
    from core.config import MedGemmaConfig

    config = MedGemmaConfig()

    # Initialize registry and discover tools
    registry = ToolRegistry()
    registry.auto_discover()

    # Initialize agent
    agent = MedGemmaAgent(config=config)
    await agent.initialize()

    yield

    # Cleanup
    if agent:
        await agent.shutdown()


# Create FastAPI app
app = FastAPI(
    title="MedGemma Agent API",
    description="REST API for the MedGemma medical AI agent framework",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=agent is not None and agent._initialized,
        tools_available=len(registry.list_tools()) if registry else 0
    )


@app.get("/tools", response_model=List[ToolInfo], tags=["Tools"])
async def list_tools():
    """List all available tools."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    tools = []
    for tool_name in registry.list_tools():
        try:
            tool = await registry.get_tool(tool_name)
            tools.append(ToolInfo(
                name=tool.name,
                description=tool.description,
                version=tool.version,
                category=tool.category.value if hasattr(tool.category, 'value') else str(tool.category),
                input_schema=tool.get_input_schema(),
                output_schema=tool.get_output_schema()
            ))
        except ToolNotFoundError:
            pass
    return tools


@app.get("/tools/{tool_name}", response_model=ToolInfo, tags=["Tools"])
async def get_tool(tool_name: str):
    """Get information about a specific tool."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    try:
        tool = await registry.get_tool(tool_name)
    except ToolNotFoundError:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return ToolInfo(
        name=tool.name,
        description=tool.description,
        version=tool.version,
        category=tool.category.value if hasattr(tool.category, 'value') else str(tool.category),
        input_schema=tool.get_input_schema(),
        output_schema=tool.get_output_schema()
    )


@app.post("/tools/{tool_name}/execute", response_model=ToolExecutionResponse, tags=["Tools"])
async def execute_tool(tool_name: str, request: ToolExecutionRequest):
    """Execute a specific tool."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    try:
        tool = await registry.get_tool(tool_name)
    except ToolNotFoundError:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    start_time = datetime.now()

    try:
        validated_input = tool.validate_input(request.input_data)
        result = await tool.execute(validated_input)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ToolExecutionResponse(
            success=result.success,
            tool_name=tool_name,
            result=result.model_dump(),
            confidence=result.confidence,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def process_query(request: QueryRequest):
    """Process a natural language query."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    start_time = datetime.now()
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Set patient context if provided
        if request.patient_context:
            agent.set_patient_context(request.patient_context)

        response = await agent.process(request.query)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return QueryResponse(
            response=response,
            session_id=session_id,
            tools_used=[],  # Could track tools used during processing
            confidence=0.85,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/async", response_model=AsyncTaskResponse, tags=["Agent"])
async def process_query_async(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a query for asynchronous processing."""
    task_id = str(uuid.uuid4())

    task_results[task_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None
    }

    async def process_task():
        try:
            task_results[task_id]["status"] = "running"

            if request.patient_context:
                agent.set_patient_context(request.patient_context)

            response = await agent.process(request.query)

            task_results[task_id]["status"] = "completed"
            task_results[task_id]["result"] = {"response": response}
            task_results[task_id]["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["error"] = str(e)
            task_results[task_id]["completed_at"] = datetime.now().isoformat()

    background_tasks.add_task(process_task)

    return AsyncTaskResponse(
        task_id=task_id,
        status="pending",
        message="Task submitted successfully"
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Agent"])
async def get_task_status(task_id: str):
    """Get the status of an async task."""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")

    task = task_results[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"],
        error=task["error"],
        created_at=task["created_at"],
        completed_at=task["completed_at"]
    )


@app.post("/analyze/image", tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    modality: str = Form(..., description="Image modality: xray, ct, mri, fundus, dermoscopy, ultrasound"),
    body_region: Optional[str] = Form(default=None, description="Body region"),
    clinical_context: Optional[str] = Form(default=None, description="Clinical context")
):
    """Analyze a medical image."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Read image data
    image_data = await file.read()

    start_time = datetime.now()

    try:
        # Determine which tool to use based on modality
        tool_map = {
            "xray": "xray_analyzer",
            "ct": "ct_analyzer",
            "mri": "mri_analyzer",
            "fundus": "fundus_analyzer",
            "dermoscopy": "dermoscopy_analyzer",
            "ultrasound": "ultrasound_analyzer",
            "histopathology": "histopath_analyzer"
        }

        tool_name = tool_map.get(modality.lower())
        if not tool_name:
            raise HTTPException(status_code=400, detail=f"Unknown modality: {modality}")

        tool = await registry.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=503, detail=f"Tool {tool_name} not available")

        # Prepare input
        import base64
        input_data = {
            "image_data": base64.b64encode(image_data).decode(),
            "image_format": file.content_type or "image/jpeg"
        }

        if body_region:
            input_data["body_region"] = body_region
        if clinical_context:
            input_data["clinical_context"] = clinical_context

        validated_input = tool.validate_input(input_data)
        result = await tool.execute(validated_input)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return JSONResponse({
            "success": result.success,
            "modality": modality,
            "result": result.model_dump(),
            "processing_time_ms": processing_time
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/document", tags=["Analysis"])
async def analyze_document(
    file: UploadFile = File(...),
    document_type: str = Form(..., description="Document type: lab, radiology, pathology, discharge, clinical_notes, prescription"),
):
    """Analyze a medical document."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Read document content
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    start_time = datetime.now()

    try:
        # Determine which tool to use
        tool_map = {
            "lab": "lab_report_parser",
            "radiology": "radiology_report_parser",
            "pathology": "pathology_report_parser",
            "discharge": "discharge_summary_parser",
            "clinical_notes": "clinical_notes_parser",
            "prescription": "prescription_parser",
            "insurance": "insurance_claims_parser"
        }

        tool_name = tool_map.get(document_type.lower())
        if not tool_name:
            raise HTTPException(status_code=400, detail=f"Unknown document type: {document_type}")

        tool = await registry.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=503, detail=f"Tool {tool_name} not available")

        input_data = {"text": text}
        validated_input = tool.validate_input(input_data)
        result = await tool.execute(validated_input)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return JSONResponse({
            "success": result.success,
            "document_type": document_type,
            "result": result.model_dump(),
            "processing_time_ms": processing_time
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Factory function for creating the app."""
    return app


def run(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
