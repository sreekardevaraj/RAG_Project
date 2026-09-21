"""
FastAPI application for RAG Chatbot V2
Wraps the entire LangGraph pipeline with REST API endpoints
http://localhost:8001/docs

"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import socket
import tempfile
from backend.rag_pipeline import build_rag_pipeline, ask_question
from backend.healthcheck import get_health_status
from backend.persistence import load_chat_history
from backend.deepeval_integration import evaluate_user_response
from config.settings import RAW_PDF_DIR
from fastapi.responses import JSONResponse


# ── Initialize FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot V2 API",
    description="Multi-Agent RAG Chatbot with PDF and Web Search capabilities",
    version="2.0.0"
)

# ── CORS Middleware ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Pipeline State ──────────────────────────────────────────────
pipeline_state = {
    "pipeline": None,
    "ready": False,
    "error": None
}


# ── Pydantic Models (Request/Response Schemas) ────────────────────────
class InitializeRequest(BaseModel):
    """Initialize and build the RAG pipeline"""
    force_rebuild: bool = False


class InitializeResponse(BaseModel):
    """Response after pipeline initialization"""
    status: str
    message: str
    ready: bool


class QuestionRequest(BaseModel):
    """User question to the chatbot"""
    question: str


class SourceDetail(BaseModel):
    """Reference detail from answer"""
    source: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None
    title: Optional[str] = None


class EvaluationResult(BaseModel):
    """DeepEval evaluation result for a single user response."""
    timestamp: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    score: Optional[float] = None
    passed: Optional[bool] = None
    model: Optional[str] = None
    error: Optional[str] = None


class AnswerResponse(BaseModel):
    """Answer response with source and references"""
    question: str
    answer: str
    source: str  # "PDF" | "Web" | "Analyst" | "Chat"
    details: List[Dict[str, Any]]
    followups: List[str]
    evaluation: Optional[EvaluationResult] = None


class PipelineStatus(BaseModel):
    """Current pipeline status"""
    ready: bool
    error: Optional[str]


class UploadResponse(BaseModel):
    """Response after file upload"""
    status: str
    filename: str
    message: str


class ExportRequest(BaseModel):
    """Request to export conversation to PDF"""
    title: str = "Chat History"
    filename: str = "chat_export.pdf"


# ── Health & Status Endpoints ──────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot V2 API",
        "status": "online",
        "docs": "/docs",
        "pipeline_ready": pipeline_state["ready"]
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    health = get_health_status()
    return {
        "status": health["status"],
        "pipeline_ready": pipeline_state["ready"],
        "error": pipeline_state["error"],
        "checks": health["checks"],
        "issues": health["issues"],
    }


@app.get("/status", response_model=PipelineStatus, tags=["Pipeline"])
def get_status():
    """Get current pipeline status"""
    return {
        "ready": pipeline_state["ready"],
        "error": pipeline_state["error"]
    }


@app.get("/api/history", tags=["Chat"])
def get_chat_history():
    """Return persisted chat history."""
    return {"messages": load_chat_history()}


# ── Pipeline Initialization ────────────────────────────────────────────

@app.post("/api/initialize", response_model=InitializeResponse, tags=["Pipeline"])
def initialize_pipeline(request: InitializeRequest):
    """
    Initialize and build the RAG pipeline.
    
    This endpoint must be called before asking questions.
    - If force_rebuild is True, rebuilds the vector index from scratch
    - Otherwise, loads existing index if available
    """
    try:
        print("[API] Initializing pipeline...")
        pipeline = build_rag_pipeline(force_rebuild=request.force_rebuild)
        
        pipeline_state["pipeline"] = pipeline
        pipeline_state["ready"] = True
        pipeline_state["error"] = None
        
        return {
            "status": "success",
            "message": "Pipeline initialized successfully",
            "ready": True
        }
    except Exception as e:
        error_msg = f"Pipeline initialization failed: {str(e)}"
        pipeline_state["error"] = error_msg
        pipeline_state["ready"] = False
        raise HTTPException(status_code=500, detail=error_msg)


# ── File Upload Endpoint ───────────────────────────────────────────────

@app.post("/api/upload", response_model=UploadResponse, tags=["Files"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for the RAG pipeline to process.
    
    The file is saved to the data/raw directory.
    After upload, call /api/initialize with force_rebuild=true to re-index.
    """
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create directory if it doesn't exist
        os.makedirs(RAW_PDF_DIR, exist_ok=True)
        
        # Save file
        file_path = os.path.join(RAW_PDF_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File '{file.filename}' uploaded successfully. Call /api/initialize with force_rebuild=true to re-index."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


# ── Question Answering Endpoint ────────────────────────────────────────

@app.post("/api/ask", response_model=AnswerResponse, tags=["Chat"])
def ask_question_endpoint(request: QuestionRequest):
    """
    Ask the chatbot a question.
    
    The question is routed by the Supervisor agent to the appropriate agent:
    - RAG Agent: Questions about the uploaded PDF
    - Web Agent: Questions needing recent/web information
    - Analyst Agent: Requests for analysis, summary, or report
    - Casual Agent: Small talk and greetings
    
    Requires: Pipeline must be initialized first (/api/initialize)
    """
    # Check if pipeline is ready
    if not pipeline_state["ready"] or pipeline_state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Call /api/initialize first."
        )
    
    try:
        # Get question from user
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Ask the pipeline
        response = ask_question(pipeline_state["pipeline"], question)
        answer = response.get("answer", "")

        # Evaluate the answer quality for this user input
        eval_result = evaluate_user_response(
            question=question,
            answer=answer,
            context=[detail.get("source") or detail.get("title") or "" for detail in response.get("details", []) if detail],
        )

        payload = {
            "question": question,
            "answer": answer,
            "source": response.get("source", "Unknown"),
            "details": response.get("details", []),
            "followups": response.get("followups", []),
            "evaluation": eval_result
        }
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")


# ── Export Endpoint ────────────────────────────────────────────────────

@app.post("/api/export", tags=["Chat"])
def export_chat(request: ExportRequest, background_tasks: BackgroundTasks):
    """
    Export conversation history to PDF (placeholder).
    
    Note: Requires integration with your export_chat_to_pdf function
    """
    try:
        if not pipeline_state["pipeline"]:
            raise HTTPException(
                status_code=503,
                detail="Pipeline not initialized"
            )
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_path = temp_file.name
        temp_file.close()
        
        # Export (you'll need to implement this based on your export function)
        # pdf_buffer = export_chat_to_pdf(pipeline_state["pipeline"], request.title)
        
        # For now, return a placeholder
        return {"status": "success", "message": "Export functionality can be integrated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ── Batch Question Endpoint ────────────────────────────────────────────

class BatchQuestionsRequest(BaseModel):
    """Request to ask multiple questions"""
    questions: List[str]


class BatchAnswerItem(BaseModel):
    """Single answer in batch response"""
    question: str
    answer: str
    source: str
    error: Optional[str] = None


@app.post("/api/batch-ask", tags=["Chat"])
def batch_ask(request: BatchQuestionsRequest):
    """
    Ask multiple questions in a single request.
    
    Returns answers to all questions.
    """
    if not pipeline_state["ready"] or pipeline_state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Call /api/initialize first."
        )
    
    results = []
    for question in request.questions:
        try:
            question = question.strip()
            if not question:
                continue
            
            response = ask_question(pipeline_state["pipeline"], question)
            results.append(BatchAnswerItem(
                question=question,
                answer=response.get("answer", ""),
                source=response.get("source", "Unknown"),
                error=None
            ))
        except Exception as e:
            results.append(BatchAnswerItem(
                question=question,
                answer="",
                source="Error",
                error=str(e)
            ))
    
    return {"results": results, "total": len(results)}


# ── Pipeline Info Endpoint ────────────────────────────────────────────

@app.get("/api/pipeline-info", tags=["Pipeline"])
def get_pipeline_info():
    """Get information about the current pipeline configuration"""
    return {
        "ready": pipeline_state["ready"],
        "agents": ["Supervisor", "RAG", "Web", "Analyst", "Casual", "FollowUp", "Memory"],
        "capabilities": [
            "PDF Question Answering (RAG)",
            "Web Search",
            "Document Analysis",
            "Casual Conversation",
            "Conversation Memory",
            "Follow-up Suggestions"
        ],
        "models": {
            "llm": "llama-3.3-70b-versatile (Groq)",
            "embeddings": "all-MiniLM-L6-v2"
        }
    }


# ── Error Handlers ─────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    def find_free_port(start_port: int = 8001, max_attempts: int = 20) -> int:
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No free port found")

    requested_port = int(os.getenv("PORT", "8001"))
    port = requested_port if requested_port != 8001 else find_free_port(8001)
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting RAG Chatbot V2 API on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="info")
