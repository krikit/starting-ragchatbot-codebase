import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

from config import config
from rag_system import RAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system with proper error handling
try:
    rag_system = RAGSystem(config)
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[str]
    session_id: str
    error: Optional[str] = None

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""

    # Check if RAG system is available
    if rag_system is None:
        logger.error("RAG system not initialized")
        return QueryResponse(
            answer="System initialization error. Please check your API key configuration and try again.",
            sources=[],
            session_id="",
            error="RAG system not initialized"
        )

    try:
        # Validate query
        if not request.query or not request.query.strip():
            return QueryResponse(
                answer="Please provide a question to search the course materials.",
                sources=[],
                session_id="",
                error="Empty query"
            )

        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            try:
                session_id = rag_system.session_manager.create_session()
            except Exception as e:
                logger.warning(f"Session creation failed, using fallback: {e}")
                session_id = "fallback_session"

        logger.info(f"Processing query: {request.query[:50]}...")

        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)

        # Check for empty or problematic responses
        if not answer or answer.strip() == "":
            answer = "I couldn't generate a response. Please try rephrasing your question."

        logger.info(f"Query processed successfully, answer length: {len(answer)}, sources: {len(sources)}")

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Query processing error: {error_msg}")

        # Provide specific error messages based on the error type
        if "api" in error_msg.lower() or "key" in error_msg.lower():
            user_message = "API configuration error. Please check your Anthropic API key and try again."
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            user_message = "Network connection error. Please check your internet connection and try again."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            user_message = "API rate limit reached. Please wait a moment and try again."
        else:
            user_message = f"An error occurred while processing your query: {error_msg}"

        return QueryResponse(
            answer=user_message,
            sources=[],
            session_id=request.session_id or "",
            error=error_msg
        )

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        logger.error(f"Error getting course stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving course statistics: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "rag_system": "initialized" if rag_system else "failed",
        "api_key": "configured" if config.ANTHROPIC_API_KEY else "missing"
    }

    if rag_system:
        try:
            course_count = rag_system.vector_store.get_course_count()
            status["courses_loaded"] = course_count
        except Exception as e:
            status["courses_loaded"] = f"error: {e}"

    return status

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    if rag_system is None:
        logger.error("Cannot load documents: RAG system not initialized")
        return

    docs_path = "../docs"
    if os.path.exists(docs_path):
        logger.info("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            logger.info(f"Loaded {courses} courses with {chunks} chunks")

            # Verify data was loaded
            final_count = rag_system.vector_store.get_course_count()
            logger.info(f"Final course count in vector store: {final_count}")

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
    else:
        logger.warning(f"Documents directory not found: {docs_path}")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")