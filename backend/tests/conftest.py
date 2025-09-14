"""
Test configuration and fixtures for the RAG chatbot system tests.
"""
import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

# Add backend directory to path so tests can import modules
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)


@pytest.fixture
def sample_search_results():
    """Fixture providing sample search results for testing"""
    from vector_store import SearchResults

    return SearchResults(
        documents=[
            "Python is a high-level programming language known for its readability.",
            "Variables in Python are created by assigning values to names."
        ],
        metadata=[
            {"course_title": "Python Basics", "lesson_number": 1},
            {"course_title": "Python Basics", "lesson_number": 2}
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def empty_search_results():
    """Fixture providing empty search results for testing"""
    from vector_store import SearchResults

    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def error_search_results():
    """Fixture providing search results with error for testing"""
    from vector_store import SearchResults

    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Database connection failed"
    )


@pytest.fixture
def mock_course_metadata():
    """Fixture providing sample course metadata"""
    return {
        'metadatas': [{
            'title': 'Python Programming',
            'instructor': 'John Doe',
            'course_link': 'https://example.com/python-course',
            'lessons_json': '''[
                {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson1"},
                {"lesson_number": 2, "lesson_title": "Variables", "lesson_link": "https://example.com/lesson2"}
            ]'''
        }]
    }


@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    from config import Config

    config = Config()
    config.OPENAI_API_KEY = "test-api-key"
    config.OPENAI_MODEL = "gpt-4"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"

    return config


@pytest.fixture
def temp_docs_dir():
    """Fixture providing a temporary docs directory with test files"""
    temp_dir = tempfile.mkdtemp()

    # Create a sample course file
    course_content = """Course: Python Programming
Instructor: John Doe
Link: https://example.com/python-course

Lesson 1: Introduction to Python
Link: https://example.com/lesson1
This is an introduction to Python programming language.
Python is easy to learn and powerful.

Lesson 2: Variables and Data Types
Link: https://example.com/lesson2
Variables in Python are created by assigning values.
Python supports various data types like strings, integers, and lists."""

    with open(os.path.join(temp_dir, "python_course.txt"), "w") as f:
        f.write(course_content)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_rag_system():
    """Fixture providing a mock RAG system"""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "Python is a high-level programming language.",
        ["Python Programming Course - Lesson 1"]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Python Programming"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    return mock_rag


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mock OpenAI client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a test response from OpenAI."
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def test_app():
    """Fixture providing a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create a minimal test app without problematic static file mounting
    app = FastAPI(title="Course Materials RAG System Test", root_path="")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Mock RAG system for testing
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "Python is a high-level programming language.",
        ["Python Programming Course - Lesson 1"]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Python Programming"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"

    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}

    return app


@pytest.fixture
def test_client(test_app):
    """Fixture providing a test client for the FastAPI app"""
    return TestClient(test_app)