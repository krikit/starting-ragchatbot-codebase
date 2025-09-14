"""
Test configuration and fixtures for the RAG chatbot system tests.
"""

import os
import sys

import pytest

# Add backend directory to path so tests can import modules
backend_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, backend_dir)


@pytest.fixture
def sample_search_results():
    """Fixture providing sample search results for testing"""
    from vector_store import SearchResults

    return SearchResults(
        documents=[
            "Python is a high-level programming language known for its readability.",
            "Variables in Python are created by assigning values to names.",
        ],
        metadata=[
            {"course_title": "Python Basics", "lesson_number": 1},
            {"course_title": "Python Basics", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
        error=None,
    )


@pytest.fixture
def empty_search_results():
    """Fixture providing empty search results for testing"""
    from vector_store import SearchResults

    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def error_search_results():
    """Fixture providing search results with error for testing"""
    from vector_store import SearchResults

    return SearchResults(
        documents=[], metadata=[], distances=[], error="Database connection failed"
    )


@pytest.fixture
def mock_course_metadata():
    """Fixture providing sample course metadata"""
    return {
        "metadatas": [
            {
                "title": "Python Programming",
                "instructor": "John Doe",
                "course_link": "https://example.com/python-course",
                "lessons_json": """[
                {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson1"},
                {"lesson_number": 2, "lesson_title": "Variables", "lesson_link": "https://example.com/lesson2"}
            ]""",
            }
        ]
    }


@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    from config import Config

    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"

    return config
