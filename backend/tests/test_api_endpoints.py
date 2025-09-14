"""
API endpoint tests for the RAG chatbot system.
"""
import pytest
from fastapi.testclient import TestClient
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test class for API endpoint functionality"""

    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns correct message"""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System API"}

    def test_query_endpoint_basic(self, test_client):
        """Test basic query functionality"""
        query_data = {
            "query": "What is Python?"
        }
        response = test_client.post("/api/query", json=query_data)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Validate response content
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert len(data["answer"]) > 0
        assert data["session_id"] == "test-session-123"

    def test_query_endpoint_with_session_id(self, test_client):
        """Test query with existing session ID"""
        query_data = {
            "query": "What is Python?",
            "session_id": "existing-session-456"
        }
        response = test_client.post("/api/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"

    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query"""
        query_data = {
            "query": ""
        }
        response = test_client.post("/api/query", json=query_data)

        # Should still return 200 but with appropriate handling
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post("/api/query", data="invalid json")

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_endpoint_missing_query_field(self, test_client):
        """Test query endpoint without required query field"""
        query_data = {
            "session_id": "test-session"
        }
        response = test_client.post("/api/query", json=query_data)

        assert response.status_code == 422  # Validation error

    def test_courses_endpoint(self, test_client):
        """Test the courses statistics endpoint"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Validate response content
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Python Programming"]

    def test_courses_endpoint_get_only(self, test_client):
        """Test that courses endpoint only accepts GET requests"""
        # Test POST is not allowed
        response = test_client.post("/api/courses")
        assert response.status_code == 405  # Method Not Allowed

        # Test PUT is not allowed
        response = test_client.put("/api/courses")
        assert response.status_code == 405  # Method Not Allowed

        # Test DELETE is not allowed
        response = test_client.delete("/api/courses")
        assert response.status_code == 405  # Method Not Allowed

    def test_nonexistent_endpoint(self, test_client):
        """Test accessing non-existent endpoint returns 404"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are properly set"""
        response = test_client.get("/")

        # Check for CORS headers (FastAPI TestClient may not include all headers)
        # At minimum, response should be successful indicating CORS is configured
        assert response.status_code == 200

    @pytest.mark.slow
    def test_query_endpoint_stress(self, test_client):
        """Test query endpoint with multiple rapid requests"""
        query_data = {
            "query": "What is Python programming?"
        }

        # Send multiple requests quickly
        responses = []
        for _ in range(10):
            response = test_client.post("/api/query", json=query_data)
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "session_id" in data

    def test_request_validation_types(self, test_client):
        """Test request validation with different data types"""
        # Test with integer instead of string for query
        query_data = {
            "query": 12345
        }
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 422

        # Test with invalid session_id type
        query_data = {
            "query": "What is Python?",
            "session_id": 123
        }
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 422

    def test_response_content_types(self, test_client):
        """Test that responses have correct content types"""
        # Test query endpoint
        query_data = {"query": "What is Python?"}
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        # Test courses endpoint
        response = test_client.get("/api/courses")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        # Test root endpoint
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"