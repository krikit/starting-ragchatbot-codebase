"""
Real integration test against the actual RAG system to identify issues.
This test runs against the real components without mocking.
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from rag_system import RAGSystem


class TestRealIntegration:
    """Test against real RAG system to identify actual issues"""

    def setup_method(self):
        """Setup real RAG system for testing"""
        # Use actual config but with a test API key
        self.config = Config()

        # Skip tests if no API key is available
        if not self.config.ANTHROPIC_API_KEY:
            pytest.skip("No ANTHROPIC_API_KEY available for integration test")

        # Initialize real RAG system
        self.rag_system = RAGSystem(self.config)

    def test_tool_manager_tools_available(self):
        """Test that tools are properly registered"""
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()

        assert len(tool_definitions) > 0, "No tools registered in tool manager"

        tool_names = [tool["name"] for tool in tool_definitions]
        assert (
            "search_course_content" in tool_names
        ), "search_course_content tool not found"
        assert "get_course_outline" in tool_names, "get_course_outline tool not found"

    def test_search_tool_direct_execution(self):
        """Test direct execution of search tool"""
        # Test with a simple query that should work even without course data
        result = self.rag_system.search_tool.execute("Python")

        # Should return a string result, not raise an exception
        assert isinstance(result, str), f"Expected string result, got {type(result)}"

        # Should not be None or empty
        assert result is not None, "Search tool returned None"
        assert result != "", "Search tool returned empty string"

        print(f"Search tool result: {result}")

    def test_vector_store_connection(self):
        """Test that vector store is properly initialized and can be queried"""
        try:
            # Try to get course count (should work even with empty database)
            course_count = self.rag_system.vector_store.get_course_count()
            assert isinstance(course_count, int), "Course count should be an integer"
            print(f"Course count: {course_count}")

            # Try a simple search
            results = self.rag_system.vector_store.search("test query")
            assert hasattr(
                results, "documents"
            ), "Search results should have documents attribute"
            assert hasattr(
                results, "error"
            ), "Search results should have error attribute"

            print(f"Search results error: {results.error}")
            print(f"Search results document count: {len(results.documents)}")

        except Exception as e:
            pytest.fail(f"Vector store connection failed: {e}")

    def test_ai_generator_basic_functionality(self):
        """Test that AI generator can generate responses"""
        try:
            # Test without tools first
            response = self.rag_system.ai_generator.generate_response(
                "What is 2 + 2?",
                conversation_history=None,
                tools=None,
                tool_manager=None,
            )

            assert isinstance(
                response, str
            ), f"Expected string response, got {type(response)}"
            assert response != "", "AI generator returned empty response"
            assert "4" in response, "AI should be able to answer basic math"

            print(f"Basic AI response: {response}")

        except Exception as e:
            pytest.fail(f"AI generator basic functionality failed: {e}")

    def test_ai_generator_with_tools(self):
        """Test that AI generator can work with tools"""
        try:
            # Test with tools but a non-course question
            response = self.rag_system.ai_generator.generate_response(
                "What is Python programming language?",
                conversation_history=None,
                tools=self.rag_system.tool_manager.get_tool_definitions(),
                tool_manager=self.rag_system.tool_manager,
            )

            assert isinstance(
                response, str
            ), f"Expected string response, got {type(response)}"
            assert response != "", "AI generator returned empty response with tools"

            print(f"AI response with tools: {response}")

        except Exception as e:
            pytest.fail(f"AI generator with tools failed: {e}")

    @pytest.mark.skipif(not Config().ANTHROPIC_API_KEY, reason="Requires API key")
    def test_full_rag_query_flow(self):
        """Test the complete RAG query flow"""
        try:
            # Test with a simple query
            query = "What is programming?"
            response, sources = self.rag_system.query(query)

            assert isinstance(
                response, str
            ), f"Expected string response, got {type(response)}"
            assert isinstance(
                sources, list
            ), f"Expected list of sources, got {type(sources)}"

            # Response should not be empty
            assert response != "", "RAG query returned empty response"

            # Should not contain error messages that indicate total failure
            assert (
                "query failed" not in response.lower()
            ), f"RAG query failed: {response}"

            print(f"Full RAG query response: {response}")
            print(f"Sources: {sources}")

        except Exception as e:
            pytest.fail(f"Full RAG query flow failed: {e}")

    def test_course_data_availability(self):
        """Test if course data is available in the system"""
        try:
            # Check if docs folder exists and has content
            docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
            if os.path.exists(docs_path):
                files = [
                    f
                    for f in os.listdir(docs_path)
                    if f.endswith((".txt", ".pdf", ".docx"))
                ]
                print(f"Found {len(files)} potential course files in docs folder")

                # Try to add course data if none exists
                if self.rag_system.vector_store.get_course_count() == 0:
                    added_courses, added_chunks = self.rag_system.add_course_folder(
                        docs_path
                    )
                    print(f"Added {added_courses} courses with {added_chunks} chunks")
            else:
                print("No docs folder found - system will have no course data")

            # Check final course count
            final_count = self.rag_system.vector_store.get_course_count()
            print(f"Final course count: {final_count}")

        except Exception as e:
            print(f"Course data check failed: {e}")

    def test_error_scenarios(self):
        """Test various error scenarios"""
        # Test with empty query
        try:
            response, sources = self.rag_system.query("")
            print(f"Empty query response: {response}")
        except Exception as e:
            print(f"Empty query error: {e}")

        # Test with very long query
        try:
            long_query = "What is " + "very " * 100 + "long query about programming?"
            response, sources = self.rag_system.query(long_query)
            print(f"Long query response length: {len(response)}")
        except Exception as e:
            print(f"Long query error: {e}")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__ + "::TestRealIntegration", "-v", "-s"])
