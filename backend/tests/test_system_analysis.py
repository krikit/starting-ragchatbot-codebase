"""
System analysis test to identify issues without requiring API keys.
This test examines the system components and identifies potential failure points.
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import Mock, patch

from config import Config
from rag_system import RAGSystem
from vector_store import SearchResults


class TestSystemAnalysis:
    """Analyze system components to identify potential failure points"""

    def setup_method(self):
        """Setup system analysis"""
        self.config = Config()
        self.config.ANTHROPIC_API_KEY = "test-key"  # Use fake key for testing

    def test_vector_store_initialization(self):
        """Test vector store can be initialized"""
        from vector_store import VectorStore

        try:
            # Initialize with test path
            vector_store = VectorStore("./test_chroma", "all-MiniLM-L6-v2", 5)
            assert vector_store is not None
            print("✅ Vector store initializes successfully")

            # Test basic functionality
            course_count = vector_store.get_course_count()
            assert isinstance(course_count, int)
            print(f"✅ Vector store course count: {course_count}")

        except Exception as e:
            pytest.fail(f"❌ Vector store initialization failed: {e}")

    def test_search_results_structure(self):
        """Test SearchResults class functionality"""
        from vector_store import SearchResults

        # Test normal results
        results = SearchResults(
            documents=["Test doc"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1],
            error=None,
        )

        assert not results.is_empty()
        assert results.error is None

        # Test empty results
        empty_results = SearchResults.empty("No results")
        assert empty_results.is_empty()
        assert empty_results.error == "No results"

        print("✅ SearchResults class works correctly")

    def test_search_tool_without_data(self):
        """Test search tool behavior with empty vector store"""
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        # Create empty vector store
        vector_store = VectorStore("./test_empty_chroma", "all-MiniLM-L6-v2", 5)

        # Create search tool
        search_tool = CourseSearchTool(vector_store)

        # Test search with no data
        result = search_tool.execute("test query")

        # Should return a meaningful message, not crash
        assert isinstance(result, str)
        assert result != ""
        print(f"✅ Search tool with empty data returns: {result}")

        # Should handle filters gracefully
        result_with_filter = search_tool.execute("test", course_name="Nonexistent")
        assert isinstance(result_with_filter, str)
        print(f"✅ Search tool with filter returns: {result_with_filter}")

    def test_tool_manager_functionality(self):
        """Test tool manager can register and execute tools"""
        from search_tools import CourseSearchTool, ToolManager
        from vector_store import VectorStore

        vector_store = VectorStore("./test_chroma_tm", "all-MiniLM-L6-v2", 5)
        search_tool = CourseSearchTool(vector_store)

        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)

        # Test tool definitions
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) > 0
        assert definitions[0]["name"] == "search_course_content"

        # Test tool execution
        result = tool_manager.execute_tool("search_course_content", query="test")
        assert isinstance(result, str)

        print("✅ Tool manager functionality works")

    def test_config_loading(self):
        """Test configuration loading"""
        config = Config()

        # Check required attributes
        assert hasattr(config, "ANTHROPIC_API_KEY")
        assert hasattr(config, "ANTHROPIC_MODEL")
        assert hasattr(config, "EMBEDDING_MODEL")
        assert hasattr(config, "CHUNK_SIZE")
        assert hasattr(config, "MAX_RESULTS")

        # Check default values
        assert config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert config.CHUNK_SIZE == 800
        assert config.MAX_RESULTS == 5

        print("✅ Configuration loads with correct defaults")

    def test_rag_system_component_initialization(self):
        """Test RAG system components can be initialized"""
        with patch("rag_system.AIGenerator") as mock_ai:
            mock_ai.return_value = Mock()

            try:
                rag_system = RAGSystem(self.config)

                # Check all components exist
                assert hasattr(rag_system, "vector_store")
                assert hasattr(rag_system, "document_processor")
                assert hasattr(rag_system, "ai_generator")
                assert hasattr(rag_system, "session_manager")
                assert hasattr(rag_system, "tool_manager")
                assert hasattr(rag_system, "search_tool")
                assert hasattr(rag_system, "outline_tool")

                print("✅ RAG system components initialize successfully")

                # Test tool registration
                tools = rag_system.tool_manager.get_tool_definitions()
                tool_names = [t["name"] for t in tools]
                assert "search_course_content" in tool_names
                assert "get_course_outline" in tool_names

                print("✅ RAG system tools are registered correctly")

            except Exception as e:
                pytest.fail(f"❌ RAG system component initialization failed: {e}")

    def test_course_data_directory(self):
        """Check if course data directory exists"""
        docs_path = os.path.join(os.path.dirname(__file__), "../../docs")

        if os.path.exists(docs_path):
            files = os.listdir(docs_path)
            course_files = [f for f in files if f.endswith((".txt", ".pdf", ".docx"))]

            print(
                f"✅ Found docs directory with {len(course_files)} potential course files:"
            )
            for file in course_files[:5]:  # Show first 5 files
                print(f"   - {file}")

            if len(course_files) == 0:
                print("⚠️  No course files found in docs directory")

        else:
            print(
                "❌ No docs directory found - this could cause 'query failed' responses"
            )

    def test_identify_potential_issues(self):
        """Identify potential issues that could cause 'query failed'"""
        issues = []

        # Check API key
        if not self.config.ANTHROPIC_API_KEY:
            issues.append("❌ No ANTHROPIC_API_KEY configured")
        else:
            print("✅ ANTHROPIC_API_KEY is configured")

        # Check docs directory
        docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
        if not os.path.exists(docs_path):
            issues.append("❌ No docs directory found for course data")

        # Check chroma db permissions
        chroma_path = "./chroma_db"
        try:
            if not os.path.exists(chroma_path):
                os.makedirs(chroma_path, exist_ok=True)
            print("✅ ChromaDB directory is accessible")
        except Exception as e:
            issues.append(f"❌ Cannot create ChromaDB directory: {e}")

        # Test vector store creation
        try:
            from vector_store import VectorStore

            test_store = VectorStore("./test_permissions", "all-MiniLM-L6-v2", 5)
            print("✅ Vector store can be created")
        except Exception as e:
            issues.append(f"❌ Cannot create vector store: {e}")

        if issues:
            print("\n🚨 POTENTIAL ISSUES IDENTIFIED:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ No obvious configuration issues detected")

        return issues


if __name__ == "__main__":
    pytest.main([__file__ + "::TestSystemAnalysis", "-v", "-s"])
