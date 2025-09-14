import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from vector_store import SearchResults
from config import Config


class TestRAGSystem:
    """Integration tests for RAG system content query handling"""

    def setup_method(self):
        """Setup for each test method"""
        # Create a test config
        test_config = Config()
        test_config.ANTHROPIC_API_KEY = "test-key"

        # Mock all external dependencies
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class, \
             patch('rag_system.SessionManager'):

            # Create mock instances
            self.mock_vector_store = Mock()
            mock_vector_store_class.return_value = self.mock_vector_store

            self.mock_ai_generator = Mock()
            mock_ai_generator_class.return_value = self.mock_ai_generator

            # Initialize RAG system
            self.rag_system = RAGSystem(test_config)

    def test_query_successful_content_search(self):
        """Test successful content query with search results"""
        # Arrange
        query = "What are Python data types?"

        # Mock AI generator response that includes tool use
        self.mock_ai_generator.generate_response.return_value = "Python has several data types including int, str, list..."

        # Mock tool manager returning sources
        self.rag_system.tool_manager.get_last_sources.return_value = ["Python Basics - Lesson 1"]

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args

        # Verify the query was passed correctly
        assert "What are Python data types?" in call_args[1]["query"]

        # Verify tools were provided
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

        # Verify response
        assert "Python has several data types" in response
        assert sources == ["Python Basics - Lesson 1"]

    def test_query_with_session_history(self):
        """Test query with conversation history"""
        # Arrange
        query = "Tell me more about that"
        session_id = "test_session_123"

        # Mock session manager
        mock_history = "Previous conversation about Python"
        self.rag_system.session_manager.get_conversation_history.return_value = mock_history

        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Based on previous context..."
        self.rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = self.rag_system.query(query, session_id=session_id)

        # Assert
        self.rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)

        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == mock_history

        # Verify session was updated
        self.rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, query, "Based on previous context..."
        )

    def test_query_no_results_scenario(self):
        """Test query when search tools find no results"""
        # Arrange
        query = "Tell me about quantum computing in this course"

        # Mock AI response when no results found
        self.mock_ai_generator.generate_response.return_value = "I couldn't find any relevant content about quantum computing in the available course materials."
        self.rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        assert "couldn't find any relevant content" in response
        assert sources == []

        # Verify tools were still provided to AI
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["tools"] is not None

    def test_query_tool_error_handling(self):
        """Test query when tool execution encounters errors"""
        # Arrange
        query = "What is machine learning?"

        # Mock AI response when tool fails
        self.mock_ai_generator.generate_response.return_value = "I encountered an error while searching for course content. Please try again."
        self.rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        assert "error while searching" in response
        assert sources == []

    def test_query_sources_reset_after_retrieval(self):
        """Test that sources are properly reset after each query"""
        # Arrange
        query = "Python basics"

        self.mock_ai_generator.generate_response.return_value = "Python is a programming language"
        self.rag_system.tool_manager.get_last_sources.return_value = ["Source 1"]

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        self.rag_system.tool_manager.get_last_sources.assert_called_once()
        self.rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_multiple_sources(self):
        """Test query returning multiple sources"""
        # Arrange
        query = "Programming fundamentals"

        self.mock_ai_generator.generate_response.return_value = "Programming involves..."
        self.rag_system.tool_manager.get_last_sources.return_value = [
            "Python Basics - Lesson 1",
            "Programming 101 - Lesson 2",
            "Computer Science - Lesson 3"
        ]

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        assert len(sources) == 3
        assert "Python Basics - Lesson 1" in sources
        assert "Programming 101 - Lesson 2" in sources
        assert "Computer Science - Lesson 3" in sources

    def test_search_tool_integration(self):
        """Test that search tool is properly integrated and configured"""
        # Verify search tool was created and registered
        assert hasattr(self.rag_system, 'search_tool')
        assert hasattr(self.rag_system, 'tool_manager')

        # Verify tool manager has the search tool
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_ai_generator_tool_configuration(self):
        """Test that AI generator receives proper tool configuration"""
        # Arrange
        query = "Test query"
        self.mock_ai_generator.generate_response.return_value = "Test response"
        self.rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        self.rag_system.query(query)

        # Assert
        call_args = self.mock_ai_generator.generate_response.call_args
        tools = call_args[1]["tools"]
        tool_manager = call_args[1]["tool_manager"]

        # Verify tools are provided
        assert tools is not None
        assert len(tools) > 0

        # Verify tool manager is the same instance
        assert tool_manager is self.rag_system.tool_manager

    def test_prompt_formatting(self):
        """Test that queries are properly formatted for AI"""
        # Arrange
        query = "Explain Python variables"
        self.mock_ai_generator.generate_response.return_value = "Variables in Python..."
        self.rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        self.rag_system.query(query)

        # Assert
        call_args = self.mock_ai_generator.generate_response.call_args
        formatted_query = call_args[1]["query"]

        # Should include the original query in a formatted prompt
        assert "Explain Python variables" in formatted_query
        assert "Answer this question about course materials" in formatted_query


class TestRAGSystemInitialization:
    """Test RAG system initialization and component setup"""

    def test_initialization_with_config(self):
        """Test RAG system initializes all components correctly"""
        # Arrange
        test_config = Config()
        test_config.ANTHROPIC_API_KEY = "test-key"

        # Mock all dependencies
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr:

            # Act
            rag_system = RAGSystem(test_config)

            # Assert
            mock_doc_proc.assert_called_once_with(test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP)
            mock_vector_store.assert_called_once_with(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
            mock_ai_gen.assert_called_once_with(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            mock_session_mgr.assert_called_once_with(test_config.MAX_HISTORY)

            # Verify components are assigned
            assert rag_system.config == test_config
            assert hasattr(rag_system, 'document_processor')
            assert hasattr(rag_system, 'vector_store')
            assert hasattr(rag_system, 'ai_generator')
            assert hasattr(rag_system, 'session_manager')
            assert hasattr(rag_system, 'tool_manager')
            assert hasattr(rag_system, 'search_tool')
            assert hasattr(rag_system, 'outline_tool')

    def test_tool_registration(self):
        """Test that tools are properly registered with tool manager"""
        # Arrange
        test_config = Config()
        test_config.ANTHROPIC_API_KEY = "test-key"

        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            # Act
            rag_system = RAGSystem(test_config)

            # Assert
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            tool_names = [tool["name"] for tool in tool_definitions]

            assert len(tool_names) == 2
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system queries"""

    def setup_method(self):
        """Setup for error handling tests"""
        test_config = Config()
        test_config.ANTHROPIC_API_KEY = "test-key"

        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class, \
             patch('rag_system.SessionManager'):

            self.mock_ai_generator = Mock()
            mock_ai_generator_class.return_value = self.mock_ai_generator

            self.rag_system = RAGSystem(test_config)

    def test_query_with_ai_generator_exception(self):
        """Test query handling when AI generator raises exception"""
        # Arrange
        query = "Test query"
        self.mock_ai_generator.generate_response.side_effect = Exception("API Error")

        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            self.rag_system.query(query)

    def test_query_with_empty_query(self):
        """Test query with empty string"""
        # Arrange
        query = ""
        self.mock_ai_generator.generate_response.return_value = "Please provide a specific question."
        self.rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["query"] is not None  # Should still format the query

    def test_query_with_none_query(self):
        """Test query with None input"""
        # Act & Assert
        with pytest.raises(Exception):
            self.rag_system.query(None)