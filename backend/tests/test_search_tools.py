import os
import sys
from unittest.mock import MagicMock, Mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""

    def setup_method(self):
        """Setup for each test method"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_successful_search(self):
        """Test successful search with results"""
        # Arrange
        mock_results = SearchResults(
            documents=["Course content about Python basics"],
            metadata=[{"course_title": "Python Programming", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("Python basics")

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics", course_name=None, lesson_number=None
        )
        assert "[Python Programming - Lesson 1]" in result
        assert "Course content about Python basics" in result
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0] == "Python Programming - Lesson 1"

    def test_execute_with_course_filter(self):
        """Test search with course name filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Advanced Python concepts"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 2}],
            distances=[0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("concepts", course_name="Advanced Python")

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="concepts", course_name="Advanced Python", lesson_number=None
        )
        assert "[Advanced Python - Lesson 2]" in result

    def test_execute_with_lesson_filter(self):
        """Test search with lesson number filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 3}],
            distances=[0.15],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("data analysis", lesson_number=3)

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="data analysis", course_name=None, lesson_number=3
        )
        assert "[Data Science - Lesson 3]" in result

    def test_execute_empty_results(self):
        """Test when search returns no results"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("nonexistent topic")

        # Assert
        assert "No relevant content found" in result
        assert len(self.search_tool.last_sources) == 0

    def test_execute_empty_results_with_filters(self):
        """Test empty results with filter information"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(
            "topic", course_name="Nonexistent Course", lesson_number=999
        )

        # Assert
        assert (
            "No relevant content found in course 'Nonexistent Course' in lesson 999"
            in result
        )

    def test_execute_search_error(self):
        """Test when search returns an error"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("any query")

        # Assert
        assert result == "Database connection failed"

    def test_execute_multiple_results(self):
        """Test handling of multiple search results"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("multiple topics")

        # Assert
        assert "[Course A - Lesson 1]" in result
        assert "[Course B]" in result
        assert "Content 1" in result
        assert "Content 2" in result
        assert len(self.search_tool.last_sources) == 2

    def test_get_tool_definition(self):
        """Test tool definition structure"""
        # Act
        definition = self.search_tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""

    def setup_method(self):
        """Setup for each test method"""
        self.mock_vector_store = Mock()
        self.outline_tool = CourseOutlineTool(self.mock_vector_store)

    def test_execute_successful_outline(self):
        """Test successful course outline retrieval"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = "Python Programming"
        mock_results = {
            "metadatas": [
                {
                    "title": "Python Programming",
                    "instructor": "John Doe",
                    "course_link": "https://example.com/python",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Basics", "lesson_link": "link1"}]',
                }
            ]
        }
        self.mock_vector_store.course_catalog.get.return_value = mock_results

        # Act
        result = self.outline_tool.execute("Python")

        # Assert
        assert "**Python Programming**" in result
        assert "Instructor: John Doe" in result
        assert "Course Link: https://example.com/python" in result
        assert "1. Basics - link1" in result

    def test_execute_course_not_found(self):
        """Test when course is not found"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = None

        # Act
        result = self.outline_tool.execute("Nonexistent Course")

        # Assert
        assert "No course found matching 'Nonexistent Course'" in result

    def test_get_tool_definition(self):
        """Test outline tool definition structure"""
        # Act
        definition = self.outline_tool.get_tool_definition()

        # Assert
        assert definition["name"] == "get_course_outline"
        assert "course_title" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_title"]


class TestToolManager:
    """Test suite for ToolManager"""

    def setup_method(self):
        """Setup for each test method"""
        self.tool_manager = ToolManager()
        self.mock_tool = Mock()
        self.mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        self.mock_tool.execute.return_value = "test result"

    def test_register_tool(self):
        """Test tool registration"""
        # Act
        self.tool_manager.register_tool(self.mock_tool)

        # Assert
        assert "test_tool" in self.tool_manager.tools
        assert self.tool_manager.tools["test_tool"] == self.mock_tool

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        definitions = self.tool_manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"

    def test_execute_tool(self):
        """Test tool execution"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        result = self.tool_manager.execute_tool("test_tool", param1="value1")

        # Assert
        self.mock_tool.execute.assert_called_once_with(param1="value1")
        assert result == "test result"

    def test_execute_nonexistent_tool(self):
        """Test execution of non-existent tool"""
        # Act
        result = self.tool_manager.execute_tool("nonexistent_tool")

        # Assert
        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self):
        """Test getting last sources from tools"""
        # Arrange
        mock_search_tool = Mock()
        mock_search_tool.get_tool_definition.return_value = {"name": "search"}
        mock_search_tool.last_sources = ["Source 1", "Source 2"]
        self.tool_manager.register_tool(mock_search_tool)

        # Act
        sources = self.tool_manager.get_last_sources()

        # Assert
        assert sources == ["Source 1", "Source 2"]

    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        # Arrange
        mock_search_tool = Mock()
        mock_search_tool.get_tool_definition.return_value = {"name": "search"}
        mock_search_tool.last_sources = ["Source 1"]
        self.tool_manager.register_tool(mock_search_tool)

        # Act
        self.tool_manager.reset_sources()

        # Assert
        assert mock_search_tool.last_sources == []
