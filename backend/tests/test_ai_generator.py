import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator tool calling functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.api_key = "test-api-key"
        self.model = "claude-sonnet-4-20250514"

        # Mock the anthropic client
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)

    def test_init(self):
        """Test AIGenerator initialization"""
        assert self.ai_generator.model == self.model
        assert self.ai_generator.base_params["model"] == self.model
        assert self.ai_generator.base_params["temperature"] == 0
        assert self.ai_generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self):
        """Test response generation without tools"""
        # Arrange
        mock_response = Mock()
        mock_response.content = [Mock(text="Simple response")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = self.ai_generator.generate_response("What is Python?")

        # Assert
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args[1]
        assert result == "Simple response"

    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        # Arrange
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = self.ai_generator.generate_response(
            "Follow up question", conversation_history="Previous conversation"
        )

        # Assert
        call_args = self.mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation" in system_content

    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but not used"""
        # Arrange
        mock_response = Mock()
        mock_response.content = [Mock(text="Response without tools")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()

        # Act
        result = self.ai_generator.generate_response(
            "General question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        call_args = self.mock_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"]["type"] == "auto"
        assert result == "Response without tools"

    def test_generate_response_with_tool_use(self):
        """Test response generation with tool usage"""
        # Arrange
        # First response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python basics"}
        mock_tool_content.id = "tool_use_123"

        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"

        # Final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Final response using search results")]

        # Set up client to return both responses
        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Search results: Python is a programming language"
        )

        tools = [{"name": "search_course_content", "description": "Search tool"}]

        # Act
        result = self.ai_generator.generate_response(
            "What is Python?", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        # Should call client twice - once for initial request, once for final response
        assert self.mock_client.messages.create.call_count == 2

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify final response
        assert result == "Final response using search results"

    def test_handle_tool_execution_single_tool(self):
        """Test tool execution handling with single tool call"""
        # Arrange
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "machine learning"}
        mock_tool_content.id = "tool_123"

        initial_response = Mock()
        initial_response.content = [mock_tool_content]

        base_params = {
            "messages": [{"role": "user", "content": "Tell me about ML"}],
            "system": "You are a helpful assistant",
        }

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "ML is a subset of AI"

        final_response = Mock()
        final_response.content = [Mock(text="Based on search: ML explanation")]
        self.mock_client.messages.create.return_value = final_response

        # Act
        result = self.ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Assert
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning"
        )

        # Verify final API call structure
        call_args = self.mock_client.messages.create.call_args
        messages = call_args[1]["messages"]

        # Should have 3 messages: original user message, assistant response, tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_123"
        assert messages[2]["content"][0]["content"] == "ML is a subset of AI"

        assert result == "Based on search: ML explanation"

    def test_handle_tool_execution_multiple_tools(self):
        """Test tool execution handling with multiple tool calls"""
        # Arrange
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "search_course_content"
        mock_tool1.input = {"query": "Python"}
        mock_tool1.id = "tool_1"

        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "get_course_outline"
        mock_tool2.input = {"course_title": "Python Course"}
        mock_tool2.id = "tool_2"

        initial_response = Mock()
        initial_response.content = [mock_tool1, mock_tool2]

        base_params = {
            "messages": [{"role": "user", "content": "Tell me about Python course"}],
            "system": "You are a helpful assistant",
        }

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Python search results",
            "Python course outline",
        ]

        final_response = Mock()
        final_response.content = [Mock(text="Combined response from both tools")]
        self.mock_client.messages.create.return_value = final_response

        # Act
        result = self.ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Assert
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify both tools were called
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("search_course_content",)
        assert calls[0][1] == {"query": "Python"}
        assert calls[1][0] == ("get_course_outline",)
        assert calls[1][1] == {"course_title": "Python Course"}

        # Verify final message structure includes both tool results
        call_args = self.mock_client.messages.create.call_args
        messages = call_args[1]["messages"]
        tool_results = messages[2]["content"]

        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[1]["tool_use_id"] == "tool_2"

        assert result == "Combined response from both tools"

    def test_system_prompt_content(self):
        """Test that system prompt contains expected tool guidance"""
        system_prompt = AIGenerator.SYSTEM_PROMPT

        # Check for key tool guidance elements
        assert "get_course_outline" in system_prompt
        assert "search_course_content" in system_prompt
        assert "One tool call per query maximum" in system_prompt
        assert "No meta-commentary" in system_prompt

    def test_error_handling_in_tool_execution(self):
        """Test error handling during tool execution"""
        # Arrange
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_123"

        initial_response = Mock()
        initial_response.content = [mock_tool_content]

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test",
        }

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution failed"

        final_response = Mock()
        final_response.content = [Mock(text="Error response")]
        self.mock_client.messages.create.return_value = final_response

        # Act
        result = self.ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Assert
        # Should still complete the flow even with tool errors
        assert result == "Error response"
        mock_tool_manager.execute_tool.assert_called_once()

    def test_non_tool_content_ignored_in_tool_execution(self):
        """Test that non-tool content blocks are ignored during tool execution"""
        # Arrange
        mock_text_content = Mock()
        mock_text_content.type = "text"

        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_123"

        initial_response = Mock()
        initial_response.content = [mock_text_content, mock_tool_content]

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test",
        }

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        final_response = Mock()
        final_response.content = [Mock(text="Final response")]
        self.mock_client.messages.create.return_value = final_response

        # Act
        result = self.ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Assert
        # Should only execute the tool_use content, not the text content
        assert mock_tool_manager.execute_tool.call_count == 1
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test"
        )
