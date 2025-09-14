"""
Test OpenAI integration to verify the system works with ChatGPT.
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import Mock, patch

from ai_generator import AIGenerator
from config import Config
from rag_system import RAGSystem


class TestOpenAIIntegration:
    """Test OpenAI integration functionality"""

    def test_ai_generator_initialization(self):
        """Test that AIGenerator initializes with OpenAI"""
        api_key = "test-key"
        model = "gpt-4o-mini"

        with patch("ai_generator.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            ai_gen = AIGenerator(api_key, model)

            assert ai_gen.model == model
            assert ai_gen.base_params["model"] == model
            mock_openai.assert_called_once_with(api_key=api_key)

    def test_tool_format_conversion(self):
        """Test conversion from Anthropic to OpenAI tool format"""
        with patch("ai_generator.openai.OpenAI"):
            ai_gen = AIGenerator("test-key", "gpt-4o-mini")

            # Anthropic format tools
            anthropic_tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course materials",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    },
                }
            ]

            # Convert to OpenAI format
            openai_tools = ai_gen._convert_tools_to_openai_format(anthropic_tools)

            expected = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_course_content",
                        "description": "Search course materials",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]

            assert openai_tools == expected

    def test_config_uses_openai(self):
        """Test that config is set up for OpenAI"""
        config = Config()

        # Should have OpenAI settings
        assert hasattr(config, "OPENAI_API_KEY")
        assert hasattr(config, "OPENAI_MODEL")
        assert config.OPENAI_MODEL == "gpt-4o-mini"

        # Should not have Anthropic settings
        assert not hasattr(config, "ANTHROPIC_API_KEY")

    def test_rag_system_uses_openai(self):
        """Test that RAG system initializes with OpenAI"""
        config = Config()
        config.OPENAI_API_KEY = "test-key"

        with patch("rag_system.AIGenerator") as mock_ai_gen:
            mock_ai_gen.return_value = Mock()

            rag_system = RAGSystem(config)

            # Verify AIGenerator was called with OpenAI parameters
            mock_ai_gen.assert_called_once_with(
                config.OPENAI_API_KEY, config.OPENAI_MODEL
            )

    def test_system_without_api_key(self):
        """Test system behavior without API key"""
        config = Config()
        # Don't set OPENAI_API_KEY

        with patch("rag_system.AIGenerator") as mock_ai_gen:
            mock_ai_gen.return_value = Mock()

            rag_system = RAGSystem(config)

            # Should still initialize but with empty key
            mock_ai_gen.assert_called_once_with("", config.OPENAI_MODEL)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_real_openai_request(self):
        """Test actual OpenAI API request (requires real API key)"""
        config = Config()

        if not config.OPENAI_API_KEY:
            pytest.skip("No OPENAI_API_KEY available")

        ai_gen = AIGenerator(config.OPENAI_API_KEY, config.OPENAI_MODEL)

        # Simple test without tools
        response = ai_gen.generate_response("What is 2 + 2?")

        assert isinstance(response, str)
        assert response != ""
        assert "4" in response

        print(f"OpenAI response: {response}")

    def test_error_handling(self):
        """Test error handling with invalid API key"""
        with patch("ai_generator.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception(
                "Invalid API key"
            )
            mock_openai.return_value = mock_client

            ai_gen = AIGenerator("invalid-key", "gpt-4o-mini")
            response = ai_gen.generate_response("test query")

            assert "error" in response.lower()
            assert "invalid api key" in response.lower()


if __name__ == "__main__":
    pytest.main([__file__ + "::TestOpenAIIntegration", "-v", "-s"])
