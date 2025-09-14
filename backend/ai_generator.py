import json
import logging
from typing import Any, Dict, List, Optional

import openai

logger = logging.getLogger(__name__)


class AIGenerator:
    """Handles interactions with OpenAI's GPT API for generating responses"""

    # Static system prompt optimized for OpenAI
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Course outline/structure questions**: Use get_course_outline tool to get complete course information including:
  * Course title and instructor
  * Course link
  * Complete lesson list with numbers, titles, and links
- **Specific content questions**: Use search_course_content tool for detailed educational materials
- **Use tools when relevant to the query**
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

When to Use Each Tool:
- **get_course_outline**: For questions about course structure, lesson lists, course outlines, "what lessons are in...", "what's covered in the course", etc.
- **search_course_content**: For detailed content within lessons, specific topics, explanations, examples from course materials

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool first, then provide comprehensive course information
- **Course content questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the tool results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked."""

    def __init__(self, api_key: str, model: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build messages array for OpenAI
        messages = []

        # Add system message with conversation context if available
        system_content = self.SYSTEM_PROMPT
        if conversation_history:
            system_content += f"\n\nPrevious conversation:\n{conversation_history}"

        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": query})

        # Prepare API call parameters
        api_params = {**self.base_params, "messages": messages}

        # Add tools if available (convert to OpenAI format)
        if tools:
            openai_tools = self._convert_tools_to_openai_format(tools)
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"

        try:
            # Get response from OpenAI
            response = self.client.chat.completions.create(**api_params)

            # Handle tool execution if needed
            if response.choices[0].finish_reason == "tool_calls" and tool_manager:
                return self._handle_tool_execution(response, messages, tool_manager)

            # Return direct response
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

    def _convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools from Anthropic format to OpenAI format"""
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def _handle_tool_execution(
        self, initial_response, messages: List[Dict], tool_manager
    ) -> str:
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """

        # Add assistant's tool call message
        assistant_message = {
            "role": "assistant",
            "content": initial_response.choices[0].message.content,
            "tool_calls": initial_response.choices[0].message.tool_calls,
        }
        messages.append(assistant_message)

        # Execute all tool calls and collect results
        for tool_call in initial_response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute tool
            tool_result = tool_manager.execute_tool(function_name, **function_args)

            # Add tool result message
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
            messages.append(tool_message)

        try:
            # Get final response without tools
            final_params = {**self.base_params, "messages": messages}

            final_response = self.client.chat.completions.create(**final_params)
            return final_response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting final response: {e}")
            return f"I used the tools but encountered an error generating the final response: {str(e)}"
