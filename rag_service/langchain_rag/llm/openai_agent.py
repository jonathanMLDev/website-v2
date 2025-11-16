"""
OpenAI LLM Agent

Implementation of LLM agent using OpenAI API with synchronous processing.
"""

import json
from typing import Any, Dict

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_agent import BaseAgent


class OpenAIAgent(BaseAgent):
    """OpenAI-based LLM agent"""

    def __init__(
        self,
        config: Any,
        api_key: str = None,
        host_url: str = None
    ):
        """
        Initialize OpenAI agent

        Args:
            config: Config object with openai_model, llm_temperature, llm_max_tokens
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            host_url: OpenAI host URL (or use OPENAI_HOST_URL env var)
        """
        super().__init__()

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        import os

        if not config:
            raise ValueError("config is required")

        # Get model from config
        if hasattr(config, 'openai_model'):
            model = config.openai_model
        else:
            model = "gpt-3.5-turbo"  # Default fallback

        # Get temperature from config
        if hasattr(config, 'llm_temperature'):
            temperature = config.llm_temperature
        else:
            temperature = 0.3  # Default fallback

        # Get max_tokens from config
        if hasattr(config, 'llm_max_tokens'):
            max_tokens = config.llm_max_tokens
        else:
            max_tokens = 500  # Default fallback

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        host_url = host_url or os.getenv("OPENAI_HOST_URL")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.client_sync = OpenAI(api_key=api_key, base_url=host_url)
        self.logger.info(f"OpenAI agent initialized with model: {model}")

    def run_llm(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 500  # noqa: ARG002
    ) -> Dict[str, Any]:
        """
        Run LLM with given prompt and system prompt

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens in response (currently not used, kept for interface compatibility)

        Returns:
            JSON result from LLM
        """
        try:
            response = self.client_sync.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                # max_tokens=max_tokens,  # Not used - model determines response length
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content.strip()
            if not result_text:
                self.logger.warning("Empty response from API")
                return {}

            try:
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                json_text = result_text[start_index:end_index]
                result = json.loads(json_text)
                return result
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}. Response text: {result_text}")
                return {}
        except Exception as e:
            self.logger.error(f"Error in run_llm: {e}")
            return {}
