"""
HuggingFace LLM Agent

Implementation of LLM agent using HuggingFace transformers.
"""

import json
from typing import Any, Dict

try:
    import torch
    from transformers import pipeline

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from .base_agent import BaseAgent


class HuggingFaceAgent(BaseAgent):
    """HuggingFace-based LLM agent using text generation models"""

    def __init__(
        self,
        config: Any,
        device: str = None
    ):
        """
        Initialize HuggingFace agent

        Args:
            config: Config object
            device: Device to use ('cpu', 'cuda', etc.). Auto-detect if None
        """
        super().__init__()

        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "HuggingFace packages not installed. "
                "Install with: pip install transformers torch"
            )

        if not config:
            raise ValueError("config is required")

        # Get temperature from config
        if hasattr(config, 'llm_temperature'):
            self.temperature = config.llm_temperature
        else:
            self.temperature = 0.7  # Default fallback

        # Get max_tokens from config
        if hasattr(config, 'llm_max_tokens'):
            self.max_tokens = config.llm_max_tokens
        else:
            self.max_tokens = 1024  # Default fallback

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Get text generation model from config or use default
        # For HuggingFace, we can use a text generation model like GPT-2, T5, etc.
        if hasattr(config, 'openai_model'):
            # Use the same model name pattern if available
            self.text_generator_model = config.openai_model
        else:
            self.text_generator_model = "gpt2"  # Default fallback

        self.model = self.text_generator_model

        # Initialize text generator lazily when needed
        self.text_generator = None

        self.logger.info(
            f"HuggingFace agent initialized with model: {self.text_generator_model} on {device}"
        )

    def run_llm(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Run LLM with given prompt and system prompt

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens in response

        Returns:
            JSON result from LLM
        """
        try:
            # Initialize text generator lazily if not already initialized
            if self.text_generator is None:
                try:
                    self.text_generator = pipeline(
                        "text-generation",
                        model=self.text_generator_model,
                        device=0 if self.device == 'cuda' else -1,
                        return_full_text=False
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to initialize text generator: {e}")
                    # Fallback: return empty dict if text generation not available
                    return {}

            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}\n\nRespond with JSON only:"

            # Generate text
            result = self.text_generator(
                full_prompt,
                max_length=len(full_prompt.split()) + max_tokens,
                num_return_sequences=1,
                temperature=self.temperature,
                do_sample=True,
                truncation=True
            )

            result_text = result[0]['generated_text'].strip() if result else ""

            if not result_text:
                self.logger.warning("Empty response from text generator")
                return {}

            # Try to extract JSON from the response
            try:
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                if start_index >= 0 and end_index > start_index:
                    json_text = result_text[start_index:end_index]
                    result = json.loads(json_text)
                    return result
                else:
                    # If no JSON found, try parsing the whole text
                    result = json.loads(result_text)
                    return result
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse JSON response: {e}. Response text: {result_text[:200]}"
                )
                # Return empty dict if JSON parsing fails
                return {}
        except Exception as e:
            self.logger.error(f"Error in run_llm: {e}")
            return {}

