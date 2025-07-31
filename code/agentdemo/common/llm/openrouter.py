"""
LLM utilities for AgentDemo using OpenRouter via OpenAI SDK.

This module provides:
- Chat completion using various models via OpenRouter
- Text embeddings using OpenAI text-embedding-3-small
- Image understanding using vision models
- Instructor integration for structured output
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from openai import OpenAI

from agentdemo.agents.deep_researcher.configuration import GEMINI_FLASH
from agentdemo.common.llm.base_delegator import BaseLLMDelegator

# Import instructor for structured output
try:
    from instructor import Instructor, Mode, patch

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    Instructor = None
    Mode = None
    patch = None

T = TypeVar("T")

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


class LLMClient(BaseLLMDelegator):
    """Client for interacting with LLMs via OpenRouter using OpenAI SDK."""

    def __init__(self, instructor: bool = False):
        """
        Initialize the LLM client.

        Args:
            instructor: Whether to enable instructor for structured output
        """

        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")

        # Initialize OpenAI client with OpenRouter configuration
        self.client = OpenAI(api_key=self.openrouter_api_key, base_url="https://openrouter.ai/api/v1")

        # Model configurations (can be overridden by environment variables)
        self.chat_model = os.getenv("CHAT_MODEL", GEMINI_FLASH)
        self.vision_model = os.getenv("VISION_MODEL", GEMINI_FLASH)

        # Initialize instructor if requested
        self.instructor = None
        if instructor:
            if not INSTRUCTOR_AVAILABLE:
                raise ValueError("Instructor is required for structured output. Install with: pip install instructor")

            # Create instructor instance for structured output
            patched_client = patch(self.client, mode=Mode.JSON)
            self.instructor = Instructor(
                client=patched_client,
                create=patched_client.chat.completions.create,
                mode=Mode.JSON,
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate chat completion using the configured model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to OpenAI API

        Returns:
            Generated text response or streaming response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )

            if stream:
                return response
            else:
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> T:
        """
        Generate structured completion using instructor.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_model: Pydantic model class for structured output
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to instructor

        Returns:
            Structured response as the specified model type
        """
        if not self.instructor:
            raise ValueError("Instructor is not enabled. Initialize LLMClient with instructor=True")

        try:
            result = self.instructor.create(
                model=self.chat_model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return result

        except Exception as e:
            logger.error(f"Error in structured completion: {e}")
            raise

    def understand_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image using the configured vision model.

        Args:
            image_path: Path to the image file
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image
        """
        try:
            # Read and encode the image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ]

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

    def understand_image_from_url(
        self,
        image_url: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image from URL using the configured vision model.

        Args:
            image_url: URL of the image
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image from URL: {e}")
            raise


# Convenience functions for easy usage
def get_llm_client() -> LLMClient:
    """
    Get an LLM client instance.

    Returns:
        LLMClient instance
    """
    return LLMClient()


def get_llm_client_instructor() -> LLMClient:
    """
    Get an LLM client instance with instructor support.

    Returns:
        LLMClient instance with instructor enabled
    """
    return LLMClient(instructor=True)
