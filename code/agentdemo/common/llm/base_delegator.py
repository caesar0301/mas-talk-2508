from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

T = TypeVar("T")


class BaseLLMDelegator:
    """Client for interacting with LLMs via OpenRouter using OpenAI SDK."""

    def __init__(self, instructor: bool = False):
        """
        Initialize the LLM client.

        Args:
            instructor: Whether to enable instructor for structured output
        """

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
        raise NotImplementedError("Chat completion is not implemented in the base class")

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
        raise NotImplementedError("Structured completion is not implemented in the base class")

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
        raise NotImplementedError("Image understanding is not implemented in the base class")

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
        raise NotImplementedError("Image understanding from URL is not implemented in the base class")
