from abc import ABC, abstractmethod

from .types import SearchResult


class BaseSearch(ABC):
    """
    Abstract base class for web search engines.

    This class defines the interface that all search engine implementations
    must follow to ensure consistent behavior across different providers.
    """

    @abstractmethod
    def search(self, query: str, **kwargs) -> SearchResult:
        """
        Perform a search query.

        Args:
            query: The search query string
            **kwargs: Additional search parameters

        Returns:
            SearchResponse: Structured search results

        Raises:
            Exception: If search fails
        """

    @abstractmethod
    def get_config(self) -> dict:
        """
        Get the current configuration of the search engine.

        Returns:
            dict: Current configuration parameters
        """
