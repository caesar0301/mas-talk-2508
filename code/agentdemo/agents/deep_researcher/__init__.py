"""
DeepResearcher Module

This module provides advanced research capabilities using LangGraph and LLM integration
with OpenRouter.

The researcher requires both OPENROUTER_API_KEY and GEMINI_API_KEY environment
variables. OPENROUTER_API_KEY is required for LLM functionality, and GEMINI_API_KEY
is required for real web search capabilities.
"""

# Enable colorful logging by default for deep_researcher
from agentdemo.common.logging_config import setup_logging

setup_logging(level="INFO", enable_colors=True)

from .configuration import Configuration
from .prompts import get_research_prompts
from .researcher import DeepResearcher
from .schemas import Reflection, SearchQueryList
from .state import Query, QueryState, ReflectionState, ResearchState, WebSearchState

__all__ = [
    "DeepResearcher",
    "ResearchState",
    "QueryState",
    "WebSearchState",
    "ReflectionState",
    "Query",
    "SearchQueryList",
    "Reflection",
    "Configuration",
    "get_research_prompts",
]
