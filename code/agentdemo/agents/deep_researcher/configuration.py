"""
Configuration management for the DeepResearcher agent.
Based on the gemini-fullstack-langgraph-quickstart backend configuration.
"""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

GEMINI_FLASH = "google/gemini-2.5-flash"


class Configuration(BaseModel):
    """The configuration for the DeepResearcher agent."""

    query_generator_model: str = Field(
        default=GEMINI_FLASH,
        metadata={"description": "The name of the language model to use for query generation."},
    )

    web_search_model: str = Field(
        default=GEMINI_FLASH,
        metadata={"description": "The name of the language model to use for web search."},
    )

    reflection_model: str = Field(
        default=GEMINI_FLASH,
        metadata={"description": "The name of the language model to use for reflection."},
    )

    answer_model: str = Field(
        default="google/gemini-2.5-pro",
        metadata={"description": "The name of the language model to use for answer finalization."},
    )

    number_of_initial_queries: int = Field(
        default=2,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    search_engine: str = Field(
        default="google",
        metadata={"description": "The search engine to use for web search. Options: google, tavily."},
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name)) for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
