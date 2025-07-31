"""
State definitions for the DeepResearcher agent.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ResearchState(TypedDict):
    """Main state for the research workflow."""

    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    context: Dict[str, Any]


class Query(TypedDict):
    """Individual query with rationale."""

    query: str
    rationale: str


class QueryState(TypedDict):
    """State for query generation."""

    query_list: List[Query]


class WebSearchState(TypedDict):
    """State for web search operations."""

    search_query: str
    id: str


class ReflectionState(TypedDict):
    """State for reflection and evaluation."""

    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class ResearchOutput(BaseModel):
    """Output from research process."""

    content: str = Field(default="")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.now)
