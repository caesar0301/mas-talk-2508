"""
DeepResearcher implementation using LangGraph and LLM integration.
Enhanced base class designed for extensibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agentdemo.common.llm.openrouter import get_llm_client_instructor
from agentdemo.common.logging_config import get_logger

from .configuration import Configuration
from .prompts import get_current_date, get_research_prompts
from .schemas import Reflection, SearchQueryList
from .state import QueryState, ReflectionState, ResearchOutput, ResearchState, WebSearchState

# Configure logging
logger = get_logger(__name__)


class DeepResearcher:
    """
    Advanced research agent using LangGraph and LLM integration.
    Base class designed for extensibility with hooks for specialized researchers.

    To create a specialized researcher, inherit from this class and override:
    - get_prompts(): Return domain-specific prompts
    - get_state_class(): Return domain-specific state class
    - customize_initial_state(): Add domain-specific state fields
    - preprocess_research_topic(): Enhance research topic preprocessing
    - generate_fallback_queries(): Customize fallback query generation
    - customize_reflection_fallback(): Customize reflection logic
    - format_final_answer(): Customize final answer formatting

    Example:
        class MySpecializedResearcher(DeepResearcher):
            def get_prompts(self):
                return my_specialized_prompts()

            def preprocess_research_topic(self, messages):
                topic = super().preprocess_research_topic(messages)
                return f"Specialized context: {topic}"
    """

    def __init__(self, configuration: Optional[Configuration] = None):
        """
        Initialize the DeepResearcher.
        Requires OPENROUTER_API_KEY, GEMINI_API_KEY, and instructor library.
        OPENROUTER_API_KEY is required for LLM functionality.
        GEMINI_API_KEY is required for real web search capabilities.
        Instructor is required for structured output.

        Args:
            configuration: Optional Configuration instance. If not provided,
                          will use default configuration from environment.
        """
        # Initialize LLM client with instructor support
        self.llm_client = get_llm_client_instructor()

        # Load prompts (can be overridden by subclasses)
        self.prompts = self.get_prompts()

        # Set configuration (use provided config or create from environment)
        self.configuration = configuration or Configuration()

        # Create the research graph
        self.graph = self._create_research_graph()

    def get_prompts(self) -> Dict[str, str]:
        """
        Get prompts for the researcher.
        Override this method in subclasses for specialized prompts.

        Returns:
            Dictionary containing all prompts for the research workflow
        """
        return get_research_prompts()

    def get_state_class(self) -> Type:
        """
        Get the state class for this researcher.
        Override this method in subclasses for specialized state.

        Returns:
            The state class to use for the research workflow
        """
        return ResearchState

    def customize_initial_state(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Customize the initial state for research.
        Override this method in subclasses to add domain-specific state.

        Args:
            user_message: User's research request
            context: Additional context for research

        Returns:
            Dictionary containing the initial state
        """
        return {
            "messages": [HumanMessage(content=user_message)],
            "context": context,
            "search_query": [],
            "web_research_result": [],
            "sources_gathered": [],
            "initial_search_query_count": self.configuration.number_of_initial_queries,
            "max_research_loops": self.configuration.max_research_loops,
            "research_loop_count": 0,
            "reasoning_model": self.configuration.reflection_model,
        }

    def preprocess_research_topic(self, messages: List[AnyMessage]) -> str:
        """
        Preprocess the research topic from messages.
        Override this method in subclasses for domain-specific preprocessing.

        Args:
            messages: List of messages from the conversation

        Returns:
            Processed research topic string
        """
        return self._get_research_topic(messages)

    def generate_fallback_queries(self, prompt: str) -> List[str]:
        """
        Generate fallback queries when structured generation fails.
        Override this method in subclasses for domain-specific fallback logic.

        Args:
            prompt: The formatted prompt

        Returns:
            List of fallback queries
        """
        return self._extract_queries_from_response(prompt)

    def customize_reflection_fallback(self, state: ResearchState, research_loop_count: int) -> Dict[str, Any]:
        """
        Customize reflection fallback behavior.
        Override this method in subclasses for domain-specific reflection logic.

        Args:
            state: Current research state
            research_loop_count: Current loop count

        Returns:
            Dictionary containing reflection results
        """
        is_sufficient = research_loop_count >= self.configuration.max_research_loops
        knowledge_gap = "" if is_sufficient else "Need more specific information and practical details"
        follow_up_queries = (
            []
            if is_sufficient
            else [f"More details about {self._get_research_topic(state['messages'])} and practical information"]
        )

        return {
            "is_sufficient": is_sufficient,
            "knowledge_gap": knowledge_gap,
            "follow_up_queries": follow_up_queries,
        }

    def format_final_answer(self, final_answer: str, sources: List[Dict[str, Any]]) -> str:
        """
        Format the final research answer.
        Override this method in subclasses for domain-specific formatting.

        Args:
            final_answer: Generated final answer
            sources: List of sources gathered during research

        Returns:
            Formatted final answer
        """
        return self._format_research_summary([final_answer], sources)

    def _create_research_graph(self) -> StateGraph:
        """Create the LangGraph research workflow."""
        state_class = self.get_state_class()
        workflow = StateGraph(state_class)

        # Add nodes
        workflow.add_node("generate_query", self._generate_query_node)
        workflow.add_node("web_research", self._google_research_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("finalize_answer", self._finalize_answer_node)

        # Set entry point
        workflow.add_edge(START, "generate_query")

        # Add conditional edges
        workflow.add_conditional_edges("generate_query", self._continue_to_web_research, ["web_research"])
        workflow.add_edge("web_research", "reflection")
        workflow.add_conditional_edges("reflection", self._evaluate_research, ["web_research", "finalize_answer"])
        workflow.add_edge("finalize_answer", END)

        return workflow.compile()

    def _format_research_summary(self, summaries: List[str], sources: List[Dict[str, Any]]) -> str:
        """
        Format research summaries with proper citations and structure.

        Args:
            summaries: List of research summaries
            sources: List of source information

        Returns:
            Formatted research summary
        """
        if not summaries:
            return "No research results available."

        # Combine summaries
        combined_summary = "\n\n---\n\n".join(summaries)

        # Add source information if available
        if sources:
            source_section = "\n\n## Sources\n"
            for i, source in enumerate(sources, 1):
                if isinstance(source, dict):
                    url = source.get("value", source.get("url", f"Source {i}"))
                    title = source.get("label", source.get("title", f"Source {i}"))
                    source_section += f"{i}. [{title}]({url})\n"
                else:
                    source_section += f"{i}. {source}\n"
            combined_summary += source_section

        return combined_summary

    def _generate_query_node(self, state: ResearchState, config: RunnableConfig) -> QueryState:
        """Generate search queries based on user request using instructor structured output."""
        # Get research topic from messages (can be customized by subclasses)
        research_topic = self.preprocess_research_topic(state["messages"])

        # Get configuration from RunnableConfig or use instance configuration
        runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = self.prompts["query_writer"].format(
            current_date=current_date,
            research_topic=research_topic,
            number_queries=state.get("initial_search_query_count", runnable_config.number_of_initial_queries),
        )

        try:
            # Generate queries using instructor with structured output
            result: SearchQueryList = self.llm_client.structured_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=SearchQueryList,
                temperature=0.7,
                max_tokens=1000,
            )

            # Create query list with rationale
            query_list = [{"query": q, "rationale": result.rationale} for q in result.query]

            logger.info(f"Generated {len(result.query)} queries: {query_list}")
            return {"query_list": query_list}

        except Exception as e:
            logger.error(f"Error in structured query generation: {e}")
            # Fallback to domain-specific extraction (can be customized by subclasses)
            queries = self.generate_fallback_queries(formatted_prompt)
            query_list = [{"query": q, "rationale": "Generated for research"} for q in queries]
            return {"query_list": query_list}

    def _extract_queries_from_response(self, prompt: str) -> List[str]:
        """Extract search queries from prompt as fallback."""
        # Simple fallback extraction when instructor fails
        lines = prompt.split("\n")
        queries = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("```"):
                # Simple heuristic to identify queries
                if len(line) > 10 and len(line) < 100:  # Reasonable query length
                    queries.append(line)

        # Limit to initial query count from configuration
        return queries[: self.configuration.number_of_initial_queries]

    def _continue_to_web_research(self, state: QueryState):
        """Send queries to web research nodes."""
        return [
            Send("web_research", {"search_query": item["query"], "id": str(idx)})
            for idx, item in enumerate(state["query_list"])
        ]

    def _google_research_node(self, state: WebSearchState, config: RunnableConfig) -> ResearchState:
        """Perform web research using real Google Search API or LLM simulation."""
        from agentdemo.common.websearch import GoogleAISearch

        try:
            search_query = state["search_query"]

            # Initialize Google AI Search client
            google_search = GoogleAISearch()

            # Get configuration from RunnableConfig or use instance configuration
            runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

            # Perform search using the new module
            result = google_search.search(
                query=search_query,
                model=runnable_config.web_search_model.split("/")[-1],
            )

            # Convert SourceInfo objects back to dictionaries for compatibility
            sources_gathered = []
            for source in result.sources:
                sources_gathered.append(source.model_dump())

            return ResearchState(
                sources_gathered=sources_gathered,
                search_query=[result.query],
                web_research_result=[result.answer] if result.answer else [],
            )

        except Exception as e:
            logger.error(f"Error in real web search: {e}")
            raise RuntimeError(f"Web search failed: {str(e)}")

    def _get_research_topic(self, messages: List[AnyMessage]) -> str:
        """
        Get the research topic from the messages.

        Args:
            messages: List of messages from the conversation

        Returns:
            Formatted research topic string
        """
        # Check if request has a history and combine the messages into a single string
        if len(messages) == 1:
            research_topic = messages[-1].content
        else:
            research_topic = ""
            for message in messages:
                if isinstance(message, HumanMessage):
                    research_topic += f"User: {message.content}\n"
                elif isinstance(message, AIMessage):
                    research_topic += f"Assistant: {message.content}\n"
        return research_topic

    def _reflection_node(self, state: ResearchState, config: RunnableConfig) -> ReflectionState:
        """Reflect on research results and identify gaps using instructor structured output."""
        # Increment research loop count
        research_loop_count = state.get("research_loop_count", 0) + 1

        # Get configuration from RunnableConfig or use instance configuration
        Configuration.from_runnable_config(config) if config else self.configuration

        # Format the prompt
        current_date = get_current_date()
        research_topic = self.preprocess_research_topic(state["messages"])
        summaries = "\n\n---\n\n".join(state.get("web_research_result", []))

        formatted_prompt = self.prompts["reflection"].format(
            current_date=current_date,
            research_topic=research_topic,
            summaries=summaries,
        )

        try:
            # Use instructor for reflection and evaluation with structured output
            result: Reflection = self.llm_client.structured_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=Reflection,
                temperature=0.5,
                max_tokens=1000,
            )

            reflection_results = {
                "is_sufficient": result.is_sufficient,
                "knowledge_gap": result.knowledge_gap,
                "follow_up_queries": result.follow_up_queries,
                "research_loop_count": research_loop_count,
                "number_of_ran_queries": len(state.get("search_query", [])),
            }

            return reflection_results

        except Exception as e:
            logger.error(f"Error in structured reflection: {e}")
            # Use customizable fallback logic
            fallback_results = self.customize_reflection_fallback(state, research_loop_count)

            return {
                **fallback_results,
                "research_loop_count": research_loop_count,
                "number_of_ran_queries": len(state.get("search_query", [])),
            }

    def _evaluate_research(self, state: ReflectionState, config: RunnableConfig):
        """Evaluate research and decide next step."""
        # Get configuration from RunnableConfig or use instance configuration
        runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

        if state["is_sufficient"] or state["research_loop_count"] >= runnable_config.max_research_loops:
            return "finalize_answer"
        else:
            return [
                Send(
                    "web_research",
                    {
                        "search_query": follow_up_query,
                        "id": str(state["number_of_ran_queries"] + int(idx)),
                    },
                )
                for idx, follow_up_query in enumerate(state["follow_up_queries"])
            ]

    def _finalize_answer_node(self, state: ResearchState, config: RunnableConfig):
        """Finalize the research answer with advanced formatting and citations."""
        # Get configuration from RunnableConfig or use instance configuration
        Configuration.from_runnable_config(config) if config else self.configuration

        current_date = get_current_date()
        research_topic = self.preprocess_research_topic(state["messages"])
        summaries = "\n---\n\n".join(state.get("web_research_result", []))

        formatted_prompt = self.prompts["answer"].format(
            current_date=current_date,
            research_topic=research_topic,
            summaries=summaries,
        )

        # Generate final answer using LLM
        messages = [{"role": "user", "content": formatted_prompt}]

        final_answer = self.llm_client.chat_completion(messages=messages, temperature=0.3, max_tokens=2000)

        # Process sources and format summary (can be customized by subclasses)
        sources = state.get("sources_gathered", [])
        formatted_summary = self.format_final_answer(final_answer, sources)

        return {
            "messages": [AIMessage(content=formatted_summary)],
            "sources_gathered": sources,
        }

    def research(
        self,
        user_message: str,
        context: Dict[str, Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> ResearchOutput:
        """
        Research a topic and return structured results.

        Args:
            user_message: User's research request
            context: Additional context for research
            config: Optional RunnableConfig for runtime configuration

        Returns:
            ResearchOutput with content and sources
        """
        try:
            # Initialize state (can be customized by subclasses)
            initial_state = self.customize_initial_state(user_message, context or {})

            # Run the research graph with optional runtime configuration
            if config:
                result = self.graph.invoke(initial_state, config=config)
            else:
                result = self.graph.invoke(initial_state)

            # Extract the final AI message
            final_message = None
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    final_message = message.content
                    break

            return ResearchOutput(
                content=final_message or "Research completed",
                sources=result.get("sources_gathered", []),
                summary=f"Research completed for topic",
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error in research: {e}")
            raise RuntimeError(f"Research failed: {str(e)}")
