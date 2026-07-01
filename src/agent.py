"""
LangGraph-based Agent for orchestrating the chat flow
Implements the state machine: clarify -> retrieve -> rank -> format -> output
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import operator

# Load environment variables for standalone execution
load_dotenv(Path(__file__).parent.parent / ".env")

from src.context import ContextExtractor, ExtractedContext
from src.llm_client import GeminiClient
from src.retriever import HybridRetriever
from src.catalog_loader import CatalogLoader
from src.models import Message, Recommendation, ChatResponse


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """State passed through agent workflow"""
    messages: List[Dict[str, str]]
    context: Optional[ExtractedContext]
    intent: str
    constraints: Dict[str, Any]
    turn_count: int
    llm_decision: Optional[Dict[str, Any]]
    retrieved_assessments: List[Dict[str, Any]]
    recommendations: List[Recommendation]
    response_text: str
    end_of_conversation: bool


# ============================================================================
# Agent Node Functions
# ============================================================================

class SHLRecommendationAgent:
    """LangGraph-based agent for SHL assessment recommendations"""

    # Maps assessment category keys to single-letter test type codes
    KEY_TO_CODE = {
        "Ability & Aptitude": "A",
        "Assessment Exercises": "A",
        "Biodata & Situational Judgment": "B",
        "Competencies": "C",
        "Development & 360": "D",
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Simulations": "S",
    }

    def __init__(self):
        """Initialize agent with all required components"""
        self.context_extractor = ContextExtractor()
        # Lazy-loaded clients and components (initialize to None)
        self.llm_client = None
        self.retriever = None
        self.catalog_loader = CatalogLoader()
        self.graph = None

        # Build the agent graph
        self._build_graph()
    
    def _init_llm(self):
        """Lazy-load LLM client"""
        if self.llm_client is None:
            self.llm_client = GeminiClient()
    
    def _init_retriever(self):
        """Lazy-load retriever"""
        if self.retriever is None:
            self.retriever = HybridRetriever()

    def _get_test_type_code(self, assessment: Dict) -> str:
        """Convert assessment keys to comma-separated test type codes (e.g. 'P,C')"""
        keys = assessment.get("keys", [])
        codes = []
        seen = set()
        for key in keys:
            code = self.KEY_TO_CODE.get(key, key[0].upper() if key else "?")
            if code not in seen:
                codes.append(code)
                seen.add(code)
        return ",".join(codes) if codes else "N"

    def _build_graph(self):
        """Build LangGraph state machine"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("extract_context", self.node_extract_context)
        workflow.add_node("check_turn_limit", self.node_check_turn_limit)
        workflow.add_node("llm_decision", self.node_llm_decision)
        workflow.add_node("retrieve_assessments", self.node_retrieve_assessments)
        workflow.add_node("rank_results", self.node_rank_results)
        workflow.add_node("format_response", self.node_format_response)
        workflow.add_node("validate_output", self.node_validate_output)
        
        # Define edges (state transitions)
        workflow.set_entry_point("extract_context")
        
        workflow.add_edge("extract_context", "check_turn_limit")
        workflow.add_conditional_edges(
            "check_turn_limit",
            self.should_terminate,
            {
                True: "retrieve_assessments",  # Hit turn limit, still retrieve before formatting
                False: "llm_decision"
            }
        )
        
        workflow.add_edge("llm_decision", "retrieve_assessments")
        workflow.add_conditional_edges(
            "retrieve_assessments",
            self.should_rank,
            {
                True: "rank_results",
                False: "format_response"
            }
        )
        
        workflow.add_edge("rank_results", "validate_output")
        workflow.add_edge("validate_output", "format_response")
        workflow.add_edge("format_response", END)
        
        self.graph = workflow.compile()
    
    def run(self, messages: List[Message]) -> ChatResponse:
        """
        Run the agent on a conversation
        
        Args:
            messages: List of Message objects
        
        Returns:
            ChatResponse
        """
        
        # Convert Message objects to dicts
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]
        
        # Initialize state
        initial_state: AgentState = {
            "messages": message_dicts,
            "context": None,
            "intent": "",
            "constraints": {},
            "turn_count": sum(1 for m in message_dicts if m["role"] == "user"),
            "llm_decision": None,
            "retrieved_assessments": [],
            "recommendations": [],
            "response_text": "",
            "end_of_conversation": False,
        }
        
        # Run graph
        result = self.graph.invoke(initial_state)
        
        # Build response
        return ChatResponse(
            reply=result["response_text"],
            recommendations=result["recommendations"],
            end_of_conversation=result["end_of_conversation"],
        )
    
    # ========================================================================
    # Node Functions
    # ========================================================================
    
    def node_extract_context(self, state: AgentState) -> AgentState:
        """Extract intent, constraints, turn count from conversation"""
        
        context = self.context_extractor.extract(state["messages"])
        
        print(f"   [Context] Intent: {context.intent[:60]}...")
        print(f"   [Context] Constraints: {context.constraints}")
        print(f"   [Context] Turn count: {context.turn_count}")
        
        state.update({
            "context": context,
            "intent": context.intent,
            "constraints": context.constraints,
            "turn_count": context.turn_count,
        })
        
        return state
    
    def node_check_turn_limit(self, state: AgentState) -> AgentState:
        """Check if we've hit the turn limit (8 user turns)"""
        if state["turn_count"] >= 8:
            # Set a default LLM decision for best-effort recommendation
            state["llm_decision"] = {
                "action": "recommend",
                "response": "We've reached the maximum number of conversation turns. Based on our discussion, here are the most relevant assessments for your needs:",
                "assessment_names": [],
            }
        return state

    def should_terminate(self, state: AgentState) -> bool:
        """Decide whether to terminate conversation"""
        return state["turn_count"] >= 8
    
    def node_llm_decision(self, state: AgentState) -> AgentState:
        """Use Gemini to decide next action"""
        
        self._init_llm()
        
        # Generate catalog summary for prompt
        all_assessments = self.catalog_loader.get_all()
        catalog_summary = self.llm_client.generate_catalog_summary(all_assessments)
        
        # Get LLM decision
        decision = self.llm_client.decide_action(
            state["messages"],
            {
                "intent": state["intent"],
                "constraints": state["constraints"],
                "prior_recommendations": state.get("prior_recommendations", [])
            },
            catalog_summary,
            state["turn_count"]
        )
        
        state["llm_decision"] = decision
        
        # DEBUG: Log the decision
        print(f"  [LLM] action='{decision.get('action')}', response='{decision.get('response', '')[:100]}...'")
        
        return state
    
    def should_rank(self, state: AgentState) -> bool:
        """Check if we need to rank assessments (compare produces no recommendations)"""
        decision = state.get("llm_decision") or {}
        action = decision.get("action", "")
        return action in ["recommend", "refine"]
    
    def node_retrieve_assessments(self, state: AgentState) -> AgentState:
        """Retrieve assessments based on context, with graceful fallback"""
        
        decision = state.get("llm_decision", {})
        action = decision.get("action", "clarify")
        
        # DEBUG: Log the action
        print(f" Retrieve Node: action='{action}'")
        
        if action not in ["recommend", "refine"]:
            print(f"  Skipping retrieval for action '{action}'")
            state["retrieved_assessments"] = []
            return state
        
        self._init_retriever()
        
        # Perform hybrid retrieval
        query = state["intent"]
        results = self.retriever.search(
            query,
            constraints=state["constraints"],
            limit=20,
            keyword_weight=0.5,
            semantic_weight=0.5,
        )
        
        # Fallback: if constraints produced fewer than 3 results, retry without
        # constraints to backfill with broader matches
        if len(results) < 3 and state.get("constraints"):
            print(f"   [Retriever] Only {len(results)} results with constraints — retrying without constraints")
            unconstrained_results = self.retriever.search(
                query,
                constraints=None,
                limit=20,
                keyword_weight=0.5,
                semantic_weight=0.5,
            )
            # Append unconstrained results that aren't already in the list
            existing_ids = {r["id"] for r in results}
            for r in unconstrained_results:
                if r["id"] not in existing_ids:
                    results.append(r)
                    existing_ids.add(r["id"])
            print(f"   [Retriever] After fallback: {len(results)} total results")
        
        state["retrieved_assessments"] = results
        
        return state
    
    def node_rank_results(self, state: AgentState) -> AgentState:
        """Rank and limit results to top 10"""
        
        # For now, trust the retriever's ranking
        # In the future, could add re-ranking based on conversation context
        top_results = state["retrieved_assessments"][:10]
        state["retrieved_assessments"] = top_results
        
        return state
    
    def node_validate_output(self, state: AgentState) -> AgentState:
        """Validate that all URLs are from catalog"""
        
        # Check all retrieved assessments have valid URLs
        for assessment in state["retrieved_assessments"]:
            url = assessment.get("url", "")
            if not self.catalog_loader.validate_url(url):
                print(f" Invalid URL detected: {url}")
                # Remove invalid URLs
                state["retrieved_assessments"] = [
                    a for a in state["retrieved_assessments"]
                    if self.catalog_loader.validate_url(a.get("url", ""))
                ]
                break
        
        return state
    
    def node_format_response(self, state: AgentState) -> AgentState:
        """Format final response"""

        decision = state.get("llm_decision") or {}
        action = decision.get("action", "clarify")
        llm_names = decision.get("assessment_names", [])

        # DEBUG: Log what the LLM decided
        print(f"  [Format] action='{action}', retrieved={len(state['retrieved_assessments'])}, llm_names={len(llm_names)}")

        # Build response text
        response_text = decision.get("response", "")

        # Build recommendations based on action
        recommendations = []
        end_of_conversation = False

        if action in ["recommend", "refine"]:
            # First, try to resolve LLM-selected assessment names from catalog
            resolved_from_llm = []
            for name in llm_names:
                assessment = self.catalog_loader.get_by_name(name)
                if assessment:
                    resolved_from_llm.append(assessment)

            # Use LLM selections if available, otherwise fall back to retriever results
            source_assessments = resolved_from_llm if resolved_from_llm else state["retrieved_assessments"]

            for assessment in source_assessments[:10]:
                recommendations.append(
                    Recommendation(
                        name=assessment["name"],
                        url=assessment["url"],
                        test_type=self._get_test_type_code(assessment),
                    )
                )

        # compare, clarify: no recommendations (response text only)
        # refuse: no recommendations, end conversation

        if action == "refuse":
            end_of_conversation = True

        # Turn limit forces end of conversation
        if state["turn_count"] >= 8:
            end_of_conversation = True

        # Provide default response text if LLM returned empty
        if not response_text:
            if action == "clarify":
                response_text = "Could you share the role, seniority level, and assessment types you're looking for?"
            elif action == "refuse":
                response_text = "I can only help with SHL assessment recommendations. Let me know if you have a hiring need I can assist with."
            elif recommendations:
                response_text = "Here are the most relevant SHL assessments for your requirements:"

        state.update({
            "response_text": response_text,
            "recommendations": recommendations,
            "end_of_conversation": end_of_conversation,
        })

        return state


if __name__ == "__main__":
    # Test agent
    agent = SHLRecommendationAgent()
    
    # Test conversation
    test_messages = [
        Message(role="user", content="We need assessments for senior Java developers"),
    ]
    
    response = agent.run(test_messages)
    print(f"\nAgent response:")
    print(f"  Reply: {response.reply[:100]}...")
    print(f"  Recommendations: {len(response.recommendations)}")
    print(f"  End: {response.end_of_conversation}")
