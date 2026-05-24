"""
LangGraph-based Agent for orchestrating the chat flow
Implements the state machine: clarify -> retrieve -> rank -> format -> output
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

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
    
    def __init__(self):
        """Initialize agent with all required components"""
        self.context_extractor = ContextExtractor()
        self.llm_client   # Lazy-loaded
        self.retriever    # Lazy-loaded
        self.catalog_loader = CatalogLoader()
        self.graph 
        
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
                True: "format_response",  # Hit turn limit, format best-effort
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
            "turn_count": len(message_dicts),
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
        
        state.update({
            "context": context,
            "intent": context.intent,
            "constraints": context.constraints,
            "turn_count": context.turn_count,
        })
        
        return state
    
    def node_check_turn_limit(self, state: AgentState) -> AgentState:
        """Check if we've hit the turn limit (8 turns)"""
        # Logic handled in conditional edge
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
        print(f"🤖 LLM Response: action='{decision.get('action')}', response='{decision.get('response', '')[:100]}...'")
        
        return state
    
    def should_rank(self, state: AgentState) -> bool:
        """Check if we need to retrieve and rank assessments"""
        decision = state.get("llm_decision", {})
        action = decision.get("action", "")
        return action in ["recommend", "refine", "compare"]
    
    def node_retrieve_assessments(self, state: AgentState) -> AgentState:
        """Retrieve assessments based on context"""
        
        decision = state.get("llm_decision", {})
        action = decision.get("action", "clarify")
        
        # DEBUG: Log the action
        print(f"🔍 Retrieve Node: action='{action}'")
        
        if action not in ["recommend", "refine", "compare"]:
            print(f"⚠️  Skipping retrieval because action is '{action}', not in ['recommend', 'refine', 'compare']")
            state["retrieved_assessments"] = []
            return state
        
        self._init_retriever()
        
        # Perform hybrid retrieval
        query = state["intent"]
        results = self.retriever.search(
            query,
            constraints=state["constraints"],
            limit=20,
            keyword_weight=0.3,
            semantic_weight=0.7,
        )
        
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
                print(f"❌ Invalid URL detected: {url}")
                # Remove invalid URLs
                state["retrieved_assessments"] = [
                    a for a in state["retrieved_assessments"]
                    if self.catalog_loader.validate_url(a.get("url", ""))
                ]
                break
        
        return state
    
    def node_format_response(self, state: AgentState) -> AgentState:
        """Format final response"""
        
        decision = state.get("llm_decision", {})
        action = decision.get("action", "clarify")
        
        # DEBUG: Log what the LLM decided
        print(f"🔍 LLM Decision: action='{action}', retrieved_assessments={len(state['retrieved_assessments'])}")
        
        # Build response text
        response_text = decision.get("response", "")
        
        # Build recommendations based on action
        recommendations = []
        end_of_conversation = False
        
        if action in ["recommend", "refine", "compare"]:
            # Create Recommendation objects
            for idx, assessment in enumerate(state["retrieved_assessments"][:10]):
                recommendations.append(
                    Recommendation(
                        name=assessment["name"],
                        url=assessment["url"],
                        test_type=assessment.get("keys", [""])[0][0] if assessment.get("keys") else "N"
                    )
                )
            
            # Mark as end if this was the final recommendation
            if action == "recommend" and state["turn_count"] >= 7:
                end_of_conversation = True
        
        # Handle turn limit scenarios
        if state["turn_count"] >= 8 and not recommendations:
            # Force best-effort shortlist
            for idx, assessment in enumerate(state["retrieved_assessments"][:5]):
                recommendations.append(
                    Recommendation(
                        name=assessment["name"],
                        url=assessment["url"],
                        test_type=assessment.get("keys", [""])[0][0] if assessment.get("keys") else "N"
                    )
                )
            end_of_conversation = True
        
        state.update({
            "response_text": response_text,
            "recommendations": recommendations,
            "end_of_conversation": end_of_conversation or action == "refuse",
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
    print(f"\n✓ Agent response:")
    print(f"  Reply: {response.reply[:100]}...")
    print(f"  Recommendations: {len(response.recommendations)}")
    print(f"  End: {response.end_of_conversation}")
