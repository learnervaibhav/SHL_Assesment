"""
Gemini LLM Client for agent reasoning
"""

import json
import os
from typing import Optional, Dict, Any, List

from google import genai
from google.genai import types


class GeminiClient:
    """Wrapper for Google Gemini API"""

    ALLOWED_ACTIONS = {"clarify", "recommend", "refine", "compare", "refuse"}

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key (from environment if not provided)
            model: Gemini model version
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not provided or set in environment")

        self.model = model
        self.client = genai.Client(api_key=self.api_key)

        print(f"Gemini client initialized ({model})")

    def decide_action(
        self,
        conversation_history: List[Dict[str, str]],
        extracted_context: Dict[str, Any],
        catalog_summary: str,
        turn_count: int,
    ) -> Dict[str, Any]:
        """
        Use Gemini to decide next action: clarify, recommend, refine, compare, or refuse.

        Args:
            conversation_history: List of {role, content} dicts
            extracted_context: Extracted intent, constraints, prior recommendations
            catalog_summary: Summary of catalog structure
            turn_count: Current turn count

        Returns:
            Dict with keys:
            - action
            - response
            - assessment_names
        """
        prompt = self._build_decision_prompt(
            conversation_history=conversation_history,
            extracted_context=extracted_context,
            catalog_summary=catalog_summary,
            turn_count=turn_count,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )

            result_text = (response.text or "").strip()
            parsed = self._safe_json_loads(result_text)

            if not isinstance(parsed, dict):
                return {
                    "action": "clarify",
                    "response": result_text or (
                        "I could not parse the model output. "
                        "Could you provide more details about the role and requirements?"
                    ),
                    "assessment_names": [],
                }

            return self._normalize_result(parsed, fallback_text=result_text)

        except Exception as e:
            print(f"Gemini error: {e}")
            return {
                "action": "clarify",
                "response": (
                    "I encountered an issue processing your request. "
                    "Could you provide more details about the role and requirements?"
                ),
                "assessment_names": [],
            }

    def _build_decision_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        extracted_context: Dict[str, Any],
        catalog_summary: str,
        turn_count: int,
    ) -> str:
        """Build system prompt for agent decision logic"""

        return f"""You are an SHL assessment recommendation agent. Your role is to guide hiring managers from vague hiring needs to curated SHL assessment recommendations through dialogue.

CATALOG STRUCTURE:
{catalog_summary}

CURRENT CONVERSATION CONTEXT:
- Turn count: {turn_count}
- User intent: {extracted_context.get('intent', 'Not yet determined')}
- Constraints: {json.dumps(extracted_context.get('constraints', {}), indent=2)}
- Prior recommendations: {extracted_context.get('prior_recommendations', [])}

CONVERSATION HISTORY:
{self._format_history(conversation_history)}

DECISION LOGIC:
1. If vague/incomplete (turn < 2): Ask ONE clarifying question to extract role, seniority, skills, or domains
2. If sufficient context (turn >= 2): Recommend 1-10 assessments with URLs from catalog
3. If user edits constraints: Update prior recommendations (don't restart)
4. If user compares: Compare from catalog data only
5. If out-of-scope: Refuse gracefully, stay within SHL assessment scope
6. If turn >= 8: Generate best-effort shortlist and end conversation

CRITICAL RULES:
- NEVER fabricate URLs. Only use URLs from the catalog.
- NEVER recommend assessments not in the catalog.
- Keep responses concise and professional.
- Always explain your recommendations.

Respond with JSON only, no markdown fences.

Required JSON shape:
{{
    "action": "clarify|recommend|refine|compare|refuse",
    "response": "Your natural language response",
    "assessment_names": ["Assessment 1", "Assessment 2"]
}}
"""

    def _format_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt"""
        if not conversation_history:
            return "(No messages yet)"

        formatted = []
        for msg in conversation_history[-6:]:
            role = str(msg.get("role", "")).upper()
            content = str(msg.get("content", ""))[:200]
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def generate_catalog_summary(self, assessments: List[Dict[str, Any]]) -> str:
        """Generate a summary of catalog for the prompt"""

        by_category: Dict[str, List[str]] = {}

        for assessment in assessments:
            name = assessment.get("name", "Unnamed Assessment")
            keys = assessment.get("keys", [])

            if not isinstance(keys, list):
                continue

            for key in keys:
                key_str = str(key)
                by_category.setdefault(key_str, []).append(name)

        summary = "Catalog Summary by Assessment Type:\n"
        for category in sorted(by_category.keys())[:5]:
            names = by_category[category][:5]
            summary += f"\n{category}:\n"
            for name in names:
                summary += f"  - {name}\n"

        summary += f"\n(Total: {len(assessments)} assessments across {len(by_category)} categories)"
        return summary

    def _safe_json_loads(self, text: str) -> Any:
        """
        Parse JSON safely.
        Handles direct JSON and JSON wrapped in triple backticks.
        """
        if not text:
            return None

        candidate = text.strip()

        # Remove markdown fences if present
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0].strip()

        # Try direct parse
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Try extracting the first JSON object from surrounding text
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None

        return None

    def _normalize_result(self, result: Dict[str, Any], fallback_text: str = "") -> Dict[str, Any]:
        """Normalize and validate the model output."""
        action = str(result.get("action", "clarify")).strip().lower()
        if action not in self.ALLOWED_ACTIONS:
            action = "clarify"

        response = str(result.get("response", "")).strip()
        if not response:
            response = fallback_text or (
                "I could not produce a valid response. "
                "Could you provide more details about the role and requirements?"
            )

        assessment_names = result.get("assessment_names", [])
        if not isinstance(assessment_names, list):
            assessment_names = []

        assessment_names = [
            str(name).strip()
            for name in assessment_names
            if str(name).strip()
        ]

        # Only some actions should return names
        if action not in {"recommend", "refine", "compare"}:
            assessment_names = []

        return {
            "action": action,
            "response": response,
            "assessment_names": assessment_names,
        }


if __name__ == "__main__":
    try:
        client = GeminiClient()

        test_history = [
            {"role": "user", "content": "We need assessments for senior leadership roles"}
        ]

        test_context = {
            "intent": "leadership hiring",
            "constraints": {"seniority": "senior", "role_type": "executive"},
            "prior_recommendations": [],
        }

        summary = (
            "Leadership assessments: OPQ32r, Leadership Report\n"
            "Personality assessments: OPQ32r, DSI"
        )

        result = client.decide_action(test_history, test_context, summary, 1)
        print("\nDecision result:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}")