"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    """Single message in conversation history"""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class Recommendation(BaseModel):
    """Assessment recommendation"""
    name: str
    url: str
    test_type: str


class ChatRequest(BaseModel):
    """Request payload for /chat endpoint"""
    messages: List[Message] = Field(..., min_items=1)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response payload for /chat endpoint"""
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for /health endpoint"""
    status: str
    timestamp: Optional[str] = None
    database: Optional[str] = None
    agent: Optional[str] = None
