"""
Main FastAPI application for SHL Assessment Recommender
Endpoints: GET /health, POST /chat
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import time
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

from src.models import (
    Message, ChatRequest, ChatResponse, HealthResponse
)
from src.database import get_db, init_db, Conversation, MessageRecord, RecommendationRecord, AccessLog
from src.agent import SHLRecommendationAgent

# ============================================================================
# Initialize App
# ============================================================================

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Multi-turn conversational agent for SHL assessment recommendations",
    version="2.0",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager
    Runs once on startup and once on shutdown
    """

    global agent

    # Startup logic
    init_db()

    # Preload agent to avoid cold-start latency
    agent = SHLRecommendationAgent()

    print("Agent and database initialized")

    yield

    # Shutdown logic (optional cleanup)
    print("Application shutdown complete")

# Global agent instance (will be set on startup)
agent: Optional[SHLRecommendationAgent] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint
    
    Returns:
        HealthResponse with status and timestamp
    """
    start_time = time.time()
    
    try:
        # Quick database check
        db.execute(text("SELECT 1"))
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return HealthResponse(
            status="ok",
            timestamp=datetime.now().isoformat(),
            database="connected",
            agent="ready"
        )
    
    except Exception as e:
        print(f" Health check failed: {e}")
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log to database
        try:
            log = AccessLog(
                conversation_id="health-check",
                endpoint="/health",
                status_code=503,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
            db.add(log)
            db.commit()
        except Exception as log_error:
            print(f"Failed to log error: {log_error}")
        
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "Service unavailable",
                "database": "disconnected"
            }
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Chat endpoint for multi-turn conversation
    
    Args:
        request: ChatRequest with messages and optional conversation_id
        db: Database session
    
    Returns:
        ChatResponse with reply, recommendations, end_of_conversation flag
    """
    
    start_time = time.time()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        # Validate request
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="Messages list cannot be empty"
            )
        
        if len(request.messages) > 0:
            if not isinstance(request.messages[0], Message):
                # Convert dicts to Message objects if needed
                request.messages = [
                    Message(**m) if isinstance(m, dict) else m
                    for m in request.messages
                ]
        
        # Load or create conversation record
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            conversation = Conversation(
                id=conversation_id,
                turns=len(request.messages)
            )
            db.add(conversation)
        else:
            conversation.turns = len(request.messages)
            conversation.updated_at = datetime.now()
        
        db.commit()
        
        # Run agent with 28-second timeout (2 seconds buffer for DB operations)
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Agent not initialized"
            )
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(agent.run, request.messages),
                timeout=28.0
            )
        except asyncio.TimeoutError:
            print(f" Agent timeout for {conversation_id} after 28 seconds")
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "Agent processing timeout",
                    "conversation_id": conversation_id
                }
            )
        
        # Store messages in database
        for msg in request.messages:
            if not db.query(MessageRecord).filter(
                MessageRecord.conversation_id == conversation_id,
                MessageRecord.content == msg.content
            ).first():
                message_record = MessageRecord(
                    conversation_id=conversation_id,
                    role=msg.role,
                    content=msg.content
                )
                db.add(message_record)
        
        # Store recommendations
        for idx, rec in enumerate(response.recommendations):
            rec_record = RecommendationRecord(
                conversation_id=conversation_id,
                assessment_name=rec.name,
                assessment_url=rec.url,
                test_type=rec.test_type,
                rank=idx + 1,
                turn_number=len(request.messages)
            )
            db.add(rec_record)
        
        # Update conversation
        conversation.is_completed = response.end_of_conversation
        if response.recommendations:
            conversation.constraints = {}  # Store final context if needed
        
        db.commit()
        
        # Log access
        response_time_ms = int((time.time() - start_time) * 1000)
        access_log = AccessLog(
            conversation_id=conversation_id,
            endpoint="/chat",
            status_code=200,
            response_time_ms=response_time_ms,
        )
        db.add(access_log)
        db.commit()
        
        # Add conversation_id to response
        response.conversation_id = conversation_id
        
        print(f"Chat response for {conversation_id}: {len(response.recommendations)} recommendations, end={response.end_of_conversation}")
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"Chat error: {e}")
        
        # Log error
        response_time_ms = int((time.time() - start_time) * 1000)
        try:
            access_log = AccessLog(
                conversation_id=conversation_id,
                endpoint="/chat",
                status_code=500,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
            db.add(access_log)
            db.commit()
        except Exception as log_error:
            print(f"Failed to log error: {log_error}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "conversation_id": conversation_id
            }
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "path": str(request.url)
        }
    )


# ============================================================================
# Server Info
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
