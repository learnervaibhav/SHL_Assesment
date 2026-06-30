# SHL Assessment Recommender

Production Deployment: https://shl-assesment-3bxd.onrender.com/docs

---

## Overview

SHL Assessment Recommender is a multi-turn conversational agent that helps hiring managers select appropriate SHL assessments based on job requirements, candidate profiles, and organizational needs. The system uses advanced retrieval techniques and generative AI to provide personalized assessment recommendations from a catalog of 377+ SHL assessments.

## Key Features

- Multi-turn conversational interface with 8 user-turn conversation limit
- Hybrid retrieval combining keyword search and semantic similarity (FAISS)
- Intelligent constraint filtering (job levels, test types, languages, assessment categories)
- LangGraph-based state machine orchestration with deterministic workflow
- Schema-compliant responses with URL validation
- Database logging for conversation history and analytics
- Sub-30 second response latency with optimized prewarming

## Architecture

### Core Components

1. **Agent (LangGraph State Machine)**
   - 7 nodes: extract_context, check_turn_limit, llm_decision, retrieve_assessments, rank_results, validate_output, format_response
   - Enforces 5 action types: clarify, recommend, refine, compare, refuse
   - Manages conversation state and turn counting

2. **Retrieval System**
   - Keyword-based search (inverted index over assessment metadata)
   - Semantic search (FAISS with sentence-transformers all-MiniLM-L6-v2)
   - Hybrid scoring: 50% keyword + 50% semantic weight
   - Query caching for frequent requests (50 query limit)

3. **LLM Integration**
   - Google Gemini 2.5 Flash as decision-making backbone
   - JSON-enforced output format via response_mime_type
   - Catalog summary provided in prompts for grounding
   - Priority rule: direct recommendations when test types specified

4. **Database Layer**
   - SQLAlchemy ORM models for PostgreSQL/SQLite
   - Conversation logging with message history and recommendations
   - Access logs for monitoring and analytics

### Design Decisions

**Why LangGraph over Chain of Thought:**
- Deterministic control flow prevents unpredictable outputs
- Explicit state transitions enforce constraint compliance
- Easier to validate schema and validate URL correctness

**Why Hybrid Retrieval:**
- Keyword search catches exact test type and role matches
- Semantic search handles user intent and synonyms
- Balanced weighting (50/50) provides reliability across domain queries

**Why Lazy Component Loading:**
- Embedding model loads only when semantic search is needed
- Prewarming focuses on assessments index (FAISS + keyword indices)
- Reduces cold-start latency while maintaining sub-2s first request penalty

**Why Lenient Constraint Filtering:**
- 377 assessments across 8 assessment types (many language-agnostic)
- Only apply language filter if assessment explicitly specifies languages
- Prevents over-filtering and zero-result scenarios

## Test Type Mapping

The system maps user-provided test type keywords to assessment catalog keys:

- Cognitive ability, numerical, verbal, reasoning -> Ability & Aptitude (32 assessments)
- Personality, behavioral measures -> Personality & Behavior (66 assessments)
- Situational judgement, SJT, scenario -> Biodata & Situational Judgment (17 assessments)
- Leadership, management potential -> Competencies, Assessment Exercises
- Skills tests -> Knowledge & Skills (240 assessments)

## Conversation Flow

### Clarify Action
Triggered when user request lacks specific test type or role information.
Agent asks targeted clarifying questions to identify hiring needs.

### Recommend Action
Triggered when constraints include test types or clear job requirements.
Agent returns 10 most relevant assessments from catalog matching criteria.

### Refine Action
Triggered when user provides feedback on previous recommendations.
Agent adjusts filters and returns updated assessment list.

### Compare Action
Triggered when user asks for comparison between assessments.
Agent provides structured comparison of selected assessments.

### Refuse Action
Triggered when request falls outside assessment recommendation scope.
Agent explains scope limitations and redirects to valid use cases.

## Constraint Filters

Constraints extracted from conversation include:

- job_levels: Entry-Level, Manager, Director, Executive, etc.
- seniority: entry, mid, senior, executive
- test_types: cognitive, personality, situational_judgement, leadership, skills
- languages: English, Spanish, French, German, Chinese, etc.
- keys: Assessment categories from catalog
- industries: Finance, Technology, Healthcare, Retail, Manufacturing, Consulting, Operations
- roles: Leadership, Technical, Sales, Customer Service, Finance, Operational

## API Endpoints

### Health Check
GET /health
Returns service status, database connection, and agent readiness.

### Chat Endpoint
POST /chat
Request body:
```
{
  "messages": [
    {
      "role": "user",
      "content": "We need cognitive and personality assessments for entry-level graduates"
    }
  ],
  "conversation_id": "optional-id-string"
}
```

Response:
```
{
  "reply": "Agent response text",
  "recommendations": [
    {
      "name": "Assessment Name",
      "url": "https://www.shl.com/products/...",
      "test_type": "A|B|C|D|K|P|S (comma-separated for multi-category)"
    }
  ],
  "end_of_conversation": false,
  "conversation_id": "conversation-uuid"
}
```

## Setup & Deployment

### Local Development

1. Clone repository and navigate to project directory
2. Create virtual environment: python -m venv .venv
3. Activate: .\.venv\Scripts\Activate.ps1
4. Install dependencies: pip install -r requirements.txt
5. Create .env file with GOOGLE_API_KEY and DATABASE_URL
6. Start server: fastapi dev main.py
7. Access API docs: http://127.0.0.1:8000/docs

### Environment Variables

- GOOGLE_API_KEY: Google Generative AI API key
- GOOGLE_MODEL_NAME: Model to use (default: gemini-2.5-flash)
- DATABASE_URL: PostgreSQL connection string or SQLite file path
- LANGSMITH_TRACING: Set to true for LangChain observability (optional)

### Docker Deployment

Build image: docker build -t shl-assessment .
Run container: docker run -p 8000:8000 shl-assessment

### Production Deployment (Render)

1. Push changes to main branch
2. Redeploy on Render platform
3. Monitor logs for successful prewarming and startup

## Data Files

The system requires the following data files in data/ directory:

- assessments.json: Complete assessment catalog (377 assessments with metadata)
- assessments_index.faiss: FAISS vector index for semantic search
- id_mapping.json: Maps FAISS indices to assessment IDs
- keyword_index.json: Inverted index for keyword-based search
- category_index.json: Assessment category mapping

## Known Limitations

- Conversation limited to 8 turns to manage token consumption
- Semantic search requires embedding model load on first request (2s penalty)
- Language constraints are best-effort matching (partial matching, case-insensitive)
- Assessment recommendations are deterministic based on catalog; new assessments require catalog update

## Testing

Run sample conversation:
```
POST /chat
{
  "messages": [{
    "role": "user",
    "content": "We run a graduate trainee scheme. Need cognitive, personality, and situational judgement tests for entry-level candidates. English-language. Give 5 recommendations."
  }]
}
```

## Project Structure

```
SHL_Assessment/
├── main.py                          FastAPI application, endpoints, and startup
│
├── src/                             Core application modules
│   ├── agent.py                     LangGraph state machine orchestration (7 nodes)
│   ├── llm_client.py                Google Gemini integration and prompt design
│   ├── retriever.py                 Hybrid keyword + semantic search (FAISS)
│   ├── context.py                   Extract constraints from conversation
│   ├── database.py                  SQLAlchemy ORM models (PostgreSQL/SQLite)
│   ├── models.py                    Pydantic schemas (ChatRequest, ChatResponse)
│   ├── catalog_loader.py            SHL assessment catalog management and validation
│   └── embeddings.py                Sentence-transformers wrapper
│
├── data/                            Assessment catalog and search indices
│   ├── assessments.json             Complete 377 assessment catalog
│   ├── assessments_index.faiss      FAISS vector index for semantic search
│   ├── id_mapping.json              Maps FAISS indices to assessment IDs
│   ├── keyword_index.json           Inverted index for keyword search
│   ├── category_index.json          Assessment category mappings
│   ├── embeddings.json              Pre-computed embeddings
│   └── README.md                    Data file documentation
│
├── GenAI_SampleConversations/       Sample multi-turn conversation examples
│   ├── C1.md through C10.md         10 example conversation flows
│   └── README.md                    Conversation documentation
│
├── requirements.txt                 Python dependencies (FastAPI, LangGraph, etc.)
├── Dockerfile                       Docker container configuration
├── docker-compose.yml               Multi-container orchestration
├── .env                             Environment variables (API keys, DB URL)
├── .gitignore                       Git ignore patterns
├── README.md                        This file
├── APPROACH.md                      Design and evaluation documentation
└── backup.sql                       Database schema backup
```

## Support & Feedback

For issues or feature requests, refer to project documentation or contact me
https://vaibhavbaranwal.me.

---

