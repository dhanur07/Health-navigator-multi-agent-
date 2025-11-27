# Health-navigator-multi-agent(ADK + VertexAI)
A multi-agent health navigator that provides safe medical information, misinformation checking, and travel health advice using Google’s ADK


---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Flow](#Flow)
- [Why Agents?](#why-agents)
- [What I Built (Architecture)](#what-i-built-architecture)
- [Agent & Tool Flow](#agent--tool-flow)
- [Project Structure](#project-structure)
- [Configuration & API Keys](#configuration--api-keys)
- [Testing the Deployed Agent](#testing-the-deployed-agent)
- [Tech Stack](#tech-stack)


---

## Problem Statement

Finding credible, safe, and context-aware health information online is hard. Search results often mix scientific content with misinformation, unsafe advice, or irrelevant pages. Users still have very basic questions:

- “Is this claim about my condition actually true?”
- “What vaccines do I need for this trip?”
- “What does this medication do?”
- “I have a chronic condition — what should my daily routine look like?”

Most tools answer these in a one-off way, without tying together: verified sources (CDC/WHO), travel advisories, chronic-condition education, or nearby care options. This project builds a single multi-agent system that can:

- Verify health claims against authoritative sources
- Summarize CDC/WHO guidance
- Explain medications safely
- Provide conservative, high-level chronic-condition education
- Tailor travel health advice to destination and risk factors

All while maintaining strict medical safety constraints.

---
## Flow

<img width="2816" height="1536" alt="WorkflowForHN" src="https://github.com/user-attachments/assets/67b63d1c-d6c6-4379-89bb-fe67adea6b19" />

---
## Why Agents?

Health questions are not single-step prompts. They often require:

- Calling external APIs and web tools
- Collecting evidence from multiple sources
- Running sequential and parallel workflows
- Persisting session state (e.g., user location)
- Applying safety rules consistently

A single “flat” model prompt quickly becomes unmanageable.

Agents (via Google’s ADK) are a better fit because they let us:

- Split responsibilities into **specialized agents** (misinformation, travel, chronic, medication)
- Add **tools** to each agent (CDC/WHO search, TuGo travel API, MCP tools, Google Search)
- Orchestrate **Sequential** and **Parallel** workflows
- Maintain **session + memory** across turns
- Centralize safety and routing in a **Router Agent**

The result is a safer, more reliable system that can reason over tools and workflows instead of ad-hoc prompts.

---

## What I Built (Architecture)

At a high level, the system is a **multi-agent health navigator** deployed as a **Vertex AI Agent Engine** using the ADK.

### Top-Level Router

- **`router_agent`** (root agent)
  - Introduces itself
  - Classifies user intent
  - Routes to exactly one of:
    - `misinformation_agent`
    - `travel_workflow_agent`
    - `chronic_workflow_agent`
    - `prescription_explainer_agent`
  - Uses `save_location` / `get_user_location` tools for chronic flows
  - Uses an `after_agent_callback` to auto-save session state to memory

### Misinformation Workflow

- **`misinformation_agent`**
  - Calls:
    - `google_search_cdc_who` via `health_search_tool` (Google Custom Search, restricted to CDC + WHO)
    - `medical_mcp_toolset` (MCP Toolset pointing to a local medical content server)
  - Compares the user’s claim vs. CDC/WHO evidence
  - Returns a verdict + sources + disclaimer

### Travel Workflow

Implemented as a **SequentialAgent**:

1. **`travel_intent_agent`**
   - Collects destination, dates, purpose, risk factors
   - Saves `travel_intent_summary` to state

2. **`travel_parallel_evidence`** (ParallelAgent)
   - **`cdc_who_travel_agent`**
     - Uses `health_search_tool` (CDC/WHO CSE)
   - **`tugo_travel_agent`**
     - Uses `tugo_travel_tool` to call the TuGo Travel Advisory API

3. **`travel_summary_agent`**
   - Reads `{+cdc_who_travel_summary}` and `{+tugo_travel_summary}`
   - Merges evidence, highlights differences, and produces:
     - Combined explanation
     - Before / during / after travel checklist
     - Strong medical disclaimer

### Chronic Condition Workflow

Also a **SequentialAgent**, with location-aware behavior via custom tools:

1. **`chronic_coach_agent`**
   - Explains the condition in simple language
   - Suggests conservative lifestyle routines
   - May use `health_search_tool` + `medical_mcp_toolset`

2. **`hospital_finder_agent`**
   - Uses `google_search_tool` to find real hospitals/clinics near the user
   - Returns a list (name, city, URL)

3. **`chronic_summary_agent`**
   - Combines `{+chronic_plan}` and `{+hospital_suggestions}`
   - Outputs a single message with education + nearby care options + disclaimers

### Prescription Explainer Workflow

- **`prescription_explainer_agent`**
  - Uses `google_search_tool`
  - Explains:
    - What the drug is for
    - Typical indications
    - Common side effects
    - High-level warnings
  - Does **not** prescribe or adjust meds

### Memory & Session

- `InMemorySessionService` – manages sessions by `session_id` / `user_id`
- `InMemoryMemoryService` – can persist session snapshots
- `auto_save_session_to_memory_callback` – runs after each agent call to store state

---

## Agent & Tool Flow
```
User
  ↓
router_agent
  ├──────────────▶ misinformation_agent
  │                   ↓
  │                   ├──▶ google_search_cdc_who (CDC/WHO Search Tool)
  │                   └──▶ medical_mcp_toolset (MCP Medical Tools)
  │
  ├──────────────▶ travel_workflow_agent (SequentialAgent)
  │                   ↓
  │                   ├──▶ travel_intent_agent
  │                   │
  │                   └──▶ travel_parallel_evidence (ParallelAgent)
  │                           ├──▶ cdc_who_travel_agent
  │                           │       └──▶ health_search_tool (CDC/WHO)
  │                           └──▶ tugo_travel_agent
  │                                   └──▶ tugo_travel_tool (TuGo API)
  │
  │                   ↓
  │                   └──▶ travel_summary_agent
  │
  ├──────────────▶ chronic_workflow_agent (SequentialAgent)
  │                   ↓
  │                   ├──▶ chronic_coach_agent
  │                   │       ├──▶ health_search_tool (optional)
  │                   │       └──▶ medical_mcp_toolset (optional)
  │                   │
  │                   ├──▶ hospital_finder_agent
  │                   │       └──▶ google_search_tool
  │                   │
  │                   └──▶ chronic_summary_agent
  │
  └──────────────▶ prescription_explainer_agent
                      ↓
                      └──▶ google_search_tool
```

## Project Structure
```
Session / Memory / Location
---------------------------
User
  ↓
router_agent
  ├──▶ get_location_tool (get_user_location)
  ├──▶ save_location_tool (save_location)
  └──▶ after_agent_callback → auto_save_session_to_memory_callback
                                   ↓
                            InMemorySessionService
                            InMemoryMemoryService

HealthNavigator_AI/
  ├── Health_Agent/
  │   ├── agent.py
  │   ├── __init__.py
  │   ├── .agent_engine_config.json
  │   ├── requirements.txt
  │   └── (optional MCP scripts / configs)
  ├── test_agent.py
  ├── .gitignore
  ├── README.md
  └── .env  (NOT committed)
```

## Configuration & API Keys

All sensitive configuration is passed via environment variables (e.g. in .env).
The key ones:

- PROJECT_ID=your-gcp-project-id
- GOOGLE_CLOUD_LOCATION=us-central1
- GOOGLE_API_KEY=your_gemini_or_google_ai_studio_key
- SEARCH_API_KEY=your_custom_search_api_key
- SEARCH_ENGINE_ID=your_cdc_who_search_engine_id
- TUGO_API_KEY=your_tugo_travel_api_key

### Note:
### CDC/WHO Custom Search Engine
The project uses a Google Programmable Search Engine (CSE) to search only:
- cdc.gov
- who.int


### Steps:
- Go to Google Programmable Search (Custom Search Engine).
- Create a new search engine:
- Sites to search: cdc.gov, who.int
- Copy the Search engine ID → set as:
SEARCH_ENGINE_ID=your_cse_id


### In Google Cloud Console:
- Enable Custom Search JSON API.
- Create an API key for it.
- Set as:
- SEARCH_API_KEY=your_custom_search_json_api_key

### For Tugo API key:
- Sign up for a TuGo developer account (or substitute your own travel advisory API).
- Obtain an API key.
- Set:
- TUGO_API_KEY=your_tugo_api_key

---

## Tech Stack

- Google ADK (Agent Development Kit)
- Vertex AI Agent Engine / Reasoning Engine Service
- Gemini 2.5 Flash / Flash-Lite
- Google Custom Search API (CDC/WHO)
- TuGo Travel Advisory API
- MCP Toolset (medical content server)
- Python 3.11, google-generativeai, google-genai, google-adk, google-cloud-aiplatform, requests, python-dotenv




