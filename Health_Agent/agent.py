import os
import asyncio
import uuid
import logging
from google.adk.runners import Runner
import sys
import requests

import google.generativeai as genai
import vertexai
from googleapiclient.discovery import build
from google.genai import types as genai_types
from dotenv import load_dotenv
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)

from google.adk.agents import (
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
)

from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.memory import InMemoryMemoryService
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import GoogleSearchTool

from google.adk.tools import load_memory, preload_memory
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools.tool_context import ToolContext
from vertexai import agent_engines


#imports environment variables from .env file
load_dotenv()  


PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"


vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
    )

# Logging / Observability

logging.basicConfig(
    level=logging.DEBUG,  # change to DEBUG while debugging
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)
logger = logging.getLogger("health_agents")

print("✅ ADK components imported successfully.")


# Auth & Config

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

#search api limits the search to specific domains - cdc.gov and who.int
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY")

#search engine id limits the search to specific domains - cdc.gov and who.int
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
MEDADAPT_PATH = os.path.abspath("../../cdmedadapt-content-server/content_server.py")


TUGO_API_KEY = os.environ.get("TUGO_API_KEY")

if not TUGO_API_KEY:
    logger.warning("TUGO_API_KEY not set. TuGo travel advisory tool will fail without it.")

missing = []
if not GOOGLE_API_KEY: missing.append("GOOGLE_API_KEY")
if not SEARCH_API_KEY: missing.append("SEARCH_API_KEY")
if not SEARCH_ENGINE_ID: missing.append("SEARCH_ENGINE_ID")
if not TUGO_API_KEY: missing.append("TUGO_API_KEY")

if missing:
    logger.warning(f"[ENV WARNING] Missing env vars: {missing}. Continuing with degraded functionality.")

retry_config = genai_types.HttpRetryOptions(
    attempts=1,
    exp_base=1,
    initial_delay=0,
    http_status_codes=[429, 500, 503, 504],
)

#  LLM config
GEMINI_FLASH = Gemini(model="gemini-2.5-flash", retry_options=retry_config)

GEMINI_LITE = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# Built-in code executor
code_executor = BuiltInCodeExecutor()

# Memory/session services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()



# Helper function – extract text from events

def extract_text(event):
    #  Check if event or content is missing
    if not event:
        return ""
    
    content = getattr(event, "content", None)
    if content is None:
        return ""
        
    # Handle simple string content
    if isinstance(content, str):
        return content
        
    # Handle complex Content objects
    if hasattr(content, "parts") and content.parts is not None:
        return "".join(
            (p.text or "") for p in content.parts 
            if hasattr(p, "text") and p.text is not None
        )
        
    #  Fallback for unexpected types
    return str(content)



# Built-in Google Search for generic web lookup 
google_search_tool = GoogleSearchTool()

#--------------------------------------
#Misinformation agent functionality
#--------------------------------------

#Custom CDC/WHO Search Tool (uses Google Custom Search to limit to cdc.gov and who.int)
def google_search_cdc_who(query: str) -> str:
    """
    Custom tool: searches CDC / WHO content using Custom Search.
    Only used for health/medical guidance and misinformation checks.
    """
    logger.info(f"[cdc_who_search] query={query}")
    try:
        service = build("customsearch", "v1", developerKey=SEARCH_API_KEY)
        result = service.cse().list(
            q=query,
            cx=SEARCH_ENGINE_ID,
            num=5,
        ).execute()

        if "items" not in result:
            return "No relevant information found on cdc.gov or who.int."

        snippets = []
        for item in result["items"]:
            snippets.append(
                f"Source URL: {item['link']}\n"
                f"Title: {item['title']}\n"
                f"Snippet: {item['snippet']}\n"
            )
        return "\n---\n".join(snippets)

    except Exception as e:
        logger.exception("Error during CDC/WHO search")
        return f"Error during search: {e}"

health_search_tool = FunctionTool(google_search_cdc_who)

# MCP toolset – connect to an MCP server that exposes medical/knowledge tools
medical_mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python3",        # or "python3" depending on your env
            args=[MEDADAPT_PATH],
            env={
                # Propagate your environment – especially if you rely on .env
                **os.environ,
                # Optionally override/add keys:
                # "NCBI_API_KEY": os.getenv("NCBI_API_KEY", ""),
            },
             timeout=120,   
        ),
    ),
)

# Misinformation / fact-checking agent
misinformation_agent = LlmAgent(
    model=GEMINI_LITE,
    name="misinformation_agent",
    instruction="""
You are a cautious public health misinformation checker.

Your job:
- Verify health claims strictly against CDC (cdc.gov) and WHO (who.int) content.
- ALWAYS call the `google_search_cdc_who` tool to retrieve evidence first.
- Compare multiple sources if available.
- Clearly state whether the claim seems CONSISTENT or INCONSISTENT with official guidance.
- Always include URLs in your answer.
- Always say: "This is not medical advice. Talk to a licensed clinician for personal decisions."
    """,
    tools=[health_search_tool, medical_mcp_toolset],
)
misinfo_tool = AgentTool(agent=misinformation_agent)

#--------------------------------------
#  Travel & Vaccine Functionality
#--------------------------------------


travel_intent_agent = LlmAgent(
    model=GEMINI_LITE,
    name="travel_intent_agent",
    instruction="""
You are a travel health intake agent.

1. Ask clarifying questions (destination country/city, travel dates, trip length, purpose, age, pregnancy, chronic conditions).
2. Summarize and store normalized info in state under keys:
   - temp:travel_destination
   - temp:travel_dates
   - temp:travel_purpose
   - temp:travel_risk_factors
Don't give any medical advice yet.
    """,
    output_key="travel_intent_summary",
)

#CDC and WHO website checker agent
cdc_who_travel_agent = LlmAgent(
    model=GEMINI_LITE,
    name="cdc_who_travel_agent",
    instruction="""
You are a CDC and WHO travel guidance summarizer.
Use tools to retrieve official CDC and WHO travel pages for the destination and time window.

Focus on:
- Required or recommended vaccines
- Malaria / mosquito-borne risk
- Food/water precautions
- Outbreak alerts

Never diagnose or prescribe. Only summarize official CDC guidance.
Always cite URLs.
    """,
    tools=[health_search_tool],
    output_key="cdc_who_travel_summary",
)

# TuGo Travel Advisory function
def tugo_travel_advisory(country: str) -> dict:
    """
    Fetches travel advisory + health/safety info for a country from TuGo's Travel Advisory API.
    """
    if not country:
        return {"error": "country is required"}

    if not TUGO_API_KEY:
        return {"error": "TUGO_API_KEY not configured on server"}

    # TuGo expects an ISO country code or country slug. For now we just lowercase
    # and strip spaces; you can improve this mapping later.
    country_slug = country.strip().lower().replace(" ", "-")

    url = f"https://api.tugo.com/v1/travelsafe/countries/{country_slug}"

    try:
        resp = requests.get(
            url,
            headers={"X-Auth-API-Key": TUGO_API_KEY},
            timeout=10,
        )
    except Exception as e:
        logger.exception("Error calling TuGo Travel Advisory API")
        return {"error": f"Request failed: {e}"}

    if resp.status_code != 200:
        return {
            "error": f"TuGo returned HTTP {resp.status_code}",
            "debug": {
                "url": url,
                "response_text": resp.text,
                "headers": dict(resp.headers)
            }
        }

    try:
        data = resp.json()
    except Exception as e:
        return {"error": f"Failed to parse JSON from TuGo: {e}", "body": resp.text}
    # Normalize output structure
    normalized = {
        "country_input": country,
        "country_resolved": data.get("country", {}).get("name") if isinstance(data, dict) else None,
        "advisories": data.get("advisories"),
        "health": data.get("health"),
        "safety": data.get("safety"),
        "entry_exit": data.get("entryExit"),
        "sources": ["TuGo Travel Advisory API (targeted primarily to Canadian travellers)"],
        "raw": data,
    }

    return normalized   


tugo_travel_tool = FunctionTool(tugo_travel_advisory)

#Tugo agent
tugo_travel_agent = LlmAgent(
    model=GEMINI_LITE,
    name="tugo_travel_agent",
    instruction="""
You are a travel guidance summarizer.
Use  tugo_travel_tool to fetch structured advisory & health/safety info for the destination and time window.

Focus on high-level public-health recommendations like:
- Required or recommended vaccines
- Malaria / mosquito-borne risk
- Food/water precautions
- Outbreak alerts
 or other health/safety risks

Always cite URLs.
    """,
    tools=[tugo_travel_tool],
    output_key="tugo_travel_summary",
)

# ParallelAgent to fetch CDC,WHO + TUGO info concurrently
travel_parallel_evidence = ParallelAgent(
    name="travel_parallel_evidence",
    sub_agents=[cdc_who_travel_agent, tugo_travel_agent],
)


# Travel advisory Summarizer
travel_summary_agent = LlmAgent(
    model=GEMINI_LITE,
    name="travel_summary_agent",
    instruction="""
You are a travel & vaccine companion.
You see two inputs:
- CDC view: {+cdc_who_travel_summary}
- WHO view: {+tugo_travel_summary}

Your tasks:
1. Reconcile them into a single, user-friendly explanation but mention which source each part comes from.
2. Highlight any differences in guidance.
3. Present a checklist:
   - Before you travel (vaccines, prescriptions, preventive steps)
   - While traveling
   - When you return
4. Strong safety language: you are NOT a doctor; user must confirm vaccines
   and prescriptions with a clinician.

Do NOT invent vaccines or treatments. If unsure, say so.
    """,
    output_key="travel_final_answer",
)

# Sequential travel workflow: intake -> parallel evidence -> summarize
travel_workflow_agent = SequentialAgent(
    name="travel_workflow_agent",
    sub_agents=[travel_intent_agent, travel_parallel_evidence, travel_summary_agent],
)
travel_tool = AgentTool(agent=travel_workflow_agent)


#---------------------------------------------
# Chronic Condition Content Coach Functioality
#---------------------------------------------

#Chronic coach for chronic condition
chronic_coach_agent = LlmAgent(
    model=GEMINI_LITE,
    name="chronic_coach_agent",
    instruction="""
You are a chronic-condition education coach.
The user is located in: {+user:location}

Goals:
- Explain the user's chronic condition in clear, simple language.
- Suggest conservative lifestyle routines consistent with mainstream guidelines.
- Suggest questions the user can ask their clinician.

STRICT SAFETY RULES:
- Never diagnose, prescribe, or change medications.
- Never tell people to start/stop medicines.
- Never give emergency advice.

When the user asks about a chronic condition:
1. Assume they already have a diagnosis or concern (e.g., "I have diabetes",
   "I have kidney stones", "I have hypertension").
2. Create a concise education plan that includes:
   - What the condition is (high level).
   - Common symptoms / risk factors (high level, no fear-mongering).
   - Lifestyle / daily routine suggestions that are safe and conservative.
   - Specific questions they can ask their clinician.
3. Always finish with: "This is educational only and not a substitute for care
   from your clinician."
""",
    tools=[health_search_tool, medical_mcp_toolset],
    output_key="chronic_plan",

)

#Finds hospital based on the location and condition
hospital_finder_agent = LlmAgent(
    model=GEMINI_LITE,
    name="hospital_finder_agent",
    instruction="""
You are a hospital finder.Users current location is: {+user:location}

Your job is to suggest a few hospitals or clinics near the users location that often treat the user's chronic condition.

Rules:

1. Get the users location and only find the hospitals near that location.
1. Use the google_search_tool to search the open web. Use queries like:
   - "hospitals near [CITY] for [CONDITION]"
   - "urology clinic near [CITY]" (for kidney stones, etc.)

2. From the search results, pick 3 to 5 realistic options. For each, return:
   - Name
   - City / neighborhood
   - One-line reason it's relevant (e.g., "large urology department")
   - URL

3. Do NOT invent hospitals. Base your output on the search results.
4. You are NOT endorsing any provider. Add a line like:
   "These are example options based on web search; they are not endorsements."

Your output should be a short bullet list of hospitals/clinics.
""",
    tools=[google_search_tool],
    output_key="hospital_suggestions",
   
)

#Summarizer agent
chronic_summary_agent = LlmAgent(
    model=GEMINI_LITE,
    name="chronic_summary_agent",
    instruction="""
You are a summarizer that combines an education plan with nearby hospital options.

You receive:
- chronic_plan: {+chronic_plan}
- hospital_suggestions: {+hospital_suggestions}

Create a single, user-friendly answer that:

1. Shows the education plan first (possibly lightly edited for flow).
2. Then adds a section like "Hospitals / clinics near you that often treat this condition"
   and lists the hospital suggestions with city , state and their website link in a clean, numbered or bulleted format.
3. Ends with a strong disclaimer that:
   - This is general information.
   - Hospital choices and treatments must be discussed with their own clinician.
   - You are not endorsing any specific provider.adk web

Do NOT add new hospitals that are not in hospital_suggestions.
""",
    output_key="chronic_final_answer",
)

# Chronic workflow: plan -> hospitals -> combined
chronic_workflow_agent = SequentialAgent(
    name="chronic_workflow_agent",
    sub_agents=[chronic_coach_agent, hospital_finder_agent, chronic_summary_agent],)     
chronic_tool = AgentTool(agent=chronic_workflow_agent)


# prescription explainer agent

prescription_explainer_agent = LlmAgent(
    model=GEMINI_LITE,
    name="prescription_explainer_agent",
    instruction="""
You are a medication + diagnosis explainer.

User may give:
- a diagnosis (e.g., "I have hypertension")
- a drug name (e.g., "metoprolol")
- or both. 
and you need to explain accordingly.

Rules:
1. If a drug is mentioned, use 'google_search_tool' to find information about the drug.
2. Explain in simple terms:
   - what the drug is for
   - how it generally works
   - what it is usually prescribed for
   - common side effects
   - is it fda approved?
   - important warnings / interactions (high level)

3. If a diagnosis is mentioned, explain:
   - what it is (high level)
   - typical goals of treatment
4. You may add any important information regarding the drug and diagnosis.
5. NEVER prescribe, change dosages, or tell user to start/stop meds.
6. End with: "This is educational only; confirm with your clinician."
Output should be clear, structured paragraphs.
""",
    tools=[google_search_tool,],
    output_key="prescription_explanation",
)
prescription_tool = AgentTool(agent=prescription_explainer_agent)


# auto save
async def auto_save_session_to_memory_callback(callback_context):
    inv = getattr(callback_context, "_invocation_context", None)
    if inv is None:
        return None
    memory_service = getattr(inv, "memory_service", None)
    session = getattr(inv, "session", None)

    if memory_service is None or session is None:
        return None

    await memory_service.add_session_to_memory(session)
    return None
     
#  Custom Function save_location
def save_location(tool_context: ToolContext, location: str) -> dict:
    """
    Saves the user's location to the current session state.
    ADK's memory layer persists the whole session, not key/value pairs.
    """
    clean_location = location.strip()
    print(f"[save_location] Saving location: {clean_location}")
    logger.info(f"[save_location] Saving location: {clean_location}")
    # Store in session state (this is what ADK persists)
    tool_context.state["user:location"] = clean_location
    # alias: tool_context.state["user:location"] = clean_location also works

    return {"status": "saved", "user:location": clean_location}
save_location_tool = FunctionTool(save_location)

#  Custom Function get_location
def get_user_location(tool_context: ToolContext) -> str:
    """
    
    Retrieves the user's location from the current session state.
    Returns 'NOT_SET' if not present.
    """
    print(f"[get_user_location] Retrieving location.")
    logger.info(f"[get_user_location] Retrieving location.")
    return tool_context.session.state.get("user:location", "NOT_SET")

get_location_tool = FunctionTool(get_user_location)


#The main Router Agent, this agent routes the workflow depending on the query
router_agent = LlmAgent(
    model=GEMINI_LITE,
    name="router_agent",
    instruction="""
You are the top-level health navigator.

Your job is to route user requests to the appropriate specialized agent.
First Introduce yourslef and tell the user you can help with:
- Verifying health claims and misinformation
- Travel health advice and vaccine recommendations
- Chronic condition education and lifestyle coaching
- Medication and prescription explanations


Classify the user's intent and delegate using the appropriate tool:
- If they ask about verifying health claims, misinformation, or "is this true?" 
  -> call `misinformation_agent`
- If they mention travel, vaccines for trips, or "I'm going to [country]" 
  -> call `travel_workflow_agent`
- If user asks about a medication, prescription, side effects, or drug purpose 
  -> call `prescription_explainer_agent`
- If they mention chronic conditions,(diabetes, hypertension, asthma, etc.)
    1. **CALL the `get_user_location` tool.**
    2. If the tool returns "NOT_SET":
        - Ask the user: "To help you find care, what city and state are you in?"
        - (Do not call the chronic agent yet).
        - When the user replies in the NEXT turn, call `save_location`.
        - call chronic_workflow_agent with the location included in your message to the agent (e.g., "User has diabetes and is in Austin, TX").
    3. If the tool returns a valid location (e.g., "Austin, TX"):
        - Call `chronic_workflow_agent` immediately.
        - Include the location in your message to the agent (e.g., "User has diabetes and is in Austin, TX").
        

Call exactly ONE agent tool per turn. After the tool executes and returns results,
pass those results through to the user as your response.
    """,
    tools=[misinfo_tool, travel_tool, chronic_tool, prescription_tool, preload_memory, save_location_tool, get_location_tool],
    after_agent_callback=auto_save_session_to_memory_callback
)

# App & Runner with session + memory + logging plugin
APP_NAME = "health_multi_agent_app"
root_agent = router_agent
resumability_config = ResumabilityConfig(is_resumable=True)
logging_plugin = LoggingPlugin()


root_app = App(
    root_agent=root_agent,
    name=APP_NAME,
    plugins=[logging_plugin],
    resumability_config=resumability_config,
    
)



async def main():
    """
    Simple CLI REPL for local testing.
    """
    user_id = "cli-user"
    # Generate a session ID once for this run
    session_id = "session-" + str(uuid.uuid4())
    
    print("🚀 Health Navigator CLI")
    print(f"Session ID: {session_id}")
    print("Type your question, or 'exit' to quit. Type '/state' to inspect memory.\n")

    runner = Runner(
        agent=root_app.root_agent,
        app_name=root_app.name,
        session_service=session_service
    )
  
    try:
        session = await session_service.create_session(
            app_name=root_app.name,
            user_id=user_id,
            session_id=session_id
        )
    except Exception as e:
        print(f"❌ Critical Error: Could not create session. {e}")
        return

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except EOFError:
            break

        if user_input.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        if user_input == "/state":
            try:
                current_session = await session_service.get_session(
                    app_name=root_app.name,
                    user_id=user_id,
                    session_id=session_id,
                )
                print("\n🧠 Session State:")
                if current_session and current_session.state:
                    for k, v in current_session.state.items():
                        print(f"  {k}: {v}")
                else:
                    print("  (Empty)")
                
                print("\n🗄️  Long-Term Memory items:")
                try:
                    memories = await memory_service.get_memories(session_id)
                    if memories:
                        for m in memories:
                            # Handle dict vs object
                            k = getattr(m, "key", m.get("key") if isinstance(m, dict) else "Unknown")
                            v = getattr(m, "value", m.get("value") if isinstance(m, dict) else "Unknown")
                            print(f"  {k}: {v}")
                    else:
                        print("  (No memories found)")
                except Exception as mem_err:
                    print(f"  (Could not fetch memories: {mem_err})")
                    
                print()
            except Exception as e:
                print(f"⚠️ Could not fetch session details: {e}")
            continue

        print("🤖 Agent: ", end="", flush=True)

        try:
            message_payload = genai_types.Content(role="user", parts=[genai_types.Part(text=user_input)])
            async for event in runner.run_async(
                session_id=session_id,
                new_message=message_payload,
                user_id=user_id,
            ):
              
                if hasattr(event, "content") and event.content:
                     text = extract_text(event)
                     if text:
                         print(text, end="", flush=True)

            print()  # newline after response

        except Exception as e:
            logger.exception("Runtime error while handling user input")
            print(f"\n⚠️ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())