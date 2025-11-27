import os
import vertexai
from vertexai import agent_engines
import asyncio

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = "us-central1"

ENGINE_NAME = "projects/1084699953366/locations/us-central1/reasoningEngines/7837332076926861312"


vertexai.init(project=PROJECT_ID, location=REGION)
# Get the most recently deployed agent
agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    print(f"✅ Connected to deployed agent: {remote_agent.resource_name}")

    async def _stream_and_print():
        async for item in remote_agent.async_stream_query(
            message="i have kidney stones, what do i do?",
            user_id="user_42",
        ):
            print(item)

    # Run the async stream
    asyncio.run(_stream_and_print())
else:
    print("❌ No agents found. Please deploy first.")
   
