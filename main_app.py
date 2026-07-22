import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
import httpx
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
import uvicorn

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("github-pr-reviewer")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"
BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = os.getenv("TOPIC_NAME", "github-pr-reviewer")
CONSUMER_GROUP_ID = os.getenv("CONSUMER_GROUP_ID", "pr-reviewer-group")

MCP_SERVERS_CONFIG = {
    "github-reader": {
        "url": "http://127.0.0.1:8000/sse",
        "transport": "sse"
    }
}

producer: AIOKafkaProducer | None = None
mcp_client: MultiServerMCPClient | None = None
review_agent = None
background_tasks = set()

MAX_CONCURRENT_REVIEWS = 5
REVIEW_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REVIEWS)

async def get_tools_with_retry(mcp_client: MultiServerMCPClient, max_retries: int = 3):
    """Load tools with retry logic"""
    for attempt in range(1, max_retries + 1):
        try:
            tools = await asyncio.wait_for(
                mcp_client.get_tools(),
                timeout=15.0,
            )
            logger.info(f"Successfully loaded {len(tools)} tools from MCP server.")
            return tools
        except asyncio.TimeoutError:
            logger.warning(f"Timeout loading tools (attempt {attempt}/{max_retries})")
        except Exception as e:
            logger.warning(f"Failed to load tools (attempt {attempt}/{max_retries}): {e}")

        if attempt < max_retries:
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to load tools after all retries")
    return []

async def post_comment_to_github(repo: str, pr_num: int, comment: str):
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_num}/comments"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-Github-Api-Version": "2022-11-28",
        "User-Agent": "MCP-Client-AI-Agent"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    payload = {"body": comment}
    timeout = httpx.Timeout(10.0, read=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 201:
                logger.info(f"Successfully posted PR review to {repo}#{pr_num}")
            else:
                logger.error(f"Failed to post comment to GitHub ({response.status_code}): {response.text}")
        except httpx.RequestError as e:
            logger.error(f"Network error while communicating with GitHub API: {e}")

async def run_pr_review_workflow(repo: str, pr_num: int, commit_sha: str):
    """Executes concurrently in background without blocking the consumer loop."""

    global review_agent

    if review_agent is None:
        logger.info(f"Agent not initialized for PR {repo}#{pr_num}")
        return

    async with REVIEW_SEMAPHORE:
        try:
            user_message = (
                f"Please fetch and review the pull request diff for repository '{repo}' "
                f"and Pull Request #{pr_num} (commit: {commit_sha})."
            )

            logger.info(f"Invoking review agent loop for {repo}#{pr_num}")
            response = await review_agent.ainvoke({"messages": [HumanMessage(content=user_message)]})
            review_content = response["messages"][-1].content

            await post_comment_to_github(repo, pr_num, review_content)
        except Exception as e:
            logger.exception(f"Unhandled error during agent execution loop for {repo}#{pr_num}: {e}")

async def start_consumer_worker():
    """Background Kafka Consumer Task"""
    logger.info("start_consumer_worker started")
    kafka_consumer = AIOKafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True
    )

    await kafka_consumer.start()
    logger.info(f"Consumer started for topic '{TOPIC_NAME}' (Group ID: '{CONSUMER_GROUP_ID}')")

    try:
        async for msg in kafka_consumer:
            payload = msg.value
            logger.info(f"Received message from partition {msg.partition} at offset {msg.offset}")

            try:
                repo_full_name = payload.get("repo_full_name")
                pr_number = payload.get("pr_number")
                commit_sha = payload.get("commit_sha")

                if repo_full_name and pr_number:
                    task = asyncio.create_task(
                        run_pr_review_workflow(
                            repo=repo_full_name,
                            pr_num=pr_number,
                            commit_sha=commit_sha
                        )
                    )
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
                else:
                    logger.warning(f"Payload missing required fields: {payload}")

            except Exception as e:
                logger.error(f"Error processing message {payload}: {e}", exc_info=True)

    except asyncio.CancelledError:
        logger.info("Cancellation received, stopping consumer loop...")
    finally:
        logger.info("Stopping Kafka Consumer...")
        await kafka_consumer.stop()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer, mcp_client, review_agent

    logger.info("Starting Kafka Producer...")
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    await producer.start()

    try:
        logger.info("Initializing MCP Client...")
        mcp_client = MultiServerMCPClient(MCP_SERVERS_CONFIG)

        langchain_tools = await get_tools_with_retry(mcp_client=mcp_client, max_retries=3)
        if not langchain_tools:
            logger.error("No tools loaded. Agent will not function properly.")
        else:
            logger.info(f"Loaded {len(langchain_tools)} tools from MCP server.")

            llm = ChatOpenAI(model="gpt-4o", temperature=0)

            system_prompt = (
                "You are an expert Senior Security & QA Engineer. "
                "When given a repository and PR number, you MUST FIRST use your available tools "
                "(such as `get_pr_diff`) to retrieve the patch diff before attempting any review. "
                "Analyze the retrieved diff for logical flaws, security vulnerabilities, or performance issues. "
                "Provide critical feedback in a clear, concise markdown format. "
                "If the code looks excellent, state explicitly that no issues were found."
            )

            review_agent = create_agent(model=llm, tools=langchain_tools, system_prompt=system_prompt)

            logger.info("Review agent created successfully and ready for reuse.")

    except Exception as e:
        logger.error(f"Failed to initialize MCP/tools/agent: {e}")
        review_agent = None

    logger.info("Starting Background Consumer Task...")
    consumer_task = asyncio.create_task(start_consumer_worker())

    yield

    logger.info("Cancelling Background Consumer Task...")
    consumer_task.cancel()

    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    if background_tasks:
        logger.info(f"Waiting for {len(background_tasks)} active review tasks to complete...")
        await asyncio.gather(*background_tasks, return_exceptions=True)

    if producer:
        logger.info("Stopping Kafka Producer...")
        await producer.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def github_webhook(request: Request):
    event = request.headers.get("X-GitHub-Event")

    if event != "pull_request":
        return {"status": "ignored", "reason": "Not a pull_request event"}

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")

    action = payload.get("action")

    if action in ["opened", "synchronize"]:
        try:
            result = {
                "repo_full_name": payload["repository"]["full_name"],
                "pr_number": payload["number"],
                "commit_sha": payload["pull_request"]["head"]["sha"]
            }
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing key in payload structure: {e}"
            )

        if producer is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Producer not initialized")

        try:
            await producer.send_and_wait(TOPIC_NAME, result)
            logger.info(f"Published PR #{result['pr_number']} event to Kafka")
        except KafkaError as ex:
            logger.error(f"Failed to publish event to Kafka: {ex}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to publish event to queue")

        return {"status": "processing"}

    return {"status": "ignored", "reason": f"Action '{action}' not handled"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)