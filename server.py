from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, status
from langchain.agents import create_agent
import os
import httpx
from contextlib import asynccontextmanager
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("github-pr-reviewer"
                           )
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

GITHUB_API = "https://api.github.com"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MCP_SERVERS_CONFIG = {
    "github-reader": {
        "url": "http://127.0.0.1:8000/sse",
        "transport": "sse"
    }
}

http_client: httpx.AsyncClient | None = None
mcp_client: MultiServerMCPClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown lifecycle resources safely."""
    global http_client, mcp_client

    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN is missing. Outgoing comments will be unauthenticated.")
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is missing. Agent execution will be failed.")

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0))

    mcp_client = MultiServerMCPClient(MCP_SERVERS_CONFIG)
    logger.info("Initialized MultiServerMCPClient instance.")

    yield

    if http_client:
        await http_client.aclose()

    logger.info("Lifecycle resources cleaned up successfully.")

app = FastAPI(lifespan=lifespan)

async def run_pr_review_workflow(repo: str, pr_num: int, commit_sha: str):
    """Executes the AI Reviewer agent loop and posts the output back to GitHub."""
    if not mcp_client:
        logger.error("MCP client not initialized. Skipping workflow.")
        return

    try:
        langchain_tools = await mcp_client.get_tools()

        if not langchain_tools:
            print(f"Error: No tools retrieved from MCP server for PR #{pr_num}")
            return

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )

        system_prompt = (
            "You are an expert Senior Security & QA Engineer. "
            "When given a repository and PR number, you MUST FIRST use your available tools "
            "(such as `get_pr_diff`) to retrieve the patch diff before attempting any review. "
            "Analyze the retrieved diff for logical flaws, security vulnerabilities, or performance issues. "
            "Provide critical feedback in a clear, concise markdown format. "
            "If the code looks excellent, state explicitly that no issues were found."
        )

        agent = create_agent(
            model=llm,
            tools=langchain_tools,
            system_prompt=system_prompt
        )

        user_message = (
            f"Please fetch and review the pull request diff for repository '{repo}' "
            f"and Pull Request #{pr_num} (commit: {commit_sha})."
        )

        logger.info(f"Invoking review agent loop for {repo}#{pr_num}")
        response = await agent.ainvoke({
            "messages": [HumanMessage(content=user_message)]
        })

        review_content = response["messages"][-1].content

        await post_comment_to_github(repo, pr_num, review_content)
    except Exception as e:
        logger.exception(f"Unhandled error during agent execution loop for {repo}#{pr_num}: {e}")

async def post_comment_to_github(repo: str, pr_num: int, comment: str):
    """Asynchronous function to post comments without blocking the event loop."""
    if not http_client:
        logger.error("HTTP client pool is uninitialized.")
        return

    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_num}/comments"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-Github-Api-Version": "2022-11-28",
        "User-Agent": "MCP-Client-AI-Agent"
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    payload = {"body": comment}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 201:
                logger.info(f"Successfully posted PR review to {repo}#{pr_num}")
            else:
                logger.error(f"Failed to post comment to GitHub ({response.status_code}):{response.text}")
        except httpx.RequestError as e:
            logger.error(f"Network error while communicating with GitHub API: {e}")


@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Secure entry point to catch incoming GitHub Pull Reqyest events."""
    event = request.headers.get("X-Github-Event")

    if event != "pull_request":
        return {"status": "ignored", "reason": "Not a pull_request event"}

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")

    action = payload.get("action")

    if action in ["opened", "synchronize"]:
        try:
            repo_full_name = payload["repository"]["full_name"]
            pr_number = payload["number"]
            commit_sha = payload["pull_request"]["head"]["sha"]
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Missing key in payload structure: {e}")

        background_tasks.add_task(
            run_pr_review_workflow,
            repo_full_name,
            pr_number,
            commit_sha
        )
        return {"status": "processing"}
    return {"status": "ignored", "reason": f"Action '{action}' not handled"}