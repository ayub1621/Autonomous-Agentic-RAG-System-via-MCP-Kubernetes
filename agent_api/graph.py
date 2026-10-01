import os
import sys
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Add the router directory to the path to import the PyTorch model
sys.path.append(os.path.abspath("../pytorch_router"))
from router import IntentRouter # pyright: ignore

# 2. Define the Agent's Memory (State)
class AgentState(TypedDict):
    query: str
    domain: str        # e.g., 'financial_data' or 'crm_data'
    context: str       # Data retrieved from the MCP server
    final_answer: str

# Initialize components
router_model = IntentRouter()
# set export GOOGLE_API_KEY="your_api_key" in the terminal
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 3. Define the Nodes
def route_query_node(state: AgentState):
    """Node 1: Uses PyTorch to classify the intent."""
    print("-> Routing Query...")
    domain = router_model.route_query(state["query"])
    return {"domain": domain}

async def fetch_mcp_context_node(state: AgentState):
    """Node 2: Connects to the appropriate MCP server based on the domain."""
    print(f"-> Fetching context for domain: {state['domain']}")
    domain = state["domain"]
    
    # simulate the MCP response:
    if domain == "financial_data":
        # Simulating invoking the yfinance MCP tool
        fetched_data = "{'symbol': 'AAPL', 'current_price': 150.25, '5_day_trend': 'Upward'}"
    elif domain == "crm_data":
        # Simulating invoking the CRM MCP tool
        fetched_data = "{'id': 'USR-001', 'name': 'Acme Corp', 'status': 'Active', 'mrr': 15000.0}"
    else:
        fetched_data = "No external data required."
        
    return {"context": fetched_data}

# Change from 'def' to 'async def'
async def generate_response_node(state: AgentState):
    """Node 3: Uses the LLM to synthesize the final answer."""
    print("-> Generating Final Response...")
    prompt = f"""
    You are an enterprise AI assistant. Answer the user's query using ONLY the provided context.
    Query: {state['query']}
    Context from MCP Tools: {state['context']}
    """
    
    # Use await and ainvoke for non-blocking execution
    response = await llm.ainvoke(prompt)
    
    if isinstance(response.content, str):
        clean_text = response.content
    elif isinstance(response.content, list):
        clean_text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in response.content
        )
    else:
        clean_text = str(response.content)

    return {"final_answer": clean_text.strip()}

# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", route_query_node)
workflow.add_node("mcp_fetcher", fetch_mcp_context_node)
workflow.add_node("generator", generate_response_node)

# Define the edges (the flow of execution)
workflow.add_edge(START, "router")
workflow.add_edge("router", "mcp_fetcher")
workflow.add_edge("mcp_fetcher", "generator")
workflow.add_edge("generator", END)

# Compile the graph
app = workflow.compile()

# 5. Test the Graph execution
if __name__ == "__main__":
    async def run_test():
        test_state = {"query": "What is the current trend for Apple stock?", "domain": "", "context": "", "final_answer": ""}
        result = await app.ainvoke(test_state)
        print(f"\nFinal Answer: {result['final_answer']}")
        
    asyncio.run(run_test())