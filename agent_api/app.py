from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import the compiled LangGraph workflow from your graph.py file
from graph import app as agent_workflow 

# 1. Initialize FastAPI
app = FastAPI(
    title="Autonomous Agent API",
    description="API serving a LangGraph agent with PyTorch routing and MCP data tools."
)

# 2. Define Request and Response Schemas
class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    query: str
    domain: str
    context: str
    final_answer: str

# 3. Create the Endpoint
@app.post("/api/v1/invoke_agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    try:
        print(f"\n[API Request Received] Query: {request.query}")
        
        # Initialize the state for LangGraph
        initial_state = {
            "query": request.query,
            "domain": "",
            "context": "",
            "final_answer": ""
        }
        
        # Await the execution of the full LangGraph graph
        result = await agent_workflow.ainvoke(initial_state)
        
        # Return the structured payload
        return AgentResponse(
            query=result["query"],
            domain=result["domain"],
            context=result["context"],
            final_answer=result["final_answer"]
        )
        
    except Exception as e:
        print(f"[API Error] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. Run the Server
if __name__ == "__main__":
    print("Starting FastAPI Server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)