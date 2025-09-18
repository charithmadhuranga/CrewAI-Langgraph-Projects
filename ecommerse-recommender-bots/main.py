from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_bot import langgraph_chat
from crewai_bot import crewai_chat

app = FastAPI(title="E-commerce Multi-User Chatbot API (Gemini + dotenv)")

class ChatRequest(BaseModel):
    user_id: str
    query: str
    mode: str  # "langgraph" or "crewai"

@app.post("/chat")
def chat(request: ChatRequest):
    if request.mode == "langgraph":
        response = langgraph_chat(request.user_id, request.query)
    elif request.mode == "crewai":
        response = crewai_chat(request.user_id, request.query)
    else:
        response = "Invalid mode. Use 'langgraph' or 'crewai'."
    return {"response": response}
