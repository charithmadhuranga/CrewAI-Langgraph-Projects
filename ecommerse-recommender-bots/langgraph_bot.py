import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from memory import (
    get_user_profile, set_user_profile,
    retrieve_context, store_conversation, update_profile
)

# Load .env
load_dotenv()

# Use Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini/gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)
search = DuckDuckGoSearchRun()

class State(dict):
    user_id: str
    query: str
    profile: dict
    history: list
    search_results: str
    recommendations: str

def load_memory(state: State):
    user_id = state["user_id"]
    profile = get_user_profile(user_id)
    if not profile:
        profile = {"budget": "under $300", "style": "casual", "brands": ["Nike", "Adidas"]}
        set_user_profile(user_id, profile)
    history = retrieve_context(user_id, state["query"])
    return {"profile": profile, "history": history}

def run_search(state: State):
    query = f"{state['query']} with preferences: {state['profile']}"
    results = search.run(query)
    return {"search_results": results}

def recommend(state: State):
    prompt = f"""
    User Profile: {state['profile']}
    Past History: {state['history']}
    Query: {state['query']}
    Search Results: {state['search_results']}

    Recommend 3 products with pros/cons, tailored to the user.
    """
    response = llm.invoke(prompt)
    return {"recommendations": response.content}

def save_memory(state: State):
    user_id = state["user_id"]
    store_conversation(user_id, state["query"], state["recommendations"])
    update_profile(user_id, f"User searched for {state['query']}")
    return {}

# Build graph
graph = StateGraph(State)
graph.add_node("load_memory", load_memory)
graph.add_node("search", run_search)
graph.add_node("recommend", recommend)
graph.add_node("save_memory", save_memory)

graph.set_entry_point("load_memory")
graph.add_edge("load_memory", "search")
graph.add_edge("search", "recommend")
graph.add_edge("recommend", "save_memory")
graph.add_edge("save_memory", END)

app_graph = graph.compile()

def langgraph_chat(user_id: str, query: str):
    result = app_graph.invoke({"user_id": user_id, "query": query})
    return result["recommendations"]
