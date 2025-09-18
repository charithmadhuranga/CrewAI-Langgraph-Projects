# memory.py
import os
import json
from uuid import uuid4
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

# LangChain Gemini embeddings wrapper
embedding_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

# Chroma expects an embedding function object that supports:
#   - __call__(self, input)
#   - embed_documents(self, input, **kwargs)
#   - embed_query(self, input, **kwargs)
class GeminiEmbeddingFunction:
    def __init__(self, emb_model):
        self._model = emb_model

    # Called by Chroma in some code paths
    def __call__(self, input, **kwargs):
        return self.embed_documents(input, **kwargs)

    # Accepts a single string or a list (keyword arg 'input' may be used)
    def embed_documents(self, input, **kwargs):
        if input is None:
            return []
        # normalize to list[str]
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)
        # embedding_model.embed_documents expects List[str]
        return self._model.embed_documents(texts)

    # For single query or list of queries.
    # If input is a list, return list of vectors. If string, return single vector.
    def embed_query(self, input, **kwargs):
        if input is None:
            return []
        if isinstance(input, list):
            # return list of vectors
            return [self._model.embed_query(t) for t in input]
        else:
            return self._model.embed_query(input)

# Initialize Chroma client (in-memory by default).
# If you want persistence on disk, see the note below.
chroma_client = chromadb.Client()

embedding_fn = GeminiEmbeddingFunction(embedding_model)

# Collections (if they already exist, create_collection will raise; we handle that)
def _create_collection_safe(name):
    try:
        return chroma_client.create_collection(name=name, embedding_function=embedding_fn)
    except Exception:
        # If it exists, return existing
        return chroma_client.get_collection(name)

conversation_store = _create_collection_safe("ecommerce_conversations")
profile_store = _create_collection_safe("ecommerce_profiles")

# ----------------------
# Conversation Memory
# ----------------------
def store_conversation(user_id: str, message: str, response: str):
    doc = f"User: {message}\nBot: {response}"
    conversation_store.add(
        documents=[doc],
        ids=[f"{user_id}_{uuid4().hex}"],
        metadatas=[{"user_id": user_id}]
    )

def retrieve_context(user_id: str, query: str, k: int = 3):
    results = conversation_store.query(
        query_texts=[query],
        n_results=k,
        where={"user_id": user_id}
    )
    docs = []
    # results["documents"] is a list (one per query) of lists
    for sub in results.get("documents", []):
        docs.extend(sub)
    return docs

# ----------------------
# Profile Builder
# ----------------------
def set_user_profile(user_id: str, profile: dict):
    """Store structured profile as JSON string."""
    profile_store.add(
        documents=[json.dumps(profile)],
        ids=[f"profile_{user_id}"],
        metadatas=[{"user_id": user_id}]
    )

def get_user_profile(user_id: str):
    # Prefer get by id (safer). If not present, return None.
    try:
        res = profile_store.get(ids=[f"profile_{user_id}"])
        docs = res.get("documents", [])
        if docs and docs[0]:
            return json.loads(docs[0])
    except Exception:
        # fallback to query
        try:
            res = profile_store.query(query_texts=[f"profile_{user_id}"], n_results=1, where={"user_id": user_id})
            docs = res.get("documents", [])
            if docs and docs[0]:
                return json.loads(docs[0][0])
        except Exception:
            pass
    return None

# ----------------------
# Adaptive Memory
# ----------------------
def update_profile(user_id: str, new_info: str):
    current = get_user_profile(user_id) or {}
    adaptive = current.get("adaptive_notes", [])
    adaptive.append(new_info)
    current["adaptive_notes"] = adaptive
    set_user_profile(user_id, current)
    return current
