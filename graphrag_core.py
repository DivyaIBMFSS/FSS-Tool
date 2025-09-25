# graphrag_core.py

import os
import json
import faiss
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import logging
import re
import google.generativeai as genai
import streamlit as st

# === STREAMLIT SECRETS ===
API_KEY = st.secrets["GEMINI_API_KEY"]
NEO4J_URI = st.secrets["NEO4J_URI"]        # e.g. "neo4j+s://xxxx.databases.neo4j.io"
NEO4J_USER = st.secrets["NEO4J_USER"]      # "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

# === GEMINI CONFIG ===
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash")

# === NEO4J CONNECTION ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_cypher(query, params=None):
    """Run Cypher query on Aura and return results as list of dicts"""
    if params is None:
        params = {}
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]

# === FAISS + EMBEDDINGS ===
index = faiss.read_index("faiss_indexmain.idx")
with open("vector_metadatamain.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === INTENT ANALYSIS ===
def analyze_question(q):
    prompt = f"""
You're analyzing a user's question to extract:
- client
- technology
- capability
- geography
- intent

Possible intents: 
- find_client_by_technology
- find_client_by_capability
- find_client_by_geography
- summarize_client_relationship
- list_files_by_client
- generate_catalog
- general_query

Return only JSON. Example:
{{
  "client": "...",
  "technology": "...",
  "capability": "...",
  "geography": "...",
  "intent": "..."
}}

User Question:
"{q}"
"""
    response = llm_model.generate_content(prompt)
    try:
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        return json.loads(match.group(0)) if match else json.loads(response.text)
    except Exception as e:
        logger.error(f"❌ Intent parse error: {e}")
        return {"client": "", "technology": "", "capability": "", "geography": "", "intent": "general_query"}

# === DOCUMENT RETRIEVAL BY CLIENT ===
def get_all_client_documents(client_name):
    query = """
    MATCH (c:Client)<-[:ABOUT]-(d:Document)
    WHERE toLower(c.name) CONTAINS toLower($client)
    RETURN d.file_name AS file_name, d.slide_number AS slide_number, d.text AS text
    ORDER BY d.file_name, d.slide_number
    """
    return run_cypher(query, {"client": client_name})

# === RAG SEARCH ===
def search_rag_context(query, top_k=100):
    vec = embedder.encode([query])[0].astype("float32").reshape(1, -1)
    _, idxs = index.search(vec, top_k)
    return [metadata[i] for i in idxs[0]]

# === FIND CLIENT FOR SLIDE ===
def find_client(file, slide):
    query = """
    MATCH (d:Document {file_name: $file, slide_number: $slide})-[:ABOUT]->(c:Client)
    RETURN c.name AS client_name
    """
    res = run_cypher(query, {"file": file, "slide": slide})
    return res[0]["client_name"] if res else "CLIENT_NOT_SPECIFIED"

# === GEMINI CALL ===
def call_gemini(prompt, temperature=0.1, max_tokens=2048):
    try:
        res = llm_model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return res.text
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return "LLM failed."

# === FORMAT CONTEXT ===
def format_context(entries, remove_warnings=True):
    result = ""
    for entry in entries:
        file = entry.get("file_name", "N/A")
        slide = entry.get("slide_number", "?")
        text = entry.get("text", "")
        client = find_client(file, slide)

        if remove_warnings and "!!! Please remove all client logos" in text:
            continue

        for word in ["Bank", "Client", "Company"]:
            text = text.replace(word, client)

        result += f"[{file} | Slide {slide}] {text.strip()}\n"
    return result, len(entries)

# === MAIN FUNCTION CALLED BY STREAMLIT ===
def process_question(question, top_k=100, temperature=0.1, max_tokens=2048):
    parsed = analyze_question(question)
    logger.info(f"Entities: {parsed}")

    if parsed["intent"] == "summarize_client_relationship" and parsed["client"]:
        docs = get_all_client_documents(parsed["client"])
        context, count = format_context(docs)
        if not context.strip():
            return "❌ No relevant content found."

        final_prompt = f"""
You are an expert IBM Consulting analyst writing a formal summary for leadership.

User asked to summarize the IBM relationship with client '{parsed['client']}'.

Write detailed paragraphs (not bullets) with proper flow. For each fact, include:
- [file_name | Slide number] after the sentence
- Cover all programs, technologies, milestones
- Use only the context provided below
- DO NOT fabricate anything
- Break into meaningful paragraphs.

Context:
{context}

Your Summary:
"""
        return call_gemini(final_prompt, temperature, max_tokens)

    # === General case with RAG ===
    rag_ctx = search_rag_context(question, top_k)
    context, _ = format_context(rag_ctx)

    final_prompt = f"""
User asked:
{question}

Relevant context from documents:
{context}

Based only on this context:
- Answer truthfully and completely.
- If it's a catalog or list, show max 20 entries, then say how many more exist.
- For lists, give a clean structured list.
- If no relevant info, say: "No relevant information was found in the database."
- Always include [file | Slide] references.
- Ignore noise lines like "!!! Please remove all client logos..."

Your Answer:
"""
    return call_gemini(final_prompt, temperature, max_tokens)
