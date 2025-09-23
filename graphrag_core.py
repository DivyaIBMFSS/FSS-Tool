# graphrag_core.py

import os
import json
import faiss
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import logging
import re
import google.generativeai as genai
                            
# Secrets from Streamlit Cloud
API_KEY = st.secrets["GEMINI_API_KEY"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

# Configure Gemini
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# Connect to Aura (not local Neo4j)
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


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
    except:
        return {"client": "", "technology": "", "capability": "", "geography": "", "intent": "general_query"}

# === DOCUMENT RETRIEVAL BY CLIENT ===
def get_all_client_documents(client_name):
    query = f"""
    MATCH (c:Client)<-[:ABOUT]-(d:Document)
    WHERE toLower(c.name) CONTAINS toLower("{client_name}")
    RETURN d.file_name AS file_name, d.slide_number AS slide_number, d.text AS text
    ORDER BY d.file_name, d.slide_number
    """
    return graph.run(query).data()

# === RAG SEARCH ===
def search_rag_context(query, top_k=100):
    vec = embedder.encode([query])[0].astype("float32").reshape(1, -1)
    _, idxs = index.search(vec, top_k)
    return [metadata[i] for i in idxs[0]]

# === FIND CLIENT FOR SLIDE ===
def find_client(file, slide):
    q = f'''
    MATCH (d:Document {{file_name: "{file}", slide_number: {slide}}})-[:ABOUT]->(c:Client)
    RETURN c.name AS client_name
    '''
    res = graph.run(q).data()
    return res[0]['client_name'] if res else "CLIENT_NOT_SPECIFIED"

# === GEMINI CALL ===
def call_gemini(prompt, temperature=0.1, max_tokens=2048):
    try:
        res = llm_model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        return res.text
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return "LLM failed."

# === FORMAT CONTEXT ===
def format_context(entries, remove_warnings=True):
    result = ""
    for entry in entries:
        file = entry.get('file_name', 'N/A')
        slide = entry.get('slide_number', '?')
        text = entry.get('text', '')
        client = find_client(file, slide)

        if remove_warnings and "!!! Please remove all client logos" in text:
            continue

        for word in ["Bank", "Client", "Company"]:
            text = text.replace(word, client)

        result += f"[{file} | Slide {slide}] {text.strip()}\n"
    return result, len(entries)

# === MAIN FUNCTION TO CALL FROM STREAMLIT ===
def process_question(question, top_k=100, temperature=0.1, max_tokens=2048):
    parsed = analyze_question(question)
    logger.info(f"Entities: {parsed}")

    if parsed['intent'] == "summarize_client_relationship" and parsed['client']:
        docs = get_all_client_documents(parsed['client'])
        context, count = format_context(docs)
        if not context.strip():
            return "❌ No relevant content found."

        final_prompt = f"""
You are an expert IBM Consulting analyst writing a formal summary for leadership.

User asked to summarize the IBM relationship with client '{parsed['client']}' or provide details for any specific information.

Write a DETAILED paragraphs (not bullets) with proper flow. For each fact, include:
- [file_name | Slide number] in brackets after the sentence
- Cover all programs, technologies, milestones and any other information asked in the question
- Use only the context provided below
- DO NOT fabricate anything
- Break the paragraphs according to the information sections and relevance of information. Create meaningful paragraphs.
- Do not try to put everything in one single paragraph when information retrieved is a lot to present.

Ignore lines like:
!!! Please remove all client logos and references before sharing any case study with customer !!!

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
- Answer the question truthfully and completely.
- If it's a catalog or list, show 20 entries in the way the user has asked, then say how many more exist and ask the user to refine the search to get specific responses.
- For list type questions give proper pointers and a list of outcome not a paragraph. When more details are asked in a list or catalogue type question answer systematically.
- If no relevant information found, say: "No relevant information was found in the database."
- Use paragraph format (NO bullet points).
- Include file name and slide number after each fact, like [Slide.pptx | Slide 3].
- Not all questions may specifically use the word summarize, you need to understand if the question is a list type or a short answer type question or answer in brief or a detailed question.
- Ignore this line if found: "!!! Please remove all client logos..."
- If the summarization answers are too long and the word limits are exceeded please ask for a word limit and try to completely summarize everything in a concise manner so that no information is missed.
- focus on being very detailed about what the user has asked you to respond.

Your Answer:
"""
    return call_gemini(final_prompt, temperature, max_tokens)

