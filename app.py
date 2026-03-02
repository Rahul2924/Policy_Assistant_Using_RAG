import os
import logging
import uuid
import streamlit as st
from dotenv import load_dotenv
from models.llm import call_llm
from models.embeddings import get_embeddings, get_embedding_dimension
from utils.rag_utils import FAISSStore, build_or_update_index_from_documents
# from utils.web_search import serpapi_search, summarize_web_results
from utils.pdf_utils import extract_text_from_pdf   # ✅ NEW
from config.config import TOP_K

load_dotenv()

# --- Streamlit page config ---
st.set_page_config(
    page_title="Policy Assistant (RAG + Groq)",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize FAISS in session_state so it persists ---
EMBED_DIM = get_embedding_dimension() or 384

if "store" not in st.session_state:
    st.session_state.store = FAISSStore(dim=EMBED_DIM)

store: FAISSStore = st.session_state.store


# --- AUTO LOAD POLICY DOCUMENTS ---
POLICY_FOLDER = "data/policies"

if "kb_loaded" not in st.session_state:
    docs = []

    if os.path.exists(POLICY_FOLDER):
        for filename in os.listdir(POLICY_FOLDER):
            path = os.path.join(POLICY_FOLDER, filename)
            text = ""

            # TXT / MD
            if filename.endswith((".txt", ".md")):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            # PDF SUPPORT ✅
            elif filename.endswith(".pdf"):
                try:
                    text = extract_text_from_pdf(path)
                except Exception as e:
                    logging.exception(f"PDF read failed: {filename} | {e}")

            else:
                continue

            if text.strip():
                docs.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "source": filename
                })

    if docs:
        build_or_update_index_from_documents(docs, store)
        st.session_state.kb_loaded = True


# --- Title and subtitle ---
st.markdown(
    """
    <h1 style="margin-bottom:0.2rem;">📘 Company Policy Assistant</h1>
    <p style="color:gray;margin-top:0;">
        Ask questions about company policies. Answers come only from official policy documents.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# --- Layout: left = chat, right = settings/docs ---
left_col, right_col = st.columns([2, 1])

# ========== RIGHT COLUMN: SETTINGS & POLICY INFO ==========
with right_col:
    st.subheader("⚙️ Settings")

    response_mode = st.radio(
        "Response mode",
        ("Concise", "Detailed"),
        help="Choose how detailed the answer should be.",
    )

    use_web = st.checkbox("Allow live web search (optional)", value=False)
    top_k = st.number_input("Top-k retrieved chunks", min_value=1, max_value=10, value=TOP_K)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2)

    st.markdown("---")

    st.subheader("📂 Policy Knowledge Base")

    sources = store.list_sources()
    if sources:
        st.markdown("**Loaded policy documents:**")
        for src, count in sources:
            st.markdown(f"- `{src}` _(chunks: {count})_")
    else:
        st.warning("No policy documents found in `data/policies/`")

    if st.button("🧹 Clear knowledge base"):
        store.clear()
        st.success("Knowledge base cleared. Restart app to reload policies.")

    st.markdown("---")

    if st.button("🧼 Clear chat history"):
        st.session_state.history = []
        st.success("Chat history cleared.")


# ========== LEFT COLUMN: CHAT INTERFACE ==========
with left_col:
    if "history" not in st.session_state:
        st.session_state.history = []

    user_query = st.text_input(
        "Ask something about company policies:",
        key="query",
        placeholder="e.g., What is the leave policy?",
    )

    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and user_query:
        st.session_state.history.append({"role": "user", "content": user_query})

        # 1) RAG retrieval
        q_emb_list = get_embeddings([user_query])
        if q_emb_list:
            q_emb = q_emb_list[0]
            retrieved = store.query(q_emb, k=top_k)
            st.caption(f"🔎 Retrieved {len(retrieved)} policy chunk(s).")
        else:
            retrieved = []

        context_text = ""
        if retrieved:
            context_text += "Policy context:\n"
            for meta, dist in retrieved:
                context_text += (
                    f"- Source: {meta.get('source')} | chunk_index: {meta.get('chunk_index')}\n"
                    f"{meta.get('text')}\n\n"
                )

        # 2) Optional web search
        # web_summary = ""
        # if use_web:
        #     try:
        #         web_hits = serpapi_search(user_query, num_results=4)
        #         if web_hits:
        #             web_summary = summarize_web_results(web_hits, call_llm)
        #     except Exception as e:
        #         logging.exception("Web search failed: %s", e)
        #         web_summary = ""

        # 3) Response mode instructions
        mode_instructions = {
            "Concise": (
                "Give a short, direct answer based ONLY on company policy. "
                "If not found, say: 'This is not covered in company policy.'"
            ),
            "Detailed": (
                "Give a detailed answer based ONLY on company policy. "
                "If policy does not contain the answer, say: 'This is not covered in company policy.' "
                "End with 3 bullet-point summary notes."
            ),
        }

        prompt = f"""
You are a STRICT COMPANY POLICY CHATBOT.
Answer ONLY using the policy document context below.
If the answer is not present, reply exactly:
"This is not covered in company policy."

User question:
{user_query}

Response style:
{mode_instructions[response_mode]}

Policy context:
{context_text}



Provide the final answer now.
""".strip()

        answer = call_llm(prompt, max_tokens=600, temperature=temperature)
        st.session_state.history.append({"role": "assistant", "content": answer})

    st.markdown("### 💬 Conversation")

    if not st.session_state.history:
        st.info("Ask a company policy question to begin.")
    else:
        for turn in st.session_state.history:
            if turn["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(turn["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(turn["content"])
