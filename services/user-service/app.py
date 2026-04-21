import os
import uuid
import httpx
import streamlit as st

CHAT_SERVICE_URL = os.environ.get("CHAT_SERVICE_URL", "http://localhost:8002")

st.set_page_config(
    page_title="Omni-Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def login(username: str, password: str) -> str | None:
    """Verifica credenciales contra el chat-service. Devuelve el token 'username:password'."""
    token = f"{username}:{password}"
    try:
        resp = httpx.get(
            f"{CHAT_SERVICE_URL}/auth/verify",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        if resp.status_code == 200:
            return token
        return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None


def chat(message: str, session_id: str, token: str) -> dict | None:
    try:
        resp = httpx.post(
            f"{CHAT_SERVICE_URL}/chat",
            json={"message": message, "session_id": session_id},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Chat error {resp.status_code}: {resp.text}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None



# ─── Session State Init ───────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Login Screen ─────────────────────────────────────────────────────────────
if st.session_state.token is None:
    st.title("🔬 Omni-Analyst")
    st.subheader("Academic Research Intelligence System")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.markdown("### Login")
            username = st.text_input("Username", value="demo")
            password = st.text_input("Password", type="password", value="demo123")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                token = login(username, password)
                if token:
                    st.session_state.token = token
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try demo/demo123 or admin/admin123")
    st.stop()

# ─── Main App ─────────────────────────────────────────────────────────────────
st.title("🔬 Omni-Analyst")
st.caption("Academic Research Intelligence System — powered by ArXiv + Azure OpenAI")

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    page = "💬 Chat"

    st.markdown("---")
    st.markdown("**Session ID**")
    st.code(st.session_state.session_id[:16] + "...", language=None)

    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.token = None
        st.session_state.messages = []
        st.rerun()

# ─── Chat Page ────────────────────────────────────────────────────────────────
if page == "💬 Chat":
    st.markdown("### Ask anything about ArXiv research papers")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("intent"):
                st.caption(f"Intent: `{msg['intent']}`")

    example_queries = [
        "What are the most influential papers on RAG?",
        "Who are the main authors working on diffusion models?",
        "Explain the attention mechanism in transformers",
        "Which institutions publish most about LLMs?",
    ]
    st.markdown("**Example queries:**")
    cols = st.columns(2)
    for i, q in enumerate(example_queries):
        if cols[i % 2].button(q, key=f"example_{i}"):
            st.session_state._pending_query = q

    user_input = st.chat_input("Ask about ArXiv papers, authors, or research concepts...")
    if hasattr(st.session_state, "_pending_query"):
        user_input = st.session_state._pending_query
        del st.session_state._pending_query

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(user_input, st.session_state.session_id, st.session_state.token)

            if response:
                st.markdown(response["answer"])
                st.caption(f"Intent: `{response.get('intent', 'unknown')}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "intent": response.get("intent", ""),
                    "cypher_used": response.get("cypher_used"),
                })
            else:
                st.error("Failed to get a response. Please try again.")

