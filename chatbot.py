import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from agno.agent import Agent
from agno.tools.exa import ExaTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.groq import Groq as AgnoGroq
from agno.storage.sqlite import SqliteStorage
from agno.agent import Agent, AgentMemory
from agno.memory.classifier import MemoryClassifier
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.memory.manager import MemoryManager
from agno.memory.summarizer import MemorySummarizer
from agno.models.groq import Groq
from agno.storage.agent.sqlite import SqliteAgentStorage
# from agno.tools.googlesearch import GoogleSearchTools
import html
import re

# Load environment variables
load_dotenv(".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

# Initialize Groq client (for Agno)
groq_model = AgnoGroq(
    id="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)

# Initialize storage components
agent_storage = SqliteAgentStorage(table_name="study_sessions", db_file="tmp/agents.db")
memory_db = SqliteMemoryDb(
    table_name="sessions",
    db_file="tmp/agent_memory.db",
)

# Initialize agent with memory
# agent = Agent(
#     model=groq_model,
#     name="AI Assistant",
#     instructions=(
#         "You are an AI assistant. "
#         "Always maintain conversation context. "
#         "If the user refers to something with pronouns like 'it', 'this', 'that', "
#         "use the last relevant subject from the conversation history."
#     ),
#     storage=SqliteStorage(table_name="sessions", db_file="agent.db"),
#     add_history_to_messages=True,     # ✅ ensures context is preserved
#     add_datetime_to_instructions=True,
#     num_history_runs=3,
#     search_previous_sessions_history=True,  # allow searching previous sessions
#     num_history_sessions=2,
#     tools=[ExaTools()]
# )

agent = Agent(
    model=groq_model,
    reasoning_model=Groq(
        id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
    ),
    name="AI Assistant",
    instructions=(
        "You are an AI assistant. "
        "Always maintain conversation context. "
        "If the user refers to something with pronouns like 'it', 'this', 'that', "
        "use the last relevant subject from the conversation history."
    ),
    memory=AgentMemory(
            db=memory_db,
            create_user_memories=True,
            update_user_memories_after_run=True,
            classifier=MemoryClassifier(
                model=Groq(id="llama-3.3-70b-versatile"),
            ),
            summarizer=MemorySummarizer(
                model=Groq(id="llama-3.3-70b-versatile"),
            ),
            manager=MemoryManager(
                model=Groq(id="llama-3.3-70b-versatile"),
                db=memory_db,
                # user_id=user_id,
            ),
        ),
    storage=agent_storage,
    add_history_to_messages=True,     # ✅ ensures context is preserved
    add_datetime_to_instructions=True,
    num_history_runs=3,
    search_previous_sessions_history=True,  # allow searching previous sessions
    num_history_sessions=2,
    tools=[ExaTools(), DuckDuckGoTools()]
)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Agno + Groq Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #d4fc79, #96e6a1);
            font-family: 'Segoe UI', sans-serif;
        }
        .stChatMessage {
            border-radius: 20px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .user {
            background-color: #d1ecf1;
            text-align: right;
        }
        .bot {
            background-color: #f8d7da;
            text-align: left;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=100)
st.sidebar.title("Agno + Groq Chatbot 🤖")
st.sidebar.markdown("### 🧠 Chat Memory")
memory_enabled = st.sidebar.toggle("Enable Chat Memory", value=True)
st.sidebar.markdown("Built using **Agno Agents** + **Groq API**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 AI Assistant")
st.caption("Ask anything — powered by Agno agent + Groq LLM")

# Download chat history
if st.session_state.chat_history:
    chat_text = "\n\n".join(
        [f"User: {msg['content']}" if msg["role"] == "user" else f"Assistant: {msg['content']}"
         for msg in st.session_state.chat_history]
    )
    st.download_button(
        label="💾 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain",
    )

# Display chat
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='stChatMessage user'>🧑‍💻: {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage bot'>🤖: {msg['content']}</div>", unsafe_allow_html=True)


# for msg in st.session_state.chat_history:
#     safe_content = html.escape(msg["content"])  # escape special chars
#     if msg["role"] == "user":
#         st.markdown(
#             f"<div class='stChatMessage user'>🧑‍💻: {safe_content}</div>",
#             unsafe_allow_html=True
#         )
#     else:
#         st.markdown(
#             f"<div class='stChatMessage bot'>🤖: {safe_content}</div>",
#             unsafe_allow_html=True
#         )


# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="input", placeholder="Ask me anything...")
    submitted = st.form_submit_button("Send")

# Process input
if submitted and user_input:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if memory_enabled:
        # Agent uses its own memory (Agno MemoryBuffer)
        response = agent.run(user_input)
    else:
        # Stateless agent call
        stateless_agent = Agent(model=groq_model, instructions="You are an AI assistant.")
        response = stateless_agent.run(user_input)
        
    if "<function=" in response.content:
        # Execute the tool call
        tool_output = agent.execute_tool_call(response)
        bot_reply = tool_output.content  # final output
    else:
        bot_reply = response.content

    bot_reply = response.content
    bot_reply = re.sub(r"<function=.*?>.*?</function>", "", bot_reply, flags=re.DOTALL).strip()

    # Save assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

    st.rerun()
