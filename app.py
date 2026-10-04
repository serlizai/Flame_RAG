"""提供 Flame 的 Streamlit 聊天界面。"""

import streamlit as st
import time
import uuid
from dotenv import load_dotenv

from backend.retrieval import agent_invoke
from backend.config import cache as backend_cache
from langchain_core.messages import HumanMessage

# 设置页面标题、图标和布局
st.set_page_config(page_title="Flame", page_icon="🔥", layout="wide")

# 加载环境变量
load_dotenv()


# 为界面添加自定义样式
def add_custom_css():
    custom_css = """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .st-chat-input {
            border-radius: 15px; padding: 10px;
            border: 1px solid #ddd; margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stButton > button {
            background-color: #0066cc; color: white;
            font-size: 16px; border-radius: 20px;
            padding: 10px 20px; margin-top: 5px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover { background-color: #0052a3; }
        .st-chat-message-assistant {
            background-color: #f7f7f7; border-radius: 15px;
            padding: 15px; margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .st-chat-message-user {
            background-color: #d9f0ff; border-radius: 15px;
            padding: 15px; margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-input-container {
            position: fixed; bottom: 0; width: 100%;
            background-color: #f0f0f0; padding: 20px;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            display: flex; gap: 10px;
        }
        .chat-input { flex-grow: 1; }
        .st-sidebar {
            background-color: #f9f9f9; padding: 20px;
        }
        .st-sidebar header {
            font-size: 20px; font-weight: bold; margin-bottom: 10px;
        }
        .st-sidebar p {
            font-size: 14px; color: #666;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


add_custom_css()


def render_citations(citations):
    for citation in citations:
        with st.expander(f"[{citation.number}] {citation.label}"):
            st.text(citation.snippet)
            if citation.source:
                st.markdown(f"[Source]({citation.source})")
    if citations:
        st.caption(
            "Bracketed citations are verified against retrieved passages; "
            "other references mentioned in the answer text are not."
        )


st.title("Flame 🔥")

# 展示侧边栏项目信息
st.sidebar.header("About Flame")
st.sidebar.markdown("""
**Flame** is an open-source AI assistant that answers questions using your documents.

_Disclaimer_: This tool is in its pilot phase, and responses may not be 100% accurate.
""")

# 在界面会话中保留同一个会话编号
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

thread_id = st.session_state.thread_id

# 从后端获取聊天历史并转换为界面消息
if "messages" not in st.session_state:
    st.session_state.messages = []

    history = backend_cache.get_chat_history(thread_id).messages
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})

# 展示已有聊天记录
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])
        render_citations(message.get("citations", []))

# 接收用户输入的问题
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
prompt = st.chat_input("Ask a question about your documents.")
st.markdown("</div>", unsafe_allow_html=True)

if prompt and prompt.strip():
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用 Agent 后端处理问题
    result, citations, thread_id = agent_invoke(prompt, session_id=thread_id)

    # 根据更新后的 Redis 聊天历史重建界面消息
    updated_history = backend_cache.get_chat_history(thread_id).messages
    st.session_state.messages = []
    for msg in updated_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})
    if st.session_state.messages and st.session_state.messages[-1]["content"] == result:
        st.session_state.messages[-1]["citations"] = citations

    # 逐词展示模型回答
    final_response = f"Flame: {result}"

    def response_generator(response):
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    with st.chat_message("assistant"):
        animated = "".join(list(response_generator(final_response)))
        st.markdown(animated)
        render_citations(citations)
