"""提供 Flame 的 Streamlit 聊天界面。"""

import streamlit as st
import uuid
from dotenv import load_dotenv

from backend.retrieval import agent_invoke
from backend.config import cache as backend_cache
from backend.products import load_products
from backend.tracing import TOOL_LABELS, routing_matches
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


def render_diagnostics(message, show_trace):
    expected = message.get("expected_tools")
    if not show_trace and expected is None and not message.get("expected_product_id"):
        return
    trace = message.get("trace")
    if trace is None:
        st.caption("这条历史消息没有工具过程记录。")
        return
    if message.get("product_name"):
        st.caption("本轮商品：" + message["product_name"])
    if trace.get("steps"):
        st.caption("实际执行流程")
        st.code(" → ".join(trace["steps"]), language="text")
    if trace.get("context_source"):
        st.caption(
            f"会话：{trace['session_id']} · 当前商品已有 {trace.get('history_turns', 0)} 轮历史"
        )
        if trace.get("original_question") != trace.get("resolved_question"):
            st.info("承接上轮话题后的实际问题：" + trace["resolved_question"])
    expected_product = message.get("expected_product_id")
    if expected_product:
        if (
            trace.get("product_id") == expected_product
            and not trace.get("clarification")
            and all(
                call["arguments"].get("product_id") == expected_product
                for call in trace.get("tools", [])
            )
        ):
            st.success("本轮商品上下文符合预期")
        else:
            st.warning("本轮商品上下文未达到演示预期，请查看解析结果。")
    if message.get("expected_memory"):
        if (
            message["expected_memory"] in message["content"]
            and trace.get("history_turns", 0) > 0
        ):
            st.success("已恢复 Pro 独立对话中的备注记忆")
        else:
            st.warning("尚未恢复预期备注，请检查是否沿用了同一会话。")
    if trace.get("clarification"):
        st.info("需要补充商品或话题，本轮未调用模型和工具。")
        return
    if expected is not None:
        if trace.get("cache_hit"):
            st.info("本轮命中缓存，没有执行工具，不计入路由验收。")
        elif routing_matches(trace, expected):
            st.success("工具路由符合预期")
        else:
            st.warning("工具路由与预期不一致，请检查下方实际调用过程。")
        st.caption("预期工具：" + "、".join(expected))
    if trace.get("cache_hit"):
        st.caption("回答来源：共享答案缓存；本轮工具调用 0 次。")
        return
    calls = trace.get("tools", [])
    if not calls:
        st.caption("本轮未调用工具，可能为问候、聊天记忆或复用已有资料。")
        return
    with st.expander(
        f"实际工具调用过程（{len(calls)} 次）", expanded=expected is not None
    ):
        for i, call in enumerate(calls, 1):
            label = TOOL_LABELS.get(call["name"], call["name"])
            with st.expander(f"{i}. {label} · {call['name']}"):
                st.caption("实际传入参数")
                st.json(call["arguments"])
                st.caption("工具返回结果")
                st.text(call["result"])
                if call["sources"]:
                    st.caption("数据来源：" + "、".join(call["sources"]))


VALIDATION_SCENARIOS = {
    "1. 商品卖点 → 知识检索": ("这款有什么卖点？", ["product_rag_tool"]),
    "2. 商品颜色 → 商品信息": ("这款有几个颜色，分别是什么？", ["product_info_tool"]),
    "3. 商品报价 → 价格工具": ("这款多少钱？", ["price_tool"]),
    "4. 七天退货 → 售后工具": ("七天能退吗，有什么条件？", ["after_sales_tool"]),
    "5. 混合问题 → 三个业务工具": (
        "这款有哪些颜色、多少钱、七天能退吗？",
        ["product_info_tool", "price_tool", "after_sales_tool"],
    ),
    "6. 不存在的规格 → 不乱报价": ("12+512 这个配置是什么价格？", ["price_tool"]),
}

CONVERSATION_DEMO = [
    (
        "请记住这款商品的备注代号是 WEEK7-PRO。X500 Pro 支持无线充电吗，最高功率是多少？",
        "vivo_x500_pro",
    ),
    ("那普通版呢？", "vivo_x500"),
    ("Pro Max 呢？", "vivo_x500_pro_max"),
    ("回到 X500 Pro：我给这款商品的备注代号是什么？", "vivo_x500_pro"),
]


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

try:
    products = load_products()
except ValueError:
    st.error("商品表无法读取，请检查 data/products.json。")
    st.stop()
product_options = [None, *products]
if "pending_product_selection" in st.session_state:
    st.session_state.selected_product = st.session_state.pop(
        "pending_product_selection"
    )
default_product = "vivo_x500_pro" if "vivo_x500_pro" in products else None
if "selected_product" not in st.session_state:
    st.session_state.selected_product = default_product
product_id = st.sidebar.selectbox(
    "当前商品",
    product_options,
    index=None,
    format_func=lambda item: products[item].name if item else "未选择商品（全库问答）",
    key="selected_product",
)
show_trace = st.sidebar.checkbox("显示工具调用过程", value=True, key="show_tool_trace")
auto_follow = st.sidebar.checkbox("根据追问自动切换商品", value=True, key="auto_follow")
if st.sidebar.button("新建对话", key="new_chat"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.week7_demo_step = 0
    st.rerun()
with st.sidebar.expander("对话工具", expanded=True):
    scenario = st.selectbox(
        "验证场景", list(VALIDATION_SCENARIOS), key="validation_scenario"
    )
    sample_question, expected_tools = VALIDATION_SCENARIOS[scenario]
    st.caption("预期工具：" + "、".join(expected_tools))
    st.caption("每次验证开启独立会话并绕过答案缓存，实际调用模型。")
    run_validation = st.button("发送验证问题", key="run_validation", type="primary")
    st.caption("路由符合预期只表示工具选对；还应核对回答、引用和工具结果。")
    coverage_display = st.empty()

with st.sidebar.expander("多轮上下文", expanded=True):
    st.caption("Pro 无线充 → 那普通版呢 → Pro Max 呢 → 切回 Pro 恢复备注。")
    st.caption("同一会话连续演示，自动承接话题；各商品的答案和备注仍独立保存。")
    if st.button("开始多轮演示", key="week7_start"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.week7_demo_step = 0
        st.session_state.pending_product_selection = "vivo_x500_pro"
        st.session_state.pending_week7_step = 0
        st.rerun()
    completed = st.session_state.get("week7_demo_step", 0)
    st.caption(f"已完成演示：{completed}/4 轮")
    advance_demo = False
    if 0 < completed < len(CONVERSATION_DEMO):
        advance_demo = st.button(
            "继续：" + CONVERSATION_DEMO[completed][0], key="week7_next"
        )
    st.caption(
        "演示绕过共享答案缓存。展开回答下方流程，可核对实际商品、补全问题和历史轮数。"
    )

# 从后端获取聊天历史并转换为界面消息
if "messages" not in st.session_state:
    st.session_state.messages = []

    history = backend_cache.get_chat_history(thread_id).messages
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})


def render_coverage():
    observed = {
        call["name"]
        for message in st.session_state.messages
        if message.get("expected_tools") is not None
        and routing_matches(message.get("trace", {}), message["expected_tools"])
        for call in message.get("trace", {}).get("tools", [])
    }
    coverage_display.metric(
        "已通过路由验证的工具类型", f"{len(observed & TOOL_LABELS.keys())}/4"
    )


render_coverage()

# 展示已有聊天记录
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])
        render_citations(message.get("citations", []))
        if role == "assistant":
            render_diagnostics(message, show_trace)

# 接收用户输入的问题
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
prompt = st.chat_input("请输入商品问题或追问，例如：那普通版呢？")
st.markdown("</div>", unsafe_allow_html=True)

if run_validation:
    st.session_state.thread_id = str(uuid.uuid4())
    thread_id = st.session_state.thread_id
    prompt = sample_question
    if product_id is None:
        prompt = "关于 vivo X500 Pro：" + prompt

week7_step = st.session_state.pop("pending_week7_step", None)
if advance_demo:
    week7_step = completed
if week7_step is not None:
    prompt = CONVERSATION_DEMO[week7_step][0]

if prompt and prompt.strip():
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    trace = {}
    try:
        with st.spinner("正在调用 Agent，完成后可查看真实工具过程…"):
            result, citations, thread_id = agent_invoke(
                prompt,
                session_id=thread_id,
                product_id=product_id,
                trace=trace,
                use_cache=not run_validation and week7_step is None,
                follow_conversation=auto_follow or week7_step is not None,
            )
    except Exception as error:
        st.error(
            f"本次请求未完成（{type(error).__name__}），请检查聊天服务和 Redis 后重试。"
        )
    else:
        resolved_product = trace.get("product_id", product_id)
        message = {
            "role": "assistant",
            "content": result,
            "citations": citations,
            "trace": trace,
            "product_name": products[resolved_product].name
            if resolved_product in products
            else "全库问答",
            "expected_tools": expected_tools if run_validation else None,
            "expected_product_id": CONVERSATION_DEMO[week7_step][1]
            if week7_step is not None
            else None,
            "expected_memory": "WEEK7-PRO" if week7_step == 3 else None,
        }
        st.session_state.messages.append(message)
        with st.chat_message("assistant"):
            st.markdown(result)
            render_citations(citations)
            render_diagnostics(message, show_trace)
        render_coverage()
        if week7_step is not None:
            st.session_state.week7_demo_step = week7_step + 1
        if resolved_product != product_id:
            st.session_state.pending_product_selection = resolved_product
        if week7_step is not None or resolved_product != product_id:
            st.rerun()
