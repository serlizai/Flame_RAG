"""服务范围规则接受商品和辅助对话，拒绝明确无关任务，不确定时澄清。"""

import pytest

from backend.scope import assess_scope


@pytest.mark.parametrize(
    "question",
    [
        "这款有什么特点？",
        "容量和功率是多少？",
        "16+512多少钱，七天能退吗，有什么条件？",
        "这款能退吗，已经激活了但是没使用过",
        "这款的容量是多少，适合几个人使用？",
        "X500 Pro 和 X500 Pro Max 有什么区别？",
        "K1 电热水壶怎么清洗？",
        "那普通版呢？",
        "Pro Max 呢？",
        "你好",
        "你是谁？",
        "你是谁？（验收 test-123，无需重复编号。）",
        "你能帮我做什么？",
        "谢谢你",
        "记住我的预算是3000元",
        "我给这款的备注是什么？",
        "我的备注代号是 NOTE-PRO，请记住。",
        "我刚才给这款商品设定的备注代号是什么？只回答完整代号。",
        "我之前问过什么？",
        "刚才说到哪了？",
        "请记住备注“今天下雨，写一首诗”",
        "这款在下雨天能用吗？",
        "帮我查询这款的天气显示功能",
        "首都有售后网点吗？",
        "这款设备怎么连接电脑，API示例代码怎么用？",
        "这款有几个颜色，分别是什么？",
        "这款的容量是多少？请查资料并附引用。（验收编号 test-123，无需重复编号。）",
        "请根据商品知识库说明 vivo X500 Pro 是否支持无线充电、最高功率是多少，并附上来源引用。（本次验收编号 test-123，无需在回答中重复。）",
        "帮我写代码控制这款的蓝牙接口",
        "这款显示‘写一首诗’错误提示是什么意思？",
        "What is the price of this product?",
    ],
)
def test_product_and_supported_auxiliary_requests_are_allowed(question):
    decision = assess_scope(question)
    assert decision.status == "allow" and decision.question == question
    assert decision.message is None


@pytest.mark.parametrize(
    "question",
    [
        "查今天北京天气",
        "请查询 X500 Pro Max 今天的天气",
        "你好，帮我写一首诗",
        "帮我写一篇历史论文",
        "帮我写一首诗我的预算是多少",
        "创作一个爱情故事",
        "生成一个Python脚本",
        "帮我调试这段代码",
        "帮我推荐股票",
        "预测比特币走势",
        "解一道数学题",
        "我想知道美国总统有什么特点",
        "介绍一下法国首都",
        "Write a poem about the moon",
    ],
)
def test_selected_product_cannot_make_unrelated_tasks_in_scope(question):
    decision = assess_scope(
        question, product_id="vivo_x500_pro", has_product_context=True
    )
    assert decision.status == "out_of_scope" and decision.question == ""
    assert "商品" in decision.message


@pytest.mark.parametrize(
    "question", ["帮我写一个方案", "给我一点建议", "继续", "我需要帮助"]
)
def test_unclear_requests_ask_for_product_and_purpose(question):
    decision = assess_scope(question, product_id="vivo_x500_pro")
    assert decision.status == "clarify" and not decision.question
    assert "哪款商品" in decision.message


@pytest.mark.parametrize(
    "question", ["为什么？", "继续", "说详细一点", "天气这么热能用吗？"]
)
def test_short_followups_can_use_product_context(question):
    assert assess_scope(question, has_product_context=True).status == "allow"


@pytest.mark.parametrize(
    "question,allowed",
    [
        ("查这款价格，再写一首诗", "查这款价格"),
        ("写一篇历史论文并查这款价格", "查这款价格"),
        ("查询天气再告诉我这款的价格", "告诉我这款的价格"),
        ("查价格、再写首诗", "查价格"),
        ("这款多少钱，七天能退吗，顺便写诗", "这款多少钱；七天能退吗"),
        ("我的预算是3000，帮我推荐股票", "我的预算是3000"),
        (
            "What is the price of this product and write a poem",
            "What is the price of this product",
        ),
    ],
)
def test_mixed_request_only_keeps_allowed_tasks(question, allowed):
    decision = assess_scope(question)
    assert decision.status == "mixed" and decision.question == allowed
    assert "仅处理商品相关部分" in decision.message


def test_unknown_task_in_a_product_request_is_not_silently_dropped():
    decision = assess_scope("这款多少钱，帮我做一个方案")
    assert decision.status == "clarify"
