"""验证商品身份、模板插值和提示词版本对应的缓存隔离。"""

import pytest

from backend import prompts as backend_prompts
import prompts


@pytest.mark.parametrize(
    "name, product_id",
    [
        ("vivo X500", "vivo_x500"),
        ("vivo X500 Pro", "vivo_x500_pro"),
        ("vivo X500 Pro Max", "vivo_x500_pro_max"),
    ],
)
def test_system_prompt_uses_selected_product(name, product_id):
    result = prompts.build_system_prompt(name, product_id)
    assert f"当前商品：{name}（product_id={product_id}）" in result
    assert "{product_context}" not in result
    assert "当前未指定商品" not in result


def test_unselected_product_does_not_default_to_a_model():
    result = prompts.build_system_prompt()
    assert "当前未指定商品" in result
    assert all(name not in result for name in ("vivo X500", "vivo_x500_pro"))


def test_template_preserves_braces_in_question_and_document_content():
    question = "请解释 {配置参数}"
    context = '[6] 商品原文：{"参数": "资料中的值"}'
    rendered = prompts.QA_PROMPT.format(context=context, input=question)
    assert question in rendered and context in rendered
    system = prompts.build_system_prompt("商品 {测试版}", "demo_product")
    assert "商品 {测试版}" in system


def test_backend_reexports_the_same_prompt_builder():
    assert backend_prompts.build_system_prompt is prompts.build_system_prompt
    assert backend_prompts.QA_PROMPT == prompts.QA_PROMPT
