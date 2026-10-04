"""构建支持历史问题改写和引用标注的经典 RAG 链路。"""

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda

from citations import annotate_documents_for_citation


def _annotate_only(documents):
    """把引用标注函数适配为仅返回文档列表的检索步骤。

    返回值作为 response['context']。下游基于同一组文档 metadata
    重新构造 Citation 列表，因此引用编号与此处写入的 citation_marker 一致。
    """
    annotated, _ = annotate_documents_for_citation(documents)
    return annotated


def get_rag_chain(llm, vector_store, system_prompt, qa_prompt):
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.3},
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    annotating_retriever = history_aware_retriever | RunnableLambda(_annotate_only)
    qa_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", qa_prompt),
        ]
    )
    document_prompt = PromptTemplate.from_template("{citation_marker}\n{page_content}")
    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt_template, document_prompt=document_prompt
    )
    rag_chain = create_retrieval_chain(annotating_retriever, question_answer_chain)
    return rag_chain
