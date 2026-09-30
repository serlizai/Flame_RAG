from backend.retrieval import agent_invoke

answer, citations, session_id = agent_invoke(
    "What does Article 21 of the Indian Constitution say?"
)

print("回答：")
print(answer)

print("\n引用：")
print(citations)

print("\nsession_id：")
print(session_id)