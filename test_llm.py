from agent import llm


response = llm.invoke(
    "Reply with exactly: Grounded Research Agent is online."
)

print(response.content)