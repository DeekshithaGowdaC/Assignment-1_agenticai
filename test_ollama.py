from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

response = llm.invoke("Explain personal finance in 3 simple points.")

print(response.content)