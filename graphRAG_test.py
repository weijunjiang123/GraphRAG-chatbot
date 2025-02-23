from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.ollama import Ollama

def chatbot(query: str):
    llm = Ollama(model="llama3.2", request_timeout=360.0)
    messages = [
        ChatMessage(
            role="system",
            content="I want you to act as an AI writing tutor. "
                    "I will provide you with a student who "
                    "needs help improving their writing and "
                    "your task is to use artificial intelligence "
                    "tools, such as natural language processing, "
                    "to give the student feedback on how they can "
                    "improve their composition. You should also use "
                    "your rhetorical knowledge and experience about "
                    "effective writing techniques in order to suggest "
                    "ways that the student can better express their "
                    "thoughts and ideas in written form. My first "
                    "request is \"I need somebody to help me "
                    "edit my master's thesis.\"",
        ),
        ChatMessage(role="user", content=query),
    ]

    response = llm.stream_chat(messages)
    for resp in response:
        yield resp.content