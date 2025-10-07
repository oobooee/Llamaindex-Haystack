from haystack.tools import tool
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator, AzureOpenAIChatGenerator 
from haystack.dataclasses import ToolCallResult
from haystack.components.agents import Agent
import json
from haystack.utils import Secret

subscription_key = ""
endpoint = ""
model_name = "gpt-4o"
deployment = "gpt-4o"

# Prompt dell’agente iniziale che decide cosa fare
AGENT1_PROMPT = """
You are an assistant that decides which agent should take care of the task.

- If the user input is about translating something, call the `agent2`.
- If the user input is about explaining a concept, call the `agent3`.

Always use a tool. Never answer directly.
"""

AGENT2_PROMPT = "You are an assistant that translates English sentences into Italian."
AGENT3_PROMPT = "You are an assistant that explains technical concepts in simple language."

@tool(name="agent2")
def agent2(text: str) -> str:
    """Tool that translates a sentence to Italian using LLM."""
    response = llm.run(messages=[
        ChatMessage.from_system(AGENT2_PROMPT),
        ChatMessage.from_user(text)
    ])
    print("\033[92m[INFO] Agent 2 response received.\033[0m")
    print("\033[93m[DEBUG] Agent 2 response:\033[0m", response["replies"][-1].text)
    return response["replies"][-1].text

@tool(name="agent3")
def agent3(text: str) -> str:
    """Tool that explains a concept using LLM."""
    response = llm.run(messages=[
        ChatMessage.from_system(AGENT3_PROMPT),
        ChatMessage.from_user(text)
    ])
    print("\033[92m[INFO] Agent 3 response received.\033[0m")
    print("\033[93m[DEBUG] Agent 3 response:\033[0m", response["replies"][-1].text)
    return response["replies"][-1].text



#llm = OpenAIChatGenerator(model="gpt-4o")
llm=AzureOpenAIChatGenerator(
        azure_endpoint=endpoint,
        api_key=Secret.from_token(subscription_key),
        azure_deployment=deployment
)
agent1 = Agent(
    chat_generator=llm,
    tools=[agent2, agent3],
    exit_conditions=["agent2", "agent3"]
)


#user_message = "Can you explain in 5 rows how transformers work?"
user_message = "Can you translate 'Hello, how are you?' into Italian?"


result1 = agent1.run(messages=[
    system_prompt := ChatMessage.from_system(AGENT1_PROMPT),
    ChatMessage.from_user(user_message)
])
print("\n\033[92m[INFO] Result: \033[0m", result1)
print("\n\033[93m[------------------------]\033[0m")
tool_msg = result1["messages"][-1]

print("\n\033[96m[DEBUG] Tool call intercepted:\033[0m")

# Verifica che sia un tool message e che _content[0] sia ToolCallResult
if tool_msg.role == "tool" and hasattr(tool_msg, "_content") and isinstance(tool_msg._content[0], ToolCallResult):
    tool_result: ToolCallResult = tool_msg._content[0]
    print(f"Tool name: {tool_result.origin.tool_name}")
    print(f"Tool input: {tool_result.origin.arguments}")
    print("\n✅ Agent 1 output:")
    print(tool_result.result)
else:
    print("❌ No tool call result found.")
    print("Message role:", tool_msg.role)
    print("Message available attributes:", dir(tool_msg))
