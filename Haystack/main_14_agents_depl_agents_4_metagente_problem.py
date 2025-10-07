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


#llm = OpenAIChatGenerator(model="gpt-4o")
llm=AzureOpenAIChatGenerator(
        azure_endpoint=endpoint,
        api_key=Secret.from_token(subscription_key),
        azure_deployment=deployment
)


@tool(name="agent2")
def agent2(text: str) -> str:
    return f"[HANDOFF] {text}"

@tool(name="agent3")
def agent3(text: str) -> str:
    return f"[HANDOFF] {text}"

@tool
def noop() -> str:
    """A dummy tool that does nothing."""
    print("\033[93m[INFO] Noop tool called, doing nothing.\033[0m")
    return "noop"


agent1 = Agent(
    chat_generator=llm,
    system_prompt=AGENT1_PROMPT,
    tools=[agent2, agent3],
    exit_conditions=["agent2", "agent3"]
)

agent2 = Agent(
    chat_generator=llm,
    system_prompt=AGENT2_PROMPT,
    tools=[noop],
)

agent3 = Agent(
    chat_generator=llm,
    system_prompt=AGENT3_PROMPT,
    tools=[noop],
)


user_message = "Can you explain how transformers work?"
#user_message = "Can you translate 'Hello, how are you?' into Italian?"


result1 = agent1.run(messages=[
    ChatMessage.from_user(user_message)
])
tool_msg = result1["messages"][-1]
print("\n✅ Agent 1 output:")
print(tool_msg)
print("\n🔸 ----------------------")


if isinstance(tool_msg._content[0], ToolCallResult):
    tool_result = tool_msg._content[0]
    tool_name = tool_result.origin.tool_name
    handoff_text = tool_result.result

    print(f"\n✅ Agent 1 handoff via tool: {tool_name}")
    print(f"➡️ Payload: {handoff_text}")

    if tool_name == "agent2":
        result2 = agent2.run(messages=[ChatMessage.from_user(handoff_text)])
        print("\n✅ Agent 2 (Summarizer) reply:")
        print(result2["messages"][-1].text)

    elif tool_name == "agent3":
        result3 = agent3.run(messages=[ChatMessage.from_user(handoff_text)])
        print("\n✅ Agent 3 (Translator) reply:")
        print(result3["messages"][-1].text)

    else:
        print("\n⚠️ Tool name does not match any known agent.")
else:
    print("\n🔸 No tool call was triggered by Agent 1.")
