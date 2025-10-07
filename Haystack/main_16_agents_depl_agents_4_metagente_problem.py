from haystack.tools import tool, Toolset
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.components.tools import ToolInvoker

import json
import uuid

# Init LLM globale
llm = OpenAIChatGenerator(model="gpt-4o")

# Prompts
AGENT1_PROMPT = """
You are an assistant that decides which agent should take care of the task.

- If the user input is about translating italian, call the `agent2`.
- If the user input is about explaining jazz music, call the `agent3`.
- If the user input is about any other topic, invoke the tool 'agent_builder' to create a new agent that can handle the task.
After invoking agent_builder , check the toolset to discover available new tools and try to solve the quastion, otherwise exit and return the string "END"
Always use a tool. Never answer directly.
"""

AGENT2_PROMPT = "You are an assistant that translates English sentences into Italian."
AGENT3_PROMPT = "You are an assistant that explains technical concepts in simple language."

AGENT_BUILDER_PROMPT = """
You are an agent builder. When invoked, you need to create a new agent that can handle the task described in the user input.
Respond with a JSON object with two keys:
- 'name': the name of the new tool to be created
- 'prompt': the system prompt of the new tool
Example:
{
  "name": "agent_weather",
  "prompt": "You are a weather assistant that answers weather-related questions."
}
"""

# Toolset globale (usato da tutti)


@tool(name="agent2")
def agent2(text: str) -> str:
    print("\033[94m[INFO] agent2 invoked.\033[0m")
    response = llm.run(messages=[
        ChatMessage.from_system(AGENT2_PROMPT),
        ChatMessage.from_user(text)
    ])
    reply = response["replies"][-1].text
    print("\033[92m[agent2 reply]\033[0m", reply)
    return reply

@tool(name="agent3")
def agent3(text: str) -> str:
    print("\033[94m[INFO] agent3 invoked.\033[0m")
    response = llm.run(messages=[
        ChatMessage.from_system(AGENT3_PROMPT),
        ChatMessage.from_user(text)
    ])
    reply = response["replies"][-1].text
    print("\033[92m[agent3 reply]\033[0m", reply)
    return reply

@tool(name="agent_builder")
def agent_builder(input_text: str) -> str:
    # Step 1 - Chiedi JSON con nome/prompt
    resp = llm.run(messages=[
        ChatMessage.from_system(AGENT_BUILDER_PROMPT),
        ChatMessage.from_user(input_text)
    ])
    output = resp["replies"][-1].text
    print(f"\033[96m[BUILDER OUTPUT]\033[0m\n{output}\n")

    try:
        data = json.loads(output)
        base_name = data.get("name", "agent_dynamic")
        prompt = data["prompt"]

        # Evita nomi duplicati
        existing_names = {tool.name for tool in toolset.tools}
        new_tool_name = base_name
        i = 1
        while new_tool_name in existing_names:
            new_tool_name = f"{base_name}_{i}"
            i += 1
    except Exception as e:
        return f"⚠️ Failed to parse tool definition: {str(e)}"

    # Step 2 - Crea tool dinamico
    @tool(name=new_tool_name)
    def dynamic_tool(input_text: str) -> str:
        print(f"\033[94m[INFO] {new_tool_name} invoked.\033[0m")
        response = llm.run(messages=[
            ChatMessage.from_system(prompt),
            ChatMessage.from_user(input_text)
        ])
        return response["replies"][-1].text

    toolset.tools.append(dynamic_tool)
    agent1.tools = toolset
    print(f"\033[92m✅ Tool '{new_tool_name}' aggiunto al Toolset.\033[0m")


toolset = Toolset([agent2, agent3, agent_builder])
# Agente iniziale
agent1 = Agent(
    chat_generator=llm,
    system_prompt=AGENT1_PROMPT,
    tools=toolset,
    exit_conditions=["agent2", "agent3"]
)

agent1.warm_up()
# Messaggio da testare
user_message = "Can you calculate the sum of 123 and 456?"

print("\n\033[93m[INFO] Running Agent 1...\033[0m")
result = agent1.run(messages=[ChatMessage.from_user(user_message)])
tool_msg = result["messages"][-1]

print("\n✅ Agent 1 final output:")
print(tool_msg.text)
