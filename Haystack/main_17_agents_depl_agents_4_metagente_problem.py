import json
from haystack.tools import tool, Toolset
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.utils import Secret

# === CONFIG AZURE LLM ===
subscription_key = ""
endpoint = ""
deployment = "gpt-4o"

llm = AzureOpenAIChatGenerator(
    azure_endpoint=endpoint,
    api_key=Secret.from_token(subscription_key),
    azure_deployment=deployment
)



@tool(name="agent_builder")
def agent_builder(input_text: str) -> str:
    try:
        data = json.loads(input_text)
        base_name = data.get("name", "agent_dynamic")
        prompt = data["prompt"]
        print(f"\033[94m[INFO] Creating tool '{base_name}' with prompt:\033[0m\n{prompt}")
        existing_names = {tool.name for tool in toolset.tools}
        new_tool_name = base_name
        i = 1
        while new_tool_name in existing_names:
            new_tool_name = f"{base_name}_{i}"
            i += 1

        @tool(name=new_tool_name)
        def dynamic_tool(user_input: str) -> str:
            print(f"\n\033[92m[INFO] Tool '{new_tool_name}' INVOCATO\033[0m")
            print(f"\033[96m[INPUT]\033[0m {user_input}")
            result = llm.run(messages=[
                ChatMessage.from_system(prompt),
                ChatMessage.from_user(user_input)
            ])

            # ✅ Gestione robusta di ChatResult o dict
            try:
                reply = result.replies[-1]
            except AttributeError:
                reply = result["replies"][-1]


            if hasattr(reply, "text"):
                return reply.text
            elif isinstance(reply, dict) and "text" in reply:
                return reply["text"]
            else:
                return str(reply)


        toolset.tools.append(dynamic_tool)
        return f"✅ Tool '{new_tool_name}' creato con prompt:\n{prompt}"

    except Exception as e:
        return f"❌ Errore nella creazione del tool: {str(e)}"

# === AGGIUNGI agent_builder al toolset ===

toolset = Toolset([agent_builder])

# === PROMPT AGENTE ORCHESTRATORE ===
AGENT1_PROMPT = """
You are an orchestrator that builds new agents by calling the `agent_builder` tool.

For each agent, prepare a JSON object with keys:
- "name": a short, unique identifier
- "prompt": the system prompt for the agent

You must invoke the tool like this:
agent_builder(input_text="{...json string...}")

The input must be passed as a **single string** using the `input_text` parameter.
Do not include markdown, explanations, or quotes around the function.

The prompt must be extremely clear and long enough to let the agent to be specialized to the tash he will have to perform.
Example:
agent_builder(input_text="{\"name\": \"code_writer\", \"prompt\": \"You write Python code from a task.\"}")

Generate one tool call per line. Only invoke the tool.
"""


# === DEFINISCI L'AGENTE 1 ===
agent1 = Agent(
    chat_generator=llm,
    system_prompt=AGENT1_PROMPT,
    tools=toolset
)
# === INPUT UTENTE ===
user_input = """
I want to build a multi-agent system for software development. You have to create agents that can work together to complete tasks.
It should include:
- one agent that writes Python code based on a natural language task
- another that writes unit tests for that code
- a third that explains the code line-by-line to beginners
"""
agent1.warm_up()
# === ESECUZIONE ===
resp = agent1.run(messages=[
    ChatMessage.from_user(user_input)
])
print (resp)
output = resp["messages"][-1].text.strip()

print(f"\n🧩 Agent1 decomposition:\n{output}\n")
print("\n📦 Toolset finale:")
for tool in toolset.tools:
    print(f" - {tool.name}")

print("\n📄 🔍 Dettagli dei tool creati:\n")

AGENT2_PROMPT = """
You are a tool inspector.

When the user asks you to inspect or verify the system, you must call all tools available in your toolset and print:

- The name of each tool
- If it is a static tool (like agent_builder) or a dynamically created one
- Its prompt (if available)

Format your response as clean readable text. Always use the tools directly — do not summarize from memory. Do not guess.

Use the tools one by one to get their details.
"""
agent2 = Agent(
    chat_generator=llm,
    system_prompt=AGENT2_PROMPT,
    tools=toolset  # contiene tutti i tool dinamici creati
)
agent2.warm_up()

response = agent2.run(messages=[
    ChatMessage.from_user("List and describe all the tools available in this system.")
])
print("\n📋 Risposta di agent2:\n")
print(response["messages"][-1].text)


AGENT3_PROMPT = """
You are a software engineering assistant that knows how to delegate tasks.

When a user describes a software-related task or problem, you must:
- Understand the request.
- Decide how to solve it by delegating it to a specialized assistant.
- Use only one tool to solve the task.
- Never explain or describe which tool you're using.
- Return only the final solution to the user.

If the task is unclear, ask the user for clarification. Never attempt to solve it yourself.
"""

agent3 = Agent(
    chat_generator=llm,
    system_prompt=AGENT3_PROMPT,
    tools=toolset  # contiene tutti i tool dinamici creati
)

response = agent3.run(messages=[
    ChatMessage.from_user("Write a Python function to validate an email address, then check if the result is correct. At the end execute them.")
])

print("\n🤖 agent3 output:\n")
print(response["messages"][-1].text)
