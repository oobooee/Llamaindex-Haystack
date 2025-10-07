import os
import asyncio
import pandas as pd
from agents.agents import workflow
from dotenv import load_dotenv
from tools.tools import load_training_data
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.utils.workflow import draw_all_possible_flows
from prompts.prompt_orig import (
    INITIAL_SUMMARIZER_PROMPT,
)

class Main:
    def __init__(self):
        self.workflow = workflow

    async def run(self):
        data = load_training_data()
        
        print("📊 Dataset caricato con successo.")
        print(f"📊 Dataset caricato con successo: {len(data)} record trovati.\n")
        readme = data[5]["readme"]
        ground_truth_description = data[5]["description"]
        print("📥 Input ricevuto:\n", readme)
  
        
        ctx = Context(workflow)
        await ctx.set("state", {
            "ground_truth_description": ground_truth_description,
            "readme": readme,
            "extracted_text": "", 
            "generated_about": "",
            "summarizer_prompt": INITIAL_SUMMARIZER_PROMPT,
            "best_prompt": INITIAL_SUMMARIZER_PROMPT,
            "best_score": 0,
            "attempt_count": 0,
            "max_attempts": 3                 
        })
        handler = self.workflow.run(user_msg=readme, ctx=ctx, stream=True)

        async for event in handler.stream_events():
            if isinstance(event, AgentOutput):
                print(f"\x1b[32m📤 Output:\x1b[0m\n{event.response.content}")
            elif isinstance(event, ToolCallResult):
                print(f"\x1b[36m🔧 Tool Result ({event.tool_name}):\x1b[0m {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"\x1b[33m🔨 Tool: {event.tool_name} - Args: {event.tool_kwargs}\x1b[0m")

        state = await ctx.get("state")
        print("\n📦 Stato finale:")
        for k, v in state.items():
            if isinstance(v, str):
                print(f"- {k}: {v[:100]}...")
            else:
                print(f"- {k}: {v}")
        print("\n✅ Workflow completato con successo!")
        draw_all_possible_flows(self.workflow, filename="workflow_AgentWF.html")
        print("\n📊 Flussi possibili disegnati con successo!")
                # 🔍 Stampa completa della memoria conversazionale
        memory = await ctx.get("memory")
        history = memory.get_all()

        print("\n🧠 Memoria conversazionale completa:")
        for i, msg in enumerate(history):
            print(f"\n--- Messaggio {i} ---")
            print(f"Ruolo: {msg.role}")
            print(f"Contenuto:\n{msg.content}")



if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
    main_flow = Main()
    asyncio.run(main_flow.run())
