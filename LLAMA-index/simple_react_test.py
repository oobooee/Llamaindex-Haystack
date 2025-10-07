import os
import asyncio
import pandas as pd
from agents.agents import workflow
from dotenv import load_dotenv
from tools.tools import load_training_data
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.prompts import PromptTemplate
from llama_index.core.agent.workflow import AgentStream
from prompts.prompt_agents import (
    COMBINE_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    TEACHER_AWARENESS_PROMPT,
    TEACHER_PROMPT,
    TEACHER_PROMPT_C,
    TEACHER_PROMPT_D,
    TEACHER_PROMPT_REACT
)

custom_teacher_prompt = PromptTemplate(
    template=TEACHER_PROMPT_REACT,
)
custom_summarizer_prompt = PromptTemplate(
    template=INITIAL_SUMMARIZER_PROMPT,
)
custom_extractor_prompt = PromptTemplate(
    template=EXTRACTOR_PROMPT,  
)
custom_combiner_prompt = PromptTemplate(
    template=COMBINE_PROMPT,
)

class Main:
    def __init__(self):
        self.workflow = workflow
    async def run(self):
        data = load_training_data()
        print(f"📊 Dataset caricato con successo: {len(data)} record trovati.\n")

        for i, row in enumerate(data):
            print(f"\n{'='*40}\n🚀 Avvio esecuzione per record #{i+1}\n{'='*40}")
            readme = row["readme"]
            ground_truth_description = row["description"]

            ctx = Context(self.workflow)
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

            handler = self.workflow.run(user_msg=readme, ctx=ctx)


            async for ev in handler.stream_events():
                # if isinstance(ev, ToolCallResult):
                #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
                if isinstance(ev, AgentStream):
                    print(f"{ev.delta}", end="", flush=True)
            # async for event in handler.stream_events():
            #     if isinstance(event, AgentOutput):
            #         content = event.response.content.strip()
            #         if content:
            #             print(f"\n📤 \033[32m[AgentOutput] {event.agent_name}:\033[0m {content}")
            #     elif isinstance(event, ToolCallResult):
            #         print(f"\n🔧 \033[36m[Tool Result] {event.tool_name}:\033[0m {event.tool_output}")

            state = await ctx.get("state")
            print("\n📦 Stato finale:")
            for k, v in state.items():
                if isinstance(v, str):
                    print(f"- {k}: {v[:100]}...")
                else:
                    print(f"- {k}: {v}")
            print("\n✅ Workflow completato per record #{i+1}")

        draw_all_possible_flows(self.workflow, filename="workflow_AgentWF.html")
        print("\n📊 Flussi possibili disegnati con successo!")


if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
    main_flow = Main()
    asyncio.run(main_flow.run())
