import os
from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
    Context, 
)
from prompts.prompt_orig import (
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    TEACHER_PROMPT,
    COMBINE_PROMPT
)
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows
from tools.tools import simple_rouge_l_score
import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult

# ---- LLM Setup ----
llm = OpenAI(model="gpt-4o")
#llm = Ollama(model=os.getenv("OLLAMA_MODEL"))

# ---- Dataset ----
data = pd.read_csv("data/train_data.csv")
rows = iter(data.itertuples())

# ---- Event Classes ----
class ExtractedEvent(Event):
    extracted_text: str
    description: str

class SummaryEvent(Event):
    summary: str
    extracted_text: str
    description: str

class EvaluatedEvent(Event):
    rouge_score: float
    summary: str
    extracted_text: str
    description: str

class PromptUpdateEvent(Event):
    new_prompt: str
    extracted_text: str
    description: str

class CombinedEvent(Event):
    result: str

# ---- Workflow ----
class MetagenteWorkflow(Workflow):
    def __init__(self, **kwargs):
        self.prompt = INITIAL_SUMMARIZER_PROMPT
        self.best_prompts = []
        self.max_attempts = 2
        super().__init__(**kwargs)

    @step
    async def Extractor_Agent(self, ctx: Context, ev: StartEvent ) -> ExtractedEvent | CombinedEvent:
        try:
            row = next(rows)
        except StopIteration:
            return CombinedEvent(result="No more Readme, let's evaluate all the prompts.....")
        await ctx.set("attempt", 0)
        prompt = EXTRACTOR_PROMPT.replace("$readme_text", row.readme)
        print(f"\n📥 Row READ:\n{row.readme}\n\n🧠 Extractor Prompt:\n{prompt}")
        response = llm.complete(prompt, timeout=600)
        return ExtractedEvent(
            extracted_text=response.text.strip(),
            description=row.description
        )

    @step
    async def Summarizer_Agent(self, ctx: Context, ev: ExtractedEvent | PromptUpdateEvent) -> SummaryEvent:
        if isinstance(ev, PromptUpdateEvent):
            self.prompt = ev.new_prompt
            extracted_text = ev.extracted_text
            description = ev.description
        else:
            extracted_text = ev.extracted_text
            description = ev.description

        filled = self.prompt.replace("$extracted_text", extracted_text)
        print(f"\n🧠 Summarizer Prompt:\n{filled}")

        response = llm.complete(filled, timeout=600)
        return SummaryEvent(
            summary=response.text.strip(),
            extracted_text=extracted_text,
            description=description
        )

    @step
    async def gen_about_evaluation_tool(self, ctx: Context, ev: SummaryEvent) -> EvaluatedEvent:
        score = simple_rouge_l_score(ev.summary, ev.description)
        print(f"\n📊 Evaluation:\nSummary: {ev.summary}\nDescription: {ev.description}\nROUGE: {score}")
        return EvaluatedEvent(
            rouge_score=score,
            summary=ev.summary,
            extracted_text=ev.extracted_text,
            description=ev.description
        )

    @step
    async def Teacher_Agent(self, ctx: Context, ev: EvaluatedEvent) -> PromptUpdateEvent:
        attempt = await ctx.get("attempt", default=0)

        if ev.rouge_score >= 0.7 or attempt >= self.max_attempts:
            print("\n✅ Prompt finale accettato.")
            self.best_prompts.append(self.prompt)
            return StartEvent()
        print(f"\n📚 Teaching Prompt (Attempt {attempt}): ROUGE {ev.rouge_score}")
        filled = TEACHER_PROMPT \
            .replace("$extracted_text", ev.extracted_text) \
            .replace("$description", ev.description) \
            .replace("$generated_about", ev.summary) \
            .replace("$rouge_score", str(ev.rouge_score)) \
            .replace("$summarizer_prompt", self.prompt)

        print(f"\n🧠 Teacher Prompt:\n{filled}")
        response = llm.complete(filled, timeout=600)
        new_prompt = response.text.strip()

        await ctx.set("attempt", attempt + 1)

        return PromptUpdateEvent(
            new_prompt=new_prompt,
            extracted_text=ev.extracted_text,
            description=ev.description
        )

    @step
    async def Prompt_Combine_Agent(self, ev: CombinedEvent) -> StopEvent:
        if not self.best_prompts:
            return StopEvent(result="Nessun prompt da combinare.")
        prompt_list = "\n---\n".join(self.best_prompts)
        filled = COMBINE_PROMPT.replace("$summarizer_list", prompt_list)
        print(f"\n🧠 COMBINE Prompt:\n{filled}")
        response = llm.complete(filled, timeout=600)
        final_prompt = response.text.strip()
        with open("final_prompt.txt", "w") as f:
            f.write(final_prompt)

        print("\n✅ Prompt finale salvato.")
        return StopEvent(result="Workflow completato.")


async def main():
    w = MetagenteWorkflow(timeout=8900, verbose=True)
    result = await w.run()
    print("\n✅ FINITO:", result)
    draw_all_possible_flows(MetagenteWorkflow, filename="workflow_simple_eventd_WF_1.html")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
