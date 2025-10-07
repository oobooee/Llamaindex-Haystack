import os
import json
import pandas as pd
import csv
from datetime import datetime
from rouge_score import rouge_scorer
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
    COMBINE_PROMPT_EVO,
    TEACHER_PROMPT_EVO,
    TEACHER_PROMPT,
    COMBINE_PROMPT,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.utils.workflow import draw_all_possible_flows
from tools.others_orig import save_parallel_train_result, save_evaluation_result
# ---- Colored Logs ----
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
RESET   = "\033[0m"
PINK    = "\033[95m"
BLACK   = "\033[30m"
DARK_GRAY = "\033[90m"
ORANGE  = "\033[38;5;208m"
WHITE   = "\033[97m"

api_version = "2024-12-01-preview"



subscription_key_mini = ""
endpoint_mini = ""
model_name_mini = "gpt-4o-mini"
deployment_mini = "gpt-4o-mini"

subscription_key = ""
endpoint = ""
model_name = "gpt-4o"
deployment = "gpt-4o"


llm_mini = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint_mini,
    api_key=subscription_key_mini,
    model=deployment_mini,
    engine=model_name_mini
)
llm = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    model=deployment,
    engine=model_name
)
# ---- LLM Setup ----
#llm = OpenAI(model="gpt-4o")
#llm_mini = OpenAI(model="gpt-4o-mini")
# ---- Dataset ----
data_train = pd.read_csv("data/TS50.csv")
rows_train = iter(data_train.itertuples())

data_test = pd.read_csv("data/test_data.csv")
rows_test = iter(data_test.itertuples())


# ---- Event Classes ----
class ExtractedEvent(Event):
    extracted_text: str
    description: str
    readme: str

class SummaryEvent(Event):
    summary: str
    extracted_text: str
    description: str
    readme: str

class EvaluatedEvent(Event):
    rouge_score: float
    summary: str
    extracted_text: str
    description: str
    readme: str

class PromptUpdateEvent(Event):
    new_prompt: str
    extracted_text: str
    description: str
    readme: str

class CombinedEvent(Event):
    result: str


# ---- Training Workflow ----
class MetagenteWorkflow(Workflow):
    def __init__(self, max_attempts: int = 15, **kwargs):
        self.prompt = INITIAL_SUMMARIZER_PROMPT
        self.best_prompt = INITIAL_SUMMARIZER_PROMPT
        self.best_score = 0.0
        self.iteration_debug = []  # ← per raccogliere tutti i dati
        self.data_prompt = []      # ← per i prompt da combinare
        self.best_prompts = []
        self.max_attempts = max_attempts
        self.train_row_index = 0
        super().__init__(**kwargs)    

    @step
    async def Extractor_Agent(self, ctx: Context, ev: StartEvent) -> ExtractedEvent | CombinedEvent:
        print(f"{BLUE}[Extractor_Agent: Extracting from README...]{RESET}")
        try:
            row = next(rows_train)
        except StopIteration:
            return CombinedEvent(result="No more Readme, let's evaluate all the prompts.....")
        await ctx.set("attempt", 0)
        print(f"{WHITE}[Evaluation Progress: Row #{self.train_row_index + 1}]{RESET}")
        self.train_row_index += 1
        prompt = EXTRACTOR_PROMPT.replace("$readme_text", row.readme)
        print(f"{DARK_GRAY}[Extractor_Agent: LLM prompt ->]\n{prompt}{RESET}")
        response = llm_mini.complete(prompt, temperature=0.0)
        print(f"{ORANGE}[Extractor_Agent: LLM output ->]\n{response.text.strip()}{RESET}")
        self.prompt = INITIAL_SUMMARIZER_PROMPT         # 🔁 Reset del prompt
        self.best_prompt = INITIAL_SUMMARIZER_PROMPT    # 🔁 Reset del best prompt
        self.best_score = 0.0    
        return ExtractedEvent(
            extracted_text=response.text.strip(),
            description=row.description,
            readme=row.readme
        )

    @step
    async def Summarizer_Agent(self, ctx: Context, ev: ExtractedEvent | PromptUpdateEvent) -> SummaryEvent:
        if isinstance(ev, PromptUpdateEvent):
            self.prompt = ev.new_prompt
        filled = self.prompt.replace("$extracted_text", ev.extracted_text)
        print(f"{CYAN}[Summarizer_Agent: LLM prompt ->]\n{filled}{RESET}")
        response = llm_mini.complete(filled, temperature=0.0)
        print(f"{CYAN}[Summarizer_Agent: LLM output ->]\n{response.text.strip()}{RESET}")
        return SummaryEvent(
            summary=response.text.strip(),
            extracted_text=ev.extracted_text,
            description=ev.description,
            readme=ev.readme
        )

    @step
    async def Teacher_Agent(self, ctx: Context, ev: EvaluatedEvent) -> PromptUpdateEvent | StartEvent:
        attempt = await ctx.get("attempt", default=0)
        print(f"{RED}[Teacher_Agent: Evaluating and updating prompt... Attempt #{attempt + 1}]{RESET}")

        # 🔥 Check if this is the best prompt so far
        if ev.rouge_score > self.best_score:
            self.best_score = ev.rouge_score
            self.best_prompt = self.prompt
            print(f"{GREEN}[Teacher_Agent: 🔄 Nuovo best prompt salvato (ROUGE={ev.rouge_score:.4f})]{RESET}")

        # Condizione di arresto
        if ev.rouge_score >= 0.7 or attempt >= self.max_attempts:
            # ✅ aggiunta manuale extra se necessario (comportamento docente)
            if self.best_score >= 0.7 and self.best_prompt not in self.data_prompt:
                self.data_prompt.append(self.best_prompt)

            self.best_prompts.append({
                "prompt": self.best_prompt,
                "rougeL": self.best_score
            })
            print(f"{ORANGE}[Teacher_Agent: ⏹️ Iterazioni terminate - si torna all'inizio]{RESET}")

            iteration_data = await ctx.get("iteration_data", [])
            self.iteration_debug.append({
                "readme": ev.readme,
                "description": ev.description,
                "iteration_debug": iteration_data,
                "best_ROUGE-L": self.best_score,
                "best_summarizer_prompt": self.best_prompt
            })
            await ctx.set("history_data", [])
            return StartEvent()
        history_data = await ctx.get("history_data", default=[])
        recent_history = history_data[-3:]
        history_lines = []
        for i, step in enumerate(recent_history):
            history_lines.append(
                f"{i+1}. Prompt: {step['summarizer_prompt']!r}, "
                f"Output: {step['generated_about']!r}, "
                f"ROUGE-L: {step['rougeL_score']:.4f}"
            )
        history_str = "\n".join(history_lines)

        # 💾 Appendi nuovo elemento allo storico per il prossimo giro
        history_data = await ctx.get("history_data", default=[])
        history_data.append({
            "summarizer_prompt": self.prompt,
            "generated_about": ev.summary,
            "rougeL_score": ev.rouge_score
        })
        await ctx.set("history_data", history_data)
        # Prompt improvement via LLM
        filled = TEACHER_PROMPT_EVO \
            .replace("$extracted_text", ev.extracted_text) \
            .replace("$description", ev.description) \
            .replace("$generated_about", ev.summary) \
            .replace("$rouge_score", str(ev.rouge_score)) \
            .replace("$summarizer_prompt", self.prompt) \
            .replace("$history_attempts", history_str)
        print(f"{YELLOW}[Teacher_Agent: LLM prompt ->]\n{filled}{RESET}")
        response = llm.complete(filled, temperature=0.7)
        new_prompt = response.text.strip()
        print(f"{MAGENTA}[Teacher_Agent: LLM output (new prompt) ->]\n{new_prompt}{RESET}")
        await ctx.set("attempt", attempt + 1)
        return PromptUpdateEvent(
            new_prompt=new_prompt,
            extracted_text=ev.extracted_text,
            description=ev.description,
            readme=ev.readme
        )


    @step
    async def gen_about_evaluation_tool_4_train(self, ctx: Context, ev: SummaryEvent) -> EvaluatedEvent:
        print(f"{YELLOW}[Evaluator: Calculating ROUGE score...]{RESET}")

        # Calcola tutte le metriche ROUGE
        scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(ev.summary, ev.description)
        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        rougeL = scores["rougeL"].fmeasure

        # Recupera lo storico dal contesto, oppure inizializzalo
        iteration_data = await ctx.get("iteration_data", default=[])

        # Aggiungi il turno attuale
        iteration_data.append({
            "summarizer_prompt": self.prompt,
            "readme": ev.readme,
            "extracted_text": ev.extracted_text,
            "description": ev.description,
            "generated_about": ev.summary,
            "rouge1_score": rouge1,
            "rouge2_score": rouge2,
            "rougeL_score": rougeL,
        })

        # Salva nel contesto per il Teacher
        await ctx.set("iteration_data", iteration_data)

        return EvaluatedEvent(
            rouge_score=rougeL,
            summary=ev.summary,
            extracted_text=ev.extracted_text,
            description=ev.description,
            readme=ev.readme
        )


    @step
    async def Prompt_Combine_Agent(self, ev: CombinedEvent) -> StopEvent:
        print(f"{GREEN}[Prompt_Combine_Agent: Combining best prompts...]{RESET}")
        if not self.best_prompts:
            return StopEvent(result="Nessun prompt da combinare.")
        
        def format_prompt(entry):
            if isinstance(entry, str):
                return f"{entry.strip()}"
            elif isinstance(entry, dict):
                return f"{entry['prompt'].strip()}\n[ROUGE-L score: {entry['rougeL']:.4f}]"
            else:
                return str(entry)

        prompt_list = "\n---\n".join([format_prompt(p) for p in self.best_prompts])
        filled = COMBINE_PROMPT_EVO.replace("$summarizer_list", prompt_list)
        print(f"{RED}[Prompt_Combine_Agent: LLM prompt ->]\n{filled}{RESET}")
        response = llm.complete(filled, temperature=0.2)
        final_prompt = response.text.strip()
        print(f"{YELLOW}[Prompt_Combine_Agent: LLM output (final prompt) ->]\n{final_prompt}{RESET}")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"final_prompt_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(final_prompt)
        self.final_prompt = final_prompt
        self.debug_result = {
            "timestamp": timestamp,
            "Train Debug": self.iteration_debug,
            "Final Summarizer Prompt": final_prompt,
        }
        #save_parallel_train_result_compatible("result/train", self.debug_result)
        save_parallel_train_result("result/train", self.debug_result)
        return StopEvent(result="Workflow completato.")


    
# ---- Evaluation Workflow ----
class MetagenteEvaluationWorkflow(Workflow):
    def __init__(self, prompt_filename:str, **kwargs):
        try:
            with open(prompt_filename, "r", encoding="utf-8") as final_prompt:
                self.summarizer_prompt = final_prompt.read().strip()
                print(f"{BLUE}[Evaluator Setup: Prompt loaded ->]\n{self.summarizer_prompt}{RESET}")
        except FileNotFoundError:
            raise RuntimeError("final_prompt.txt non trovato; eseguire prima il training.")
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.debug_result = {
            "data_debug": [],
            "avg_rouge1_score": 0.0,
            "avg_rouge2_score": 0.0,
            "avg_rougeL_score": 0.0,
        }
        self.total_rouge1 = 0.0
        self.total_rouge2 = 0.0
        self.total_rougeL = 0.0
        self.count = 0
        self.test_row_index = 0
        super().__init__(**kwargs)

    @step
    async def Extractor_Agent(self, ctx: Context, ev: StartEvent) -> ExtractedEvent | StopEvent:
        print(f"{BLUE}[Extractor_Agent - Eval: Extracting from test README...]{RESET}")
        try:
            row = next(rows_test)
        except StopIteration:
            if self.count > 0:
                self.debug_result["avg_rouge1_score"] = self.total_rouge1 / self.count
                self.debug_result["avg_rouge2_score"] = self.total_rouge2 / self.count
                self.debug_result["avg_rougeL_score"] = self.total_rougeL / self.count
            else:
                self.debug_result["avg_rouge1_score"] = 0.0
                self.debug_result["avg_rouge2_score"] = 0.0
                self.debug_result["avg_rougeL_score"] = 0.0

            save_evaluation_result("result/test", self.debug_result)
            return StopEvent(result="Workflow completato.")
        print(f"{WHITE}[Evaluation Progress: Row #{self.test_row_index + 1}]{RESET}")
        self.test_row_index += 1
        prompt = EXTRACTOR_PROMPT.replace("$readme_text", row.readme)
        print(f"{DARK_GRAY}[Extractor_Agent - Eval: LLM prompt ->]\n{prompt}{RESET}")
        response = llm_mini.complete(prompt, temperature=0.0)
        print(f"{ORANGE}[Extractor_Agent - Eval: LLM output ->]\n{response.text.strip()}{RESET}")

        return ExtractedEvent(
            extracted_text=response.text.strip(),
            description=row.description,
            readme=row.readme
        )

    @step
    async def Summarizer_Agent(self, ctx: Context, ev: ExtractedEvent ) -> SummaryEvent:
        filled = self.summarizer_prompt.replace("$extracted_text", ev.extracted_text)
        print(f"{RED}[Summarizer_Agent: LLM prompt ->]\n{filled}{RESET}")
        response = llm_mini.complete(filled, temperature=0.0)
        print(f"{CYAN}[Summarizer_Agent: LLM output ->]\n{response.text.strip()}{RESET}")
        return SummaryEvent(
            summary=response.text.strip(),
            extracted_text=ev.extracted_text,
            description=ev.description,
            readme=ev.readme
        )

    @step
    async def gen_about_evaluation_tool_4_test(self, ctx: Context, ev: SummaryEvent) -> StartEvent:
        print(f"{YELLOW}[Evaluator: Calculating ROUGE score...]{RESET}")

        # Calcola tutte le metriche ROUGE
        scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(ev.summary, ev.description)
        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        rougeL = scores["rougeL"].fmeasure
        print(f"{YELLOW}[Evaluator: ROUGE-1 Score: {rouge1:.4f}]{RESET}")
        print(f"{YELLOW}[Evaluator: ROUGE-2 Score: {rouge2:.4f}]{RESET}")
        print(f"{YELLOW}[Evaluator: ROUGE-L Score: {rougeL:.4f}]{RESET}")

        # qui aggiungere il debug per il csv
        self.debug_result["data_debug"].append({
            "extracted_text": ev.extracted_text,  # idem
            "description": ev.description,
            "generated_about": ev.summary,
            "rouge1_score": rouge1,
            "rouge2_score": rouge2,
            "rougeL_score": rougeL,
        })

        self.total_rouge1 += rouge1
        self.total_rouge2 += rouge2
        self.total_rougeL += rougeL
        self.count += 1

        return StartEvent(
            rouge_score=rougeL,
            summary=ev.summary,
            extracted_text=ev.extracted_text,
            description=ev.description,
            readme=ev.readme
        )


# ---- Main ----
async def main():
    # w = MetagenteWorkflow(timeout=None , verbose=True)
    # await w.run()
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # draw_all_possible_flows(MetagenteWorkflow, filename=f"workflow_train_{timestamp}.html")


    e = MetagenteEvaluationWorkflow(timeout=None , prompt_filename="final_prompt_2025-06-11_11-05-50.txt", verbose=True)
    await e.run()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    draw_all_possible_flows(MetagenteEvaluationWorkflow, filename=f"workflow_test_{timestamp}.html")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
