from haystack import super_component, Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack import component
from haystack.tools import tool
import pandas as pd
import os
from prompt_orig import EXTRACTOR_PROMPT, INITIAL_SUMMARIZER_PROMPT, TEACHER_PROMPT, COMBINE_PROMPT

@tool
def noop() -> str:
    return "noop"

@component
class MessageTextExtractor:
    @component.output_types(text=str)
    def run(self, messages: list[ChatMessage]):
        return {"text": messages[-1].text if messages else ""}

@component
class RougeEvaluator:
    @component.output_types(score=float)
    def run(self, generated_about: str, ground_truth: str):
        from rouge_score import rouge_scorer
        score = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(ground_truth, generated_about)
        print(f"\033[93m[ROUGE-L SCORE] → {score['rougeL'].fmeasure:.4f}\033[0m")
        return {"score": score["rougeL"].fmeasure}


@super_component
class ExtractSummarizeEvaluate:
    def __init__(self, max_iterations: int = 3, threshold: float = 0.7):
        self.max_iterations = max_iterations
        self.threshold = threshold
        llm = OpenAIChatGenerator(model="gpt-4o")
        extractor_agent = Agent(llm, tools=[], system_prompt=EXTRACTOR_PROMPT)
        summarizer_agent = Agent(llm, tools=[], system_prompt=INITIAL_SUMMARIZER_PROMPT)
        self.teacher_agent = Agent(llm, tools=[], system_prompt=TEACHER_PROMPT)

        self.pipeline = Pipeline()
        self.pipeline.add_component("extractor", extractor_agent)
        self.pipeline.add_component("summarizer", summarizer_agent)
        self.pipeline.add_component("extract_text", MessageTextExtractor())
        self.pipeline.add_component("summary_text", MessageTextExtractor())
        self.pipeline.add_component("rouge", RougeEvaluator())

        self.pipeline.connect("extractor", "extract_text")
        self.pipeline.connect("extractor", "summarizer")
        self.pipeline.connect("summarizer", "summary_text")
        self.pipeline.connect("summary_text.text", "rouge.generated_about")

    def run_training_loop(self, readme: str, description: str) -> dict:
        summarizer_prompt = ChatMessage.from_system(INITIAL_SUMMARIZER_PROMPT)
        best_score = 0.0
        best_prompt = None
        
        for i in range(self.max_iterations):
            self.pipeline.components["summarizer"].system_prompt = summarizer_prompt.text

            result = self.pipeline.run({
                "extractor": {"messages": [ChatMessage.from_user(readme)]},
                "rouge": {"ground_truth": description}
            }, include_outputs_from={"extract_text", "summary_text", "rouge"})

            extracted = result["extract_text"]["text"]
            generated = result["summary_text"]["text"]
            rouge = result["rouge"]["score"]

            if rouge > best_score:
                best_score = rouge
                best_prompt = summarizer_prompt

            if rouge >= self.threshold:
                break

            teacher_input = f"""
                <EXTRACTED_TEXT>\n{extracted}\n</EXTRACTED_TEXT>
                <GROUND_TRUTH>\n{description}\n</GROUND_TRUTH>
                <GENERATED>\n{generated}\n</GENERATED>
                <ROUGE>\n{rouge}\n</ROUGE>
                <PROMPT>\n{summarizer_prompt.text}\n</PROMPT>
                """
            teacher_response = self.teacher_agent.run(messages=[
                ChatMessage.from_system(TEACHER_PROMPT),
                ChatMessage.from_user(teacher_input)
            ])
            summarizer_prompt = ChatMessage.from_system(teacher_response["messages"][-1].text)

        return {"best_prompt": best_prompt.text}



@super_component
class Combine:
    def __init__(self):
        generator = OpenAIChatGenerator(model="gpt-4o")
        combine_agent = Agent(generator, tools=[], system_prompt=COMBINE_PROMPT)

        self.pipeline = Pipeline()
        self.pipeline.add_component("agent", combine_agent)

    def run_combination(self, summarizer_list: list[str]) -> dict:
        messages = [
            ChatMessage.from_system(COMBINE_PROMPT),
            ChatMessage.from_user("\n\n".join(summarizer_list))
        ]
        result = self.pipeline.run({"agent": {"messages": messages}})
        return {"final_prompt": result["agent"]["replies"][-1].text}


# Carica dataset
df = pd.read_csv("data/train_data.csv")
extract_summarize_evaluate = ExtractSummarizeEvaluate(max_iterations=5, threshold=0.75)
combine = Combine()
best_prompts = []


for idx, row in df.iterrows():
    result = extract_summarize_evaluate.run_training_loop(readme=row["readme"], description=row["description"])
    best_prompts.append(result["best_prompt"])

combine_result = combine.run_combination(prompt_list=best_prompts)
final_prompt = combine_result["final_prompt"]

# Salvataggio
os.makedirs("result/train", exist_ok=True)
with open("result/train/final_combined_prompt.txt", "w", encoding="utf-8") as f:
    f.write(final_prompt)

print("\n\033[92m✅ FINAL PROMPT SAVED TO result/train/final_combined_prompt.txt\033[0m")
