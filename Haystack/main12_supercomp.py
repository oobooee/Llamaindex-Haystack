from haystack import super_component, Pipeline, SuperComponent
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




class ExtractSummarizeEvaluate(SuperComponent):
    def __init__(self, max_iterations: int = 3, threshold: float = 0.7):
        self.max_iterations = max_iterations
        self.threshold = threshold

        generator = OpenAIChatGenerator(model="gpt-4o")

        extractor = Agent(generator, tools=[], system_prompt=EXTRACTOR_PROMPT)
        summarizer = Agent(generator, tools=[], system_prompt=INITIAL_SUMMARIZER_PROMPT)
        self.teacher_agent = Agent(generator, tools=[], system_prompt=TEACHER_PROMPT)

        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("summarizer", summarizer)
        pipeline.add_component("extract_text", MessageTextExtractor())
        pipeline.add_component("summary_text", MessageTextExtractor())
        pipeline.add_component("rouge", RougeEvaluator())

        pipeline.connect("extractor", "extract_text")
        pipeline.connect("extractor", "summarizer")
        pipeline.connect("summarizer", "summary_text")
        pipeline.connect("summary_text.text", "rouge.generated_about")

        input_mapping = {
            "readme": "extractor.messages",
            "description": "rouge.ground_truth"
        }

        output_mapping = {
            "extract_text.text": "extracted_text",
            "summary_text.text": "generated_about",
            "rouge.score": "rouge_score"
        }

        super().__init__(pipeline=pipeline, input_mapping=input_mapping, output_mapping=output_mapping)

    def run_training_loop(self, readme: str, description: str) -> dict:
        summarizer_prompt = ChatMessage.from_system(INITIAL_SUMMARIZER_PROMPT)
        best_score = 0.0
        best_prompt = None

        for i in range(self.max_iterations):
            self.pipeline.components["summarizer"].system_prompt = summarizer_prompt.text

            result = self.run(readme=[ChatMessage.from_user(readme)], description=description)
            extracted = result["extracted_text"]
            generated = result["generated_about"]
            rouge = result["rouge_score"]

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
            teacher_out = self.teacher_agent.run(messages=[
                ChatMessage.from_system(TEACHER_PROMPT),
                ChatMessage.from_user(teacher_input)
            ])
            summarizer_prompt = ChatMessage.from_system(teacher_out["messages"][-1].text)

        return {"best_prompt": best_prompt.text}




class Combine(SuperComponent):
    def __init__(self):
        generator = OpenAIChatGenerator(model="gpt-4o")
        combine_agent = Agent(generator, tools=[], system_prompt=COMBINE_PROMPT)

        pipeline = Pipeline()
        pipeline.add_component("agent", combine_agent)

        input_mapping = {"prompt_list": "agent.messages"}
        output_mapping = {"agent.replies": "combined_prompt"}

        super().__init__(pipeline=pipeline, input_mapping=input_mapping, output_mapping=output_mapping)

    def run_combination(self, prompt_list: list[str]) -> dict:
        messages = [
            ChatMessage.from_system(COMBINE_PROMPT),
            ChatMessage.from_user("\n\n".join(prompt_list))
        ]
        return self.run(prompt_list=messages)



# Carica dataset
df = pd.read_csv("data/train_data.csv")
extract_summar_teach = ExtractSummarizeEvaluate(max_iterations=5, threshold=0.75)
combine = Combine()
best_prompts = []
for idx, row in df.iterrows():
    result = extract_summar_teach.run_training_loop(readme=row["readme"], description=row["description"])
    best_prompts.append(result["best_prompt"])

combine_result = combine.run_combination(prompt_list=best_prompts)
final_prompt = combine_result["combined_prompt"][-1].text

