import os
import pandas as pd
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from rouge_score import rouge_scorer
from haystack.tools import tool
from prompt_orig import EXTRACTOR_PROMPT, INITIAL_SUMMARIZER_PROMPT, TEACHER_PROMPT, COMBINE_PROMPT, TEACHER_PROMPT_ALPHA

@component
class RougeEvaluator:
    @component.output_types(score=float)
    def run(self, generated_about: str, ground_truth: str):
        print("\033[90mGround truth:\033[0m", ground_truth)
        print("\033[90mGenerated about:\033[0m", generated_about)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(ground_truth, generated_about)
        print(f"\033[96m[METRIC] ROUGE-L score: {score['rougeL'].fmeasure:.4f}\033[0m")
        return {"score": score["rougeL"].fmeasure}


@component
class MessageTextExtractor:
    @component.output_types(text=str)
    def run(self, messages: list[ChatMessage]):
        return {"text": messages[0].text if messages else ""}


@tool
def calculate_rouge_score(description: str, generated_about: str) -> float:
    """Compute ROUGE-L score between generated and ground truth descriptions."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(description, generated_about)
    print(f"\033[96m[TOOL FUNCTION CALL] ROUGE-L score: {score['rougeL'].fmeasure:.4f}\033[0m")
    global_scorer = score["rougeL"].fmeasure
    return global_scorer

@tool
def noop() -> str:
    """A dummy tool that does nothing."""
    return "noop"

from haystack.tools import tool
import pandas as pd



# Agent definitions
extractor_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    exit_conditions=["text"],
    max_agent_steps=100,
    tools=[noop]
)

summarizer_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    exit_conditions=["text"],
    max_agent_steps=100,
    tools=[noop]
)

teacher_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    max_agent_steps=100,  # massimo numero di passi dell'agente       
    tools=[calculate_rouge_score],
    exit_conditions=["text"]  # terminerà quando riceve una risposta testuale
)


combine_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    exit_conditions=["text"],
    max_agent_steps=100,
    tools=[noop]
)

# Prompts
extractor_agent_prompt = ChatMessage.from_system(EXTRACTOR_PROMPT)
summarizer_agent_prompt_start = ChatMessage.from_system(INITIAL_SUMMARIZER_PROMPT)
teacher_agent_prompt = ChatMessage.from_system(TEACHER_PROMPT_ALPHA)
combine_agent_prompt = ChatMessage.from_system(COMBINE_PROMPT)

train_df = pd.read_csv("data/train_data1.csv")
print(f"\033[92m[INFO] Dataset loaded with {len(train_df)} rows.\033[0m")

max_iterations = 2
threshold = 0.7
best_prompts = []



for idx, row in train_df.iterrows():
    print(f"\n\033[94m🔄 [ROW {idx+1}]\033[0m")
    readme = row["readme"]
    description = row["description"]

    iteration = 0
    best_score = 0
    best_prompt = summarizer_agent_prompt_start
    summarizer_prompt = summarizer_agent_prompt_start

    while iteration < max_iterations:
        iteration += 1
        print(f"\n\033[95m[INFO] Iteration {iteration}...\033[0m")

        # 1. ExtractorAgent
        extractor_result = extractor_agent.run(messages=[
            extractor_agent_prompt,
            ChatMessage.from_user(readme)
        ])
        extracted_text = extractor_result["messages"][-1].text
        print(f"\033[93m[EXTRACTED TEXT]\033[0m\n{extracted_text}")

        # 2. SummarizerAgent
        summarizer_result = summarizer_agent.run(messages=[
            summarizer_prompt,
            ChatMessage.from_user(extracted_text)
        ])
        generated_about = summarizer_result["messages"][-1].text
        print(f"\033[92m[GENERATED ABOUT]\033[0m\n{generated_about}")

        # 3. ROUGE Evaluation


        # 4. TeacherAgent
        teacher_input = f"""
            <EXTRACTED_TEXT>
            {extracted_text}
            </EXTRACTED_TEXT>

            <GROUND_TRUTH DESCRIPTION>
            {description}
            </GROUND_TRUTH DESCRIPTION>

            <GENERATED_DESCRIPTION>
            {generated_about}
            </GENERATED_DESCRIPTION>
           
            <CURRENT_PROMPT>
            {summarizer_prompt}
            </CURRENT_PROMPT>
        """


        print(f"\033[93m[TEACHER INPUT]\033[0m\n{teacher_input}")
        teacher_result = teacher_agent.run(messages=[
            teacher_agent_prompt,
            ChatMessage.from_user(teacher_input)
        ])
        print(f"\033[93m[TEACHER RESULT]\033[0m\n{teacher_result}")
        new_summarizer_prompt = teacher_result["messages"][-1].text
        print(f"\033[91m[TEACHER FEEDBACK]\033[0m\n{new_summarizer_prompt}")
        summarizer_prompt = ChatMessage.from_user(new_summarizer_prompt)

    if best_prompt:
        best_prompts.append(best_prompt)

# CombineAgent finale
summarizer_list = "\n\n".join([prompt.text for prompt in best_prompts])
print("\n\033[93m[INFO] Summarizer list...\033[0m")
combine_result = combine_agent.run(messages=[
    combine_agent_prompt,
    ChatMessage.from_user(summarizer_list)
])
final_prompt = combine_result["messages"][-1].text

print("\n\033[92m✅ [FINAL COMBINED PROMPT]\033[0m")
print(final_prompt)
