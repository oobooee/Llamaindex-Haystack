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
def calculate_rouge_score(generated_about: str, ground_truth: str) -> float:
    """Compute ROUGE-L score between generated and ground truth descriptions."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(ground_truth, generated_about)
    return score["rougeL"].fmeasure

@tool
def noop() -> str:
    """A dummy tool that does nothing."""
    return "noop"

# Agent definitions
extractor_agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4o-mini",
        generation_kwargs={"temperature": 0.0},
        api_key=Secret.from_env_var("OPENAI_API_KEY")
    ),
    tools=[noop]
)

summarizer_agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4o-mini",
        generation_kwargs={"temperature": 0.0},
        api_key=Secret.from_env_var("OPENAI_API_KEY")
    ),
    system_prompt=INITIAL_SUMMARIZER_PROMPT,
    tools=[noop]
)

teacher_agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4o",
        generation_kwargs={"temperature": 0.7},
        api_key=Secret.from_env_var("OPENAI_API_KEY")
    ),
    system_prompt=TEACHER_PROMPT,
    tools=[noop]
)

combine_agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4o",
        generation_kwargs={"temperature": 0.2},
        api_key=Secret.from_env_var("OPENAI_API_KEY")
    ),
    tools=[noop]
)

# Prompts
extractor_agent_prompt = ChatMessage.from_system(EXTRACTOR_PROMPT)
summarizer_agent_prompt_start = ChatMessage.from_system(INITIAL_SUMMARIZER_PROMPT)
teacher_agent_prompt = ChatMessage.from_system(TEACHER_PROMPT)
combine_agent_prompt = ChatMessage.from_system(COMBINE_PROMPT)

train_df = pd.read_csv("data/train_data.csv")
print(f"\033[92m[INFO] Dataset loaded with {len(train_df)} rows.\033[0m")

max_iterations = 15
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


        summ_input=[
            summarizer_prompt,
            ChatMessage.from_user(extracted_text)
        ]
        print(f"\033[93m[SUMMARIZER INPUT]\033[0m\n{summ_input}")
        summarizer_result = summarizer_agent.run(summ_input)

        generated_about = summarizer_result["messages"][-1].text
        print(f"\033[92m[GENERATED ABOUT]\033[0m\n{generated_about}")

        # 3. ROUGE Evaluation
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_score = scorer.score(description, generated_about)["rougeL"].fmeasure
        print(f"\033[96m[ROUGE-L]: {rouge_score:.4f}\033[0m")

        if rouge_score > best_score:
            best_score = rouge_score
            best_prompt = summarizer_prompt
            print(f"\033[92m[INFO] New best score: {best_score:.4f}\033[0m")
            print(f"\033[92m[INFO] New best prompt:\033[0m\n{best_prompt.text}")
        if rouge_score >= threshold:
            print("\033[92m[INFO] Threshold reached, stopping early.\033[0m")
            break

        # 4. TeacherAgent
        teacher_input = f"""<EXTRACTED_TEXT>
{extracted_text}
</EXTRACTED_TEXT>

<GROUND_TRUTH DESCRIPTION>
{description}
</GROUND_TRUTH DESCRIPTION>

<GENERATED_DESCRIPTION>
{generated_about}
</GENERATED_DESCRIPTION>

<ROUGE_SCORE>
{rouge_score}
</ROUGE_SCORE>

<CURRENT_PROMPT>
{summarizer_prompt.text}
</CURRENT_PROMPT>"""

        print(f"\033[93m[TEACHER INPUT]\033[0m\n{teacher_input}")
        teacher_result = teacher_agent.run(messages=[
            teacher_agent_prompt,
            ChatMessage.from_user(teacher_input)
        ])
        new_summarizer_prompt = teacher_result["messages"][-1].text
        if "$extracted_text" not in new_summarizer_prompt:
            new_summarizer_prompt += "\n\n<EXTRACTED_README>\n$extracted_text\n</EXTRACTED_README>"
        print(f"\033[91m[TEACHER FEEDBACK]\033[0m\n{new_summarizer_prompt}")
        summarizer_prompt = ChatMessage.from_user(new_summarizer_prompt)

    if best_prompt:
        best_prompts.append(best_prompt)

# CombineAgent finale
summarizer_list = "\n\n".join([prompt.text for prompt in best_prompts])
print("\n\033[93m[INFO] [SUMMARIZER LIST 4 combiner]\033[0m")
print(summarizer_list)
print("-" * 80)
combine_list=[
    combine_agent_prompt,
    ChatMessage.from_user(summarizer_list)
]
print("\n\033[93m [INFO] [COMBINE INPUT]\033[0m")
print(combine_list)
# Final combined prompt
# Extract the final prompt from the CombineAgent result
combine_result = combine_agent.run(combine_list)
final_prompt = combine_result["messages"][-1].text
if "$extracted_text" not in final_prompt:
    final_prompt += "\n\n<EXTRACTED_README>\n$extracted_text\n</EXTRACTED_README>"

print("\n\033[92m✅ [FINAL COMBINED PROMPT]\033[0m")
print(final_prompt)

# Save output
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("result/train", exist_ok=True)
with open(f"result/train/final_combined_prompt{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write(final_prompt)
