import os
import pandas as pd
from haystack import Pipeline, component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.tools import tool
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

# Load dataset
print("\033[92m[INFO] Loading dataset from data/train_data1.csv...\033[0m")
train_df = pd.read_csv("data/train_data.csv")
print(f"\033[92m[INFO] Dataset loaded with {len(train_df)} rows.\033[0m")

max_iterations = 2
threshold = 0.7
best_prompts = []

for idx, row in train_df.iterrows():
    print(f"\n\033[94m🔄 Processing row {idx + 1}/{len(train_df)}...\033[0m")
    iteration = 0
    best_score = 0
    best_teacher_prompt = None
    readme = row["readme"]
    description = row["description"]
    summarizer_prompt = ChatMessage.from_system(INITIAL_SUMMARIZER_PROMPT)

    while iteration < max_iterations:
        iteration += 1
        print(f"\n\033[96m[INFO] Iteration {iteration}...\033[0m")

        pipeline = Pipeline()

        extractor_agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o", api_key=Secret.from_env_var("OPENAI_API_KEY")),
            tools=[noop],
            system_prompt=EXTRACTOR_PROMPT
        )
        summarizer_agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o", api_key=Secret.from_env_var("OPENAI_API_KEY")),
            tools=[noop],
            system_prompt=summarizer_prompt.text
        )
        teacher_agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o", api_key=Secret.from_env_var("OPENAI_API_KEY")),
            tools=[noop],
            system_prompt=TEACHER_PROMPT
        )

        pipeline.add_component("extractor_agent", extractor_agent)
        pipeline.add_component("summarizer_agent", summarizer_agent)
        pipeline.add_component("extract_text", MessageTextExtractor())
        pipeline.add_component("summary_text", MessageTextExtractor())
        pipeline.add_component("rouge", RougeEvaluator())

        pipeline.connect("extractor_agent", "extract_text")
        pipeline.connect("extractor_agent", "summarizer_agent")
        pipeline.connect("summarizer_agent", "summary_text")
        pipeline.connect("summary_text.text", "rouge.generated_about")

        result = pipeline.run({
            "extractor_agent": {"messages": [ChatMessage.from_user(readme)]},
            "rouge": {"ground_truth": description}
        }, include_outputs_from={"extract_text", "summary_text", "rouge"})

        extracted_text = result["extract_text"]["text"]
        generated_about = result["summary_text"]["text"]
        rouge_score = result["rouge"]["score"]

        print(f"\033[94m🟦 Extracted Text:\033[0m\n{extracted_text}")
        print(f"\033[93m🟨 Generated About:\033[0m\n{generated_about}")

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

<ROUGE_SCORE>
{rouge_score}
</ROUGE_SCORE>

<CURRENT_PROMPT>
{summarizer_prompt}
</CURRENT_PROMPT>
        """

        teacher_result = teacher_agent.run(messages=[
            ChatMessage.from_system(TEACHER_PROMPT),
            ChatMessage.from_user(teacher_input)
        ])
        teacher_response = teacher_result["messages"][-1].text

        print(f"\033[95m📘 Teacher Feedback:\033[0m\n{teacher_response}")

        if rouge_score > best_score:
            best_score = rouge_score
            best_teacher_prompt = ChatMessage.from_system(teacher_response)

        if rouge_score >= threshold:
            print(f"\033[92m[✓] Threshold reached ({rouge_score:.4f}), stopping.\033[0m")
            break
        else:
            summarizer_prompt = ChatMessage.from_user(teacher_response)

    if best_teacher_prompt:
        best_prompts.append(best_teacher_prompt)

# Combine agent
combine_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o", api_key=Secret.from_env_var("OPENAI_API_KEY")),
    tools=[noop],
    system_prompt=COMBINE_PROMPT
)

combined_input = "\n\n".join([prompt.text for prompt in best_prompts])
print("\n\033[94m[INFO] Combining best prompts...\033[0m")
print(f"\033[90m[DEBUG] Combined input:\033[0m\n{combined_input}")

combine_result = combine_agent.run(messages=[
    ChatMessage.from_system(COMBINE_PROMPT),
    ChatMessage.from_user(combined_input)
])
final_prompt = combine_result["messages"][-1].text

print("\n\033[92m✅ FINAL COMBINED PROMPT:\033[0m")
print(final_prompt)

# Save output
os.makedirs("result/train", exist_ok=True)
with open("result/train/final_combined_prompt.txt", "w", encoding="utf-8") as f:
    f.write(final_prompt)
