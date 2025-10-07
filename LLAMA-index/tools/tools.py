from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.workflow import Context
from metric.rouge import ROUGE
import pandas as pd
import os




# 🔹 External function, callable as LLM tool or utility
def load_training_data(csv_path: str = None):
    """
    Reads the train_data.csv file and returns a list of records (dict), as in the original project.
    """
    if csv_path is None:
        csv_path = os.path.join(os.getcwd(), "data", "train_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    data = pd.read_csv(csv_path)
    expected_cols = {"description", "readme"}
    if not expected_cols.issubset(data.columns):
        raise ValueError(f"The CSV must contain the following columns: {expected_cols}")

    return data.to_dict(orient="records")

async def get_summarizer_prompt(ctx: Context) -> str:
    state = await ctx.get("state")
    summarizer_prompt = state["summarizer_prompt"]
    print(f"\n🔵 Prompt retrieved from state:\n{summarizer_prompt}")
    return summarizer_prompt

async def save_extracted_text(ctx: Context, extracted_text: str) -> str:
    state = await ctx.get("state")
    state["extracted_text"] = extracted_text
    await ctx.set("state", state)
    print(f"\n🟡 Extracted text saved in state:\n{extracted_text}")
    return "Extracted text saved"

async def save_summary(ctx: Context, summary: str) -> str:
    state = await ctx.get("state")
    #state["summary"] = summary
    state["generated_about"] = summary
    await ctx.set("state", state)
    print(f"\n🔵 About saved in state:\n{summary}")
    return "Summary saved"

async def save_new_summarizer_prompt(ctx: Context, new_summarizer_prompt) -> str:
    state = await ctx.get("state")
    best_score = state["best_score"]
    rouge_score = state["rouge_score"]
    attempt_count = state["attempt_count"] + 1
    max_attempts = state["max_attempts"] + 1
    state["max_attempts"] = attempt_count
    state["attempt_count"] = attempt_count
    if rouge_score > best_score:
        state["best_prompt"] = new_summarizer_prompt
        state["best_score"] = rouge_score
        print(f"\n🔵 New best prompt saved in state:")
        state["summarizer_prompt"] = new_summarizer_prompt
    else:
        state["summarizer_prompt"] = state["best_prompt"]
    print(f"\n🔁 Iterazione n. {attempt_count})")



async def calculate_rouge_score(ctx: Context) -> str:
    state = await ctx.get("state")
    generated_about = state["generated_about"]
    ground_truth_description = state["ground_truth_description"]
    rouge_score = simple_rouge_l_score(generated_about, ground_truth_description)
    state["rouge_score"] = rouge_score
    print("📊 ROUGE-L Score calculated:", rouge_score)
    print("✅ Context updated:")
    return f"New rouge_score calculated is {rouge_score}"

def simple_rouge_l_score(generated_about: str, ground_truth_description: str) -> float:
    rouge = ROUGE()
    rouge1_score = rouge.get_Rouge1(string_1=generated_about, string_2=ground_truth_description)
    rouge2_score = rouge.get_Rouge2(string_1=generated_about, string_2=ground_truth_description)
    rougeL_score = rouge.get_RougeL(string_1=generated_about, string_2=ground_truth_description)
    print(f"ROUGE-1: {rouge1_score:.3f}, ROUGE-2: {rouge2_score:.3f}, ROUGE-L: {rougeL_score:.3f}")
    return rougeL_score
