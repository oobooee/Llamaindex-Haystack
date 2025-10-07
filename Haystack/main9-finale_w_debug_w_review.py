import os
import re
import pandas as pd
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.tools import tool
from tools.others_orig import save_parallel_train_result  # o dove hai definito la funzione
from prompt_orig import EXTRACTOR_PROMPT, INITIAL_SUMMARIZER_PROMPT, TEACHER_PROMPT, COMBINE_PROMPT, TEACHER_PROMPT_ALPHA, ANALYSIS_PROMPT, ANALYSIS_SUMMARIZER_PROMPT, SEQUENTIAL_TEACHER_PROMPT
from tools.others_orig import save_evaluation_result  # Assicurati che sia definita lì
from datetime import datetime as dt
from metric.rouge import ROUGE


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


analysis_agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4o",
        #generation_kwargs={"temperature": 0.3},
        api_key=Secret.from_env_var("OPENAI_API_KEY")
    ),
    tools=[noop]  # non servono tool per ora
)

analysis_summarizer_agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4o",
        #generation_kwargs={"temperature": 0.3},
        api_key=Secret.from_env_var("OPENAI_API_KEY")
    ),
    tools=[noop]
)

seq_teacher_agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4o",
        generation_kwargs={"temperature": 0.7},
        api_key=Secret.from_env_var("OPENAI_API_KEY")
    ),
    tools=[noop]
)



# Prompts
extractor_agent_prompt = ChatMessage.from_system(EXTRACTOR_PROMPT)
summarizer_agent_prompt_start = ChatMessage.from_system(INITIAL_SUMMARIZER_PROMPT)
analysis_agent_prompt = ChatMessage.from_system(ANALYSIS_PROMPT)
analysis_summarizer_agent_prompt = ChatMessage.from_system(ANALYSIS_SUMMARIZER_PROMPT)
seq_teacher_agent_prompt = ChatMessage.from_system(SEQUENTIAL_TEACHER_PROMPT)

# ------------------------- TRAIN PHASE ------------------------- 
train_df = pd.read_csv("data/train_data.csv")
print(f"\033[92m[INFO] Dataset loaded with {len(train_df)} rows.\033[0m")



max_iterations = 15

best_prompts = []
train_debug = []

# best_summarizer_prompt = summarizer_agent_prompt_start
# summarizer_prompt = summarizer_agent_prompt_start
# worst_summarizer_prompt  = summarizer_agent_prompt_start
# best_score = 0
# best_analysis_summary = ""
# best_extractor_prompt = extractor_agent_prompt
# best_summarizer_prompt = summarizer_prompt
# worst_score = 1
# worst_analysis_summary = ""
# worst_summarizer_prompt = summarizer_prompt
# all_prompts_history = []

# for iteration in range(max_iterations):
#     print(f"Iteration #{iteration}:")
#     iteration_analysis = []
#     data_debug = []
#     total_rouge_score = 0
#     total_rouge1 = 0
#     total_rouge2 = 0
#     for idx, row in train_df.iterrows():
#         print(f"\n\033[94m🔄 [ROW {idx+1}]\033[0m")
#         readme = row["readme"]
#         description = row["description"]
#         iteration_debug = []
        
#         print(f"\n\033[95m[INFO] Iteration {iteration}...\033[0m")
        
#         extractor_result = extractor_agent.run(messages=[
#             extractor_agent_prompt,
#             ChatMessage.from_user(readme)
#         ])
#         extracted_text = extractor_result["messages"][-1].text

#         print(f"\033[93m[EXTRACTOR- EXTRACTED TEXT]\033[0m\n{extracted_text}")

#         print(f"\033[92m[SUMMARIZER PROMPT]\033[0m\n{summarizer_prompt.text}")
#         summ_input=[
#             summarizer_prompt,
#             ChatMessage.from_user(extracted_text)
#         ]
#         summarizer_result = summarizer_agent.run(summ_input)

#         generated_about = summarizer_result["messages"][-1].text
#         print(f"\033[92m[SUMMARIZER -GENERATED ABOUT]\033[0m\n{generated_about}")

#         # Calcola tutte le metriche
#         rougeL_score = ROUGE().get_RougeL(string_1=generated_about, string_2=description)
#         total_rouge_score += rougeL_score
#         rouge1_score = ROUGE().get_Rouge1(string_1=generated_about, string_2=description)
#         total_rouge1 += rouge1_score
#         rouge2_score = ROUGE().get_Rouge2(string_1=generated_about, string_2=description)
#         total_rouge2 += rouge2_score
#         print(f"\033[92m[ROUGE1 SCORE]\033[0m {rouge1_score}")
#         print(f"\033[92m[ROUGE2 SCORE]\033[0m {rouge2_score}")
#         print(f"\033[92m[ROUGEL SCORE]\033[0m {rougeL_score}")
#         ground_truth = description
#         analysis_input = f"""
#             <ACTUAL_DESCRIPTION>
#             {ground_truth}
#             </ACTUAL_DESCRIPTION>
            
#             <GENERATED_DESCRIPTION>
#             {generated_about}
#             </GENERATED_DESCRIPTION>
            
#             <ROUGE_SCORE>
#             {rougeL_score}
#             </ROUGE_SCORE>"""

        
#         analysis_print = analysis_agent.run(messages=[
#             analysis_agent_prompt,
#             ChatMessage.from_user(analysis_input)
#         ])
#         analysis_reasoning = analysis_print["messages"][-1].text
#         print(f"\033[94m[ANALYZER - ANALYSIS FEEDBACK]\033[0m\n{analysis_reasoning}")
#         iteration_analysis.append(
#             {
#                 "description": description,
#                 "generated_about": generated_about,
#                 "analysis_reasoning": analysis_reasoning,
#             }
#         )
#         #print(f"\033[88m[ITER AN]\033[0m\n{iteration_analysis}")

#     analysis_summary_input = f"""
#     <ANALYSIS_RESULT>
#     {iteration_analysis}    
#     </ANALYSIS_RESULT>
#     """      
#     prompt_as_  = [
#         analysis_summarizer_agent_prompt,
#         ChatMessage.from_user(analysis_summary_input)
#     ]
#     #print(f"\033[93m[ANALYSIS SUMMARIZER INPUT]\033[0m\n{prompt_as_}")
#     analysis_summary = analysis_summarizer_agent.run(messages=prompt_as_)
#     print(f"\033[93m[ANALYZER SUMMARIZER OUTPUT]\033[0m\n{analysis_summary['messages'][-1].text}")
#     analysis_summary_1 = analysis_summary["messages"][-1].text


#     avg_rouge_score = total_rouge_score / len(train_df)
#     avg_rouge1_score = total_rouge1 / len(train_df)
#     avg_rouge2_score = total_rouge2 / len(train_df)
#     print(f"\033[88m[AVG ROUGE SCORE]\033[0m {avg_rouge_score}")
#     print(f"\033[88m[AVG ROUGE1 SCORE]\033[0m {avg_rouge1_score}")
#     print(f"\033[88m[AVG ROUGE2 SCORE]\033[0m {avg_rouge2_score}")

#     # 6. Update Best/Worst
#     if avg_rouge_score > best_score:
#         best_score = avg_rouge_score
#         best_analysis_summary = analysis_summary_1
#         best_extractor_prompt = extractor_agent_prompt
#         best_summarizer_prompt = summarizer_prompt
#         best_rouge1 = avg_rouge1_score
#         best_rouge2 = avg_rouge2_score

#     if avg_rouge_score < worst_score:
#         worst_score = avg_rouge_score
#         worst_analysis_summary = analysis_summary_1
#         worst_summarizer_prompt = summarizer_prompt

    


#     teacher_input = SEQUENTIAL_TEACHER_PROMPT.format(
#     current_summarizer_prompt=summarizer_prompt.text,
#     current_analysis=analysis_summary_1,
#     best_summarizer_prompt=best_summarizer_prompt.text,
#     best_score=best_score,
#     best_analysis_result=best_analysis_summary,
#     intermediate_summarizer_prompts =all_prompts_history,
#     worst_summarizer_prompt=worst_summarizer_prompt.text,
#     worst_score=worst_score,
#     worst_analysis_result=worst_analysis_summary
# )


#     print(f"\033[93m[TEACHER INPUT]\033[0m\n{teacher_input}")
#     seq_teacher_result = seq_teacher_agent.run(messages=[
#         ChatMessage.from_system(teacher_input)
#     ])

    
#     new_summarizer_prompt = seq_teacher_result["messages"][-1].text
#     print(f"\033[93m[NEW -- TEACHER RESULT]\033[0m\n{new_summarizer_prompt}")
#     cleaned_prompt = re.sub(r"</?Prompt>", "", new_summarizer_prompt)
#     cleaned_prompt = cleaned_prompt.replace("\n\n", "\n").strip()
#     summarizer_prompt = ChatMessage.from_system(cleaned_prompt)
#     all_prompts_history.append(summarizer_prompt.text.strip())



# # Save output
# timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
# print(f"\033[92m[INFO] Best summarizer prompt:\033[0m\n{best_summarizer_prompt.text}")
# print(f"\033[92m[INFO] Best analysis summary:\033[0m\n{best_analysis_summary}")
# print(f"\033[92m[INFO] Best score: {best_score}\033[0m")
# print(f"\033[92m[INFO] Worst summarizer prompt:\033[0m\n{worst_summarizer_prompt.text}")
# print(f"\033[92m[INFO] Worst analysis summary:\033[0m\n{worst_analysis_summary}")
# print(f"\033[92m[INFO] Worst score: {worst_score}\033[0m")


# os.makedirs("result/train", exist_ok=True)
# with open(f"result/train/final_combined_prompt{timestamp}.txt", "w", encoding="utf-8") as f:
#     f.write(best_summarizer_prompt.text)


# ------------------------- TEST PHASE ------------------------- #

print("\n\033[95m[INFO] 🧪 INIZIO FASE DI TEST...\033[0m")

test_df = pd.read_csv("data/test_data.csv")
print(f"\033[92m[INFO] Dataset di test caricato con {len(test_df)} righe.\033[0m")

# Salva il prompt di test usato (per tracciabilità)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("result/test", exist_ok=True)
test_prompt_path = "result/train/final_combined_prompt20250609_151756.txt"
with open(test_prompt_path, "r", encoding="utf-8") as f:
    test_prompt = f.read()
print(f"\033[92m[INFO] Test prompt caricato correttamente:\033[0m\n{test_prompt}")


summarizer_agent_prompt_start = ChatMessage.from_system(test_prompt)


data_debug = []
rouge1_total = 0
rouge2_total = 0
rougeL_total = 0

for idx, row in test_df.iterrows():
    print(f"\n\033[94m[Test Row {idx+1}]\033[0m")

    readme = row["readme"]
    ground_truth = row["description"]

    # Estrazione
    extractor_result = extractor_agent.run(messages=[
        extractor_agent_prompt,
        ChatMessage.from_user(readme)
    ])
    extracted_text = extractor_result["messages"][-1].text
    print(f"\033[93m[EXTRACTED TEXT]\033[0m\n{extracted_text}")
    # Sintesi
    summ_input=[
            summarizer_agent_prompt_start,
            ChatMessage.from_user(extracted_text)
        ]
    print(f"\033[93m[SUMMARIZER INPUT]\033[0m\n{summ_input}")
    summarizer_result = summarizer_agent.run(summ_input)
    print(f"\033[93m[SUMMARIZER RESULT]\033[0m\n{summarizer_result}")
    generated_about = summarizer_result["messages"][-1].text
    print(f"\033[92m[GENERATED ABOUT]\033[0m\n{generated_about}")

    # ROUGE evaluation
    rougeL_score = ROUGE().get_RougeL(string_1=generated_about, string_2=ground_truth)
    rouge1_score = ROUGE().get_Rouge1(string_1=generated_about, string_2=ground_truth)
    rouge2_score = ROUGE().get_Rouge2(string_1=generated_about, string_2=ground_truth)
    rouge1 = rouge1_score
    rouge2 = rouge2_score
    rougeL = rougeL_score
    print(f"\033[96m[ROUGE-1]: {rouge1:.4f}\033[0m")
    print(f"\033[96m[ROUGE-2]: {rouge2:.4f}\033[0m")
    print(f"\033[96m[ROUGE-L]: {rougeL:.4f}\033[0m")
    rouge1_total += rouge1
    rouge2_total += rouge2
    rougeL_total += rougeL

    data_debug.append({
        "description": ground_truth,
        "generated_about": generated_about,
        "rouge1_score": rouge1,
        "rouge2_score": rouge2,
        "rougeL_score": rougeL
    })

# Calcola le medie
avg_rouge1 = rouge1_total / len(data_debug)
avg_rouge2 = rouge2_total / len(data_debug)
avg_rougeL = rougeL_total / len(data_debug)

# Prepara il dizionario di debug
debug_result = {
    "data_debug": data_debug,
    "avg_rouge1_score": avg_rouge1,
    "avg_rouge2_score": avg_rouge2,
    "avg_rougeL_score": avg_rougeL,
}

# Salvataggio finale
save_evaluation_result("result/test", debug_result)
print("\n\033[92m✅ [INFO] Risultati di test salvati correttamente.\033[0m")
