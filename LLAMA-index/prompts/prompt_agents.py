EXTRACTOR_PROMPT = """
Your task is to shorten and extract only the introduction and description information from the README of a Github repository. You are given the following README text from a GitHub repository:
# Steps
- **Identify the structure of the repository**: The README file is a structure text file that might contains many sections such as introduction, description, installation, contributing, license,...
- **Remove all sections that are not relevant to the introduction or description of the repository**: Irrelevant sections might include technical guidance (installing/running/specification... instruction), repository structure/table of contents, contributions/references,...
- **Remove all unnecessary links/tags**: Identify all links/tags that DO NOT contribute to the description of the repository. You must remove all of these reference links and tags.
- **Return only text that is relevant to the description of the repository**: The output should only contains the text that is relevant to the introduction/description of the repository, including the project name/title, project tagline/functional description/purpose statement/overview. DO NOT include any output identifications such as: "Here's the ..." or "Extracted README:"
- **You must store this output into the shared context under the key 'extracted_text' for the next agent.**
- **Once finished, pass control to SummarizerAgent**: After extracting the relevant information, hand off to the SummarizerAgent."
"""


INITIAL_SUMMARIZER_PROMPT = """
Summarize the following extracted text from a Github repository README into a short term/phrase introducing the repository:
<EXTRACTED_README>
$extracted_text
</EXTRACTED_README>

The output should include only a short term/phrase introducing the repository.
Once finished, hand off to TeacherAgent**: After summarizing the extracted text, save the summary with your function and than hand off to the TeacherAgent with the summary as output

"""

TEACHER_AWARENESS_PROMPT = """
You are a professional Prompt Engineer. 
# Steps:
- ** first print all the variables described below**

<EXTRACTED_TEXT>
$extracted_text
</EXTRACTED_TEXT>
<GENERATED_ABOUT>
$generated_about
</GENERATED_ABOUT>
<ROUGE_SCORE>
$rouge_score
</ROUGE_SCORE>
<GROUND_TRUTH DESCRIPTION>
$description
</GROUND_TRUTH DESCRIPTION>
<SUMMARIZER_PROMPT>
$summarizer_prompt
</SUMMARIZER_PROMPT>

- Calculate the ROUGE score and go on
<ROUGE_SCORE>
$rouge_score
</ROUGE_SCORE>
"""

TEACHER_PROMPT_C = """
You are a professional Prompt Engineer working with a Large Language Model (LLM) to help developers automatically generate a short description term/phrase from an extracted README of a GitHub repository.

Your role is to **evaluate the current summarization prompt**, review the quality of the generated summary (compared to the ground truth), and produce an improved version of the summarization prompt for the next iteration.

You MUST generate a new version of the summarizer prompt **at each cycle**, even if the current one seems reasonable. The new prompt should retain most of the original structure, but incorporate any opportunity for improvement, refinement, clarification, or better alignment with the target output. Minor wording changes, better instructions, or even reordering steps are acceptable.

# Constraints:

- DO NOT include reasoning or explanation in your final output.
- DO NOT mention or reference "ground truth" in the final prompt.
- DO NOT use markers like "New Prompt:", "Here is the prompt", etc.
- Your output must be ONLY the final optimized prompt string.

# Workflow Steps:

1. **First step (mandatory):**  
   Call the tool `calculate_rouge_score` (no arguments). This evaluates the ROUGE score between the current generated summary and the ground truth description.

2. **Analyze the data:**  
   Use the following input:
   <EXTRACTED_TEXT>
   $extracted_text
   </EXTRACTED_TEXT>

   <GROUND_TRUTH_DESCRIPTION>
   $ground_truth_description
   </GROUND_TRUTH_DESCRIPTION>

   <GENERATED_ABOUT>
   $generated_about
   </GENERATED_ABOUT>

   <ROUGE_SCORE>
   $rouge_score
   </ROUGE_SCORE>

3. **Review the current summarizer prompt:**
   <SUMMARIZER_PROMPT>
   $summarizer_prompt
   </SUMMARIZER_PROMPT>

   Use the evaluation above to understand where it might be lacking (e.g., not focusing on tagline, not being concise, being unclear).

   
4. **Create a new summarizer prompt ($new_summarizer_prompt):**  
   You MUST make a small improvement to the `$summarizer_prompt`. 
   The new prompt must be better aligned with the expected summarization behavior. 
   Focus on clarity, precision, or ordering of instructions. Even if minor, a change is required.
   The prompt at the end must have the string "when finished pass control to TeacherAgent."
   
5.**Call the tool `save_new_summarizer_prompt`  passing the `$new_summarizer_prompt` as argument.

6. **Decide next step:**
   - If `rouge_score > 0.7`, call hand off to `PromptCombinerAgent`.
   - Otherwise, call hand off to `SummarizerAgent`, passing `$new_summarizer_prompt` as argument.

"""
TEACHER_PROMPT_D = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short description term/phrase containing the key concept/idea from an extracted text of the README of a GitHub repository. Your task is to modify and improve the current prompt of the LLM based on the result of testing on data that includes a README and a ground truth description.

# Step 1 – Analyze the input
Analyze the following data:

<EXTRACTED_TEXT>
$extracted_text
</EXTRACTED_TEXT>

<GROUND_TRUTH_DESCRIPTION>
$description
</GROUND_TRUTH_DESCRIPTION>

<GENERATED_DESCRIPTION>
$generated_about
</GENERATED_DESCRIPTION>

<ROUGE_SCORE>
$rouge_score
</ROUGE_SCORE>

<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>

# Step 2 – Evaluate and refine
- Compare the beginning of the extracted text with the ground truth description.  
  If the ground truth is already present as a tagline or purpose statement, include an instruction to prefer it.
- Modify the prompt **minimally**, preserving its structure and intention.
- DO NOT mention the ground truth in the new prompt.
- DO NOT include reasoning, explanations, or any output identifiers like “New Prompt:”.
- The output must be the **raw string** of the new prompt.
- The new prompt MUST end with:  
  `"Once you have generated the summary, hand off to TeacherAgent."`

---

Thought: I will evaluate the summary quality with the ROUGE-L score.
Action: calculate_rouge_score
Action Input: {}

Thought: I will now slightly rewrite the summarization prompt to improve it while preserving its original intent.
Action: save_new_summarizer_prompt
Action Input: {
  "new_summarizer_prompt": "Please analyze the following extracted content from a GitHub repository README and write a concise, meaningful expression that introduces the repository. Prefer existing taglines or overview phrases when available. Once you have generated the summary, hand off to TeacherAgent."
}

Thought: I will hand off to the SummarizerAgent to regenerate the summary using the improved prompt.
Action: hand off
Action Input: {
  "to_agent": "SummarizerAgent",
  "reason": "Retry summarization using the new prompt",
  "new_summarizer_prompt": "Please analyze the following extracted content from a GitHub repository README and write a concise, meaningful expression that introduces the repository. Prefer existing taglines or overview phrases when available. Once you have generated the summary, hand off to TeacherAgent."
}
"""


TEACHER_PROMPT_REACT = """
You are a professional Prompt Engineer using a Large Language Model (LLM) to optimize summarization prompts for a system that extracts concise phrases from GitHub README files.

Your goal is to review the current performance of the summarization prompt generated by the SummarizerAgent and improve it based on the generated output and a known reference description.

You will proceed in a ReAct style: think step-by-step, perform actions via tools, observe results, and update your thinking accordingly. You are allowed to take multiple steps to reach your final decision.

DO NOT include any reasoning, comments, or headers in your final output. The final answer must be a raw string: the new improved prompt. All other reasoning must happen as thoughts and tool uses.

## Context

The following variables may be available to you:
- `extracted_text`: content extracted from the README
- `ground_truth_description`: the ideal description for the repo
- `generated_about`: the summary currently produced by the SummarizerAgent
- `summarizer_prompt`: the current prompt used by the SummarizerAgent


## Task

Think through the following:

1. First, **calculate the ROUGE score** by calling the tool `calculate_rouge_score(ground_truth_description, generated_about)`. This is always your first step. The `rouge_score` is the ROUGE-L score comparing `generated_about` to the ground_truth_description

2. Then, **analyze** the `summarizer_prompt`, `generated_about`, and `ground_truth_description` to see what guidance is missing or unclear. Focus on alignment, conciseness, and fidelity to purpose.

3. Use your analysis to **propose a refined version** of the summarizer prompt. The new prompt must keep most of the original structure, but include at least one small meaningful improvement: better wording, clearer instructions, or reordering of steps.

4. **Save the new prompt** by calling `save_new_summarizer_prompt(new_prompt)`.

5. **Choose the next step**:
   - If `rouge_score > 0.7`, hand off to `PromptCombinerAgent`.
   - Otherwise, hand off to `SummarizerAgent`, with the new_prompt as input.

## Reminder

- NEVER output explanations in the final result.
- NEVER say "Here's the prompt".
- ALWAYS the new_prompt must be the prompt only with this string attached as is: 
  `"Once you have generated the summary, hand off to TeacherAgent."`


"""


TEACHER_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the README of a Github repository. Your task is to modify and improve the current prompt of the LLM based on the result of testing on a data include a README and a ground truth description. As the new prompt will not include the ground truth description, DO NOT mention about the ground truth description in the new prompt. DO NOT include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the new prompt for the LLM

# Steps:
- **First mandatory, calculate the ROUGE score**: You have to calculate the rouge_score that compare the generated_about and the description.
- **Analyze the data for testing**: Analyze the following data include an extracted_text from a README and a ground truth description from a GitHub repository:
- **Review the current result**: Review the generated about  using the extracted text its ROUGE score on the ground truth description to identify improvements that could be made:
- **Prioritize extracting existing tagline/functional description/purpose statement/overview**: Compare the text from the beginning of the extracted text from README and the ground truth description. If the ground truth description is already existed in this extracted text as a tagline/functional description/purpose statement/overview, you must include in the new prompt the instruction to prioritize using it.
- **Modify the current summarizer prompt**: Identify mistakes and lacking instructions in the current summarizer prompt from the result of the above review. You should preserve the current summarizer prompt as much as possible and only make small changes to the prompt based on the identified mistakes and lacking instructions.
- **You must store this output into the shared context under the key 'summarizer_prompt' and hand off to the the next agent.**
"""

TEACHER_PROMPT_ORIG = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the README of a Github repository. Your task is to modify and improve the current prompt of the LLM based on the result of testing on a data include a README and a ground truth description. As the new prompt will not include the ground truth description, DO NOT mention about the ground truth description in the new prompt. DO NOT include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the new prompt for the LLM
 
# Steps:
- **Calculate the ROUGE score**: You have to calculate the rouge_score that compare the generated_about and the description.
- **Analyze the data for testing**: Analyze the following data include an extracted text from a README and a ground truth description from a GitHub repository:
<EXTRACTED_TEXT>
$extracted_text
</EXTRACTED_TEXT>
 
<GROUND_TRUTH DESCRIPTION>
$description
</GROUND_TRUTH DESCRIPTION>
- **Review the current result**: Review the generated description using the extracted text its ROUGE score on the ground truth description to identify improvements that could be made:
<GENERATED_DESCRIPTION>
$generated_about
</GENERATED_DESCRIPTION>
<ROUGE_SCORE>
$rouge_score
</ROUGE_SCORE>
- **Prioritize extracting existing tagline/functional description/purpose statement/overview**: Compare the text from the beginning of the extracted text from README and the ground truth description. If the ground truth description is already existed in this extracted text as a tagline/functional description/purpose statement/overview, you must include in the new prompt the instruction to prioritize using it.
- **Modify the current prompt**: Identify mistakes and lacking instructions in the current prompt from the result of the above review. You should preserve the current prompt as much as possible and only make small changes to the prompt based on the identified mistakes and lacking instructions.
<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>
- **You must store this 'summarizer_prompt' and then hand off to the the next agent.**
"""




COMBINE_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the README of a Github repository. Your task is to combine several candidate prompts for the LLM into a final prompt.
 
# Steps:
- **Review all candidate prompts**: Analyze the following prompts to identify common parts to be included in the final prompt and also includes specific details or conditional key points from these prompts to be included in the final prompt
<CANDIDATE_PROMPTS>
$summarizer_list
</CANDIDATE_PROMPTS>
- **Generate a final prompt**: Based on the common parts and conditional key points, generate a final prompt for the LLM.

# Output Format:
Do not include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the prompt for the LLM
"""

ANALYSIS_PROMPT = """
A Large Language Model is used to auto-summarize Github README section and generate a concise description for a Github repository. Your task is to analyze its result to provide a short advice on how to improve the generated description. You will be provided with the actual and the generated description section of a Github repository and the ROUGE score between them:
<ACTUAL_DESCRIPTION>
$ground_truth
</ACTUAL_DESCRIPTION>
 
<GENERATED_DESCRIPTION>
$generated_about
</GENERATED_DESCRIPTION>
 
<ROUGE_SCORE>
$score
</ROUGE_SCORE>
 
# Steps:
- List the differences between the actual and the generated description section that results in the ROUGE score.
- Choose one main reason among the differences that best represents the ROUGE score.
- The output must be only one short advise on how to improve the generated description from the README base on that main reason.
# Output Format:
1 concise and short advice sentence
"""

ANALYSIS_SUMMARIZER_PROMPT = """
A Large Language Model is used to auto-summarize Github README section and generate a concise description for a Github repository. You are given the following evaluating results on a dataset comparing Github repository descriptions generated from a detail README by the Large Language Model and the actual descriptions:
<ANALYSIS_RESULT>
$analysis_result
</ANALYSIS_RESULT>
# Steps:
- Review the overall tendency of the analysis results.
- Summary one main point that best represents the analysis results.
- Give only one advise on how to improve the generated description.
# Output Format:
1 concise and short advice sentence
"""

SEQUENTIAL_TEACHER_PROMPT = """
You are a professional Prompt Engineer. You are using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from the README of a Github repository. The LLM include a prompt to extract the relevant information from the README and a prompt to generate a short description term/phrase contain key concept/idea. Your task is to optimize the current prompt to improve the performance of the LLM.
<CURRENT_PROMPT>
$current_summarizer_prompt
</CURRENT_PROMPT>
 
You should use the following advising analysis to help you optimize the prompts of the two agents:
<ANALYSIS_RESULT>
$current_analysis
</ANALYSIS_RESULT>
 
Here is an example of prompts that have good performance. FOLLOW this example to optimize the prompt of the LLM.
<GOOD_PROMPT>
$best_summarizer_prompt
</GOOD_PROMPT>
<SCORE>
$best_score
</SCORE>
<ADVISING_ANALYSIS>
$best_analysis_result
</ADVISING_ANALYSIS>
 
Here is an example of prompts that have bad performance. AVOID the mistakes of this example to optimize the prompt of the LLM.
<BAD_PROMPT>
$worst_summarizer_prompt
</BAD_PROMPT>
<SCORE>
$worst_score
</SCORE>
<ADVISING_ANALYSIS>
$worst_analysis_result
</ADVISING_ANALYSIS>
 
You should preserve the current prompt as much as possible and only make small changes to the prompt based on the advising analysis. You must include in the detail part that the description is "A shortest term or phrase include only the concept/idea of the repository, without any explanations or details". The answer must only include the new prompt for the LLM
 
# Output Format:
Prompt: <Prompt>
"""

OPTIMIZED_SUMMARIZER_PROMPT = """Summarize the following extracted text from a Github repository README into a short term/phrase introducing the repository. You should always try to look for an existing project tagline, functional description, purpose statement, or overview at the beginning of the extracted text and prioritize using this exact sentence or phrase. Ensure the output closely matches the original phrasing when appropriate. The output should include only a brief term/phrase introducing the repository.  
<EXTRACTED_README>  
$extracted_text  
</EXTRACTED_README>"""
