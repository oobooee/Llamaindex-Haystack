from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI, AzureOpenAI
from llama.index.llms.azure_openai import AzureOpenAI

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import ReActAgent
from tools.tools import save_extracted_text, get_summarizer_prompt, save_summary, calculate_rouge_score, save_new_summarizer_prompt
from dotenv import load_dotenv
load_dotenv()
import os
from llama_index.core.prompts import PromptTemplate

from prompts.prompt_agents import (
    COMBINE_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    TEACHER_AWARENESS_PROMPT,
    TEACHER_PROMPT,
    TEACHER_PROMPT_C,
    TEACHER_PROMPT_D,
    TEACHER_PROMPT_REACT
)

custom_teacher_prompt = PromptTemplate(
    template=TEACHER_PROMPT_REACT,
)
custom_summarizer_prompt = PromptTemplate(
    template=INITIAL_SUMMARIZER_PROMPT,
)
custom_extractor_prompt = PromptTemplate(
    template=EXTRACTOR_PROMPT,  
)
custom_combiner_prompt = PromptTemplate(
    template=COMBINE_PROMPT,
)



llm_mini = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
llm= OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

#llm = Ollama(model=os.getenv("OLLAMA_MODEL"))

extractor_agent = ReActAgent(
    name="ExtractorAgent",
    description="Extracts the relvant infos from a README.",
    llm=llm_mini,
    tools=[save_extracted_text],
    can_handoff_to=["SummarizerAgent"],
    verbose=False,
    temperature=0.3
)

summarizer_agent = ReActAgent(
    name="SummarizerAgent",
    description="Summarizes the extracted text.",
    llm=llm_mini,
    tools=[get_summarizer_prompt, save_summary],
    can_handoff_to=["TeacherAgent"],
    verbose=False,
    temperature=0.3
)

teacher_agent = ReActAgent( 
    name="TeacherAgent",
    description="Extracts or evaluates keywords and improves summary.",
    llm=llm,
    tools=[calculate_rouge_score, save_new_summarizer_prompt],
    can_handoff_to=["SummarizerAgent", "PromptCombinerAgent"],
    verbose=False,
    temperature=0.3
    
)

combiner_agent = ReActAgent(
    name="PromptCombinerAgent",
    description="Combines the output into a final optimized summarization prompt.",
    llm=llm,
    tools=[],
    can_handoff_to=[],
    verbose=False,
    temperature=0.2
)
# react-agent = ReActAgent(
#     name="ReactAgent",
#     description="A ReAct agent that can adapt its behavior based on the context.",
#     system_prompt=TEACHER_AWARENESS_PROMPT,
#     llm=llm,
#     tools=[calculate_rouge_score, store_new_summarizer_prompt],
#     can_handoff_to=["SummarizerAgent", "PromptCombinerAgent"],
#     verbose=True,
#     temperature=0.7
# )

# 2. Workflow sequenziale

extractor_agent.update_prompts({"react_header": custom_extractor_prompt})
summarizer_agent.update_prompts({"react_header": custom_summarizer_prompt})
teacher_agent.update_prompts({"react_header": custom_teacher_prompt})
combiner_agent.update_prompts({"react_header": custom_combiner_prompt})




prompt_dict = extractor_agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")

prompt_dict = summarizer_agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")

prompt_dict = teacher_agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")

prompt_dict = combiner_agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")    

workflow = AgentWorkflow(  # ✅
    agents=[
        extractor_agent,
        summarizer_agent,
        teacher_agent,
        combiner_agent,
    ],
    
    root_agent="ExtractorAgent"
)

