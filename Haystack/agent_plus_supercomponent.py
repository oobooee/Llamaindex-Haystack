from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters.html import HTMLToDocument
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.dataclasses import ChatMessage
from haystack.core.pipeline import Pipeline
from haystack.tools.tool import Tool
from haystack.core.super_component import SuperComponent
from haystack.tools import ComponentTool
import os

# Build the nested search tool pipeline using haystack.core.pipeline.Pipeline
search_component = Pipeline()

search_component.add_component("search", SerperDevWebSearch(top_k=10,))
search_component.add_component("fetcher", LinkContentFetcher(timeout=3, raise_on_failure=False, retry_attempts=2))
search_component.add_component("converter", HTMLToDocument())
search_component.add_component("builder", ChatPromptBuilder(
    template=[ChatMessage.from_user("""
{% for doc in docs %}
<search-result url="{{ doc.meta.url }}">
{{ doc.content|default|truncate(25000) }}
</search-result>
{% endfor %}
""")],
    variables=["docs"],
    required_variables=["docs"]
))

search_component.connect("search.links", "fetcher.urls")
search_component.connect("fetcher.streams", "converter.sources")
search_component.connect("converter.documents", "builder.docs")

# Wrap in SuperComponent + ComponentTool
super_tool_component = SuperComponent(pipeline=search_component)
search_tool = ComponentTool(
    name="search",
    description="Use this tool to search for information on the internet.",
    component=super_tool_component
)

# Build the Chat Generator
chat_generator = OpenAIChatGenerator()

# Create the Agent
agent = Agent(
    chat_generator=chat_generator,
    tools=[search_tool],
    system_prompt="""
You are a deep research assistant.
You create comprehensive research reports to answer the user's questions.
You use the 'search'-tool to answer any questions.
You perform multiple searches until you have the information you need to answer the question.
Make sure you research different aspects of the question.
Use markdown to format your response.
When you use information from the websearch results, cite your sources using markdown links.
It is important that you cite accurately.
""",
    exit_conditions=["text"],
    max_agent_steps=100,
    raise_on_tool_invocation_failure=False
)

agent.warm_up()

# Answer builder
answer_builder = AnswerBuilder()

# Simulate input
query = "What are the latest updates on the Artemis moon mission?"
messages = [ChatMessage.from_user(query)]

# Run agent
agent_output = agent.run(messages=messages)

# Filter replies with valid 'text' only
valid_replies = [msg for msg in agent_output["messages"] if getattr(msg, "text", None)]

answers = answer_builder.run(query=query, replies=valid_replies)

# Print the result
for answer in answers["answers"]:
    print(answer)