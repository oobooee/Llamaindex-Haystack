# Testing Multi-Agent Architectures with LLMs using LlamaIndex and Haystack

A multi-agent architecture for exploring LLM-based summarization tasks using both LlamaIndex and Haystack frameworks.

## Requirements

- A dedicated Conda environment with either Haystack or LlamaIndex installed  
- Python 3.10 or later

## Configuration

The project uses a `.env` file to manage LLM API credentials and endpoint configuration.  
The University of L’Aquila (UNIVAQ) provides an Azure for Students license with €100 in free credits, allowing the use of Azure OpenAI services.

Example `.env` file:

```env
# Azure OpenAI (UNIVAQ - Azure for Students)
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>

# OpenAI (alternative)
OPENAI_API_KEY=<your-openai-key>
```

Replace the placeholders with your actual credentials before running the code.

## Execution

Each Python file in this repository is self-contained and performs a specific function or test.  
You can run them individually, for example:

```bash
python main9-finale_w_debug.py
```
