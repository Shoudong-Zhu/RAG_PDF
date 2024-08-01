# Uber Quarterly Report RAG

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions about Uber's quarterly reports using a combination of Pinecone, Groq, and Azure OpenAI. The system retrieves relevant information from parsed documents and uses large language models to generate accurate responses.

## Setup Instructions

### Prerequisites

1. Python 3.7+
2. Pinecone account and API key
3. Groq account and API key
4. Azure OpenAI account and API key
5. Necessary Python packages (listed below)

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd <repository-directory>

```

2. **Create a virtual environment and activate it:**
```bash
python -m venv RAG_PDF_env
source RAG_PDF_env/bin/activate  # On Windows, use `RAG_PDF_env\Scripts\activate`
```
3. **Install the required packages:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
export LLAMAPARSE_API_KEY=<your-llamaparse-api-key>
export PINECONE_API_KEY=<your-pinecone-api-key>
export AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
export AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
export GROQ_API_KEY=<your-groq-api-key>
```