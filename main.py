import os

from dotenv import load_dotenv

from create_vector_db import create_vector_db
from parse_document import parse_pdf_to_markdown
from run_queries import main as run_queries_main

# Load environment variables
load_dotenv()

def main():
    # Check if environment variables are set
    if not os.getenv('LLAMAPARSE_API_KEY'):
        print("Error: LLAMAPARSE_API_KEY is not set.")
        return

    if not os.getenv('GROQ_API_KEY'):
        print("Error: GROQ_API_KEY is not set.")
        return

    if not os.getenv('AZURE_OPENAI_API_KEY'):
        print("Error: AZURE_OPENAI_API_KEY is not set.")
        return

    if not os.getenv('AZURE_OPENAI_ENDPOINT'):
        print("Error: AZURE_OPENAI_ENDPOINT is not set.")
        return

    if not os.getenv('PINECONE_API_KEY'):
        print("Error: PINECONE_API_KEY is not set.")
        return

    # Step 1: Parse document
    markdown_file = parse_pdf_to_markdown('Uber_Quarter.pdf')
    
    if markdown_file is None:
        print("Failed to parse the document. Exiting.")
        return

    # Step 2: Create vector database
    create_vector_db(markdown_file)
    
    # Step 3: Run queries
    run_queries_main()

if __name__ == "__main__":
    main()
