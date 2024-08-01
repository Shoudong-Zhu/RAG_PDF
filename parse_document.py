import os

import nest_asyncio
from llama_parse import LlamaParse

# Apply the nest_asyncio to allow nested event loops
nest_asyncio.apply()

def parse_pdf_to_markdown(pdf_path):
    api_key = os.getenv('LLAMAPARSE_API_KEY')
    
    if not api_key:
        print("Error: LLAMAPARSE_API_KEY is not set.")
        return None

    # Print the API key for debugging (remove this in production)
    print(f"Using LLAMAPARSE_API_KEY: {api_key}")

    # Initialize LlamaParse with your API key
    llama_parse = LlamaParse(
        api_key=api_key, 
        result_type="markdown", 
        verbose=True
    )

    try:
        # Parse the PDF document
        documents = llama_parse.load_data(pdf_path)
        
        if not documents:
            raise ValueError("No documents were parsed. Please check the file and API key.")
        
        # Print the length of the documents for debugging
        print(f"len(documents): {len(documents)}")

        # Extract the parsed content and concatenate into a single string
        markdown_content = "\n\n".join([doc.text for doc in documents])

        # Save the markdown content to a file
        markdown_file = 'parsed_uber_report.md'
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        return markdown_file

    except Exception as e:
        print(f"Error while parsing the file '{pdf_path}': {e}")
        return None
