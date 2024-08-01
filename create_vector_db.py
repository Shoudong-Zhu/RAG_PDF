import os

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec


def create_vector_db(markdown_file):
    # Load the parsed markdown document
    with open(markdown_file, 'r') as f:
        markdown_content = f.read()

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunk_texts = splitter.split_text(markdown_content)

    # Create Document objects from chunks
    documents = [Document(page_content=chunk) for chunk in chunk_texts]

    # Log the documents being embedded
    for i, doc in enumerate(documents):
        print(f"Document {i}: {doc.page_content[:200]}...")

    # Create embeddings for the chunks using Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")

    # Initialize Pinecone
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "uber-quarter"
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # adjust to the correct embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    docsearch = PineconeVectorStore.from_documents(documents=documents, 
                                                   index_name=index_name, 
                                                   embedding=embeddings,
                                                   namespace="wondervector5000")

    # docsearch.save("uber-quarter")

    return docsearch
