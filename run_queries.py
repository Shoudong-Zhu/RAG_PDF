# import os

# from groq import Groq
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings.sentence_transformer import \
#     SentenceTransformerEmbeddings
# from langchain_openai import (AzureChatOpenAI, AzureOpenAI,
#                               AzureOpenAIEmbeddings)
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec

# # Azure OpenAI credentials
# azure_client = AzureOpenAI(
#     api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_version="2023-03-15-preview"
# )
# deployment_name = "gpt-4o-2"

# # Groq initialization
# groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

# def main():
#     """
#     This is the main function that runs the application. It initializes the Azure OpenAI client and the SentenceTransformer model,
#     retrieves relevant excerpts from documents based on the user's question,
#     generates a response to the user's question using a pre-trained model, and displays the response.
#     """

#     embedding_function = AzureOpenAIEmbeddings(model="text-embedding-ada-002")

#     # Initialize Pinecone
#     pinecone_api_key = os.getenv('PINECONE_API_KEY')
#     pinecone_index_name = "uber-quarter"
#     pc = Pinecone(api_key=pinecone_api_key)

#     # Create index if it doesn't exist
#     if pinecone_index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=pinecone_index_name,
#             dimension=1536,  # ensure the dimension matches the embedding model
#             metric='cosine',
#             spec=ServerlessSpec(
#                 cloud='aws',
#                 region='us-east-1'
#             )
#         )

#     docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)

    
#     # Initialize the LangChain object for chatting with the LLM without knowledge from Pinecone
#     llm = AzureChatOpenAI(
#         api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
#         azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
#         model_name="gpt-4o-2",
#         temperature=0.0,
#         api_version="2023-03-15-preview"  # API version
#     )

    
#     index = pc.Index(pinecone_index_name)
#     namespace = "wondervector5000"
#     for ids in index.list(namespace=namespace):
#         query = index.query(
#             id=ids[0], 
#             namespace=namespace, 
#             top_k=1,
#             include_values=True,
#             include_metadata=True
#         )
#         print(query)
#     # Initialize the LangChain object for retrieving information from Pinecone
#     knowledge = PineconeVectorStore.from_existing_index(
#         index_name=pinecone_index_name,
#         namespace="wondervector5000",
#         embedding=embedding_function
#     )

#     # Initialize the LangChain object for chatting with the LLM with knowledge from Pinecone
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=knowledge.as_retriever()
#     )

#     # Display the title and introduction of the application
#     print("Uber Quarterly Report RAG")
#     print("""
#     Welcome! Ask questions about Uber's quarterly reports, like "What were Uber's earnings in the last quarter?" or "Describe Uber's growth strategy." The app matches your question to relevant excerpts from the reports and generates a response using a pre-trained model.
#     """)

#     while True:
#         # Get the user's question
#         user_question = input("Ask a question about Uber's quarterly report (or type 'exit' to quit): ")

#         if user_question.lower() in ['exit', 'quit']:
#             print("Exiting the application. Goodbye!")
#             break

#         if user_question:
#             # Use Pinecone knowledge
#             print("Chat with knowledge:")
#             try:
#                 response_with_knowledge = qa.invoke(user_question).get("result")
#                 print(response_with_knowledge)
#             except Exception as e:
#                 print(f"Error during knowledge retrieval: {e}")

#             # Without Pinecone knowledge
#             print("\nChat without knowledge:")
#             try:
#                 response_without_knowledge = llm.invoke(user_question).content
#                 print(response_without_knowledge)
#             except Exception as e:
#                 print(f"Error during LLM response generation: {e}")

# if __name__ == "__main__":
#     main()


import os

from groq import Groq
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings
from langchain_openai import (AzureChatOpenAI, AzureOpenAI,
                              AzureOpenAIEmbeddings)
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Azure OpenAI credentials
azure_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2023-03-15-preview"
)
deployment_name = "gpt-4o-2"

# Groq initialization
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

def generate_groq_response(user_question, relevant_excerpts):
    """
    This function generates a response using Groq for LLM inference.
    Parameters:
    user_question (str): The question asked by the user.
    relevant_excerpts (str): A string containing the most relevant excerpts from the documents.
    Returns:
    str: A string containing the response to the user's question.
    """
    system_prompt = '''
    You are a financial analyst. Given the user's question and relevant excerpts from the documents, answer the question by including direct quotes from the excerpts.
    '''
    
    try:
        # Generate a response to the user's question using Groq
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {user_question}\n\nRelevant Excerpts:\n\n{relevant_excerpts}"}
            ],
            model='llama3-8b-8192'
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        print(f"Error during Groq response generation: {e}")
        return str(e)

def main():
    """
    This is the main function that runs the application. It initializes the Azure OpenAI client and the SentenceTransformer model,
    retrieves relevant excerpts from documents based on the user's question,
    generates a response to the user's question using a pre-trained model, and displays the response.
    """

    embedding_function = AzureOpenAIEmbeddings(model="text-embedding-ada-002")

    # Initialize Pinecone
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = "uber-quarter"
    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if it doesn't exist
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,  # ensure the dimension matches the embedding model
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)

    # Initialize the LangChain object for chatting with the LLM without knowledge from Pinecone
    llm = AzureChatOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        model_name="gpt-4o-2",
        temperature=0.0,
        api_version="2023-03-15-preview"  # API version
    )

    index = pc.Index(pinecone_index_name)
    namespace = "wondervector5000"
    for ids in index.list(namespace=namespace):
        query = index.query(
            id=ids[0], 
            namespace=namespace, 
            top_k=1,
            include_values=True,
            include_metadata=True
        )
        print(query)

    # Initialize the LangChain object for retrieving information from Pinecone
    knowledge = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index_name,
        namespace="wondervector5000",
        embedding=embedding_function
    )

    # Initialize the LangChain object for chatting with the LLM with knowledge from Pinecone
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge.as_retriever()
    )

    # Display the title and introduction of the application
    print("Uber Quarterly Report RAG")
    print("""
    Welcome! Ask questions about Uber's quarterly reports, like "What were Uber's earnings in the last quarter?" or "Describe Uber's growth strategy." The app matches your question to relevant excerpts from the reports and generates a response using a pre-trained model.
    """)

    while True:
        # Get the user's question
        user_question = input("Ask a question about Uber's quarterly report (or type 'exit' to quit): ")

        if user_question.lower() in ['exit', 'quit']:
            print("Exiting the application. Goodbye!")
            break

        if user_question:
            # Retrieve relevant excerpts from Pinecone
            try:
                relevant_excerpts = "\n\n".join([doc.page_content for doc in knowledge.similarity_search(user_question)])
                print(f"Relevant Excerpts:\n{relevant_excerpts}")
            except Exception as e:
                print(f"Error retrieving relevant excerpts: {e}")
                relevant_excerpts = ""
            print("-----------------------------------------")

            # Use Pinecone knowledge with Azure OpenAI
            print("Chat with knowledge (Azure OpenAI):")
            try:
                response_with_knowledge = qa.invoke(user_question).get("result")
                print(response_with_knowledge)
            except Exception as e:
                print(f"Error during Azure OpenAI knowledge retrieval: {e}")
            print("-----------------------------------------")
            
            # Use Pinecone knowledge with Groq
            print("Chat with knowledge (Groq):")
            try:
                response_with_groq = generate_groq_response(user_question, relevant_excerpts)
                print(response_with_groq)
            except Exception as e:
                print(f"Error during Groq response generation: {e}")
            print("-----------------------------------------")
            
            # Without Pinecone knowledge
            print("\nChat without knowledge:")
            try:
                response_without_knowledge = llm.invoke(user_question).content
                print(response_without_knowledge)
            except Exception as e:
                print(f"Error during LLM response generation: {e}")

if __name__ == "__main__":
    main()
