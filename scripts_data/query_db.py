from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
from typing import Dict, Any

class OmicsRAGPipeline:
    def __init__(self, vector_store_path: str):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store_path = vector_store_path
        self.db = None
        self.qa_chain = None

    def load_vector_store(self):
        self.db = DeepLake(dataset_path=self.vector_store_path, embedding=self.embeddings, read_only=True)

    def create_rag_pipeline(self):
        if self.db is None:
            raise ValueError("Vector store is not loaded. Call load_vector_store() first.")

        retriever = self.db.as_retriever(search_kwargs={"k": 5})

        custom_prompt_template = """
        You are an AI assistant specialized in omics data analysis. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Question: {question}


        Answer:"""

        PROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["question"]
        )

        llm = ChatOllama(
            model="mathstral:7b-v0.1-q6_K",
            temperature=0.2,
            max_tokens=512,
            top_p=0.5,
        )
        print("Creating RAG pipeline...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            raise ValueError("RAG pipeline is not created. Call create_rag_pipeline() first.")
        print("Question: ", question)
        result = self.qa_chain({"query": question})
        print("Result: ", result)
        return result

def query_vector_db(question: str):
    vector_store_path = "./omics_vector_store"
    rag_pipeline = OmicsRAGPipeline(vector_store_path)
    
    rag_pipeline.load_vector_store()
    rag_pipeline.create_rag_pipeline()

    # Example usage
    result = rag_pipeline.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print("Source documents:")
    for doc in result['source_documents']:
        print(f"- {doc.metadata['id']}: {doc.page_content[:100]}...")

