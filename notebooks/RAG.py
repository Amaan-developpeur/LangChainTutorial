import os
import pdfplumber
import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path

BASE_DIR = Path().resolve().parent
DATA_DIR = BASE_DIR / "data"


class DocumentProcessor:
    def __init__(self, DATA_DIR, chunk_size=300, overlap=50, min_chunk_len=30):
        self.data_dir = DATA_DIR
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_len = min_chunk_len

    def _extract_chunks_from_page(self, text, filename, page_num):
        words = text.strip().split()
        chunks = []
        for word in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[word:word + self.chunk_size]
            if len(chunk_words) < self.min_chunk_len:
                continue
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{filename}_p{page_num}_c{word}"
            chunks.append(
                {
                    "filename": filename,
                    "page": page_num,
                    "Retrieval_ID": chunk_id,
                    "text": chunk_text
                }
            )
        return chunks

    def extract_from_pdfs(self):
        all_chunks = []
        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith(".pdf")]

        for filename in pdf_files:
            pdf_path = os.path.join(self.data_dir, filename)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        continue
                    text = text.replace("\n", " ").strip()
                    chunks = self._extract_chunks_from_page(text, filename, page_num)
                    all_chunks.extend(chunks)

        return pd.DataFrame(all_chunks)


class VectorStoreManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_transformer = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_storage = None

    def build_store(self, df):
        documents = [
            Document(
                page_content=row["text"],
                metadata={
                    "filename": row["filename"],
                    "page": row["page"],
                    "Retrieval_ID": row["Retrieval_ID"]
                }
            )
            for index, row in df.iterrows()
        ]
        self.vector_storage = Chroma.from_documents(documents, self.embedding_transformer)
        return self.vector_storage

    def get_retriever(self, k=3):
        if not self.vector_storage:
            raise ValueError("Vector storage has not been initialized. Call build_store() first.")
        return self.vector_storage.as_retriever(search_kwargs={"k": k})


class FinancialQA_Assistant:
    def __init__(self, model="mistral", temperature=0.3):
        self.llm = Ollama(model=model, temperature=temperature)
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            input_key="user_input"
        )
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "context", "user_input"],
            template="""
            Act as a Financial QA Assistant and provide structural insights using 
            the {context} and the {chat_history} to answer: {user_input}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def run(self, user_input, retriever):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        return self.chain.run(user_input=user_input, context=context)


class FinancialQAPipeline:
    def __init__(self, data_dir):
        self.processor = DocumentProcessor(data_dir)
        self.vector_manager = VectorStoreManager()
        self.assistant = FinancialQA_Assistant()

    def run(self, user_query):
        df = self.processor.extract_from_pdfs()
        self.vector_manager.build_store(df)
        retriever = self.vector_manager.get_retriever(k=3)
        return self.assistant.run(user_query, retriever)


if __name__ == "__main__":
    pipeline = FinancialQAPipeline(DATA_DIR)
    answer = pipeline.run("Explain the brief summary of the document")
    print(answer)
