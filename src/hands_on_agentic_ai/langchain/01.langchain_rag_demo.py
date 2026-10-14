from IPython.terminal.embed import embed
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pathlib import Path
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader

load_dotenv()

# Create the llm
llm = ChatOpenAI(model="gpt-4o-mini")

# Load Docs using the LangChain methods

pdf_loader = PyPDFDirectoryLoader("Docs")
pdf_docs = pdf_loader.load()

# Create Embeddings and load to Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(pdf_docs)

# Older way of loading files using Pypdf instead now we use the langchain in this demo
# def doc_loader(folder_name: str):
#     documents = []
#     directory = Path(folder_name)
#
#     for file_path in directory.rglob("*"):
#         if file_path.suffix in {".pdf", ".txt"}:
#             if file_path.suffix == ".pdf":
#                 reader = PdfReader(str(file_path))
#                 content = "\n".join(page.extract_text() for page in reader.pages)
#                 documents.append({"filePath": str(file_path.relative_to(directory)), "content": content+ "\n"})
#     return documents
#
# def create_context(documents: list):
#     context = ""
#     for document in documents:
#         context += f"___{document['filePath']}___\n {document['content']}\n\n"
#     return context

while True:
    user_Prompt = input("Please enter your question: ")
    retrieved_docs = vectorstore.similarity_search(user_Prompt, k=2)

    context = "\n".join(doc.page_content for doc in retrieved_docs)
    system_prompt = f"You are an AI Agent that can answers questions using RAG {context} from the given context only and not from your trained data."
    messages = [SystemMessage(content=system_prompt)]

    if user_Prompt == "exit":
        break

    messages.append(HumanMessage(content=user_Prompt))
    response = llm.invoke(messages)

    print(response.response_metadata)
    print(response.usage_metadata)
    print(response)


