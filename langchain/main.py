from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

FILE_PATH =  "../ccna.pdf"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:latest"
CHROMA_DB =  './chroma_db'
def get_llm():
    model = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.7,
    num_predict=1024,
    )
    return model

def get_embedding():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

def load_pdf():
    try:
        reader = PdfReader(FILE_PATH)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text
    except Exception as e:
        print(e)
        return None
    
def chunking_text(text):
    chunk_size = 500
    chunk_overlap=100
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    return texts


def store_in_vector_database(chunks):
    embedding = get_embedding()
    
    vector_db = Chroma.from_texts(
        texts=chunks,   
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    print("Vector DB stored successfully : ", vector_db)
    return vector_db

def search_query(vector_db,query):
    result = vector_db.similarity_search(
        query=query,
        k=4
    )
    return result 

def ask_llm(data,query,llm):
    context = "\n\n".join(doc.page_content for doc in data)


    prompt = f"""
            You are a helpful tutor.

            Answer the user's question using ONLY the information provided in the context below.

            Instructions:
            - Give a complete answer.
            - Explain in simple words.
            - Include examples if available in the context.
            - If commands are available, show all commands completely.
            - Explain step-by-step.
            - Do not stop in the middle of command output.
            - If answer spans multiple steps, continue until explanation is complete.
            - If answer is not found, say:
            "I couldn't find that in the document."

        Context:
        {context}

        Question:
        {query}
        """
    
    response = llm.stream(prompt)
    print("Bot : ",end="")
    chat_reponse = ""
    for chunk in response:
        print(chunk,end="",flush=True)
        chat_reponse += chunk
    print()

def load_vector_db():
    embedding = get_embedding()

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )



def main():
    llm = get_llm()
    if Path(CHROMA_DB).exists():
        vector_db = load_vector_db()
    else:
        pdf = load_pdf()
        chunks = chunking_text(pdf)
        vector_db = store_in_vector_database(chunks)
    
    while True:
        user_pmt = input("User : ")
        if user_pmt.strip().lower() == "exit":
            break

        searched_data = search_query(vector_db,user_pmt)
       
        ask_llm(searched_data,user_pmt,llm)
if __name__ == "__main__":
        main()