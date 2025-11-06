from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore

INPUT_FILE = "./rag/infoscuola.pdf"
OUTPUT_DB = "./rag/infoscuola.db"

# carico il file
loader = PyPDFLoader(INPUT_FILE)
doc = loader.load()
print(f"Numero di pagine caricate: {len(doc)}")

# divido il testo in chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    is_separator_regex=False
)
chunks = text_splitter.split_documents(doc)
print(f"Numero di chunk creati: {len(chunks)}")

# creo gli embeddings (vettori)
embeddings = OllamaEmbeddings(model="embeddinggemma")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
print("Vettori creati e memorizzati nel vectorstore.")

# salva su file
vectorstore.dump(OUTPUT_DB)