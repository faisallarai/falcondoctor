from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader, DirectoryLoader
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore, LocalFileStore, RedisStore

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatLiteLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def fetch():
  print('loading...')
  loader = DirectoryLoader('./data/', glob="**/*.csv", loader_cls=CSVLoader, show_progress=True, use_multithreading=True)
  documents = loader.load()
  return documents

def chunk(documents):
  print('chunking...')
  tokenizer = GPT2TokenizerFast.from_pretrained("Ransaka/gpt2-tokenizer-fast")
  text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=1024, chunk_overlap=10)
  docs = text_splitter.split_documents(documents)
  return docs

def embed():
  print('embedding...')
  underlying_embeddings = HuggingFaceEmbeddings()
  # store = InMemoryByteStore()
  # store = LocalFileStore("./cache/")
  store = RedisStore(redis_url="redis://localhost:6379")
  cached_embedder = CacheBackedEmbeddings.from_bytes_store(
      underlying_embeddings, store, namespace=underlying_embeddings.model_name
  )
  return cached_embedder

def start(db):
  # Create a retriever
  print('retrieving...')
  retriever = db.as_retriever()

  # Augment the prompt
  print('augmenting...')
  template = """You are an assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.
  Use three sentences maximum and keep the answer concise.
  Question: {question}
  Context: {context}
  Answer:
  """
  prompt = ChatPromptTemplate.from_template(template)

  print(prompt)

  # Create the RAG chain
  print('chaining...')
  model = "tiiuae/falcon-180B"
  llm = ChatLiteLLM(
      streaming=True,
      verbose=True,
      callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
      model_name=model,
      temperature=0
  )

  rag_chain = (
      {"context": retriever,  "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

  query = "How to test for Mpox?"
  rag_chain.invoke(query)

async def insert(docs, embeddings):
  print('inserting...')
  db = await FAISS.afrom_documents(docs, embeddings)
  return db

async def search(db, query):
  print('searching...')
  docs = await db.asimilarity_search(query)
  return docs