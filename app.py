import os, asyncio, redis, pinecone, litellm, time

from helpers import insert, search, fetch, chunk, embed

from pprint import pprint
from decouple import config
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, RedisCache

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain.chat_models import ChatLiteLLM, ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.vectorstores import FAISS, Pinecone

os.environ["HUGGINGFACE_API_KEY"] = config("HUGGINGFACE_API_KEY")

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# set_llm_cache(InMemoryCache())
litellm.set_verbose=True
redis_url = "redis://localhost:6379/0"
redis_client = redis.Redis.from_url(redis_url)
set_llm_cache(RedisCache(redis_client))

# Collect and Load data
documents = fetch()

# Chunk data
docs = chunk(documents)

## Embed the chunks
embeddings = embed()

text = "How to test for Mpox?"
query_result = embeddings.embed_query(text)
print(query_result[:3])

## Store the embeddings
# db = asyncio.run(insert(docs, embeddings))
# db = FAISS.from_documents(docs, embeddings)

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY") or '51b22bf7-8308-4dd5-964d-f20fc0250766',  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV") or 'gcp-starter',  # next to api key in console
)

index_name = "falcondoctor"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(name=index_name, metric="cosine", dimension=768)

    while not pinecone.describe_index(index_name).status['ready']:
    	time.sleep(1)

index = pinecone.Index(index_name)
index.describe_index_stats()

print(index.describe_index_stats())

db = Pinecone.from_documents(docs, embeddings, index_name=index_name)

index.describe_index_stats()
print(index.describe_index_stats())

# if you already have an index, you can load it like this
# db = Pinecone.from_existing_index(index_name, embeddings)

# Search
query = "How to test for Mpox?"
# docs = asyncio.run(search(db, query))
docs = db.similarity_search(query)
print(docs)

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
print('creating chat model...')
model = "tiiuae/falcon-180B"

llm = ChatLiteLLM(
   streaming=True,
   verbose=True,
   callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
   model=model,
   model_kwargs={'options': {'wait_for_model': True}},
   custom_llm_provider="huggingface",
   temperature=0
)

print('sleeping...')
# time.sleep(600)

print('rag chaining...')
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print('asking question...')
query = "How to test for Mpox?"
rag_chain.invoke(query)