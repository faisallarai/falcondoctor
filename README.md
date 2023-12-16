
##

```sh
docker build -t faisallarai/falcondoctor:v1.0.0 .
docker tag faisallarai/falcondoctor:v1.0.0 faisallarai/falcondoctor:v1.0.0-release
docker push faisallarai/falcondoctor:v1.0.0-release
```

### Load

- `python main.py -d load <path>`
- `python main.py --doctor load <path>`

### Start

- `python main.py -d start <path>`
- `python main.py --doctor start <path>`

## Framework
```sh
pip install langchain
pip install "langserve[all]"
pip install langchain-cli
pip install langsmith
```

## Vector Store

```sh
pip install faiss-cpu
```

# Embeddings

```sh
pip install sentence_transformers
```

# Providers

```sh
pip install openai
pip install huggingface_hub
```




