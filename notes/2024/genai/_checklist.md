# RAG
- Which model
    - architecture
    - mixture
    - Context length
    - Tranined on which data
        - Is it close to internal data
        - Is there a conflict in pre-trained vs new data
    - Size
    - Cost
    - Quantization
        - gptq, gguf, awq, exl2, hqq
    - alignment
        - dpo, ppo, orpo, kto, ipo, nac
        - dno
- Which framework
    - llamaindex
    - langchain
    - dspy
    - semantic kernel
    - autogen
    - taskWeive
- Embedding using which model
- Chunking style
    - raptor
    - semantic
    - doc summary
    - colbertv2
    - agentic
    - length
    - stop words
- How to choose vector store
    - familiarity
    - felxible
    - ease of implementation
        - abstraction
        - Integration
    - performance
        - Queries per second (QPS)
        - Recall rate
    - scalability
        - Vector dimensions supported
        - Number of embeddings
    - Consider trade-offs between cost, recall, throughput and latency
- Which vector store : 
    - pinecone (cloud)
    - chroma
    - Faiss
    - Vespa
    - Milvus
    - weaviate (cloud)
    - redis
    - AWS aurora with pgvector postgres
    - AWS Open search
- Which fine tuning
    - peft
    - lora
    - qlora
    - 1 bit
    - reft
    - LISA
    - LONGLORA, LOFTQ, RSLORA, QLORA, LORA+, GALORE, DORA, NEFT, unsloth, PISSA
- Caching required?
- Feedback mechanism
    - store for that
    - logs
- Data
    - is conversion of pre processing it into a specific format required?
- Should we look into RAFT?
- Evaluation

# Agent





