- Before starting an LLM project, conduct impact & risk assessment. After POC, conduct maturity assessment.
    - identify the problem if llm is req
    - identify profit , not tech for the sake of tech
    - cost
    - define time line
    - define impact
- Risks
    - Customer facing 
    - Preventing facing hallucinations 
    - privacy concern
    - ai bias
    - data sec breach
    - downtime
    - human oversight and feeedback
    - misuse
- Questions to ask
    - which
        - llm
        - verison
        - token : Byte pair encoding, Wordpience, sentence piece, unigram, Byte Latent
        - useage cost
        - latency, 
    - embeddings : which model, how parsing occurs, how chunking occurs, computational expense
    - store embedding : how to udpate, how to store chuncks, metadata storage, chunking strategy
    - retrieving : How many chunks are retrieved, combining chunks, relevancy ranking, Metadata filtering, Which metadata was retrieved with the embeddings, Which similarity algorithm is used
    - prompt strategy : ICL, query enrichment before sending, meta prompting, prompt expansion
- Fine tuning
    - Corresponding code commit. Infrastructure used for fine-tuning & serving.
    - What model artifact is produced.
    - What training data is used. retraining strategy & frequency. Methodology (supervised, self-supervised, RLHF).
- Tools : 
    - version control : github, hitlab
    - cicd : jenkins, gitlab cicd
    - orchestration :L airflow, databricks, aws
    - mdoel regustry and tracking :  mlflow, sagemaker, vertex
    - container registry :  ecr, docekr hub
    - serving : k8, databricks, azure, sagemaker
    - evaluation :  grafana, elastic, 
    - vectore db : https://superlinked.com/vector-db-comparison
# RAG
- Which model
    - architecture : rag, knowleegde grahp, crag, self rag, reranking
    - CAG will work instead of RAG? or hybrid?
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
        - dpo, ppo, orpo, kto, ipo, nac, apo
        - dno
    - Guardrails
        - nemo
        - guardralis ai
        - aws guardrails
        - llamaguard
        - prompt injection
        - anti jailbreak
    - how to test guardrails
        - [spikee](https://spikee.ai/)
        - [pyrit](https://github.com/Azure/PyRIT)
    - Security:
        - vul scan [garak](https://github.com/NVIDIA/garak)
        - Plexiglass
        - PurpleLlama
        - Rebuff
        - Garak
        - LLMFuzzer
        - LLM Guard
        - Vigil
        - jailbreak-evaluation
        - Prompt Fuzzer
        - WhistleBlower
        - Open-Prompt-Injection
        - Agentic Radar
        - https://github.com/msoedov/agentic_security
    - Techniques to improve
        - multi query retrival
        - having context extraction using one llm and another to get structured query 
            - to act as a filter in case of conditions in user query
        - BM25 : for retrival as well as reranking
        - BM41
        - Dense passage retriver
        - Haystack retriver
        - CRAG : judge t5
        - self-RAG
        - Self-CRAG
        - Autorag
        - hippo rag
        - Reranking : cross encoder, api based like cohere, 
        - flash rank : in different sizes 
        - Main-RAG
        - Hyde : hypothetical answer using 2nd llm along with actual query
        - Pre retreval
            - query routing
            - query rewriting
            - query expansion
        - Post retrival 
            - rerank
            - summary
            - fusion
        - hybrid rag : graph + vector rag
        - knowledge graph rag : networkx, neo4j, grafitti, zap, rdflib
        - Light rag
        - LongRAG
        - OP-RAG by NVIDIA for long context
        - Temprature
            - top p and top k
        - model merging mergekit, LM-Cocktail
        - DFT positional encofing for faithfulness
        - CoPe
        - RoPe (RoFormer)
        - Rags to riches
        - Hallucination Detection Model in RAG Pipeline - Lynx 8B
        - contextual retrival by anthropic
        - PyMuPDF4LLM for RAG
        - Rule RAG #TODO
        - ATRag #TODO
        - Report generator - llama index's agent
        - AstuteRag : to prevent contradicatory knowledge
    - Citation required? 
        - llama index citation query engine
        - from langchain.chains import create_citation_fuzzy_match_chain
    - Metadata details : hierarchical details
- vision rag
    - colpali
    - clip :  two encoders are trained, openai
    - Align : same but from google
    - siglip : adds sigmoid so that two batches are not processed
    - EVA
    - MetaClip
    - CLIPA
    - DFN
    - lit : pre-trained image encoder is froozen and only text is trained
    - Siglit
    - imagebind :  multimodal
    - FuDD
    - 
- Which framework
    - llamaindex
    - langchain
    - dspy (ml for better prompt)
    - Verba
    - semantic kernel
    - autogen
    - taskWeive
    - Haystack
    - embedAnything
- Embedding using which model
    - OpenAl
    - bge-large
    - lim-embedder
    - Cohere-v2
    - Cohere-v3
    - Voyage
    - JinaAl
    - Fine tuning : linear adapter, 
    - Qwen embeeding
    - Gemini embedding
    - Reranker 
    - matryoshka embedding support
- Chunking style
    - raptor
    - semantic
        - cluster based
    - doc summary
    - colbertv2
    - PLAID
    - agentic
    - length
        - character
        - token
    - stop words
    - Recursive
        - character
        - token
    - topical
    - late chunking
    - llm as chunker
    - chonkie
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
    - Indexing
    - Search algorithm : hswn, diskann, scann, iVF, PQ
    - Similarity alogorithm
    - Distributed architecture
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
    - qdrant
    - lancedb (on the fly)
- Caching required?
    - prompt chaching
- Feedback mechanism
    - store for that
    - logs
- Data
    - is conversion of pre processing it into a specific format required?
- Should we look into RAFT? : trainign llm on classifying relevant document
- Evaluation
    - LangSmith
    - RAGAS
    - Prometheus 2 : judge jury
    - AWS clarify
    - Giskard
    - GPT Eval
    - GPT Score
    - WEAT
    - BertScore
    - Vectara
    - replacing judge with juries
    - Panel of LLMs
    - weights and bias
    - Phoenix eval
    - MMLU
    - big bench
    - MATH
    - human eval
    - Vicuna bench
    - MT bench
    - Alpacafarm
    - Truelens
    - multiple logically related questions maxsat solver
    - self-refinement
    - DeepEval
    - AutoRAGAS (costly)
    - CICD CircleCI
    - ChainPoll
    - RealHall Closed
    - OpenEval
    - Anthropic's Circuit tracer : how it said what it said, OS [link](https://github.com/safety-research/circuit-tracer)
    - lilypad : to trace llms resposnes [link](https://github.com/Mirascope/lilypad)
- metrics
    - f1-over-words
    - Precision@k
    - NDCG
    - Hit rate
    - MRR : Mean Reciprocal Rank (MRR)the quality of information retrieval systems
    - Correctness, haluucaination, toxicity
    - faithfullness, 
    - adherence to guidlines
- Optimization strategy
    - Better Parsers
    - Chunk Sizes
    - Prompt Engineering
        - dspy
        - textgrad
        - automatic PR
        - conversational PE
    - Grpah Tracing in ICL 
    - Customizing Models
    - Metadata Filtering
    - Recursive Retrieval
    - Embedded Tables
    - Small-to-big Retrieval (parent document, small chunks but with links to parent for whole)
    - Automerge chunking (hierachical searching and expansion)
    - contextual compression (generation side pruning, using llm or embedding model)
    - Knowledge base
    - React
    - Chain of thougth (CoT)
        - Reward modeling
        - Self verification
        - Search methods
        - Best-of-N sampling
        - STaR algorithm
        - Verifier
        - Monte carlo Tree search (MCTS)
    - Tree of thought (beam search)
    - Language Agent Tree search (LATS)
    - Buffer of thought
# Local run
- lm studio
- llama cpp
- ollama
- anything llm

# Parsers
- langchain aprser 
- PyMuPDF4LLM (not that good with tables)
- Unstructured (uses most of these and models from HF)
- llama parse (paid api)
- docling (uses HF models, good)
- LLMSherpa
- PageMage
- pypdf2
- pdfminer
- pdfplumber (not good with tables)
- pdfquery
- camleot
- tabula-py (needs java)
- pdfreader (gets images out, but depedns on the pdf)
- pymupdf/fitz
- pypdfium2
- marker (not for commercial use, uses HF models)
- extarctable
- spire.pdf


# Fine tune
- Which fine tuning
    - peft
    - lora
    - MORA
    - qlora
    - 1 bit
    - reft
    - LISA
    - LONGLORA, LOFTQ, RSLORA, QLORA, LORA+, GALORE, DORA, NEFT, unsloth, PISSA, Relora, ReMora, QALora, 
    - LORA-XS
    - VERA
- Which toolNN
    - unsloth
    - onnx
    - llama factory
    - swift
    - DeepSpeed
- which library
    - auto GPTQ
    - auto round
    - bitsandbytes
    - auto awq
# Back up llm

# Inferencing
- Test time
- sleep time 

# Multimodal LLM
- QwenVL
- Pixtral
- OpenAI o1
- Gemini 1.5
- MedGemma 3 for medical
- Bagel MoT

# resoning API
- Forge

# Agent : authentication, extensive unit test, reAct, loop, observe, action
- Crewai
- pydantic ai and grpah
- MS Autogen
- langflow
- MS semantic kernel
- OpenAI Swarm
- Langraph
    - LangGraph Multi-Agent Supervisor
- Llamaindex workflow
- ChatDev
- Stack
- PhiData (OS)
- Camel AI
- smolagent

# Agent protocol
- only llm to agent
    - MCP
    - agents.json
- agent to agent : find, communicate, neogtiate, authenticate, debate
    - A2A
    - A connect P : by cisco, also like a2a, how to host and launch agent, 
    - A communication P : by IBM, BeeAI, fork of MCP, more lie a2a
    - A Network P : a2a but on blockchain
    - A Interaction Trwansaction P : also on blackchain, focuses on how much will an agent cost, uses bidding
    - Agora : new take, start with NL then upgrade the protocol for sharing, there by saving tokens
- domain specific protocols
    - human interaction protocol
        - PXP 
        - LOKA
    - Robot agent interaction protocol
        - CrowdES
        - SPP : spatial populaiton protocol
    - LMOS : Large model operating system, from IBM again
    - AP : same as agents.json

# Deep research
- deerflow : https://github.com/bytedance/deer-flow

## Agent framework properties 04/25
- state management : mutable (PyDantic G), immutable (LanGraph)
- node calling (pydantic g) or edge flow defining (langraph, node agnostic, but command)
- data sharing : nodes dont know other nodes data (pydantic g), langgraph shared state
- Dependency injection : pasing non data objects to nodes (connector)
- state persistance dumping : states can be stored as checkpoints
- node dependencies : node has state and dependncy passed separately
- strucutred output : for deterministic flow
- async-ness : by default
- reuseable nodes : if they return the next node, it cant be resused
- non-leaky abstraction : prompt conflict
- oops heavy : pydantic
- can a node call two nodes simultaneously?

# Memory
- memgpt
- LangMem

# SQL
- Lida
- vanna
- pandasai
- 


# Speed it up
When a Retrieval-Augmented Generation (RAG) system experiences latency issues, several strategies can be employed to improve its speed. Here's an exhaustive list of potential optimizations:

**Optimizing the Retrieval Stage:**

* **Efficient Indexing and Search in the Vector Database:**
    * **Choose an appropriate vector database:** Evaluate different vector databases (like Pinecone, Faiss, Milvus, etc.) based on their performance characteristics and suitability for the specific use case. The sources use Pinecone in their example.
    * **Optimize indexing parameters:** Fine-tune indexing parameters, such as the number of clusters or the distance metric used, to improve search speed. 
    * **Use approximate nearest neighbor search:** Consider techniques like Locality Sensitive Hashing (LSH) for faster retrieval at a slight cost to accuracy.

* **Reduce the Number of Documents Retrieved:** 
    * **Fine-tune the similarity threshold:** Adjust the threshold for document similarity to retrieve a smaller, more relevant set of documents. 
    * **Employ smarter pre-filtering:** Use heuristics or rule-based filters to narrow down the search space before querying the vector database.

* **Optimize Embedding Computation:** 
    * **Use smaller embedding models:** Consider using smaller or more efficient embedding models if the accuracy trade-off is acceptable.
    * **Batch embedding computations:**  Process multiple documents or queries simultaneously to reduce overhead.
    * **Cache frequently used embeddings:** Store embeddings for frequently accessed documents in a cache to avoid recomputation. 

**Optimizing the Generation Stage:**

* **Use a Smaller or More Efficient LLM:**
    * **Quantize the LLM:** Employ quantization techniques to reduce the precision of model weights and activations, leading to faster inference and smaller model sizes.
    * **Distill knowledge into a smaller model:** Train a smaller "student" model to mimic the behavior of the larger LLM using knowledge distillation techniques. The sources mention that a smaller model trained this way is guaranteed to both be smaller and improve latency.
* **Reduce the Context Length:** 
    * **Limit the number of retrieved documents passed to the LLM:**  Experiment with passing only the most relevant documents to the LLM. 
    * **Use summarization techniques:** Summarize retrieved documents before feeding them to the LLM to reduce the input sequence length. The sources explain that this can be done using an external LLM.
* **Optimize LLM Inference:**
    * **Compile the LLM:**  Compile the LLM to optimize its execution on the target hardware. This leads to improved efficiency, resource utilization, and cost.
    * **Use a more efficient inference engine:** Explore optimized inference engines or libraries specifically designed for fast LLM inference. For example, vLLM, Hugging Face's TGI, or OpenLLM are libraries that can be used for easy deployment of LLM inference services.
    * **Employ caching:** Cache LLM outputs for frequently used prompts or inputs. 
* **Stream Responses:** Stream LLM-generated text to the user as it's being generated, improving perceived latency and responsiveness. Streaming allows users to see the text as it's being produced, even if the entire generation process takes some time.

**Optimizing the Overall System:**

* **Distributed Computing:**
    * **Parallelize retrieval and generation:** Distribute the workload across multiple GPUs or computing nodes. This can be done using libraries like Ray, which enables scaling to multiple GPUs by running each pipeline on a separate node. Accelerate is another library that can be used for parallelizing and scaling to multiple GPUs.
    * **Optimize data transfer:** Minimize data transfer between nodes by using efficient communication protocols and data serialization techniques. For example, using gRPC can improve network I/O performance, especially for large payloads often seen in LLM applications.
* **Caching:** 
    * **Cache retrieval results:** Store frequently used retrieval results in a cache to avoid repeated searches. 
    * **Cache LLM generations:** Cache LLM outputs for commonly used prompts or document combinations. 
* **Hardware Optimization:** 
    * **Use faster storage:** Employ high-speed SSDs or NVMe drives to reduce storage access latency. This is particularly relevant for storing and retrieving embeddings and documents in the vector database. 
    * **Use powerful GPUs:** Utilize high-performance GPUs to accelerate both embedding computations and LLM inference. 
* **Code Optimization:**
    * **Profile the code:** Identify performance bottlenecks in the retrieval and generation processes. 
    * **Optimize data structures and algorithms:** Employ efficient data structures and algorithms to minimize computational overhead. 
* **Load Balancing and Autoscaling:**
    * **Distribute traffic:** Use load balancing to distribute incoming requests across multiple instances of the RAG system. 
    * **Implement autoscaling:** Automatically adjust the number of instances based on the workload to handle varying traffic patterns. This ensures efficient resource utilization and avoids bottlenecks during peak usage.

Addressing latency in a RAG system often involves a combination of these techniques, carefully chosen based on the specific system architecture, requirements, and performance bottlenecks.


# better accuracy

Several techniques can be implemented to enhance the retrieval stage of a Retrieval-Augmented Generation (RAG) system, improving speed and accuracy. Here's an exhaustive list of potential optimizations focusing on the retrieval aspect:

### **Enhancing the Vector Database:**

*   **Selecting the Right Vector Database:**  The choice of vector database significantly impacts search speed and scalability. Evaluate different options like **Pinecone**, **Faiss**, **Weaviate**, **Milvus**, and **Pine cone** based on their performance, indexing methods, and suitability for your data and query patterns. The sources mention **Pinecone** as a practical example for implementing RAG.
*   **Optimizing Indexing Parameters:** Fine-tuning the indexing parameters of your chosen vector database can significantly enhance search speed. Experiment with parameters like:
    *   **Number of clusters:** This influences the granularity of the search space.
    *   **Distance metric:**  Choose the appropriate distance metric (Euclidean, cosine similarity, etc.) based on the nature of your embeddings and the desired similarity measure.

### **Optimizing the Search Strategy:**

*   **Employing Approximate Nearest Neighbor Search:** For faster retrieval, particularly in large-scale databases, consider using approximate nearest neighbor search (ANN) techniques like **Locality Sensitive Hashing (LSH)**. ANN methods sacrifice some accuracy for substantial speed improvements.
*   **Implementing Hybrid Search:** Combine semantic search with keyword search to leverage the strengths of both approaches. Keyword search can be useful for exact phrase matching, while semantic search excels at capturing meaning and context.
*   **Using Query Rewriting:** If your RAG system is a chatbot, use an LLM to rewrite verbose or context-dependent user queries into concise and focused queries optimized for the retrieval step. This can improve the relevance of retrieved documents. 

### **Refining Text Chunking:**

*   **Choosing the Right Chunking Strategy:** The way you divide documents into chunks before embedding them affects the expressiveness of your search index. Consider: 
    *   **Sentence-level chunking:**  Suitable for short documents or when fine-grained retrieval is necessary.
    *   **Paragraph-level chunking:** Effective for longer documents with clear paragraph structures.
    *   **Fixed-length chunking:**  Useful when document structure is inconsistent.
    *   **Overlapping chunks:** Include overlapping segments to preserve context and capture concepts spanning multiple sentences. The sources highlight this as an effective strategy.
*   **Adding Context to Chunks:** Enhance chunk embeddings by incorporating surrounding text:
    *   Include the document title in each chunk to provide document-level context.
    *   Add text snippets from preceding and following sentences to enrich the representation of the current chunk.

### **Optimizing Embedding Computation and Storage:**

*   **Fine-Tuning Embedding Models:** Improve the retrieval relevance by fine-tuning the embedding model on a dataset of relevant query-document pairs. This aligns the model with your specific notion of relevance, as demonstrated in the sources.
*   **Using Smaller or More Efficient Embedding Models:** If accuracy trade-offs are acceptable, consider using smaller embedding models or those specifically optimized for efficiency.
*   **Batching Embedding Computations:**  Process multiple documents or queries simultaneously to reduce overhead. This optimizes the embedding generation process.
*   **Caching Frequently Used Embeddings:** Store embeddings for frequently accessed documents in a cache to avoid recomputation, speeding up the retrieval process.

### **Managing Computational Resources:**

*   **Using Efficient Hardware:** Employing high-speed storage, like SSDs or NVMe drives, for the vector database reduces storage access latency. Using powerful GPUs can accelerate both embedding computations and LLM inference.
*   **Parallelizing Retrieval:** Utilize distributed computing techniques to parallelize the retrieval process across multiple GPUs or computing nodes. Libraries like **Ray** or **Accelerate** can enable this scaling.

By implementing a combination of these techniques, tailored to your specific RAG system and data characteristics, you can significantly improve the speed and effectiveness of the retrieval stage. 


# Guides

https://gradientflow.com/the-new-era-of-efficient-llm-deployment/
https://a16z.com/emerging-architectures-for-llm-applications/
https://www.youtube.com/watch?v=Z3-HddkYgyI

Gateway | External and internal models, quota management, utilization, general RAI checks, etc.
Orchestration | On top of langchain; tool selection (RAG)
Memory & EBR | Context compression; few shots
LLM Model Training / Adaptation / Fine-Tuning | LoRA, RLHF
LLM Model Serving | Multi-GPU; multi-machines
Trust & Responsible Al | Prompt injection / leakage defenses etc.
Dev Tools Prompt, Playground, Evaluation (human & algorithmic evaluation; batch evaluation)


- Understand what users can do, expect the unexpected