## [Stanford CS25: V3 I Retrieval Augmented Language Models](https://youtu.be/mE7IDf2SmJg)
Release date : 25/01/24
### Idea
- 

### Details
- limitations
    - Hallucination : make up with high confidence
    - Attribution : why they say what they say
    - Staleness : cutoff date for training
    - Revisions : bespokeness is missing
    - Customization : working on ones own data
- Solution : to give it external memory
- Contextualization will have retriver and adiional context to the generator

- Pradigms
    - closed vs open book 
    - parametric vs semi parametric

- revising index by swapping it out will prevent staleness
- grounding by pointing it back to source can reduce hallucination
- Even though the strucure of RAG is simple, there are mnany questions like
    - scaling
    - chunking
    - encoding
    - retiriving
#### Architectures
- Frozen RAG: training happens ones and while using only testing happens
    - In context learning
- to make this better one must start from retriver
    - Sparse : TFIDF and BM25
        - used in DrQA 2017
    - Dense : OrQA
        - Dense Passage Retriever
        - ORCA
- VBD in GPU
    - FAISS
    - based on cosine score between embedding
- ColBert
    - Siamese model

#### State of the art
- SPLADE : sparse meets dense with query expansion
- DRAGON : Generalized dense retriever via progressive data augmentation (Lin et al 2023).
- hybdrid search and algorithms better than direct dot product

#### Contextualizing the Retriever for the Generator
- RePlug (Shi et al 2023)
- In-Context RALM
    - Use "Frozen RAG" with BM25 and then "specialize" only the retrieval part via reranking:
        - 0 shot with an LM
        - Trained reranker
#### Contextualization of both retiriver and generator
- RAG (Lewis et al 2020)
    - RAG seq model
    - RAG token model
- Whole point of RAG: Freezing Suboptimal!
- Limitation of RAG is small k (not for large docs). Can we do the fusion in the decoder directly?
    - FiD (Izacard & Grave 2020)
- Pure decoder model
    - KNN-LM (Khandelwal et al 2019)
- Retro (Borgeaud et al 2022) - pretrain from scratch?
- Retro++ (Wang, Ping, Xu et al 2023)
    - might not work
#### Contextualization of notht with doc encoder
- REALM (Guu et al 2020)
    - Downside: not really generative, BERT all the way.
    - Every weight update the encoding happens so expensive
- Atlas: Deep Dive - How to train the retriever? (1)
    - Retriever-updates:
        - FiD-style "attention distillation"
        - EMDR? (RAG-style loss but stopgradient on the LLM)
        - Likelihood distillation (KL div over the posterior log likelihood for adding a doc)
        - Leave-one-out (KL div over the posterior log likelihood if we had removed a doc)

- SILO (Min, Gururangan et al 2023) : safe data source
- Lost in the Middle (Liu et al 2023)
- WebGPT (Nakano et al 2021)
- Toolformer (Shick et al 2021)
- Self-RAG (Asai et al 2023)
- InstructRetro (Wang et al 2023) / RA-DIT (Lin, Chen, Chen et al 2023)
- Advanced Frozen RAG
    - Frameworks: Llamalndex, LangChain
    - Vector databases (eg Chroma, Weaviate, etc) are all making FRAG easy
    - Child-Parent RecursiveRetriever: go from small child chunks to bigger parent chunks
    - Hybrid search: combine sparse and dense results using reciprocal rank fusion
    - Using zero-shot LLM rerankers on dense retrieval results
    - HyDE: Hypothetical Document Embeddings (Gao, Ma et al 2022) = Given query, write hypothetical answers, embed those and find relevant docs

#### Unanswered questions
- Joint from-scratch pretraining is still underexplored
- What do scaling laws look like?
    - Scale the LM? In terms of params or tokens? o Scale the retriever? In terms of params or chunks? o What if we scale the index - size during inference?
- Can we fully decouple memorization from generalization? Decouple knowledge from generation?
- Can we move beyond bi-encoder vector databases, or does it all just become rerankers?
- Are there smart ways to create (synthetic) data for RAG?
- How do we properly evaluate RAG systems?


#### Other works
- Multimodal RAG (Gur et al 2021; Yasunaga et al 2023)
    - Cross-Modal Retrieval Augmentation for Multi-Modal Classification
    - Retrieval-Augmented Multimodal Language Modeling

#### RAG 2.0
- Systems over models!!
- Optimize it all - why not backprop into the chunker?
- Trading off cost and quality
- Zero-shot domain generalization

### Resource
- 

### misc
 
---
