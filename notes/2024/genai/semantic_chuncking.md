## [Semantic Chunking for RAG](https://youtu.be/dt1Iobn_Hw0)
Release date : 08/03/24
### Idea
- Chunking is more like an art
- Sematic chunking is about asking the meaning of each chunk
- Aim
    - process
    - contexualize importance
    - best practice
    - assess the output using RAG assessment
- Index
    - intro
    - recursive
    - semantic
    - assess

### Details
#### Intro
- break text into smaller pieces to fit in context window
- Finding the right length of chunk depends on the context 
- This leads to better retrival in RAG
    - Retrival is fetching dense vectors from DB
    - A query is chunked, embedded and similar embeddings are found in VBD to act as output
    - Chunking can be before embedding or aftr, affects the output
- Chunking methods
    - Fixed size
    - recursive : ???
    - doc specific
    - scemantic : ???
    - agentic : start with normal chunking, then use LLM to reason if the chunking is good enough
- FIxed size:
    - number of chars
- overlap : 
    - by sentence
    - by para
- Recurssive : fixed size plus natual language flow
    - De-facto standard Size + Overlap still used
    - LangChain separators
    - .\n\n - double new lines  
    - " "
- Semantic : use embeddings of individual sentences to make more meaningful chunks.
    - accurately maintain info in embedding (small doc)
    - retain context in each chunk (big doc)
    - Steps:
        1. Split doc into sentences
        2. Compare 1,2,3 --> 4,5,6 and so on
        3. How similar (in embedding space)?
        4. If similar, keep together
        5. If too different, split apart
    - Steps:
        - Split sentences
        - Index sentences on position
        - Group: Choose how many sentences on either side Calculate: distances between groups of sentences
        - Merge: Groups based on similarity above thresholds

- Code explanation skipped
- Rag Eval
    - context precision
    - context recall
    - answer relevance
    - faithfulness

### Resource
- [5 levels](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [LG](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker)
- [LI](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/)
- [youtube](https://www.youtube.com/watch?v=8OJC21T2SL4)

### misc
 
---
