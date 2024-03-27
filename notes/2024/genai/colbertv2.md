## [Supercharge Your RAG with Contextualized Late Interactions](https://youtu.be/xTzUn3G9YA0)
Release date : 22/03/24
### Idea
- When a document chunck is big, the whole information needs to be stored in one vector embedding
- This can lead to loss of info

### Details
- col : contextualized later interaction
#### Mechanism
1. both chunks and query needs to be tokenized
2. then each and every toekn gets embedded
3. For each token in the query and chunk embedding, similarity score is calaulated
4. Now each chuck is assigned a score
    - each token contributes to the score than getting compressed in a vector and losing value

#### Usage
- COlbert can be used else RAGatuille which can call COlbert2.0 can be used
- Orca pdf is being used in this case as input
- indexing 
```python
RAG.index(
    collection=List_pdf_documents,
    index_name="orca",
    max_document_length=256, 
    split_documents =True,
) 
```
- Comparing colvert with openai embedding model
- Langchain can be used along with ragatuollie
- If the embedding model has a small size then this can help in making it perform better

### Resource
- [github colbertv2](https://github.com/stanford-futuredata/ColBERT)
- [github Ragtouille](https://github.com/bclavie/RAGatouille?tab=readme-ov-file)

### misc
 
---
