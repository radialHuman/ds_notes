## [The 5 Levels Of Text Splitting For Retrieval](https://youtu.be/8OJC21T2SL4)
Release date : 08/01/24
### Idea
- LLM models perform better if trained on custom data for custom tasks (increase signla to noise ratio)
- LLm apps have context length, which limits the amount of info that can be in i/p
- text splitting or chucking is to split i/p to make it optimal for llm
- Goal : to find optimal way to pass data to model for the task
- For RAG applications, chuncking method is important
- Not all emthods work for all tasks, needs evaluation

### Details
#### 1. Character splitting
- Character splitting is the most basic form of splitting up your text. It is the process of simply dividing your text into N-character sized chunks regardless of their content or form
- Pros: Easy & Simple
- Cons: Very rigid and doesn't take into account the structure of your text

#### 2. Resursive splitting
- The problem with Level #1 is that we don't take into account the structure of our document at all. We simply split by a fix number of characters.
- The Recursive Character Text Splitter helps with this. With it, we'll specify a series of separatators which will be used to split our docs.
- from langchain.text_splitter import RecursiveCharacterTextSplitter
- 
#### 3. Doc specific splitting
- The Markdown, Python, and JS splitters will basically be similar to Recursive Character, but with different separators.
- from langchain.text_splitter import MarkdownTextSplitter
- from langchain.text_splitter import PythonCodeTextSplitter
- from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

#### 4. Scemantic splitting
- Its at embedding level
- Takes meniang of the sentence ot make chuncks
- Embeddings represent the semantic meaning of a string. They don't do much on their own, but when compared to embeddings of other texts you can start to infer the relationship between chunks. 
- lean into this property and explore using embeddings to find clusters of semantically similar texts.
- SKIPPING code explantion

#### 5. Agentic splitting
1. | would get myself a scratch piece of paper or notepad
2. I'd start at the top of the essay and assume the first part will be a chunk (since we don't have any yet)
3. Then | would keep going down the essay and evaluate if a new sentence or piece of the essay should be a part of the first chunk,
if not, then create a new one
4. Then keep doing that all the way down the essay until we got to the end. 
- This can be done using an "agent"
- https://python.langchain.com/docs/templates/propositional-retrieval
- First the chunkcs are run through to get propotitions then relevant chunks are grouped


#### Alternative representation
- Apart from chunkcing
- Shoudl the mebdding in the next step be of raw text or alternative rep of the text?
- Multi-Vector Indexing - This is when you do semantic search for a vector that is derived from something other than your raw text
    - like summary of chuck
- Hypothetical questions
    - trainign on made up qna rather than just text
- Parent Document Retriever (PDR)
    - Much like the previous two, Parent Document Retriever builds on the concept of doing semantic search on a varied representation of your data.
    - 


### Resource
- 

### misc
 
---
