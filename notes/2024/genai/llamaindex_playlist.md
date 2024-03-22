# [Llamaindex](https://www.youtube.com/playlist?list=PLZoTAELRMXVNOWh1SDXt5NFujQMOt-CWy)

## [Announcement](https://youtu.be/1eym7BTnuNg)
### langchain vs llamaindex
#### llamaindex
- connect custom data sources to llms
- indexing is done on the data to get metadata to query from
- quick response based on indexing
- doc Q&A, chatbot based on data, knwoledge agent, structured analysis
- strength :
    - data ingestion via apis, many data format support, 
    - indexing data for vector db and database providers
    - query interface 
    - supports unstructred, semi and structred data

#### langchain
- 


#### difference: llamaindex vs langchain
feature | llamaindex | langchain |
|-|-|-|
Primary focus | Intelligent search and data indexing and retrieval  |  Building a wide range of Gen Al applications
Data handling | ingesting, structuring, and accessing private or domain-specific data  | Loading, processing, and indexing data for various uses
Customization | Offers tools for integrating private data into LLMs  | Highly customizable, it allows users to chain multiple tools and components
Flexibility | Specialized for efficient and fast search  | General-purpose framework with more flexibility in application behavior
Supported LLMs | Connects to any LLM provider like OpenAl, Antropic, HuggingFace, and Al21  | Support for over 60 LLMs, including popular farmewois ke Opera, hugging face, a21
use case | Best for applications that require quick data lookup and retrieval  | Suitable for applications that require complex interactions like chatbots, GQA, summarization
integration | Functions as a smart storage mechanism  |  multiple tools linking
main use | for search | wide range general purpose

#### use together llamaindex + langchain
- First index the data using llamaindex
- then user queries on the index using llamaindex
- the output of index, combined with prompt and data is sent to llm which send back the o/p to user using langchain

---

## [End to end RAG LLM App Using Llamaindex and OpenAI- Indexing and Querying Multiple pdf's](https://youtu.be/hH4WkgILUD4)
### DEMO notebook

## [Step-by-Step Guide to Building a RAG LLM App with LLamA2 and LLaMAindex](https://youtu.be/f-AXdiCyiT8?)
### 
