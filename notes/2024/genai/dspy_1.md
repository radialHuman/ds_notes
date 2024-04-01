## [DSPy: Transforming Language Model Calls into Smart Pipelines](https://youtu.be/NoaDWKHdkHg)
Release date : 01/04/24
### Idea
- d-s-py
- Information retrival with bert
- llm for search and maintain quality - colBert
- RAG 
- DSpy generalizes llm based task into expressive optimized programming model
- Contextual late interaction over bert

### Details
- Evolution fo RAG
    - till 2023, RAG was all about FT retriver and base llm together (by meta)
    - So that llm with existing knowledge will learn the new data also and become an expert
        - How to sueprvise retiriver for complex tasks?
    - Next came few shot prompting without FT
- Complex retriver
    - where there is dependencies of knowledge and info is scattered all over the doc
    - Multihop retrival
    - Late inetraction
        - how to model retrival?
            - encode doc and query to dense vector 
            - Having one vector contain all the info is too much to ask for
        - Col says 
            - doc should be represented as a matrix of many small vector
            - at the elvel of token
            - if optimized, for infra, then it can be of high quality
            - with less training data
            - 1k queries of COLBERT = 50K SFT LLM (claimed)
    - Porgramming and pipeline
        - Finding a right architecture with layers
        - Focus is not on training or prompting but on brekaing down problems
        - so that complex task can be understood
- Vector encoding for DB
    - Different ways to vector doc and send to VDB needs to be changed
    - Colbert is there in Vespa for searching
    - Not faster but with higher quality with complex tasks domain and less data domain
- Bert vs new model
    - For pipelines for LLMs : llama 2, mistral, gpt
    - For modelling retrival for doc :  bert like models are right fit
    - retirval is a spl case, as it is scale dependednt and no matter the scale the output is expected fast
- Design principle
    - few shot
    - chain of thought
    - these are not sustainable
    - Prompt must go with proper stacking
    - Customized prompting for each model is fragile
        - if the underlying model changes it breaks
        - if the task is too complicated it breaks
        - if the domain is different it breaks
    - Not scalable as a pipeline
    - dspy modules aims at creating structure which remains constant whiel the underlying thimgs change without affecting the quality
- MLOPS challenges
    - stacks, chaining, n/w in llm are devices to program
    - must eb standardize else there are tooo many fuzzy things like json format
    - proper workflow than prompt first route
- Pipeline breaks complex into small easy task so smaller and less expansive models can be used multiple times there by reducing cost
- DSPy
    - 

### Resource
- [github](https://github.com/stanfordnlp/dspy)

### misc
 
---
