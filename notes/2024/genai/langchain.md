# [Complete Langchain GEN AI Crash Course With 6 End To End LLM Projects With OPENAI,LLAMA2,Gemini Pro](https://youtu.be/aWKrL4z5H6w)

## Installation
- Need langchain pip
- huggingface_hug pip
## details
- temprature is a paramterer to control creativity and diversity with 1 being wild imagination


## 28:30 Prompt template 
- to set what input and output should be expected, som sort of standardization
```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt_template=PromptTemplate(input_variables=[ 'country' ],
template="Tell me the capital of this {country}" )

prompt_template.format(country="Japan")

chain = LLMChain(llm=llm, prompt = prompt_template)
chain.run("Italy")
```
## 35:35 Sequential Chain
- one i/p for multiple chains
- 
SKIPPING
## 2:18:00 End to end example 