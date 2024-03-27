# [LangChain How to and guides](https://www.youtube.com/playlist?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)
- https://github.com/samwit/langchain-tutorials
- https://github.com/samwit/llm-tutorials
## [LangChain Basics Tutorial #1 - LLMs & PromptTemplates with Colab](https://youtu.be/J_0qvRt4LNk?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)
Release date : 02/03/23
### Idea
- LangChain is built around Prompts, to interact with the s/w stack
- to connect llms with other tools, apis and dbs
- to use prompt template 

### Details
#### Prompt template
```python
from langchain import PromptTemplate

restaurant_template = """I want you to act as a naming consultant for new restaurants.
Return a list a restaurant names. Each name should be short, catchy and easy to remember.
What are some good names for a restaurant that is {restaurant_desription}"""

prompt = PromptTemplate( input_variables=["restaurant_desription"],
template=template,)

description = "a Greek place that serves fresh lamb souvlakis and other Greek " 
description_02 = "a burger place that is themed with baseball memorabilia" 
description_03 = "a cafe that has live hard rock music and memorabilia"
## to see what the prompt will be like 
prompt_template.format (restaurant_desription=description)

```
#### To query LLM
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)
# Run the chain only specifying the input variable.
print(chain.run(description_02))
```

#### To do few shot learning
```python
from langchain import PromptTemplate, FewShotPromptTemplate

# First, create the list of few shot examples.
examples = [ {"word": "happy", ""antonym": "sad"}, {"word": "tall", ""antonym": "short"},
# Next, we specify the template to format the examples we have provided
# We use the ~PromptTemplate™ class for this.
example formatter_template = """
Word: {word}
Antonym: {antonym}\n
"""
example prompt = PromptTemplate(
input_variables=["word", "antonym"],
template=example formatter_template,)

# Finally, we create the ~FewShotPromptTemplate object.
few_shot_prompt = FewShotPromptTemplate( 
    # These are the examples we want to insert into the prompt.
    examples=examples,
    # This is how we want to format the examples when we insert them into the prompt.
    example_prompt=example_prompt,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="Give the antonym of every input",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Word: {input}\nAntonym:",
    # The input variables are the variables that the overall prompt expects.
    input_variables=["input"],
    # The example separator is the string we will use to join the prefix,
    example separator="\n\n",
)

print(few_shot_prompt.format(input="big"))  

```
### Resource
- 

### misc
 
---
## [LangChain Basics Tutorial #2 Tools and Chains](https://youtu.be/hI2BY7yl_Ac?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)
Release date : 02/03/23
### Idea
- Tools and chains 
- Tools are submodules to link to LLM
    - python REPL
    - serpAPI
    - Wolfram alpha
    - google search
- Chain are links of tools
    - output of one tool can be connected to another
    - input => llm -> google search -> llm -> final o/p

### Details
#### Types of chain
- generic : llm chain, transformation chain (like a function on an o/p), sequential chain to make tools connect and execute 1/1
- utility : LLMMath, PAL, SQLDB, API, LLMBash, LLMChecker, Request
- async :

#### PAL chain
- Converts text into resoning using python and run it

#### SQLDB chain
- convert nlp into SQL 

#### Bash chain
- run shell commands

#### Request chain
- maniputlate html page of a site

### Code
- llm -> prompt -> template -> LLM
```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model_name='text-davinci-003',
temperature=0,
max_tokens = 256)

article = """..."""

fact_extraction_prompt = PromptTemplate(
input_variables=["text_input"],
template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep {}")

fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
facts = fact_extraction_chain.run(article)
print(facts)


# change the system role
investor_update_prompt = PromptTemplate(
input_variables=["facts"],
template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short {}:")

investor_update_chain = LLMChain(llm=llm, prompt=investor_update_prompt)
investor_update = investor_update_chain.run(facts)
print(investor_update)
len(investor_update)

# another one
triples_prompt = PromptTemplate( I
input_variables=["facts"], template="Take the following list of facts and turn them into triples for a knowledge graph:\n\n {facts}")

triples_chain = LLMChain(llm=llm, prompt=triples_prompt)
triples = triples_chain.run(facts)
print(triples)
len(triples)


# combingn them altogether
from langchain.chains import SimpleSequentialChain, SequentialChain
full_chain = SimpleSequentialChain(chains=[fact_extraction_chain, investor_update_chain], verbose=True)
response = full_chain.run(article)


```
- Chains can be made out of tools or smaller chains

```python
# PAL CHAIN
from langchain.chains import PALChain
llm = OpenAI(model_name='code-davinci-002',
temperature=0, max_tokpns512 )

pal_chain = PALChain.from_math_prompt(llm, verbose=True)

questionon = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four ¢
question_02= "The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they a eS
pal_chain.run(question_02)

```

```python
# API CHAIN
from langchain import OpenAI
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
llm = OpenAI(temperature=0, max_tokens=100)
from langchain.chains.api import open_meteo_docs
chain_new = APIChain.from_llm_and_api _docs(llm, open_meteo docs.OPEN METEO DOCS, verbose=True)
chain_new.run('What is the temperature like right now in Bedok, Singapore in degrees Celcius?')

```

### Resource
- [doc](https://python.langchain.com/docs)

### misc
 
---
## [ChatGPT API Announcement & Code Walkthrough with LangChain](https://youtu.be/phHqvLHCwH4?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)
Release date : 02/03/23
### Idea
- LangChain with OpenAI 3.5 turbo

### Details
```python
from langchain import PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, OpenAIChat

prefix messages = [{"role": "system", "content": "You are a helpful history professor named Kate."}]

llm = OpenAIChat(model_name='gpt-3.5-turbo',
temperature=0,
prefix_messages=prefix_messages,
max_tokens = 256)


template = """Take the following question: {user_input}
Answer it in an informative and intersting but conscise way for someone who is new to this topic"""
prompt = PromptTemplate(template=template, input_variables=["user_input"])

llm_chain = LLMChain(prompt=prompt, llm=l1m) 
user_input = "When was Marcus Aurelius the emperor of Rome?"
llm_chain.run(user_input)

llm_chain = LLMChain(prompt=prompt, llm=l1m)
user_input = "Who was Marcus Aurelius married to?"
llm_chain.run(user_input)
```

### Resource
- 

### misc
 
---

## [LangChain - Conversations with Memory (explanation & code walkthrough)](https://youtu.be/X550Zbz_ROE?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)
Release date : 06/03/23
### Idea
- 

### Details
- 

### Resource
- 

### misc
 
---
