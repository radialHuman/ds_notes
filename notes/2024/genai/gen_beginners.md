## [01 - Introduction to Generative AI and LLMs](https://youtu.be/vf_mZrn8ibc?list=PLmsFUfdnGr3zAgBMu4l1W713a0W__zAMl)
### Concepts
#### Tokenization
- To break down input text into set of characters which are mapped to token indices
- output is text which has more probability of occuring after the last word
- might not be the top most probability, to maintain creativity, randomness is introduced, hence not the same o/p for same i/p
- i/p is knowns as prompt in natual language
- o/p is completion

## [02 - Exploring and comparing different LLM types](https://youtu.be/J1mWzw0P74c?list=PLmsFUfdnGr3zAgBMu4l1W713a0W__zAMl)
- Foundational models 
    - unsupervised or self supervised data
    - large billions parameters
    - base of new models
    - exp : gtp3.5
- LLMS
    - part of foundational model
    - exp : chatGPT
    - open : llama
    - Proprietary : openai
- embeddings : string to number representation
### encoder-decoder
- ...
### decoder only
- ...
### encoder only
- ...

### Fine tuning (easy to diffucult, chaep to expensive, low to high quality)
1. prompt with context
    - one shot
    - few shot
2. RAG
    - vector db
3. fine tuned model
    - customize model with updating wieght and bias
    - high quality labelled data avaialble
4. built from scratch
    - too much work


## [04 - Understanding Prompt Engineering Fundamentals](https://youtu.be/R3sHRPP2G7A?list=PLmsFUfdnGr3zAgBMu4l1W713a0W__zAMl)
- art and repetition
- effective use of the input text 
- depends on the type of application
- to control randomness, hallucination and diversity of its capability
- output bulleted, specifc file format

## [05 - Creating advanced prompts](https://www.youtube.com/watch?v=32GBH6BTWZQ&list=PLmsFUfdnGr3zAgBMu4l1W713a0W__zAMl&index=4&pp=iAQB)
### Techniques
- Few shot prompting, 
    - It's a single prompt with a few examples.
- Chain-of-thought, 
    - tells the LLM how to break down a problem into steps.
    - give an example
- Generated knowledge
    - provide generated facts or knowledge additionally to your prompt.
    - RAG
    - adking for specific details in a specifc manner
- Least to most, like chain-of-though, this technique is about
    - breaking down a problem in series of steps and then ask these steps to be performed in order.
    - ML steps : data collection -> cleaning -> modeling -> preformance
- Self-refine
    - this technique is about critiquing the LLM's output and then asking it to improve.
    - ask it to improve for some reason like security
    - Maieutic prompting.
        - What you want here is to ensure the LLM answer is correct and you ask it to explain various parts of the answer

--- ## skipping the rest
