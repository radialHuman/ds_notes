## [The A to Z of Prompt Engineering - Intro to Prompt Engineering](https://youtu.be/QYs90ps6rxk)
Release date : 29/03/24
### Idea
- Intro to Prompt Engineering — Today
- Common Prompt Engineering Techniques — Apr 2"
- Advanced Prompt Engineering Techniques — Apr 9"

### Details
#### Models in Azure
- multi modal generation : gpt, claude
- Emebdding generation : text-embedding
- fine tuning
- Depending on the model and its purpose, prompt differs

#### types of prompt in stateless model
- system : to tell the model how to behave or act
- user : the i/p from user
- response

#### Things to keep in mind
- Use the latest model for best result
- Put instructions at the beginning of the prompt and use ### or """ to separate the instruction and context
- Be specific, descriptive and as detailed as possible about the desired context, outcome, length, format, style, etc
- Articulate the desired output format through examples
- Reduce "fluffy" and imprecise descriptions
- Instead of just saying what not to do, say what to do instead
- Use all caps for heading
- use --- for seperation of sections
- its not just about sending words but also the way it looks (presentation)
    - XML or markdown format also can be sued as they are in the dataset
- by giving a persona
    - Models perform better when given a specific area of expertise to focus on
    - Imagine a use case of code conversion/translation using LLMs
- Prime the output
    - Add phrases at the end of the prompt to obtain a model response in a desired form
- Instruct the model to reason on a prompt before jumping into answering
- LLMs often perform better if the task is broken down into smaller step (useful for bigger query)
- Chain of Thought Prompting
    - Instruct model to proceed step-by-step and present all the steps involved

#### Learnign techniques
- Zero-shot - Predicting with no sample provided
- One-shot - Predicting with one sample provided
- Few-shot — Predicting with a few samples provided

- self reflection


#### Prevent hallucination
- Without good prompt design and grounding techniques, models are likely to hallucinate or make up answers.
- The danger is that models often generate highly convincing and plausible sounding answers so great care must be taken to design safety mitigations and ground model answers in facts.
- Good prompt engineering can instruct the model towards what it should and should not
generate. This includes instructing the model to only use information from specific sources and not to extrapolate.
- Grounding solves this by anchoring model completions to factual information.
- In addition to using grounding techniques, you most likely want to reduce the temperature setting.
- High temperature settings (closer td 1) allow the model to introduce more randomness into the response which®ncreases the chance of hallucinations. Lower temperature values make the model responses more predictable.
- Top Probabilities is another parameter that effects model randomness.

#### Grounding and prompt stuffing
- Because GPT models are generative they do not generate "factual" or "truthful" information. True answers may be generated because they are probabilistically the right answer, but the model cannot differentiate between true and untrue answers. It is simply choosing the most likely tokens based on the context and pretraining
- Grounding is the process of getting generative Al models to be grounded in true information so that they produce correct answers. This is generally done by combining information retrieval techniques (e.g. search and queries) with generative Al models to generate truthful answers. »
- One technique for accomplishing this is prompt stuffing where the document or data containing the true answer are fed to the prompt and the model is instructed to only use information from the data in the prompt. This is often done as part of metacontext design.

#### Fact checking
- Instruct the model to answer by referring to context provided
- Instruct the model to create a citation list and reference it within its answer
- Instruct the model on what to say when no relevant data is provided to answer a question

#### Gaurdrails
#### Prevent jailbreaking
#### parameter control
- Temprature 
- top_p

### Resource
- 

### misc
 
---
