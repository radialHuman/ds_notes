## [Generative AI Foundations on AWS | Part 1: Introduction to foundation models](https://youtu.be/oYm66fHqHUM?list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF)
### Intro
- Customize models via
    - Pretraining
    - Fine-tuning 
    - Retrieval augmented generation
    - Prompt engineering
    - trade off between Accuracy and Complexity & cost
- human feedback via RL (reward model)
- timeline
    1. transformers 2017
    2. bert 2018
    3. megatron-lm 2019
    4. gpt 3 2020
    5. InstructGPT, Cohere, Al21, GPT-J 2021
    6. PaLM, Bloom, ChatGPT 2022
    7. GPT4, anthropic, stable diffusion 2023
- AWS partner : Al21 Jurassic-2 Jumbo Instruct
- Demo
    - falcon model foundation model in AWS sagemaker
    - playground to chat directly without deploying
    - skipping demo as its just an intro

## [Part 2: Pick the right foundation model](https://www.youtube.com/watch?v=EVqTWGafpfo&list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF&index=2&pp=iAQB)
### Considerations
- modalities, task, size,
- accuracy, ease-of-use, licensing, previous
- examples, external benchmarks
- MODALITY : language or vision
    - CLIP : image to text model
    - stable diffusion, deepFloyd : text to image
    - gpt, bert : text to text 
    - dreambooth : image to image
- ML task can be recast as generative without building specific classic model for the task
    - just prompt if the sentence is +ve or -ve
    - one foundation model for all tasks
- Size of the model
    - GPU computation
- generative types
    - decoder only
    - encoder-decoder
- define accuracy
- open or proprietary model
- find reviews online, paper, examples
- Develop unit tests and edge cases for your domain.
- find, use, master external benchmarks.
    - standford's Holistic Evaluation of Language Models  (HELM), hugging face
- Get ready when a better model comes in
- Demo
    - jumpstart endpoint in sagemaker
    - skipping

## [Part 3: Prompt engineering and fine-tuning](https://youtu.be/RK9bLf8a5Lo?list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF)
### Prompt engineering: zero-shot, single-shot, few-shot
- desired o/p by manipulating input text
- this can help in creating prompt template, which will take in just the parameters which vary for user but use the same way of prompting as a whole
- the way of finding which i/p gives best o/p is via syntax hacking
    - varies from model to model 
- zero shot : no spl i/p
- single and few shot : gives 1 or more examples in prompt
- prompt tuning : train task specific vectors 
### Instruction examples: Summarization, classification, translation
- Intructed models perform way better
- less halucination and gibberish
- Instruction tuning uses supervised learning to adapt the model's behavior
    - create own i/p and desired o/p as a classification task
    - can be done on a base foundational model
    - to make sure the model follow instructions
- prompting techniques
    - zero shot is y:?
    - single shot : is like x:y::y:?
    - few shot is multiple single shot
- commands
    - paste entire text of a paragraph and type summarize
- classification task
    - paste everything and in the end just ask a question about it
- translate
    - paste and type translate it to some language
    - can be used for code
### fine tuning : when prompt eng. fails
- create new model artifact
- types
    - Parameter efficient fine-tuning : hugging face 
        - adjusting weights
    - LoRA : low rank adpatation
        - find under utilized aspects of model and fine tune that
        - https://github.com/huggingface/peft
    - prefix tuning : hugging face
        - 
    - Transfer learning : add more layers for bespoke tasks and train them
        - more compute efficient
    - classic fine tuning : fine tuning a LLM with extra head 
        - more complicated than just transfer
    - continued pretraining : 
        - not sure what this was
### Fine-tuning: classic, parameter efficient, controlled
- [notebooks](https://github.com/aws/amazon-sagemaker-examples)
### Hands-on walk through: SageMaker JumpStart
- Skipping

## [Part 4: Pretraining a new foundation model](https://youtu.be/0xfe54_pYIQ)
### Prerequisite
- Must have trried all the techq mentioned above to make an existing model better at the task
    - 0,1,few,parameter fine tuning, classic fine tuning
- needs a lot of gpus running for more than 1 month to pretrain
- TBs in vector database
- tens of computer nodes
- so business case is strong only then do this
- prove at 1% of data that pretraining gets significant results before adpating it fully
### Steps
1. data in bucket
2. process the training data
3. Optimize data storage using FSX cluster
4. notebook for test training scripts
5. scale up
6. evaluate
## [Part 5: Preparing data and training at scale](https://youtu.be/QpPpbM0FQ1Y)
## [Part 6: RL with human feedback](https://youtu.be/An-ha4YzxXo)
## [Part 7: Deploying a foundation model](https://youtu.be/TGCe3FXDgGY)