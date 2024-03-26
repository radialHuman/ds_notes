# [Aligning LLMs with Direct Preference Optimization](https://youtu.be/QXVCqtAZAn4)
Release date : Feb 9 2024
### Idea
- Stanford found new way to align llms which was done via Human reinforcement learning (RLHF)
- Simple and memory efficient way

### Details
- What is alignment?
    - most base models are trained on general internet data
    - to make it work for a bespoke task, it needs supervised fine tuning
    - which is possible by adding a few 1000 exmaples of q&A with human input
    - this is done via SFT 
    - but the human input might be biased so there is a f/b mechanism in place to test with a few 100 exmp.
    - which will help eliminate the human bias
- old way RLHF
    - Step 1 : humans involved in creating datasets for fine tuning
        - they did not have good datasets in 2023
        - but now they do :
            - OpenHermes
            - dolphin
            - no robots
            - ultrachat
            - oasst2
            - OpenOrca
    - Step 2 : collect multiple answers from the model and rank them to retrain with reward involved
        - the ranking of the o/ps are done by human labelers
        - public datsets that can help in this teps are : 
            - hh-rlhf
            - ultrafeedback
            - orca-dpo-paris
            - helpsteer
    - Step 3 : combine the models from step 1 and 2
    - Disadvantage 
        - human bias
        - reward hacking by reward model to misunderstand or overfit
        - this can lead to gibberish
            - To avoid this OpenAI used KL penalty, 
            - this meaures distance between original and reward oriented chat model
        - RL is unstable
            - many hyper params to tune
        - CHat model, reward model and the last model to optimize between both
            - 3 models to train
            - this needs a lot of compute power
- DPO : Direct Preference Optimization
    - No need of RL
    - Instead of maximizing reward model
    - The objective needs to be changed so that directly encodes score which was supposed to come from reward model
    - So one model is reduced here
    - Mechanism (equation)
        - First a prompt is taken and then binary preferences is taken if the response is good or bad
        - Log proabilities of two terms are calculated to see if the model predicted the right response
            1. for good response it has the model to optimize divided by reference mode to normalize and avoid drifting too far
            2. similar for bad response 
        - This is combined with a hyper param beta to be tuned
        - and is maximized
        - Since its log prob,. its differentiable
        - So backprop can help in optimizing the model
    - Update process
        - Has 3 main terms
        - Weighting factor (sigmoid function) : finding difference between reward of getting incorrect response vs correct response
            - If model is making wrong reponse then this will increase to 1 and will penalize the model
        - likelihood of right answer
        - likelihood of wrong answer
- Zephyr is the LLM made using DPO
- HF's TRL can be sued to DPO
- Axolotl also is another library
- Extensions
    - (Identity) IPO : adds a regularization term to avoid overfitting
    - KTO (Kahneman-Tversky Optimization): Dispenses with binary preference altogether
        - stanford and COntextual AI 
        - loss function is altered since labelling good and bad is expensive
    - Iterative DPO (online) : Combines rejection sampling with DPO by snorkel
        - sequence of improved model while trainign
    - #TODO find out more about other RPO etc
- Demo
    - pretrainig -> supervised fine tuning -> Alignment
    - SFT
        - load datset -> format it into template -> SFT
    - template wit spl tokens using transformers.AutoTokenizer
        - chatml
        - llama2
        - zephyr
    - system message : how the llm must act
    - trl.SFTtrainer
        - training arguments
        - trainer
    - DPO
        - load dataset
        - labeled as choosen or rejected 
        - chat template
        - ```python
            def apply_chat_template(example, tokenizer):
                prompt_messages = example["chosen"][:-1] # Prepend a system message if the first message is not a system message
                if example["chosen"][0]["role"] != "system":
                    prompt_messages.insert(0, {"role": "system", "content": ""}) # Now we extract the final turn to define chosen/rejected responses
                    chosen_messages = example["chosen"][-1:] 
                    rejected_messages = example["rejected"}[-1:]
                    example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                    example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
                    example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            raw_datasets = raw_datasets.map( apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
            ```
        - DPO
            - ```python
                from trl import DPOTrainer
                from transformers import TratningArguments

                model_id = 'huggingface/Mistral-78-v6.1-DP0"

                training_args = TrainingArguments(
                    learning rate=5.8e-6
                    gradient_accumulation_steps=2
                    lr_scheduler_type="cosine"
                    max_Length=1024
                    max_prompt_length=512
                    num_tratn_epochs=1
                    per_device train_batch size=4
                    per_device_eval_batch_size=8,
                )

                BETA = 0.01 # DPO
                LOSS_TYPE = "sigmoid™ # DPO

                trainer = DPOTrainer( model_id
                    args=training_ args
                    beta=BETA
                    train_dataset=raw_datasets["train"]
                    eval_dataset=raw_datasets[*test"]
                    tokenizer=tokenizer
                    max_length=training_args.max_Length
                    max_prompt_length=training_args.max_prompt_length
                    loss_type=LOSS_TYPE,
                )

                trainer.train()
            ```
        - Tips to train 
            - Beta: test from 0.01 - 1.0
            - Learning rate: much smaller than for SFT ~1@@x smaller (5E-7)
            - Batch size: tradeoff between global batch size and n epochs
            - Optimizer: Adam appears better than RMSProp
            - Scheduler: Cosine > Linear 
            - The best SFT model != Best DPO model
            - LoRA: Appears to regularize the model compared to full fine-tune
    - Human evaluation is the best
    - Automated evaluzation
        openLLM LeaderrBoard - Not Chatbot focused, leakage, overfitting
        MT Bench - Usage
        Alpaca Eval - Usage LLamaindex (RAG)
        Human Eval - Lmsys Chatbot Arena
    - 

### Resource
- [Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb](https://colab.research.google.com/drive/1WNSVtM82oknmzL1QrJlNu--yNaWbp6o9?usp=sharing)
- [Fine-tune a SFT model with direct preference optimization (DPO).ipynb](https://colab.research.google.com/drive/1mWiOFBy3zY6OdINEvHN9EPoQ_VIvfFKw?usp=sharing)
- [slides](https://docs.google.com/presentation/d/1S8ao40-CdclRU0D2D9FdyN5x8fZL1Iv5/edit#slide=id.p1)
- [paper](https://arxiv.org/pdf/2310.16944)
- [training datasets](https://huggingface.co/collections/HuggingFaceH4/awesome-sft-datasets-65788b571bf8e371c4e4241a)
- [handbook](https://github.com/huggingface/alignment-handbook)
- [f/b datasets](https://huggingface.co/collections/HuggingFaceH4/awesome-feedback-datasets-6578d0dc8628ec00e90572eb)
- [beta effect](https://huggingface.co/blog/pref-tuning)

### misc
- DOUBTs
    - Ranking in RLHF had multiple o/ps whiel in DPO its just good or bad, is it losing here? : may be infuture they will include ranked 38:50

