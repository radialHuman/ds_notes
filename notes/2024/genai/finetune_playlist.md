# [Finetuning playlist](https://www.youtube.com/playlist?list=PLrLEqwuz-mRIEtuUEN8sse2XyksKNN4Om)

## [LLM Fine Tuning Crash Course: 1 Hour End-to-End Guide](https://youtu.be/mrKuDK9dGlg?list=PLrLEqwuz-mRIEtuUEN8sse2XyksKNN4Om)

### Training LLM
- Three approches
    - pretraining
    - fine tuning using tools like xoltols???
    - lora and qlora
#### 1. Pre-traning
- needs a massive data, text etc
- identify model architecture
- tokenizer : to encode and decode data, depending on the task
- dataset is preprocessed using tokenizers vocab to make it suitable for the model to ingest and learn from
    - mapping tokens to corresponding ids
    - incorporating spl tokens like masking
- In the pre training phase, model learns to predict the next sentence (text gen model, causal lang model) / filling missing words (masked lang model)
    - needs optmizing to various parameters like maximum likelihood of predicting the next word
    - self-supervised training is used 
    - to understand the semantics, lang patterns, general knowledge of language
    - lacks specific knowledge of domain/task
#### 2. Fine tuning
- Its like choosing a degree after 12th, specialization in a domain or task after getting a general knowledge of all the subjects
- Instruction tuned models steps
    - dataset of instruction and response pair is required (not as big as pre-training but still in 000s)
    - Optimize pretrained model to optimizes for task specific loss function 
    - prams of the pretrainded models are adjusted gradient based optmization algorithms (recent use cases)
        - like SGD, adam, etc
    - Improvement techqieus like these also can be used
        - lerning rate scheduling
        - regularization like drop outs or weight decay or early stopping
#### 3. LoRA (low rank adaptation of LLM)
- Used to make LLM run of less resources
- reduces GPU memory 3x (still too much so Quantize it in the enxt section)
- reduces learnable params by 10kx
- 

#### 4. QLORA
- To further reduce the resource and memory constriants, bitsandbytes library is used to reduce precision of floating values of prams
- Lossless quantization is not possible, but quite close

### Things to consoder
- llama2, mistral 7b can be used for free for commercial purpose
- [transformer mat 101](https://blog.eleuther.ai/transformer-math/) to find the compute required
- gpu on rent
    - runpod
    - vastai
    - lambda labs
    - AWS SM
    - Google colab
- gather dataset fom huggingface dataset
    - diversity in data
    - size of data set 10k q&a
    - quality of data, places a huge role too
        - phi-llm
        - orca
- [git repo to clone](https://github.com/OpenAccess-AI-Collective/axolotl)

### In detail
- LORA
    - makes the training process faster and consumes less memory using pairs of rank decomposition to create update matrices
    - presevation of the pre-trained weights is important to retain existing knowledge while adpating to the new data
    - portability of the trained weights : the rank decomposed matrices can be used anywhere else
    - Integration with attention layer : ???
    - hyperparameters in mistral qlora.yml in the above axolotl link
        - lora_r: lora rank, determines the # of rank decompostion matrices, recomemded is 8, proportional to better result but needs more compute and if the rank is the highest (hidden size of the model, in the config.json, or Trasformers automodel) is = full param fine tuning
        - lora_alpha: scaling factor which determines how much it has to learn from the new data. Lower the alpha more inclination towards original data's knowledge
        - lora_dropout: 0.05
        - lora_target_linear: true
        - lora_fan_in_fan_out:
        - lora_target_modules:
            - gate_proj : 
            - down_proj
            - up_proj
            - q_proj : query vector
            - v_proj : value vectors
            - k_proj : key vector
            - o_proj : output vectors
        - llm_head :  o/p layer of the model, for data has custom and complicated syntax
        - embed_token : 
        - norm : for stability and convergence of the model 
        - backpropogation : for gradients through a frozen 4bit quantized 
        - nf4 : 4 bit normal float datatype
        - page optimizers
        - doublt bit quantization
        - wandb : weights and biases 
        - spl tokens
        - num_epochs : hyper pram in GD
            - epoch is when the whole datset is aprsed
            - batch is when the model is updated
            - 1 epoch can contain multiple batches
    - for different models names might vary, details can be found by for looping over model.state_dict().keys()

## [Train a Small Language Model for Disease Symptoms | Step-by-Step Tutorial](https://youtu.be/1ILVm4IeNY8?list=PLrLEqwuz-mRIEtuUEN8sse2XyksKNN4Om)
### Aim
- To enter disease and get back its symptoms using GPT (small LM)
- https://huggingface.co/distilbert/distilgpt2
- 124M gpt2  -> 82 M distilgpt
- dataset : https://huggingface.co/datasets/QuyenAnhDE/Diseases_Symptoms

### Tokenization
- ngram, bigram are the original LM
- SKIPPING code explanation

## [Fine Tune Phi-2 Model on Your Dataset](https://youtu.be/eLy74j0KCrY?list=PLrLEqwuz-mRIEtuUEN8sse2XyksKNN4Om)
- MS's [phi2](https://huggingface.co/microsoft/phi-2) fine tuned for custom dataset
```python
!pip install -q torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate einops tqdm scipy

import os
from dataclasses import dataclass, field
from typing import Optional
import torch from datasets
import load_dataset, load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
HfArgumentParser, TrainingArguments )

from tqdm.notebook import tqdm

from trl import SFTTrainer

from huggingface_hub import interpreter_login
interpreter_login()
# enter the api key

# get the dataset, 3.5k rows
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")
dataset

# convert it to df
import pandas as pd
df = pd.DataFrame(dataset)

# formating for SLM's consumption, is deifferent for other LLMs, look for documentation
def format_row(row):
    question = row[ 'Context' ]
    answer = row[ 'Response' ]
    formatted_string = f"[INST] {question} [/INST] {answer} "
    return formatted_string

df['Formatted'] = df.apply(format_row, axis=1)

new_df = df.rename(columns = {'Formatted': 'Text'})
new_df = new_df[['Text']]

# finetuning
base_model = "microsoft/phi-2""
new_model = "phi2-mental-health" #after applying lora to original one

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_ side  = "right"
bnb_config = BitsAndBytesConfig( 
    load_in_4bit = True, # loading qlora version of the model
    bnb_4bit_quant_type = "nf4", # quantile quantization, smaller size format while maintaining a format
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = False # to avoid loss of performance
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config = bnb_config,
    trust_remote_code = True,  # for HF API key
    flash_attn = True, # flash_attn2 is not avaialbe for this right now
    flash_rotary = True, # 
    fused_dense = True, # gpu memory
    low_cpu_mem_usage = True, # 
    device_map = {" ", @}, # cuda
    revision = "refs/pr/23" # for commit to HF
)

model.config.use_cache = False # why?
model.config.pretraining_tp = 1 # 
model = prepare_model_for_kbit_training(model , use_gradient_checkpointing= True) #  

training_arguments = TrainingArguments(
    output_dir = "./mhGPT",
    num_train_epochs = 2,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 32,
    evaluation_strategy = "steps",
    eval_steps = 1500,
    logging_steps = 15,
    optim = "paged_adamw_8bit",
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    save_steps = 1500,
    warmup_ratio = 0.05,
    weight_decay = 0.01,
    max_steps = -1
)
peft_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    lora_dropout = 8.95,
    bias_type = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["Wgqkv", "fc1","fc2"]
)

trainer = SFTTrainer(
     model = model,
     train_dataset = training_dataset,
     peft_config = peft_config,,
     dataset_text_field = "Text"
     max_sequence = 690,
     tokenizer = tokenizer,
     args = training_arguments,
)

# time taking trainign
trainer.train()

```

## [llamafactory](https://youtu.be/iMD7ba1hHgw?list=PLrLEqwuz-mRIEtuUEN8sse2XyksKNN4Om)
- finetuning mistral using llmfactory (low code)
    - Various models: LLaMA, Mistral, Mixtral-MoE, Qwen, Yi, Gemma, Baichuan, ChatGLM, Phi, etc.
    - Integrated methods: (Continuous) pre-training, supervised fine-tuning, reward modeling, PPO and DPO.
    - Scalable resources: 32-bit full-tuning, 16-bit freeze-tuning, 16-bit LoRA and 2/4/8-bit QLoRA via AQLM/AWQ/GPTQ/LLM.int8.
    - Advanced algorithms: GaLore, DoRA, LongLoRA, LLaMA Pro, LoRA+, LoftQ and Agent tuning.
    - Practical tricks: FlashAttention-2, Unsloth, RoPE scaling, NEFTune and rsLoRA.
    - Experiment monitors: LlamaBoard, TensorBoard, Wandb, MLflow, etc.
    - Faster inference: OpenAI-style API, Gradio UI and CLI with vLLM worker.
- dataset :  MattCoddity/dockexNLcommands
- aim : to train to get docker commands
- [llamafactory](https://github.com/hiyouga/LLaMA-Factory)
- skipping as its low code


## [Make LLM Fine Tuning 5x Faster with Unsloth](https://youtu.be/sIFokbuATX4?list=PLrLEqwuz-mRIEtuUEN8sse2XyksKNN4Om)
- Optimizes a lot to gain speed [github](https://github.com/unslothai/unsloth)
- llama and mistral architecture
- Installation depends on GPU being used
- [trl](https://github.com/huggingface/trl)
- rope sclaing is inbuilt
- FastLanguageModel is used here
- SFTTrainer
- imdb Dataset is used frmo HF
- 