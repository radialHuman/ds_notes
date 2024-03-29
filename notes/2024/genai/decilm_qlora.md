## [Webinar: How to Fine-Tune LLMs with QLoRA](https://youtu.be/9Ieaf42tOnw)
Release date : 
### Idea
- What's the Difference Between a Base Model and a Fine-tuned Model?
- Prompt Engineering, Fine-tuning, RAG, PEFT
- Issues with Fine-tuning
- Fine-tuning for the GPU Poor
- LoRA and QLORA overview

### Details
- Pre-training: Learns from massive text data to predict words and understand grammer
- Fine-tuning: Specializes on specific tasks using narrower datasets.
    - model params will get updated
- Prompt Engineering: Refines model input to guide its output.
- RAG (Retrieval Augmented Generation): Merges prompt engineering with database querying for context-rich answers,by adding custom data to llms
- Full Fine-tuning: Adjusts all parameters of the LLM using task-specific data, expensive
    - High computational costs due to updating billions of parameters. 
    - Significant memory needs requiring advanced hardware.
    - Time-intensive and demands expertise for large-scale models.
    - Issues like catastrophic forgetting, storage, hyperparameter tuning
- Parameter-efficient Fine-tuning (PEFT): Modifies select parameters for more efficient adaptation, less expsive, size remain the same
    - Extra component or part of pretrained LLM
    - just adjusting few parameters
    - either train some portion of roginal
    - else add adapters and tune them only
- QLORA
    - LORA : lowering the size of matrixes containing change in weights from original in a decomposed form ther eby reducing size
        - targetting certain attention and linear layers
        - making them into two 
        - B with all 0 and A with random normally distributed values, which are to be trained
        - LLMs are inherently low rank already
    - QLORA : what ever the param values are of the LORA representation, decreasing the precision of the floating points there by reducing the overlal size of the model drastically
        - 4-bit NormalFloat: A new data type that keeps up with 16-bit performance. Ideal for Neural Networks: Works great where weights are normally disstributed, a common scenario in neural networks.
        - Efficient Format: Combines a shared exponent across parameters with a 4-bitt mantissa for each, using only a quarter of the bits of 16-bit floats.
        - Dynamic Range & Precision: Achieves a dynamic range similar to higher bit- widths and minimizes quantization error across typical weight distributions.
        - double quantization
            - Twice the Quantization: First quantizes network parameters, then quantizes the quantization levels.
            - Reduces Bit Requirements: Drops quantization constant bit needs significantly. e & Counterintuitive but Effective: 
            - Uses redundancy to compress the model with minimal performance loss.
            - Massive Storage Savings: Reduces overhead from 0.5 to 0.125 bits per parameter.
            - Keeps Computational Performance: Maintains model's computational efficiency while boosting storage efficiency.
        - Page optimizers
            - Automatically offload data to handle memory spikes, keeping training fast on GPU and adaptive to memory needs.
        - 
```python
# Here's what you need to get started fine-tuning with QLoRA using HuggingFace's SFTTrainer:
# + Alanguage model « A tokenizer
# + Adataset « Two configuration files: BitsAndBytesConfig and LoraConfig
# « You'll also need to use the prepare_model_for_kbit_training and get_peft_model functions from the peft library to, well prepare your model and get it into a peft format.
# « And, training parameters for the SFTTrainer

# First, just a couple of preliminaries.
# « Set the UTF encoding for the environment so that peft doesn't yell at you during installation. 
# + Create a directory for checkpoints + Install libraries that you will need
# « Log into HuggingFace so you can immediately push the trained model to the hub without baby sitting the notebook

import os 
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8' 
os.environ['LC_CTYPE'] = 'en_US.UTF-8'

from huggingface_hub import notebook_login
notebook_login()

!pip install -qq git+https://github.com/huggingface/peft.git # configurationa dn helper functions
!pip install -qq accelerate # for easy gpu running
!pip install -qq datasets # to get data from huggin face
!pip install -qq trl # to get supervised traininer
!pip install -qq transformers # to get pre tuned models
!pip install -qq bitsandbytes # to do QLora
!pip install -qq safetensors # 
# note: flash attention installation can take a long time 
!pip install flash-attn —no-build-isolation # there is a version 2 avaialbe as of 03/24

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
import torch


#This configuration below is literally the only thing that differentiates LoRA from QLoRA.
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True, # enables 4 bit quantization
    bnb_4bit_use_double_quant=True, # enables double quntization for even smaller size
    bnb_4bit_quant_type="nf4", # calling in spl data format, 4 bit normal float
    bnb_4bit_compute_dtype=torch.bfloat16, # 
)


# Load data
model_name = "Deci/DeciLM-6b" # can be replaced with any model

decilm_6b = AutoModelForCausalLM.from_pretrained( 
    model_name,
    quantization_config = bnb_config,
    device_map = "auto",
    use_cache=False,
    trust_remote_code=True,
    use_flash_attention_2=True, # to foucs on attention layers better
)
tokenizer = AutoTokenizer.from_pretrained(model_name) 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# custom data to be trained on
from datasets import load_dataset
dataset = "TeamDLD/neurips_challenge_dataset" # can be replaced based on use case
data = load_dataset (dataset)

# format the data as per use case or add tokens using functions SKIPPING

# trian test split, based on gpu space, needs modification
from typing import Tuple
from datasets import concatenate_datasets, Dataset
import os
def split_dataset_by_category( data: Dataset, total_samples: int, category_field: str, category_values: list, category_ratios: list, seed: int = 42
) —> Tuple[Dataset, Dataset, Dataset]:
'''Splits a dataset into categories with specified ratios and then into train, validation, and test splits.
Parameters:
- data (Dataset): The dataset to be split.
- total_samples (int): Total number of samples in the final subset.
- category_field (str): The field in the dataset to filter by for categories. — category v list): A list of tegory values to filter .
- category_values (list): A list of category values to filter by.
- category_ratios (list): A list of ratios corresponding to each category in 'category_values'. - seed (int): Random seed for shuffling the dataset.
Returns: - Three separate Dataset objects: train_data, val_data, test_data

'''
    subsets = [] 
    for value, ratio in zip(category_values, category_ratios): 
        samples = int(total_samples * ratio) 
        filtered_dataset = data.filter(lambda example: example[category_field) == value, num_proc=os.cpu_count())
        subset = filtered_dataset.shuffle(seed=seed).select(range(samples) )
        subsets. append(subset)
    # Concatenate the subsets final_subset = concatenate_datasets (subsets) .shuffle(seed=seed)
    # Split the dataset into train, test, and validation sets
    train_test_split = final_subset.train_test_split(test_size=0.25)
    train_data = train_test_split['train']
    test_data = train_test_split['test']
    train_val_split = train_data.train_test_split(test_size=0.3333) # 0.25 of the remaining 75% to make it 25% of
    train_data = train_val_split['train']
    val_data = train_val_split['test']

    return train_data, val_data, test_data

train_data, val_data, test_data = split_dataset_by_category(
    data=data_with_type,
    total_samples=7500,
    category_field='source',
    category_values=[
        'nampdn-ai/tiny-codes'
        'emrgnt—cmplxty/sciphi-textbooks—are-all-you-need'
        '@-hero/0IG-small-chip2'
        'wenhu/TheoremQA'
        'iamtarun/code_instructions_120k_alpaca'
        'Nan-Do/code-search-net-python'
        'lighteval/logiqa_harness'
        'WizardLM/WizardLM_evol_instruct_70k'
        'databricks/databricks—dolly-15k'
        'Lighteval/boolq_helm' 
    ],
    category_ratios=[0.15,
    0.05,
    6.2,
    0.05,
    0.08,
    0.08,
    0.08,
    @.15,
    0.08,
    0.08]
    )

# QLORA hypertuning 
import peft
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model 
# we set our lora config to be the same He qlora
lora_config = LoraConfig(
    r=16, # rank, more compelx data needs higher r, hjigher for nuanced task, a good value is 256
    lora_alpha=32, # usually double of rank, higher is closer to original llm than new data
    lora_dropout=0.1, # 13B, 0.05 for larger models
    bias="none",
    task_type="CAUSAL_LM"
)


# prepare model, based on config mentioned above
decilm_6b = prepare_model_for_kbit_training(decitm_6b)

# get the peft model
decilm_6b = get_peft_model(decilm_6b, lora_config)

# trainign parametrs
import transformers from transformers import TrainingArguments

training_args = TrainingArguments(output_dir=output_dir
    evaluation_strategy="steps"
    do_eval=True
    auto_find_batch_size=True # Find a correct batch size that fits the size of Data. log_level="debug"
    optim="paged_adamw_32bit" # can be 8 bit too
    save_steps=25
    logging_steps=25
    learning_rate=3e-4
    weight_decay=0.01
    max_steps=125 # mmore will need mroe time 
    warmup_steps=25
    bf16=True
    tf32=True,
    gradient_ checkpointing=True,
    max_grad_norm=0.3, #from the paper
    lr_scheduler_type="reduce_lr_on_plateau",

)
# bringing it all together
trainer = SFTTrainer(
    model=decilm_6b,
    args=training_args,
    peft_config=lora_config,
    tokenizer=tokenizer,
    formatting_func=example_formatter, # function whichw as skipped
    train_dataset=train_data,
    eval_dataset=val_data,
    max_seq_length=4096,
    dataset_num_proc=os.cpu_count(),
    packing=True
    )
trainer.train()

trainer.save_model()

# Now, merge the adapter weights to the base LLM.
from peft import AutoPeftModelForCausalLM
instruction_tuned_model = AutoPeftModelForCausalLM. from_pretrained( 
    training_args.output_dir, 
    torch_dtype=torch. bfloat16, 
    device_map = 'auto', 
    trust_remote_code=True,
)
merged_model = instruction_tuned_model.merge_and_unload()


# push my merged weights to the HuggingFace Hub.
HF_USERNAME = "harpreetsahota" 
HF_REPO_NAME = "DecilLM-6B-instruction—tuned"
merged_model.push_to_hub( f"{HF_USERNAME}/{HF_REPO_NAME}")
tokenizer.push_to_hub( f" {HF_USERNAME} /{HF_REPO_NAME}"')

tokenizer.pad_token_id = tokenizer.eos_token_id
def get_outputs(inputs):
    outputs = instruction_tuned_model.generate( 
        input_ids=inputs["input_ids"]
        attention_mask=inputs ["attention_mask"]
        max_lLength=1000
        do_sample=True
        early_stopping=True
        num_beams=5
        temperature=0.01
        eos_token_id=tokenizer.eos_token_id,
    )
    return outputs


def generation_formatter(example):
    # Joins the columns 'instruction' and 'input' into a string with each part separated by a newline character. Adds "### Response:\n" at the end to indicate where the model's output should start. If 'input' is None, it is replaced with "### Input:\n".
    # Parameters: - example (dict): A dictionary representing a row with the keys 'instruction', 'input'.
    # Returns: - str: A formatted string ready to be used as input for generation from an LLM.
    # Check if 'input' is None and substitute the placeholder text
    input_text = "### Input:\n" if example['input'] is None else example['input'] # Return the formatted string with placeholders for 'input' and 'response' 
    return f"{example['instruction']}\n{input_text}\n### Response: \n"

print(generation_formatter(test_data [42] ))

# generate mdol response
input_sentences = tokenizer(generation_formatter(test_data[42]) return_tensors="pt").to('cuda')
outputs_sentence = get_outputs(input_sentences)
results = tokenizer.batch_decode(outputs_sentence, skip_special_tokens=True)
print(results[0] )


```
### Resource
- [blog](https://deci.ai/blog/how-to-instruction-tune-a-base-llm-using-qlora-with-decilm-6b/)
- [blog](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html)
- [youtube](https://www.youtube.com/watch?v=fQirE9N5q_Y&pp=ygURdGltIGRlbG1lZXIgcWxvcmE%3D)

### misc
 
---
