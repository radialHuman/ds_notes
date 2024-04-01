# [Fine tuning Optimizations - DoRA, NEFT, LoRA+, Unsloth](https://youtu.be/ae2lbmtTY5A)
- DoRA: https://arxiv.org/abs/2402.09353
- LoRA+: https://arxiv.org/abs/2402.12354
- Unsloth: https://github.com/unslothai/unsloth
- NEFT: https://arxiv.org/abs/2310.05914
- Transformers integration: https://huggingface.co/docs/trl/main/en/sft_trainer

## LORA : Low-Rank Adaptation
- To avoid trainign all the parameters for bespoke tasks
- LLMs ahave multiple modules, like in transformers diagram and modules have multiple matrices with param values
- If one matrix has 1024x1024  then it has 1M params
- Instead of tuning them all, an adapter is used which is 1M matrix is represented using 2 small matrices whose product is the big one
- ex: 1024x1 and 1x1024 matrices
- so only 16k pramas needs to be trained while the original one is freezed
- then trained is added to the original
- The advantage here is less # of updates, less memory and resouce.
- Also since all the params are updated in a group, there is a smoothing effect when compared to individual pram update
- Wn = Wo + Bt*A
- where Wo is the original freezed parameter and Bt and A are the adapters
- Here B is all 0 and A is random values , both are trainable

## LORA+ : Efficient Low-Rank Adaptation
- Variant of LoRA where learning rate is involved in training Bt and A
- If A is trained with LR then Bt is with 16x LR since its all 0 and must converge faster
- Optimizer needs to be modified to train B faster than A
- Not implemented in huggingFace yet may be in future, custom functions will not be required

## DORA : Weight-Decomposed Low-Rank Adaptation
- Here the Wo is seen as M*D which is magnitude and directional matrix
- M remains like a scalar while LoRA is implemented on D
- Wn = M(D+ Bt*A)
- M Bt and A are all trianable
- Training M is like changing its size
- Training D is like changing its direction using LoRA
- implementation is same as lora, just twio changes
    - user_dora=True
    - lora_magnitude_vector must be there in target module

## NEFT
- Adding gausian noise to embedding layers while training
- Not restricted to LoRA can be applied where embedding is involved
- Makes better high level features generalization
- just need to add NEFTUNE NOISE _ALPHA=5 in stftrainer

## UNSLOTH
- COMBINATION OF MANY SMALL TECHNICAL SPEED UPS
- https://unsloth.ai/blog/mistral-benchmark
- Ways to optimize mat mul was found and implemented
- Can be used only in llama and FastLanguageModel not CausalModel
- lora_dropout cant be used in this
- 

## CODE
- SKIPPING ntoebook setup
### LORA+
```python
# since its not implemented, a custom optimizer needs to be created
#TODO
```
### DORA
```python
# #If using DoRA (this may soon not be needed as DoRA will be part of transformers
!pip uninstall peft -y 
!pip install git+https://github.com/BenjaminBossan/peft.git@feat-dora -q


target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # "self_attn.rotary_emb.inv_freq",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lora_magnitude_vector", # <- for DORA
    # "input_layernorm.weight",
    # "post_attention_layernorm.weight",
    # "model.norm.weight",
    # "lm_head.weight", # ""dense_h_to_4h", #for falcon # "dense_4h_to_h", #for falcon # "query_key_value", #for falcon # "dense" #for falcon
],
lora_dropout=0.1,
bias="none",
task_type="CAUSAL_LM",
use_dora=True # <- for DoRA
```
### NEFT
```python
# under trainer = SFTTrainer(

# optimizers=(optimizer, None) # Comment in for LoRA+
neftune_noise_alpha = 5 # Add in noise to embeddings! ,<- for NEFT
```
### UNSLOTH
```python

```
