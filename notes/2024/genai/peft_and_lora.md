# [Fine-tuning LLMs with PEFT and LoRA](https://youtu.be/Us5ZFp16PaU)

## Resources
- https://huggingface.co/blog/peft
- https://github.com/samwit/langchain-tutorials
- https://github.com/samwit/llm-tutorials

## PEFT
- by HF has multiple techniques to reduce the size of the model
- Prefix tuning
- P tuning
- Prompt tuning
- LORA
- Advantage
    - Freeze most of the prams and train only a few
    - original knoweledeg is not lost
    - extra added weights are only trained
- notebook demo
    - peft using bitsandbytes
    - lora checkpoints
    - hugging face hub is where fine tuned model can be saved to prevent it from losing when the notebook restarts
    ```python
    # load and convert the model
    import os os.environ[ "CUDA_VISIBLE_DEVICES" ]="0"
    import torch import torch.nn as nn
    import bitsandbytes as bnb from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1"
        load_in_8bit=True,
        device_map='auto',)
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1" )

    # freeze the original weights and prams
    for param in model.parameters():
        param.requires grad = False # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    model.gradient_checkpointing_enable() # reduce number of stored activations
    model.enable input_require grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.1lm_head)

    # setting up lora apadters
    def print_trainable parameters (model):
        # Prints the number of trainable parameters in the model.
        trainable params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable,params += param.numel() print (
        f"trainable params: {trainable params} || all params: {all_param} || trainablet: {100 * trainable_params})

    # config lora
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=16, #attlention heads
        lora_alpha=32, #alpha scaling
        # target_modules=["q proj", "v_proj"], #if you know the
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM" )# set this for CLM (GTP, decoder only) or Seq2Seq (T5)
    model = get_peft_model(model, config)
    print_trainable_parameters (model)
    ```
- data modeification as per syntax
-