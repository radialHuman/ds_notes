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