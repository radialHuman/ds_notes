# [Finetuning LLM](https://www.youtube.com/playlist?list=PLZoTAELRMXVN9VbAx5I2VvloTtYmlApe3)
## Steps By Step Tutorial To Fine Tune LLAMA 2 With Custom Dataset Using LoRA And QLoRA Techniques

### 1. Parameter-Efficient Transfer Learning for NLP (peft)
- !pip install -q accelerate==0.21.@ peft==@.4.0 bitsandbytes==@.40.2 transformers==4.31.@ trl==@.4.7
- LoRA: Low-Rank Adaptation of LLM
- bitandbytes : quantization : weights are floating values, round off to reduce size of the model
- peft
    - most the weights of an LLM is freezed but some will get retrained to gain better scores
- in this case llama 2 is used to train
    - prompt template used here is 
    <s>[INST] <<SYS>>
    System prompt <</SYS>> 
    User prompt [/INST] Model answer </s>
    - dataset needs to be converted into this format
    - the data set used here has aa different format
    - then a sample of newly formatted dataset is taken 
- llmama 2 is a huge model with 7b parameters
- needs gpus for storing and other computation
- which might not be avaialable or cost effective
- hence quantizing using peft the model will help 
- moreover qlora and lora can be used to reduce the precision of the weights to reduce size further
    - qlora has 2 hypertuning parameter rank, dropout and scaling (alpha)
    - many such parameters are required, can be found in documentation
    - load it into self-finetuning trainer

---

## [In depth intution of quantization](https://youtu.be/6S59Y0ckTm4)
### 1. Quantization
- to reduce memory format, by reducing precision of weights and other parameters
- FP : full precision/single precision
- example : weights stroed in 32 bit FP can be reduced to int 8 or 16 FP (half precision) bit to make the model consumeable by local or smaller hardware
- useful to deploy in edge devices or running things locally
- CON : loss of info as usualy
- overcome using : 
    - ...
### 2. Caliberation
- How to perform quantization/convertion to lower memory format
- Symmetric quantization
    - like batch normalization (weights are centred around 0)
    - symmetric unsigned int 8 Q
        - unsigned FP 16 is stored with 1 bit for sign, 5 bit for non-decimal number and the rest for decimal
        - min max scaler can be used to reduce the number range into something smaller with equivalent values
        - example 0-2000 -> 0-255
        - scaling factor = (2000-0)/(255-0)
        - all the numbers in 0-2000 will be then divided by scaling factor
        - and then round them off to get a quantized int weight
- Asymetric quantization
    - the values of weights are not equally or evenly distributed
    - #DOUBT how to find that?
    - example : -20  to 2000 -> 0-255
    - minimum after scaling is -5 which needs to be offsetted by adding zeropoint to all the numbers
- hence quantization has 2 parameters : scale and zero point (0 for symmetric)

### 3. Modes of quantization
1. Post training ; first the model is trained, and the weights are fixed, then quantization is done to use it
    - loss of data so less accuracy
2. Quantization aware training : trained model, quantized, using new training data fine tune it

---

## [Qlora and LORA maths](https://youtu.be/l5a_uKnbEr4)
### LoRA
- used while finetuning a base foundational model
1. full parameter fine tuning is expensive
    - time and computation wise
    - model monitoring, model inference, gpu, ram
2. domain specific fine tuning
3. specific task fine tuning
- to over come disadvantages of full para. fine tuning use LORA or QLORA (lora 2.0)
#### Math
- while full para. tuning, the change in weights are captured in smaller matrix
- 3x3 <-> 3x1 and 1x3
- done via matrix decomposition based on RANK
- There will be a loss in precision, but the number of paramters stored will be reduced
- W + dalta(w) = W + BA
- where b and a are the smaller matrix
- the rank of these smaller matrix can be increased but will be less than or equal to the full param. fine tuned 
- since there are 3 matrix in transformers  q,k,v and o there will be 4 ranks
- earlier tech of fine tuning : prefix embed, prefix layer, adapter
- rank is proportional to complex learning task
### QLORA
- quantization of the values of decompoised matrix to move form full precision to half and so on to reduce the size even further

---

## [LLMops](https://youtu.be/4ijnajzwor8)
- When an llm app needs various api calls to get data, many configs and keys needs to be taken care of.
- Platform vext is being used here to simplify all the piepline
- no code ui based
- SKIPPING 

## [1bit LLM](https://youtu.be/wN07Wwtp6LE)
- bitnet is where parametes are either -1, 0 or 1
- simplifies mutiplication and addition to just addition
- less requirement of gpu
- feature filtering is easy as it makes the value 0
- absolute mean quantization function is used to convert values to -1,0,1
- memory and latency is both less when compared to actual model
