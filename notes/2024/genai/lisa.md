## [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://youtu.be/BYZ7H9JR9mU)
Release date : 01/04/24
### Idea
- LORA's performance still fails to match full parameter training in most large-scale fine-tuning settings
- we investigate layerwise properties of LoRA on fine-tuning tasks and observe an uncommon skewness of weight norms across different layers
- Layerwise Importance Sampled AdamW (LISA)
- which applies the idea of importance sampling to different layers in LLMs and randomly freeze most middle layers during optimization
- less GPU memory consumption, LISA surpasses LoRA or even full parameter tuning in downstream fine-tuning tasks

### Details
- 

### Resource
- [paper](https://arxiv.org/abs//2403.17919.pdf)

### misc
 
---
