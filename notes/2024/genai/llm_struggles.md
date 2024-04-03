## [Long-context LLMs Struggle with Long In-context Learning](https://youtu.be/bYiGlr777hg)
Release date : 03/04/24
### Idea
- Context windows are increasing from 32k to 2M
- This can help them do 
    - Long doc Q&A
    - Multi doc summarization
    - code understanding at repo level

### Details
- How does long context work:
    - alabi and rope embedding on short sequences
    - and apply them to longer oens during inference
    - context window sliding and segmentation in transformers
    - state state and recurrent models also helps
- Evaluation of these long windows is done by
    - LM perpexity over long document
    - pass key retirival task
    - and long doc Q&A and summarization
- Might not be a good infdicator for long seq task
- Like in school's comprehension, rather than understanding the whole para, based on the question, just a small part might be considered to answer
- ICL : in context learning on extreme label classification tasks
    -  this needs scanning the entire space and testing llms ability to comprehend the entire seq
    - long ICL bench has 6 levels
    - performance decreases with complex and more input length
    - distribution of labels impact the preformace of llms
    - extreme label space
- Result
    - Open Transformers outs perform SSSSS
    - API based outperforms both
    - 

### Resource
- [paper](https://arxiv.org/abs//2404.02060)

### misc
 
---
