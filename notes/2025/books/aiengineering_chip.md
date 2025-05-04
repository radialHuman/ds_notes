## Chapter # : 1
### Idea
- building applcaitions
- using these tools to solve actual problems
- github code exc and exam

### Details
- some concepts are same as in ml
- but the complexity and the scale is huge
- new challenges : ai engineering
- focusing on
    - understanding ml and ai to integrate it and build solutions
- questions 
    - does ai fit here? may be its not required
    - should it be built form ground up?
    - why and if
- use cases
    - coding
    - image and video geenration
    - writing help
    - education 
    - chat bots
    - info gatehring
        - nlp to db
        - organization
        - workflow automation
            - agents
    - synthetic data generation
    - personalization
- llm vs foundation models
    - multimodal : foundation
    - llm are just text
- build vs buy
    - buy if complex and expsive to build
    - build if control security and long term cost 
- critical vs complementatry ai distinction
    - need vs want
- Ai engineering stack
    - layer 1 : applciation : find whats the use of the app, experiemnt with existing models, poc quick
    - layer 2 : model development 
    - layer 3 : infra

- sampling : how they work internally, can lead to hallucination
- training data :  if your knowledge is not in the data, it can no way geenrate it
    
### Resource


### misc

---

## Chapter # : 2
### Idea
- how foundational models are made?
- designing and behaviours

### Details
- training data
    - quality
    - diversity
    - language dominance
    - domain specific
- Architecture
    - seq to seq was the beginning
    - encoder compresses the informaiton into representation
    - decoder takes the representation and generates the output
    - usually used in rnns, in sequential manner
    - attention  :no more sequentially, it should all be process in one go
        - dynamically wieghthe different parts of input
        - k q v matrix
        - mlp helps learn non linear complex real world data
        - the activations ened not be advanced even the simple ones will work well here
        - 

### Resource


### misc
