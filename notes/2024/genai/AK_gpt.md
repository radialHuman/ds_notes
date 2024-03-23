# [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY)
- training on small shakespear data
- Token : peices of words which is hot transformers consume and spit out text
- https://github.com/karpathy/nanoGPT
- model and train.py
- encoding text to numbers as tokenizers, many schemas can be sued
    - google sues sentencepiece (sub-word unit)
    - open ai : tiktoken
    ```python
         import tiktoken
         enc = tiktoken.get_encoding( "gpt2")
         enc.n_vocab
        50257
         enc.encode("hid there")
        ih, 4178, 612)
         enc.decode([71, 4178, 612])
        'hii there'
    ```
- all the text converted into a long list of numbers using simple string to ascii 
- split into validation and train
- block is a chunck of data for training 1/1
- since there has to be till a word and next, the total size of one training block is block size +1
- its so that the context of a word is learnt based on al the words that came before it in a block
- to make things faster, multiple blocks can be sent to train as batches
- bigram demo SKIPPING
## 42:00 Self attention block 
- 