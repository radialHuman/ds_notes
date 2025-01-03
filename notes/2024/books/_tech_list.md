**Prompt Engineering:**

*   Sub-techniques:
    *   In-context learning
    *   Chain-of-thought
    *   Tree-of-thought
    *   Zero-shot prompting
    *   One-shot prompting
    *   Few-shot prompting
*   Libraries: 
    *   LangChain
*   Tools: 
    *   Galactica
    *   ChatGPT

**Retrieval-Augmented Generation (RAG):**

*   Sub-techniques:
    *   Dense retrieval
    *   Reranking
*   Libraries:
    *   Cohere
    *   Sentence Transformers
    *   LangChain
*   Tools:
    *   Vector databases

Vector databases utilize various techniques to efficiently store and retrieve high-dimensional vectors, primarily focusing on approximate nearest neighbor (ANN) search algorithms. These algorithms strive to find vectors that are "close" to a given query vector, even within massive datasets. Here is a list of commonly used algorithms:

**Exact Search:**

*   **Brute-Force Search (Linear Scan):** This straightforward approach calculates the distance between the query vector and every vector in the database. While simple, it's computationally expensive and impractical for large datasets.

**Approximate Nearest Neighbor (ANN) Search:**

*   **Tree-based Methods:**
    *   **KD-Tree:** Partitions the data space into hierarchical regions, enabling efficient pruning of the search space. Effective for low-dimensional data but struggles with high dimensionality. 
    *   **Ball Tree:** Similar to KD-Trees but partitions using hyperspheres instead of hyperplanes, better handling high-dimensional data. 
*   **Hashing-based Methods:**
    *   **Locality-Sensitive Hashing (LSH):** Uses hash functions to map similar vectors into the same buckets, reducing the search space. Offers probabilistic guarantees but can be sensitive to parameter tuning. 
*   **Graph-based Methods:**
    *   **Hierarchical Navigable Small World (HNSW):** Constructs a navigable graph connecting data points, allowing efficient traversal and approximate nearest neighbor search. Known for its good performance and scalability.
*   **Clustering-based Methods:**
    *   **K-Means Tree:** Combines K-Means clustering with tree-based search, offering a balance between accuracy and speed.
*   **Product Quantization (PQ):** Divides the vector space into subspaces and quantizes each subspace independently, compressing vectors and enabling faster distance calculations.
*   **Inverted File Index (IVF):** Clusters vectors and builds an inverted index based on cluster assignments, speeding up search by narrowing down candidates.

**Hybrid Approaches:**

*   Many vector databases combine multiple techniques to optimize performance. For instance, **FAISS**, a popular vector search library, often uses a combination of **IVF and PQ** for efficient retrieval.

**Other Techniques:**

*   **Filtering:** Vector databases may use metadata filters to restrict the search space, improving retrieval speed. For example, filtering by document type or date range before performing similarity search.

The choice of algorithm depends on factors like:

*   Dataset size
*   Dimensionality of the vectors
*   Accuracy requirements
*   Latency constraints
*   Available computing resources

The sources highlight the use of **HNSW** as an effective indexing technique and mention **FAISS** as a popular library for vector search, often employing a hybrid approach combining **IVF and PQ**. Additionally, they emphasize the importance of filtering based on metadata to improve retrieval speed.


**Quantization:**

*   Sub-techniques:
    *   GGUF model compression
    *   INT8 Quantization 
*   Libraries: 
    *   llama.cpp
    *   GPTQ
*   Tools: 
    *   TensorRT

**Text Classification:**

*   Sub-techniques:
    *   Supervised Classification
    *   Unsupervised Classification
    *   Zero-shot classification
    *   Few-shot classification
*   Libraries:
    *   Sentence Transformers
    *   Transformers
*   Tools:
    *   BERT
    *   RoBERTa
    *   DistilBERT
    *   ALBERT
    *   DeBERTa
    *   Flan-T5
    *   GPT-3.5 

**Text Clustering:**

*   Sub-techniques:
    *   Dimensionality Reduction
*   Libraries: 
    *   BERTopic
*   Tools:

**Topic Modeling:**

*   Sub-techniques:
    *   Bag-of-words
    *   c-TF-IDF
*   Libraries: 
    *   BERTopic
    *   datamapplot
*   Tools: 
    *   KeyBERT

**Creating Text Embedding Models:**

*   Sub-techniques:
    *   Contrastive learning
    *   Cosine similarity loss
    *   MNR loss
*   Libraries: 
    *   Sentence-transformers
*   Tools: 
    *   BERT

**Fine-tuning:**

*   Sub-techniques:
    *   Supervised fine-tuning
    *   Preference tuning
    *   Instruction fine-tuning
    *   Classification fine-tuning
*   Libraries: 
    *   OpenAI API
*   Tools:
    *   GPT-2

**Prompt Tuning:**

*   Libraries: 
    *   PEFT
    *   Transformers

**Knowledge Distillation:**

*   Sub-techniques:
    *   Soft Targets

**Mixture of Experts Finetuning:**

*   Libraries: 
    *   DeepSpeed

**Other tools mentioned:**

*   whylogs
*   Langkit
*   vLLM
*   OpenLLM
*   Text-Generation-Inference (TGI)
*   TitanML
*   Kubernetes
*   Llama.cpp
*   tiktoken
*   prodi.gy 
*   doccano
*   d5555/TagEditor
*   Praat 
*   Galileo

The sources do not specify sub-techniques, libraries, or tool names for **Text Clustering**.


# Transformers 

The Transformer architecture is a deep neural network design that has revolutionized natural language processing (NLP) tasks. At its heart lies the concept of **self-attention**, allowing the model to weigh the importance of different words in a sentence to better understand their context and relationships. The Transformer can be broadly divided into two main submodules: **the encoder and the decoder**. While the original Transformer utilized both encoder and decoder for machine translation, modern LLMs like GPT primarily employ the **decoder-only architecture** for text generation.

Here's a breakdown of each layer within the Transformer, particularly focusing on the decoder:

**1. Input Processing:**

*   **Tokenization:** The input text is first broken down into individual words or subword units called tokens using a tokenizer, such as the Byte Pair Encoding (BPE) tokenizer. 
## **Embedding Layer:** 
Each token is then converted into a numerical vector representation called an embedding. These embeddings capture semantic information about the tokens.
The **embedding layer** in a Transformer is a crucial component responsible for converting text input into numerical representations that the model can understand and process. It acts as a bridge between the discrete world of words and the continuous vector space where the Transformer operates. Here's a detailed explanation of how the embedding layer functions:

**1. Tokenization:** Before embedding, the input text is divided into individual units called tokens. These tokens can represent words, subwords, or even characters, depending on the chosen tokenizer. For example, the sentence "This is an example." might be tokenized into ["This", "is", "an", "ex", "ample", "."].

**2. Creating an Embedding Vocabulary:** The embedding layer relies on a vocabulary of all the possible tokens that the model can encounter. Each unique token in the vocabulary is assigned a specific index or ID.

**3. Embedding Matrix:** The core of the embedding layer is an embedding matrix. This matrix is a table where each row corresponds to a token in the vocabulary, and each column represents a dimension in the embedding space.  The dimensions of the matrix are:
    * **Rows:** Number of tokens in the vocabulary (`vocab_size`)
    * **Columns:** Embedding dimension (`emb_dim`), which is a hyperparameter that determines the size of the embedding vectors. Common values are 128, 256, 512, 768, or larger.

**4. Initialization:** Initially, the embedding matrix is filled with random values or pre-trained weights from other models. These values will be adjusted during the model's training process.

**5. Lookup Operation:**  During training or inference, when the model encounters a token, the embedding layer performs a lookup operation. It uses the token's ID to retrieve the corresponding row from the embedding matrix. This row becomes the embedding vector for that token.

**6. Embedding Vectors:**  The embedding vector is a dense numerical representation of the token in a continuous vector space. Each dimension in the embedding vector captures some aspect of the token's meaning or its relationship to other tokens.

**Example:**

Consider a vocabulary of 5 tokens: ["This", "is", "an", "example", "."] and an embedding dimension of 3.  The embedding matrix might look like this (with simplified random values):

| Token     | Dim 1 | Dim 2 | Dim 3 |
|-----------|-------|-------|-------|
| This      | 0.2   | -0.5  | 0.8   |
| is        | 0.9   | 0.1   | -0.3  |
| an        | -0.1  | 0.6   | 0.4   |
| example   | 0.5   | -0.2  | 0.7   |
| .         | -0.7  | 0.3   | -0.1  |

If the model encounters the token "example", the embedding layer retrieves the fourth row of the matrix: [0.5, -0.2, 0.7], which becomes the embedding vector for "example."

**Key Points:**

*   **Contextual Embeddings:** LLMs typically create **contextualized output embeddings**, meaning the embedding for a token can change depending on its surrounding words in a sentence.
*   **Training and Optimization:** The embedding matrix is a trainable parameter of the Transformer, and its values are updated during the model's training process to optimize performance on the specific task.
*   **Semantic Similarity:** Embedding vectors with similar meanings or contexts tend to be closer together in the embedding space. This property is useful for tasks like semantic search, where semantically similar documents can be retrieved efficiently.

**In summary, the embedding layer transforms text input into numerical vectors that represent the meaning and context of tokens, providing a foundation for the Transformer to process and understand language.** 



*   **Positional Encoding:**  Since the Transformer processes tokens in parallel, it lacks inherent information about the order of words in the sentence. Positional encodings are added to the embeddings to provide this sequential information. 

**RoPE** stands for **Rotary Positional Embeddings**. It is a method for encoding positional information in a Transformer model that captures both **absolute and relative token positions**.  

Here are the key aspects of RoPE:

* **Functionality:** RoPE enables the model to understand the order of tokens in a sequence, a crucial aspect of language understanding.
* **Implementation:** Instead of using static, absolute embeddings added at the beginning of the forward pass, RoPE encodes positional information by rotating vectors in the embedding space.
    * The positional information is applied during the attention step, specifically to the queries and keys matrices, just before they are multiplied for relevance scoring. 
* **Benefits:**  RoPE captures both the absolute position of a token and its relative position to other tokens, providing a more comprehensive representation of word order than traditional positional embeddings.

RoPE is used in recent Transformer architectures like **Llama 2 and 3**. 


**2. Decoder Blocks:**

*   A GPT model consists of multiple stacked decoder blocks.  The number of blocks can vary depending on the model size. For instance, the 124-million parameter GPT-2 model has 12 decoder blocks.
*   **Masked Multi-Head Attention:** This layer is the core of the Transformer's ability to understand relationships between words in a sentence. It allows each token to attend to all preceding tokens in the sequence, calculating attention weights based on their relevance to the current token. The "masked" aspect ensures that the model can only attend to past tokens, preventing it from "looking into the future" during text generation.
    *   Within the multi-head attention, the input embeddings are projected into three separate matrices: **Queries (Q), Keys (K), and Values (V)**. These matrices are used to compute attention scores, indicating the importance of each token relative to others. 
    *   Multiple "heads" of attention are computed in parallel, allowing the model to capture different aspects of relationships within the sentence.
    *   The outputs from all attention heads are then concatenated and projected back into the original embedding dimension.
*   **Layer Normalization:** This layer normalizes the outputs from the attention layer to stabilize training and improve model convergence.
*   **Feedforward Network:** A fully connected feedforward neural network is applied to each token's representation, further transforming the embeddings to capture more complex relationships. It typically consists of two linear layers with a non-linear activation function like GELU (Gaussian Error Linear Unit) in between.
*   **Shortcut Connections (Residual Connections):**  These connections add the original input to the output of each layer (attention and feedforward), helping with gradient flow during training and allowing the model to learn deeper representations.

**3. Output Layer:**

*   **Final Layer Normalization:**  The outputs from the last decoder block are normalized again.
*   **Linear Output Layer:** This layer projects the final embeddings into the vocabulary space of the tokenizer. The output is a probability distribution over all tokens in the vocabulary, representing the model's prediction for the next token in the sequence. 

**4. Text Generation:**

*   **Decoding Strategy:**  The model selects the next token based on the predicted probability distribution. Common decoding strategies include:
    *   **Greedy Decoding:** Selecting the most probable token at each step.
    *   **Beam Search:** Exploring multiple probable sequences to find the most likely overall sequence.
    *   **Sampling Techniques:** Introducing randomness to generate more diverse outputs, controlled by parameters like temperature and top-k/top-p sampling.
*   **Iterative Process:** The generated token is added to the input sequence, and the process repeats until a specific end-of-sequence token is generated or a maximum length is reached. 

Modern LLMs build upon this basic Transformer architecture, often incorporating advanced techniques like:

*   **Mixture of Experts (MoE):** Replaces the feedforward layers in the Transformer with multiple expert networks, improving efficiency and scaling. Libraries like **DeepSpeed** offer specialized support for MoE fine-tuning.
*   **Parameter-Efficient Fine-tuning (PEFT):** Optimizes fine-tuning for specific tasks by updating only a small subset of model parameters, as opposed to training the entire model. Techniques like **LoRA (Low-Rank Adaptation)**, adapters, and prompt tuning are commonly used.
*   **Reinforcement Learning from Human Feedback (RLHF):** Aligns LLM outputs with human preferences by incorporating feedback mechanisms during training.

Understanding these layers and techniques provides a solid foundation for comprehending the inner workings of modern LLMs.
