# [Step-by-step Code for Knowledge Graph Construction](https://youtu.be/OsnM8YTFwk4)

from gliner import GLiNER

def merge_entities(entities):
    '''
    to get entities for graph
    '''
    if not entities:
        return []
    merged = []
    current = entities[0] 
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
    else:
        merged.append(current)
        current = next_entity
        # Append the last entity
    merged.append(current)
    return merged

model = GLiNER.from_pretrained("numind/NuZero_token")

# intersted entities
labels = ["organization", "medicine", "side effects", "product", "chemical composition"] # =====> this is to be modified as epr use case
labels = [l.lower() for l in labels]

text = ""  # =====> this is to be modified as epr use case, will contain the entire text to extarct entites dform

entities = model.predict_entities(text, labels)

entities = merge_entities(entities)

for entity in entities:
    print(entity["text"], "=>", entity["label"])

# if the text is too long then it has to be chucked depending on the sieze of the gliner window like in RAG

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter( chunk_size=200,chunk_overlap=20, separators=("\n\n", "\n"))
chunks = text_splitter.split_text(text)

# 

def split_text_with_overlap(text, num_sentences=3, overlap=1):
    """
    Splits the input text into multiple chunks, where each chunk contains
    a specified number of sentences with an overlap of a specified number
    of sentences.

    Args:
    - text (str): The input text to be split.
    - num_sentences (int): The number of sentences to include in each chunk.
      Default is 3.
    - overlap (int): The number of sentences to overlap between adjacent
      chunks. Default is 1.

    Returns:
    - A list of text chunks, where each chunk contains a specified number
      of sentences with an overlap of a specified number of sentences.
    """

    # Tokenize the text into sentences using NLTK
    sentences = nltk.sent_tokenize(text)

    # Initialize the list of text chunks
    chunks = []

    # Split the sentences into chunks with the specified overlap
    start_idx = 0
    while start_idx < len(sentences):
        end_idx = min(start_idx + num_sentences, len(sentences))
        chunk = ' '.join(sentences[start_idx:end_idx])
        chunks.append(chunk)
        start_idx += num_sentences - overlap

    return chunks


# another way
lables = [] #  =====>  this is to be modified as epr use case

chunks_entities = []
entity_list = []
duplicates = set()

for text in tqdm(chunks):
    entities = model.predict_entities(text, labels, threshold=0.7)
    entities = merge_entities(entities)
    chunk_entities = set()
    for entity in entities: 
        # print(entity["text"], "=>", entity["label"])
        chunk_entities.add(entity ("text") )
        if entity["text"] in duplicates:
            continue
        duplicates.add(entity ("text") )
        entity_list.append((entity["text"], "=>", entity["label"]))
    chunks_entities.append(lList(chunk_entities) )

# creating speerate list, AS THE PER THE LABELS ABOVE

persons = [] # =====>  this is to be modified as epr use case
orgs = []
locs = []
awards = []
movies = []

for e in entity_list:
    s,p,o = e
    if o == 'person':
        persons.append(s. lower())
    elif o == 'organization':
        orgs.append(s. lower())
    elif o == 'location':
        locs.append(s. lower())
    elif o == 'award':
        awards.append(s. lLower())
    elif o == 'movie':
        movies.append(s. lower())

# To get relation between entites using FEW SHOT

system_message = """Extract all the relationships between the following entities ONLY baseD ON
 Return a list of JSON objects.
For example:
    <Examples>
    [{{"subject": "John", "relationship": "lives in", "object": "US"}},
    {{"subject": "Eifel towel", "relationship": "is located in", "object": "Paris"}},
    {{"subject": "Hayao Miyazaki", "relationship": "is", "object": "Japanese animator"}}]
</Examples>
- ONLY return triples and nothing else. None of 'subject', 'relationship' and 'object' can be empty.
Entities: \n\nf{entities}
"""

i=3
ents = format_entities(chunks_entities [i] )
text = chunks[i]
user_message = "Context: {text}\n\nTriples:"
response = completion(
    model="gpt-3.5-turbo",
    messages=[{"content": system_message.format(entities=ents),"role": "system"},
    {"content": user_message, "role" : ""}], max_tokens=1000, format = "json")

triples = json. loads(response.choices[0].message. content)
triples

# do it for all the chunkcs

import time
errors = []
all_triples = []
for i in tqdm(range(len(chunks_entities) )):
    try:
        ents = format_entities(chunks_entities [i] )
        text = chunks[i]
        user_message = "Context: {text}\n\nTriples:"
        response = completion(model="gpt-3.5-turbo",
        messages=[{"content": system_message.format(entities=ents),"role": "system"},
        {"content": user_message, "role" : ""}], max_tokens=1000, format = "json")

        triples = json.loads(response.choices[0].message. content)
        all_triples.append(triples)
        time.sleep(3)
    except Exception as e:
        print(f"Error for chunk {i}, {e}")
        errors.append(response.choices[0].message.content)
        all_triples.append([])

# combine this json and entity pos
from pyvis.network import Network
import networkx as nx
G = nx.Graph()
for items in all_triples:
    for item in items:
        try:
            node_1 = item["subject"]
            node_2 = item["object"]
            G.add_node(node_1, title=node_1, color=get_color(node_1), size=get_size(node_1), label=node_1)
            G.add_node(node_2, title=node_2, color=get_color(node_2), size=get_size(node_2), label=node_2)
            G.edge(node_1, node_2, title=item["relationship"], weight=4)
        except Exception as e:
            print(f"Error in item: {item}")

# show grpah

nt = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
nt.from_nx(G)
# nt.toggle_physics (True)
nt. force_atlas_2based(central_gravity=0.015, gravity=-31)
nt.show("graph.html", notebook=False)
# Generate the HTML
# html = nt.generate_html()
# # Write the HTML to a file # with open("graph.html", "w") as file: 4 file.write(html)
# # Display the graph in a Jupyter Notebook
from IPython.display import IFrame
IFrame("graph.html", width=1000, height=800)



# 