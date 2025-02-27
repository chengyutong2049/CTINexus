---
title: Entity Merging (EM)
parent: Knowledge Graph Construction (KGC)
nav_order: 2
---

# Entity Merging (EM)
{: .no_toc }

The EM module processes CTI reports to identify and merge similar entities within extracted triplets. It uses embeddings to measure similarity and clusters entities that are semantically similar. This process helps in reducing redundancy and improving the quality of the extracted knowledge.
{: .fs-5 .fw-300 }

![](../../../assets/images/em.png)

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Architecture

```yaml
EM/
├── main.py              # Entry point for the merging pipeline
├── Merger.py            # Core merging processor
├── LLMMerger.py         # Handles LLM-based merging logic
├── preprocess/          # Preprocessing scripts
│   ├── main.py          # Entry point for preprocessing
│   ├── Preprocessor.py  # Preprocessing logic
│   └── config/      # Configuration directory for preprocessing
│       └── config.yaml # Default configuration file
└── config/              # Configuration directory
    └── example.yaml     # Default configuration file      
```

## Technical Components

### Main Pipeline (`main.py`)
The entry point that orchestrates the merging process:
```python
@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(config: DictConfig):
    for CTI_Source in os.listdir(config.inSet):
        annotated_CTI_Source = os.listdir(config.outSet)
        if CTI_Source in annotated_CTI_Source:
            continue
            
        # Process each file in source directory
        FolderPath = os.path.join(config.inSet, CTI_Source)
        for JSONFile in os.listdir(FolderPath):
            Merger(config).merge(CTI_Source, JSONFile)
```

### Entity Merger (`Merger.py`)
The `Merger` class handles the complete entity merging workflow:
1. Input Processing:
```python
def merge(self, CTI_source, inFile):
    self.CTI_source = CTI_source
    self.inFile = inFile
    self.inFile_path = os.path.join(self.config.inSet, self.CTI_source, self.inFile)
    self.build_tag_dict(self.inFile_path)
``` 

2. Tag Dictionary Construction:
```python
def build_tag_dict(self, inFile_path):
    with open(inFile_path, 'r') as f:
        self.js = json.load(f)
    for triple in self.js["EA"]["aligned_triplets"]:
        for key, node in triple.items():
            if key in ["subject", "object"]:
                self.update_tag_dict(node)
```

3. Entity Merging:
```python
for mention_class, node_list in self.tag_dict.items():
    if len(node_list) == 1:
        continue
    LLMMerger(self).merge(node_list)
```

4. Result Storage:
```python
outfolder = os.path.join(self.config.outSet, self.CTI_source)
os.makedirs(outfolder, exist_ok=True)
outfile_path = os.path.join(outfolder, self.inFile)
with open(outfile_path, 'w') as f:
    json.dump(self.js, f, indent=4)
```

### LLM-Based Merging (`LLMMerger.py`)
Handles the merging of entities using embeddings:

1. Embedding Generation:
```python
def get_embeddings(self, node_list):
    openai.api_key = self.config.api_key
    node_embedding_list = []
    for (mention_id, mention_text) in node_list:
        response = openai.Embedding.create(
            input=mention_text,
            engine=self.config.embedding_model
        )
        node_embedding_list.append((mention_id, response['data'][0]['embedding']))
    return node_embedding_list
```

2. Clustering:
```python
for (mention_id_1, mention_embedding_1), (mention_id_2, mention_embedding_2) in itertools.combinations(node_embedding_list, 2):
    similarity = 1 - cosine(mention_embedding_1, mention_embedding_2)
    if similarity > self.config.similarity_threshold:
        # Cluster logic
```

3. Cluster Assignment:
```python
for _, cluster in enumerate(clusters):
    if len(cluster) > 1:
        for mention_id in cluster:
            _node = self.retrieve_node(mention_id)
            _node["entity_id"] = self.merger.entity_id
            _node["mentions_merged"] = [self.retrieve_mention_text(mention_id) for mention_id in cluster]
            _node["entity_text"] = self.get_freq_mentions(_node["mentions_merged"])
    else:
        for mention_id in cluster:
            _node = self.retrieve_node(mention_id)
            _node["entity_id"] = self.merger.entity_id
            _node["mentions_merged"] = [self.retrieve_mention_text(mention_id)]
            _node["entity_text"] = _node["mentions_merged"][0]
    self.merger.entity_id += 1
```

### Preprocessing (`preprocess/Preprocessor.py`)
Prepares the data for merging by aligning triplets and assigning mention IDs:
```python
def preprocess(self, CTI_source, inFile):
    inFilePath = os.path.join(self.config.inSet, self.CTI_source, self.inFile)
    with open(inFilePath, 'r') as f:
        js = json.load(f)

    jsr = copy.deepcopy(js)
    jsr["EA"] = {}
    jsr["EA"]["aligned_triplets"] = js["ET"]["typed_triplets"]
    ID = 0
    for triple in jsr["EA"]["aligned_triplets"]:
        for key, entity in triple.items():
            if key in ["subject", "object"]:
                entity["mention_id"] = ID
                ID += 1
                entity['mention_text'] = entity.pop('text')
                entity["mention_class"] = entity.pop('class')
                if isinstance(entity["mention_class"], dict):
                    entity["mention_class"] = list(entity["mention_class"].keys())[0]

    jsr["EA"]["mentions_num"] = ID
    
    outfolder = os.path.join(self.config.outSet, self.CTI_source)
    os.makedirs(outfolder, exist_ok=True)
    outfile_path = os.path.join(outfolder, self.inFile)
    with open(outfile_path, 'w') as f:
        json.dump(jsr, f, indent=4)
```

## Configuration
The module uses [Hydra] for configuration management. Key parameters in `example.yaml`:
```yaml
# Input/Output paths
inSet: <*>                  # Input directory with preprocessed triplets
outSet: <*>                 # Output directory

# OpenAI settings
api_key: <*>                # API key (recommend using environment variables)
embedding_model: <*>        # Embedding model to use
similarity_threshold: <*>   # Similarity threshold for clustering
```

## Usage Instructions

1. Configure Settings:
    * Set input directory with preprocessed triplet files
    * Set output directory for merged results
    * Configure OpenAI API key (preferably using environment variables)
    * Select appropriate embedding model and similarity threshold

2. Run Preprocessing:
```bash
cd KGC/EM/preprocess
python main.py
```

3. Run Merging:
```bash
cd KGC/EM
python main.py
``` 

## Input Format
The module expects files containing preprocessed triplets with mention IDs:
```json
{
  "EA": {
    "aligned_triplets": [
      {
        "subject": {"mention_id": 0, "mention_text": "APT28", "mention_class": "THREAT_ACTOR"},
        "relation": "uses",
        "object": {"mention_id": 1, "mention_text": "Zebrocy", "mention_class": "MALWARE"}
      },
      {
        "subject": {"mention_id": 2, "mention_text": "Zebrocy", "mention_class": "MALWARE"},
        "relation": "targets",
        "object": {"mention_id": 3, "mention_text": "government entities", "mention_class": "ORGANIZATION"}
      }
    ]
  }
}
```

## Output Format

The module enhances the input files by adding merged entities:

```json
{
  "EA": {
    "aligned_triplets": [
      {
        "subject": {"mention_id": 0, "mention_text": "APT28", "mention_class": "THREAT_ACTOR", "entity_id": 0, "mentions_merged": ["APT28"], "entity_text": "APT28"},
        "relation": "uses",
        "object": {"mention_id": 1, "mention_text": "Zebrocy", "mention_class": "MALWARE", "entity_id": 1, "mentions_merged": ["Zebrocy"], "entity_text": "Zebrocy"}
      },
      {
        "subject": {"mention_id": 2, "mention_text": "Zebrocy", "mention_class": "MALWARE", "entity_id": 1, "mentions_merged": ["Zebrocy"], "entity_text": "Zebrocy"},
        "relation": "targets",
        "object": {"mention_id": 3, "mention_text": "government entities", "mention_class": "ORGANIZATION", "entity_id": 2, "mentions_merged": ["government entities"], "entity_text": "government entities"}
      }
    ],
    "response_time": ...,
    "embedding_model": "...",
    "entity_num": ...
  }
}
```


[Hydra]: https://hydra.cc/docs/intro