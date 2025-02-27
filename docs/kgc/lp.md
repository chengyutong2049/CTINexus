---
title: Link Prediction (LP)
parent: Knowledge Graph Construction (KGC)
nav_order: 3
---

# Link Prediction (LP)

The LP module processes CTI reports to predict and establish links between entities within extracted triplets. It leverages LLMs to infer relationships, thereby improving the quality and utility of the extracted knowledge.
{: .fs-5 .fw-300 }

![](../../../assets/images/lp-cap.png)

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Architecture
```yaml
LP/
├── main.py              # Entry point for the link prediction pipeline
├── Linker.py            # Core linking processor
├── LLMLinker.py         # Handles LLM-based linking logic
├── LLMCaller.py         # Interface to LLM API
├── UsageCalculator.py   # Tracks token usage and costs
└── config/              # Configuration directory
    └── example.yaml     # Default configuration file
```

## Technical Components

### Main Pipeline (`main.py`)

The entry point that orchestrates the link prediction process:
```python
@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(config: DictConfig):
    annotated_CTI_Sources = os.listdir(config.outSet)
    for CTI_Source in os.listdir(config.inSet):
        if CTI_Source in annotated_CTI_Sources:
            continue
        FolderPath = os.path.join(config.inSet, CTI_Source)
        for JSONFile in os.listdir(FolderPath):
            Linker(config, CTI_Source, JSONFile)
``` 


### Linker (`Linker.py`)
The `Linker` class handles the complete link prediction workflow:
1. Graph Construction:
Builds an [adjacency list](https://en.wikipedia.org/wiki/Adjacency_list) representation of the knowledge graph. Each entity becomes a node, and each triplet establishes an undirected edge between subject and object entities. This structure enables efficient graph traversal and component analysis.
```python
# Fill the graph structure with entity connections
for triplet in self.aligned_triplets:
    subject_entity_id = triplet["subject"]["entity_id"]
    object_entity_id = triplet["object"]["entity_id"]
    
    if subject_entity_id not in self.graph:
        self.graph[subject_entity_id] = []
    if object_entity_id not in self.graph:
        self.graph[object_entity_id] = []
    
    # Add undirected edges
    self.graph[subject_entity_id].append(object_entity_id)
    self.graph[object_entity_id].append(subject_entity_id)
```

2. Disconnected Subgraph Identification:
Uses [depth-first search (DFS)](https://en.wikipedia.org/wiki/Depth-first_search) to identify disconnected components in the knowledge graph. For each unvisited node, it starts a new DFS traversal to collect all connected nodes into a subgraph. This allows the system to identify isolated knowledge clusters that require connecting.
```python
def find_disconnected_subgraphs(self):
    self.visited = set()
    subgraphs = []

    for start_node in self.graph.keys():
        if start_node not in self.visited:
            # For each new subgraph found, collect its nodes
            current_subgraph = set()
            self.dfs_collect(start_node, current_subgraph)
            subgraphs.append(current_subgraph)

    return subgraphs
```

3. Main Node Identification:
Identifies the central entity within each subgraph by calculating [node degrees](https://en.wikipedia.org/wiki/Degree_(graph_theory)) (number of connections). The entity with the highest degree is considered the main node of the subgraph, representing its most central concept.
```python
def get_main_node(self, subgraph):
    # Count node degrees
    outdegrees = defaultdict(int)
    self.directed_graph = {}
    
    # Build directed graph
    for triplet in self.aligned_triplets:
        subject_entity_id = triplet["subject"]["entity_id"]
        object_entity_id = triplet["object"]["entity_id"]
        if subject_entity_id not in self.directed_graph:
            self.directed_graph[subject_entity_id] = []
        self.directed_graph[subject_entity_id].append(object_entity_id)
        outdegrees[subject_entity_id] += 1
        outdegrees[object_entity_id] += 1
        
    # Find the node with maximum degree
    max_outdegree = 0
    main_node = None
    for node in subgraph:
        if outdegrees[node] > max_outdegree:
            max_outdegree = outdegrees[node]
            main_node = node
    return main_node
```

4. Topic Node Identification:
Identifies the overall topic node of the entire knowledge graph by first determining the largest subgraph (by node count) and then finding its central entity. This approach assumes that the largest connected component contains the report's primary subject matter.
```python
def get_topic_node(self, subgraphs):
    # The subgraph with the most nodes is considered the main subgraph
    max_node_num = 0
    for subgraph in subgraphs:
        if len(subgraph) > max_node_num:
            max_node_num = len(subgraph)
            main_subgraph = subgraph
    # Find the main node of the largest subgraph
    return self.get_node(self.get_main_node(main_subgraph))
```


### LLM-Based Linking (`LLMLinker.py`)
Handles the linking of entities using LLMs:
1. Link Generation Process:
Iterates through each main node from the disconnected subgraphs and generates a prompt asking the LLM to predict a relationship between that node and the identified topic node. It then formats the response into a standardized triple format, handles potential hallucinations, and collects usage statistics.
```python
def link(self):
    for main_node in self.main_nodes:
        prompt = self.generate_prompt(main_node)
        llmCaller = LLMCaller(self.config, prompt)
        self.llm_response, self.response_time = llmCaller.call()
        self.usage = UsageCalculator(self.llm_response).calculate()
        self.response_content = json.loads(self.llm_response.choices[0].message.content)
        
        # Extract the predicted relationship components
        try:
            pred_sub = self.response_content["predicted_triple"]['subject']
            pred_obj = self.response_content["predicted_triple"]['object']
            pred_rel = self.response_content["predicted_triple"]['relation']
        except:
            values = list(self.response_content.values())
            pred_sub, pred_rel, pred_obj = values[0], values[1], values[2]
            
        # Format the predicted relationship properly
        if pred_sub == main_node["entity_text"] and pred_obj == self.topic_node["entity_text"]:
            new_sub = {"entity_id": main_node["entity_id"], "mention_text": main_node["entity_text"]}
            new_obj = self.topic_node
        elif pred_obj == main_node["entity_text"] and pred_sub == self.topic_node["entity_text"]:
            new_sub = self.topic_node
            new_obj = {"entity_id": main_node["entity_id"], "mention_text": main_node["entity_text"]}
        else:
            # Handle hallucination cases
            new_sub = {"entity_id": "hallucination", "mention_text": "hallucination"}
            new_obj = {"entity_id": "hallucination", "mention_text": "hallucination"}

        self.predicted_triple = {"subject": new_sub, "relation": pred_rel, "object": new_obj}
        self.predicted_triples.append(self.predicted_triple)
        self.response_times.append(self.response_time)
        self.usages.append(self.usage)

    return self.construct_lp_output()
```

2. Prompt Generation:
Generates a customized prompt for the LLM using [Jinja2] templates. It passes the main node, topic node, and original CTI text as context to help the LLM predict meaningful relationships. The prompt is also stored for future reference and debugging.
```python
def generate_prompt(self, main_node):
    env = Environment(loader=FileSystemLoader(self.config.link_prompt_folder))
    parsed_template = env.parse(env.loader.get_source(env, self.config.link_prompt_file)[0])
    template = env.get_template(self.config.link_prompt_file)
    variables = meta.find_undeclared_variables(parsed_template)

    if variables is not {}: # if template has variables
        User_prompt = template.render(main_node=main_node["entity_text"], 
                                     CTI=self.js["CTI"]["text"], 
                                     topic_node=self.topic_node["entity_text"])
    else:
        User_prompt = template.render()
        
    prompt = [{"role": "user", "content": User_prompt}]

    # Store the prompt for reference
    subFolderPath = os.path.join(self.config.link_prompt_set, self.CTI_Source)
    os.makedirs(subFolderPath, exist_ok=True)
    with open(os.path.join(subFolderPath, self.inFile.split('.')[0] + ".txt"), 'w') as f:
        f.write(json.dumps(User_prompt, indent=4).replace("\\n", "\n").replace('\\"', '\"'))
    return prompt
```




### LLM API Communication  (`LLMCaller.py`)
Handles the communication with OpenAI's API, ensuring responses are formatted as JSON objects and tracking the time needed for generation. It includes a rate-limiting mechanism to prevent API throttling.

```python
def call(self):
    client = OpenAI(api_key=self.config.api_key)
    startTime = time.time()
    response = client.chat.completions.create(
        model = self.config.model,
        response_format = { "type": "json_object" },
        messages = self.prompt,
        max_tokens= 4096,
    )
    endTime = time.time()
    #pause for 5 seconds to avoid exceeding the rate limit
    time.sleep(5)
    generation_time = endTime - startTime
    return response, generation_time
```

### Usage Tracking (`UsageCalculator.py`)
Calculates and tracks token usage and associated costs by reading pricing data from a configuration file and applying it to the actual token counts from the API response.


```python
def calculate(self):
    with open (model_price_menu, "r") as f:
        data = json.load(f)
    iprice = data[self.model]["input"]
    oprice = data[self.model]["output"]
    
    usageDict = {}
    usageDict["model"] = self.model
    usageDict["input"] = {
        "tokens": self.response.usage.prompt_tokens,
        "cost": iprice*self.response.usage.prompt_tokens
    }
    usageDict["output"] = {
        "tokens": self.response.usage.completion_tokens,
        "cost": oprice*self.response.usage.completion_tokens
    }
    usageDict["total"] = {
        "tokens": self.response.usage.prompt_tokens+self.response.usage.completion_tokens,
        "cost": iprice*self.response.usage.prompt_tokens+oprice*self.response.usage.completion_tokens
    }
    return usageDict
```

## Configuration
The module uses [Hydra] for configuration management. Key parameters in `example.yaml`:
```yaml
# Input/Output paths
inSet: <*>               # Input directory for processed entity files
outSet: <*>              # Output directory for link prediction results

# OpenAI settings
model: <*>               # LLM model to use
api_key: <*>             # API key (recommend using environment variables)

# Prompt settings
link_prompt_folder: <*>  # Directory with templates
link_prompt_file: <*>    # Template file to use
link_prompt_set: <*>     # Where to save prompts
```

## Usage Instructions

Configure Settings:
1. Set input directory with preprocessed triplet files
* Set output directory for linked results
* Configure OpenAI API key (preferably using environment variables)
* Select appropriate prompt template
2. Run Link Prediction:
```bash
cd KGC/LP
python main.py
```
## Input Format
The module expects files containing preprocessed triplets with entity IDs:
```json
{
  "EA": {
    "aligned_triplets": [
      {
        "subject": {"entity_id": 0, "mention_text": "APT28", "mention_class": "THREAT_ACTOR"},
        "predicate": "uses",
        "object": {"entity_id": 1, "mention_text": "Zebrocy", "mention_class": "MALWARE"}
      },
      {
        "subject": {"entity_id": 2, "mention_text": "Zebrocy", "mention_class": "MALWARE"},
        "predicate": "targets",
        "object": {"entity_id": 3, "mention_text": "government entities", "mention_class": "ORGANIZATION"}
      },
      ...
    ]
  }
}
```

## Output Format
The module enhances the input files by adding predicted links:
```json
{
  "EA": { ... },
  "LP": {
    "predicted_links": [
      {
        "subject": {"entity_id": 0, "mention_text": "APT28"},
        "relation": "uses",
        "object": {"entity_id": 1, "mention_text": "Zebrocy"}
      },
      {
        "subject": {"entity_id": 2, "mention_text": "Zebrocy"},
        "relation": "targets",
        "object": {"entity_id": 3, "mention_text": "government entities"}
      },
      ...
    ],
    "response_time": ...,
    "usage": {
      "model": "...",
      "input": {"tokens": ..., "cost": ...},
      "output": {"tokens": ..., "cost": ...},
      "total": {"tokens": ..., "cost": ...}
    },
    "model": "...",
    "topic_node": {
      "entity_id": ...,
      "entity_text": "..."
    },
    "main_nodes": [
      {
        "entity_id": ...,
        "entity_text": "..."
      }
    ],
    "subgraphs": [
      [0, 1, ...],
      [2, 3, ...],
      ...
    ],
    "subgraph_num": ...
  }
}
```

[Hydra]: https://hydra.cc/docs/intro
[Jinja2]: https://jinja.palletsprojects.com/en/3.0.x/
