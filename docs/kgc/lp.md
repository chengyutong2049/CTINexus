---
title: Link Prediction (LP)
parent: Knowledge Graph Construction (KGC)
nav_order: 3
---

# Link Prediction (LP)

The LP module processes CTI reports to predict and establish links between entities within extracted triplets. It leverages LLMs to infer relationships, thereby improving the quality and utility of the extracted knowledge.
{: .fs-5 .fw-300 }

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
import os 
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from Linker import Linker

@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(config: DictConfig):
    annotated_CTI_Sources = os.listdir(config.outSet)
    for CTI_Source in os.listdir(config.inSet):
        if CTI_Source in annotated_CTI_Sources:
            continue
        FolderPath = os.path.join(config.inSet, CTI_Source)
        for JSONFile in os.listdir(FolderPath):
            Linker(config, CTI_Source, JSONFile)

if __name__ == "__main__":
    run()
``` 

### Linker (`Linker.py`)
The `Linker` class handles the complete link prediction workflow:
1. Input Processing:
```python
def __init__(self, config: DictConfig, CTI_Source, inFile):
    self.config = config
    self.CTI_Source = CTI_Source
    self.inFile = inFile

    infile_path = os.path.join(self.config.inSet, self.CTI_Source, self.inFile)
    with open(infile_path, 'r') as f:
        self.js = json.load(f)
        self.aligned_triplets = self.js["EA"]["aligned_triplets"]

    self.graph = {}
    self.build_graph()
    self.subgraphs = self.find_disconnected_subgraphs()
    self.main_nodes = self.find_main_nodes()
    self.topic_node = self.get_topic_node(self.subgraphs)
    self.main_nodes = [node for node in self.main_nodes if node["entity_id"] != self.topic_node["entity_id"]]
    self.js["LP"] = LLMLinker(self).link()
    self.js["LP"]["topic_node"] = self.topic_node
    self.js["LP"]["main_nodes"] = self.main_nodes
    self.js["LP"]["subgraphs"] = [list(subgraph) for subgraph in self.subgraphs]
    self.js["LP"]["subgraph_num"] = len(self.subgraphs)
    
    outfolder = os.path.join(self.config.outSet, self.CTI_Source)
    os.makedirs(outfolder, exist_ok=True)
    outfile_path = os.path.join(outfolder, self.inFile)

    with open(outfile_path, 'w') as f:
        json.dump(self.js, f, indent=4)
```

2. Graph Construction:
```python
def build_graph(self):
    for triplet in self.aligned_triplets:
        subject_entity_id = triplet["subject"]["entity_id"]
        object_entity_id = triplet["object"]["entity_id"]
        
        if subject_entity_id not in self.graph:
            self.graph[subject_entity_id] = []
        if object_entity_id not in self.graph:
            self.graph[object_entity_id] = []
        
        self.graph[subject_entity_id].append(object_entity_id)
        self.graph[object_entity_id].append(subject_entity_id)
```

3. Subgraph Identification:
```python
    def find_disconnected_subgraphs(self):
        self.visited = set()
        subgraphs = []

        for start_node in self.graph.keys():
            if start_node not in self.visited:
                current_subgraph = set()
                self.dfs_collect(start_node, current_subgraph)
                subgraphs.append(current_subgraph)

        return subgraphs

    def dfs_collect(self, node, current_subgraph):
        if node in self.visited:
            return
        self.visited.add(node)
        current_subgraph.add(node)
        for neighbour in self.graph[node]:
            self.dfs_collect(neighbour, current_subgraph)
```

4. Main Node Identification:
```python
    def find_main_nodes(self):
        main_nodes = []
        for subgraph in self.subgraphs:
            main_node_entity_id = self.get_main_node(subgraph)
            main_node = self.get_node(main_node_entity_id)
            main_nodes.append(main_node)
        return main_nodes

    def get_main_node(self, subgraph):
        outdegrees = defaultdict(int)
        for triplet in self.aligned_triplets:
            subject_entity_id = triplet["subject"]["entity_id"]
            object_entity_id = triplet["object"]["entity_id"]
            outdegrees[subject_entity_id] += 1
            outdegrees[object_entity_id] += 1
        max_outdegree = 0
        main_node = None
        for node in subgraph:
            if outdegrees[node] > max_outdegree:
                max_outdegree = outdegrees[node]
                main_node = node
        return main_node

    def get_node(self, entity_id):
        for triplet in self.aligned_triplets:
            for key, node in triplet.items():
                if key in ["subject", "object"]:
                    if node["entity_id"] == entity_id:
                        return node
```

5. Topic Node Identification:
```python
    def get_topic_node(self, subgraphs):
        max_node_num = 0
        for subgraph in subgraphs:
            if len(subgraph) > max_node_num:
                max_node_num = len(subgraph)
                main_subgraph = subgraph
        return self.get_node(self.get_main_node(main_subgraph))
```

### LLM-Based Linking (`LLMLinker.py`)
Handles the linking of entities using LLMs:
1. Link Prediction:
```python
    def link(self):
        for main_node in self.main_nodes:
            prompt = self.generate_prompt(main_node)
            llmCaller = LLMCaller(self.config, prompt)
            self.llm_response, self.response_time = llmCaller.call()
            self.usage = UsageCalculator(self.llm_response).calculate()
            self.response_content = json.loads(self.llm_response.choices[0].message.content)
            pred_sub, pred_rel, pred_obj = self.extract_predicted_triple(self.response_content, main_node)
            self.predicted_triple = self.construct_predicted_triple(pred_sub, pred_rel, pred_obj, main_node)
            self.predicted_triples.append(self.predicted_triple)
            self.response_times.append(self.response_time)
            self.usages.append(self.usage)

        return self.construct_lp_output()

    def extract_predicted_triple(self, response_content, main_node):
        try:
            pred_sub = response_content["predicted_triple"]['subject']
            pred_obj = response_content["predicted_triple"]['object']
            pred_rel = response_content["predicted_triple"]['relation']
        except:
            values = list(response_content.values())
            pred_sub, pred_rel, pred_obj = values[0], values[1], values[2]
        return pred_sub, pred_rel, pred_obj

    def construct_predicted_triple(self, pred_sub, pred_rel, pred_obj, main_node):
        if pred_sub == main_node["entity_text"] and pred_obj == self.topic_node["entity_text"]:
            new_sub = {"entity_id": main_node["entity_id"], "mention_text": main_node["entity_text"]}
            new_obj = self.topic_node
        elif pred_obj == main_node["entity_text"] and pred_sub == self.topic_node["entity_text"]:
            new_sub = self.topic_node
            new_obj = {"entity_id": main_node["entity_id"], "mention_text": main_node["entity_text"]}
        else:
            new_sub = {"entity_id": "hallucination", "mention_text": "hallucination"}
            new_obj = {"entity_id": "hallucination", "mention_text": "hallucination"}
        return {"subject": new_sub, "relation": pred_rel, "object": new_obj}

    def construct_lp_output(self):
        return {
            "predicted_links": self.predicted_triples,
            "response_time": sum(self.response_times),
            "model": self.config.model,
            "usage": {
                "input": {
                    "tokens": sum([usage["input"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["input"]["cost"] for usage in self.usages])
                },
                "output": {
                    "tokens": sum([usage["output"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["output"]["cost"] for usage in self.usages])
                },
                "total": {
                    "tokens": sum([usage["total"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["total"]["cost"] for usage in self.usages])
                }
            }
        }
```

2. Prompt Generation:
```python
    def generate_prompt(self, main_node):
        env = Environment(loader=FileSystemLoader(self.config.link_prompt_folder))
        parsed_template = env.parse(env.loader.get_source(env, self.config.link_prompt_file)[0])
        template = env.get_template(self.config.link_prompt_file)
        variables = meta.find_undeclared_variables(parsed_template)

        if variables:
            User_prompt = template.render(main_node=main_node["entity_text"], CTI=self.js["CTI"]["text"], topic_node=self.topic_node["entity_text"])
        else:
            User_prompt = template.render()
        prompt = [{"role": "user", "content": User_prompt}]

        subFolderPath = os.path.join(self.config.link_prompt_set, self.CTI_Source)
        os.makedirs(subFolderPath, exist_ok=True)
        with open(os.path.join(subFolderPath, self.inFile.split('.')[0] + ".txt"), 'w') as f:
            f.write(json.dumps(User_prompt, indent=4).replace("\\n", "\n").replace('\\"', '\"'))
        return prompt
```

### LLM API Caller (`LLMCaller.py`)
Handles communication with the OpenAI API:
```python
from openai import OpenAI
import time
from omegaconf import DictConfig

class LLMCaller:
    def __init__(self, config: DictConfig, prompt) -> None:
        self.config = config
        self.prompt = prompt

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
        time.sleep(5)  # Pause to avoid exceeding rate limit
        generation_time = endTime - startTime
        return response, generation_time
```

### Usage Calculator (`UsageCalculator.py`)
Calculates token usage and associated costs:
```python
import json
class UsageCalculator:
    def __init__(self, response) -> None:
        self.response = response
        self.model = response.model

    def calculate(self):
        with open ("Toolbox/menu/menu.json", "r") as f:
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
defaults:  
  - _self_

addition:
  darkreading:
    - Bluetooth-Flaw.json
  securityweek:
    - wallescape.json
  thehackernews:
    - ESG.json
  threatPost:
    - H0lyGh0st.json
  trendmicro:
    - Akira.json

inSet: /home/yutong/CTINexus/dataset/Merger-output-large-GT
outSet: /home/yutong/CTINexus/dataset/Linker-outputs-GT

## openai config
model: gpt-4-0125-preview
api_key: sk-*** # API key (recommend using environment variables)

## prompt constructor
link_prompt_folder: Toolbox/LinkerPrompt
link_prompt_file: GT.jinja

## prompt store
link_prompt_set: /home/yutong/CTINexus/dataset/Linker-outputs-GT/prompt_store/metrics
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
      }
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
      }
    ],
    "response_time": 2.35,
    "model": "gpt-4-0125-preview",
    "usage": {
      "input": {
        "tokens": 1024,
        "cost": 0.01
      },
      "output": {
        "tokens": 512,
        "cost": 0.03
      },
      "total": {
        "tokens": 1536,
        "cost": 0.04
      }
    },
    "topic_node": {
      "entity_id": 0,
      "entity_text": "APT28"
    },
    "main_nodes": [
      {
        "entity_id": 1,
        "entity_text": "Zebrocy"
      }
    ],
    "subgraphs": [
      [0, 1],
      [2, 3]
    ],
    "subgraph_num": 2
  }
}
```

[Hydra]: https://hydra.cc/docs/intro