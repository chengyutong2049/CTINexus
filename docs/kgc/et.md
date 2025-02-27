---
title: Entity Typing (ET)
parent: Knowledge Graph Construction (KGC)
nav_order: 1
---

# Entity Typing (ET)
{: .no_toc }

The ET module enhances triplets (subject-relation-object structures) extracted by the Information Extraction (IE) module by assigning specific semantic types to entities.
{: .fs-5 .fw-300 }

![](../../../assets/images/et.png)

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Architecture

```yaml
ET/
├── main.py              # Entry point with different execution modes
├── LLMTagger.py         # Core tagging processor
├── LLMCaller.py         # LLM API interface
├── usageCalculator.py   # Tracks token usage and costs
└── config/              # Configuration directory
    └── example.yaml     # Default configuration file
```

## Technical Components
### Main Pipeline (`main.py`)
The entry point that orchestrates the tagging process:
```python   
@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(config: DictConfig):
    for CTI_source in os.listdir(config.inSet):
        # Skip already processed sources
        annotatedCTICource = [dir for dir in os.listdir(config.outSet)]
        if CTI_source in annotatedCTICource:
            continue
            
        # Process each file in source directory
        inFolder = os.path.join(config.inSet, CTI_source)
        for file in os.listdir(inFolder):
            LLMTagger(config).tag(CTI_source, file)
```

### Entity Tagger (`LLMTagger.py`)
The `LLMTagger` class handles the complete entity typing workflow:
1. Input Processing:
```python
def tag(self, CTI_Source, file):
    inFile_path = os.path.join(self.config.inSet, CTI_Source, file)
    with open(inFile_path, 'r') as f:
        js = json.load(f)
        triples = js["IE"]["triplets"]
```

2. Prompt Generation:
```python
self.prompt = self.generate_prompt(triples)
# Store generated prompt for reference
folderPath = os.path.join(self.config.tag_prompt_store, CTI_Source)
os.makedirs(folderPath, exist_ok=True)
self.promptPath = os.path.join(folderPath, file.split('.')[0] + ".txt")
with open(self.promptPath, 'w') as f:
    f.write(json.dumps(self.prompt[0]["content"], indent=4).replace("\\n", "\n").replace('\\"', '\"'))
```

3. LLM API Calling:
```python
self.response, self.response_time = LLMCaller(self.config, self.prompt).call()
self.usage = UsageCalculator(self.response).calculate()
self.response_content = json.loads(self.response.choices[0].message.content)
```

4. Result Storage:
```python
outfolder = os.path.join(self.config.outSet, CTI_Source)
os.makedirs(outfolder, exist_ok=True)
outfile_path = os.path.join(outfolder, file)
with open(outfile_path, 'w') as f:
    js["ET"] = {}
    js["ET"]["typed_triplets"] = self.response_content["tagged_triples"]
    js["ET"]["response_time"] = self.response_time
    js["ET"]["Demo_num"] = self.config.shot
```

### LLM Integration (`LLMCaller.py`)
Handles communication with the OpenAI API:
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
    response_time = endTime - startTime
    return response, response_time
```

### Cost Tracking (`usageCalculator.py`)
Calculates token usage and associated costs:
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
inSet: <*>                # Input triplets directory
outSet: <*>               # Output directory

# Model configuration
model: <*>                # LLM model to use
api_key: <*>              # API key (recommend using environment variables)

# Prompt settings
tag_prompt_folder: <*>    # Directory with templates
tag_prompt_file: <*>      # Template file to use
tag_prompt_store: <*>     # Where to save prompts
shot: <*>                 # Number of examples in prompt
```

## Usage Instructions
1. Configure Settings:
    * Set input directory with triplet files (from IE module)
    * Set output directory for typed results
    * Configure OpenAI API key (preferably using environment variables)
    * Select appropriate template and shot count
2. Run the Module:
```bash
cd KGC/ET
python main.py
```

## Input Format
The module expects files containing extracted triplets from the `IE` module:
```json
{
  "CTI": {
    "text": "Original CTI text...",
    "link": "source_link"
  },
  "IE": {
    "triplets": [
      {"subject": "APT28", "relation": "uses", "object": "Zebrocy"},
      {"subject": "Zebrocy", "relation": "targets", "object": "government entities"},
      {"subject": "APT28", "relation": "associated with", "object": "Russia"}
    ]
  }
}
```

## Output Format
The module enhances the input files by adding an ET section with typed entities:
```json
{
  "CTI": { ... },
  "IE": { ... },
  "ET": {
    "typed_triplets": [
      {
        "subject": {"text": "APT28", "type": "THREAT_ACTOR"},
        "relation": "uses",
        "object": {"text": "Zebrocy", "type": "MALWARE"}
      },
      {
        "subject": {"text": "Zebrocy", "type": "MALWARE"},
        "relation": "targets",
        "object": {"text": "government entities", "type": "ORGANIZATION"}
      },
      {
        "subject": {"text": "APT28", "type": "THREAT_ACTOR"},
        "relation": "associated with",
        "object": {"text": "Russia", "type": "LOCATION"}
      }
    ],
    "response_time": ...,
    "Demo_num": ...
  }
}
```

## Entity Type Ontology
The module by default uses the [MALONT] ontology to assign types to entities, including:
```json
[
    "Account",
    "Action",
    "Attacker",
    "Campaign",
    "Event",
    "Exploit Target Object",
    "Host",
    {
        "Indicator":[
            "Adress",
            "Email",
            "File",
            "Hash"
        ]
    },
    "Information",
    "Location",
    "Malware",
    "Malware Characteristic",
    "Malware Family",
    "Organization",
    "Software",
    "System",
    "Vulnerability",
    "This entity cannot be classified into any of the existing types"
]
``` 


[Hydra]: https://hydra.cc/docs/intro
[MALONT]: https://github.com/aiforsec/MALOnt