---
title: Information Extraction (IE)
nav_order: 2
---

# Information Extraction (IE)
{: .no_toc }

The IE module automatically processes unstructured CTI reports and extracts structured information in the form of triplets (subject-predicate-object). It uses demonstration-based learning to improve extraction accuracy and incorporates multiple LLM backends.
{: .fs-5 .fw-300 }

![](../../assets/images/ie-cap.png)

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}
---
## Architecture

```yaml
IE/
├── main.py                  # Pipeline entry point
├── LLMAnnotator.py          # Core annotation processor
├── promptConstructor.py     # Builds prompts using templates
├── demoRetriever.py         # Retrieves relevant demonstration examples
├── LLMcaller.py             # Interfaces with different LLMs
├── responseParser.py        # Parses and structures LLM responses
├── usageCalculator.py       # Calculates API usage and costs
├── instructionLoader.py     # Loads instruction templates
└── config/                  # Configuration directory
    └── example.yaml         # Default configuration file
```

## Technical Components

### Main Pipeline (`main.py`)

The main pipeline orchestrates the entire extraction process using Hydra for configuration management:
```python
@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(config: DictConfig):
    for CTI_Source in os.listdir(config.inSet):
        annotatedCTICource = [dir for dir in os.listdir(config.outSet)]
        if CTI_Source in annotatedCTICource:
            continue
            
        # Process files in each source directory
        FolderPath = os.path.join(config.inSet, CTI_Source)
        for JSONFile in os.listdir(FolderPath):
            LLMAnnotator(config, CTI_Source, JSONFile).annotate()
```
### LLM Annotation Process (`LLMAnnotator.py`)

The `LLMAnnotator` class coordinates the complete annotation workflow:

1.  Loads the input CTI report
2.  Retrieves relevant demonstrations (if configured)
3.  Constructs prompts using templates
4.  Calls the LLM API
5.  Parses responses and structures the output
6.  Saves results with metadata

### Demonstration Retrieval (`demoRetriever.py`)
Supports multiple strategies for selecting demonstration examples:
* **kNN**: Finds semantically similar examples using TF-IDF vectorization and distance metrics
* **Random**: Provides random examples from the demonstration set
* **Fixed** examples can also be specified

### Prompt Construction (`promptConstructor.py`)
Uses Jinja2 templating to build prompts with a flexible structure:

```python
def generate_prompt(self):
    env = Environment(loader=FileSystemLoader(self.config.ie_prompt_set))
    DymTemplate = self.templ
    template_source = env.loader.get_source(env, DymTemplate)[0]
    parsed_content = env.parse(template_source)
    variables = meta.find_undeclared_variables(parsed_content)
    
    # Load and render template with appropriate variables
    template = env.get_template(DymTemplate)
    # ...
```
### LLM Integration (`LLMcaller.py`)
Supports multiple language model backends:

* OpenAI Models: GPT-4 and variants
* Llama: Via local Ollama API or Hugging Face
* Qwen: Via local Ollama API

### Response Processing (`responseParser.py`)
Parses LLM responses and structures the extracted information:
```python
def parse(self):
    self.output = {
        "CTI": self.query,
        "annotator": self.JSONResp if get_char_before_hyphen(self.config.model) == "gpt" else {"triplets": self.JSONResp, "triples_count": len(self.JSONResp)},
        "link": self.link,
        "usage": UsageCalculator(self.llm_response).calculate() if get_char_before_hyphen(self.config.model) == "gpt" else None,
        "prompt": self.prompt,
    }
    
    # Calculate triplet counts and additional metadata
    # ...
```

### Usage Statistics (`usageCalculator.py`)
Calculates and tracks token usage and costs for API-based models:
```python
def calculate(self):
    with open ("Toolbox/menu/menu.json", "r") as f:
        data = json.load(f)
    iprice = data[self.model]["input"]
    oprice = data[self.model]["output"]
    
    # Calculate input, output, and total costs
    # ...
```

## Configuration
The module uses [Hydra] for configuration management. Key parameters in example.yaml:
```yaml
config_name: IE-GT                      # Configuration profile name
inSet: dataset/190CTI+12Source          # Input directory for CTI sources
outSet: dataset/IE-GT                   # Output directory for results
model: gpt-4o-2024-08-06                # LLM model identifier
retriever:
  type: "kNN"                           # Demonstration retrieval method
  permutation: asc                      # Order of retrieved examples
shot: 3                                 # Number of demonstrations to include
demo_set: dataset/demoSet               # Directory with demonstration examples
ie_prompt_set: Toolbox/IE_Prompts       # Directory with prompt templates
templ: QD4.jinja                        # Template file to use
ie_prompt_store: Toolbox/PromptStore/IE-GT  # Storage for used prompts
```

## Usage Instructions

1. Setup Configuration:
    * Modify `example.yaml` to specify input/output paths
    * Set your LLM model and API key (or use environment variables)
    * Configure demonstration parameters (shot count, retriever type)
    * Select appropriate prompt template
2. Run the Pipeline:
```bash
cd IE
python main.py
```

## Output Structure
For each processed CTI report, the module generates:

1. Structured JSON output:
```json
{
  "CTI": "Original CTI text...",
  "IE": {
    "triplets": [
      {"subject": "...", "predicate": "...", "object": "..."},
      ...
    ],
    "triples_count": 5,
    "cost": {
      "model": "gpt-4o-2024-08-06",
      "input": {"tokens": 1024, "cost": 0.01},
      "output": {"tokens": 512, "cost": 0.03},
      "total": {"tokens": 1536, "cost": 0.04}
    },
    "time": 2.45,
    "Prompt": {
      "constructed_prompt": "path/to/prompt",
      "prompt_template": "QD4.jinja",
      "demo_retriever": "kNN",
      "demos": ["example1.json", "example2.json", "example3.json"],
      "demo_number": 3,
      "permutation": "asc"
    }
  }
}
```
2. Prompt Archives:
    * Each prompt is saved for reproducibility and debugging

## Key Features
* Multi-model Support: Works with OpenAI GPT models, Llama, and Qwen 🚀
* Few-Shot Learning: Uses relevant examples to guide extraction 🚀
* Flexible Templating: Uses Jinja2 for adaptive prompt construction 🚀
* Smart Demo Retrieval: kNN-based selection of similar examples 🚀
* Usage Tracking: Calculates and tracks token usage and API costs 🚀
* Reproducibility: Saves all prompts and configuration details 🚀

## Extension Points
* Custom Templates: Add new templates to `IE_Prompts`
* New Models: Extend `LLMcaller.py` to support additional LLMs
* Retrieval Methods: Implement alternatives to kNN in `demoRetriever.py`
* Output Formats: Modify `responseParser.py` for different output structures

## Dependencies
* Python 3.8+
* Hydra
* Jinja2
* OpenAI API (for GPT models)
* Ollama (for local Llama/Qwen deployment)
* scikit-learn (for kNN retrieval)
* NLTK (for text processing)

[Hydra]: https://hydra.cc/docs/intro