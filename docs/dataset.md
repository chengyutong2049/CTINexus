---
title: Dataset
# parent: Experiments
nav_order: 4
---

# Dataset
{: .no_toc }

The CTINEXUS dataset is a comprehensive collection of annotated Cyber Threat Intelligence (CTI) reports designed to evaluate end-to-end knowledge graph construction systems. Unlike existing benchmarks that focus solely on triplet extraction from outdated reports, this dataset encompasses the complete pipeline including cybersecurity triplet extraction, hierarchical entity alignment, and long-distance relation prediction.
{: .fs-5 .fw-300 }


## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
## Dataset Statistics

* 150 CTI reports published from May 2023 onwards
* Sourced from 10 cybersecurity organizations (approximately 15 reports per source)
* Publishers include Trend Micro, Symantec, The Hacker News, and others
* Contains 4,292 mentions, 2,528 entities, and 2,503 relations

## Source Distribution

Reports were collected from reputable cybersecurity sources to ensure diversity and quality of threat intelligence:

| Source    | Number of Reports    |
|:---------------|:-------------------------|
| [Trend Micro] | 15 |
| [Symantec] | 15 |
| [The Hacker News] | 15 |
| [AVERTIUM] | 15 |
| [Bleeping Computer] | 15 |
| [Dark Reading] | 15 |
| [Google TAG] | 15 |
| [Microsoft] | 15 |
| [Security Week] | 15 |
| [Threat Post] | 15 |




## Dataset Format

The dataset follows a JSON structure that captures the progressive enrichment of CTI information:

```json
{
  "CTI": {
    "text": "Original report content...",
    "source": "Publisher name"
  },
  "IE": {
    "triplets": [
      {"subject": "Entity1", "predicate": "relation", "object": "Entity2"},
      ...
    ]
  },
  "EA": {
    "aligned_triplets": [
      {
        "subject": {"mention_id": 0, "mention_text": "Entity1", "entity_id": 1, ...},
        "predicate": "relation",
        "object": {"mention_id": 1, "mention_text": "Entity2", "entity_id": 2, ...}
      },
      ...
    ]
  },
  "LP": {
    "predicted_links": [
      {"subject": {...}, "relation": "implicit_relation", "object": {...}},
      ...
    ]
  }
}
```

## Features
This dataset enables researchers and practitioners to:

* Evaluate end-to-end Cyber Security Knowledge Graph (CSKG) construction systems 🚀
* Benchmark performance on modern threat intelligence (post-May 2023) 🚀
* Test capabilities across multiple KG construction tasks rather than just triplet extraction 🚀
* Develop more robust CTI analysis tools leveraging knowledge graph approaches 🚀





[Trend Micro]: https://www.trendmicro.com/
[Symantec]: https://www.symantec.com/
[The Hacker News]: https://thehackernews.com/
[AVERTIUM]: https://www.avertium.com/
[Bleeping Computer]: https://www.bleepingcomputer.com/
[Dark Reading]: https://www.darkreading.com/
[Google TAG]: https://blog.google/threat-analysis-group/
[Microsoft]: https://www.microsoft.com/
[Security Week]: https://www.securityweek.com/
[Threat Post]: https://threatpost.com/