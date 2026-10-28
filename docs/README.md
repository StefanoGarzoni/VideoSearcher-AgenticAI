# VideoSearcher documentation

**Start here:** [README](../README.md) · [Project scope](PROJECT_SCOPE.md) · [Unavailable private components](PRIVATE_COMPONENTS.md)

## For readers

| Document | Why read it |
|---|---|
| [Architecture](ARCHITECTURE.md) | Services, agents and data flow |
| [Orchestrator design](ORCHESTRATOR_DESIGN.md) | Routing/handoff chains and examples |
| [Vision-language components](VLM_COMPONENTS.md) | VQA, segmentation and model integration |
| [Design decisions](DESIGN_DECISIONS.md) | Engineering tradeoffs |
| [Extension tools](EXTENSION_TOOLS.md) | Web search, summarization, scene detection |
| [MCP transport](STREAMABLE_HTTP_ARCHITECTURE.md) | Streamable HTTP details |

## For research and evaluation

| Document | Why read it |
|---|---|
| [Results](RESULTS.md) | Baselines, confirmation runs, limits of interpretation |
| [Model comparison](MODEL_COMPARISON.md) | Metric definitions and runner setup |
| [Hyperparameter tuning](HYPERPARAMETER_TUNING.md) | Bayesian search and confirmation methodology |
| [CodeCarbon](CODECARBON.md) | Energy measurement design |
| [Test suite](../tests/README.md) | Scenarios and metric checks |

## For developers and reviewers

| Document | Why read it |
|---|---|
| [Setup](SETUP.md) | What can be tested locally |
| [Reproducibility](REPRODUCIBILITY.md) | Evidence and environmental limitations |
| [Private components](PRIVATE_COMPONENTS.md) | What is missing and why |
| [Project scope](PROJECT_SCOPE.md) | Shared authorship, source boundaries, licensing |
| [Repository preparation notes](REPOSITORY_NOTES.md) | Changes compared with the supplied export |
| [Historical technical guide](TECHNICAL_GUIDE.md) | Full original instructions, including historical inaccessible dependencies |
| [Model registry](../local-ai-service/ADDING_MODELS.md) | Local GGUF configuration |

## Presentations

- [System architecture and VLM interactions (PDF)](presentations/VideoSearcher_VLM_Interaction.pdf)
- [Benchmark and Bayesian tuning (PDF)](presentations/VideoSearcher_Benchmark_Defense.pdf)
- [Benchmark editable slides (PPTX)](presentations/VideoSearcher_Benchmark_Defense.pptx)

## Public distribution and attribution

- [Public-release checklist](PUBLICATION_CHECKLIST.md) — what was excluded and which rights must still be verified
- [Third-party notices](THIRD_PARTY_NOTICES.md) — acknowledgments for vendored chat templates and other external components
