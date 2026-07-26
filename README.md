# CS336: Language Modeling from Scratch

Selected implementations and experiments based on Stanford CS336 course, focused on understanding LLM development from model construction through data quality and evaluation.

## Highlights

- Built core language-model components from first principles, including BPE tokenization, Transformer layers, optimization, training, and evaluation.
- Achieved **3.56 validation loss within a one-hour H100 compute budget**.
- Implemented filtering and deduplication methods for preparing higher-quality pretraining data.
- Established a reproducible zero-shot baseline for evaluating mathematical reasoning.

## Projects

| Project | Focus | Result |
|---|---|---|
| [Assignment 1](./assignment1/) | From-scratch language-model implementation and training | **3.56 validation loss in 60 minutes on one H100** |
| [Assignment 4](./assignment4/) | Web-data filtering and deduplication | Completed and validated the deduplication pipeline |
| [Assignment 5](./assignment5/) | Zero-shot mathematical-reasoning evaluation | Baseline evaluation completed; post-training was not pursued |

I selected the parts of CS336 most relevant to my interests in model architecture, data quality, and rigorous evaluation. I stopped Assignment 5 after establishing the zero-shot baseline to prioritize an independent model-design project.

## Next Steps: Project Le Mot Juste
I am starting Le Mot Juste, an independent evaluation project examining whether LLM translations preserve the emotions, tone, and psychological voice of French literature.

The first study will compare Claude, ChatGPT, Gemini, and a published English translation on selected passages from Maupassant’s Le Horla. I will establish a reproducible prompting and evaluation protocol, analyze model-specific failure modes, and publish the findings. Later studies will extend the framework to Flaubert, Stendhal, and Proust.
