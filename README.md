# Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation


## ❤️ Community Support


We will actively maintain this repository by incorporating new research as it emerges. If you have any suggestions regarding our taxonomy, find any missed papers, or update any preprint arXiv paper that has been accepted to some venue, feel free to send us an email or submit a **pull request** using the following markdown format.

```markdown
Paper Title, <ins>Conference/Journal/Preprint, Year</ins>  [[pdf](link)] [[other resources](link)].
```

## 📌 What is This Survey About?

Data contamination has received increasing attention in the era of large language models (LLMs) due to their reliance on vast Internet-derived training corpora. To mitigate the risk of potential data contamination, LLM benchmarking has undergone a transformation from *static* to *dynamic* benchmarking. In this work, we conduct an in-depth analysis of existing *static* to *dynamic* benchmarking methods aimed at reducing data contamination risks. We first examine methods that enhance *static* benchmarks and identify their inherent limitations. We then highlight a critical gap—the lack of standardized criteria for evaluating *dynamic* benchmarks. Based on this observation, we propose a series of optimal design principles for *dynamic* benchmarking and analyze the limitations of existing *dynamic* benchmarks. This survey provides a concise yet comprehensive overview of recent advancements in data contamination research, offering valuable insights and a clear guide for future research efforts.

## 🤔 What is data contamination?

Data contamination occurs when benchmark data is inadvertently included in the training phase of language models, leading to an inflated and misleading assessment of their performance. While this issue has been recognized for some time—stemming from the fundamental machine learning principle of separating training and test sets—it has become even more critical with the advent of LLMs. These models often scrape vast amounts of publicly available data from the Internet, significantly increasing the likelihood of contamination. Furthermore, due to privacy and commercial concerns, tracing the exact training data for these models is challenging, if not impossible, complicating efforts to detect and mitigate potential contamination.

## ❓ Why do we need this survey?
<!-- ![img/image.jpg](img/image.jpg) -->
This survey is necessary to address the growing issue of data contamination in LLM benchmarking, which compromises the reliability of **static benchmarks** that rely on fixed, human-curated datasets. While methods like data encryption and post-hoc contamination detection attempt to mitigate this issue, they have inherent limitations. **Dynamic benchmarking** has emerged as a promising alternative, yet existing reviews focus primarily on post-hoc detection and lack a systematic analysis of dynamic methods. Moreover, no standardized criteria exist for evaluating these benchmarks. To bridge this gap, we comprehensively review contamination-free benchmarking strategies, assess their strengths and limitations, and propose evaluation criteria for dynamic benchmarks, offering insights to guide future research and standardization.

## 📖 Table of Content
- [Static Benchmarking](#Static-Benchmarking)
    - [Static Benchmarking Applications](#Static-Benchmarking-Applications)
      - [Math](#Math)
      - [Knowledge](#Knowledge)
      - [Coding](#Coding)
      - [Instruction Following](#Instruction-Following)
      - [Reasoning](#Reasoning)
      - [Safety](#Safety)
      - [Language](#Language)
      - [Reading Comprehension](#Reading-Comprehension)
    - [Methods for Mitigation](#Methods-For-Mitigation)
      - [Canary String](#Canary-String)
      - [Encryption](#Encryption)
      - [Label Protection](#Label-Protection)
      - [Post-hoc Detection](#Post-Hoc-Detection)
- [Dynamic Benchmarking](#Dynamic-Benchmarking)
  - [Dynamic Benchmark Application](#Dynamic-Benchmark-Application)
    - [Temporal Cutoff](#Temporal-Cutoff)
    - [Rule-Based Generation](#Rule-Based-Generation)
      - [Template-Based](#Template-Based)
      - [Table-Based](#Table-Based)
      - [Graph-Based](#Graph-Based)
    - [LLM-Based Generation](#LLM-Based-Generation)
      - [Benchmark Rewriting](#Benchmark-Rewriting)
      - [Interactive Evaluation](#Interactive-Evaluation)
      - [Multi-Agent Evaluation](#Multi-Agent-Evaluation)
    - [Hybrid Generation](#Hybrid-Generation)






## Static Benchmarking
### Static Benchmark Application
#### Math
- Training Verifiers to Solve Math Word Problems, <ins>arXiv, 2021</ins> [[Paper](https://arxiv.org/pdf/2110.14168)] [[Code](https://github.com/openai/grade-school-math)]
- Measuring Mathematical Problem Solving With the MATH Dataset, <ins>NeurIPS, 2021</ins> [[Paper](https://arxiv.org/abs/2103.03874)] [[Code](https://github.com/hendrycks/math)]
#### Knowledge
- TriviaQA: A Large Scale Distantly Supervised Challenge Dataset
for Reading Comprehension, <ins>ACL, 2017</ins> [[Paper](https://aclanthology.org/P17-1147/)] [[Code](https://nlp.cs.washington.edu/triviaqa/)]
- Natural questions: a benchmark for question answering research, <ins>TACL, 2019</ins> [[Paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question)] [[Code](https://ai.google.com/research/NaturalQuestions)]
- Measuring Massive Multitask Language Understanding, <ins>ICLR, 2021</ins> [[Paper](https://openreview.net/forum?id=d7KBjmI3GmQ)] [[Code](https://github.com/hendrycks/test)]
- Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.824/)] [[Code](https://github.com/suzgunmirac/BIG-Bench-Hard)]
- AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-naacl.149/)] [[Code](https://github.com/ruixiangcui/AGIEval)]
- Are We Done with MMLU?, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2406.04127)] [[Code](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0)]
- MMLU-Pro: A More Robust and Challenging
Multi-Task Language Understanding Benchmark, <ins>NeurIPS, 2024</ins> [[Paper](https://arxiv.org/pdf/2406.01574)] [[Code](https://github.com/TIGER-AI-Lab/MMLU-Pro)]
- Capabilities of Large Language Models in Control Engineering:
A Benchmark Study on GPT-4, Claude 3 Opus, and Gemini 1.0 Ultra, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2404.03647)] 
- GPQA: A Graduate-Level Google-Proof
Q&A Benchmark, <ins>COLM, 2024</ins> [[Paper](https://arxiv.org/pdf/2311.12022)] [[Code](https://github.com/idavidrein/gpqa/)]
- Length-Controlled AlpacaEval:
A Simple Way to Debias Automatic Evaluators, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.04475)] [[Code](https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file)]
- FROM CROWDSOURCED DATA TO HIGH-QUALITY
BENCHMARKS: ARENA-HARD AND BENCHBUILDER
PIPELINE, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2406.11939)] [[Code](https://github.com/lmarena/arena-hard-auto)]
- Fact, Fetch, and Reason: A Unified Evaluation of
Retrieval-Augmented Generation, <ins>NAACL, 2025</ins> [[Paper](https://arxiv.org/pdf/2409.12941)] [[Code](https://huggingface.co/datasets/google/frames-benchmark)]
- AIME., [[Website](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOorI76-rO7SIb5k4OFKc-0omPLPimr5TnY6Phqz-PW8q6WsfYOiz)]
- CNMO., [[Website](https://www.cms.org.cn/Home/comp/comp/cid/12.html)]
#### Coding
- Evaluating Large Language Models Trained on Code, <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/pdf/2107.03374)] [[Code](https://github.com/openai/human-eval)]
- Program Synthesis with Large Language Models, <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/pdf/2108.07732)] [[Code](https://github.com/google-research/google-research/tree/master/mbpp)]
- SWE-bench: Can Language Models Resolve Real-world Github Issues?, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2310.06770)] [[Code](https://www.swebench.com/)]
- SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?, <ins>ICLR, 2025</ins> [[Paper](https://arxiv.org/abs/2410.03859)] [[Code](https://www.swebench.com/multimodal)]
- Codeforces: Competitive programming platform., [[Website](https://codeforces.com/)] 
- Aider., [[Website](https://aider.chat/)] 
#### Instruction Following 
- Instruction-Following Evaluation for Large Language
Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2311.07911)] [[Code](https://github.com/google-research/google-research/tree/master/instruction_following_eval)]
- C-EVAL: A Multi-Level Multi-Discipline Chinese
Evaluation Suite for Foundation Models, <ins>NeurIPS, 2023</ins> [[Paper](https://github.com/hkust-nlp/ceval)] [[Code](https://github.com/qinyiwei/InfoBench)]
- INFOBENCH: Evaluating Instruction Following Ability
in Large Language Models, <ins>ACL, 2024</ins> [[Paper](https://arxiv.org/pdf/2401.03601)] [[Code](https://github.com/qinyiwei/InfoBench)]
#### Reasoning
- Can a Suit of Armor Conduct Electricity?
A New Dataset for Open Book Question Answering, <ins>EMNLP, 2018</ins> [[Paper](https://aclanthology.org/D18-1260.pdf)] [[Code](https://leaderboard.allenai.org/open_book_qa)]
- Think you have Solved Question Answering?
Try ARC, the AI2 Reasoning Challenge, <ins>Arxiv, 2018</ins> [[Paper](https://arxiv.org/pdf/1803.05457)] [[Code](https://huggingface.co/datasets/allenai/ai2_arc)]
- HellaSwag: Can a Machine Really Finish Your Sentence?, <ins>ACL, 2019</ins> [[Paper](https://arxiv.org/pdf/1905.07830)] [[Code](https://rowanzellers.com/hellaswag/)]
- WINOGRANDE: An Adversarial Winograd Schema Challenge at Scale,<ins>ACL, 2019</ins>[[Paper](https://arxiv.org/pdf/1907.10641)] [[Code](https://winogrande.allenai.org/)]
- COMMONSENSEQA: A Question Answering Challenge Targeting
Commonsense Knowledge
, <ins>NAACL, 2019</ins> [[Paper](https://aclanthology.org/N19-1421.pdf)] [[Code](https://github.com/jonathanherzig/commonsenseqa)]
- SOCIAL IQA: Commonsense Reasoning about Social Interactions, <ins>EMNLP, 2019</ins> [[Paper](https://arxiv.org/pdf/1904.09728)] [[Code](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/social_iqa/README.md)]
- PIQA: Reasoning about Physical Commonsense in Natural Language, <ins>AAAI, 2020</ins> [[Paper](https://arxiv.org/abs/1911.11641)] [[Code](https://yonatanbisk.com/piqa/)]
- CHINESE SIMPLEQA: A CHINESE FACTUALITY EVALUATION FOR LARGE LANGUAGE MODELS, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2411.07140)] [[Code](https://openstellarteam.github.io/ChineseSimpleQA/)]
#### Safety
- REALTOXICITYPROMPTS:
Evaluating Neural Toxic Degeneration in Language Models, <ins>EMNLP, 2020</ins> [[Paper](https://aclanthology.org/2020.findings-emnlp.301.pdf)] [[Code](https://github.com/allenai/real-toxicity-prompts)]
- TOXIGEN: A Large-Scale Machine-Generated Dataset for Adversarial
and Implicit Hate Speech Detection, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.234.pdf)] [[Code](https://github.com/microsoft/toxigen)]
#### Language
- GLUE: A Multi-Task Benchmark and Analysis Platform
for Natural Language Understanding, <ins>EMNLP, 2018</ins> [[Paper](https://aclanthology.org/W18-5446.pdf)] [[Code](https://paperswithcode.com/paper/glue-a-multi-task-benchmark-and-analysis)]
- SuperGLUE: A Stickier Benchmark for
General-Purpose Language Understanding Systems, <ins>NeurIPS, 2019</ins> [[Paper](https://arxiv.org/pdf/1905.00537)] [[Code](https://huggingface.co/datasets/aps/super_glue)]
- CLUE: A Chinese Language Understanding Evaluation Benchmark, <ins>COLING, 2020</ins> [[Paper](https://aclanthology.org/2020.coling-main.419.pdf)] [[Code](https://github.com/CLUEbenchmark/CLUE)]
- CLUE: A Chinese Language Understanding Evaluation Benchmark, <ins>COLING, 2020</ins> [[Paper](https://aclanthology.org/2020.coling-main.419.pdf)] [[Code](https://github.com/CLUEbenchmark/CLUE)]
- Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.824/)] [[Code](https://github.com/suzgunmirac/BIG-Bench-Hard)]
#### Reading Comprehension
- Know What You Don’t Know: Unanswerable Questions for SQuAD, <ins>ACL, 2018</ins> [[Paper](https://aclanthology.org/P18-2124.pdf)] [[Code](https://rajpurkar.github.io/SQuAD-explorer/)]
- QuAC : Question Answering in Context, <ins>EMNLP, 2018</ins> [[Paper](https://aclanthology.org/D18-1241.pdf)] [[Code](https://quac.ai/)]
- BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions
, <ins>NAACL, 2019</ins> [[Paper](https://aclanthology.org/N19-1300.pdf)] [[Code](https://github.com/google-research-datasets/boolean-questions)]
### Methods for Mitigation
#### Canary String
#### Encryption
#### Label Protection
#### Post-hoc Detection
## Dynamic Benchmarking
### Dynamic Benchmark Application
#### Temporal Cutoff
#### Rule-Based Generation
##### Template-Based
##### Table-Based
##### Graph-Based
#### LLM-Based Generation
##### Benchmark Rewriting
##### Interactive Evaluation
##### Multi-Agent Evaluation
#### Hybrid Generation



 <!-- [^1]: This table was updated Dec 2023. This table will require updates as cool new frameworks are being released frequently and current frameworks continue to mature at an accelerated rate. So please feel free to suggest any important distinguishing features or popular new frameworks-->

](https://hackmd.io/l1wtpy8XSsiNIuzg1BqXsA?both)
