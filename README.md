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
![img/image.jpg](img/image.jpg)
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
- Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating
Data Contamination by Evaluation Benchmarks
, <ins>EMNLP, 2023</ins> [[Paper](https://aclanthology.org/2023.emnlp-main.308.pdf)] [[Code](https://github.com/google-research-datasets/boolean-questions)]
#### Encryption
- Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating
Data Contamination by Evaluation Benchmarks
, <ins>EMNLP, 2023</ins> [[Paper](https://aclanthology.org/2023.emnlp-main.308.pdf)] [[Code](https://github.com/google-research-datasets/boolean-questions)]
- Rethinking Benchmark and Contamination for Language Models with
Rephrased Samples
, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2311.04850)] [[Code](https://github.com/lm-sys/llm-decontaminator)]
- TRUCE: Private Benchmarking to Prevent
Contamination and Improve Comparative Evaluation
of LLMs, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2403.00393)] [[Code](https://github.com/microsoft/private-benchmarking?tab=readme-ov-file)]
#### Label Protection
- GLUE: A Multi-Task Benchmark and Analysis Platform
for Natural Language Understanding, <ins>EMNLP, 2018</ins> [[Paper](https://aclanthology.org/W18-5446.pdf)] [[Code](https://paperswithcode.com/paper/glue-a-multi-task-benchmark-and-analysis)]
- SuperGLUE: A Stickier Benchmark for
General-Purpose Language Understanding Systems, <ins>NeurIPS, 2019</ins> [[Paper](https://arxiv.org/pdf/1905.00537)] [[Code](https://huggingface.co/datasets/aps/super_glue)]
- Evaluating Large Language Models Trained on Code, <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/pdf/2107.03374)] [[Code](https://github.com/openai/human-eval)]
#### Post-hoc Detection
- Quantifying Contamination in Evaluating Code Generation
Capabilities of Language Models, <ins>ACL, 2024</ins> [[Paper](https://aclanthology.org/2024.acl-long.761.pdf)] [[Code](https://github.com/yale-nlp/code-llm-contamination)]
- Platypus: Quick, Cheap, and Powerful
Refinement of LLMs, <ins>NeurIPS Workshop, 2023</ins> [[Paper](https://arxiv.org/pdf/2308.07317)] [[Code](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)]
- Textbooks Are All You Need, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2306.11644)] [[Code](https://github.com/kyegomez/phi-1)]
- An Open-Source Data Contamination Report for Large Language Models, <ins>EMNLP, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-emnlp.30.pdf)] [[Code](https://github.com/liyucheng09/Contamination_Detector)]
- Benchmarking Benchmark Leakage in Large Language
Models, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2308.07317)] [[Code](https://github.com/GAIR-NLP/benbench)]
- Investigating the Impact of Data Contamination of Large Language Models
in Text-to-SQL Translation, <ins>ACL, 2024</ins> [[Paper](https://arxiv.org/pdf/2402.08100)] [[Code](https://github.com/ART-Group-it/Termite)]
- Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4
, <ins>EMNLP, 2023</ins> [[Paper](https://aclanthology.org/2023.emnlp-main.453.pdf)] [[Code](https://github.com/StellaAthena/speak-memory)]
- Time Travel in LLMs: Tracing Data Contamination in Large Language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/pdf/2308.08493)] [[Code](https://github.com/shahriargolchin/time-travel-in-llms)]
- DE-COP: Detecting Copyrighted Content in Language Models Training Data, <ins>ICML, 2024</ins> [[Paper](https://arxiv.org/pdf/2402.09910)] [[Code](TVdRFwWMxqZMVRKnRALCmKoLjv6G1VwWnD)]
- Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models
, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/abs/2311.06233)] [[Code](https://github.com/shahriargolchin/DCQ)]
- Fool Your (Vision and) Language Model With Embarrassingly Simple Permutations
, <ins>ICML, 2024</ins> [[Paper](https://arxiv.org/pdf/2310.01651)] [[Code](https://github.com/ys-zong/FoolyourVLLMs)]
- ConStat: Performance-Based Contamination
Detection in Large Language Models
, <ins>NeurIPS, 2024</ins> [[Paper](https://arxiv.org/pdf/2405.16281)] [[Code](https://github.com/eth-sri/ConStat)]

## Dynamic Benchmarking
### Dynamic Benchmark Application
#### Temporal Cutoff
- AntiLeak-Bench: Preventing Data Contamination by Automatically
Constructing Benchmarks with Updated Real-World Knowledge
, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2412.13670)] [[Code](https://github.com/bobxwu/AntiLeak-Bench)]
- LiveBench: A Challenging, Contamination-Free LLM Benchmark
, <ins>ICLR, 2025</ins> [[Paper](https://arxiv.org/pdf/2406.19314)] [[Code](https://github.com/LiveBench/LiveBench)]
- ACADEMICEVAL: LIVE LONG-CONTEXT LLM
BENCHMARK, <ins>Arxiv, 2025</ins> [[Paper](https://openreview.net/pdf?id=iRYExPKnxm)] 
- LiveCodeBench: Holistic and Contamination Free Evaluation of
Large Language Models for Code
, <ins>ICLR, 2025</ins> [[Paper](https://arxiv.org/pdf/2403.07974)] [[Code](https://github.com/LiveCodeBench/LiveCodeBench)]
- LEVERAGING ONLINE OLYMPIAD-LEVEL
MATH PROBLEMS FOR LLMS TRAINING AND
CONTAMINATION-RESISTANT EVALUATION
, <ins>Arxiv, 2025</ins> [[Paper](https://arxiv.org/pdf/2501.14275)] [[Code](https://github.com/DSL-Lab/aops)]
- ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities
, <ins>ICLR, 2025</ins> [[Paper](https://arxiv.org/abs/2409.19839)] [[Code](https://www.forecastbench.org/)]
#### Rule-Based Generation
##### Template-Based
- GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models
, <ins>ICLR, 2025</ins> [[Paper](https://arxiv.org/pdf/2410.05229)] [[Code](https://github.com/apple/ml-gsm-symbolic)]
- Mathador-LM: A Dynamic Benchmark for Mathematical Reasoning
on Large Language Models
, <ins>EMNLP, 2024</ins> [[Paper](https://aclanthology.org/2024.emnlp-main.946.pdf)] [[Code](https://github.com/IST-DASLab/Mathador-LM)]
- MMLU-CF: A Contamination-free Multi-task Language Understanding
Benchmark
, <ins>EMNLP, 2024</ins> [[Paper](https://arxiv.org/pdf/2412.15194)] [[Code](https://github.com/microsoft/MMLU-CF)]
##### Table-Based
- S3Eval: A Synthetic, Scalable, Systematic Evaluation Suite for
Large Language Models
, <ins>NAACl, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-long.69.pdf)] [[Code](https://github.com/lfy79001/S3Eval)]
##### Graph-Based
- DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks
, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/pdf/2309.17167)] [[Code](https://github.com/microsoft/promptbench)]
- NPHardEval: Dynamic Benchmark on Reasoning Ability of Large
Language Models via Complexity Classes
, <ins>ACL, 2024</ins> [[Paper](https://aclanthology.org/2024.acl-long.225.pdf)] [[Code](https://github.com/casmlab/NPHardEval)]
- ON MEMORIZATION OF LARGE LANGUAGE MODELS
IN LOGICAL REASONING
, <ins>NeurIPS Workshop, 2024</ins> [[Paper](https://arxiv.org/pdf/2410.23123)] [[Code](https://memkklogic.github.io/)]
#### LLM-Based Generation
##### Benchmark Rewriting
- Automating Dataset Updates Towards Reliable and Timely Evaluation of Large Language Models
, <ins>NeurIPS, 2024</ins> [[Paper](https://openreview.net/pdf?id=EvEqYlQv8T)] [[Code](https://yingjiahao14.github.io/Automating-DatasetUpdates/)]
- Inference-Time Decontamination: Reusing Leaked Benchmarks for Large
Language Model Evaluation
, <ins>EMNLP, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-emnlp.532.pdf)] [[Code](https://github.com/8188zq/Inference-Time-Decontamination)]
- StructEval: Deepen and Broaden Large Language Model Assessment
via Structured Evaluation
, <ins>ACL, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.314.pdf)] [[Code](https://github.com/c-box/StructEval)]
- VarBench: Robust Language Model Benchmarking Through Dynamic
Variable Perturbation
, <ins>EMNLP, 2024</ins> [[Paper](https://arxiv.org/pdf/2406.17681)] [[Code](https://github.com/qbetterk/VarBench)]
- Automating Dataset Updates Towards Reliable and Timely Evaluation of Large Language Models
, <ins>NeurIPS, 2024</ins> [[Paper](https://openreview.net/pdf?id=EvEqYlQv8T)] [[Code](https://yingjiahao14.github.io/Automating-DatasetUpdates/)]
##### Interactive Evaluation
- LLM-AS-AN-INTERVIEWER:
Beyond Static Testing Through Dynamic LLM Evaluation
, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2412.10424)] [[Code](https://github.com/interview-eval/)]
- TreeEval: Benchmark-Free Evaluation of Large Language Models through Tree
Planning
, <ins>NeurIPS, 2024</ins> [[Paper](https://arxiv.org/pdf/2402.13125)] [[Code](https://github.com/Ashura5/TreeEval)]
- KIEval: A Knowledge-grounded Interactive Evaluation Framework for
Large Language Models
, <ins>ACL , 2024</ins> [[Paper](https://aclanthology.org/2024.acl-long.325.pdf)] [[Code](https://github.com/zhuohaoyu/KIEval)]
##### Multi-Agent Evaluation
- Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation
, <ins>COLING , 2025</ins> [[Paper](https://aclanthology.org/2025.coling-main.223.pdf)] [[Code](https://github.com/NanshineLoong/Self-Evolving-Benchmark)]
- BENCHAGENTS: Automated Benchmark Creation with Agent Interaction
, <ins>Arxiv , 2024</ins> [[Paper](https://arxiv.org/pdf/2410.22584)]
#### Hybrid Generation
- LatestEval: Addressing Data Contamination in Language Model Evaluation
through Dynamic and Time-Sensitive Test Construction
, <ins>AAAI, 2024</ins> [[Paper](https://arxiv.org/pdf/2312.12343)] [[Code](https://github.com/liyucheng09/LatestEval)]
- DARG: Dynamic Evaluation of Large Language
Models via Adaptive Reasoning Graph
, <ins>NeurIPS, 2024</ins> [[Paper](https://openreview.net/pdf?id=5IFeCNA7zR)] [[Code](https://github.com/SALT-NLP/DARG)]
- C2LEVA: Toward Comprehensive and Contamination-Free
Language Model Evaluation
, <ins>AAAI, 2024</ins> [[Paper](https://arxiv.org/pdf/2412.04947)] 


 <!-- [^1]: This table was updated Dec 2023. This table will require updates as cool new frameworks are being released frequently and current frameworks continue to mature at an accelerated rate. So please feel free to suggest any important distinguishing features or popular new frameworks-->

