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
- [🤖 Model-Centric Methods](#Model-Centric-Methods) 
  - [Model Compression](#Model-Compression) 
    - [Quantization](#Quantization)
      - [Post-Training Quantization](#Post-Training-Quantization)
        - [Weight-Only Quantization](#Weight-Only-Quantization)
        - [Weight-Activation Co-Quantization](#Weight-Activation-Co-Quantization)
        - [Evaluation of Post-Training Quantization](#Evaluation-of-Post-Training-Quantization)


🤖 
## Model-Centric Methods
### Model Compression
#### Quantization
##### Post-Training Quantization
###### Weight-Only Quantization
- I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17849)] 
- IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01241)] 
- OmniQuant: OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- OneBit: Towards Extremely Low-bit Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.11295)]
- GPTQ: Accurate Quantization for Generative Pre-trained Transformers, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13304)] [[Code](https://github.com/jerry-chee/QuIP)]
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- OWQ: Lessons Learned from Activation Outliers for Weight Quantization in Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.02272)] [[Code](https://github.com/xvyaward/owq)]
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2306.03078)] [[Code](https://github.com/Vahe1994/SpQR)]
- FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs, <ins>NeurIPS-ENLSP, 2023</ins> [[Paper](https://arxiv.org/abs/2308.09723)]
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=dXiGWqBoxaD)] [[Code](https://github.com/TimDettmers/bitsandbytes)]
- Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2208.11580)] [[Code](https://github.com/IST-DASLab/OBC)]
- QuantEase: Optimization-based Quantization for Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.01885)] [[Code](https://github.com/linkedin/QuantEase)]
###### Weight-Activation Co-Quantization
- Rotation and Permutation for Advanced Outlier Management and Efficient Quantization of LLMs, <ins>NeurIPS, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] 
- OmniQuant: OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- Intriguing Properties of Quantization at Scale, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.19268)]
- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2303.08302)] [[Code](https://github.com/microsoft/DeepSpeed)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats, <ins>NeurIPS-ENLSP, 2023</ins> [[Paper](https://arxiv.org/abs/2307.09782)] [[Code](https://github.com/microsoft/DeepSpeed)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization, <ins>ISCA, 2023</ins> [[Paper](https://arxiv.org/abs/2304.07493)] [[Code](https://github.com/clevercool/ANT-Quantization)]
- RPTQ: Reorder-based Post-training Quantization for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.01089)] [[Code](https://github.com/hahnyuan/RPTQ4LLM)]
- Outlier Suppression+: Accurate Quantization of Large Language Models by Equivalent and Optimal Shifting and Scaling, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.09145)] [[Code](https://github.com/ModelTC/Outlier_Suppression_Plus)]
- QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.08041)]
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers, <ins>NeurIPS, 2022</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html)]

###### Evaluation of Post-Training Quantization
 - Evaluating Quantized Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.18158)]
##### Quantization-Aware Training
- The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.17764)]
- FP8-LM: Training FP8 Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.18313)]
- Training and inference of large language models using 8-bit floating point, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.17224)]
- BitNet: Scaling 1-bit Transformers for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.11453)]
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17888)] [[Code](https://github.com/facebookresearch/LLM-QAThttps://github.com/facebookresearch/LLM-QAT)]
- Compression of Generative Pre-trained Language Models via Quantization, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.331.pdf)]
#### Parameter Pruning
##### Structured Pruning
- Compact Language Models via Pruning and Knowledge Distillation, <ins>arXiv, 2024</ins> [[Paper](https://www.arxiv.org/abs/2407.14679)] 
- A deeper look at depth pruning of LLMs, <ins>arXiv, 2024</ins> [[Paper](https://www.arxiv.org/abs/2407.16286)] 
- Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.20541)] 
- Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models, <ins>ICLR, 2024</ins> [[Paper](https://openreview.net/forum?id=Tr0lPx9woF)] 
- BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.16880)] 
- ShortGPT: Layers in Large Language Models are More Redundant Than You Expect, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.03853)] 
- NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.09773)] 
- SliceGPT: Compress Large Language Models by Deleting Rows and Columns, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2401.15024)] [[Code](https://github.com/microsoft/TransformerCompression?utm_source=catalyzex.com)]
- LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.18356)]
- LLM-Pruner: On the Structural Pruning of Large Language Models, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11627)] [[Code](https://github.com/horseee/LLM-Pruner)]
- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning, <ins> NeurIPS-ENLSP, 2023</ins> [[Paper](https://arxiv.org/abs/2310.06694)] [[Code](https://github.com/princeton-nlp/LLM-Shearing)]
- LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, <ins>arXiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2305.18403)]
- LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models, <ins>ICML, 2024</ins> [[Paper](https://arxiv.org/abs/2404.09695)][[Code](https://github.com/lihuang258/LoRAP)]
##### Unstructured Pruning
- MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models, <ins>NIPS, 2024</ins> [[Paper](https://arxiv.org/abs/2409.17481)] 
- Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2310.08915)] 
- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- A Simple and Effective Pruning Approach for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]
- One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2310.09499v1.pdf)]
#### Low-Rank Approximation
- SVD-LLM: Singular Value Decomposition for Large Language Model Compression, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.07378)] [[Code](https://github.com/AIoT-MLSys-Lab/SVD-LLM)]
- ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.05821)] [[Code](https://github.com/hahnyuan/ASVD4LLM)]
- Language model compression with weighted low-rank factorization, <ins>ICLR, 2022</ins> [[Paper](https://arxiv.org/abs/2207.00112)]
- TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on the Tensor-Train Decomposition, <ins>arXiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.00526)]
- LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, <ins>ICML, 2023</ins>  [[Paper](https://arxiv.org/abs/2306.11222)] [[Code](https://github.com/yxli2123/LoSparse)]
#### Knowledge Distillation
##### White-Box KD
- DDK: Distilling Domain Knowledge for Efficient Large Language Models <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.16154)]
- Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.02657)]
- DistiLLM: Towards Streamlined Distillation for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.03898)] [[Code](https://github.com/jongwooko/distillm)]
- Towards the Law of Capacity Gap in Distilling Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.07052)] [[Code](https://github.com/GeneZC/MiniMA)]
- Baby Llama: Knowledge Distillation from an Ensemble of Teachers Trained on a Small Dataset with no Performance Penalty, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.02019)]
- Knowledge Distillation of Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.08543)] [[Code](https://github.com/microsoft/LMOps/tree/main/minillm)]
- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.13649)]
- Propagating Knowledge Updates to LMs Through Distillation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09306)] [[Code](https://github.com/shankarp8/knowledge_distillation)]
- Less is More: Task-aware Layer-wise Distillation for Language Model Compression, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/pdf/2210.01351.pdf)]
- Token-Scaled Logit Distillation for Ternary Weight Generative Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.06744)]
##### Black-Box KD
- Zephyr: Direct Distillation of LM Alignment, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.09571)]
- Instruction Tuning with GPT-4, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.03277)] [[Code](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)]
- Lion: Adversarial Distillation of Closed-Source Large Language Model, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.12870)] [[Code](https://github.com/YJiangcm/Lion)]
- Specializing Smaller Language Models towards Multi-Step Reasoning, <ins>ICML, 2023</ins> [[Paper](https://aclanthology.org/2022.findings-naacl.169.pdf)] [[Code](https://github.com/FranxYao/FlanT5-CoT-Specialization)]
- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2305.02301)]
- Large Language Models Are Reasoning Teachers, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2212.10071)] [[Code](https://github.com/itsnamgyu/reasoning-teacher)]
- SCOTT: Self-Consistent Chain-of-Thought Distillation, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2305.01879)] [[Code](https://github.com/wangpf3/consistent-CoT-distillation)]
- Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14050)]
- Distilling Reasoning Capabilities into Smaller Language Models, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.441/)] [[Code](https://github.com/kumar-shridhar/Distiiling-LM)]
- In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2212.10670)]
- Explanations from Large Language Models Make Small Reasoners Better, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2210.06726)]
- DISCO: Distilling Counterfactuals with Large Language Models, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2212.10534)] [[Code](https://github.com/eric11eca/disco)]
##### Parameter-Sharing
- MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.16840)] 
### Efficient Pre-Training
#### Mixed Precision Training
- Bfloat16 Processing for Neural Networks, <ins>ARITH, 2019</ins> [[Paper](https://ieeexplore.ieee.org/document/8877390)]
- A Study of BFLOAT16 for Deep Learning Training, <ins>arXiv, 2019</ins> [[Paper](https://arxiv.org/abs/1905.12322)]
- Mixed Precision Training, <ins>ICLR, 2018</ins> [[Paper](https://openreview.net/forum?id=r1gs9JgRZ)]
#### Scaling Models
- lemon: lossless model expansion, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2310.07999)] 
- Preparing Lessons for Progressive Training on Language Models, <ins>AAAI, 2024</ins> [[Paper](https://arxiv.org/abs/2401.09192)] 
- Learning to Grow Pretrained Models for Efficient Transformer Training, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/pdf?id=cDYRS5iZ16f)] [[Code](https://github.com/VITA-Group/LiGO)]
- 2x Faster Language Model Pre-training via Masked Structural Growth, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.02869)]
- Reusing Pretrained Models by Multi-linear Operators for Efficient Training, <ins>NeurIPS, 2023</ins> [[Paper](https://openreview.net/pdf?id=RgNXKIrWyU)]
- FLM-101B: An Open LLM and How to Train It with $100 K Budget, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2309.03852.pdf)] [[Code](https://huggingface.co/CofeAI/FLM-101B)]
- Knowledge Inheritance for Pre-trained Language Models, <ins>NAACL, 2022</ins> [[Paper](https://aclanthology.org/2022.naacl-main.288/)] [[Code](https://github.com/thunlp/Knowledge-Inheritance)]
- Staged Training for Transformer Language Models, <ins>ICML, 2022</ins> [[Paper](https://proceedings.mlr.press/v162/shen22f/shen22f.pdf)] [[Code](https://github.com/allenai/staged-training)]
#### Initialization Techniques
- Deepnet: Scaling transformers to 1,000 layers, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2203.00555)] [[Code](https://github.com/microsoft/torchscale)]
- ZerO Initialization: Initializing Neural Networks with only Zeros and Ones, <ins>TMLR, 2022</ins> [[Paper](https://openreview.net/pdf?id=1AxQpKmiTc)] [[Code](https://github.com/jiaweizzhao/ZerO-initialization)]
- Rezero is All You Need: Fast Convergence at Large Depth, <ins>UAI, 2021</ins> [[Paper](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf)] [[Code](https://github.com/majumderb/rezero)]
- Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks, <ins>NeurIPS, 2020</ins> [[Paper](https://papers.neurips.cc/paper/2020/file/e6b738eca0e6792ba8a9cbcba6c1881d-Paper.pdf)]
- Improving Transformer Optimization Through Better Initialization, <ins>ICML, 2020</ins> [[Paper](https://proceedings.mlr.press/v119/huang20f/huang20f.pdf)] [[Code](https://github.com/layer6ai-labs/T-Fixup)]
- Fixup Initialization: Residual Learning without Normalization, <ins>ICLR, 2019</ins> [[Paper](https://openreview.net/pdf?id=H1gsz30cKX)]
- On Weight Initialization in Deep Neural Networks, <ins>arXiv, 2017</ins> [[Paper](https://arxiv.org/abs/1704.08863)]
#### Training Optimizers
- Towards Optimal Learning of Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.17759)] [[Code](https://github.com/microsoft/LMOps/tree/main/learning_law)]
- Symbolic Discovery of Optimization Algorithms, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.06675)]
- Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14342)] [[Code](https://github.com/Liuhong99/Sophia)]



 <!-- [^1]: This table was updated Dec 2023. This table will require updates as cool new frameworks are being released frequently and current frameworks continue to mature at an accelerated rate. So please feel free to suggest any important distinguishing features or popular new frameworks-->

