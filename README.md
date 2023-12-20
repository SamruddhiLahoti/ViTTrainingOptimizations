<base target="_blank">

# ViT Training Optimizations

The goal of this project is to conduct an ablation study to assess the individual and combined impact of four novel training optimization techniques on Vision Transformers (ViTs). The techniques to be evaluated are [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf), [DeCAtt (loss regularizer)](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Bhattacharyya_DeCAtt_Efficient_Vision_Transformers_With_Decorrelated_Attention_Heads_CVPRW_2023_paper.pdf), and [Mixed-Resolution Tokenization](https://arxiv.org/abs/2304.00287). This study aims to provide insights into their effectiveness in improving the training process and model performance of ViTs.

## A Brief Overview of the Optimizations

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness}

[FlashAttention](https://arxiv.org/pdf/2205.14135.pdf) is an IO-aware, exact attention algorithm that utilizes tiling to minimize memory reads/writes between GPU high bandwidth memory (HBM) and on-chip SRAM. This approach results in fewer HBM accesses than standard attention, making it optimal for a range of SRAM sizes and improving overall efficiency. The optimization is achieved by computing the softmax reduction without access to the whole input during gradient calculation and not storing the large intermediate attention matrix for the backward pass.

### DeCAtt: Efficient Vision Transformers with Decorrelated Attention Heads

The [De-Correlation Loss](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Bhattacharyya_DeCAtt_Efficient_Vision_Transformers_With_Decorrelated_Attention_Heads_CVPRW_2023_paper.pdf) paradigm minimizes the cross-correlation among the heads of each layer of vision transformers. This loss acts as a regularizer that mitigates overfitting as well as improves efficiency of the model as a whole. Decorrelating attention heads leads to significant boost in model performance and similar accuracy can be achieved with approximately 2.5 to 3 times lesser parameters. 

Let $`A`$ be the unrolled matrix of each attention map of heads with a dimension of $`B\times h\times (nd)`$ where $`B`$ is the batch size, $`h`$ is the number of heads, $`n`$ is the number of patches and $`d`$ is the dimension of query, key and value vectors. Then the regularization is defined as follows -

```math
C_1 = \dfrac{AA^T}{|A|^2}
```
```math
C_2[...]  = \frac{1}{B}\sum_{i=0}^{B} (C_1[...,i])^2
```
```math
\mathbb{L}_{DeCAtt} = \sum_{i=0}^{N} \sum_{j=0}^{N} C_2[i, j](i \neq j)
```
And the final update equation is
```math
\mathbb{L}_{total} = \mathbb{L}_{CE} + \lambda\mathbb{L}_{DeCAtt}
```

$`\lambda`$ is the decorrelation weight factor and $`\mathbb{L}_{CE}`$ is the Cross-Entropy (CE) loss.
Only the diagonal elements are summed since those are the ones that represent the de-correlation loss.

### Vision Transformers with Mixed-Resolution Tokenization

The vast majority of Vision Transformers use uniform patch tokenization, splitting the image into a spatially regular grid of equal-size patches. The [paper](https://arxiv.org/abs/2304.00287) introduces a novel image tokenization scheme, replacing the standard uniform grid with a mixed-resolution sequence of tokens, where each token represents a patch of arbitrary size. The image is split into a mixed-resolution patch mosaic according to a saliency scorer, and a standard Transformer architecture is employed with 2D position embeddings. These novel models are also less sensitive to out-of-distribution input lengths, showing a lower accuracy drop with respect to their retrained counterparts, and providing a better inference-time compute-accuracy tradeoff with a single model.

## Project Milestones

|    | Model                             | Status |
|:-: | :-------------------------------- | :----: |
| 1  | Baseline ViT                      | done   |
| 2  | ViT DeCAtt                        | done   |
| 3  | ViT MRT                           | done   |
| 4  | ViT FlashAttention                | done   |
| 5  | ViT MRT & FlashAttention          | done   |
| 6  | ViT DeCAtt & FlashAttention       | done   |
| 7  | ViT DeCAtt & MRT                  | done   |
| 8  | ViT DeCAtt, MRT & FlashAttention  | done   |

## Implementation Details

hardware, platform, framework, dataset used, functionalities, and limitations

#### Hardware
All experiments were run on Tesla-V100 GPUs provided by the NYU HPC team.

#### Framework and Libraries
Implementation and trainig of models done using the pytorch framework and associated libraries

#### Datasets used
We have used 2 image classification datasets, CIFAR10 and CIFAR100, to run our experiments and verify the results.

#### Limitation
The ImageNet dataset is too big to compute on for the purpose of this project. But the Mixed-Resolution tokenizer doesn't work for small image sizes because splitting the already very small patches (4✕4 sized) almost takes us to pixel space. So, the benchmarks were reached with resized images CIFAR10/CIFAR100 images.


## Repository Overview

* [vit.py](https://github.com/SamruddhiLahoti/ViTTrainingOptimizations/blob/main/vit.py): contains implementation of the ViT model.
* [vit-training.ipynb](https://github.com/SamruddhiLahoti/ViTTrainingOptimizations/blob/main/vit-training.ipynb): Contains driver code to train models 1-4 in the table above. The calculated statistics (train/val accuracy, train time & train loss) are stored in the stats folder. Model with the highest val accuracy for each experiment is stored in saved_models.
* [vit-ablation-training.ipynb](https://github.com/SamruddhiLahoti/ViTTrainingOptimizations/blob/main/vit-ablation-training.ipynb): contains driver code to train models 5-8 in the table above. The calculated statistics (train/val accuracy, train time & train loss) are stored in the stats folder. Model with the highest val accuracy for each experiment is stored in saved_models.
* [vit-inference.ipynb](https://github.com/SamruddhiLahoti/ViTTrainingOptimizations/blob/main/vit-inference.ipynb): plots the statistucs calculated for all modeles trained above against the baseline ViT model.

## Results

### CIFAR10

|    | Model                             |       Num Params       |          Train Time          | Max Val Accuracy |
|:-: | :-------------------------------- | :--------------------: | :--------------------------: | :--------------: |
| 1  | Baseline ViT                      |       N = 11.9M        |         T = 31 mins          |      78.39%      |
| 2  | ViT DeCAtt                        | 4.2M $`\approx`$ 0.35N |   26 mins $`\approx`$ 0.84T  |      80.31%      |
| 3  | ViT MRT                           |   $`\approx`$ 0.35N    |     |           |
| 4  | ViT FlashAttention                |   $`\approx`$ 0.35N    |   20 mins $`\approx`$ 0.65T  |      79.19%      |
| 5  | ViT MRT & FlashAttention          |   $`\approx`$ 0.35N    |    |           |
| 6  | ViT DeCAtt & FlashAttention       |   $`\approx`$ 0.35N    |  20.5 mins $`\approx`$ 0.66T |      79.62%      |
| 7  | ViT DeCAtt & MRT                  |   $`\approx`$ 0.35N    |     |           |
| 8  | ViT DeCAtt, MRT & FlashAttention  |   $`\approx`$ 0.35N    |     |            |


### CIFAR100

|    | Model                             |       Num Params       |          Train Time          | Max Val Accuracy |
|:-: | :-------------------------------- | :--------------------: | :--------------------------: | :--------------: |
| 1  | Baseline ViT                      |       N = 11.9M        |        T = 30.5 mins         |      45.72%      |
| 2  | ViT DeCAtt                        | 4.2M $`\approx`$ 0.35N | 25.26 mins $`\approx`$ 0.83T |      51.33%      |
| 3  | ViT MRT                           |   $`\approx`$ 0.35N    |     |           |
| 4  | ViT FlashAttention                |   $`\approx`$ 0.35N    |   20 mins $`\approx`$ 0.65T  |      50.5%       |
| 5  | ViT MRT & FlashAttention          |   $`\approx`$ 0.35N    |    |           |
| 6  | ViT DeCAtt & FlashAttention       |   $`\approx`$ 0.35N    |  17.2 mins $`\approx`$ 0.56T |      49.98%       |
| 7  | ViT DeCAtt & MRT                  |   $`\approx`$ 0.35N    |     |           |
| 8  | ViT DeCAtt, MRT & FlashAttention  |   $`\approx`$ 0.35N    |     |            |