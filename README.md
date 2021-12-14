  
# Revisiting stock price prediction models with RNN-based Neural Architecture Search
Kerem Guventurk (kg2900), Hojae Yoon (hy2714)

## Description of Project
### Problem Statement
In this project we experimented in defining a RNN search space in NAS since it's not defined well in literature.

### Problem Motivation
Most NAS studies focus on CNN based tasks however RNN based search space is not well defined in NAS literature. On top of that, the types of networks that are built to solve RNN problems are drastically different than that of CNN problems.

So the major questions we tried to answer in this project were:
1) How to formulate search space?
2) How to tune training hyper parameters?
3) How to implement training in existing frameworks?

### Background Work
In building this project, we refered and utilized these papers, tutorials, and repositories:
1) Efficient Architecture Search by Network Transformation
(https://arxiv.org/abs/1707.04873)
2) Neural Architecture Search with Controller RNN
(https://github.com/titu1994/neural-architecture-search)
3) A Technical Guide on RNN/LSTM/GRU for Stock Price Prediction
(https://medium.com/swlh/a-technical-guide-on-rnn-lstm-gru-for-stock-price-prediction-bce2f7f30346)
4) Stock Price Prediction Using CNN and LSTM-Based Deep Learning Models
(https://arxiv.org/pdf/2010.13891.pdf)

The code for this project was an adapted version of #2 (Neural Architecture Search with Controller RNN). While that project focused on exploring a CNN search space with RL agents, we adapted the code to be explore a RNN search space. Below is how the CNN search space was defined before:

![image](https://user-images.githubusercontent.com/44733338/146050569-14c70c9c-18e0-4c79-ae56-0e8ae0eeba8c.png)

### Our Approach
Below is our implementation of the NAS model combining the repository mentioned above and the high-level idea in "Efficient Architecture Search by Network Transformation"

Algorithm:
RL based NAS inspired by Efficient Architecture Search by Network Transformation
1. Train 300 different models for 10 epochs and selects a subset of well performing models
2. Selects the well performing models as the baseline for the next RL exploration stage
3. Trains 300 different models for 10 epochs with a higher exploitation rate (80%)
4. Selects the best (1) model and trains for 100 epochs

Functions:
- Dynamically resize width and depth
- Dynamically generates new models with diverse architectures
- RNN controller outputs probabilities of each block (with parameters) being selected

Comparison of 4 approaches
1. Vanilla LSTM
2. Vanilla GRU
3. NAS with basic search space (LSTM stacking with Dropout)
4. NAS with diverse search space (LSTM, GRU, Activation Functions, Dropout)

Below is how we defined the search space and states which were fed to the RL-based controller:

Search space:
![image](https://user-images.githubusercontent.com/44733338/146051142-c9974e1b-3cf2-46c1-aad8-47fd83691e4b.png)
<img width="200" height="400" alt="Screen Shot 2021-12-14 at 12 41 31 PM" src="https://user-images.githubusercontent.com/44733338/146051174-dbfa9927-9870-437f-814e-82c6027ab79f.png">

State representation:
![image](https://user-images.githubusercontent.com/44733338/146051254-ab9c59cc-0cdd-4005-8340-d5cfc5f4f590.png)

Solution Diagram / Architecture:

![image](https://user-images.githubusercontent.com/44733338/146052177-7faa2766-8c81-4c5a-8155-5b52c6055ab1.png)

### Implementation Details

Dataset: Daily and Intraday Price + Volume Data For All U.S. Stocks & ETFs > 1 Hour > Tesla
(https://www.kaggle.com/borismarjanovic/daily-and-intraday-stock-price-data)

Parameters:
- Batch Size = 32
- Sequence Length = 5
- Num of Max Layers = 8
- Num of of Runs = 2
- Trials Per Run = 300
- Num of Epochs per Trial = 10
- Exploration Rate Run 1 = 0.8
- Exploration Rate Run 2 = 0.2
- Regularization = 1e-3

Train Environment: GCP: Nvidia Tesla V100

ML Framework: Tensorflow

Types of Layer: LSTM, GRU, Dropout, Dense

Type of Activations: tanh, ReLU, sigmoid, linear, None

## Results
Graphs of each model on test data (TESLA stock after 2017):

Vanilla LSTM            |  Vanilla GRU
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/44733338/146055820-cedcccc1-3462-4e4e-a91d-4e11fed21255.png)  |  ![](https://user-images.githubusercontent.com/44733338/146056158-00e1db76-17f0-494c-9613-c08b105c3948.png)
Basic NAS Search Space           |  Diverse NAS Search Space  
![](https://user-images.githubusercontent.com/44733338/146056162-86c0b87a-7160-434b-8db7-1cff54b5fe56.png)  |  ![](https://user-images.githubusercontent.com/44733338/146056173-64faae48-c0cd-45fe-876b-dbe2b0ffcfbb.png)



Performance comparison of models:

![image](https://user-images.githubusercontent.com/44733338/146054160-4c7e2c57-e562-4490-8d5f-f5f54ef2615a.png)

Discovered architectures:

Vanilla LSTM            |  Vanilla GRU
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/44733338/146054520-f3f5c2ee-9e56-41be-87d6-351985edf4e8.png)  |  ![](https://user-images.githubusercontent.com/44733338/146054560-b434b52c-7440-4eae-bca6-34c14496198a.png)
Basic NAS Search Space           |  Diverse NAS Search Space  
![](https://user-images.githubusercontent.com/44733338/146055206-97579ac4-740e-4ce1-8990-b3f86c696bb5.png)  |  ![](https://user-images.githubusercontent.com/44733338/146055212-7be699b5-37e5-4c90-8de3-d7ad7a2f82c4.png)

Expanding TSLA stock trained model to other stocks

![image](https://user-images.githubusercontent.com/44733338/146055391-9433e84d-effb-4e99-bb88-262d223263ed.png)


```
## News
- Next generation of ProxylessNAS: [Once-for-All](https://github.com/mit-han-lab/once-for-all) (First place in the 3rd and 4th [Low-Power Computer Vision Challenge](https://lpcv.ai/competitions/2019)). 
- First place in the Visual Wake Words Challenge, TF-lite track, @CVPR 2019
- Third place in the Low Power Image Recognition Challenge (LPIRC), classification track, @CVPR 2019

## Performance
Without any proxy, directly and efficiently search neural network architectures on your target **task** and **hardware**! 

Now, proxylessnas is on [PyTorch Hub](https://pytorch.org/hub/pytorch_vision_proxylessnas/). You can load it with only two lines!

```python
target_platform = "proxyless_cpu" # proxyless_gpu, proxyless_mobile, proxyless_mobile14 are also avaliable.
model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=True)
```


![](https://file.lzhu.me/projects/proxylessNAS/figures/proxyless_nas.png)

<p align="center">
    <img src="https://file.lzhu.me/projects/proxylessNAS/figures/proxyless_bar.png" width="80%" />
</p>

![](https://file.lzhu.me/projects/proxylessNAS/figures/proxyless_compare.png)

<table>
<tr>
    <th> Mobile settings </th><th> GPU settings </th>
</tr>
<tr>
    <td>
    <img src="https://file.lzhu.me/projects/proxylessNAS/figures/proxyless_vs_mobilenet.png" width="100%" />
    </td>
<td>

| Model                | Top-1    | Top-5    | Latency | 
|----------------------|----------|----------|---------| 
| MobilenetV2          | 72.0     | 91.0     | 6.1ms   |
| ShufflenetV2(1.5)    | 72.6     | -        | 7.3ms   |
| ResNet-34            | 73.3     | 91.4     | 8.0ms   |
| MNasNet(our impl)    | 74.0     | 91.8     | 6.1ms   | 
| ProxylessNAS (GPU)   | 75.1     | 92.5     | 5.1ms   |

</td>
</tr>
<tr>
    <th> ProxylessNAS(Mobile) consistently outperforms MobileNetV2 under various latency settings.  </th>
    <th> ProxylessNAS(GPU) is 3.1% better than MobilenetV2 with 20% faster. </th>
</tr> 



</td></tr> </table>

<!-- <p align="center">
    <img src="https://file.lzhu.me/projects/proxylessNAS/figures/proxyless_vs_mobilenet.png" width="50%" />
    </br>
    <a> ProxylessNAS consistently outperforms MobileNetV2 under various latency settings. </a>
</p> -->

## Specialization

People used to deploy one model to all platforms, but this is not good. To fully exploit the efficiency, we should specialize architectures for each platform.

![](https://file.lzhu.me/projects/proxylessNAS/figures/specialization.jpg)
![](https://file.lzhu.me/projects/proxylessNAS/figures/specialized_archs.png)

We provide a [visualization](https://file.lzhu.me/projects/proxylessNAS/visualization.mp4) of search process. Please refer to our [paper](https://arxiv.org/abs/1812.00332) for more results.
 
# How to use / evaluate 
* Use
    ```python
    # pytorch 
    from proxyless_nas import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14, proxyless_cifar
    net = proxyless_cpu(pretrained=True) # Yes, we provide pre-trained models!
    ```
    ```python
    # tensorflow
    from proxyless_nas_tensorflow import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
    tf_net = proxyless_cpu(pretrained=True)
    ```

    If the above scripts failed to download, you download it manually from [Google Drive](https://drive.google.com/drive/folders/1qIaDsT95dKgrgaJk-KOMu6v9NLROv2tz?usp=sharing) and put them under  `$HOME/.torch/proxyless_nas/`.

* Evaluate

    `python eval.py --path 'Your path to imagent' --arch proxyless_cpu  # pytorch ImageNet`
    
    `python eval.py -d cifar10 # pytorch cifar10 `
    
    `python eval_tf.py --path 'Your path to imagent' --arch proxyless_cpu  # tensorflow`


## File structure

* [search](./search): code for neural architecture search.
* [training](./training): code for training searched models.
* [proxyless_nas_tensorflow](./proxyless_nas_tensorflow): pretrained models for tensorflow.
* [proxyless_nas](./proxyless_nas): pretrained models for PyTorch.

## Related work on automated model compression and acceleration:

[Once for All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791) (ICLR'20, [code](https://github.com/mit-han-lab/once-for-all))

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf) (ICLR’19)

[AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf) (ECCV’18)

[HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/pdf/1811.08886.pdf)  (CVPR’19, oral)
	
[Defenstive Quantization: When Efficiency Meets Robustness](https://openreview.net/pdf?id=ryetZ20ctX) (ICLR'19)

