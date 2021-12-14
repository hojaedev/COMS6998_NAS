  
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

Implementation Details: 

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
![](https://user-images.githubusercontent.com/44733338/146056535-79163d0d-88a7-4596-b8c6-6782ad1362d1.png)  |  ![](https://user-images.githubusercontent.com/44733338/146055820-cedcccc1-3462-4e4e-a91d-4e11fed21255.png)
Basic NAS Search Space           |  Diverse NAS Search Space  
![](https://user-images.githubusercontent.com/44733338/146056162-86c0b87a-7160-434b-8db7-1cff54b5fe56.png)  |  ![](https://user-images.githubusercontent.com/44733338/146056346-3388528b-d497-4a6a-84ec-dd76779de58b.png)

Performance comparison of models:
![image](https://user-images.githubusercontent.com/44733338/146057042-684c278a-72d5-4264-9285-3fe2df32704b.png)

Discovered architectures:

Vanilla LSTM            |  Vanilla GRU
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/44733338/146054520-f3f5c2ee-9e56-41be-87d6-351985edf4e8.png)  |  ![](https://user-images.githubusercontent.com/44733338/146054560-b434b52c-7440-4eae-bca6-34c14496198a.png)
Basic NAS Search Space           |  Diverse NAS Search Space  
![](https://user-images.githubusercontent.com/44733338/146055206-97579ac4-740e-4ce1-8990-b3f86c696bb5.png)  |  ![](https://user-images.githubusercontent.com/44733338/146055212-7be699b5-37e5-4c90-8de3-d7ad7a2f82c4.png)

Expanding TSLA stock trained model to other stocks

![image](https://user-images.githubusercontent.com/44733338/146055391-9433e84d-effb-4e99-bb88-262d223263ed.png)

## Observations

1) From our experiments, it can be observed that simpler models perform better on stock price prediction than others.
2) While NAS with diverse search space has yielded the greatest accuracy on the TESLA dataset, its training time of 9 hours is significantly higher than a vanilla GRU (~40 secs).
3) Although the accuracy of NAS w/ diverse search space doesn't suffer significantly when testing other datasets, it's clear that there has been some overfitting since NAS w/ basic search space and simple GRU model suffer less. This might be due to the high exploitation rate we have selected for Run #2, the fact that we train the best model 100 epochs more without early stopping, and/or the best architecture for this task happened to not have any dropout layers.
4) Overall, our implementation of NAS works well in generating a model that performs well given a task; however, simpler models do almost as good and have less training cost.

## Description of Repository

The code was run primarily on Google Colab; all of the colab files that were run are in the colab_files folder.

The colab file which we used to train and compare models that were found in our experiments is in final_results and named Final_Results.ipynb.

The entire code is in train.py. In train, there are:
1) Script to generate train and test data from "data/tsla.us.txt".
2) StateSpace class (which provides utility functions for holding "states" / "actions" that the controller must use to train and predict.
3) Controller class(utility class to manage the RNN Controller)
4) model_fn function (which is used to generate our neural networks)
5) NetworkManager class (helper class to manage the generation of subnetwork training given a dataset)
6) Script to perform NAS training by utilizing the classes / functions described above.

## Example Commands to execute the code

The code can be executed simply by running
```
python train.py --num_layers=8 ...
```
The arguments and their default values are:

```
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--max_trials', type=int, default=300)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--child_batchsize', type=int, default=128)
parser.add_argument('--exploration', type=float, default=0.8)
parser.add_argument('--regularization', type=float, default=1e-3)
parser.add_argument('--controller_cells', type=int, default=32)
parser.add_argument('--embedding_dim', type=int, default=20)
parser.add_argument('--accuracy_beta', type=float, default=0.8)
parser.add_argument('--clip_rewards', type=float, default=0.0)
parser.add_argument('--restore_controller', type=bool, default=True)
parser.add_argument('--sequence_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_features', type=int, default=1)
```

The loss, reward, and all of the parameter values for the neural networks experimented on will be written to a file called train_history.csv. The time taken for the NAS to be completed would be written in train_results.txt
