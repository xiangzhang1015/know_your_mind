# adaptiveeegclassification
The reusable codes and datasets for paper: Adaptive Brain Signal Classification with Reinforced Attentive Convolutional Neural Networks.

1. The Dueling DQN codes including: 
  1.1 Environment code: env_spatial_autoregressive.py, which is the most valuable part. Please design your environment based on your classification task. It can not only used on EEG data classification but can be used for any policy optimization.
  1.2 Agent: RL_brain.py. 
  1.3 Main: run_this.py

2. In this paper we regard the combination of the convolutional mapping procedure and the classifier as the reward model. To make it more readable and help to design your own reward model, we separately write the reward model as a single python file: B_1DCNN.py. Please design your own reward model corresponding to your problem and insert it in the env_spatial_autoregressive.py. I left the insert position blank and added comments already.

3. All the download method is given for all the public and local datasets.

For more details, please contact 1015xzhang@gmail.com
