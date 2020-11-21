# Anomaly Detection-Based Unknown Face Presentation Attack Detection
This repository contains Pytorch implementation of our paper titled:  

[Anomaly Detection-Based Unknown Face Presentation Attack Detection](https://arxiv.org/abs/2007.05856) which was accepted at *IEEE International Joint Conference on Biometrics, 2020.*  

>In this method we train a convolutional neural network with only bonafide(real data). To model the attacked distribution we use a pseudo-negative gaussian whose mean is calculated using a running batch average. This is in accordance to the intuition that attacked data will lie in close proximity to the real data in high dimensional space.

Experiments are performed on multiple datasets, description of each is given in detail in the paper. The slides and video for the [presentation](https://drive.google.com/file/d/1qW149mPkgdrtU1ajmOjoRWN6k4j_bcZ1/view?usp=sharing) at the conference are also attached for which we won the Best Audience Choice Presentation Award!  

To run the code for say ReplayAttack dataset, one can use the following command:  
`python run.py --train_dataset_name replayattack --dev_dataset_name replayattack --exp_name *NAME_OF_YOUR_EXPERIMENT*`  

To edit the hyperparameters one can edit the `run_parameters` dictionary in `parameters.py`  

To test the model one can use the following command:  
`python test.py --train_dataset_name replayattack --dev_dataset_name replayattack --test_dataset_name replayattack --exp_name *NAME_OF_YOUR_EXPERIMENT*`  

We also give a small script to test visualize the features learned by the penultimate layer for classification using one-class method as described above. The script can be edited and run with the following command:  

`python tsne_demo.py --train_dataset_name replayattack --dev_dataset_name replayattack --test_dataset_name replayattack --exp_name tpc_one_github --vis_class 2`  

where `--vis_class 2` specifies the subject of whom you want to visualize the features. One can compare these features and see that as training progresses, the cluster of real data becomes more compact, and hence perform better spoof classification.  

If you have any questions about the code or the dataset, be sure to reach out by opening an issue. Thanks!

PS. Due to privacy concerns I cannot the share the dataset until the agreement is signed by the recieving party. Further, the code is always free to use. Enjoy!
