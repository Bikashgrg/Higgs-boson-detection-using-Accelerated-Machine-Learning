# Higgs-boson-detection-using-Accelerated-Machine-Learning 

The Random Forest and XGBoost model has been used in this project to train datasets in both CPU and GPU to classify Higgs boson signal from the background signal based on the dataset provided by our professors Dr. Carl Chalmers and Dr. Paul Fergus. 

## The Data
The provided dataset contains almost 11 million simulated collison occurences. The data was generated using Monte Carlo simulations. A list of 28 features (21 low-level and seven high-level) and the target feature (1 for signal and 0 for background) is present in the given dataset. Some of features name are: **lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi** and many more which can be seen in the given link (https://archive.ics.uci.edu/ml/datasets/HIGGS). 

The accelerator's particle detectors evaluate kinematic features in the first 21 characteristics (columns 2 to 22). The final seven traits, which are high-level features (columns 23-29) and are functions of the first 21 qualities, were devised by physicists to distinguish between the two groups.

Since, I am going to use GPU to perform this project I am going to use GPU supported libraries and framework such as RAPIDS that contains cuml, cudf. The reason I am using RAPIDS is that it will help me easily accelerate the workflows without going into the details of CUDA programming. Also, I am going to run the given models in CPU for the comparison.

## Benchmark comparison between CPU and GPU

### Random Forest (CPU)

![RF_CPU](https://user-images.githubusercontent.com/29011734/164761111-fbcb036f-8f86-467e-bc6e-24ee4a922655.png)

The reason for using lower value of parameters for training the RF and XGBoost model in CPU rather than GPU is that using GPU parameters in CPU took way too long, as shown by the training time of the third parameter in the above image, where I used one of the GPU parameters.

### Random Forest (GPU)

![RF_GPU](https://user-images.githubusercontent.com/29011734/164761230-5459926c-71c2-4d60-826d-d3493995b56c.png)

### Final conclusion (RF)

The benchmark for random forest shows that training time on GPU is significantly less than on CPU. Interestingly, the only GPU parameter values I ran on CPU show that accuracy is similar or slightly better in CPU (74.05%) compared to GPU (74.02%), but GPU crushes CPU on training time. For the same selected instances and parameter values where *n_estimator = 150* and *max_depth = 20* (with selected 21 features but all features with values less than 3), it can be seen that the GPU is almost 177 times faster than the CPU, saving a significant amount of time that can be used to improve the model.

### XGBoost (CPU)

![XG_CPU](https://user-images.githubusercontent.com/29011734/164761341-5eb3b24f-f4b1-4241-94ea-9a4add5f8298.png)

### XGBoost (GPU)

![XG_GPU](https://user-images.githubusercontent.com/29011734/164761371-9821b593-fc36-4f79-958f-a7cd728e644d.png)

### Final conclusion (XGBoost)

Even with the XGBoost model, the training time on GPU is significantly faster than on CPU. In this model, I ran two parameters that are similar to each other, and once again, CPU (83.45%, 82.90%) slightly outperforms GPU (83.30%, 82.89%) in terms of accuracy. However, GPU outperforms CPU in training time, as GPU was more than 300 times faster on both occasions. It can be shown that the GPU in the XGBoost model is more time efficient than the RF model, which was 177 times faster.

If you want to have a greater accuracy, even if it is a slight one when training time is not a concern, then go with CPU because it can be seen in both models above that when the same parameter is utilised for CPU and GPU, CPU performs slightly better in terms of accuracy.

### Reasons for the selected model parameters

To determine the ideal parameters, I started with one parameter, such as the number of trees in RF, and gradually increased the number of trees to check if there was any gain in accuracy. Then, after choosing the optimum number of trees, I begin with another parameter starting with low values to determine which value helps to raise the accuracy, and so on.

I attempted and ran many various parameters, but only a few perimeters comparisons are presented in the comparison table or image above since the accuracy that other parameters provided when I used them was not good enough or better than the ones I selected. Also, the table would be excessively long as some of the parameters made accuracy poorer, such as 57% or 60%, which wasn't worth it.

## Thank you
