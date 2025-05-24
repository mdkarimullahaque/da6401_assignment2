 Here we are implementing a CNN model on the  [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) from scratch and tuned the hyperparameters to achieve optimal performance. I utilized Python along with required packages from PyTorch and Torchvision.

 # Model Architecture
Convolutional Layers: 5 layers with varying kernel sizes and filter organizations.

Batch normalisation in convolution and dense layers

Activation: ReLU, GELU, SiLU, Mish (applied after each convolution layer).

Max-Pooling: Applied after each activation layer.

Dropout in convolution and dense layers

Dense Layer: 1 dense layer with 128 or 256 neurons.

Output Layer: 10 neurons for 10 classes in the iNaturalist dataset.

# Hyperparameters 
Kernel Size (Size of Filters): [[3,3,3,3,3], [3,5,5,7,7], [7,7,5,5,3]]


Dropout: [0.2, 0.3]


Activation Function: ['ReLU', 'GELU', 'SiLU', 'Mish']


Batch Normalization: [Yes, No]


Filter Organization: [[32,32,32,32,32], [128, 128, 64, 64,32], [32, 64,128,256,512]]



Data Augmentation:[Yes, No]

Number of nodes in dense layer : [128, 256]


# Training Process


Split training data into 80:20 for training and validation.


Use bayesian sweep feature from wandb for hyperparameter tuning.


Selected best hyperparameter configuration based on validation accuracy.


# Best Hyperparameters


Kernel Size: [3,3,3,3,3]


Dropout: 0.2


Activation Function: Mish


Batch Normalization: No


Filter Organization: [32,32,32,32,32]


Data Augmentation: No


Neurons in Dense Layer: 256


# Model Evaluation
Tested the best model on the test data.


Reported accuracy on the test set.
