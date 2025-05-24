# Part B: Fine-Tuning a Pre-Trained Model (ResNet50)
# Overview


In this part, I fine-tuned a pre-trained ResNet50 model using the iNaturalist dataset. I loaded the pre-trained weights from ImageNet and adjusted the model for the smaller number of classes in iNaturalist.

Training a deep learning model from scratch on a large training dataset is time-consuming due to the vast number of parameters that need to be optimized.

# Fine-Tuning Strategies

Freezing all layers except the last layer.


Freezing up to k layers and fine-tuning the rest.


Freezing last fully connected layers and training the conv layers.


# Result Analysis


Explored freezing 70%, 80%, and 90% of layers. This freezed model giving good validation accuracy.


Explored Freezing all layers except the last layer giving good validation accuracy.

Recorded validation accuracies for different fine-tuning strategies and layer percentages.
