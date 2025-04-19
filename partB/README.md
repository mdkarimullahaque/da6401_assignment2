## Question 2:
Strategies for Fine-Tuning

Freezing all layers except the last layer: Freeze the weights of all layers except the last fully connected layer and train only the last layer.

Freezing upto $k$ layers: Freeze the weights of the initial $k$ layers (e.g., all convolutional layers) and train the later layers.

Fine-Tuning All Layers: Unfreeze all layers and fine-tune the entire model.
