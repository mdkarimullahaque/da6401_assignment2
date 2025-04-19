# DA6401 - Assignment 2

MD Karimulla Haque MA23C021

In Part A and Part B of this assignment will build and experiment with CNN based image classifiers using a subset of the [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip).

# Part A
## Question 1:
CNN Network code block - 
```python
    def _create_conv_layers(self):
        layers = []
        in_channels = self.input_shape[0]
        out_channels = self.num_filters

        if self.filter_organization == 'double':
            out_channels_list = [out_channels] + [out_channels * 2**i for i in range(1, 5)]
        elif self.filter_organization == 'halve':
            out_channels_list = [out_channels * 2**i for i in range(4, -1, -1)]
        else:
            out_channels_list = [out_channels] * 5

        for out_channels in out_channels_list:
            layers.append(nn.Conv2d(in_channels, out_channels, self.filter_size, padding='same'))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(self.get_activation())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _create_dense_layers(self):
        layers = [
            nn.Flatten(),
            nn.Linear(self.conv_output_size(), self.dense_neurons),
            self.get_activation(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dense_neurons, self.num_classes)
        ]
        return nn.Sequential(*layers)

```
## Question 2:
For wandb sweep configuration:

```python
sweep_config = {
    'method': 'random',  # Specify the search method
    'parameters': {
        'num_filters': {'values': [32, 64, 128]},
        'activation': {'values': ['relu', 'gelu', 'silu', 'mish']},
        'data_augmentation': {'values': [True, False]},
        'batch_norm': {'values': [True, False]},
        'dropout': {'values': [0.2, 0.3]},
        'filter_organization': {'values': ['same', 'double', 'halve']}
    }
}

```

## Question 4:
Best model configuration and stat:

| dropout | filter_organization | num_filters | epoch | train_accuracy_epoch | train_accuracy_step | train_loss_epoch | train_loss_step | trainer/global_step | val_accuracy | val_loss | Test_Accuracy|
|---------|---------------------|-------------|-------|----------------------|---------------------|------------------|-----------------|---------------------|--------------|----------|----------|
| 0.3     | same                | 32          | 9     | 0.4874               | 0.5938              | 1.47             | 1.423           | 3129                | 0.426        | 1.673    | 0.3969|

# Part B

## Question 2:
Strategies for Fine-Tuning

Freezing all layers except the last layer: Freeze the weights of all layers except the last fully connected layer and train only the last layer.

Freezing upto $k$ layers: Freeze the weights of the initial $k$ layers (e.g., all convolutional layers) and train the later layers.

Fine-Tuning All Layers: Unfreeze all layers and fine-tune the entire model.




