# Road extraction project

The objective of this project is to segment the roads present on satellite images. 
The predictions are made by a Unet network with a resnet50 backbone pre-trained on imagenet.

## Sample predictions
Below are some examples predicted by the network on the test set

![test_prediction_7](https://user-images.githubusercontent.com/73244633/159383975-1b246e40-e4bc-40bc-9064-25c9083ed99c.png)
![test_prediction_10](https://user-images.githubusercontent.com/73244633/159383978-3ec0cccd-7843-4bb3-b260-e11920e2ae18.png)
![test_prediction_12](https://user-images.githubusercontent.com/73244633/159383986-d26f9047-b975-4083-9543-3cd232d86a4e.png)
![test_prediction_24](https://user-images.githubusercontent.com/73244633/159383989-4f0b1591-7d66-450c-8899-e083a4999370.png)

## Usage

- Run `training.py` to train the Unet network
- Run `visualise.py` to visualise the predictions
