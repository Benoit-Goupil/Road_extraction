import cv2
import os
import random
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_preparation import Road_Dataset, validation_transform

def visualise_path_images(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    fig = plt.figure() 

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Satellite Image') 

    fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title('Mask') 

    plt.show()

def visualise_dataset_images(dataset, i):
    img, mask = dataset[i]
    mask = np.transpose(mask,(1,2,0))
    mask = np.argmax(mask, axis=-1)
    
    img = np.transpose(img,(1,2,0))

    fig = plt.figure() 

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Satellite Image') 

    fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title('Mask') 

    plt.show()


def visualise_test_predictions(dataset, i, device, save=False, save_path=None, show=True, model_path=None):

    model = torch.load(model_path)
    model.eval()
    img, mask = dataset[i]
    with torch.no_grad():
        result = model(img.to(device).unsqueeze(0))
    result = result.detach().squeeze().cpu().numpy()
    result = np.transpose(result,(1,2,0))
    result = np.argmax(result, axis=-1)

    mask = np.transpose(mask,(1,2,0))
    mask = np.argmax(mask, axis=-1)
    
    img = np.transpose(img,(1,2,0))
    fig = plt.figure() 

    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Satellite Image') 

    fig.add_subplot(1, 3, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title('Real Mask') 

    fig.add_subplot(1, 3, 3)
    plt.imshow(result)
    plt.axis('off')
    plt.title('Predicted Mask') 

    if show: plt.show()
    if save: plt.savefig(os.path.join(save_path, 'test_prediction_{}'.format(i)), dpi=1000)

    plt.close()



if __name__ == '__main__':

    CURRENT_PATH = os.path.curdir
    DATA_PATH = os.path.join(CURRENT_PATH,"data")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_df = pd.read_csv(os.path.join(DATA_PATH, 'metadata.csv'))

    
    test_dataset = Road_Dataset(meta_df=meta_df, transformation=validation_transform, data_path=DATA_PATH, split='test')

    for i in range(len(test_dataset)):
       visualise_test_predictions(
                                    dataset=test_dataset, 
                                    i=i, 
                                    device=DEVICE, 
                                    show=False, 
                                    save=True, 
                                    save_path=os.path.join(CURRENT_PATH, 'test_predictions'), 
                                    model_path=os.path.join(CURRENT_PATH, 'model', 'best_model_49.pth')
                                    )