import torch
import os

import segmentation_models_pytorch as smp
import pandas as pd

from torch.utils.data import DataLoader

from dataset_preparation import Road_Dataset
from dataset_preparation import train_transform, validation_transform


if __name__ == '__main__':

    CURRENT_PATH = os.path.curdir
    DATA_PATH = os.path.join(CURRENT_PATH,"data")



    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=2, 
        activation='sigmoid',
    )

    model = torch.load(os.path.join(CURRENT_PATH, 'model', 'best_model_29.pth'))
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')

    meta_df = pd.read_csv(os.path.join(DATA_PATH, 'metadata.csv'))

    train_dataset = Road_Dataset(meta_df=meta_df, transformation=train_transform, data_path=DATA_PATH, split='train')
    validation_dataset = Road_Dataset(meta_df=meta_df, transformation=validation_transform, data_path=DATA_PATH, split='val')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=8)
    #TRAINING 

    TRAINING = True
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    loss = smp.utils.losses.DiceLoss()

    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00008)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    validation_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )


    for i in range(30, EPOCHS+30):
        print('Epoch : ', i)
        train_epoch.run(train_loader)
        validation_epoch.run(validation_loader)
        torch.save(model, os.path.join(CURRENT_PATH, 'model', 'best_model_{}.pth'.format(i)))