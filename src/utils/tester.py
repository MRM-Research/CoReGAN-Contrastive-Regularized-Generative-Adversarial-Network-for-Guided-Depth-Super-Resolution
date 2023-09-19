import wandb
from .train_utils import ValidEpoch
from .dataloader import Dataset
from .transformations import  resize
from .model import Unet
from torch.utils.data import DataLoader
from .model import Discriminator
import torch

def test(hr_test_dir,
        tar_test_dir,
        batch_size,
        encoder='resnet34', 
        encoder_weights='imagenet', 
        device='cuda',
        loss_weight=0.5,
        gan_type='standard',
        model_path='./best_model.pth',
        ):

    activation = 'tanh' 
    # create segmentation model with pretrained encoder
    model = Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        encoder_depth = 5,
        classes=1, 
        activation=activation,
        fusion=True,
        contrastive=True,
    )

    disc = Discriminator().to(device)

    test_dataset = Dataset(
        hr_test_dir,
        tar_test_dir,
        augmentation=None, 
        preprocessing=True,
        resize = resize()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_epoch = ValidEpoch(
        model=model,
        discriminator=disc, 
        loss_weight=loss_weight,
        device=device,
        verbose=True,
        gan_type=gan_type,
        batch_size=batch_size
    )

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()

    min_test_mse = 1e4
    min_test_mae = 1e4
    max_test_ssim = 0
    max_test_psnr = 0
    for i in range(0, 1):
        
        print('\nEpoch: {}'.format(i))
        test_logs = test_epoch.run(test_loader)
        
        print(test_logs)
    print(f"max test ssim: {test_logs['SSIM']} max test psnr: {test_logs['PSNR']} min test mse: {test_logs['MSE']} min test mae: {test_logs['MAE']}")

def test_model(configs):
    test(configs['hr_test_dir'],
         configs['tar_test_dir'],configs['batch_size'], configs['encoder'],
         configs['encoder_weights'], configs['device'],
         configs['loss_weight'],  configs['gan_type'], configs['model_path'])