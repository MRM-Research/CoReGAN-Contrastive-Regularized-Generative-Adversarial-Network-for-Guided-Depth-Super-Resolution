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

    model = model.load_state_dict(torch.load(model_path))
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
        wandb.log({'epoch':i+1,
                    'test_loss':test_logs['LOSS'],
                    'test_ssim':test_logs['SSIM'],
                    'test_psnr':test_logs['PSNR'],
                    'test_mse':test_logs['MSE'],
                    'test_mae':test_logs['MAE'],
                    })
        #do something (save model, change lr, etc.)
        wandb.config.update({'min_test_mae':min_test_mae,'min_test_mse':min_test_mse, 'max_test_ssim':max_test_ssim, 'max_test_psnr':max_test_psnr}, allow_val_change=True)
        torch.save(model.state_dict(), './best_test_model.pth')
        print('Test model saved!')
    print(f'max test ssim: {max_test_ssim} max test psnr: {max_test_psnr} min test mse: {min_test_mse} min test mae: {min_test_mae}')

def test_model(configs):
    test(configs['hr_test_dir'],
         configs['tar_test_dir'],configs['batch_size'], configs['encoder'],
         configs['encoder_weights'], configs['device'],
         configs['loss_weight'],  configs['gan_type'], configs['model_path'])