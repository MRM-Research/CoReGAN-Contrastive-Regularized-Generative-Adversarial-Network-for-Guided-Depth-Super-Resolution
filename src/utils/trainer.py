import wandb
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss #custom_lossv
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing,resize
from .model import Unet
import torch
from torch.utils.data import DataLoader
def train(epochs, 
          batch_size, 
          hr_dir, 
          tar_dir, 
          hr_val_dir, 
          tar_val_dir, 
          encoder='resnet34', 
          encoder_weights='imagenet', 
          device='cuda', 
          lr=1e-4,
          beta=1, 
          loss_weight=0.5
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

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = Dataset(
        hr_dir,
        tar_dir,
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        resize = resize()
    )
    valid_dataset = Dataset(
        hr_val_dir,
        tar_val_dir,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        resize = resize()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)#, drop_last=True)

    loss = custom_loss(batch_size, beta=beta, loss_weight=loss_weight)

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),
    ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        optimizer=optimizer,
        device=device,
        verbose=True,
        contrastive=True
    )
    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        device=device,
        verbose=True,
        contrastive=True
    )

    min_mse = 0
    min_mae = 0
    max_ssim = 0
    max_psnr = 0
    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        print(train_logs)
        wandb.log({'epoch':i+1,
                    't_loss':train_logs['custom_loss'],
                    'v_loss':valid_logs['custom_lossv'],
                    't_ssim':train_logs['ssim'],
                    'v_ssim':valid_logs['ssim'],
                    't_psnr':train_logs['psnr'],
                    'v_psnr':valid_logs['psnr'],
                    't_mse':train_logs['mse'],
                    'v_mse':valid_logs['mse'],
                    't_mae':train_logs['mae'],
                    'v_mae':valid_logs['mae']
                    })
        # do something (save model, change lr, etc.)
        if min_mse >= valid_logs['mse']:
            min_mse = valid_logs['mse']
            min_mae = valid_logs['mae']
            max_psnr = valid_logs['psnr']
            max_ssim = valid_logs['ssim']
            wandb.config.update({'min_mae':min_mae,'min_mse':min_mse, 'max_ssim':max_ssim, 'max_psnr':max_psnr}, allow_val_change=True)
            torch.save(model.state_dict(), './best_model.pth')
            print('Model saved!')
    print(f'max ssim: {max_ssim} max psnr: {max_psnr} min mse: {min_mse} min mae: {min_mae}')

def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['hr_dir'],
         configs['tar_dir'], configs['th_dir'], configs['hr_val_dir'],
         configs['tar_val_dir'], configs['th_val_dir'], configs['encoder'],
         configs['encoder_weights'], configs['device'], configs['lr'], configs['beta'], configs['loss_weight'])
    
# 2. In metrics, change back to 0, 1 from -1, 1 , rest remains the same - dont clamp --> aryan
# 5. change wand.config.update and init - init mse and mae as big value
# 6. custom loss v in val epoch
         