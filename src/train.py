import argparse
from utils.trainer import train_model
import wandb

def main(args):
    config = {
        'hr_dir': args.hr_dir,
        'tar_dir': args.tar_dir,
        'hr_val_dir':args.hr_val_dir,
        'tar_val_dir':args.tar_val_dir,
        'hr_test_dir':args.hr_test_dir,
        'tar_test_dir':args.tar_test_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device':args.device,
        'encoder': args.encoder,
        'encoder_weights': args.encoder_weights,
        'lr': args.lr,
        'beta': args.beta,
        'loss_weight_loss': args.loss_weight_loss,
        'loss_weight_gan': args.loss_weight_gan,
        'gan_type': args.gan_type
    }
    wandb.init(project="DepthMapSR", entity="kasliwal17",
               config={'beta':args.beta, 
                       'lr':args.lr, 
                       'max_ssim':0,
                       'max_psnr':0, 
                       'min_mse': 1e6, 
                       'min_mae': 1e6, 
                       'loss_weight': args.loss_weight, 
                       'encoder':args.encoder,
                       'BS': args.batch_size,
                       'gan_type': args.gan_type
                       }, allow_val_change=True)
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=False, default='/kaggle/input/nyuv2-labled-dataset/images')
    parser.add_argument('--tar_dir', type=str, required=False, default='/kaggle/input/nyuv2-labled-dataset/depths')
    parser.add_argument('--hr_val_dir', type=str, required=False, default='')
    parser.add_argument('--tar_val_dir', type=str, required=False, default='')
    parser.add_argument('--hr_test_dir', type=str, required=False, default='')
    parser.add_argument('--tar_test_dir', type=str, required=False, default='')
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=250)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, required=False, default='imagenet')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--beta', type=float, required=False, default=1)
    parser.add_argument('--loss_weight_loss', type=float, required=False, default=0.5)
    parser.add_argument('--loss_weight_gan', type=float, required=False, default=2000)
    parser.add_argument('--gan_type', type=str, required=False, default='standard')
    arguments = parser.parse_args()
    main(arguments)
