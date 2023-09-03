import argparse
from .utils.trainer import train_model
import wandb

def main(args):
    config = {
        'hr_dir': args.hr_dir,
        'th_dir': args.th_dir,
        'tar_dir': args.tar_dir,
        'hr_val_dir': args.hr_val_dir,
        'th_val_dir': args.th_val_dir,
        'tar_val_dir': args.tar_val_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device':args.device,
        'encoder': args.encoder,
        'encoder_weights': args.encoder_weights,
        'lr': args.lr,
        'beta': args.beta,
        'loss_weight': args.loss_weight
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
                       'BS': args.batch_size
                       }, allow_val_change=True)
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=False, default='./Dataset/train_val/training/train_VIS_HR')
    parser.add_argument('--th_dir', type=str, required=False, default='./Dataset/train_val/training/train_input_THER_LR_bicubic/X8')
    parser.add_argument('--tar_dir', type=str, required=False, default='./Dataset/train_val/training/train_output_gt_THER_HR')
    parser.add_argument('--hr_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_VIS_HR')
    parser.add_argument('--th_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_input_THER_LR_bicubic/X8')
    parser.add_argument('--tar_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_output_gt_THER_HR')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--epochs', type=int, required=False, default=250)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, required=False, default='imagenet')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--beta', type=float, required=False, default=1)
    parser.add_argument('--loss_weight', type=float, required=False, default=0.5)
    arguments = parser.parse_args()
    main(arguments)
