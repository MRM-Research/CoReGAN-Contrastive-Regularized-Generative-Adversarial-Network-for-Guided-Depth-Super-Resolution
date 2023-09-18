import argparse
from utils.tester import test

def main(args):
    config = {
        'hr_test_dir': args.hr_test_dir,
        'tar_test_dir': args.tar_test_dir,
        'batch_size': args.batch_size,
        'device':args.device,
        'model_path': args.model_path,
        'loss_weight': args.loss_weight,
        'encoder': args.encoder,
        'encoder_weights': args.encoder_weights,
        'gan_type': args.gan_type,
        
    }
    test(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_test_dir', type=str, required=False, default='/content/drive/MyDrive/nyuv2-dataset/test/images')
    parser.add_argument('--tar_test_dir', type=str, required=False, default='/content/drive/MyDrive/nyuv2-dataset/test/depths')
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--loss_weight', type=float, required=False, default=0.5)
    parser.add_argument('--gan_type', type=str, required=False, default='lsgan')
    parser.add_argument('--model_path', type=str, required=True, default='./best_model.pth')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, required=False, default='imagenet')
    arguments = parser.parse_args()
    main(arguments)
