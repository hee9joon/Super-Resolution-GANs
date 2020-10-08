import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--sort', type=str, default='SRGAN', choices=['SRGAN', 'ESRGAN'])
parser.add_argument('--disc_type', type=str, default='conv', choices=['fcn', 'conv', 'patch'])

parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size for train. Recommend 8 for SRGAN and 2 for ESRGAN.')
parser.add_argument('--val_batch_size', type=int, default=1, help='mini-batch size for validation')
parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--crop_size', type=int, default=128, help='image crop size')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')

parser.add_argument('--lambda_adversarial', type=float, default=1e-3, help='lambda for Adversarial Loss used for SRGAN')
parser.add_argument('--lambda_tv', type=float, default=2e-8, help='lambda for Total Variation Loss used for SRGAN')

parser.add_argument('--lambda_content', type=float, default=1, help='lambda for Content Loss used for ESRGAN')
parser.add_argument('--lambda_bce', type=float, default=5e-3, help='lambda for Binary Cross Entropy Loss used for ESRGAN')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for both discriminator and generator networks')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=100, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, options: [Step, Plateau, Cosine]')

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--inference_path', type=str, default='./results/inference/', help='inference path')

parser.add_argument('--num_epochs', type=int, default=150, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=10, help='save model weights for every default epoch')

config = parser.parse_args()

if __name__ == '__main__':
    print(config)