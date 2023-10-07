import os
import time
import torch
import argparse
from tqdm import tqdm
from math import sqrt,exp
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.text_im_dataset import ImageFolder
from tools.model import VQVAE
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ExponentialLR
from tools.utils import TensorboardSummary

def main(args):
    save_path_prefix = args.save_path_prefix+time.strftime('%m%d_%H%M', time.localtime(time.time()))
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)

    composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size, args.crop_size)),            
            transforms.ToTensor()])

    train_set = ImageFolder(args.data_dir, args.train_list, composed_transforms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    vae = VQVAE(image_size = args.crop_size)    
    vae = vae.cuda()

    vae_optimizer = torch.optim.Adam(vae.parameters(),lr=args.lr)
    vae_schedule = ExponentialLR(optimizer = vae_optimizer, gamma = args.lr_decay_rate)

    model_name = os.path.join(save_path_prefix,'vae.pth')
        
    summary = TensorboardSummary(save_path_prefix)
    writer = summary.create_summary()

    # starting temperature
    temp = 1.0
    global_step = 0
    for epoch in range(args.num_epochs):
        vae_loss = 0.0
        tbar = tqdm(train_loader)
        for index, images in enumerate(tbar):
            images = images.cuda()

            loss, recons = vae(images,temp=temp)
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            
            vae_loss += loss.item()            
            tbar.set_description('epoch: %d/%d vae_loss: %.3f' % (epoch+1, args.num_epochs, vae_loss / (index + 1)))
            writer.add_scalar('vae_loss_iter', loss.item(), global_step)
            lr = vae_schedule.get_last_lr()[0]
            writer.add_scalar('lr_iter', lr, global_step)

            if global_step % 100 == 0:                
                with torch.no_grad():
                    codes = vae.get_codebook_indices(images[:args.num_images_display])
                    hard_recons = vae.decode(codes)

                images, recons = map(lambda t: t[:args.num_images_display], (images, recons))
                images, recons, hard_recons = map(lambda t: t.detach().cpu(), (images, recons, hard_recons))
                images = make_grid(images.float(), nrow = int(sqrt(args.num_images_display)), normalize = True, range = (0, 1))
                recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(args.num_images_display)), normalize = True, range = (-1, 1)), (recons, hard_recons))
                
                summary.visualize_vae_image(writer,images,recons,hard_recons,global_step)

                # temperature anneal
                temp = max(temp * exp(-args.anneal_rate * global_step), args.temp_min)
          
            global_step += 1
           
        torch.save(vae.state_dict(), model_name)        
        vae_schedule.step()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--save_path_prefix', type=str, default='./vqvae_checkpoint/')
    parser.add_argument('--data_dir', type=str, default='/iarai/home/yonghao.xu/Data/RSICD/',help='dataset path.')   
    parser.add_argument("--train_list", type=str, default='./dataset/RSICD_train.txt',help="training list file.")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_rate', type=float, default=0.996)
    parser.add_argument('--anneal_rate', type=float, default=1e-6)
    parser.add_argument('--temp_min', type=float, default=0.5)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_images_display', type=int, default=4)
    main(parser.parse_args())
    
