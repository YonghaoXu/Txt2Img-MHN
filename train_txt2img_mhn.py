import os
import time
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from tools.model import VQVAE, VQGanVAE, Txt2ImgMHN
from torch.optim.lr_scheduler import ExponentialLR
from tools.utils import *
from dataset.text_im_dataset  import TextImageDataset

def main(args):    
    tokenizer = YttmTokenizer()  
    if args.vae_type==0:
        vae = VQVAE(image_size=args.crop_size)            
        saved_state_dict = torch.load(args.vqvae_path)
        vae.load_state_dict(saved_state_dict)
        save_path_prefix = './mhn_vqvae_checkpoint/'+time.strftime('%m%d_%H%M', time.localtime(time.time()))
        model_name = os.path.join(save_path_prefix,'mhn_vqvae.pth')
    elif args.vae_type==1:
        vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path)
        save_path_prefix = './mhn_vqgan_checkpoint/'+time.strftime('%m%d_%H%M', time.localtime(time.time()))
        model_name = os.path.join(save_path_prefix,'mhn_vqgan.pth')

    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)      
    im_path = os.path.join(save_path_prefix,'im_log/') 
    if os.path.exists(im_path)==False:
        os.makedirs(im_path)
    
    train_set = TextImageDataset(
        args.data_dir,
        args.train_list,
        text_len=args.text_seq_len,
        image_size=args.crop_size,
        tokenizer=tokenizer
    )   
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = Txt2ImgMHN(vae=vae, num_text_tokens=tokenizer.vocab_size, num_prototype=args.num_prototype) 
    model = torch.nn.DataParallel(model).cuda()

    # optimizer
    model_optimizer = torch.optim.Adam(get_trainable_params(model),lr=args.lr)
    model_schedule = ExponentialLR(optimizer = model_optimizer, gamma = args.lr_decay_rate)

    summary = TensorboardSummary(save_path_prefix)
    writer = summary.create_summary()
    global_step = 0
    for epoch in range(args.num_epochs):
        ce_loss = 0.0
        tbar = tqdm(train_loader)
        for index, (text, images) in enumerate((tbar)):
            text, images = map(lambda t: t.cuda(), (text, images))     
            loss = model(text, images, return_loss=True).mean() 
            
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            ce_loss += loss.item()            
            tbar.set_description('epoch: %d/%d cross entropy loss: %.3f' % (epoch+1, args.num_epochs, ce_loss / (index + 1)))
            writer.add_scalar('ce_loss_iter', loss.item(), global_step)
            lr = model_schedule.get_last_lr()[0]
            writer.add_scalar('lr_iter', lr, global_step)

            if global_step % 100 == 0:
                sample_text = text[:1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)         
                image = model.module.generate_images(text[:1], filter_thres=0.9)  # topk sampling at 0.9                
                summary.visualize_gen_image(writer,image,decoded_text,global_step,im_path,args.vae_type)
            global_step += 1
           
        torch.save(model.state_dict(), model_name)        
        model_schedule.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--vae_type', type=int, default=1, help='vae_type==0: vqvae; vae_type==1: vqgan')
    parser.add_argument('--data_dir', type=str, default='/iarai/home/yonghao.xu/Data/RSICD/', help='dataset path.')   
    parser.add_argument("--train_list", type=str, default='./dataset/RSICD_train.txt', help="training list file.")
    parser.add_argument('--vqvae_path', type=str, default='./vae.pth')   
    parser.add_argument('--vqgan_model_path', type=str, default='./last.ckpt')    
    parser.add_argument('--vqgan_config_path', type=str, default='./2022-07-19T16-49-53-project.yaml')   
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=4.5e-3)
    parser.add_argument('--lr_decay_rate', type=float, default=0.999)
    parser.add_argument('--loss_img_weight', type=float, default=1)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=224)
    parser.add_argument('--num_prototype', type=int, default=1000)
    parser.add_argument('--text_seq_len', type=int, default=40)
    main(parser.parse_args())