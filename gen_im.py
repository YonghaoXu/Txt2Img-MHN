import os
import time
import torch
import argparse
from tqdm import tqdm
from tools.utils import *
from tools.model import Txt2ImgMHN,VQVAE,VQGanVAE
from torchvision.utils import save_image
from dataset.text_im_dataset  import TextDataset
from torch.utils.data import DataLoader

def main(args):    
    tokenizer = YttmTokenizer()      
    if args.vae_type==0:
        vae = VQVAE()            
        saved_state_dict = torch.load(args.vqvae_path)
        vae.load_state_dict(saved_state_dict)
        save_path_prefix = './gen_vqvae/'+time.strftime('%m%d_%H%M', time.localtime(time.time()))
        model = Txt2ImgMHN(vae=vae, num_text_tokens=tokenizer.vocab_size, num_prototype=args.num_prototype)  
        saved_state_dict = torch.load(args.mhn_vqvae_path)
    elif args.vae_type==1:
        vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path)
        save_path_prefix = './gen_vqgan/'+time.strftime('%m%d_%H%M', time.localtime(time.time()))
        model = Txt2ImgMHN(vae=vae, num_text_tokens=tokenizer.vocab_size, num_prototype=args.num_prototype)  
        saved_state_dict = torch.load(args.mhn_vqgan_path)

    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)         
    
    new_params = model.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
   
    model.load_state_dict(new_params)
    model = model.cuda()
    
    test_set = TextDataset(
        args.data_dir,
        args.test_list,
        tokenizer=tokenizer
    )   
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

    tbar = tqdm(test_loader)
    for index, (text, name) in enumerate(tbar):    
        text = text.cuda()
        text = text.repeat(args.num_gen_per_image,1)     
        tbar.set_description('Generating from the %d/%d text' % (index+1, len(test_loader)))
        output = model.generate_images(text, filter_thres = args.filter_thres).cpu()   
        for i in range(args.num_gen_per_image):
            save_image(output[i,:,:,:], os.path.join(save_path_prefix, name[0]+'_'+str(i)+'.png'), normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--vae_type', type=int, default=1, help='vae_type==0: vqvae; vae_type==1: vqgan')
    parser.add_argument('--data_dir', type=str, default='/iarai/home/yonghao.xu/Data/RSICD/',help='dataset path.')   
    parser.add_argument("--test_list", type=str, default='./dataset/RSICD_test.txt',help="test list file.")
    parser.add_argument('--vqvae_path', type=str, default='./vae.pth')   
    parser.add_argument('--vqgan_model_path', type=str, default='./last.ckpt')    
    parser.add_argument('--vqgan_config_path', type=str, default='./2022-07-19T16-49-53-project.yaml')   
    parser.add_argument('--mhn_vqvae_path', type=str, default='./mhn_vqvae.pth')   
    parser.add_argument('--mhn_vqgan_path', type=str, default='./mhn_vqgan.pth')   
    parser.add_argument('--num_gen_per_image', type=int, default=10)
    parser.add_argument('--filter_thres', type=float, default=0.9)    
    parser.add_argument('--num_prototype', type=int, default=1000)
    main(parser.parse_args())
