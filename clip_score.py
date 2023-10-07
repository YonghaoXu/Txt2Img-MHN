import os
import argparse
from tqdm import tqdm
from dataset.text_im_dataset  import CLIP_GenTextImageDataset
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def main(args):    

    save_path_prefix = args.save_path_prefix
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)     

    # Instantiate the CLIPProcessor object and load the pre-trained weights
    processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2",device="cuda")

    # Instantiate the CLIPModel object and load the pre-trained weights
    model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2").to("cuda")

    test_set = CLIP_GenTextImageDataset(
        img_folder=args.gen_dir,
        txt_folder=args.data_dir,
        list_path=args.test_list,
        image_size=args.crop_size
    )   

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

    tbar = tqdm(test_loader)
    sim_avg = 0.
    for index, (text, images) in enumerate((tbar)):        
        image_inputs = processor(images=images, return_tensors="pt").pixel_values.cuda()
        text_inputs = processor(text=text, return_tensors="pt").input_ids.cuda()

        image_embeddings = model.get_image_features(image_inputs)
        text_embeddings = model.get_text_features(text_inputs)

        similarity_score = (image_embeddings @ text_embeddings.T).max().item()/100.

        sim_avg += similarity_score
        tbar.set_description('Processing the %d/%d sample' % (index+1, len(test_loader)))
        
    print(args.gen_dir.split('/')[-1]+" clip score: ", sim_avg/(index+1))
    np.save(save_path_prefix+args.gen_dir.split('/')[-1]+'_clip_score_'+str(int(sim_avg/(index+1)*10000))+'.npy',sim_avg/(index+1))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--save_path_prefix', type=str, default='./clip_score/')
    parser.add_argument('--gen_dir', type=str, default='/iarai/home/yonghao.xu/Code/Ready/txt2img/gen_vqgan/',help='dataset path.')   
    parser.add_argument('--data_dir', type=str, default='/iarai/home/yonghao.xu/Data/RSICD/',help='dataset path.')   
    parser.add_argument('--test_list', type=str, default='./dataset/RSICD_test.txt',help='test list file.')
    parser.add_argument('--crop_size', type=int, default=256)
    
    main(parser.parse_args())
