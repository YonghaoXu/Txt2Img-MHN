import os
import torch
import numpy as np
from pathlib import Path
import youtokentome as yttm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_vae_image(self, writer,images,recons,hard_recons,global_step):
        writer.add_image('Ori Image', images, global_step)        
        writer.add_image('Reconstruction', recons, global_step)        
        writer.add_image('Hard Reconstruction', hard_recons, global_step)

    def visualize_gen_image(self, writer,image,text,global_step,path,vae_type=0):             
        plt.figure(figsize=(10,3))
        if vae_type == 0:
            image = np.moveaxis(image.squeeze().cpu().numpy(),0,-1)*0.5+0.5
        else:
            image = np.moveaxis(image.squeeze().cpu().numpy(),0,-1)
        image[image>1] = 1
        image[image<0] = 0
        plt.imshow(image)
        plt.title(text[0])        
        plt.axis('off')
        plt.savefig(path+str(global_step)+'.png')
        writer.add_figure('Generation', plt.gcf(), global_step)

class YttmTokenizer:
    def __init__(self, bpe_path = './tools/rsicd_vocab_5000.txt'):
        bpe_path = Path(bpe_path)
        assert bpe_path.exists(), f'BPE json path {str(bpe_path)} does not exist'
        tokenizer = yttm.BPE(model = str(bpe_path))
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size()

    def decode(self, tokens, pad_tokens = set()):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens, ignore_ids = pad_tokens.union({0}))

    def encode(self, texts):
        encoded = self.tokenizer.encode(texts, output_type = yttm.OutputType.ID)
        return list(map(torch.tensor, encoded))

    def tokenize(self, texts, context_length = 256, truncate_text = False):
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = self.encode(texts)
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]
