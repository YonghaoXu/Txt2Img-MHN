import torch
from torch import nn, einsum
import torch.nn.functional as F
from math import log2,sqrt
from einops import rearrange
from axial_positional_embedding import AxialPositionalEmbedding
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ
import importlib

def is_empty(t):
    return t.nelement() == 0

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_sample(t, temperature = 1., dim = -1):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return ((t / temperature) -log(-log(noise))).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def max_neg_value(t):
    return -torch.finfo(t.dtype).max
    
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x):
        return self.fn(x) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        x = self.fn(x)
        return self.norm_out(x)

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class VQVAE(nn.Module):    
    def __init__(self,image_size=256,num_tokens=2048,codebook_dim=512,num_layers=3,num_resnet_blocks=2,
                    hidden_dim=64,channels=3,temperature=0.9):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))
        enc_chans = [channels, *enc_chans]
        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))
        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))
        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.mse_loss

    def norm(self, images, normalization=((0.5,) * 3, (0.5,) * 3)):
        means, stds = map(lambda t: torch.as_tensor(t).to(images), normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(self,img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))
        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(self,img,return_loss=True,return_recons=True,return_logits=False,temp=None):
        img = self.norm(img)
        logits = self.encoder(img)
        if return_logits:
            return logits

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)
        if not return_loss:
            return out

        # reconstruction loss
        recon_loss = self.loss_fn(img, out)
        if not return_recons:
            return recon_loss
        return recon_loss, out

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()

        model_path = vqgan_model_path
        config_path = vqgan_config_path
        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models
        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(log2(f)/log2(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented

class HopfieldLayer(nn.Module):
    def __init__(self,dim,n_prototype=1000,dropout=0.1):
        super().__init__()
        self.beta = 1./sqrt(dim)
        self.lookup_matrix = nn.Linear(dim, n_prototype, bias = False)
        self.content_matrix = nn.Linear(n_prototype,dim,bias = False)
        self.softmax = torch.softmax
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        lookup = self.softmax(self.lookup_matrix(x) * self.beta, dim=-1)
        content = self.content_matrix(lookup)
        return self.dropout(content)

class SelfAttention(nn.Module):
    def __init__(self,dim,seq_len,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.beta = 1./sqrt(dim_head)
        self.heads = heads
        self.seq_len = seq_len
        self.to_qk = nn.Linear(dim, inner_dim*2, bias = False)
        self.to_out = nn.Sequential(
                        nn.Linear(inner_dim, dim),
                        nn.Dropout(dropout),
                        )
        self.softmax = torch.softmax
        
    def forward(self, x):
        h, device = self.heads, x.device
        qk = self.to_qk(x).chunk(2, dim = -1)       
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qk)        

        bqk = torch.einsum('b h i d, b h j d -> b h i j', q* self.beta, k)
        mask_value = max_neg_value(bqk)

        # causality
        i, j = bqk.shape[-2:]
        mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()    
        bqk = self.softmax(bqk.masked_fill_(mask, mask_value), dim=-1)

        bqkk = torch.einsum('b h i j, b h j d -> b h i d', bqk, k)
        out = rearrange(bqkk, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class SelfAttention_qkv(nn.Module):
    def __init__(self,dim,seq_len,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.beta = 1./sqrt(dim_head)
        self.heads = heads
        self.seq_len = seq_len
        self.q = nn.Linear(dim, inner_dim, bias = False)
        self.k = nn.Linear(dim, inner_dim, bias = False)
        self.v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Sequential(
                        nn.Linear(inner_dim, dim),
                        nn.Dropout(dropout),
                        )
        self.softmax = torch.softmax
        
    def forward(self, x):
        h, device = self.heads, x.device
        q = rearrange(self.q(x), 'b n (h d) -> b h n d', h = h)
        k = rearrange(self.k(x), 'b n (h d) -> b h n d', h = h)
        v = rearrange(self.v(x), 'b n (h d) -> b h n d', h = h)         

        bqk = torch.einsum('b h i d, b h j d -> b h i j', q* self.beta, k)
        mask_value = max_neg_value(bqk)

        # causality
        i, j = bqk.shape[-2:]
        mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()    
        bqk = self.softmax(bqk.masked_fill_(mask, mask_value), dim=-1)

        bqkv = torch.einsum('b h i j, b h j d -> b h i d', bqk, v)
        out = rearrange(bqkv, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class PrototypeBlock(nn.Module):
    def __init__(self,dim,n_block,seq_len,heads=8,dim_head=64,num_prototype=1000):    
        super().__init__()    
        self.seq_len = seq_len
        self.layers = nn.ModuleList([])    
        for i in range(n_block):                 
            self.layers.append(nn.ModuleList([
                LayerScale(dim,i+1,PreNorm(dim,HopfieldLayer(dim,num_prototype))),
                LayerScale(dim,i+1,PreNorm(dim,SelfAttention(dim,seq_len=seq_len,heads=heads,dim_head=dim_head))),
            ]))
        pos_emb = None      
        self.register_buffer('pos_emb', pos_emb)
    def forward(self, x):
        for (f, g) in self.layers:
            x = x + f(x)
            x = x + g(x)
        return x

class Txt2ImgMHN(nn.Module):
    def __init__(self,vae,num_text_tokens=5000,dim=512,text_seq_len=40,n_block=10,heads=8,dim_head=48,num_prototype=1000):    
        super().__init__()
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2
        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)
      
        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)
        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim)  # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        self.total_seq_len = text_seq_len + image_seq_len
        self.total_tokens = num_text_tokens + num_image_tokens

        self.vae = vae
        set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.prototype_learning = PrototypeBlock(dim=dim,seq_len=self.total_seq_len,n_block=n_block,heads=heads,dim_head=dim_head,num_prototype=num_prototype)

        self.lookup = nn.Linear(dim, num_prototype,bias=False)        
        self.softmax = torch.softmax
        self.beta = 1./sqrt(dim)
        self.content = nn.Sequential(
            nn.Linear(num_prototype, self.total_tokens,bias=False),
            nn.LayerNorm(self.total_tokens),
        )

        seq_range = torch.arange(self.total_seq_len)
        logits_range = torch.arange(self.total_tokens)
        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )

        self.register_buffer('logits_mask', logits_mask, persistent=False)

    @torch.no_grad()
    @eval_decorator
    def generate_images(self,text,clip = None,filter_thres = 0.5,temperature = 1.,cond_scale = 1.):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len
        text = text[:, :text_seq_len] # make sure text is within bounds
        out = text
       
        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len
            text, image = out[:, :text_seq_len], out[:, text_seq_len:]          
            logits = self(text, image)
       
            if cond_scale != 1:
                # discovery by Katherine Crowson
                # https://twitter.com/RiversHaveWings/status/1478093658716966912
                null_cond_logits = self(text, image, null_cond_prob = 1.)
                logits = null_cond_logits + (logits - null_cond_logits) * cond_scale

            logits = logits[:, -1, :]        
            filtered_logits = top_k(logits, thres = filter_thres)       
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sample -= (num_text_tokens if is_image else 0) # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample[:, None]), dim=-1)

        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)
        return images

    def forward(self,text,image = None,return_loss = False,null_cond_prob = 0.):   
        assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len

        # randomly remove text condition with <null_cond_prob> probability
        if null_cond_prob > 0:
            null_mask = prob_mask_like((batch,), null_cond_prob, device = device)
            text *= rearrange(~null_mask, 'b -> b 1')

        # make sure padding in text tokens get unique padding token id
        text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # add <bos>
        text = F.pad(text, (1, 0), value = 0)
        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        seq_len = tokens.shape[1]
        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4
            if is_raw_image:
                image_size = self.vae.image_size
                assert tuple(image.shape[1:]) == (3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'
                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(image_emb)
            tokens = torch.cat((tokens, image_emb), dim = 1)
            seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained
        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        prototype_learning = self.prototype_learning(tokens)
        lookup = self.softmax(self.lookup(prototype_learning)*self.beta, dim=-1)
        logits = self.content(lookup)
  
        # mask logits to make sure text predicts text (except last token), and image predicts image
        logits_mask = self.logits_mask[:, :seq_len]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if not return_loss:
            return logits
        assert exists(image), 'when training, image must be supplied'

        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim = 1)
        logits = rearrange(logits, 'b n c -> b c n')
        loss = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])
        return loss
