"""
@Modified from:
    https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    https://github.com/mseitzer/pytorch-fid
    https://github.com/lzhbrian/metrics
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from tools.model import inception_v3

from scipy.stats import entropy
from scipy import linalg
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

CUR_DIRNAME = os.path.dirname(os.path.abspath(__file__))

def default_loader(path):
    return Image.open(path).convert('RGB')
 
class scene_dataset(Dataset):
    def __init__(self, root_dir, pathfile, transform=None, loader=default_loader, mode='gen'):
        pf = open(pathfile, 'r')
        imgs = []
        if mode=='gen':
            for line in pf:
                line = line.rstrip('\n')
                words = line.split()
                name = words[0]
                imgs.append(root_dir+words[0])
        elif mode=='ori':
            for line in pf:
                line = line.rstrip('\n')
                words = line.split('/')[-1].split(' ')[0]
                name = words
                imgs.append(root_dir+words)

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        pf.close()
 
    def __getitem__(self, index):
        fn = self.imgs[index]
        img = self.loader(fn)        
        if self.transform is not None:            
            img = self.transform(img)       
        return img
 
    def __len__(self):
        return len(self.imgs)

def read_stats_file(filepath):
    """read mu, sigma from .npz"""
    if filepath.endswith('.npz'):
        f = np.load(filepath)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        raise Exception('ERROR! pls pass in correct npz file %s' % filepath)
    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths %s, %s' % (mu1.shape, mu2.shape)
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions %s, %s' % (sigma1.shape, sigma2.shape)
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class ScoreModel:
    def __init__(self, mode, cuda=True,
                 stats_file='', mu1=0, sigma1=0):
        """
        Computes the inception score of the generated images
            cuda -- whether or not to run on GPU
            mode -- image passed in inceptionV3 is normalized by mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                and in range of [-1, 1]
                1: image passed in is normalized by mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                2: image passed in is normalized by mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]
        """
        # load mu, sigma for calc FID
        self.calc_fid = False
        if stats_file:
            self.calc_fid = True
            self.mu1, self.sigma1 = read_stats_file(stats_file)
        elif type(mu1) == type(sigma1) == np.ndarray:
            self.calc_fid = True
            self.mu1, self.sigma1 = mu1, sigma1

        # Set up dtype
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            self.dtype = torch.FloatTensor

        # setup image normalization mode
        self.mode = mode
        if self.mode == 1:
            transform_input = True
        elif self.mode == 2:
            transform_input = False
        else:
            raise Exception("ERR: unknown input img type, pls specify norm method!")

        
        self.inception_model = inception_v3(pretrained=False, aux_logits=False)
        self.inception_model.fc = torch.nn.Linear(self.inception_model.fc.in_features, 30)
        
        dirpath = './pretrained_inception/'        
        model_path = os.listdir(dirpath)
        for filename in model_path: 
            filepath = os.path.join(dirpath, filename)           
            if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
                print(os.path.join(dirpath, filename))
                model_path_resume = os.path.join(dirpath, filename)
        
        saved_state_dict = torch.load(model_path_resume)    
        new_params = self.inception_model.state_dict().copy()    
        for i,j in zip(saved_state_dict,new_params):    
            new_params[j] = saved_state_dict[i]
        self.inception_model.load_state_dict(new_params)
        
        self.inception_model.eval().type(self.dtype)
        # self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(self.dtype)

        # remove inception_model.fc to get pool3 output 2048 dim vector
        self.fc = self.inception_model.fc
        self.inception_model.fc = nn.Sequential()

        # wrap with nn.DataParallel
        self.inception_model = nn.DataParallel(self.inception_model)
        self.fc = nn.DataParallel(self.fc)

    def __forward(self, x):
        """
        x should be N x 3 x 299 x 299
        and should be in range [-1, 1]
        """
        _,x = self.inception_model(x)
        pool3_ft = x.data.cpu().numpy()

        x = self.fc(x)
        preds = F.softmax(x, 1).data.cpu().numpy()
        return pool3_ft, preds

    @staticmethod
    def __calc_is(preds, n_split, return_each_score=False):
        """
        regularly, return (is_mean, is_std)
        if n_split==1 and return_each_score==True:
            return (scores, 0)
            # scores is a list with len(scores) = n_img = preds.shape[0]
        """

        n_img = preds.shape[0]
        # Now compute the mean kl-div
        split_scores = []
        for k in range(n_split):
            part = preds[k * (n_img // n_split): (k + 1) * (n_img // n_split), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
            if n_split == 1 and return_each_score:
                return scores, 0
        return np.mean(split_scores), np.std(split_scores)

    @staticmethod
    def __calc_stats(pool3_ft):
        mu = np.mean(pool3_ft, axis=0)
        sigma = np.cov(pool3_ft, rowvar=False)
        return mu, sigma

    def get_score_image_tensor(self, imgs_nchw, mu1=0, sigma1=0,
                               n_split=10, batch_size=32, return_stats=False,
                               return_each_score=False):
        """
        param:
            imgs_nchw -- Pytorch Tensor, size=(N,C,H,W), in range of [-1, 1]
            batch_size -- batch size for feeding into Inception v3
            n_splits -- number of splits
        return:
            is_mean, is_std, fid
            mu, sigma of dataset
            regularly, return (is_mean, is_std)
            if n_split==1 and return_each_score==True:
                return (scores, 0)
                # scores is a list with len(scores) = n_img = preds.shape[0]
        """

        n_img = imgs_nchw.shape[0]

        assert batch_size > 0
        assert n_img > batch_size

        pool3_ft = np.zeros((n_img, 2048))
        preds = np.zeros((n_img, 1000))
        for i in tqdm(range(np.int32(np.ceil(1.0 * n_img / batch_size)))):
            batch_size_i = min((i+1) * batch_size, n_img) - i * batch_size
            batchv = Variable(imgs_nchw[i * batch_size:i * batch_size + batch_size_i, ...].type(self.dtype))
            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)

        # if want to return stats
        # or want to calc fid
        if return_stats or \
                type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            mu2, sigma2 = self.__calc_stats(pool3_ft)

        if self.calc_fid:
            mu1 = self.mu1
            sigma1 = self.sigma1

        is_mean, is_std = self.__calc_is(preds, n_split, return_each_score)

        fid = -1
        if type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        if return_stats:
            return is_mean, is_std, fid, mu2, sigma2
        else:
            return is_mean, is_std, fid

    def get_score_image_iter(self, imgs, mu1=0, sigma1=0,
                               n_split=10, batch_size=32, return_stats=False,
                               return_each_score=False):
        """
        param:
            imgs_nchw -- Pytorch Tensor, size=(N,C,H,W), in range of [-1, 1]
            batch_size -- batch size for feeding into Inception v3
            n_splits -- number of splits
        return:
            is_mean, is_std, fid
            mu, sigma of dataset
            regularly, return (is_mean, is_std)
            if n_split==1 and return_each_score==True:
                return (scores, 0)
                # scores is a list with len(scores) = n_img = preds.shape[0]
        """
        
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
        n_img = len(imgs)


        assert batch_size > 0
        assert n_img > batch_size

        pool3_ft = np.zeros((n_img, 2048))
        preds = np.zeros((n_img, 30))

        tbar = tqdm(dataloader)
        for i, batch in enumerate(tbar, 0):
            batch = batch.type(self.dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)
            tbar.set_description('Evaluation for the %d/%d batch' % (i+1, len(dataloader)))

        # if want to return stats
        # or want to calc fid
        if return_stats or \
                type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            mu2, sigma2 = self.__calc_stats(pool3_ft)

        if self.calc_fid:
            mu1 = self.mu1
            sigma1 = self.sigma1

        is_mean, is_std = self.__calc_is(preds, n_split, return_each_score)

        fid = -1
        if type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        if return_stats:
            return is_mean, is_std, fid, mu2, sigma2
        else:
            return is_mean, is_std, fid

    def get_score_dataset(self, dataset, mu1=0, sigma1=0,
                          n_split=10, batch_size=32, return_stats=False,
                          return_each_score=False):
        """
        get score from a dataset
        param:
            dataset -- pytorch dataset, img in range of [-1, 1]
            batch_size -- batch size for feeding into Inception v3
            n_splits -- number of splits
        return:
            is_mean, is_std, fid
            mu, sigma of dataset
            regularly, return (is_mean, is_std)
            if n_split==1 and return_each_score==True:
                return (scores, 0)
                # scores is a list with len(scores) = n_img = preds.shape[0]
        """

        n_img = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        pool3_ft = np.zeros((n_img, 2048))
        preds = np.zeros((n_img, 30))
        for i, batch in tqdm(enumerate(dataloader, 0)):
            batch = batch.type(self.dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]
            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)

        # if want to return stats
        # or want to calc fid
        if return_stats or \
                type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            mu2, sigma2 = self.__calc_stats(pool3_ft)

        if self.calc_fid:
            mu1 = self.mu1
            sigma1 = self.sigma1

        is_mean, is_std = self.__calc_is(preds, n_split, return_each_score)

        fid = -1
        if type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        if return_stats:
            return is_mean, is_std, fid, mu2, sigma2
        else:
            return is_mean, is_std, fid


if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gen_dir', type=str, default='/iarai/home/yonghao.xu/Code/Txt2Img-MHN/gen_vqgan/0721_0455/',help='path for the generated images')   
    parser.add_argument('--data_dir', type=str, default='/iarai/home/yonghao.xu/Data/RSICD/',help='path for the original images')   
    parser.add_argument('--blur', type=int, default=0)   
    args = parser.parse_args()

    is_fid_model = ScoreModel(mode=2, cuda=True)

    composed_transforms = transforms.Compose([
        transforms.Resize(size=(256,256)),            
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    print ("Calculating Inception Score and FID Score...")

    if args.blur == 0:
        img_list_tensor1 = scene_dataset(root_dir=args.gen_dir,pathfile='./dataset/RSICD_gen.txt', transform=composed_transforms)
    else:
        composed_transforms_blur = transforms.Compose([
            transforms.Resize(size=(256,256)),            
            transforms.ToTensor(),
            transforms.GaussianBlur(1+2*args.blur),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])        
        img_list_tensor1 = scene_dataset(root_dir=args.gen_dir,pathfile='./dataset/RSICD_gen.txt', transform=composed_transforms_blur)

    img_list_tensor2 = scene_dataset(root_dir=args.data_dir,pathfile='./dataset/RSICD_test.txt', transform=composed_transforms,mode='ori')

    print('Calculating 1st stat ...')
    is_mean1, is_std1, _, mu1, sigma1 = \
        is_fid_model.get_score_image_iter(img_list_tensor1, n_split=1, return_stats=True)

    print('Calculating 2nd stat ...')
    is_mean2, is_std2, fid = is_fid_model.get_score_image_iter(img_list_tensor2,
                                                                    mu1=mu1, sigma1=sigma1,
                                                                    n_split=1)
 
    print('1st IS score =', is_mean1, ',', is_std1)
    print('2nd IS score =', is_mean2, ',', is_std2)
    print('FID =', fid)

    result = np.zeros((3,))
    result[0] = is_mean1
    result[1] = is_mean2
    result[2] = fid
    np.save(args.gen_dir.split('/')[-2]+'_blur_'+str(args.blur)+'_is_'+str(is_mean1)+'_fid_'+str(fid)+'.npy',result)
