from torch.utils.data import Dataset
from PIL import Image

category_dict={'airport': 0, 'bareland': 1, 'baseballfield': 2, 'beach': 3, 'bridge': 4, 'center': 5, 'church': 6, 'commercial': 7,\
               'denseresidential': 8, 'desert': 9, 'farmland': 10, 'forest': 11, 'industrial': 12, 'meadow': 13, 'mediumresidential': 14, 'mountain': 15, \
               'parking': 16, 'park': 17, 'playground': 18, 'pond': 19, 'port': 20, 'railwaystation': 21, 'resort': 22, 'river': 23, 'school': 24, \
               'sparseresidential': 25, 'square': 26, 'stadium': 27, 'storagetanks': 28, 'viaduct': 29}

def default_loader(path):
    return Image.open(path).convert('RGB')
 
class scene_dataset(Dataset):
    def __init__(self, root_dir, pathfile='./dataset/RSICD_test.txt', transform=None, loader=default_loader, mode='ori', num_gen_per_image=10):
        pf = open(pathfile, 'r')
        imgs = []
        if mode=='ori':
            for line in pf:
                line = line.rstrip('\n')
                words = line.split()
                name = words[0]
                imgs.append((root_dir+name+'.jpg',category_dict[name.split("_")[0]],name))
        elif mode=='gen':
            for line in pf:
                line = line.rstrip('\n')
                words = line.split()
                name = words[0]
                for i in range(num_gen_per_image):
                    imgs.append((root_dir+name+'_'+str(i)+'.png',category_dict[name.split("_")[0]],name+'_'+str(i)))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        pf.close()
 
    def __getitem__(self, index):
        fn, label, name = self.imgs[index]
        img = self.loader(fn)        
        if self.transform is not None:            
            img = self.transform(img)       
        return img,label,name
 
    def __len__(self):
        return len(self.imgs)