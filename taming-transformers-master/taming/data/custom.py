from torch.utils.data import Dataset
import os.path as osp
import PIL

from taming.data.base import ImagePaths

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, folder='/iarai/home/yonghao.xu/Data/RSICD/'):
        super().__init__()
        #with open(training_images_list_file, "r") as f:
        #    paths = f.read().splitlines()
        paths = [osp.join(folder, i_id.strip()+'.jpg') for i_id in open(training_images_list_file)]   
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, folder='/iarai/home/yonghao.xu/Data/RSICD/'):
        super().__init__()
        #with open(test_images_list_file, "r") as f:
        #    paths = f.read().splitlines()
        paths = [osp.join(folder, i_id.strip()+'.jpg') for i_id in open(test_images_list_file)]   
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)



class ImageFolder(Dataset):   
    def __init__(self, list_path, transform, folder='/iarai/home/yonghao.xu/Data/RSICD/'):
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]       
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(folder, name+'.jpg')            
            self.files.append({
                "img": img_file
            })        

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, ind):
        image_file = self.files[ind]["img"]      
        image_tensor = self.transform(PIL.Image.open(image_file))
        return image_tensor
