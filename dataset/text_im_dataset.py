from random import randint, choice
import PIL
from torch.utils.data import Dataset
from torchvision import transforms as T
import os.path as osp
from pathlib import Path

class TextImageDataset(Dataset):
    def __init__(self, folder, list_path, text_len=40, image_size=256, truncate_captions=False, resize_ratio=0.75, tokenizer=None, shuffle=False):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.img_ids = [i_id.strip() for i_id in open(list_path)]       
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(folder, name+'.jpg')
            text_file = osp.join(folder, name+'.txt')
            self.files.append({
                "img": img_file,
                "text": text_file,
            })        

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        datafiles = self.files[ind]        
        image_file = datafiles["img"]
        text_file = datafiles["text"] 
        descriptions = Path(text_file).read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor


class ImageFolder(Dataset):   
    def __init__(self, folder, list_path, transform):
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


class TextDataset(Dataset):
    def __init__(self, folder, list_path, text_len=40, truncate_captions=False, tokenizer=None, shuffle=False):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.img_ids = [i_id.strip() for i_id in open(list_path)]       
        self.files = []

        for name in self.img_ids:
            text_file = osp.join(folder, name+'.txt')
            self.files.append({
                "text": text_file,
                "name": name,
            })        

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        datafiles = self.files[ind]      
        text_file = datafiles["text"] 
        descriptions = Path(text_file).read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = descriptions[0]

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        
        return tokenized_text,datafiles["name"] 
    
    
class CLIP_GenTextImageDataset(Dataset):
    def __init__(self, img_folder, txt_folder, list_path, image_size=256,):

        super().__init__()
        self.img_ids = [i_id.strip() for i_id in open(list_path)]       
        self.files = []

        for name in self.img_ids:
            for i in range(10):
                img_file = osp.join(img_folder, name+'_'+str(i)+'.png')
                text_file = osp.join(txt_folder, name+'.txt')
                self.files.append({
                    "img": img_file,
                    "text": text_file,
                })        

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        datafiles = self.files[ind]        
        image_file = datafiles["img"]
        text_file = datafiles["text"] 
        descriptions = Path(text_file).read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = descriptions[0]
       
        image_tensor = self.image_transform(PIL.Image.open(image_file))
       
        return description,image_tensor
