import os
import numpy as np
import time
import torch
import argparse
from torch.utils import data
from torchvision import transforms
from tools.utils import test_acc
import tools.model as models
from torch.utils.data import Dataset
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')
 
class scene_dataset(Dataset):
    def __init__(self, root_dir, pathfile, transform=None, loader=default_loader):
        pf = open(pathfile, 'r')
        imgs = []
        for line in pf:
            line = line.rstrip('\n')
            words = line.split()
            name = words[0].split('/')[-1].split('.')[0]
            imgs.append((root_dir+words[0],int(words[1]),name))

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
 
def main(args):
    num_classes = 30
    classname = ('sparseresidential','resort','railwaystation',
                    'parking','bareland','stadium',
                    'mountain','school','playground',
                    'desert','baseballfield','commercial',
                    'storagetanks','river','meadow',
                    'center','airport','square',
                    'park','beach','bridge',
                    'industrial','church','mediumresidential',
                    'forest','farmland','port',
                    'pond','denseresidential','viaduct')

    print_per_batches = args.print_per_batches
    save_path_prefix = './pretrained_inception/'    
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)

    composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size,args.crop_size)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    train_loader = data.DataLoader(
        scene_dataset(root_dir=args.root_dir,pathfile='./dataset/RSICD_train.txt', transform=composed_transforms),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = data.DataLoader(
        scene_dataset(root_dir=args.root_dir,pathfile='./dataset/RSICD_test.txt', transform=composed_transforms),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    Model = models.inception_v3(pretrained=True, aux_logits=False)  
    Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    
    Model = torch.nn.DataParallel(Model).cuda()
    Model_optimizer = torch.optim.Adam(Model.parameters(),lr=args.lr)
    num_batches = len(train_loader)
    
    cls_loss = torch.nn.CrossEntropyLoss()
    num_steps = args.num_epochs*num_batches
    hist = np.zeros((num_steps,3))
    index_i = -1
    
    for epoch in range(args.num_epochs):
        for batch_index, src_data in enumerate(train_loader):
            index_i += 1
    
            tem_time = time.time()
            Model.train()
            Model_optimizer.zero_grad()

            X_train, Y_train, _ = src_data
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
            
            _,output = Model(X_train)

            # CE Loss            
            _, src_prd_label = torch.max(output, 1)            
            cls_loss_value = cls_loss(output, Y_train)
            cls_loss_value.backward()

            Model_optimizer.step()            
            
            hist[index_i,0] = time.time()-tem_time
            hist[index_i,1] = cls_loss_value.item()   
            hist[index_i,2] = torch.mean((src_prd_label == Y_train).float()).item() 

            tem_time = time.time()
            if (batch_index+1) % print_per_batches == 0:
                print('Epoch %d/%d:  %d/%d Time: %.2f cls_loss = %.3f acc = %.3f \n'\
                %(epoch+1, args.num_epochs,batch_index+1,num_batches,
                np.mean(hist[index_i-print_per_batches+1:index_i+1,0]),
                np.mean(hist[index_i-print_per_batches+1:index_i+1,1]),
                np.mean(hist[index_i-print_per_batches+1:index_i+1,2])))

    OA_new,_ = test_acc(Model,classname, val_loader, epoch+1,num_classes,print_per_batches=10)    
    model_name = 'epoch_'+str(epoch+1)+'_OA_'+repr(int(OA_new*10000))+'.pth'

    print('Save Model') 
    torch.save(Model.state_dict(), os.path.join(save_path_prefix, model_name))
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/iarai/home/yonghao.xu/Data/RSICD_cls/',help='dataset path.')   
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--print_per_batches', type=int, default=5)
    main(parser.parse_args())
