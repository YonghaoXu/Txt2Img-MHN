import torch
import numpy as np

def test_acc(model,classname, data_loader, epoch,num_classes, print_per_batches=10):

    model.eval()
    
    class_name_list = classname
    num_classes = len(classname)
    num_batches = len(data_loader)

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    total = 0
    correct = 0
    class_acc = np.zeros((num_classes,1))
    for batch_idx, data in enumerate(data_loader):

        images, labels = data[0].cuda(),data[1].cuda()
        batch_size = labels.size(0)
        _,outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total += batch_size
        correct += (predicted == labels).sum().item()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            
        if (batch_idx+1)%print_per_batches == 0:
            print('Epoch[%d]-Validation-[%d/%d] Batch OA: %.2f %%' % (epoch,batch_idx+1,num_batches,100.0 * (predicted == labels).sum().item() / batch_size))

    for i in range(num_classes):
        class_acc[i] = 1.0*class_correct[i] / class_total[i]
        print('---------------Accuracy of %12s : %.2f %%---------------' % (
            class_name_list[i], 100 * class_acc[i])) 
    acc = 1.0*correct / total
    print('---------------Epoch[%d]Validation-OA: %.2f %%---------------' % (epoch,100.0 * acc))
    print('---------------Epoch[%d]Validation-AA: %.2f %%---------------' % (epoch,100.0 * np.mean(class_acc)))
    return acc,class_acc
