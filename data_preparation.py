"""
This script provides necessary preprocessing for the RSICD dataset 
(since the original RSICD dataset only provides a complete 'json' file to store all text descriptions).

- Step 1. Extract separate '.txt' files for the text descriptions of each image.
- Step 2. Generate the text tokenizer using the 'youtokentome' package.

You can skip step 2 as we already provide the tokenizer in './tools/rsicd_vocab_5000.txt'.
"""


# Step 1. Extract separate '.txt' files for the text descriptions of each image.
# Input:
# - dataset_rsicd.json: the original json file in the RSICD dataset
# Output:
# - RSICD_text.txt:     the txt file containing all text descriptions
# - RSICD_train.txt:    the txt file containing the file names of all training samples
# - RSICD_test.txt:     the txt file containing the file names of all test samples

import os
jsonfile = open('/Path/To/RSICD/annotations_rsicd/dataset_rsicd.json', 'r')
caption_all_txt = open('RSICD_text.txt', 'w')
train_list = open('RSICD_train.txt', 'w')
test_list = open('RSICD_test.txt', 'w')

caption_all = jsonfile.readlines()[0].split("filename\": \"")
for i in range(1,len(caption_all)):
    image_name = caption_all[i].split('.jpg')[0]
    print(image_name)
    text_file_name = os.path.join('/Path/To/RSICD/',image_name+'.txt')

    text_file = open(text_file_name, 'w')
    tem_caption = caption_all[i].split('raw\": \"')
    for j in range(1,len(tem_caption)):
        caption = tem_caption[j].split(' .')[0]
        if caption == tem_caption[j]: #bad case in 00744.jpg
            caption = "a bustling block includes a few buildings, many cars and a football field marked with \'spartanas\'"
        print(caption)
        text_file.writelines([caption,"\n"])
        caption_all_txt.writelines([caption,"\n"])
        
    if tem_caption[-1].split("split\": ")[1].split(',')[0].split('\"')[1]=='train':
        train_list.writelines([image_name,"\n"])
    else:
        test_list.writelines([image_name,"\n"])
        
    text_file.close()
caption_all_txt.close()
train_list.close()
test_list.close()

# max length of the text descriptions
text_file = open('RSICD_text.txt', 'r')
max_length = 1
caption_all = text_file.readlines()
for i in range(len(caption_all)):    
    text_length = len(caption_all[i].split(' '))
    if text_length>max_length:
        print(caption_all[i])
        max_length = text_length
text_file.close()
print(len(caption_all))
print(max_length) # should be 34


# (Optional) Step 2. Generate the text tokenizer using the 'youtokentome' package.
# Input:
# - dataset_rsicd.json:     the original json file in the RSICD dataset
# Output:
# - rsicd_vocab_5000.txt:   the obtained tokenizer model

'''
import youtokentome as yttm

train_data_path = './dataset/RSICD_text.txt'
model_path = "./tools/rsicd_vocab_5000.txt"

# Training model
yttm.BPE.train(data=train_data_path, vocab_size=5000, model=model_path)
# Loading model
bpe = yttm.BPE(model=model_path)
# Two types of tokenization
print(bpe.encode(['a playground is next to a parking lot and many buildings'], output_type=yttm.OutputType.ID))
print(bpe.encode(['a playground is next to a parking lot and many buildings'], output_type=yttm.OutputType.SUBWORD))
'''