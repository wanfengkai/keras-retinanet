import csv
import os
import numpy as np
from shutil import copy2,rmtree,move
list_map=[
    ('01','holly shit'),
    ('02','larger shit'),
    ('03','shuangxing?'),
    ('04','wtf'),
    ('05','yyy'),
    ('06','double kill'),
    ('07','triple kill'),
    ('08','ultra kill'),
    ('09','rampage'),
    ('10','god like'),
    ('11','unstoppable '),
    ('12','Stop'),
    ('13','bg'),
    ('14','bg'),
]
dict_map = dict(list_map)

def get_txt_from_folder(new_txt,path):
    for folder in os.listdir(path):
        if 'label' in folder:
            # label files folder.
            label_folder_full_path = os.path.join(path,folder)
            for label_txt in os.listdir(label_folder_full_path):
                # get video number
                print(label_txt)
                basename = os.path.basename(os.path.join(label_folder_full_path,label_txt))
                video_number = basename.split('.')[0]
                with open(new_txt,'a+') as f:
                    writer = csv.writer(f)
                    with open(file = os.path.join(label_folder_full_path,label_txt),mode = 'r+') as txt_file:
                        reader = csv.reader(txt_file,delimiter = '_')
                        # Skip the header
                        next(reader)
                        for line in reader:
                            # Get useful information
                            frame_num = line[0]
                            cls = line[1]
                            llx = line[2]
                            lly = line[3]
                            urx = line[-2]
                            ury = line[-1]
                            # write to new file
                            new_line=[]
                            #TODO:image folder path 'imgs' should be changed as your need, This is not pythonic.
                            img_dir = os.path.join(path,'img')
                            frame_num = int(frame_num)
                            img_basename=video_number+'_'+str(frame_num)+'.png'
                            img_path=os.path.join(img_dir,img_basename)
                            new_line.append(img_path)
                            new_line.append(llx)
                            new_line.append(lly)
                            new_line.append(urx)
                            new_line.append(ury)
                            real_class = dict_map.get(cls)
                            new_line.append(real_class)
                            writer.writerow(new_line)
def mkdir_if_not_exists(fullpath):
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

def split_files_into_corresponding_folders(original_label_folder,new_folder,img_folder,ratio=0.8):
    # Create folders.
    train_folder = os.path.join(new_folder,'train')
    valid_folder = os.path.join(new_folder, 'valid')
    train_label_path = os.path.join(train_folder,'label')
    valid_label_path = os.path.join(valid_folder,'label')
    train_img_path = os.path.join(train_folder,'img')
    valid_img_path = os.path.join(valid_folder,'img')
    mkdir_if_not_exists(train_label_path)
    mkdir_if_not_exists(valid_label_path)
    mkdir_if_not_exists(train_img_path)
    mkdir_if_not_exists(valid_img_path)

    # Random split the train and valid for label folder.
    original_list = os.listdir(original_label_folder)
    number = len(original_list)
    num_train_split = int(np.floor(number * ratio))
    np.random.shuffle(original_list)
    train_labels_txt_list=original_list[0:num_train_split]
    valid_labels_txt_list=original_list[num_train_split:-1]
    dst = train_label_path
    for file_name in train_labels_txt_list:
        full_path = os.path.join(original_label_folder,file_name)
        copy2(full_path,dst)
    dst = valid_label_path
    for file_name in valid_labels_txt_list:
        full_path = os.path.join(original_label_folder,file_name)
        copy2(full_path,dst)
    # Image split
    for file_name in train_labels_txt_list:
        n1,n2 = file_name.split('.')[0].split('_')
        for img_name in os.listdir(img_folder):
            nn1, nn2 = img_name.split('.')[0].split('_')[0:2]
            if  (nn1,nn2)== (n1,n2):
                copy2(os.path.join(img_folder,img_name),train_img_path)

    # Image split
    for file_name in valid_labels_txt_list:
        n1,n2 = file_name.split('.')[0].split('_')
        for img_name in os.listdir(img_folder):
            nn1, nn2 = img_name.split('.')[0].split('_')[0:2]
            if  (nn1,nn2)== (n1,n2):
                copy2(os.path.join(img_folder,img_name),valid_img_path)



def split_label_files_wrt_test_folder(original_train_image_folder,original_test_image_folder,
                                      original_label_path,new_folder,temp_path):

    # Create folders.
    train_folder = os.path.join(new_folder,'train')
    valid_folder = os.path.join(new_folder, 'valid')
    train_label_path = os.path.join(train_folder,'label')
    valid_label_path = os.path.join(valid_folder,'label')
    train_img_path = os.path.join(train_folder,'img')
    valid_img_path = os.path.join(valid_folder,'img')
    mkdir_if_not_exists(train_label_path)
    mkdir_if_not_exists(valid_label_path)
    mkdir_if_not_exists(train_img_path)
    mkdir_if_not_exists(valid_img_path)
    # Copy all image files to coresponding folder.
    for file in os.listdir(original_test_image_folder):
        copy2(os.path.join(original_test_image_folder,file),valid_img_path)
    for file in os.listdir(original_train_image_folder):
        copy2(os.path.join(original_train_image_folder,file),train_img_path)

    # find name pattern and split label txt files.
    for labels in os.listdir(original_label_path):
        copy2(os.path.join(original_label_path,labels),temp_path)

    # For test file
    # attempted_string=[]
    # Set is more elegant than list here.
    attempted_string=set()
    for img_name in os.listdir(valid_img_path):
        # get the first two number
        nn1, nn2 = img_name.split('.')[0].split('_')[0:2]
        name_str_pre = nn1+'_'+nn2
        if name_str_pre not in attempted_string:
            move(os.path.join(temp_path,name_str_pre+'.txt'),valid_label_path)
            attempted_string.add(name_str_pre)
    # For train file just move the rest
    for lbl in os.listdir(temp_path):
        move(os.path.join(temp_path, lbl),train_label_path)


if __name__ == '__main__':

    # Only excute once
    # The new folder is the folder where You want to save all the splited training and valid data.
    new_folder = '/home/frank/big_Od'
    rmtree(new_folder)
    # #original_label_folder is the place where you stored the virtual label txts
    # #img_folder is where your virtual images are, which is just the ~/Virtual mentioned  in shoushoukanREADME
    # split_files_into_corresponding_folders(original_label_folder='/home/frank/OD/ShoushouCycleGAN/virtual2real/labels/unreal',
    #                                        new_folder=new_folder,
    #                                        img_folder='/home/frank/OD/ShoushouCycleGAN/virtual2real/virtual_img')


    valid_path = os.path.join(new_folder,'valid')
    train_path = os.path.join(new_folder,'train')
    new_txt = os.path.join(new_folder,'new_txt.txt')

    # Here is what you need to change the path

    temp_path =  '/home/frank/OD_temp'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    # Here you set the original TRaining Set
    original_test_image_folder = 'TrainA'
    # Here you set the original test Set.
    original_label_path = 'TestB'
    split_label_files_wrt_test_folder(original_test_image_folder,
                                      original_label_path, new_folder, temp_path)


    get_txt_from_folder(new_txt=new_txt,path=train_path)
    get_txt_from_folder(new_txt=new_txt,path=valid_path)

    # Find that all the 300 frame is missing. So remove them.
    clean_txt = os.path.join(new_folder,'clean_txt.txt')
    with open(clean_txt,'w+') as w:
        writer = csv.writer(w)
        with open(new_txt,'r+') as f:
           reader = csv.reader(f)
           for line in reader:
               if '300' not in line[0]:
                   writer.writerow(line)
