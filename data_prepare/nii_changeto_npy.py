import nibabel as nib
import os
import numpy as np

img_path = '../data/train_image/'
seg_path = '../data/train_mask/'
saveimg_path = '../data/train_img/'
saveseg_path = '../data/train_msk/'

img_names = os.listdir(img_path)
seg_names = os.listdir(seg_path)

for img_name in img_names:
    print(img_name)
    img = nib.load(img_path + img_name).get_fdata()  # 载入
    img = np.array(img)
    img = img.transpose(2, 1, 0)
    print(img.shape)
    np.save(saveimg_path + str(img_name).split('.')[0] + '.npy', img)  # 保存

for seg_name in seg_names:
    print(seg_name)
    seg = nib.load(seg_path + seg_name).get_fdata()
    seg = np.array(seg)
    seg = seg.transpose(2, 1, 0)
    print(seg.shape)
    np.save(saveseg_path + str(seg_name).split('.')[0] + '.npy', seg)
