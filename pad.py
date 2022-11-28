import pickle
import os
import numpy as np
import nibabel as nib

modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
    'root': '../data/Data_Training',
    'flist': 'train.txt',
    'has_label': False
}

test_set = {
    'root': '../data/Data_Validation',
    'flist': 'test.txt',
    'has_label': False
}


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
    images = np.array(nib_load(path + 'flair.nii.gz'), dtype='float32', order='C')
    print(images)


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

    for path in paths:
        process_f32b0(path, has_label)


if __name__ == '__main__':
    doit(test_set)
