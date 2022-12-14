import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from skimage.transform import resize
import time
import sys

sys.path.append('../utilspag')
from utils import *


def normalize(slice, bottom=99.5, down=0.5):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp


def generate_subimage(ct_array, seg_array, numz, numx, numy, blockz, blockx, blocky,
                      idx, origin, direction, xyz_thickness, savedct_path, savedseg_path, ct_file):
    num_z = (ct_array.shape[0] - blockz) // numz + 1  # math.floor()
    num_x = (ct_array.shape[1] - blockx) // numx + 1
    num_y = (ct_array.shape[2] - blocky) // numy + 1
    print("wode", num_z, num_y, num_x)
    for z in range(numz):
        for x in range(numx):
            for y in range(numy):
                seg_block = seg_array[z * num_z:z * num_z + blockz, x * num_x:x * num_x + blockx,
                            y * num_y:y * num_y + blocky]
                if seg_block.any():
                    ct_block = ct_array[z * num_z:z * num_z + blockz, x * num_x:x * num_x + blockx,
                               y * num_y:y * num_y + blocky]
                    saved_ctname = os.path.join(savedct_path, 'volume-' + str(idx) + '.npy')
                    saved_segname = os.path.join(savedseg_path, 'segmentation-' + str(idx) + '.npy')
                    np.save(saved_ctname, ct_block)
                    np.save(saved_segname, seg_block)

                    # saved_ctname = os.path.join(savedct_path, 'volume-' + str(idx) + '.nii')
                    # saved_segname = os.path.join(savedseg_path, 'segmentation-' + str(idx) + '.nii')
                    #
                    # saved_preprocessed(ct_block, origin, direction, xyz_thickness, saved_ctname)
                    # saved_preprocessed(seg_block, origin, direction, xyz_thickness, saved_segname)

                    idx = idx + 1
    return idx


def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)


def get_realfactor(spa, xyz, ct_array):
    resize_factor = spa / xyz
    print('resize', resize_factor)
    new_real_shape = ct_array.shape * resize_factor[::-1]
    print('new', new_real_shape)
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / ct_array.shape
    return real_resize_factor


def preprocess():
    start_time = time.time()
    ##########hyperparameters1##########
    images_path = '../LITS2017/CT'
    labels_path = '../LITS2017/seg'
    if not os.path.exists(images_path):
        print("images_path ?????????")
    if not os.path.exists(labels_path):
        print("labels_path ?????????")

    savedct_path = '../data/train_image'
    savedseg_path = '../data/train_mask'

    trainImage = savedct_path
    trainMask = savedseg_path
    if not os.path.exists(trainImage):
        os.makedirs(savedct_path)
        print("trainImage ????????????????????????")
    if not os.path.exists(trainMask):
        os.makedirs(savedseg_path)
        print("trainMask ????????????????????????")

    # ??????????????????
    saved_idx = 0
    expand_slice = 10
    new_spacing = [0.8, 0.8, 1.5]
    blockz = 64
    blockx = 128
    blocky = 160  # ?????????????????????
    numz = 6
    numx = 5
    numy = 4
    for ct_file in os.listdir(images_path):  # num_file
        ct = sitk.ReadImage(os.path.join(images_path, ct_file),
                            sitk.sitkInt16)  # sitk.sitkInt16 Read one image using SimpleITK
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        spacing = np.array(list(ct.GetSpacing()))
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(labels_path, ct_file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        print('-------', ct_file, '-------')
        print('original space', np.array(ct.GetSpacing()))
        print('original shape and spacing:', ct_array.shape, spacing)

        # step1: spacing interpolation
        real_resize_factor = get_realfactor(spacing, new_spacing, ct_array)
        # ????????????out_spacing????????????size
        ct_array = ndimage.zoom(ct_array, real_resize_factor, order=3)
        # ?????????????????????????????????????????????????????????????????????????????????,????????????????????????????????????
        # ??????order=1?????????zoomed seg unique = [0,1,2]
        seg_array = ndimage.zoom(seg_array, real_resize_factor, order=0)

        print('new space', new_spacing)
        print('zoomed shape:', ct_array.shape, ',', seg_array.shape)

        # step2 :get mask effective range(startpostion:endpostion)
        pred_liver = seg_array.copy()
        pred_liver[pred_liver > 0] = 1
        bb = find_bb(pred_liver)
        ct_array = ct_array[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        seg_array = seg_array[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        print('effective shape:', ct_array.shape, ',', seg_array.shape)
        # step3:?????????Normalization
        ct_array_nor = normalize(ct_array)

        if ct_array.shape[0] < blockz:
            print('generate no subimage !')
        else:
            saved_idx = generate_subimage(ct_array_nor, seg_array, numz, numx, numy, blockz, blockx, blocky,
                                          saved_idx, origin, direction, new_spacing, savedct_path, savedseg_path,
                                          ct_file)

        print('Time {:.3f} min'.format((time.time() - start_time) / 60))
        print(saved_idx)


if __name__ == '__main__':
    start_time = time.time()
    logfile = '../logs/printLog0117'
    if os.path.isfile(logfile):
        os.remove(logfile)
    sys.stdout = Logger(logfile)  # see utilspag.py

    preprocess()

    # Decide preprocess of different stride and window
    # Decide_preprocess(blockzxy,config)

    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
