import numpy as np
import SimpleITK as sitk
from glob import glob


def resize_image_itk(ct_image, newSize, resamplemethod):
    resampler = sitk.ResampleImageFilter()
    originSize = ct_image.GetSize()  # 原来的体素块尺寸
    print(originSize)
    originSpacing = ct_image.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(ct_image)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(ct_image)  # 得到重新采样后的图像
    return itkimgResampled

if __name__ == '__main__':
    label = True
    if label == True:
        img_path = '../data/train_image/'
        image_file = glob(img_path + '*.nii')
        # print(len(image_file))
    else:
        img_path = '../data/train_mask/'
        image_file = glob(img_path + '*.nii')
        # print(len(image_file))

    for i in range(len(image_file)):
        itkimage = sitk.ReadImage(image_file[i])
        print(itkimage.GetSize())
        # if label==True and itkimage.GetSize() != (160, 128, 64):
        #     itkimgResampled = resize_image_itk(itkimage, (160, 128, 64),
        #                                        resamplemethod=sitk.sitkLinear) #这里要注意：mask用最近邻插值，CT图像用线性插值
        #     sitk.WriteImage(itkimgResampled, '../data/save_train_new/' + image_file[i][len(img_path):])
        # elif label==False and itkimage.GetSize() != (160, 128, 64):
        #     itkimgResampled = resize_image_itk(itkimage, (160, 128, 64),
        #                                        resamplemethod=sitk.sitkNearestNeighbor)
        #     sitk.WriteImage(itkimgResampled, '../data/save_train_mask/' + image_file[i][len(img_path):])

