import SimpleITK as sitk
import numpy as np
import os
import json
from functools import partial
import multiprocessing as mp

def sitkReadZoom(path, newSize=None, centerCrop=None):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    sitk_img = reader.Execute()



    if centerCrop:
        x, y, z = sitk_img.GetSize()
        outputSpacing, size, outputOrigin, outputDirection = sitk_img.GetSpacing(), (
            int(x / 2), int(y / 2), z), sitk_img.GetOrigin(), sitk_img.GetDirection()
        img = sitk.GetArrayFromImage(sitk_img)
        cropX, cropY = int(x / 4), int(y / 4)
        img = img[:, cropX:-cropX, cropY:-cropY]

        sitk_img = sitk.GetImageFromArray(img)
        sitk_img.SetSpacing(outputSpacing)
        sitk_img.SetDirection(outputDirection)

    if newSize:
        newX, newY, newZ = newSize
        dx, dy, dz = sitk_img.GetSpacing()
        x, y, z = sitk_img.GetSize()
        if newZ == -1:
            newZ = z
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(((x * dx / newX), (y * dy / newY), (z * dz) / newZ))
        resample.SetSize((newX, newY, newZ))
        resample.SetOutputOrigin(sitk_img.GetOrigin())
        resample.SetOutputDirection(sitk_img.GetDirection())
        newImage = resample.Execute(sitk_img)
        # no nuyl
        newImage = sitk.Cast(sitk.RescaleIntensity(newImage), sitk.sitkUInt8)  # nyul  

        out_array = sitk.GetArrayFromImage(newImage)
        out_array = out_array / 255  

        return out_array
    else:
        # no nuyl
        sitk_img = sitk.Cast(sitk.RescaleIntensity(sitk_img), sitk.sitkUInt8)  # nyul
        out_array = sitk.GetArrayFromImage(sitk_img)
        return out_array



def generate_patch(index, data_path, pat_path_list, centerROI, outpath):
    pat = pat_path_list[index]
    print(pat)
    vol_all = []
    for series in ['sag PDW', 'cor PDW', 'axi PDW']:
        imgPath = os.path.join(data_path, pat, series)
        try:
            vol = sitkReadZoom(imgPath, (512, 512, 24))  # (288,288,24)
        except:
            return None
        roi = [256, 256, 12, 512, 512, 24]
        _cx, _cy, _cz, X, Y, Z = roi
        vol = np.transpose(vol, [1, 2, 0])
        if series == 'sag PDW':
            _x, _y, _z = vol.shape
            if _cx > 341:
                _cx = 256
            if _cx <171:
                _cx = 256
            if _cy > 341:
                _cy = 256
            if _cy < 171:
                _cy = 256
            vol = vol[_cy - 170:_cy + 170, _cx - 170:_cx + 170, :]
        elif series == 'cor PDW':
            _y, _z, _x = vol.shape
            slice_y = (vol[256, :, 12] > np.min(vol[256, :, 12]))
            y_offset = slice_y.argmax(axis=0)
            slice_y_reverse = slice_y[::-1]
            y_offset_2 = slice_y_reverse.argmax(axis=0)

            _cz, _cy = int(_cz / Z * (_z - y_offset - y_offset_2) + y_offset), (int(_cy / Y * _y))
            if _cz > 341:
                _cz = 256
            if _cz <171:
                _cz = 256
            if _cy > 341:
                _cy = 256
            if _cy < 171:
                _cy = 256
            _rcz, _rcy = (_cz - 170, _cz + 170), (_cy - 170, _cy + 170)
            vol = vol[_rcy[0]:_rcy[1], _rcz[0]:_rcz[1], :]
        elif series=='axi PDW':
            _z, _x, _y = vol.shape
            _cz, _cx = int(_cz / Z * _z), int(_cx / X * _x)
            if _cz > 341:
                _cz = 256
            if _cz <171:
                _cz = 256
            if _cx > 341:
                _cx = 256
            if _cx < 171:
                _cx = 256
            _rcz, _rcx = (_cz - 170, _cz + 170), (_cx - 170, _cx + 170)
            vol = vol[_rcz[0]:_rcz[1], _rcx[0]:_rcx[1], :]
        if vol.shape[0] == 0 or vol.shape[1] == 0 or vol.shape[2] == 0:
            print('shape error')
            continue
        vol = np.transpose(vol, (2, 0, 1))  # augmentation debug
        vol_all.append(vol)

    vol_all = np.array(vol_all)
    np.save(os.path.join(outpath, pat+'.npy'), vol_all)


def run(center_pathdict):
    
    ROI_path = center_pathdict.roi
    Output_path = center_pathdict.cache
    data_path = center_pathdict.data

    if not os.path.exists(Output_path):
        os.makedirs(Output_path)
    with open(ROI_path, 'r') as f:
        centerROI = json.load(f)

    data_list = os.listdir(data_path)

    
    data_list_new = []
    for pat in data_list:
        if not pat+'.npy' in os.listdir(Output_path):
            data_list_new.append(pat)
    deal_partial = partial(generate_patch, data_path=data_path, pat_path_list=data_list_new, centerROI=centerROI, outpath = Output_path)
    number_processes = 50
    pool = mp.Pool(processes=number_processes)

    res = pool.map(deal_partial, list(range(len(data_list_new))))