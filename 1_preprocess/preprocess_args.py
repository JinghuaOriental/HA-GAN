'''
Author: JinghuaOriental 1795185859@qq.com
Date: 2024-04-11 19:02:52
LastEditors: JinghuaOriental 1795185859@qq.com
LastEditTime: 2024-04-12 18:11:13
FilePath: /hyj/python_files2/Generate/GANs/HAGAN/HA-GAN/preprocess_arg.py
Description: preprocess
'''
import os
import glob
import multiprocessing as mp
import argparse
import yaml
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

def resize_img(img, low_threshold, high_threshold, trim_blank_slices, img_size):
    """ Resize and rescale the input image """
    nan_mask = np.isnan(img)  # Remove NaN
    img[nan_mask] = low_threshold
    img = np.interp(img, [low_threshold, high_threshold], [-1,1])

    if trim_blank_slices:
        valid_plane_i = np.mean(img, (1,2)) != -1  # Remove blank axial planes
        img = img[valid_plane_i,:,:]

    img = resize(img, (img_size, img_size, img_size), mode='constant', cval=-1)
    return img

def batch_resize(batch_idx, input_data_dir, output_data_dir, low_threshold, high_threshold, trim_blank_slices, img_size, num_jobs):
    """ Resize and preprocess a batch of images """
    # 下面这个img_list不能放到本函数内，因为会导致不同进程读取同一批数据
    # 不对，可以放在本函数内，img_list是全局变量
    img_list = glob.glob(os.path.join(input_data_dir, "*.nii.gz"))
    os.makedirs(output_data_dir, exist_ok=True)
    for idx, _ in enumerate(img_list):
        if idx % num_jobs != batch_idx:
            continue
        img_name = img_list[idx].split('/')[-1]
        npy_save_path = os.path.join(output_data_dir, f"{img_name.split('.')[0]}.npy")
        if os.path.exists(npy_save_path):
            # skip images that already finished pre-processing
            continue
        try:
            # img_path = os.path.join(input_data_dir, img_list[idx])
            img_path = img_list[idx]
            img = sitk.ReadImage(img_path)
        except OSError as e: 
            # skip corrupted images
            print(e)
            print("Image loading error:", img_name)
            continue 
        img = sitk.GetArrayFromImage(img)
        try:
            img = resize_img(img, low_threshold, high_threshold, trim_blank_slices, img_size)
        except OSError as e: # Some images are corrupted
            print(e)
            print("Image resize error:", img_name)
            continue
        # preprocessed images are saved in numpy arrays
        np.save(npy_save_path, img)

def load_config_yaml(file_path):
    """
    加载指定路径的YAML配置文件，并返回解析后的字典对象。

    参数:
        file_path (str): YAML配置文件的路径。

    返回:
        dict: 解析后的YAML配置内容。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file '{file_path}' not found.") from exc
    except yaml.YAMLError as e:
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            raise ValueError(f'Error parsing YAML file at line {mark.line + 1}, column {mark.column + 1}: {e}') from e
        raise ValueError("Error parsing YAML file:", e) from e

    return config

def read_config(config_yaml_path='preprocess_args_config.yaml'):
    """
    读取配置文件，并返回解析后的字典对象。

    返回:
        dict: 解析后的YAML配置内容。
    """
    # 默认情况下，尝试从YAML配置文件加载参数
    default_args = load_config_yaml(config_yaml_path)

    parser = argparse.ArgumentParser(description='Image resizing and preprocessing script.')

    # 添加参数时，使用default参数设置从YAML文件中获取的默认值
    parser.add_argument('--num_jobs', type=int, default=default_args['num_jobs'], help='Number of parallel jobs to run')
    parser.add_argument('--img_size', type=int, choices=[128, 256], default=default_args['img_size'], help='Resized output size')
    parser.add_argument('--input_data_dir', type=str, default=default_args['input_data_dir'], help='Directory containing input images')
    parser.add_argument('--output_data_dir', type=str, default=default_args['output_data_dir'], help='Directory for output data')
    parser.add_argument('--low_threshold', type=int, default=default_args['low_threshold'], help='Low intensity threshold')
    parser.add_argument('--high_threshold', type=int, default=default_args['high_threshold'], help='High intensity threshold')
    parser.add_argument('--trim_blank_slices', action='store_true', default=default_args.get('trim_blank_slices', True), help='Trim blank axial slices')

    # 解析命令行参数
    args = parser.parse_args()

    return args

def main():
    """ Main function """

    args = read_config(config_yaml_path='preprocess_args_config.yaml')
    
    processes = []
    for i in range(args.num_jobs):
        processes.append(mp.Process(target=batch_resize, args=(i, args.input_data_dir, args.output_data_dir, args.low_threshold, args.high_threshold, args.trim_blank_slices, args.img_size, args.num_jobs)))
    for p in processes:
        p.start()
        p.join()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main()