'''
Author: JinghuaOriental 1795185859@qq.com
Date: 2024-04-12 14:48:28
LastEditors: JinghuaOriental 1795185859@qq.com
LastEditTime: 2024-04-13 09:58:42
FilePath: /hyj/python_files2/Generate/GANs/HAGAN/HA-GAN/2_train/load_args.py
Description: 加载YAML配置文件
'''

import argparse
import yaml


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
            default_args = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"default_args file '{file_path}' not found.") from exc
    except yaml.YAMLError as e:
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            raise ValueError(f'Error parsing YAML file at line {mark.line + 1}, column {mark.column + 1}: {e}') from e
        raise ValueError("Error parsing YAML file:", e) from e

    return default_args

def get_args(config_yaml_path='config.yaml'):
    """
    读取配置文件，并返回解析后的字典对象。

    返回:
        dict: 解析后的YAML配置内容。
    """
    # 默认情况下，尝试从YAML配置文件加载参数
    default_args = load_config_yaml(config_yaml_path)


    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='PyTorch HA-GAN Training')

    # 添加命令行参数，并使用YAML配置文件中的值作为默认值
    parser.add_argument('--is_pretrained', type=bool, default=default_args['is_pretrained'], help="use pre-trained model or not")
    parser.add_argument('--pretrained-models-dir', type=str, default=default_args['pretrained_models_dir'], help="path to the pre-trained models directory")
    parser.add_argument('--batch-size', type=int, default=default_args['batch_size'],
                        help='mini-batch size (default: 4), this is the total '
                             'batch size of all GPUs')
    parser.add_argument('--workers', type=int, default=default_args['workers'],
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--img-size', type=int, default=default_args['img_size'],
                        help='size of training images (default: 256, can be 128 or 256)')
    parser.add_argument('--num-iter', type=int, default=default_args['num_iter'],
                        help='number of iteration for training (default: 80000)')
    parser.add_argument('--log-iter', type=int, default=default_args['log_iter'],
                        help='number of iteration between logging (default: 20)')
    parser.add_argument('--continue-iter', type=int, default=default_args['continue_iter'],
                        help='continue from a checkpoint that has run for n iteration  (0 if a new run)')
    parser.add_argument('--latent-dim', type=int, default=default_args['latent_dim'],
                        help='size of the input latent variable')
    parser.add_argument('--g-iter', type=int, default=default_args['g_iter'],
                        help='number of generator pass per iteration')
    parser.add_argument('--lr-g', type=float, default=default_args['lr_g'],
                        help='learning rate for the generator')
    parser.add_argument('--lr-d', type=float, default=default_args['lr_d'],
                        help='learning rate for the discriminator')
    parser.add_argument('--lr-e', type=float, default=default_args['lr_e'],
                        help='learning rate for the encoder')
    parser.add_argument('--data-dir', type=str, default=default_args['data_dir'],
                        help='path to the preprocessed data folder')
    parser.add_argument('--exp-name', type=str, default=default_args['exp_name'],
                        help='name of the experiment')
    parser.add_argument('--fold', type=int, default=default_args['fold'],
                        help='fold number for cross validation')

    # 条件生成的配置
    parser.add_argument('--lambda-class', type=float, default=default_args['lambda_class'],
                        help='weights for the auxiliary classifier loss')
    parser.add_argument('--num-class', type=int, default=default_args['num_class'],
                        help='number of class for auxiliary classifier (0 if unconditional)')

    # 解析命令行参数
    args = parser.parse_args()

    return args
