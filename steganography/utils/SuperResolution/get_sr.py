import argparse
import json
import os
import time
import sys

import numpy as np
import tensorflow as tf
import torch

from steganography.utils.SuperResolution.utils import image_utils
from steganography.utils.SuperResolution.srgraph import SRGraph

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config_path = "D:\ChromDownload\srzoo-master\srzoo-master\configs\edsr_baseline.json"
config_path_rcan_8 = "D:\ChromDownload\srzoo-master\srzoo-master\configs\\rcan.json"
config_path_edsr = "D:\ChromDownload\srzoo-master\srzoo-master\configs\edsr.json"


model_path = "D:\ChromDownload\srzoo-master\srzoo-master\model\edsr_baseline_x2.pb"
model_path_rcan_8 = "D:\ChromDownload\srzoo-master\srzoo-master\model\\rcan_x8.pb"
model_path_rcan_4 = "D:\ChromDownload\srzoo-master\srzoo-master\model\\rcan_x4.pb"
model_path_edsr_3 = "D:\ChromDownload\srzoo-master\srzoo-master\model\edsr_x3.pb"

# input_path = "D:\ChromDownload\srzoo-master\srzoo-master\src"
input_path = "D:\learning\pythonProjects\HiddenWatermark1\\test\\test_result"
output_path = "D:\ChromDownload\srzoo-master\srzoo-master\out"



# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='path of the config file (.json)', default=config_path_rcan_8)
parser.add_argument('--model_path', help='path of the model file (.pb)',default=model_path_rcan_4)
parser.add_argument('--input_path', default=input_path, help='folder path of the lower resolution (input) images')
parser.add_argument('--output_path', default=output_path, help='folder path of the high resolution (output) images')
parser.add_argument('--scale', default=8, help='upscaling factor')
parser.add_argument('--self_ensemble', action='store_true', help='employ self ensemble')
parser.add_argument('--cuda_device', default='-1', help='CUDA device index to be used (will be set to the environment variable \'CUDA_VISIBLE_DEVICES\')')
args = parser.parse_args()


# constants
IMAGE_EXTS = ['.png', '.jpg']


def main():
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
  tf.logging.set_verbosity(tf.logging.INFO)

  # SR config
  with open(args.config_path, 'r') as f:
    sr_config = json.load(f)

  # SR graph
  sr_model = SRGraph()
  sr_model.prepare(scale=args.scale, standalone=True, config=sr_config, model_path=args.model_path)

  # image reader/writer
  image_reader = image_utils.ImageReader()
  image_writer = image_utils.ImageWriter()

  # image path list
  image_path_list = []
  for root, _, files in os.walk(args.input_path):
    for filename in files:
      for ext in IMAGE_EXTS:
        if (filename.lower().endswith(ext)):
          image_name = os.path.splitext(filename)[0]
          input_path = os.path.join(root, filename)
          output_path = os.path.join(args.output_path, '%s.png' % (image_name))

          image_path_list.append([input_path, output_path])
  tf.logging.info('found %d images' % (len(image_path_list)))

  # run a dummy image to initialize internal graph
  input_image = np.zeros([32, 32, 3], dtype=np.uint8)
  sr_model.get_output([input_image])

  # iterate
  running_time_list = []
  for input_path, output_path in image_path_list:
    input_image = image_reader.read(input_path)

    running_time = 0.0

    if (args.self_ensemble):
      output_images = []
      ensemble_running_time_list = []
      for flip_index in range(2): # for flipping
        input_image = np.transpose(input_image, axes=(1, 0, 2))

        for rotate_index in range(4): # for rotating
          input_image = np.rot90(input_image, k=1, axes=(0, 1))

          t1 = time.perf_counter()
          output_image = sr_model.get_output([input_image])[0]
          t2 = time.perf_counter()
          ensemble_running_time_list.append(t2 - t1)

          output_image = np.clip(output_image, 0, 255)

          output_image = np.rot90(output_image, k=(3-rotate_index), axes=(0, 1))
          if (flip_index == 0):
            output_image = np.transpose(output_image, axes=(1, 0, 2))
          output_images.append(output_image)
      
      output_image = np.mean(output_images, axis=0)
      running_time = np.sum(ensemble_running_time_list)
    
    else:
      t1 = time.perf_counter()
      output_image = sr_model.get_output([input_image])[0]
      t2 = time.perf_counter()
      running_time = (t2 - t1)

      output_image = np.clip(output_image, 0, 255)
    
    output_image = np.round(output_image)
    
    image_writer.write(output_image, output_path)
    tf.logging.info('%s -> %s, %.3f sec' % (input_path, output_path, running_time))
    running_time_list.append(running_time)
  
  # finalize
  tf.logging.info('finished')
  tf.logging.info('averaged running time per image: %.3f sec' % (np.mean(running_time_list)))


def super_resolution(img:torch.Tensor,
                     model_path = "D:\ChromDownload\srzoo-master\srzoo-master\model\edsr_baseline_x2.pb",
                     config_path= "D:\ChromDownload\srzoo-master\srzoo-master\configs\edsr_baseline.json",
                     scale = 8):
  #
  # img.tolist() -> 将数据转换为list
  #
  #
  # config_path = "D:\ChromDownload\srzoo-master\srzoo-master\configs\edsr_baseline.json"
  # model_path = "D:\ChromDownload\srzoo-master\srzoo-master\model\edsr_baseline_x2.pb"
  #

  # img 是 b c h w 数据 需要进行 suqeeze(0) 然后transpose（1,2,0）
  img = img.squeeze(0).permute(1,2,0)
  with open(config_path, 'r') as f:
    sr_config = json.load(f)


  sr_model = SRGraph()
  sr_model.prepare(scale=scale, standalone=True, config=sr_config, model_path=model_path)
  # 首先执行了一步初始化
  # run a dummy image to initialize internal graph
  input_image = np.zeros([32, 32, 3], dtype=np.uint8)
  sr_model.get_output([input_image])
  # get_output 需要传入的参数 list包括的 uint8 的图片形式
  out_put = sr_model.get_output([img.tolist()])[0]


  return torch.from_numpy(out_put).permute(2,1,0)


if __name__ == '__main__':
  main()
