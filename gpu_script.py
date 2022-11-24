
pip install tensorflow-gpu

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

############## to start working on a GPU #############
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # just use one GPU on big machine
import torch
assert torch.cuda.device_count() == 1


import tensorflow as tf
tf.config.list_physical_devices('GPU')
#################################################
#################################################
#moniter the gpu usage form terminal 
https://pypi.org/project/gpu-utils/#description


pip install gpustat
gpustat -cp

nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1
################################################################
################################################################
