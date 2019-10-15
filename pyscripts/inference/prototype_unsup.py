from __future__ import print_function

import argparse
import os
import time
import math

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import network.vmf.common_utils as common_utils
import network.vmf.eval_utils as eval_utils
from PIL import Image

from seg_models.models.pspnet import pspnet_resnet101 as model
from seg_models.image_reader import VMFImageReader
import utils.general

IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)


def get_arguments():
  """Parse all the arguments provided from the CLI.
    
  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description='Extracting Prototypes for Semantic Segmentation')
  parser.add_argument('--data-dir', type=str, default='',
                      help='/path/to/dataset.')
  parser.add_argument('--data-list', type=str, default='',
                      help='/path/to/datalist/file.')
  parser.add_argument('--input-size', type=str, default='512,512',
                      help='Comma-separated string with H and W of image.')
  parser.add_argument('--strides', type=str, default='512,512',
                      help='Comma-separated string with strides of H and W.')
  parser.add_argument('--num-classes', type=int, default=21,
                      help='Number of classes to predict.')
  parser.add_argument('--ignore-label', type=int, default=255,
                      help='Index of label to ignore.')
  parser.add_argument('--restore-from', type=str, default='',
                      help='Where restore model parameters from.')
  parser.add_argument('--save-dir', type=str, default='',
                      help='/path/to/save/predictions.')
  parser.add_argument('--colormap', type=str, default='',
                      help='/path/to/colormap/file.')
  # vMF parameters
  parser.add_argument('--embedding_dim', type=int, default=32,
                      help='Dimension of the feature embeddings.')
  parser.add_argument('--num_clusters', type=int, default=5,
                      help='Number of kmeans clusters along each axis')
  parser.add_argument('--kmeans_iterations', type=int, default=10,
                      help='Number of kmeans iterations.')


  return parser.parse_args()

def load(saver, sess, ckpt_path):
  """Load the trained weights.
  
  Args:
    saver: TensorFlow saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  """ 
  saver.restore(sess, ckpt_path)
  print('Restored model parameters from {}'.format(ckpt_path))

def parse_commastr(str_comma):
  """Read comma-sperated string.
  """
  if '' == str_comma:
    return None
  else:
    a, b =  map(int, str_comma.split(','))

  return [a,b]

def main():
  """Create the model and start the Inference process.
  """
  args = get_arguments()
    
  # Parse image processing arguments.
  input_size = parse_commastr(args.input_size)
  strides = parse_commastr(args.strides)
  assert(input_size is not None and strides is not None)
  h, w = input_size
  innet_size = (int(math.ceil(h/8)), int(math.ceil(w/8)))


  # Create queue coordinator.
  coord = tf.train.Coordinator()

  # Load the data reader.
  with tf.name_scope('create_inputs'):
    reader = VMFImageReader(
        args.data_dir,
        args.data_list,
        None,
        False, # No random scale.
        False, # No random mirror.
        False, # No random crop, center crop instead
        args.ignore_label,
        IMG_MEAN)

    image_batch = tf.expand_dims(reader.image, dim=0)
    label_batch = tf.expand_dims(reader.label, dim=0)
    cluster_label_batch = tf.expand_dims(reader.cluster_label, dim=0)
    loc_feature_batch = tf.expand_dims(reader.loc_feature, dim=0)

  # Create network and output prediction.
  outputs = model(image_batch,
                  args.embedding_dim,
                  False,
                  True)

  # Grab variable names which should be restored from checkpoints.
  restore_var = [
    v for v in tf.global_variables() if 'crop_image_batch' not in v.name]
    
  # Output predictions.
  output = outputs[0]
  output = tf.image.resize_bilinear(
      output,
      tf.shape(image_batch)[1:3,])
  embedding = common_utils.normalize_embedding(output)

  shape = embedding.get_shape().as_list()
  batch_size = shape[0]

  labels = label_batch
  initial_cluster_labels = cluster_label_batch[0, :, :]
  location_features = tf.reshape(loc_feature_batch[0, :, :], [-1, 2])

  prototype_feature_list = []
  prototype_label_list = []
  for bs in range(batch_size):
    cur_labels = tf.reshape(labels[bs], [-1])
    cur_cluster_labels = tf.reshape(initial_cluster_labels, [-1])
    cur_embedding = tf.reshape(embedding[bs], [-1, args.embedding_dim])

    (prototype_features,
     prototype_labels,
     _) = eval_utils.extract_trained_prototypes(
         cur_embedding, location_features, cur_cluster_labels,
         args.num_clusters * args.num_clusters,
         args.kmeans_iterations, cur_labels,
         1, args.ignore_label,
         'semantic')

    prototype_feature_list.append(prototype_features)
    prototype_label_list.append(prototype_labels)

  prototype_features = tf.concat(prototype_feature_list, axis=0)
  prototype_labels = tf.concat(prototype_label_list, axis=0)

    
  # Set up tf session and initialize variables. 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
    
  sess.run(init)
  sess.run(tf.local_variables_initializer())
    
  # Load weights.
  loader = tf.train.Saver(var_list=restore_var)
  if args.restore_from is not None:
    load(loader, sess, args.restore_from)
    
  # Start queue threads.
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  # Create directory for saving prototypes.
  save_dir = os.path.join(args.save_dir, 'prototypes')
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
  # Iterate over testing steps.
  with open(args.data_list, 'r') as listf:
    num_steps = len(listf.read().split('\n'))-1

  for step in range(num_steps):
    (batch_prototype_features_np,
     batch_prototype_labels_np) = sess.run(
      [prototype_features, prototype_labels])

    if step == 0:
      prototype_features_np = batch_prototype_features_np
      prototype_labels_np = batch_prototype_labels_np
    else:
      prototype_features_np = np.concatenate(
          [prototype_features_np, batch_prototype_features_np], axis=0)
      prototype_labels_np = np.concatenate(
          [prototype_labels_np,
           batch_prototype_labels_np], axis=0)

    if (step + 1) % 100 == 0:
      print('Processed batches: ', (step + 1), '/', num_steps)

  print ('Total number of prototypes extracted: ',
         len(prototype_labels_np))
  np.save(
      tf.gfile.Open('%s/%s.npy' % (save_dir, 'prototype_features'),
                     mode='w'), prototype_features_np)
  np.save(
      tf.gfile.Open('%s/%s.npy' % (save_dir, 'prototype_labels'),
                     mode='w'), prototype_labels_np)


  coord.request_stop()
  coord.join(threads)
    
if __name__ == '__main__':
    main()
