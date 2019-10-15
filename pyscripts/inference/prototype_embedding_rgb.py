from __future__ import print_function

import argparse
import os
import time
import math
from tqdm import tqdm

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

    image = reader.image
    label = reader.label
    image_list = reader.image_list
  image_batch = tf.expand_dims(image, dim=0)
  label_batch = tf.expand_dims(label, dim=0)

  # Create input tensor to the Network
  crop_image_batch = tf.placeholder(
      name='crop_image_batch',
      shape=[1,input_size[0],input_size[1],3],
      dtype=tf.float32)

  # Create network and output prediction.
  outputs = model(crop_image_batch,
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
      [input_size[0], input_size[1]])

  # Input full-sized embedding
  label_input = tf.placeholder(
      tf.int32, shape=[1, None, None, 1])
  embedding_input = tf.placeholder(
      tf.float32, shape=[1, None, None, args.embedding_dim])
  embedding = common_utils.normalize_embedding(embedding_input)
  loc_feature = tf.placeholder(
      tf.float32, shape=[1, None, None, 2])
  rgb_feature = tf.placeholder(
      tf.float32, shape=[1, None, None, 3])

  # Combine embedding with location features and kmeans
  shape = tf.shape(embedding)
  cluster_labels = common_utils.initialize_cluster_labels(
      [args.num_clusters, args.num_clusters],
      [shape[1], shape[2]])
  embedding = tf.reshape(embedding, [-1, args.embedding_dim])
  labels = tf.reshape(label_input, [-1])
  cluster_labels = tf.reshape(cluster_labels, [-1])
  location_features = tf.reshape(loc_feature, [-1, 2])
  rgb_features = common_utils.normalize_embedding(
      tf.reshape(rgb_feature, [-1, 3])) / args.embedding_dim

    # Collect pixels of valid semantic classes.
  valid_pixels = tf.where(
      tf.not_equal(labels, args.ignore_label))
  labels = tf.squeeze(tf.gather(labels, valid_pixels), axis=1)
  cluster_labels = tf.squeeze(tf.gather(cluster_labels, valid_pixels), axis=1)
  embedding = tf.squeeze(tf.gather(embedding, valid_pixels), axis=1)
  location_features = tf.squeeze(
      tf.gather(location_features, valid_pixels), axis=1)
  rgb_features = tf.squeeze(tf.gather(rgb_features, valid_pixels), axis=1)

  # Generate cluster labels via kmeans clustering.
  embedding_with_location = tf.concat(
      [embedding, location_features, rgb_features], 1)
  embedding_with_location = common_utils.normalize_embedding(
      embedding_with_location)
  cluster_labels = common_utils.kmeans_with_initial_labels(
      embedding_with_location,
      cluster_labels,
      args.num_clusters * args.num_clusters,
      args.kmeans_iterations)
  _, cluster_labels = tf.unique(cluster_labels)

  # Find pixels of majority semantic classes.
  select_pixels, prototype_labels = eval_utils.find_majority_label_index(
      labels, cluster_labels)

  # Calculate the prototype features.
  cluster_labels = tf.squeeze(tf.gather(cluster_labels, select_pixels), axis=1)
  embedding = tf.squeeze(tf.gather(embedding, select_pixels), axis=1)

  prototype_features = common_utils.calculate_prototypes_from_labels(
      embedding, cluster_labels)


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


  pbar = tqdm(range(num_steps))
  for step in pbar:
    image_batch_np, label_batch_np = sess.run(
        [image_batch, label_batch])

    img_size = image_batch_np.shape
    padded_img_size = list(img_size)  # deep copy of img_size

    if input_size[0] > padded_img_size[1]:
      padded_img_size[1] = input_size[0]
    if input_size[1] > padded_img_size[2]:
      padded_img_size[2] = input_size[1]
    padded_img_batch = np.zeros(padded_img_size,
                                dtype=np.float32)
    img_h, img_w = img_size[1:3]
    padded_img_batch[:, :img_h, :img_w, :] = image_batch_np

    stride_h, stride_w = strides
    npatches_h = math.ceil(1.0*(padded_img_size[1]-input_size[0])/stride_h) + 1
    npatches_w = math.ceil(1.0*(padded_img_size[2]-input_size[1])/stride_w) + 1

    # Create the ending index of each patch.
    patch_indh = np.linspace(
        input_size[0], padded_img_size[1], npatches_h, dtype=np.int32)
    patch_indw = np.linspace(
        input_size[1], padded_img_size[2], npatches_w, dtype=np.int32)
    
    # Create embedding holder.
    padded_img_size[-1] = args.embedding_dim
    embedding_all_np = np.zeros(padded_img_size,
                                dtype=np.float32)
    for indh in patch_indh:
      for indw in patch_indw:
        sh, eh = indh-input_size[0], indh  # start & end ind of H
        sw, ew = indw-input_size[1], indw  # start & end ind of W
        cropimg_batch = padded_img_batch[:, sh:eh, sw:ew, :]

        embedding_np = sess.run(output, feed_dict={
            crop_image_batch: cropimg_batch})
        embedding_all_np[:, sh:eh, sw:ew, :] += embedding_np

    embedding_all_np = embedding_all_np[:, :img_h, :img_w, :]
    loc_feature_np = common_utils.generate_location_features_np([padded_img_size[1], padded_img_size[2]])
    feed_dict = {label_input: label_batch_np,
                 embedding_input: embedding_all_np,
                 loc_feature: loc_feature_np,
                 rgb_feature: padded_img_batch}

    (batch_prototype_features_np,
     batch_prototype_labels_np) = sess.run(
      [prototype_features, prototype_labels],
      feed_dict=feed_dict)

    if step == 0:
      prototype_features_np = batch_prototype_features_np
      prototype_labels_np = batch_prototype_labels_np
    else:
      prototype_features_np = np.concatenate(
          [prototype_features_np, batch_prototype_features_np], axis=0)
      prototype_labels_np = np.concatenate(
          [prototype_labels_np,
           batch_prototype_labels_np], axis=0)


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
