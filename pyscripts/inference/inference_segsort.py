"""Single-scale inference script for predicting segmentations using SegSort."""

from __future__ import print_function

import argparse
import math
import os
import time
import scipy.io

import network.segsort.common_utils as common_utils
import network.segsort.eval_utils as eval_utils
import tensorflow as tf
import numpy as np

from PIL import Image
from seg_models.image_reader import SegSortImageReader
from seg_models.models.pspnet import pspnet_resnet101 as model
from tqdm import tqdm


IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)


def get_arguments():
  """Parse all the arguments provided from the CLI.
    
  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description='Inference for Semantic Segmentation')
  parser.add_argument('--data_dir', type=str, default='',
                      help='/path/to/dataset.')
  parser.add_argument('--data_list', type=str, default='',
                      help='/path/to/datalist/file.')
  parser.add_argument('--input_size', type=str, default='512,512',
                      help='Comma-separated string with H and W of image.')
  parser.add_argument('--strides', type=str, default='512,512',
                      help='Comma-separated string with strides of H and W.')
  parser.add_argument('--num_classes', type=int, default=21,
                      help='Number of classes to predict.')
  parser.add_argument('--ignore_label', type=int, default=255,
                      help='Index of label to ignore.')
  parser.add_argument('--restore_from', type=str, default='',
                      help='Where restore model parameters from.')
  parser.add_argument('--save_dir', type=str, default='',
                      help='/path/to/save/predictions.')
  parser.add_argument('--colormap', type=str, default='',
                      help='/path/to/colormap/file.')
  # SegSort parameters.
  parser.add_argument('--prototype_dir', type=str, default='',
                      help='/path/to/prototype/file.')
  parser.add_argument('--embedding_dim', type=int, default=32,
                      help='Dimension of the feature embeddings.')
  parser.add_argument('--num_clusters', type=int, default=5,
                      help='Number of kmeans clusters along each axis')
  parser.add_argument('--kmeans_iterations', type=int, default=10,
                      help='Number of kmeans iterations.')
  parser.add_argument('--k_in_nearest_neighbors', type=int, default=15,
                      help='K in k-nearest neighbor search.')

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
  """Create the model and start the Inference process."""
  args = get_arguments()
    
  # Create queue coordinator.
  coord = tf.train.Coordinator()

  # Load the data reader.
  with tf.name_scope('create_inputs'):
    reader = SegSortImageReader(
        args.data_dir,
        args.data_list,
        parse_commastr(args.input_size),
        False,  # No random scale
        False,  # No random mirror
        False,  # No random crop, center crop instead
        args.ignore_label,
        IMG_MEAN)

    image_list = reader.image_list
    image_batch = tf.expand_dims(reader.image, dim=0)
    label_batch = tf.expand_dims(reader.label, dim=0)
    cluster_label_batch = tf.expand_dims(reader.cluster_label, dim=0)
    loc_feature_batch = tf.expand_dims(reader.loc_feature, dim=0)
    height = reader.height
    width = reader.width

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

  # Prototype placeholders.
  prototype_features = tf.placeholder(tf.float32,
                                      shape=[None, args.embedding_dim])
  prototype_labels = tf.placeholder(tf.int32)

  # Combine embedding with location features.
  embedding_with_location = tf.concat([embedding, loc_feature_batch], 3)
  embedding_with_location = common_utils.normalize_embedding(
      embedding_with_location)

  # Kmeans clustering.
  cluster_labels = common_utils.kmeans(
      embedding_with_location,
      [args.num_clusters, args.num_clusters],
      args.kmeans_iterations)
  test_prototypes = common_utils.calculate_prototypes_from_labels(
      embedding, cluster_labels)

  # Predict semantic labels.
  semantic_predictions, _ = eval_utils.predict_semantic_instance_labels(
      cluster_labels,
      test_prototypes,
      prototype_features,
      prototype_labels,
      None,
      args.k_in_nearest_neighbors)
  semantic_predictions = tf.cast(semantic_predictions, tf.uint8)
  semantic_predictions = tf.squeeze(semantic_predictions)
    

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

  # Get colormap.
  map_data = scipy.io.loadmat(args.colormap)
  key = os.path.basename(args.colormap).replace('.mat','')
  colormap = map_data[key]
  colormap *= 255
  colormap = colormap.astype(np.uint8)

  # Create directory for saving predictions.
  pred_dir = os.path.join(args.save_dir, 'gray')
  color_dir = os.path.join(args.save_dir, 'color')
  if not os.path.isdir(pred_dir):
    os.makedirs(pred_dir)
  if not os.path.isdir(color_dir):
    os.makedirs(color_dir)
    
  # Iterate over testing steps.
  with open(args.data_list, 'r') as listf:
    num_steps = len(listf.read().split('\n'))-1

  # Load prototype features and labels.
  prototype_features_np = np.load(
      os.path.join(args.prototype_dir, 'prototype_features.npy'))
  prototype_labels_np = np.load(
      os.path.join(args.prototype_dir, 'prototype_labels.npy'))

  feed_dict = {prototype_features: prototype_features_np,
               prototype_labels: prototype_labels_np}

  for step in tqdm(range(num_steps)):
    semantic_predictions_np, height_np, width_np = sess.run(
        [semantic_predictions, height, width], feed_dict=feed_dict)

    semantic_predictions_np = semantic_predictions_np[:height_np, :width_np]

    basename = os.path.basename(image_list[step])
    basename = basename.replace('jpg', 'png')

    predname = os.path.join(pred_dir, basename)
    Image.fromarray(semantic_predictions_np, mode='L').save(predname)

    colorname = os.path.join(color_dir, basename)
    color = colormap[semantic_predictions_np]
    Image.fromarray(color, mode='RGB').save(colorname)

  coord.request_stop()
  coord.join(threads)
    
if __name__ == '__main__':
    main()
