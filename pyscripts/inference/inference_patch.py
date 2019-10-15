from __future__ import print_function

import argparse
import copy
import os
import time
import math

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
import network.vmf.common_utils as common_utils
import network.vmf.eval_utils as eval_utils
import network.vmf.vis_utils as vis_utils

from seg_models.models.pspnet import pspnet_resnet101 as model
from seg_models.image_reader import VMFImageReader
import utils.general
import utils.html_helper as html_helper

IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)


def get_arguments():
  """Parse all the arguments provided from the CLI.
    
  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description='Inference for Semantic Segmentation')
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
  """Create the model and start the Inference process.
  """
  args = get_arguments()
    
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

    image_list = reader.image_list
    image = reader.image
    cluster_label = reader.cluster_label
    loc_feature = reader.loc_feature
    height = reader.height
    width = reader.width

  # Create network and output prediction.
  outputs = model(tf.expand_dims(image, dim=0),
                  args.embedding_dim,
                  False,
                  True)

  # Grab variable names which should be restored from checkpoints.
  restore_var = [v for v in tf.global_variables()]
    
  # Output predictions.
  output = outputs[0]
  output = tf.image.resize_bilinear(
      output,
      tf.shape(image)[:2,])
  embedding = common_utils.normalize_embedding(output)
  embedding = tf.squeeze(embedding, axis=0)

  image = image[:height, :width]
  embedding = tf.reshape(
      embedding[:height, :width], [-1, args.embedding_dim])
  cluster_label = tf.reshape(cluster_label[:height, :width], [-1])
  loc_feature = tf.reshape(
      loc_feature[:height, :width], [-1, 2])

  # Prototype placeholders.
  prototype_features = tf.placeholder(tf.float32,
                                      shape=[None, args.embedding_dim])
  prototype_labels = tf.placeholder(tf.int32)

  # Combine embedding with location features and kmeans
  embedding_with_location = tf.concat([embedding, loc_feature], 1)
  embedding_with_location = common_utils.normalize_embedding(
      embedding_with_location)
  cluster_label = common_utils.kmeans_with_initial_labels(
      embedding_with_location,
      cluster_label,
      args.num_clusters * args.num_clusters,
      args.kmeans_iterations)
  _, cluster_labels = tf.unique(cluster_label)
  test_prototypes = common_utils.calculate_prototypes_from_labels(
      embedding, cluster_labels)

  cluster_labels = tf.reshape(cluster_labels, [height, width])

  # Predict semantic labels.
  similarities = tf.matmul(test_prototypes,
                           prototype_features,
                           transpose_b=True)
  _, k_predictions = tf.nn.top_k(similarities, k=args.k_in_nearest_neighbors, sorted=True)

  prototype_semantic_predictions = eval_utils.k_nearest_neighbors(
      k_predictions, prototype_labels)
  semantic_predictions = tf.gather(prototype_semantic_predictions,
                                   cluster_labels)
#  semantic_predictions = tf.squeeze(semantic_predictions)

  # Visualize embedding using PCA
  embedding = vis_utils.pca(tf.reshape(embedding, [1, height, width, args.embedding_dim]))
  embedding = ((embedding - tf.reduce_min(embedding)) / 
               (tf.reduce_max(embedding) - tf.reduce_min(embedding)))
  embedding = tf.cast(embedding * 255, tf.uint8)
  embedding = tf.squeeze(embedding, axis=0)

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
  cluster_dir = os.path.join(args.save_dir, 'cluster')
  embedding_dir = os.path.join(args.save_dir, 'embedding')
  patch_dir = os.path.join(args.save_dir, 'test_patches')
  if not os.path.isdir(pred_dir):
    os.makedirs(pred_dir)
  if not os.path.isdir(color_dir):
    os.makedirs(color_dir)
  if not os.path.isdir(cluster_dir):
    os.makedirs(cluster_dir)
  if not os.path.isdir(embedding_dir):
    os.makedirs(embedding_dir)
  if not os.path.isdir(patch_dir):
    os.makedirs(patch_dir)
    
  # Iterate over testing steps.
  with open(args.data_list, 'r') as listf:
    num_steps = len(listf.read().split('\n'))-1

  # Load prototype features and labels
  prototype_features_np = np.load(
      os.path.join(args.prototype_dir, 'prototype_features.npy'))
  prototype_labels_np = np.load(
      os.path.join(args.prototype_dir, 'prototype_labels.npy'))

  feed_dict = {prototype_features: prototype_features_np,
               prototype_labels: prototype_labels_np}

  f = html_helper.open_html_for_write(os.path.join(args.save_dir, 'index.html'),
                                      'Visualization for Segment Collaging')
  for step in range(num_steps):
    image_np, semantic_predictions_np, cluster_labels_np, embedding_np, k_predictions_np = sess.run(
        [image, semantic_predictions, cluster_labels, embedding, k_predictions],
        feed_dict=feed_dict)

    imgname = os.path.basename(image_list[step])
    basename = imgname.replace('jpg', 'png')

    predname = os.path.join(pred_dir, basename)
    Image.fromarray(semantic_predictions_np, mode='L').save(predname)

    colorname = os.path.join(color_dir, basename)
    color = colormap[semantic_predictions_np]
    Image.fromarray(color, mode='RGB').save(colorname)

    clustername = os.path.join(cluster_dir, basename)
    cluster = colormap[cluster_labels_np]
    Image.fromarray(cluster, mode='RGB').save(clustername)

    embeddingname = os.path.join(embedding_dir, basename)
    Image.fromarray(embedding_np, mode='RGB').save(embeddingname)

    image_np = (image_np + IMG_MEAN).astype(np.uint8)
    for i in range(np.max(cluster_labels_np) + 1):
      image_temp = copy.deepcopy(image_np)
      image_temp[cluster_labels_np != i] = 0
      coords = np.where(cluster_labels_np == i)
      crop = image_temp[np.min(coords[0]):np.max(coords[0]), np.min(coords[1]):np.max(coords[1])]
      scipy.misc.imsave(patch_dir + '/' + basename + str(i).zfill(3) + '.png', crop)

    html_helper.write_vmf_to_html(f, './images/' + imgname, './labels/' + basename,
                                  './color/' + basename, './cluster/' + basename, 
                                  './embedding/' + basename, './test_patches/' + basename, './patches/', k_predictions_np)

    if (step + 1) % 100 == 0:
      print('Processed batches: ', (step + 1), '/', num_steps)

  html_helper.close_html(f)
  coord.request_stop()
  coord.join(threads)
    
if __name__ == '__main__':
    main()
