from __future__ import print_function

import argparse
import os
import time
import math

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
import cv2
import network.vmf.common_utils as common_utils
import network.vmf.eval_utils as eval_utils

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
      description='Inference of Semantic Segmentation.')
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
  parser.add_argument('--flip-aug', action='store_true',
                      help='Augment data by horizontal flipping.')
  parser.add_argument('--scale-aug', action='store_true',
                      help='Augment data with multi-scale.')
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
  """Create the model and start the inference process.
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
    image_list = reader.image_list
  image_batch = tf.expand_dims(image, dim=0)

  # Create multi-scale augmented datas.
  rescale_image_batches = []
  is_flipped = []
  scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75] if args.scale_aug else [1]
  for scale in scales:
    h_new = tf.to_int32(
        tf.multiply(tf.to_float(tf.shape(image_batch)[1]), scale))
    w_new = tf.to_int32(
        tf.multiply(tf.to_float(tf.shape(image_batch)[2]), scale))
    new_shape = tf.stack([h_new, w_new])
    new_image_batch = tf.image.resize_images(image_batch, new_shape)
    rescale_image_batches.append(new_image_batch)
    is_flipped.append(False)

  # Create horizontally flipped augmented datas.
  if args.flip_aug:
    for i in range(len(scales)):
      img = rescale_image_batches[i]
      is_flip = is_flipped[i]
      img = tf.squeeze(img, axis=0)
      flip_img = tf.image.flip_left_right(img)
      flip_img = tf.expand_dims(flip_img, axis=0)
      rescale_image_batches.append(flip_img)
      is_flipped.append(True)

  # Create input tensor to the Network
  crop_image_batch = tf.placeholder(
      name='crop_image_batch',
      shape=[1,input_size[0],input_size[1],3],
      dtype=tf.float32)

  # Create network.
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

  embedding_input = tf.placeholder(
      tf.float32, shape=[1, None, None, args.embedding_dim])
  embedding = common_utils.normalize_embedding(embedding_input)
  loc_feature = tf.placeholder(
      tf.float32, shape=[1, None, None, 2])

  # Prototype placeholders.
  prototype_features = tf.placeholder(tf.float32,
                                      shape=[None, args.embedding_dim])
  prototype_labels = tf.placeholder(tf.int32)
  
  # Combine embedding with location features and kmeans
  shape = tf.shape(embedding)#.get_shape().as_list()
#  loc_feature = tf.expand_dims(
#      common_utils.generate_location_features([shape[1], shape[2]], 'float'), 0)
  embedding_with_location = tf.concat([embedding, loc_feature], 3)
  embedding_with_location = common_utils.normalize_embedding(
      embedding_with_location)
  cluster_labels = common_utils.kmeans(
      embedding_with_location,
      [args.num_clusters, args.num_clusters],
      args.kmeans_iterations)
  _, cluster_labels = tf.unique(tf.reshape(cluster_labels, [-1]))
  test_prototypes = common_utils.calculate_prototypes_from_labels(
      embedding, cluster_labels)

  # Predict semantic labels.
  similarities = tf.matmul(test_prototypes,
                           prototype_features,
                           transpose_b=True)
  _, k_predictions = tf.nn.top_k(similarities, k=args.k_in_nearest_neighbors, sorted=True)
  k_predictions = tf.gather(prototype_labels, k_predictions)
  k_predictions = tf.gather(k_predictions, cluster_labels)
  k_predictions = tf.reshape(k_predictions, 
                             [shape[0], shape[1], shape[2], args.k_in_nearest_neighbors])
    
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

  # Load prototype features and labels
  prototype_features_np = np.load(
      os.path.join(args.prototype_dir, 'prototype_features.npy'))
  prototype_labels_np = np.load(
      os.path.join(args.prototype_dir, 'prototype_labels.npy'))
  feed_dict = {prototype_features: prototype_features_np,
               prototype_labels: prototype_labels_np}

  for step in range(num_steps):
    rescale_img_batches = sess.run(rescale_image_batches)
    # Final segmentation results (average across multiple scales).
    scale_ind = 2 if args.scale_aug else 0
    final_lab_size = list(rescale_img_batches[scale_ind].shape[1:])
    final_lab_size[-1] = args.num_classes
    final_lab_batch = np.zeros(final_lab_size)

    # Iterate over multiple scales.
    for img_batch,is_flip in zip(rescale_img_batches, is_flipped):
      img_size = img_batch.shape
      padimg_size = list(img_size) # deep copy of img_size

      padimg_h, padimg_w = padimg_size[1:3]
      input_h, input_w = input_size

      if input_h > padimg_h:
        padimg_h = input_h
      if input_w > padimg_w:
        padimg_w = input_w
      # Update padded image size.
      padimg_size[1] = padimg_h
      padimg_size[2] = padimg_w
      padimg_batch = np.zeros(padimg_size, dtype=np.float32)
      img_h, img_w = img_size[1:3]
      padimg_batch[:, :img_h, :img_w, :] = img_batch

      stride_h, stride_w = strides
      npatches_h = math.ceil(1.0*(padimg_h-input_h)/stride_h) + 1
      npatches_w = math.ceil(1.0*(padimg_w-input_w)/stride_w) + 1

      # Create padded prediction array.
      pred_size = list(padimg_size)
      pred_size[-1] = args.num_classes
      predictions_np = np.zeros(pred_size, dtype=np.int32)

      # Create the ending index of each patch.
      patch_indh = np.linspace(
          input_h, padimg_h, npatches_h, dtype=np.int32)
      patch_indw = np.linspace(
          input_w, padimg_w, npatches_w, dtype=np.int32)

      pred_size[-1] = args.embedding_dim
      embedding_all_np = np.zeros(pred_size, dtype=np.float32)
      for indh in patch_indh:
        for indw in patch_indw:
          sh, eh = indh-input_h, indh  # start & end ind of H
          sw, ew = indw-input_w, indw  # start & end ind of W
          cropimg_batch = padimg_batch[:, sh:eh, sw:ew, :]
#          feed_dict[crop_image_batch] = cropimg_batch

          embedding_np = sess.run(output, feed_dict={
              crop_image_batch: cropimg_batch})
          embedding_all_np[:, sh:eh, sw:ew, :] += embedding_np

      loc_feature_np = common_utils.generate_location_features_np([padimg_h, padimg_w])
      feed_dict[embedding_input] = embedding_all_np
      feed_dict[loc_feature] = loc_feature_np
      k_predictions_np = sess.run(k_predictions, feed_dict=feed_dict)
      for c in range(args.num_classes):
        predictions_np[:, :, :, c] += np.sum(
            (k_predictions_np == c).astype(np.int), axis=3)

      predictions_np = predictions_np[0, :img_h, :img_w, :]
      lab_batch = predictions_np.astype(np.float32)
#      lab_batch = np.argmax(pred_batch, axis=2).astype(np.float32)
      # Rescale prediction back to original resolution.
      lab_batch = cv2.resize(
          lab_batch,
          (final_lab_size[1], final_lab_size[0]),
          interpolation=cv2.INTER_LINEAR)
      if is_flip:
        # Flipped prediction back to original orientation.
        lab_batch = lab_batch[:, ::-1, :]
      final_lab_batch += lab_batch

    final_lab_ind = np.argmax(final_lab_batch, axis=-1)
    final_lab_ind = final_lab_ind.astype(np.uint8)

    basename = os.path.basename(image_list[step])
    basename = basename.replace('jpg', 'png')

    predname = os.path.join(pred_dir, basename)
    Image.fromarray(final_lab_ind, mode='L').save(predname)

    colorname = os.path.join(color_dir, basename)
    color = colormap[final_lab_ind]
    Image.fromarray(color, mode='RGB').save(colorname)

    if (step + 1) % 100 == 0:
      print('Processed batches: ', (step + 1), '/', num_steps)

  coord.request_stop()
  coord.join(threads)
    
if __name__ == '__main__':
    main()
