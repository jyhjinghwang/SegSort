"""Image reader for semantic segmentation."""
# This code is borrowed and modified from:
# https://github.com/DrSleep/tensorflow-deeplab-resnet/blob/master/deeplab_resnet/image_reader.py

import network.segsort.common_utils as common_utils
import numpy as np
import tensorflow as tf


def image_scaling(img, label):
  """Randomly scales the images between 0.5 to 1.5 times the original size.

  Args:
    img: A 4-D float tensor of images with shape 
      `[batch_size, height_in, width_in, channels]`.
    label: A 3-D integer tensor of labels with shape 
      `[batch_size, height_in, width_in]`.

  Returns:
    img: A 4-D float tensor of randomly scaled images with shape
      `[batch_size, height_out, width_out, channels]`. 
    label: A 3-D integer tensor of randomly scaled labels with shape
      `[batch_size, height_out, width_out]`.
  """
  scale = tf.random_uniform(
      [1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
  h_new = tf.to_int32(tf.to_float(tf.shape(img)[0]) * scale)
  w_new = tf.to_int32(tf.to_float(tf.shape(img)[1]) * scale)
  new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
  img = tf.image.resize_images(img, new_shape)
  # Rescale labels by nearest neighbor sampling.
  label = tf.image.resize_nearest_neighbor(
      tf.expand_dims(label, 0), new_shape)
  label = tf.squeeze(label, squeeze_dims=[0])
   
  return img, label


def image_mirroring(img, label):
  """Randomly horizontally mirrors the images and their labels.

  Args:
    img: A 4-D float tensor of images with shape
      `[batch_size, height, width, channels]`.
    label: A 3-D integer tensor of labels with shape
      `[batch_size, height, width]`.

  Returns:
    img: A 4-D float tensor of randomly mirrored images with shape
      `[batch_size, height, width, channels]`.
    label: A 3-D integer tensor of randomly mirrored labels with shape
      `[batch_size, height, width]`.
  """
  distort_left_right_random = tf.random_uniform(
      [1], 0, 1.0, dtype=tf.float32)
  distort_left_right_random = distort_left_right_random[0]

  mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
  mirror = tf.boolean_mask([0, 1, 2], mirror)
  img = tf.reverse(img, mirror)
  label = tf.reverse(label, mirror)

  return img, label


def crop_and_pad_image_and_labels(image,
                                  label,
                                  crop_h,
                                  crop_w,
                                  ignore_label=255,
                                  random_crop=True):
  """Randomly crops and pads the images and their labels.

  Args:
    img: A 4-D float tensor of images with shape
      `[batch_size, height_in, width_in, channels]`.
    label: A 3-D integer tensor of labels with shape
      `[batch_size, height_in, width_in]`.
    crop_h: An integer for the desired output height.
    crop_w: An integer for the desired output width.
    ignore_label: The semantic label to ignore.
    random_crop: A boolean that enables random cropping.

  Returns:
    img: A 4-D float tensor of (randomly) cropped images with shape
      `[batch_size, height_out, width_out, channels]`.
    label: A 4-D integer tensor of (randomly) cropped labels with shape
      `[batch_size, height_out, width_out, 1]`.
  """
  # Needs to be subtracted and later added due to 0 padding.
  label = tf.cast(label, dtype=tf.float32)
  label = label - ignore_label 

  # Concatenate images with labels, which makes random cropping easier.
  combined = tf.concat(axis=2, values=[image, label]) 
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined,
      0,
      0,
      tf.maximum(crop_h, image_shape[0]),
      tf.maximum(crop_w, image_shape[1]))
    
  last_image_dim = image.get_shape().as_list()[-1]
  last_label_dim = label.get_shape().as_list()[-1]

  if random_crop:
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w, last_image_dim + last_label_dim])
  else:
    combined_crop = tf.image.resize_image_with_crop_or_pad(
        combined_pad,
        crop_h,
        crop_w)

  img_crop = combined_crop[:, :, :last_image_dim]
  label_crop = combined_crop[:, :, last_image_dim:]
  label_crop = label_crop + ignore_label
  label_crop = tf.cast(label_crop, dtype=tf.int32)
    
  # Set static shape so that tensorflow knows shape at running. 
  img_crop.set_shape((crop_h, crop_w, last_image_dim))
  label_crop.set_shape((crop_h,crop_w, last_label_dim))

  return img_crop, label_crop  


def read_labeled_image_list(data_dir, data_list):
  """Reads txt file containing paths to images and ground truth masks.
    
  Args:
    data_dir: A string that specifies the root of images and masks.
    data_list: A string that specifies the relative locations of images and masks.
       
  Returns:
    images: A list of strings for image paths.
    masks: A list of strings for mask paths.
  """
  f = open(data_list, 'r')
  images = []
  masks = []
  for line in f:
    try:
      image, mask = line.strip("\n").split(' ')
    except ValueError: # Adhoc for test.
      image = mask = line.strip("\n")
    images.append(data_dir + image)
    masks.append(data_dir + mask)
  return images, masks


def read_unsup_labeled_image_list(data_dir, data_list):
  """Reads txt file containing paths to images and masks for unsupervised SegSort.

  Args:
    data_dir: A string that specifies the root of images and masks.
    data_list: A string that specifies the relative locations of images and masks.

  Returns:
    images: A list of strings for image paths.
    masks: A list of strings for mask paths.
    segs: A list of strings for boundary segmentation paths.
  """
  f = open(data_list, 'r')
  images = []
  masks = []
  segs = []
  for line in f:
    try:
      image, mask = line.strip("\n").split(' ')
    except ValueError: # Adhoc for test.
      image = mask = line.strip("\n")
    seg = mask.replace('segcls', 'hed')
    images.append(data_dir + image)
    masks.append(data_dir + mask)
    segs.append(data_dir + seg)
  return images, masks, segs


def read_images_from_disk_for_segsort(input_queue,
                                      input_size,
                                      random_scale,
                                      random_mirror,
                                      random_crop,
                                      ignore_label,
                                      img_mean,
                                      num_clusters=None):
  """Reads one image and its label from input queue and preprocesses them.
    
  Args:
    input_queue: A tensorflow queue with paths to the image and its mask.
    input_size: A tuple with entries of height and width. If None, return
      images of original size.
    random_scale: A boolean that enables random scaling.
    random_mirror: A boolean that enables random flipping.
    ignore_label: The semantic label to ignore.
    img_mean: A float vector indicating the mean colour values of RGB channels.
    num_clusters: An integer scalar for number of clusters along each axis.
      
  Returns:
    img: A 3-D float tensor of an image with shape `[height, width, 3]`.
    semantic_label: A 3-D integer tensor of a semantic mask with shape `[height, width, 1]`.
    cluster_label: A 3-D integer tensor of a cluster mask with shape `[height, width, 1]`.
    loc_feature: A 3-D float tensor of a location mask with shape `[height, width, 2]`.
    height: The height of the output image.
    width: The width of the output image.
  """
  img_contents = tf.read_file(input_queue[0])
  label_contents = tf.read_file(input_queue[1])
    
  img = tf.image.decode_jpeg(img_contents, channels=3)
  img = tf.cast(img, dtype=tf.float32)
  # Extract mean.
  img -= img_mean

  label = tf.image.decode_png(label_contents, channels=1)

  # Generate cluster label and location feature
  height = tf.shape(img)[0]
  width = tf.shape(img)[1]
  if num_clusters is None:
    num_clusters = [5, 5]
  cluster_label = tf.expand_dims(common_utils.initialize_cluster_labels(
      num_clusters, [height, width]), 2)
  loc_feature = common_utils.generate_location_features([height, width],
                                                         feature_type='int')
  label = tf.cast(label, tf.int32)
  label = tf.concat([label, cluster_label, loc_feature], axis=2)

  # Data augmentation
  if input_size is not None:
    h, w = input_size

    # Randomly scale the images and labels.
    if random_scale:
      img, label = image_scaling(img, label)

    # Randomly mirror the images and labels.
    if random_mirror:
      img, label = image_mirroring(img, label)

    # Randomly crops the images and labels.
    img, label = crop_and_pad_image_and_labels(
        img, label, h, w, ignore_label, random_crop)

  # Seperate label
  semantic_label = label[:, :, 0:1]
  cluster_label = label[:, :, 1:2]
  yx_features = label[:, :, 2:4]

  yx_features = tf.cast(yx_features, tf.float32)
  y_features = yx_features[:, :, 0:1] / tf.cast(height, tf.float32)
  x_features = yx_features[:, :, 1:2] / tf.cast(width, tf.float32)
  loc_feature = tf.concat([y_features, x_features], 2)

  return img, semantic_label, cluster_label, loc_feature, height, width


def read_images_from_disk(input_queue,
                          input_size,
                          random_scale,
                          random_mirror,
                          random_crop,
                          ignore_label,
                          img_mean):
  """Reads one image and its corresponding label and perform pre-processing.

  Args:
    input_queue: A tensorflow queue with paths to the image and its mask.
    input_size: A tuple with entries of height and width. If None, return
      images of original size.
    random_scale: enable/disable random_scale for randomly scaling images
      and their labels.
    random_mirror: enable/disable random_mirror for randomly and horizontally
      flipping images and their labels.
    ignore_label: A number indicating the index of label to ignore.
    img_mean: A vector indicating the mean colour values of RGB channels.

  Returns:
    Two tensors: the decoded image and its mask.
  """
  img_contents = tf.read_file(input_queue[0])
  label_contents = tf.read_file(input_queue[1])

  img = tf.image.decode_jpeg(img_contents, channels=3)
  img = tf.cast(img, dtype=tf.float32)
  # Extract mean.
  img -= img_mean

  label = tf.image.decode_png(label_contents, channels=1)

  if input_size is not None:
    h, w = input_size

    # Randomly scale the images and labels.
    if random_scale:
      img, label = image_scaling(img, label)

    # Randomly mirror the images and labels.
    if random_mirror:
      img, label = image_mirroring(img, label)

    # Randomly crops the images and labels.
    img, label = crop_and_pad_image_and_labels(
      img, label, h, w, ignore_label, random_crop
    )

  return img, label


def read_images_from_disk_for_unsup_segsort(input_queue,
                                            input_size,
                                            random_scale,
                                            random_mirror,
                                            random_crop,
                                            ignore_label,
                                            img_mean):
  """Reads one image from input queue and preprocesses them for unsupervised SegSort.

  Args:
    input_queue: A tensorflow queue with paths to the image and its mask.
    input_size: A tuple with entries of height and width. If None, return
      images of original size.
    random_scale: A boolean that enables random scaling.
    random_mirror: A boolean that enables random flipping.
    ignore_label: The semantic label to ignore.
    img_mean: A float vector indicating the mean colour values of RGB channels.

  Returns:
    img: A 3-D float tensor of an image with shape `[height, width, 3]`.
    label: A 3-D integer tensor of a semantic mask with shape 
      `[height, width, 1]`, not used in training.
    cluster_label: A 3-D integer tensor of a boudary mask with shape 
      `[height, width, 1]`.
  """
  img_contents = tf.read_file(input_queue[0])
  label_contents = tf.read_file(input_queue[1])
  cluster_label_contents = tf.read_file(input_queue[2])

  img = tf.image.decode_jpeg(img_contents, channels=3)
  img = tf.cast(img, dtype=tf.float32)
  # Extract mean.
  img -= img_mean

  label = tf.image.decode_png(label_contents, channels=1)
  cluster_label = tf.image.decode_png(cluster_label_contents,
                                      channels=1)
  label = tf.concat([label, cluster_label], axis=2)

  if input_size is not None:
    h, w = input_size

    # Randomly scale the images and labels.
    if random_scale:
      img, label = image_scaling(img, label)

    # Randomly mirror the images and labels.
    if random_mirror:
      img, label = image_mirroring(img, label)

    # Randomly crops the images and labels.
    img, label = crop_and_pad_image_and_labels(
      img, label, h, w, ignore_label, random_crop
    )

  cluster_label = label[:, :, 1:2]
  label = label[:, :, 0:1]

  return img, label, cluster_label


class ImageReader(object):
  """
  Reads images and corresponding into a Tensorflow queue.
  """
  def __init__(self, data_dir, data_list, input_size,
               random_scale, random_mirror, random_crop,
               ignore_label, img_mean):
    """
    Initializes an ImageReader.
          
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form
        '/path/to/image /path/to/mask'.
      input_size: a tuple with (height, width) values, to which all the
        images will be resized.
      random_scale: whether to randomly scale the images.
      random_mirror: whether to randomly mirror the images.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
    """
    self.data_dir = data_dir
    self.data_list = data_list
    self.input_size = input_size
          
    self.image_list, self.label_list = read_labeled_image_list(
        self.data_dir, self.data_list)
    self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
    self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
    self.queue = tf.train.slice_input_producer(
        [self.images, self.labels],
        shuffle=input_size is not None) # not shuffling if it is val
    self.image, self.label = read_images_from_disk(
        self.queue,
        self.input_size,
        random_scale,
        random_mirror,
        random_crop,
        ignore_label,
        img_mean) 

  def dequeue(self, num_elements):
    """Packs images and labels into a batch.
        
    Args:
      num_elements: A number indicating the batch size.
          
    Returns:
      image_batch: A 4-D float tensor of an image with shape 
        `[batch_size, height, width, 3]`.
      label_batch: A 4-D integer tensor of a semantic mask with shape
        `[batch_size, height, width, 1]`.
    """
    image_batch, label_batch = tf.train.batch(
        [self.image, self.label],
        num_elements,
        num_threads=2)
    return image_batch, label_batch


class SegSortImageReader(object):
  """
  Reads images and corresponding into a Tensorflow queue.
  """
  def __init__(self, data_dir, data_list, input_size,
               random_scale, random_mirror, random_crop,
               ignore_label, img_mean, num_clusters=None):
    """
    Initializes a SegSortImageReader.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form
        '/path/to/image /path/to/mask'.
      input_size: a tuple with (height, width) values, to which all the
        images will be resized.
      random_scale: whether to randomly scale the images.
      random_mirror: whether to randomly mirror the images.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      num_clusters: a tuple indicating how many clusters along each axis.
        Leave it as None would be initialized to [5, 5].
    """
    self.data_dir = data_dir
    self.data_list = data_list
    self.input_size = input_size

    self.image_list, self.label_list = read_labeled_image_list(
        self.data_dir, self.data_list)
    self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
    self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
    self.queue = tf.train.slice_input_producer(
        [self.images, self.labels],
        shuffle=(random_scale or random_mirror or random_crop)) # not shuffling if it is val
    self.image, self.label, self.cluster_label, self.loc_feature, self.height, self.width = (
        read_images_from_disk_for_segsort(
            self.queue,
            self.input_size,
            random_scale,
            random_mirror,
            random_crop,
            ignore_label,
            img_mean,
            num_clusters))
    
  def dequeue(self, num_elements):
    """Packs images and labels into a batch.

    Args:
      num_elements: A number indicating the batch size.

    Returns:
      image_batch: A 4-D float tensor of an image with shape
          `[batch_size, height, width, 3]`.
      label_batch: A 4-D integer tensor of a semantic mask with shape
          `[batch_size, height, width, 1]`.
      cluster_label_batch: A 4-D integer tensor of an initial cluster
          mask with shape `[batch_size, height, width, 1]`.
      loc_feature_batch: A 4-D float tensor of a 2D location feature mask
          with shape `[batch_size, height, width, 2]`.
    """
    image_batch, label_batch, cluster_label_batch, loc_feature_batch = tf.train.batch(
        [self.image, self.label, self.cluster_label, self.loc_feature],
        num_elements,
        num_threads=2)

    return image_batch, label_batch, cluster_label_batch, loc_feature_batch


class SegSortUnsupImageReader(object):
  """
  Reads images and corresponding contour masksinto a Tensorflow queue.
  """

  def __init__(self, data_dir, data_list, input_size,
               random_scale, random_mirror, random_crop,
               ignore_label, img_mean):
    """
    Initializes a SegSortUnsupImageReader.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form
        '/path/to/image /path/to/mask'.
      input_size: a tuple with (height, width) values, to which all the
        images will be resized.
      random_scale: whether to randomly scale the images.
      random_mirror: whether to randomly mirror the images.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
    """
    self.data_dir = data_dir
    self.data_list = data_list
    self.input_size = input_size

    self.image_list, self.label_list, self.cluster_label_list = (
        read_unsup_labeled_image_list(self.data_dir, self.data_list))
    self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
    self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
    self.cluster_labels = tf.convert_to_tensor(self.cluster_label_list,
                                               dtype=tf.string)
    self.queue = tf.train.slice_input_producer(
        [self.images, self.labels, self.cluster_labels],
        shuffle=(random_scale or random_mirror or random_crop)) # not shuffling if it is val
    self.image, self.label, self.cluster_label = (
        read_images_from_disk_for_unsup_segsort(
            self.queue,
            self.input_size,
            random_scale,
            random_mirror,
            random_crop,
            ignore_label,
            img_mean))

  def dequeue(self, num_elements):
    """Packs images and labels into a batch.

    Args:
      num_elements: A number indicating the batch size.

    Returns:
      image_batch: A 4-D float tensor of an image with shape
          `[batch_size, height, width, 3]`.
      label_batch: A 4-D integer tensor of a semantic mask with shape
          `[batch_size, height, width, 1]`.
      cluster_label_batch: A 4-D integer tensor of a contour segmentation
          mask with shape `[batch_size, height, width, 1]`.
    """
    image_batch, label_batch, cluster_label_batch = tf.train.batch(
        [self.image, self.label, self.cluster_label],
        num_elements,
        num_threads=2)

    return image_batch, label_batch, cluster_label_batch
