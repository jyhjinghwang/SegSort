"""Common utility functions for SegSort."""

import numpy as np
import tensorflow as tf


def normalize_embedding(embedding):
  """Normalizes embedding by L2 norm.

  This function is used to normalize embedding so that the embedding features
  lie on a unit hypersphere.

  Args:
    embedding: An N-D float tensor with feature embedding in the last dimension.

  Returns:
    An N-D float tensor with the same shape as input embedding with feature
      embedding normalized by L2 norm in the last dimension.
  """
  return embedding / tf.norm(embedding, axis=-1, keep_dims=True)


def calculate_prototypes_from_labels(embedding,
                                     labels,
                                     max_label=None):
  """Calculates prototypes from labels.

  This function calculates prototypes (mean direction) from embedding features
  for each label. This function is also used as the m-step in k-means
  clustering.

  Args:
    embedding: A 2-D or 4-D float tensor with feature embedding in the last
      dimension (embedding_dim).
    labels: An N-D int32 label map for each embedding pixel.
    max_label: The maximum value of the label map. Calculated on-the-fly if not
      specified.

  Returns:
    A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.
  """
  embedding = tf.reshape(embedding, [-1, tf.shape(embedding)[-1]])
  labels = tf.reshape(labels, [-1])

  if max_label is None:
    max_label = tf.reduce_max(labels) + 1
  one_hot_labels = tf.one_hot(labels, tf.cast(max_label, tf.int32))
  prototypes = tf.matmul(one_hot_labels, embedding, transpose_a=True)
  prototypes = normalize_embedding(prototypes)
  return prototypes


def find_nearest_prototypes(embedding, prototypes):
  """Finds the nearest prototype for each embedding pixel.

  This function calculates the index of nearest prototype for each embedding
  pixel. This function is also used as the e-step in k-means clustering.

  Args:
    embedding: An N-D float tensor with embedding features in the last
      dimension (embedding_dim).
    prototypes: A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.

  Returns:
    A 1-D int32 tensor with length `[num_pixels]` containing the index of the
      nearest prototype for each pixel.
  """
  embedding = tf.reshape(embedding, [-1, tf.shape(prototypes)[-1]])
  similarities = tf.matmul(embedding, prototypes, transpose_b=True)
  return tf.argmax(similarities, axis=1)


def kmeans_with_initial_labels(embedding,
                               initial_labels,
                               max_label=None,
                               iterations=10):
  """Performs the von-Mises Fisher k-means clustering with initial labels.

  Args:
    embedding: A 2-D float tensor with shape `[num_pixels, embedding_dim]`.
    initial_labels: A 1-D integer tensor with length [num_pixels]. K-means
      clustering will start with this cluster labels if provided.
    max_label: An integer for the maximum of labels.
    iterations: Number of iterations for the k-means clustering.

  Returns:
    A 1-D integer tensor of the cluster label for each pixel.
  """
  if max_label is None:
    max_label = tf.reduce_max(initial_labels) + 1
  labels = initial_labels
  for _ in range(iterations):
    # M-step of the vMF k-means clustering.
    prototypes = calculate_prototypes_from_labels(embedding, labels, max_label)
    # E-step of the vMF k-means clustering.
    labels = find_nearest_prototypes(embedding, prototypes)
  return labels


def kmeans(embedding, num_clusters, iterations=10):
  """Performs the von-Mises Fisher k-means clustering.

  Args:
    embedding: A 4-D float tensor with shape
      `[batch, height, width, embedding_dim]`.
    num_clusters: A list of 2 integers for number of clusters in y and x axes.
    iterations: Number of iterations for the k-means clustering.

  Returns:
    A 3-D integer tensor of the cluster label for each pixel with shape
      `[batch, height, width]`.
  """
#  shape = embedding.get_shape().as_list()
  shape = tf.shape(embedding)
  labels = initialize_cluster_labels(num_clusters, [shape[1], shape[2]])

  embedding = tf.reshape(embedding, [-1, shape[3]])
  labels = tf.reshape(labels, [-1])

  labels = kmeans_with_initial_labels(embedding, labels, iterations=iterations)

  labels = tf.reshape(labels, [shape[0], shape[1], shape[2]])
  return labels


def initialize_cluster_labels(num_clusters, img_dimensions):
  """Initializes uniform cluster labels for an image.

  This function is used to initialize cluster labels that uniformly partition
  a 2-D image.

  Args:
    num_clusters: A list of 2 integers for number of clusters in y and x axes.
    img_dimensions: A list of 2 integers for image's y and x dimension.

  Returns:
    A 2-D int32 tensor with shape specified by img_dimension.
  """
  yx_range = tf.cast(tf.ceil(tf.cast(img_dimensions, tf.float32) /
                             tf.cast(num_clusters, tf.float32)), tf.int32)
  y_labels = tf.reshape(tf.range(img_dimensions[0]) // yx_range[0], [-1, 1])
  x_labels = tf.reshape(tf.range(img_dimensions[1]) // yx_range[1], [1, -1])
  labels = y_labels + (tf.reduce_max(y_labels) + 1) * x_labels
  return labels


def generate_location_features(img_dimensions, feature_type='int'):
  """Calculates location features for an image.

  This function generates location features for an image. The 2-D location
  features range from 0 to 1 for y and x axes each.

  Args:
    img_dimensions: A list of 2 integers for image's y and x dimension.
    feature_type: The data type of location features, integer or float.

  Returns:
    A 3-D float32 tensor with shape `[img_dimension[0], img_dimension[1], 2]`.

  Raises:
    ValueError: Type of location features is neither 'int' nor 'float'.
  """
  if feature_type == 'int':
    y_features = tf.range(img_dimensions[0])
    x_features = tf.range(img_dimensions[1])
  elif feature_type == 'float':
    y_features = (tf.range(img_dimensions[0], dtype=tf.float32) /
                  img_dimensions[0])
    x_features = (tf.range(img_dimensions[1], dtype=tf.float32) /
                  img_dimensions[1])
  else:
    raise ValueError('Type of location features should be either int or float.')

  x_features, y_features = tf.meshgrid(x_features, y_features)
  location_features = tf.stack([y_features, x_features], axis=2)
  return location_features


def generate_location_features_np(img_dimensions):
  y_features = np.linspace(0, 1, img_dimensions[0])
  x_features = np.linspace(0, 1, img_dimensions[1])

  x_features, y_features = np.meshgrid(x_features, y_features)
  location_features = np.expand_dims(np.stack([y_features, x_features], axis=2), 0)
  return location_features

def prepare_prototype_labels(semantic_labels, instance_labels, offset=256):
  """Prepares prototype labels from semantic and instance labels.

  This function generates unique prototype labels from semantic and instance
  labels. Note that instance labels sometimes can be cluster labels.

  Args:
    semantic_labels: A 1-D integer tensor for semantic labels.
    instance_labels: A 1-D integer tensor for instance labels.
    offset: An integer for instance offset.

  Returns:
    unique_instance_labels: A 1-D integer tensor for unique instance labels with
      the same length as the input semantic labels.
    prototype_labels: A 1-D integer tensor for the semantic labels of
      prototypes with length as the number of unique instances.
  """
  instance_labels = tf.cast(instance_labels, tf.int64)
  semantic_labels = tf.cast(semantic_labels, tf.int64)
  prototype_labels, unique_instance_labels = tf.unique(
      tf.reshape(semantic_labels + instance_labels * offset, [-1]))

  unique_instance_labels = tf.cast(unique_instance_labels, tf.int32)
  prototype_labels = tf.cast(prototype_labels % offset, tf.int32)
  return unique_instance_labels, prototype_labels
