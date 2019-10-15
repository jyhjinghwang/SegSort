"""Utility functions for SegSort evaluation."""

import network.segsort.common_utils as common_utils
import tensorflow as tf


def find_majority_label_index(labels, cluster_labels):
  """Finds indices of pixels that belong to their majority label in a cluster.

  Args:
    labels: A 1-D integer tensor for panoptic or semantic labels.
    cluster_labels: A 1-D integer tensor for cluster labels.

  Returns:
    select_pixels: A 2-D integer tensor for indices of pixels that belong to
      their majority label in a cluster.
    majority_labels: A 1-D integer tensor for the panoptic or semantic
      label for each cluster with length `[num_clusters]`.
  """
  one_hot_labels = tf.one_hot(
      labels, tf.cast(tf.reduce_max(labels) + 1, tf.int32))
  one_hot_cluster_labels = tf.one_hot(
      cluster_labels, tf.cast(tf.reduce_max(cluster_labels) + 1, tf.int32))

  accumulate_labels = tf.matmul(one_hot_cluster_labels,
                                one_hot_labels,
                                transpose_a=True)
  majority_labels = tf.cast(tf.argmax(accumulate_labels, axis=1), tf.int32)

  semantic_cluster_labels = tf.gather(majority_labels, cluster_labels)
  select_pixels = tf.where(tf.equal(semantic_cluster_labels, labels))
  return select_pixels, majority_labels


def extract_trained_prototypes(embedding,
                               location_features,
                               cluster_labels,
                               num_clusters,
                               kmeans_iterations,
                               panoptic_labels,
                               panoptic_label_divisor,
                               ignore_label,
                               evaluate_semantic_or_panoptic):
  """Extracts the trained prototypes in an image.

  Args:
    embedding: A 2-D float tensor with shape `[pixels, embedding_dim]`.
    location_features: A 2-D float tensor for location features with shape
      `[pixels, 2]`.
    cluster_labels: A 1-D integer tensor for cluster labels for all pixels.
    num_clusters: An integer scalar for total number of clusters.
    kmeans_iterations: Number of iterations for the k-means clustering.
    panoptic_labels: A 1-D integer tensor for panoptic labels for all pixels.
    panoptic_label_divisor: An integer constant to separate semantic and
      instance labels from panoptic labels.
    ignore_label: The semantic label to ignore.
    evaluate_semantic_or_panoptic: A boolean that specifies whether to evaluate
      semantic or panoptic segmentation.

  Returns:
    prototype_features: A 2-D float tensor for prototype features with shape
      `[num_prototypes, embedding_dim]`.
    prototype_labels: A 1-D integer tensor for prototype labels.
  """
  # Collect pixels of valid semantic classes.
  valid_pixels = tf.where(
      tf.not_equal(panoptic_labels // panoptic_label_divisor, ignore_label))
  panoptic_labels = tf.squeeze(tf.gather(panoptic_labels, valid_pixels), axis=1)
  cluster_labels = tf.squeeze(tf.gather(cluster_labels, valid_pixels), axis=1)
  embedding = tf.squeeze(tf.gather(embedding, valid_pixels), axis=1)
  location_features = tf.squeeze(
      tf.gather(location_features, valid_pixels), axis=1)

  # Generate cluster labels via kmeans clustering.
  embedding_with_location = tf.concat([embedding, location_features], 1)
  embedding_with_location = common_utils.normalize_embedding(
      embedding_with_location)
  cluster_labels = common_utils.kmeans_with_initial_labels(
      embedding_with_location,
      cluster_labels,
      num_clusters,
      kmeans_iterations)
  _, cluster_labels = tf.unique(cluster_labels)

  if evaluate_semantic_or_panoptic == 'panoptic':
    # Calculate semantic and unique instance labels for all pixels.
    label_mapping, unique_panoptic_labels = tf.unique(panoptic_labels)

    # Find pixels of majority classes.
    select_pixels, majority_labels = find_majority_label_index(
        unique_panoptic_labels, cluster_labels)
  else:
    # Find pixels of majority semantic classes.
    semantic_labels = panoptic_labels // panoptic_label_divisor
    select_pixels, majority_labels = find_majority_label_index(
        semantic_labels, cluster_labels)

  cluster_labels = tf.squeeze(tf.gather(cluster_labels, select_pixels), axis=1)
  embedding = tf.squeeze(tf.gather(embedding, select_pixels), axis=1)

  # Calculate the majority semantic and instance label for each prototype.
  if evaluate_semantic_or_panoptic == 'panoptic':
    prototype_panoptic_labels = tf.gather(label_mapping, majority_labels)
    prototype_semantic_labels = (prototype_panoptic_labels //
                                 panoptic_label_divisor)
    prototype_instance_labels = majority_labels
  else:
    prototype_semantic_labels = majority_labels
    prototype_instance_labels = tf.zeros_like(majority_labels)

  # Calculate the prototype features.
  prototype_features = common_utils.calculate_prototypes_from_labels(
      embedding, cluster_labels)

  return (prototype_features,
          prototype_semantic_labels,
          prototype_instance_labels)


def k_nearest_neighbors(k_predictions, prototype_labels):
  """Predicts the majority labels given the top k predictions.

  Args:
    k_predictions: Top k nearest prototypes.
    prototype_labels: A 1-D integer tensor for semantic or instance labels of
      trained prototypes with length `[num_prototypes]`.

  Returns:
    prototype_predictions: A 1-D integer tensor for the label prediction for
      each protoype with length `[num_prototypes]`.
  """
  k_predictions = tf.gather(prototype_labels, k_predictions)
  k_predictions = tf.one_hot(k_predictions, tf.reduce_max(k_predictions)+1)

  prototype_predictions = tf.argmax(tf.reduce_sum(k_predictions, axis=1),
                                    axis=1, output_type=tf.int32)
  return prototype_predictions


def predict_semantic_instance_labels(cluster_labels,
                                     test_prototypes,
                                     prototype_features,
                                     prototype_semantic_labels,
                                     prototype_instance_labels,
                                     k=1):
  """Predicts semantic label for every pixel in different clusters.

  This function predicts semantic label for every pixel by using k-nearest
  neighbors, searching through training prototypes.

  Args:
    cluster_labels: A 3-D integer tensor for cluster labels with shape
      `[batch, height, width]`.
    test_prototypes: A 2-D float tensor for prototypes in all clusters with
      shape `[num_clusters, embedding_dim]`.
    prototype_features: A 2-D float tensor for training prototypes with the
      embedding features in the last dimension.
    prototype_semantic_labels: A 1-D integer tensor for trained prototype
      semantic labels with length `[num_prototypes]`.
    prototype_instance_labels: A 1-D integer tensor for trained prototype
      instance labels with length `[num_prototypes]`.
    k: The number of nearest neighbors to search, or k in k-nearest neighbors.

  Returns:
    semantic_predictions: A 3-D integer tensor for the semantic prediction for
      each pixel with shape `[batch, height, width]`.
    semantic_predictions: A 3-D integer tensor for the instance prediction for
      each pixel with shape `[batch, height, width]`.
  """
  similarities = tf.matmul(test_prototypes,
                           prototype_features,
                           transpose_b=True)
  _, k_predictions = tf.nn.top_k(similarities, k=k, sorted=False)

  prototype_semantic_predictions = k_nearest_neighbors(
      k_predictions, prototype_semantic_labels)
  semantic_predictions = tf.gather(prototype_semantic_predictions,
                                   cluster_labels)

  if prototype_instance_labels is None:
    instance_predictions = None
  else:
    prototype_instance_predictions = k_nearest_neighbors(
        k_predictions, prototype_instance_labels)
    _, prototype_instance_predictions = tf.unique(
        prototype_instance_predictions)
    instance_predictions = tf.gather(prototype_instance_predictions,
                                     cluster_labels)

  return semantic_predictions, instance_predictions


def predict_all_labels(embedding,
                       num_clusters,
                       kmeans_iterations,
                       prototype_features,
                       prototype_semantic_labels,
                       prototype_instance_labels,
                       k_in_nearest_neighbors,
                       panoptic_label_divisor,
                       class_has_instances_list):
  """Predicts panoptic, semantic, and instance labels using the vMF embedding.

  Args:
    embedding: A 4-D float tensor with shape
      `[batch, height, width, embedding_dim]`.
    num_clusters: A list of 2 integers for number of clusters in y and x axes.
    kmeans_iterations: Number of iterations for the k-means clustering.
    prototype_features: A 2-D float tensor for trained prototype features with
      shape `[num_prototypes, embedding_dim]`.
    prototype_semantic_labels: A 1-D integer tensor for trained prototype
      semantic labels with length `[num_prototypes]`.
    prototype_instance_labels: A 1-D integer tensor for trained prototype
      instance labels with length `[num_prototypes]`.
    k_in_nearest_neighbors: The number of nearest neighbors to search,
      or k in k-nearest neighbors.
    panoptic_label_divisor: An integer constant to separate semantic and
      instance labels from panoptic labels.
    class_has_instances_list: A list of thing classes, which have instances.

  Returns:
    panoptic_predictions: A 1-D integer tensor for pixel panoptic predictions.
    semantic_predictions: A 1-D integer tensor for pixel semantic predictions.
    instance_predictions: A 1-D integer tensor for pixel instance predictions.
  """
  # Generate location features and combine them with embedding features.
  shape = embedding.get_shape().as_list()
  location_features = common_utils.generate_location_features(
      [shape[1], shape[2]], 'float')
  location_features = tf.expand_dims(location_features, 0)
  embedding_with_location = tf.concat([embedding, location_features], 3)
  embedding_with_location = common_utils.normalize_embedding(
      embedding_with_location)

  # Kmeans clustering.
  cluster_labels = common_utils.kmeans(
      embedding_with_location,
      num_clusters,
      kmeans_iterations)
  test_prototypes = common_utils.calculate_prototypes_from_labels(
      embedding, cluster_labels)

  # Predict semantic and instance labels.
  semantic_predictions, instance_predictions = predict_semantic_instance_labels(
      cluster_labels,
      test_prototypes,
      prototype_features,
      prototype_semantic_labels,
      prototype_instance_labels,
      k_in_nearest_neighbors)

  # Refine instance labels.
  class_has_instances_list = tf.reshape(class_has_instances_list, [1, 1, 1, -1])
  instance_predictions = tf.where(
      tf.reduce_all(tf.not_equal(tf.expand_dims(semantic_predictions, 3),
                                 class_has_instances_list), axis=3),
      tf.zeros_like(instance_predictions),
      instance_predictions)

  # Combine semantic and panoptic predictions as panoptic predictions.
  panoptic_predictions = (semantic_predictions * panoptic_label_divisor +
                          instance_predictions)

  return (panoptic_predictions,
          semantic_predictions,
          instance_predictions,
          cluster_labels)


def oracle_region(embedding, labels):
  """Finds the oracle region for debugging purpose.

  Args:
    embedding: A 4-D float tensor with shape
      `[batch, height, width, embedding_dim]`.
    labels: A 3-D integer tensor for ground truth label maps.

  Returns:
    cluster_labels: A 3-D integer tensor for the cluster label for each pixel
      with shape `[batch, height, width]`.
    prototypes: A 2-D float tensor for cluster prototypes with embedding
      features in the last dimension.
  """
  shape = embedding.get_shape().as_list()
  _, cluster_labels = tf.unique(tf.reshape(labels, [-1]))

  one_hot_labels = tf.one_hot(cluster_labels, tf.reduce_max(labels) + 1)
  prototypes = tf.matmul(one_hot_labels,
                         tf.reshape(embedding, [-1, shape[3]]),
                         transpose_a=True)

  cluster_labels = tf.reshape(cluster_labels, [shape[0], shape[1], shape[2]])
  return cluster_labels, prototypes


def predict_oracle_labels(cluster_labels, labels, ignore_label=255):
  """Predicts semantic clusters with oracle labels for debugging purpose.

  Args:
    cluster_labels: A 3-D integer tensor for cluster labels with shape
      `[batch, height, width]`.
    labels: A 3-D integer tensor for ground truth labels with shape
      `[batch, height, width]`.
    ignore_label: A integer constant specifying the ignored label.

  Returns:
    semantic_predictions: A 3-D integer tensor for the semantic prediction for
      each pixel with shape `[batch, height, width]`.
  """
  shape = tf.shape(cluster_labels)
  cluster_labels = tf.cast(tf.reshape(cluster_labels, [-1]), tf.int32)
  labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)

  # Treat void labels as background class.
  labels = tf.where(tf.equal(labels, ignore_label),
                    tf.zeros_like(labels),
                    labels)

  one_hot_cluster_labels = tf.one_hot(cluster_labels,
                                      tf.reduce_max(cluster_labels) + 1)
  one_hot_labels = tf.one_hot(labels, tf.reduce_max(labels) + 1)

  region_semantic = tf.matmul(one_hot_cluster_labels,
                              one_hot_labels,
                              transpose_a=True)
  region_semantic = tf.argmax(region_semantic, axis=1)

  semantic_predictions = tf.gather(region_semantic, cluster_labels)
  semantic_predictions = tf.reshape(semantic_predictions, shape)
  return semantic_predictions
