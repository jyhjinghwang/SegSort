"""Utility functions for visualizing embeddings."""

import tensorflow as tf


def calculate_principal_components(embedding, num_components=3):
  """Calculates the principal components given the embedding features.

  Args:
    embedding: A 2-D float tensor with embedding features in the last dimension.
    num_components: the number of principal components to return.

  Returns:
    A 2-D float tensor with principal components in the last dimension.
  """
  embedding -= tf.reduce_mean(embedding, axis=0, keep_dims=True)
  sigma = tf.matmul(embedding, embedding, transpose_a=True)
  _, u, _ = tf.svd(sigma)
  return u[:, :num_components]


def pca(embedding, num_components=3, principal_components=None):
  """Conducts principal component analysis on the embedding features.

  This function is used to reduce the dimensionality of the embedding, so that
  we can visualize the embedding as an RGB image.

  Args:
    embedding: A 4-D float tensor with shape
      [batch, height, width, embedding_dims].
    num_components: The number of principal components to be reduced to.
    principal_components: A 2-D float tensor used to convert the embedding
      features to PCA'ed space, also known as the U matrix from SVD. If not
      given, this function will calculate the principal_components given inputs.

  Returns:
    A 4-D float tensor with shape [batch, height, width, num_components].
  """
#  shape = embedding.get_shape().as_list()
  shape = tf.shape(embedding)
  embedding = tf.reshape(embedding, [-1, shape[3]])

  if principal_components is None:
    principal_components = calculate_principal_components(embedding,
                                                          num_components)
  embedding = tf.matmul(embedding, principal_components)

  embedding = tf.reshape(embedding,
                         [shape[0], shape[1], shape[2], num_components])
  return embedding
