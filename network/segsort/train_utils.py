"""Utility functions for training with SegSort."""

import tensorflow as tf

from network.segsort import common_utils


def add_segsort_loss(embedding,
                     labels,
                     embedding_dim,
                     ignore_label,
                     concentration,
                     cluster_labels,
                     num_clusters,
                     kmeans_iterations,
                     num_banks,
                     loc_features,
                     loss_scope=None):
  """Adds SegSort loss for logits."""

  with tf.name_scope(loss_scope, 'segsort_loss',
                     (embedding, labels, embedding_dim, ignore_label,
                      concentration, cluster_labels, num_clusters,
                      kmeans_iterations, num_banks, loc_features)):
    # Normalize embedding.
    embedding = common_utils.normalize_embedding(embedding)

    label_divisor = 256
    num_total_clusters = num_clusters * num_clusters

    shape = embedding.get_shape().as_list()
    batch_size = shape[0]
    labels = tf.reshape(labels, shape=[batch_size, -1])
    cluster_labels = tf.reshape(cluster_labels, [batch_size, -1])
    semantic_labels = labels

    # Process labels and embedding for each image.
    # Note that the for loop is necessary to separate k-means clustering.
    semantic_labels_list = []
    cluster_labels_list = []
    embedding_list = []
    for bs in range(batch_size):
      cur_semantic_labels = tf.reshape(semantic_labels[bs], [-1])
      cur_cluster_labels = tf.reshape(cluster_labels[bs], [-1])
      cur_loc_features = tf.reshape(loc_features[bs], [-1, 2])
      cur_embedding = tf.reshape(embedding[bs], [-1, embedding_dim])

      valid_pixels = tf.where(tf.not_equal(cur_semantic_labels, ignore_label))
      cur_semantic_labels = tf.squeeze(tf.gather(cur_semantic_labels,
                                                 valid_pixels), axis=1)
      cur_cluster_labels = tf.squeeze(tf.gather(cur_cluster_labels,
                                                valid_pixels), axis=1)
      cur_loc_features = tf.squeeze(tf.gather(cur_loc_features, valid_pixels), axis=1)
      cur_embedding = tf.squeeze(tf.gather(cur_embedding, valid_pixels), axis=1)

      embedding_with_loc = tf.concat([cur_embedding, cur_loc_features], 1)
      embedding_with_loc = common_utils.normalize_embedding(embedding_with_loc)
      cur_cluster_labels = common_utils.kmeans_with_initial_labels(
          embedding_with_loc,
          cur_cluster_labels,
          num_total_clusters,
          kmeans_iterations)
      # Add offset to the cluster labels to separate clusters of different
      # images. The offset should be greater than number of total clusters.
      cur_cluster_labels += bs * label_divisor

      semantic_labels_list.append(cur_semantic_labels)
      cluster_labels_list.append(cur_cluster_labels)
      embedding_list.append(cur_embedding)
    semantic_labels = tf.concat(semantic_labels_list, 0)
    cluster_labels = tf.concat(cluster_labels_list, 0)
    embedding = tf.concat(embedding_list, 0)
    unique_instance_labels, prototype_labels = (
        common_utils.prepare_prototype_labels(semantic_labels,
                                              cluster_labels,
                                              label_divisor))

    # Retrieve the memory banks if specified.
    if num_banks > 0:
      num_prototypes = num_total_clusters * batch_size
      memory_list, memory_labels_list = _retrieve_memory(num_prototypes,
                                                         num_banks,
                                                         embedding_dim)
      memory = tf.concat(memory_list, axis=0)
      memory_labels = tf.concat(memory_labels_list, axis=0)
    else:
      memory = memory_labels = None

    loss, new_memory, new_memory_labels = _calculate_vmf_loss(
        embedding,
        semantic_labels,
        unique_instance_labels,
        prototype_labels,
        concentration,
        memory,
        memory_labels,
        batch_size)

    # Update the memory bank to cache prototypes in the current batch.
    update_memory_list = []
    if num_banks > 0:
      for b in range(num_banks-1):
        update_memory_list.append(tf.assign(memory_list[b], memory_list[b+1]))
        update_memory_list.append(tf.assign(memory_labels_list[b],
                                            memory_labels_list[b+1]))
      update_memory_list.append(tf.assign(memory_list[num_banks-1],
                                          new_memory[:num_prototypes, :]))
      update_memory_list.append(tf.assign(memory_labels_list[num_banks-1],
                                          new_memory_labels[:num_prototypes]))

    with tf.control_dependencies(update_memory_list):
      return loss


def add_unsupervised_segsort_loss(embedding,
                                  concentration,
                                  cluster_labels,
                                  num_banks=0,
                                  loss_scope=None):
  with tf.name_scope(loss_scope, 'unsupervised_segsort_loss',
                     (embedding, concentration, cluster_labels,
                      num_banks)):

    # Normalize embedding.
    embedding = common_utils.normalize_embedding(embedding)
    shape = embedding.get_shape().as_list()
    batch_size = shape[0]
    embedding_dim = shape[3]

    # Add offset to cluster labels.
    max_clusters = 256
    offset = tf.range(0, max_clusters * batch_size, max_clusters)
    cluster_labels += tf.reshape(offset, [-1, 1, 1, 1])
    _, cluster_labels = tf.unique(tf.reshape(cluster_labels, [-1]))

    # Calculate prototypes.
    embedding = tf.reshape(embedding, [-1, embedding_dim])
    prototypes = common_utils.calculate_prototypes_from_labels(
      embedding, cluster_labels)

    similarities = _calculate_similarities(embedding, prototypes,
                                           concentration, batch_size)

    # Calculate the unsupervised loss.
    self_indices = tf.concat(
      [tf.expand_dims(tf.range(tf.shape(similarities)[0]), 1),
       tf.expand_dims(cluster_labels, 1)], axis=1)
    numerator = tf.reshape(tf.gather_nd(similarities, self_indices), [-1])
    denominator = tf.reduce_sum(similarities, axis=1)

    probabilities = tf.divide(numerator, denominator)
    return tf.reduce_mean(-tf.log(probabilities))


def _calculate_vmf_loss(embedding, semantic_labels, unique_instance_labels,
                        prototype_labels, concentration,
                        memory=None, memory_labels=None, split=1):
  """Calculates the von-Mises Fisher loss given semantic and cluster labels.

  Args:
    embedding: A 2-D float tensor with shape [num_pixels, embedding_dim].
    semantic_labels: A 1-D integer tensor with length [num_pixels]. It contains
      the semantic label for each pixel.
    unique_instance_labels: A 1-D integer tensor with length [num_pixels]. It
      contains the unique instance label for each pixel.
    prototype_labels: A 1-D integer tensor with length [num_prototypes]. It
      contains the semantic label for each prototype.
    concentration: A float that controls the sharpness of cosine similarities.
    memory: A 2-D float tensor for memory prototypes with shape
      `[num_prototypes, embedding_dim]`.
    memory_labels: A 1-D integer tensor for labels of memory prototypes with
      length `[num_prototypes]`.
    split: An integer for number of splits of matrix multiplication.

  Returns:
    loss: A float for the von-Mises Fisher loss.
    new_memory: A 2-D float tensor for the memory prototypes to update with
      shape `[num_prototypes, embedding_dim]`.
    new_memory_labels: A 1-D integer tensor for labels of memory prototypes to
      update with length `[num_prototypes]`.
  """
  prototypes = common_utils.calculate_prototypes_from_labels(
      embedding, unique_instance_labels, tf.size(prototype_labels))

  if memory is not None:
    memory = common_utils.normalize_embedding(memory)
    rand_index = tf.random_shuffle(tf.range(tf.shape(prototype_labels)[0]))
    new_memory = tf.squeeze(tf.gather(prototypes, rand_index))
    new_memory_labels = tf.squeeze(tf.gather(prototype_labels, rand_index))
    prototypes = tf.concat([prototypes, memory], 0)
    prototype_labels = tf.concat([prototype_labels, memory_labels], 0)
  else:
    new_memory = new_memory_labels = None

  similarities = _calculate_similarities(embedding, prototypes,
                                         concentration, split)

  log_likelihood = _calculate_log_likelihood(
      similarities, unique_instance_labels, semantic_labels, prototype_labels)

  loss = tf.reduce_mean(log_likelihood)
  return loss, new_memory, new_memory_labels


def _calculate_log_likelihood(
    similarities, unique_instance_labels, semantic_labels, prototype_labels,
    use_nca=True):
  """Calculates log-likelihood of each pixel belonging to a certain cluster.

  This function calculates log-likelihood of each pixel belonging to a certain
  cluster. This log-likelihood is then used as maximum likelihood estimation.

  Args:
    similarities: A 2-D float tensor with shape [num_pixesl, num_prototypes].
    unique_instance_labels: A 1-D integer tensor with length [num_pixels]. It
      contains the unique instance label for each pixel.
    semantic_labels: A 1-D integer tensor with length [num_pixels]. It contains
      the semantic label for each pixel.
    prototype_labels: A 1-D integer tensor with length [num_prototypes]. It
      contains the semantic label for each prototype
    use_nca: A boolean that specifies whether to use the neighborhood component
      analysis to estimate the likelihood.

  Returns:
    log_likelihood: A 1-D float tensor with length [num_pixels]. It is the
      negative log-likelihood for each pixel.
  """
  semantic_labels = tf.reshape(semantic_labels, [-1, 1])
  prototype_labels = tf.reshape(prototype_labels, [1, -1])

  self_indices = tf.concat(
      [tf.expand_dims(tf.range(tf.shape(similarities)[0]), 1),
       tf.expand_dims(unique_instance_labels, 1)], axis=1)
  self_similarities = tf.reshape(tf.gather_nd(similarities, self_indices), [-1])
  if use_nca:
    same_semantic_array = tf.cast(
        tf.equal(semantic_labels, prototype_labels), tf.float32)
    selfout = (tf.reduce_sum(similarities * same_semantic_array, axis=1) -
               self_similarities)
    numerator = tf.where(tf.greater(selfout, 0),
                         selfout, self_similarities)
  else:
    numerator = self_similarities

  diff_semantic_array = tf.cast(
      tf.not_equal(semantic_labels, prototype_labels), tf.float32)
  denominator = (tf.reduce_sum(similarities * diff_semantic_array, axis=1) +
                 numerator)

  probabilities = tf.divide(numerator, denominator)
  log_likelihood = -tf.log(probabilities)
  return log_likelihood


def _calculate_similarities(
    embedding, prototypes, concentration, split=1):
  """Calculates cosine similarities between embedding and prototypes.

  This function calculates cosine similarities between embedding and prototypes.
  It splits the matrix multiplication to prevent an unknown bug in Tensorflow.

  Args:
    embedding: A 2-D float tensor with shape [num_pixels, embedding_dim].
    prototypes: A 2-D float tenwor with shape [num_prototypes, embedding_dim].
    concentration: A float that controls the sharpness of cosine similarities.
    split: An integer for number of splits of matrix multiplication.

  Returns:
    similarities: A 2-D float tensor with shape [num_pixesl, num_prototypes].
  """
  # This for loop is used to prevent an unknown bug in Tensorflow.
  # https://yaqs.googleplex.com/eng/q/4520049116446720
  if split > 1:
    step_size = tf.shape(embedding)[0] // split
    for s in range(split):
      if s < split - 1:
        embedding_temp = embedding[step_size*s:step_size*(s+1)]
      else:
        embedding_temp = embedding[step_size*s:]
      pre_similarities_temp = tf.matmul(
          embedding_temp, prototypes, transpose_b=True) * concentration
      if s == 0:
        pre_similarities = pre_similarities_temp
      else:
        pre_similarities = tf.concat([pre_similarities,
                                      pre_similarities_temp], 0)
  else:
    pre_similarities = tf.matmul(
        embedding, prototypes, transpose_b=True) * concentration
  similarities = tf.exp(pre_similarities)
  return similarities


def _retrieve_memory(num_prototypes, num_banks, embedding_dim):
  """Retrieves prototype memory variables cached from the previous batches.

  Args:
    num_prototypes: An integer that specifies the total number of prototypes in
      a batch.
    num_banks: Number of memory banks for caching the training prototypes.
    embedding_dim: An integer that refers to the number of embedding dimensions.

  Returns:
    memory_list: A list of 2-D float Tensorflow variables for prototype memory
      each with shape `[num_prototypes, embedding_dim]`.
    memory_labels_list: A list of 1-D integer Tensorflow variables for prototype
      labels each with length `[num_prototypes]`.
  """
  memory_list = []
  memory_labels_list = []
  for b in range(num_banks):
    memory_list.append(tf.get_variable(
        'memory_'+str(b),
        shape=[num_prototypes, embedding_dim],
        initializer=tf.truncated_normal_initializer(),
        trainable=False))
    memory_labels_list.append(tf.get_variable(
        'memory_labels_'+str(b),
        dtype=tf.int32,
        shape=[num_prototypes],
        initializer=tf.constant_initializer(0),
        trainable=False))
  return memory_list, memory_labels_list
