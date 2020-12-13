import tensorflow as tf

@tf.function
def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, kernel, 1, padding='SAME')
    keep = tf.cast(tf.equal(heat, hmax), tf.float32)
    return heat * keep

@tf.function
def heatmap_to_keypoints(batch_heatmaps):

    batch, height, width, n_points = tf.shape(batch_heatmaps)[0], tf.shape(
        batch_heatmaps)[1], tf.shape(batch_heatmaps)[2], tf.shape(batch_heatmaps)[3]

    batch_heatmaps = nms(batch_heatmaps)

    flat_tensor = tf.reshape(batch_heatmaps, (batch, -1, n_points))

    # Argmax of the flat tensor
    argmax = tf.argmax(flat_tensor, axis=1)
    argmax = tf.cast(argmax, tf.int32)
    scores = tf.math.reduce_max(flat_tensor, axis=1)

    # Convert indexes into 2D coordinates
    argmax_y = argmax // width
    argmax_x = argmax % width
    argmax_y = tf.cast(argmax_y, tf.float32) / float(height)
    argmax_x = tf.cast(argmax_x, tf.float32) / float(width)

    # Shape: batch * 3 * n_points
    batch_keypoints = tf.stack((argmax_x, argmax_y, scores), axis=1)
    # Shape: batch * n_points * 3
    batch_keypoints = tf.transpose(batch_keypoints, [0, 2, 1])

    return batch_keypoints