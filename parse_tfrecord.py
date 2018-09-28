import tensorflow as tf

def bounding_box(xmins, xmaxs, ymins, ymaxs):
    xmins = tf.reshape(xmins,[-1, tf.shape(xmins)[0]])
    xmaxs = tf.reshape(xmaxs, [-1, tf.shape(xmaxs)[0]])
    ymins = tf.reshape(ymins, [-1, tf.shape(ymins)[0]])
    ymaxs = tf.reshape(ymaxs, [-1, tf.shape(ymaxs)[0]])
    return tf.concat([ymins, xmins, ymaxs, xmaxs], axis=1)

def _parse_function(example_proto):
    keys_to_features = {
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
        'image/filename': tf.FixedLenFeature((), tf.string),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'xmins': tf.VarLenFeature(dtype=tf.float32),
        'ymins': tf.VarLenFeature(dtype=tf.float32),
        'xmaxs': tf.VarLenFeature(dtype=tf.float32),
        'ymaxs': tf.VarLenFeature(dtype=tf.float32),
        'image/label': tf.VarLenFeature(dtype=tf.int64)
    } 

    features = tf.parse_example([example_proto], features=keys_to_features)
    
    filename = features['image/filename'][0]
    image = tf.image.decode_jpeg(features['image/encoded'][0])
    xmins = tf.sparse_tensor_to_dense(features['xmins'])
    xmaxs = tf.sparse_tensor_to_dense(features['xmaxs'])
    ymins = tf.sparse_tensor_to_dense(features['ymins'])
    ymaxs = tf.sparse_tensor_to_dense(features['ymaxs'])
    label = tf.sparse_tensor_to_dense(features['image/label'])[0]
    '''
    is_empty = tf.not_equal(tf.size(xmins), 0)
    bbox = tf.cond(is_empty, lambda: bounding_box(xmins, xmaxs, ymins, ymaxs), lambda: tf.constant([]))
    '''
    bbox = bounding_box(xmins, xmaxs, ymins, ymaxs)
    image = tf.tile(image, [1,1,3])
    return filename, image, label, bbox

def filter_func(filename, image, label, bbox):
    return tf.reduce_all(tf.not_equal(label, 0))

def get_iterator(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.filter(filter_func)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    return iterator