# Separate file so we don't accidentally import tensorflow/keras before this function
# in utils.py or calling code

# Using this sys.argv workaround instead of argparse so we can use argparse downstream w/o conflicts
# (can't figure out more elegant solution)
import sys
if '--split-gpu' in sys.argv:
    sys.argv.remove('--split-gpu')
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.40
    set_session(tf.Session(config=config))