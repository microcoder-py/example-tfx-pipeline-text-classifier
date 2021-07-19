import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

_feature = 'text'
_label = 'label'

_vocab_size = #add number to define total vocabulary size
_max = #add number for max length of any sentence

def _transformed_name(key, is_input = False):
  if (is_input == True):
    return key + '_xf_input'

  return key+'_xf'


def tokenizer(text):

  reshape_and_split = tf.strings.split( tf.reshape(text,[-1])).to_sparse()

  tokenize = tft.compute_and_apply_vocabulary(
      reshape_and_split, default_value = _vocab_size, top_k = _vocab_size
  )

  convert_to_dense = tf.sparse.to_dense(tokenize, default_value = -1)

  padding_config = [[0,0], [0, _max]]
  dense_padded = tf.pad(convert_to_dense, padding_config, 'CONSTANT', -1)
  dense_max_len = tf.slice(dense_padded, [0,0], [-1, _max])
  dense_max_len += 1

  return dense_max_len

def preprocessing_fn(inputs):

  data = {
      _transformed_name(_label): inputs[_label],
      _transformed_name(_feature, True): tokenizer(inputs[_feature]) 
  }

  return data
