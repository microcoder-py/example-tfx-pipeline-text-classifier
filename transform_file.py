import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

_feature = 'text'
_label = 'label'

_vocab_size = #add number to define total vocabulary size
_max = #add number for max length of any sentence

#This function helps TFX know which label is input, and which is output. 
#If we have a feature which is an input, we rename it to feature_xf_input and it is mapped to the input feature_xf
#If we have a feature that is an output, we only append _xf, and it is used as output for the output labelled feature_xf
def _transformed_name(key, is_input = False):
  if (is_input == True):
    return key + '_xf_input'

  return key+'_xf'

#Making tokens from input sentences is a little different than in TF.
#All aggregation functions, that need information from the whole database are performed in this file
#Here, we need to compute vocabulary across the whole database, so we will compute it here
def tokenizer(text):

  #The shape of a string tensor is (). 
  #We first reshape it to [-1] which changes the shape to (1,)
  #This string tensor can be split easily and turned into a sparse tensor
  reshape_and_split = tf.strings.split( tf.reshape(text,[-1])).to_sparse()

  #Using the above sparse tensor, we find the vocabulary using the TFT function compute_and_apply_vocabulary
  tokenize = tft.compute_and_apply_vocabulary(
      reshape_and_split, default_value = _vocab_size, top_k = _vocab_size
  )

  #Converting sparse representation to dense for further processing
  convert_to_dense = tf.sparse.to_dense(tokenize, default_value = -1)

  #The outputs are expected to be of a fixed size since it is a FixedLengthFeature
  #We first pad the dense tensor with max length 0s from the end 
  padding_config = [[0,0], [0, _max]]
  dense_padded = tf.pad(convert_to_dense, padding_config, 'CONSTANT', -1)
  
  #Now we take a slice that gives us the first max length characters. This is our final sentence
  dense_max_len = tf.slice(dense_padded, [0,0], [-1, _max])
  
  #The reason we are adding the 1 here is because TFT does not recognize OOV tokens by default as 0
  #So we simply tokenize the whole vocabulary, and then add 1 to all of them so that if any of them is OOV,
  #it is recognised as 0 
  dense_max_len += 1

  return dense_max_len

#This function is simply sanitising the inputs
def preprocessing_fn(inputs):

  data = {
      _transformed_name(_label): inputs[_label],
      _transformed_name(_feature, True): tokenizer(inputs[_feature]) 
  }

  return data
