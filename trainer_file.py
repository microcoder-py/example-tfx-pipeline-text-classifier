import tensorflow as tf
from tensorflow import keras

from typing import List, Text
import absl
import tensorflow_transform as tft

import tfx_bsl
from tfx_bsl.tfxio import dataset_options

import tfx
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs

_feature = 'text'
_label = 'label'
_vocab_size = #add size of vocabulary, or simply use a shared variable
_embedding_dims = #add required embedding dimensions for the vocabulary


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_specifications = tf_transform_output.raw_feature_spec()
    feature_specifications.pop(_label)
    parsed_feature_list = tf.io.parse_example(serialized_tf_examples, feature_specifications)
    transformed_features = model.tft_layer(parsed_feature_list)
    return model(transformed_features)

  return serve_tf_examples_fn


def build_keras_model() -> keras.Model:

  model = keras.Sequential([
      keras.layers.Embedding(
          _vocab_size + 2,
          _embedding_dims,
          name='text_xf'),
      keras.layers.Bidirectional(
          keras.layers.LSTM(2)),
      keras.layers.Dense(4, activation='relu'),
      keras.layers.Dense(1)
  ])

  model.compile(
      loss=keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer='adam',
      metrics=['accuracy'])

  model.summary(print_fn=absl.logging.info)
  return model

def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200):

  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key= 'label_xf'),
      tf_transform_output.transformed_metadata.schema)
  

  return dataset

def run_fn(fn_args: FnArgs):
  
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  tf.print(tf_transform_output)
  tft_layer = tf_transform_output.transform_features_layer()
  print(tft_layer)
  
  train_set = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size = 2
  )

  eval_dataset = _input_fn(
    fn_args.eval_files,
    fn_args.data_accessor,
    tf_transform_output,
    batch_size = 2
  )

  model = build_keras_model()

  _num_epochs = #add number of epochs
  
  model.fit(
    train_set,
    epochs=_num_epochs,
    steps_per_epoch=fn_args.train_steps,
    validation_data=eval_dataset,
    validation_steps=fn_args.eval_steps
  )

  signatures = {
    'serving_default':
      _get_serve_tf_examples_fn(model,tf_transform_output).get_concrete_function(
          tf.TensorSpec(
              shape = [None],
              dtype = tf.string,
              name = 'examples'
          )
      )
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures = signatures)
