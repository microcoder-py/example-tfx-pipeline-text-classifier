# Example TFX Pipeline for Text Classification

While trying to build my own production ML pipeline, I was unable to understand what was happening under the hood. Found a few examples, but none of them clear enough, so I decided to build a couple of my own so I and anybody else who may need it can use them.

The fundamentals of this are fairly simple. We have several steps in building a model

1. Find the data
2. Evaluate the data
3. Preprocess the data
4. Train the model with the data
5. Save the trained model wuth serving signatures
6. Push the saved model into production

Reading the code, it will be fairly simple to understand with attached documentation what each component is doing

This example is only illustrative, not exhaustive. There are several other components that can be attached to the pipeline for more completeness, but to get started with, this is a sufficiently deep overview

## Data
You can really use any data that is in a CSV format, with columns of ``label, text``

If you want to use a different file format, you will need to make respective changes to the ``ExampleGen`` component in the file ``pipeline.py``

If you wish to carry out any more preprocessing, kindly add the needed steps in the function ``preprocessing_fn()`` in the file ``transform_file.py`` (e.g. stripping away HTML tags, reducing punctuation)

Just save the file in a directory, and specify the location of the directory in the variable ``_data_root`` in the file ``pipeline.py``

## Model
If you want to change the model you are making use of, go to the file ``trainer_file.py`` and modify the function ``build_keras_model()`` according to what you require

Defining how to run the training session, including parallelisation, data ingestion options etc can be done by modifying the ``trainer_file.py`` module

## Execution
After adding all the global variables, and downloading data, creating directories for each requirement, just run the file ``pipeline.py`` with the command. Python3 is recommended

``>>python3 pipeline.py``

It will automatically trigger the pipeline execution

Please make sure the following libraries are installed:

1. TensorFlow
``pip3 install tensorflow``

2. TensorFlow Extended
``pip3 install tfx``

3. TFX BSL
``pip3 install tfx-bsl``
