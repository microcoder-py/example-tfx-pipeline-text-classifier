import tensorflow as tf
import tfx
import tfx_bsl
#Define global parameters in the  beginning itself to avoid confusion towards the end
_data_path = #add path to directory which contains all the training files
_serving_dir = #add path to directory where you will be storing the final trained model
_num_train_steps = #add number
_num_eval_steps = #add number

_pipeline_name = #name the pipeline
_pipeline_root = #specify location of pipeline storage
_metadata_root = #specify location for sqlite db that will facilitate metadata storage
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    '--direct_num_workers=0',
]

#EXAMPLE GENERATOR
#This pipeline component defines how we are supposed to ingest data given several files
#We can use it to define how we are ingesting data as well, how we split it, etc
from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2

output_config = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(
        splits = [
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
        ]
    )
)

example_gen = CsvExampleGen(input_base=_data_root, output_config=output_config)

#STATISTICS GENERATOR
#This component generates numbers to understand possible skew in data, and other factors
#This is optional. If you do not want to use it, remove it as an argument whenever it is being
#consumed by future components
from tfx.components import StatisticsGen

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

#SCHEMA GENERATOR
#This component defines the schema for the data
#This is optional. If you do not want to use it, remove it as an argument whenever it is being
#consumed by future components
from tfx.components import SchemaGen

schema_gen = SchemaGen(
    statistics = statistics_gen.outputs['statistics'], infer_feature_shape = True
)

#TRANSFORM COMPONENT
#This component carries out all transformations needed in preprocessing for the whole batch
#We need to provide it a python file that contains a function named preprocessing_fn
#The component will use the function to execute preprocessing as needed
from tfx.components import Transform

transform_file = 'transform_file.py'

transform = Transform(
    examples = example_gen.outputs['examples'],
    schema = schema_gen.outputs['schema'],
    module_file = transform_file
)

#TRAINER MODULE
#We need to provide it a python file that contains a run_fn, which defines how it will be 
#training the module
#The output is a trained model, which needs to be saved at a location the pusher component can use
from tfx.components import Trainer
from tfx.proto import trainer_pb2

trainer_file = "trainer_file.py"

trainer = Trainer(
    module_file = trainer_file,
    examples = transform.outputs['transformed_examples'],
    transform_graph = transform.outputs['transform_graph'],
    schema = schema_gen.outputs['schema'],
    train_args = trainer_pb2.TrainArgs(num_steps = _num_train_steps)),
    eval_args = trainer_pb2.EvalArgs(num_steps = _num_eval_steps)
)

#PUSHER COMPONENT
#This defines how we will be pushing the saved model into serving 
from tfx.components import Pusher
from tfx.proto import pusher_pb2

pusher = Pusher(
      model=trainer.outputs['model'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=_serving_dir)))

if __name__ == "__main__":
  
    from tfx.orchestration import pipeline
    from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
    
    #Here we define the components involved in the pipeline
    pipeline_components = [
      example_gen,
      statistics_gen,
      schema_gen,
      transform,
      trainer,
      pusher
    ]
    
    #We build a pipeline with our options
    tfx_pipeline = pipeline.Pipeline(
        pipeline_name = _pipeline_name,
        pipeline_root = _pipeline_root,
        components = pipeline_components,
        metadata_connection_config = metadata.sqlite_metadata_connection_config(metadata_path)
        enable_cache = False,
        beam_pipeline_args = _beam_pipeline_args
    )
    
    #We execute the pipeline
    #The pipeline can be executed component by component using the TFX Interactive Context 
    BeamDagRunner().run(tfx_pipeline)
