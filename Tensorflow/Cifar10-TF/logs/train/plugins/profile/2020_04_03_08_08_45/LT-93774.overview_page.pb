�	:��H�@:��H�@!:��H�@	�y)�1�P?�y)�1�P?!�y)�1�P?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$:��H�@K�4vW@A�Pk�w-�@Y��H�}}?*	�������@2T
Iterator::Prefetch::GeneratorX9��v�?!++{:f�J@)X9��v�?1++{:f�J@:Preprocessing2b
+Iterator::Model::Prefetch::Rebatch::BatchV2�m4��@�?!A�M:��E@)'�W��?1~�޴7@:Preprocessing2x
AIterator::Model::Prefetch::Rebatch::BatchV2::Shuffle::TensorSlice@~8gDi�?!�����-@)~8gDi�?1�����-@:Preprocessing2k
4Iterator::Model::Prefetch::Rebatch::BatchV2::Shuffle@e�X��?!�V��3@)���K7�?1?��#j@:Preprocessing2F
Iterator::Model����Mb�?!:�a�:�?)p_�Q�?1ꘑC�?:Preprocessing2I
Iterator::Prefetch� �	��?!�T�CuH�?)� �	��?1�T�CuH�?:Preprocessing2Y
"Iterator::Model::Prefetch::Rebatch��"��~�?!1�l���E@)ŏ1w-!?1xm����?:Preprocessing2P
Iterator::Model::Prefetch �o_�y?!AEf1��?) �o_�y?1AEf1��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*moderate2B13.3 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	K�4vW@K�4vW@!K�4vW@      ��!       "      ��!       *      ��!       2	�Pk�w-�@�Pk�w-�@!�Pk�w-�@:      ��!       B      ��!       J	��H�}}?��H�}}?!��H�}}?R      ��!       Z	��H�}}?��H�}}?!��H�}}?JCPU_ONLY2blackY      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendationR
nomoderate"B13.3 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 