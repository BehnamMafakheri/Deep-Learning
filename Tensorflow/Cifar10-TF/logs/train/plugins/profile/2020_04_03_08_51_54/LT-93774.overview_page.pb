�	ˡE�sE�@ˡE�sE�@!ˡE�sE�@	�4��PkW?�4��PkW?!�4��PkW?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ˡE�sE�@�q���@A��q��@Yvq�-�?*	     ձ@2T
Iterator::Prefetch::Generator�3��7@!c�ކ�T@)�3��7@1c�ކ�T@:Preprocessing2b
+Iterator::Model::Prefetch::Rebatch::BatchV21�*���?!�rY1P0@)��S㥛�?1 ��v� @:Preprocessing2x
AIterator::Model::Prefetch::Rebatch::BatchV2::Shuffle::TensorSlice@-!�lV�?!�6풴�@)-!�lV�?1�6풴�@:Preprocessing2k
4Iterator::Model::Prefetch::Rebatch::BatchV2::Shuffle@�:M��?! [��׏@)�e�c]ܶ?1Αx2�L�?:Preprocessing2F
Iterator::Model��H�}�?!јoR:0�?)=�U����?1F�?���?:Preprocessing2I
Iterator::Prefetch�St$���?!F]t�E�?)�St$���?1F]t�E�?:Preprocessing2Y
"Iterator::Model::Prefetch::Rebatch/n���?!����q0@)�~j�t�x?1�T�D���?:Preprocessing2P
Iterator::Model::PrefetchU���N@s?!��J�h[�?)U���N@s?1��J�h[�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�q���@�q���@!�q���@      ��!       "      ��!       *      ��!       2	��q��@��q��@!��q��@:      ��!       B      ��!       J	vq�-�?vq�-�?!vq�-�?R      ��!       Z	vq�-�?vq�-�?!vq�-�?JCPU_ONLY2blackY      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 