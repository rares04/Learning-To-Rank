Ñ
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18¨§

rank_net_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namerank_net_1/dense_5/kernel

-rank_net_1/dense_5/kernel/Read/ReadVariableOpReadVariableOprank_net_1/dense_5/kernel*
_output_shapes

:*
dtype0

rank_net_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namerank_net_1/dense_5/bias

+rank_net_1/dense_5/bias/Read/ReadVariableOpReadVariableOprank_net_1/dense_5/bias*
_output_shapes
:*
dtype0

rank_net_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namerank_net_1/dense_3/kernel

-rank_net_1/dense_3/kernel/Read/ReadVariableOpReadVariableOprank_net_1/dense_3/kernel*
_output_shapes

:*
dtype0

rank_net_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namerank_net_1/dense_3/bias

+rank_net_1/dense_3/bias/Read/ReadVariableOpReadVariableOprank_net_1/dense_3/bias*
_output_shapes
:*
dtype0

rank_net_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namerank_net_1/dense_4/kernel

-rank_net_1/dense_4/kernel/Read/ReadVariableOpReadVariableOprank_net_1/dense_4/kernel*
_output_shapes

:*
dtype0

rank_net_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namerank_net_1/dense_4/bias

+rank_net_1/dense_4/bias/Read/ReadVariableOpReadVariableOprank_net_1/dense_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
­
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*è
valueÞBÛ BÔ

	dense
o
oi_minus_oj
trainable_variables
regularization_losses
	variables
	keras_api

signatures

	0

1
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
­

layers
metrics
non_trainable_variables
trainable_variables
regularization_losses
layer_regularization_losses
	variables
layer_metrics
 
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
h

kernel
bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
RP
VARIABLE_VALUErank_net_1/dense_5/kernel#o/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUErank_net_1/dense_5/bias!o/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

&layers
'non_trainable_variables
trainable_variables
regularization_losses
(metrics
)layer_regularization_losses
	variables
*layer_metrics
 
 
 
­

+layers
,non_trainable_variables
trainable_variables
regularization_losses
-metrics
.layer_regularization_losses
	variables
/layer_metrics
_]
VARIABLE_VALUErank_net_1/dense_3/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUErank_net_1/dense_3/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUErank_net_1/dense_4/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUErank_net_1/dense_4/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

	0

1
2
3
 
 
 
 

0
1
 

0
1
­

0layers
1non_trainable_variables
trainable_variables
regularization_losses
2metrics
3layer_regularization_losses
 	variables
4layer_metrics

0
1
 

0
1
­

5layers
6non_trainable_variables
"trainable_variables
#regularization_losses
7metrics
8layer_regularization_losses
$	variables
9layer_metrics
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_input_2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ø
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2rank_net_1/dense_3/kernelrank_net_1/dense_3/biasrank_net_1/dense_4/kernelrank_net_1/dense_4/biasrank_net_1/dense_5/kernelrank_net_1/dense_5/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1628591
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¹
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-rank_net_1/dense_5/kernel/Read/ReadVariableOp+rank_net_1/dense_5/bias/Read/ReadVariableOp-rank_net_1/dense_3/kernel/Read/ReadVariableOp+rank_net_1/dense_3/bias/Read/ReadVariableOp-rank_net_1/dense_4/kernel/Read/ReadVariableOp+rank_net_1/dense_4/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1628704
¼
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerank_net_1/dense_5/kernelrank_net_1/dense_5/biasrank_net_1/dense_3/kernelrank_net_1/dense_3/biasrank_net_1/dense_4/kernelrank_net_1/dense_4/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1628732±ù
Ô
é
#__inference__traced_restore_1628732
file_prefix.
*assignvariableop_rank_net_1_dense_5_kernel.
*assignvariableop_1_rank_net_1_dense_5_bias0
,assignvariableop_2_rank_net_1_dense_3_kernel.
*assignvariableop_3_rank_net_1_dense_3_bias0
,assignvariableop_4_rank_net_1_dense_4_kernel.
*assignvariableop_5_rank_net_1_dense_4_bias

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5·
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ã
value¹B¶B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity©
AssignVariableOpAssignVariableOp*assignvariableop_rank_net_1_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¯
AssignVariableOp_1AssignVariableOp*assignvariableop_1_rank_net_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2±
AssignVariableOp_2AssignVariableOp,assignvariableop_2_rank_net_1_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¯
AssignVariableOp_3AssignVariableOp*assignvariableop_3_rank_net_1_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_rank_net_1_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¯
AssignVariableOp_5AssignVariableOp*assignvariableop_5_rank_net_1_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6Ö

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Í
¬
D__inference_dense_5_layer_call_and_return_conditional_losses_1628601

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³4

"__inference__wrapped_model_1628441
input_1
input_25
1rank_net_1_dense_3_matmul_readvariableop_resource6
2rank_net_1_dense_3_biasadd_readvariableop_resource5
1rank_net_1_dense_4_matmul_readvariableop_resource6
2rank_net_1_dense_4_biasadd_readvariableop_resource5
1rank_net_1_dense_5_matmul_readvariableop_resource6
2rank_net_1_dense_5_biasadd_readvariableop_resource
identityÆ
(rank_net_1/dense_3/MatMul/ReadVariableOpReadVariableOp1rank_net_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net_1/dense_3/MatMul/ReadVariableOp­
rank_net_1/dense_3/MatMulMatMulinput_10rank_net_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_3/MatMulÅ
)rank_net_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp2rank_net_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net_1/dense_3/BiasAdd/ReadVariableOpÍ
rank_net_1/dense_3/BiasAddBiasAdd#rank_net_1/dense_3/MatMul:product:01rank_net_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_3/BiasAdd
rank_net_1/dense_3/LeakyRelu	LeakyRelu#rank_net_1/dense_3/BiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_3/LeakyReluÊ
*rank_net_1/dense_3/MatMul_1/ReadVariableOpReadVariableOp1rank_net_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*rank_net_1/dense_3/MatMul_1/ReadVariableOp³
rank_net_1/dense_3/MatMul_1MatMulinput_22rank_net_1/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_3/MatMul_1É
+rank_net_1/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp2rank_net_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+rank_net_1/dense_3/BiasAdd_1/ReadVariableOpÕ
rank_net_1/dense_3/BiasAdd_1BiasAdd%rank_net_1/dense_3/MatMul_1:product:03rank_net_1/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_3/BiasAdd_1
rank_net_1/dense_3/LeakyRelu_1	LeakyRelu%rank_net_1/dense_3/BiasAdd_1:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rank_net_1/dense_3/LeakyRelu_1Æ
(rank_net_1/dense_4/MatMul/ReadVariableOpReadVariableOp1rank_net_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net_1/dense_4/MatMul/ReadVariableOpÐ
rank_net_1/dense_4/MatMulMatMul*rank_net_1/dense_3/LeakyRelu:activations:00rank_net_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_4/MatMulÅ
)rank_net_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp2rank_net_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net_1/dense_4/BiasAdd/ReadVariableOpÍ
rank_net_1/dense_4/BiasAddBiasAdd#rank_net_1/dense_4/MatMul:product:01rank_net_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_4/BiasAdd
rank_net_1/dense_4/LeakyRelu	LeakyRelu#rank_net_1/dense_4/BiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_4/LeakyReluÊ
*rank_net_1/dense_4/MatMul_1/ReadVariableOpReadVariableOp1rank_net_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*rank_net_1/dense_4/MatMul_1/ReadVariableOpØ
rank_net_1/dense_4/MatMul_1MatMul,rank_net_1/dense_3/LeakyRelu_1:activations:02rank_net_1/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_4/MatMul_1É
+rank_net_1/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp2rank_net_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+rank_net_1/dense_4/BiasAdd_1/ReadVariableOpÕ
rank_net_1/dense_4/BiasAdd_1BiasAdd%rank_net_1/dense_4/MatMul_1:product:03rank_net_1/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_4/BiasAdd_1
rank_net_1/dense_4/LeakyRelu_1	LeakyRelu%rank_net_1/dense_4/BiasAdd_1:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rank_net_1/dense_4/LeakyRelu_1Æ
(rank_net_1/dense_5/MatMul/ReadVariableOpReadVariableOp1rank_net_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net_1/dense_5/MatMul/ReadVariableOpÐ
rank_net_1/dense_5/MatMulMatMul*rank_net_1/dense_4/LeakyRelu:activations:00rank_net_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_5/MatMulÅ
)rank_net_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp2rank_net_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net_1/dense_5/BiasAdd/ReadVariableOpÍ
rank_net_1/dense_5/BiasAddBiasAdd#rank_net_1/dense_5/MatMul:product:01rank_net_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_5/BiasAddÊ
*rank_net_1/dense_5/MatMul_1/ReadVariableOpReadVariableOp1rank_net_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*rank_net_1/dense_5/MatMul_1/ReadVariableOpØ
rank_net_1/dense_5/MatMul_1MatMul,rank_net_1/dense_4/LeakyRelu_1:activations:02rank_net_1/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_5/MatMul_1É
+rank_net_1/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp2rank_net_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+rank_net_1/dense_5/BiasAdd_1/ReadVariableOpÕ
rank_net_1/dense_5/BiasAdd_1BiasAdd%rank_net_1/dense_5/MatMul_1:product:03rank_net_1/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/dense_5/BiasAdd_1»
rank_net_1/subtract_1/subSub#rank_net_1/dense_5/BiasAdd:output:0%rank_net_1/dense_5/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/subtract_1/sub
rank_net_1/activation/SigmoidSigmoidrank_net_1/subtract_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rank_net_1/activation/Sigmoidu
IdentityIdentity!rank_net_1/activation/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
º
Ã
 __inference__traced_save_1628704
file_prefix8
4savev2_rank_net_1_dense_5_kernel_read_readvariableop6
2savev2_rank_net_1_dense_5_bias_read_readvariableop8
4savev2_rank_net_1_dense_3_kernel_read_readvariableop6
2savev2_rank_net_1_dense_3_bias_read_readvariableop8
4savev2_rank_net_1_dense_4_kernel_read_readvariableop6
2savev2_rank_net_1_dense_4_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1492e02d5ab04ed49f5378ab24f18226/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename±
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ã
value¹B¶B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slicesþ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_rank_net_1_dense_5_kernel_read_readvariableop2savev2_rank_net_1_dense_5_bias_read_readvariableop4savev2_rank_net_1_dense_3_kernel_read_readvariableop2savev2_rank_net_1_dense_3_bias_read_readvariableop4savev2_rank_net_1_dense_4_kernel_read_readvariableop2savev2_rank_net_1_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
´
¬
D__inference_dense_4_layer_call_and_return_conditional_losses_1628487

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
¬
D__inference_dense_3_layer_call_and_return_conditional_losses_1628457

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
~
)__inference_dense_5_layer_call_fn_1628610

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_16285162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ä
%__inference_signature_wrapper_1628591
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_16284412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
£
q
G__inference_subtract_1_layer_call_and_return_conditional_losses_1628541

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§$
Ä
G__inference_rank_net_1_layer_call_and_return_conditional_losses_1628552
input_1
input_2
dense_3_1628468
dense_3_1628470
dense_4_1628498
dense_4_1628500
dense_5_1628527
dense_5_1628529
identity¢dense_3/StatefulPartitionedCall¢!dense_3/StatefulPartitionedCall_1¢dense_4/StatefulPartitionedCall¢!dense_4/StatefulPartitionedCall_1¢dense_5/StatefulPartitionedCall¢!dense_5/StatefulPartitionedCall_1
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3_1628468dense_3_1628470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_16284572!
dense_3/StatefulPartitionedCall
!dense_3/StatefulPartitionedCall_1StatefulPartitionedCallinput_2dense_3_1628468dense_3_1628470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_16284572#
!dense_3/StatefulPartitionedCall_1·
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_1628498dense_4_1628500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_16284872!
dense_4/StatefulPartitionedCall½
!dense_4/StatefulPartitionedCall_1StatefulPartitionedCall*dense_3/StatefulPartitionedCall_1:output:0dense_4_1628498dense_4_1628500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_16284872#
!dense_4/StatefulPartitionedCall_1·
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_1628527dense_5_1628529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_16285162!
dense_5/StatefulPartitionedCall½
!dense_5/StatefulPartitionedCall_1StatefulPartitionedCall*dense_4/StatefulPartitionedCall_1:output:0dense_5_1628527dense_5_1628529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_16285162#
!dense_5/StatefulPartitionedCall_1­
subtract_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*dense_5/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_subtract_1_layer_call_and_return_conditional_losses_16285412
subtract_1/PartitionedCall
activation/SigmoidSigmoid#subtract_1/PartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Sigmoid¼
IdentityIdentityactivation/Sigmoid:y:0 ^dense_3/StatefulPartitionedCall"^dense_3/StatefulPartitionedCall_1 ^dense_4/StatefulPartitionedCall"^dense_4/StatefulPartitionedCall_1 ^dense_5/StatefulPartitionedCall"^dense_5/StatefulPartitionedCall_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dense_3/StatefulPartitionedCall_1!dense_3/StatefulPartitionedCall_12B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dense_4/StatefulPartitionedCall_1!dense_4/StatefulPartitionedCall_12B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dense_5/StatefulPartitionedCall_1!dense_5/StatefulPartitionedCall_1:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
å
Ë
,__inference_rank_net_1_layer_call_fn_1628571
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_rank_net_1_layer_call_and_return_conditional_losses_16285522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
«
s
G__inference_subtract_1_layer_call_and_return_conditional_losses_1628616
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
 
X
,__inference_subtract_1_layer_call_fn_1628622
inputs_0
inputs_1
identityÕ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_subtract_1_layer_call_and_return_conditional_losses_16285412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Í
¬
D__inference_dense_5_layer_call_and_return_conditional_losses_1628516

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
~
)__inference_dense_3_layer_call_fn_1628642

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_16284572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
¬
D__inference_dense_3_layer_call_and_return_conditional_losses_1628633

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
~
)__inference_dense_4_layer_call_fn_1628662

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_16284872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
¬
D__inference_dense_4_layer_call_and_return_conditional_losses_1628653

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*è
serving_defaultÔ
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
;
input_20
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¸a
Ù
	dense
o
oi_minus_oj
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*:&call_and_return_all_conditional_losses
;__call__
<_default_save_signature"ú
_tf_keras_modelà{"class_name": "RankNet", "name": "rank_net_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "RankNet"}}
.
	0

1"
trackable_list_wrapper
ì

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"Ç
_tf_keras_layer­{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [4, 8]}}
°
trainable_variables
regularization_losses
	variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"¡
_tf_keras_layer{"class_name": "Subtract", "name": "subtract_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "subtract_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [4, 1]}, {"class_name": "TensorShape", "items": [4, 1]}]}
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
Ê

layers
metrics
non_trainable_variables
trainable_variables
regularization_losses
layer_regularization_losses
	variables
layer_metrics
;__call__
<_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
Aserving_default"
signature_map
ñ

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
*B&call_and_return_all_conditional_losses
C__call__"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 16, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [4, 7]}}
ò

kernel
bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
*D&call_and_return_all_conditional_losses
E__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 8, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [4, 16]}}
+:)2rank_net_1/dense_5/kernel
%:#2rank_net_1/dense_5/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

&layers
'non_trainable_variables
trainable_variables
regularization_losses
(metrics
)layer_regularization_losses
	variables
*layer_metrics
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

+layers
,non_trainable_variables
trainable_variables
regularization_losses
-metrics
.layer_regularization_losses
	variables
/layer_metrics
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
+:)2rank_net_1/dense_3/kernel
%:#2rank_net_1/dense_3/bias
+:)2rank_net_1/dense_4/kernel
%:#2rank_net_1/dense_4/bias
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

0layers
1non_trainable_variables
trainable_variables
regularization_losses
2metrics
3layer_regularization_losses
 	variables
4layer_metrics
C__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

5layers
6non_trainable_variables
"trainable_variables
#regularization_losses
7metrics
8layer_regularization_losses
$	variables
9layer_metrics
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
½2º
G__inference_rank_net_1_layer_call_and_return_conditional_losses_1628552î
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
¢2
,__inference_rank_net_1_layer_call_fn_1628571î
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
2
"__inference__wrapped_model_1628441Þ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
î2ë
D__inference_dense_5_layer_call_and_return_conditional_losses_1628601¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_5_layer_call_fn_1628610¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_subtract_1_layer_call_and_return_conditional_losses_1628616¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_subtract_1_layer_call_fn_1628622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
;B9
%__inference_signature_wrapper_1628591input_1input_2
î2ë
D__inference_dense_3_layer_call_and_return_conditional_losses_1628633¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_3_layer_call_fn_1628642¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_4_layer_call_and_return_conditional_losses_1628653¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_4_layer_call_fn_1628662¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¾
"__inference__wrapped_model_1628441X¢U
N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_3_layer_call_and_return_conditional_losses_1628633\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_3_layer_call_fn_1628642O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_4_layer_call_and_return_conditional_losses_1628653\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_4_layer_call_fn_1628662O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_5_layer_call_and_return_conditional_losses_1628601\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_5_layer_call_fn_1628610O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÕ
G__inference_rank_net_1_layer_call_and_return_conditional_losses_1628552X¢U
N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¬
,__inference_rank_net_1_layer_call_fn_1628571|X¢U
N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÒ
%__inference_signature_wrapper_1628591¨i¢f
¢ 
_ª\
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
,
input_2!
input_2ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿÏ
G__inference_subtract_1_layer_call_and_return_conditional_losses_1628616Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
,__inference_subtract_1_layer_call_fn_1628622vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ