??
??
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
dtypetype?
?
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
?
rank_net_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namerank_net_1/dense_5/kernel
?
-rank_net_1/dense_5/kernel/Read/ReadVariableOpReadVariableOprank_net_1/dense_5/kernel*
_output_shapes

:*
dtype0
?
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
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
rank_net_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namerank_net_1/dense_3/kernel
?
-rank_net_1/dense_3/kernel/Read/ReadVariableOpReadVariableOprank_net_1/dense_3/kernel*
_output_shapes

:*
dtype0
?
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
?
rank_net_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namerank_net_1/dense_4/kernel
?
-rank_net_1/dense_4/kernel/Read/ReadVariableOpReadVariableOprank_net_1/dense_4/kernel*
_output_shapes

:*
dtype0
?
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
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
 Adam/rank_net_1/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/rank_net_1/dense_5/kernel/m
?
4Adam/rank_net_1/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/rank_net_1/dense_5/kernel/m*
_output_shapes

:*
dtype0
?
Adam/rank_net_1/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/rank_net_1/dense_5/bias/m
?
2Adam/rank_net_1/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/rank_net_1/dense_5/bias/m*
_output_shapes
:*
dtype0
?
 Adam/rank_net_1/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/rank_net_1/dense_3/kernel/m
?
4Adam/rank_net_1/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/rank_net_1/dense_3/kernel/m*
_output_shapes

:*
dtype0
?
Adam/rank_net_1/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/rank_net_1/dense_3/bias/m
?
2Adam/rank_net_1/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/rank_net_1/dense_3/bias/m*
_output_shapes
:*
dtype0
?
 Adam/rank_net_1/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/rank_net_1/dense_4/kernel/m
?
4Adam/rank_net_1/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/rank_net_1/dense_4/kernel/m*
_output_shapes

:*
dtype0
?
Adam/rank_net_1/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/rank_net_1/dense_4/bias/m
?
2Adam/rank_net_1/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/rank_net_1/dense_4/bias/m*
_output_shapes
:*
dtype0
?
 Adam/rank_net_1/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/rank_net_1/dense_5/kernel/v
?
4Adam/rank_net_1/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/rank_net_1/dense_5/kernel/v*
_output_shapes

:*
dtype0
?
Adam/rank_net_1/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/rank_net_1/dense_5/bias/v
?
2Adam/rank_net_1/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/rank_net_1/dense_5/bias/v*
_output_shapes
:*
dtype0
?
 Adam/rank_net_1/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/rank_net_1/dense_3/kernel/v
?
4Adam/rank_net_1/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/rank_net_1/dense_3/kernel/v*
_output_shapes

:*
dtype0
?
Adam/rank_net_1/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/rank_net_1/dense_3/bias/v
?
2Adam/rank_net_1/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/rank_net_1/dense_3/bias/v*
_output_shapes
:*
dtype0
?
 Adam/rank_net_1/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/rank_net_1/dense_4/kernel/v
?
4Adam/rank_net_1/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/rank_net_1/dense_4/kernel/v*
_output_shapes

:*
dtype0
?
Adam/rank_net_1/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/rank_net_1/dense_4/bias/v
?
2Adam/rank_net_1/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/rank_net_1/dense_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?#
value?#B?# B?#
?
	dense
o
oi_minus_oj
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures


0
1
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratemEmFmGmHmImJvKvLvMvNvOvP
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?

layers
 layer_regularization_losses
!layer_metrics
regularization_losses
	variables
"non_trainable_variables
#metrics
trainable_variables
 
h

kernel
bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

kernel
bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
RP
VARIABLE_VALUErank_net_1/dense_5/kernel#o/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUErank_net_1/dense_5/bias!o/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

,layers
-layer_regularization_losses
.layer_metrics
regularization_losses
	variables
/non_trainable_variables
0metrics
trainable_variables
 
 
 
?

1layers
2layer_regularization_losses
3layer_metrics
regularization_losses
	variables
4non_trainable_variables
5metrics
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUErank_net_1/dense_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUErank_net_1/dense_3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUErank_net_1/dense_4/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUErank_net_1/dense_4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE


0
1
2
3
 
 
 

60
 

0
1

0
1
?

7layers
8layer_regularization_losses
9layer_metrics
$regularization_losses
%	variables
:non_trainable_variables
;metrics
&trainable_variables
 

0
1

0
1
?

<layers
=layer_regularization_losses
>layer_metrics
(regularization_losses
)	variables
?non_trainable_variables
@metrics
*trainable_variables
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
4
	Atotal
	Bcount
C	variables
D	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

C	variables
us
VARIABLE_VALUE Adam/rank_net_1/dense_5/kernel/m?o/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/rank_net_1/dense_5/bias/m=o/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/rank_net_1/dense_3/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/rank_net_1/dense_3/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/rank_net_1/dense_4/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/rank_net_1/dense_4/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Adam/rank_net_1/dense_5/kernel/v?o/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/rank_net_1/dense_5/bias/v=o/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/rank_net_1/dense_3/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/rank_net_1/dense_3/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/rank_net_1/dense_4/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/rank_net_1/dense_4/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2rank_net_1/dense_3/kernelrank_net_1/dense_3/biasrank_net_1/dense_4/kernelrank_net_1/dense_4/biasrank_net_1/dense_5/kernelrank_net_1/dense_5/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_407560
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-rank_net_1/dense_5/kernel/Read/ReadVariableOp+rank_net_1/dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-rank_net_1/dense_3/kernel/Read/ReadVariableOp+rank_net_1/dense_3/bias/Read/ReadVariableOp-rank_net_1/dense_4/kernel/Read/ReadVariableOp+rank_net_1/dense_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/rank_net_1/dense_5/kernel/m/Read/ReadVariableOp2Adam/rank_net_1/dense_5/bias/m/Read/ReadVariableOp4Adam/rank_net_1/dense_3/kernel/m/Read/ReadVariableOp2Adam/rank_net_1/dense_3/bias/m/Read/ReadVariableOp4Adam/rank_net_1/dense_4/kernel/m/Read/ReadVariableOp2Adam/rank_net_1/dense_4/bias/m/Read/ReadVariableOp4Adam/rank_net_1/dense_5/kernel/v/Read/ReadVariableOp2Adam/rank_net_1/dense_5/bias/v/Read/ReadVariableOp4Adam/rank_net_1/dense_3/kernel/v/Read/ReadVariableOp2Adam/rank_net_1/dense_3/bias/v/Read/ReadVariableOp4Adam/rank_net_1/dense_4/kernel/v/Read/ReadVariableOp2Adam/rank_net_1/dense_4/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_407730
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerank_net_1/dense_5/kernelrank_net_1/dense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_raterank_net_1/dense_3/kernelrank_net_1/dense_3/biasrank_net_1/dense_4/kernelrank_net_1/dense_4/biastotalcount Adam/rank_net_1/dense_5/kernel/mAdam/rank_net_1/dense_5/bias/m Adam/rank_net_1/dense_3/kernel/mAdam/rank_net_1/dense_3/bias/m Adam/rank_net_1/dense_4/kernel/mAdam/rank_net_1/dense_4/bias/m Adam/rank_net_1/dense_5/kernel/vAdam/rank_net_1/dense_5/bias/v Adam/rank_net_1/dense_3/kernel/vAdam/rank_net_1/dense_3/bias/v Adam/rank_net_1/dense_4/kernel/vAdam/rank_net_1/dense_4/bias/v*%
Tin
2*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_407815??
?
?
C__inference_dense_3_layer_call_and_return_conditional_losses_407418

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
F__inference_rank_net_1_layer_call_and_return_conditional_losses_407513
input_1
input_2
dense_3_407429
dense_3_407431
dense_4_407459
dense_4_407461
dense_5_407488
dense_5_407490
identity??dense_3/StatefulPartitionedCall?!dense_3/StatefulPartitionedCall_1?dense_4/StatefulPartitionedCall?!dense_4/StatefulPartitionedCall_1?dense_5/StatefulPartitionedCall?!dense_5/StatefulPartitionedCall_1?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3_407429dense_3_407431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4074182!
dense_3/StatefulPartitionedCall?
!dense_3/StatefulPartitionedCall_1StatefulPartitionedCallinput_2dense_3_407429dense_3_407431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4074182#
!dense_3/StatefulPartitionedCall_1?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_407459dense_4_407461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4074482!
dense_4/StatefulPartitionedCall?
!dense_4/StatefulPartitionedCall_1StatefulPartitionedCall*dense_3/StatefulPartitionedCall_1:output:0dense_4_407459dense_4_407461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4074482#
!dense_4/StatefulPartitionedCall_1?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_407488dense_5_407490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_4074772!
dense_5/StatefulPartitionedCall?
!dense_5/StatefulPartitionedCall_1StatefulPartitionedCall*dense_4/StatefulPartitionedCall_1:output:0dense_5_407488dense_5_407490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_4074772#
!dense_5/StatefulPartitionedCall_1?
subtract_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*dense_5/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_subtract_1_layer_call_and_return_conditional_losses_4075022
subtract_1/PartitionedCall?
activation/SigmoidSigmoid#subtract_1/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
activation/Sigmoid?
IdentityIdentityactivation/Sigmoid:y:0 ^dense_3/StatefulPartitionedCall"^dense_3/StatefulPartitionedCall_1 ^dense_4/StatefulPartitionedCall"^dense_4/StatefulPartitionedCall_1 ^dense_5/StatefulPartitionedCall"^dense_5/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dense_3/StatefulPartitionedCall_1!dense_3/StatefulPartitionedCall_12B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dense_4/StatefulPartitionedCall_1!dense_4/StatefulPartitionedCall_12B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dense_5/StatefulPartitionedCall_1!dense_5/StatefulPartitionedCall_1:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?;
?
__inference__traced_save_407730
file_prefix8
4savev2_rank_net_1_dense_5_kernel_read_readvariableop6
2savev2_rank_net_1_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_rank_net_1_dense_3_kernel_read_readvariableop6
2savev2_rank_net_1_dense_3_bias_read_readvariableop8
4savev2_rank_net_1_dense_4_kernel_read_readvariableop6
2savev2_rank_net_1_dense_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_rank_net_1_dense_5_kernel_m_read_readvariableop=
9savev2_adam_rank_net_1_dense_5_bias_m_read_readvariableop?
;savev2_adam_rank_net_1_dense_3_kernel_m_read_readvariableop=
9savev2_adam_rank_net_1_dense_3_bias_m_read_readvariableop?
;savev2_adam_rank_net_1_dense_4_kernel_m_read_readvariableop=
9savev2_adam_rank_net_1_dense_4_bias_m_read_readvariableop?
;savev2_adam_rank_net_1_dense_5_kernel_v_read_readvariableop=
9savev2_adam_rank_net_1_dense_5_bias_v_read_readvariableop?
;savev2_adam_rank_net_1_dense_3_kernel_v_read_readvariableop=
9savev2_adam_rank_net_1_dense_3_bias_v_read_readvariableop?
;savev2_adam_rank_net_1_dense_4_kernel_v_read_readvariableop=
9savev2_adam_rank_net_1_dense_4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d275c1c004c64601a0adad936323b6a5/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?
B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_rank_net_1_dense_5_kernel_read_readvariableop2savev2_rank_net_1_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_rank_net_1_dense_3_kernel_read_readvariableop2savev2_rank_net_1_dense_3_bias_read_readvariableop4savev2_rank_net_1_dense_4_kernel_read_readvariableop2savev2_rank_net_1_dense_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_rank_net_1_dense_5_kernel_m_read_readvariableop9savev2_adam_rank_net_1_dense_5_bias_m_read_readvariableop;savev2_adam_rank_net_1_dense_3_kernel_m_read_readvariableop9savev2_adam_rank_net_1_dense_3_bias_m_read_readvariableop;savev2_adam_rank_net_1_dense_4_kernel_m_read_readvariableop9savev2_adam_rank_net_1_dense_4_bias_m_read_readvariableop;savev2_adam_rank_net_1_dense_5_kernel_v_read_readvariableop9savev2_adam_rank_net_1_dense_5_bias_v_read_readvariableop;savev2_adam_rank_net_1_dense_3_kernel_v_read_readvariableop9savev2_adam_rank_net_1_dense_3_bias_v_read_readvariableop;savev2_adam_rank_net_1_dense_4_kernel_v_read_readvariableop9savev2_adam_rank_net_1_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : ::::: : ::::::::::::: 2(
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
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
C__inference_dense_5_layer_call_and_return_conditional_losses_407570

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_3_layer_call_and_return_conditional_losses_407602

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
F__inference_subtract_1_layer_call_and_return_conditional_losses_407585
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?4
?
!__inference__wrapped_model_407402
input_1
input_25
1rank_net_1_dense_3_matmul_readvariableop_resource6
2rank_net_1_dense_3_biasadd_readvariableop_resource5
1rank_net_1_dense_4_matmul_readvariableop_resource6
2rank_net_1_dense_4_biasadd_readvariableop_resource5
1rank_net_1_dense_5_matmul_readvariableop_resource6
2rank_net_1_dense_5_biasadd_readvariableop_resource
identity??
(rank_net_1/dense_3/MatMul/ReadVariableOpReadVariableOp1rank_net_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net_1/dense_3/MatMul/ReadVariableOp?
rank_net_1/dense_3/MatMulMatMulinput_10rank_net_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_3/MatMul?
)rank_net_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp2rank_net_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net_1/dense_3/BiasAdd/ReadVariableOp?
rank_net_1/dense_3/BiasAddBiasAdd#rank_net_1/dense_3/MatMul:product:01rank_net_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_3/BiasAdd?
rank_net_1/dense_3/LeakyRelu	LeakyRelu#rank_net_1/dense_3/BiasAdd:output:0*'
_output_shapes
:?????????2
rank_net_1/dense_3/LeakyRelu?
*rank_net_1/dense_3/MatMul_1/ReadVariableOpReadVariableOp1rank_net_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*rank_net_1/dense_3/MatMul_1/ReadVariableOp?
rank_net_1/dense_3/MatMul_1MatMulinput_22rank_net_1/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_3/MatMul_1?
+rank_net_1/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp2rank_net_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+rank_net_1/dense_3/BiasAdd_1/ReadVariableOp?
rank_net_1/dense_3/BiasAdd_1BiasAdd%rank_net_1/dense_3/MatMul_1:product:03rank_net_1/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_3/BiasAdd_1?
rank_net_1/dense_3/LeakyRelu_1	LeakyRelu%rank_net_1/dense_3/BiasAdd_1:output:0*'
_output_shapes
:?????????2 
rank_net_1/dense_3/LeakyRelu_1?
(rank_net_1/dense_4/MatMul/ReadVariableOpReadVariableOp1rank_net_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net_1/dense_4/MatMul/ReadVariableOp?
rank_net_1/dense_4/MatMulMatMul*rank_net_1/dense_3/LeakyRelu:activations:00rank_net_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_4/MatMul?
)rank_net_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp2rank_net_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net_1/dense_4/BiasAdd/ReadVariableOp?
rank_net_1/dense_4/BiasAddBiasAdd#rank_net_1/dense_4/MatMul:product:01rank_net_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_4/BiasAdd?
rank_net_1/dense_4/LeakyRelu	LeakyRelu#rank_net_1/dense_4/BiasAdd:output:0*'
_output_shapes
:?????????2
rank_net_1/dense_4/LeakyRelu?
*rank_net_1/dense_4/MatMul_1/ReadVariableOpReadVariableOp1rank_net_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*rank_net_1/dense_4/MatMul_1/ReadVariableOp?
rank_net_1/dense_4/MatMul_1MatMul,rank_net_1/dense_3/LeakyRelu_1:activations:02rank_net_1/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_4/MatMul_1?
+rank_net_1/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp2rank_net_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+rank_net_1/dense_4/BiasAdd_1/ReadVariableOp?
rank_net_1/dense_4/BiasAdd_1BiasAdd%rank_net_1/dense_4/MatMul_1:product:03rank_net_1/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_4/BiasAdd_1?
rank_net_1/dense_4/LeakyRelu_1	LeakyRelu%rank_net_1/dense_4/BiasAdd_1:output:0*'
_output_shapes
:?????????2 
rank_net_1/dense_4/LeakyRelu_1?
(rank_net_1/dense_5/MatMul/ReadVariableOpReadVariableOp1rank_net_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net_1/dense_5/MatMul/ReadVariableOp?
rank_net_1/dense_5/MatMulMatMul*rank_net_1/dense_4/LeakyRelu:activations:00rank_net_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_5/MatMul?
)rank_net_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp2rank_net_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net_1/dense_5/BiasAdd/ReadVariableOp?
rank_net_1/dense_5/BiasAddBiasAdd#rank_net_1/dense_5/MatMul:product:01rank_net_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_5/BiasAdd?
*rank_net_1/dense_5/MatMul_1/ReadVariableOpReadVariableOp1rank_net_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*rank_net_1/dense_5/MatMul_1/ReadVariableOp?
rank_net_1/dense_5/MatMul_1MatMul,rank_net_1/dense_4/LeakyRelu_1:activations:02rank_net_1/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_5/MatMul_1?
+rank_net_1/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp2rank_net_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+rank_net_1/dense_5/BiasAdd_1/ReadVariableOp?
rank_net_1/dense_5/BiasAdd_1BiasAdd%rank_net_1/dense_5/MatMul_1:product:03rank_net_1/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net_1/dense_5/BiasAdd_1?
rank_net_1/subtract_1/subSub#rank_net_1/dense_5/BiasAdd:output:0%rank_net_1/dense_5/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
rank_net_1/subtract_1/sub?
rank_net_1/activation/SigmoidSigmoidrank_net_1/subtract_1/sub:z:0*
T0*'
_output_shapes
:?????????2
rank_net_1/activation/Sigmoidu
IdentityIdentity!rank_net_1/activation/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
C__inference_dense_5_layer_call_and_return_conditional_losses_407477

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_3_layer_call_fn_407611

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4074182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_4_layer_call_and_return_conditional_losses_407622

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_4_layer_call_fn_407631

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4074482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?k
?
"__inference__traced_restore_407815
file_prefix.
*assignvariableop_rank_net_1_dense_5_kernel.
*assignvariableop_1_rank_net_1_dense_5_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate0
,assignvariableop_7_rank_net_1_dense_3_kernel.
*assignvariableop_8_rank_net_1_dense_3_bias0
,assignvariableop_9_rank_net_1_dense_4_kernel/
+assignvariableop_10_rank_net_1_dense_4_bias
assignvariableop_11_total
assignvariableop_12_count8
4assignvariableop_13_adam_rank_net_1_dense_5_kernel_m6
2assignvariableop_14_adam_rank_net_1_dense_5_bias_m8
4assignvariableop_15_adam_rank_net_1_dense_3_kernel_m6
2assignvariableop_16_adam_rank_net_1_dense_3_bias_m8
4assignvariableop_17_adam_rank_net_1_dense_4_kernel_m6
2assignvariableop_18_adam_rank_net_1_dense_4_bias_m8
4assignvariableop_19_adam_rank_net_1_dense_5_kernel_v6
2assignvariableop_20_adam_rank_net_1_dense_5_bias_v8
4assignvariableop_21_adam_rank_net_1_dense_3_kernel_v6
2assignvariableop_22_adam_rank_net_1_dense_3_bias_v8
4assignvariableop_23_adam_rank_net_1_dense_4_kernel_v6
2assignvariableop_24_adam_rank_net_1_dense_4_bias_v
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?
B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_rank_net_1_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp*assignvariableop_1_rank_net_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_rank_net_1_dense_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_rank_net_1_dense_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_rank_net_1_dense_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_rank_net_1_dense_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_rank_net_1_dense_5_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp2assignvariableop_14_adam_rank_net_1_dense_5_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp4assignvariableop_15_adam_rank_net_1_dense_3_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_rank_net_1_dense_3_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_rank_net_1_dense_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_rank_net_1_dense_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_rank_net_1_dense_5_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_rank_net_1_dense_5_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_rank_net_1_dense_3_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_rank_net_1_dense_3_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_rank_net_1_dense_4_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_rank_net_1_dense_4_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25?
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
$__inference_signature_wrapper_407560
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_4074022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
p
F__inference_subtract_1_layer_call_and_return_conditional_losses_407502

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
W
+__inference_subtract_1_layer_call_fn_407591
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_subtract_1_layer_call_and_return_conditional_losses_4075022
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
+__inference_rank_net_1_layer_call_fn_407532
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_rank_net_1_layer_call_and_return_conditional_losses_4075132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
C__inference_dense_4_layer_call_and_return_conditional_losses_407448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_5_layer_call_fn_407579

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_4074772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?m
?
	dense
o
oi_minus_oj
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
*Q&call_and_return_all_conditional_losses
R__call__
S_default_save_signature"?
_tf_keras_model?{"class_name": "RankNet", "name": "rank_net_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "RankNet"}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
.

0
1"
trackable_list_wrapper
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"?
_tf_keras_layer?{"class_name": "Subtract", "name": "subtract_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "subtract_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
?
iter

beta_1

beta_2
	decay
learning_ratemEmFmGmHmImJvKvLvMvNvOvP"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?

layers
 layer_regularization_losses
!layer_metrics
regularization_losses
	variables
"non_trainable_variables
#metrics
trainable_variables
R__call__
S_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
,
Xserving_default"
signature_map
?

kernel
bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 16, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
?

kernel
bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
*[&call_and_return_all_conditional_losses
\__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 8, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
+:)2rank_net_1/dense_5/kernel
%:#2rank_net_1/dense_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

,layers
-layer_regularization_losses
.layer_metrics
regularization_losses
	variables
/non_trainable_variables
0metrics
trainable_variables
U__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

1layers
2layer_regularization_losses
3layer_metrics
regularization_losses
	variables
4non_trainable_variables
5metrics
trainable_variables
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)2rank_net_1/dense_3/kernel
%:#2rank_net_1/dense_3/bias
+:)2rank_net_1/dense_4/kernel
%:#2rank_net_1/dense_4/bias
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

7layers
8layer_regularization_losses
9layer_metrics
$regularization_losses
%	variables
:non_trainable_variables
;metrics
&trainable_variables
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

<layers
=layer_regularization_losses
>layer_metrics
(regularization_losses
)	variables
?non_trainable_variables
@metrics
*trainable_variables
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
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
?
	Atotal
	Bcount
C	variables
D	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
0:.2 Adam/rank_net_1/dense_5/kernel/m
*:(2Adam/rank_net_1/dense_5/bias/m
0:.2 Adam/rank_net_1/dense_3/kernel/m
*:(2Adam/rank_net_1/dense_3/bias/m
0:.2 Adam/rank_net_1/dense_4/kernel/m
*:(2Adam/rank_net_1/dense_4/bias/m
0:.2 Adam/rank_net_1/dense_5/kernel/v
*:(2Adam/rank_net_1/dense_5/bias/v
0:.2 Adam/rank_net_1/dense_3/kernel/v
*:(2Adam/rank_net_1/dense_3/bias/v
0:.2 Adam/rank_net_1/dense_4/kernel/v
*:(2Adam/rank_net_1/dense_4/bias/v
?2?
F__inference_rank_net_1_layer_call_and_return_conditional_losses_407513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *N?K
I?F
!?
input_1?????????
!?
input_2?????????
?2?
+__inference_rank_net_1_layer_call_fn_407532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *N?K
I?F
!?
input_1?????????
!?
input_2?????????
?2?
!__inference__wrapped_model_407402?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *N?K
I?F
!?
input_1?????????
!?
input_2?????????
?2?
C__inference_dense_5_layer_call_and_return_conditional_losses_407570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_5_layer_call_fn_407579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_subtract_1_layer_call_and_return_conditional_losses_407585?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_subtract_1_layer_call_fn_407591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:B8
$__inference_signature_wrapper_407560input_1input_2
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_407602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_3_layer_call_fn_407611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_4_layer_call_and_return_conditional_losses_407622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_4_layer_call_fn_407631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_407402?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "3?0
.
output_1"?
output_1??????????
C__inference_dense_3_layer_call_and_return_conditional_losses_407602\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_407611O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_4_layer_call_and_return_conditional_losses_407622\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_4_layer_call_fn_407631O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_5_layer_call_and_return_conditional_losses_407570\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_5_layer_call_fn_407579O/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_rank_net_1_layer_call_and_return_conditional_losses_407513?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "%?"
?
0?????????
? ?
+__inference_rank_net_1_layer_call_fn_407532|X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "???????????
$__inference_signature_wrapper_407560?i?f
? 
_?\
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????"3?0
.
output_1"?
output_1??????????
F__inference_subtract_1_layer_call_and_return_conditional_losses_407585?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
+__inference_subtract_1_layer_call_fn_407591vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????