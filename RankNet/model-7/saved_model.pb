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
rank_net/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namerank_net/dense_2/kernel
?
+rank_net/dense_2/kernel/Read/ReadVariableOpReadVariableOprank_net/dense_2/kernel*
_output_shapes

:*
dtype0
?
rank_net/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namerank_net/dense_2/bias
{
)rank_net/dense_2/bias/Read/ReadVariableOpReadVariableOprank_net/dense_2/bias*
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
rank_net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namerank_net/dense/kernel

)rank_net/dense/kernel/Read/ReadVariableOpReadVariableOprank_net/dense/kernel*
_output_shapes

:*
dtype0
~
rank_net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namerank_net/dense/bias
w
'rank_net/dense/bias/Read/ReadVariableOpReadVariableOprank_net/dense/bias*
_output_shapes
:*
dtype0
?
rank_net/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namerank_net/dense_1/kernel
?
+rank_net/dense_1/kernel/Read/ReadVariableOpReadVariableOprank_net/dense_1/kernel*
_output_shapes

:*
dtype0
?
rank_net/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namerank_net/dense_1/bias
{
)rank_net/dense_1/bias/Read/ReadVariableOpReadVariableOprank_net/dense_1/bias*
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
Adam/rank_net/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/rank_net/dense_2/kernel/m
?
2Adam/rank_net/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_2/kernel/m*
_output_shapes

:*
dtype0
?
Adam/rank_net/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/rank_net/dense_2/bias/m
?
0Adam/rank_net/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/rank_net/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/rank_net/dense/kernel/m
?
0Adam/rank_net/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense/kernel/m*
_output_shapes

:*
dtype0
?
Adam/rank_net/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/rank_net/dense/bias/m
?
.Adam/rank_net/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/rank_net/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/rank_net/dense_1/kernel/m
?
2Adam/rank_net/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_1/kernel/m*
_output_shapes

:*
dtype0
?
Adam/rank_net/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/rank_net/dense_1/bias/m
?
0Adam/rank_net/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/rank_net/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/rank_net/dense_2/kernel/v
?
2Adam/rank_net/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_2/kernel/v*
_output_shapes

:*
dtype0
?
Adam/rank_net/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/rank_net/dense_2/bias/v
?
0Adam/rank_net/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/rank_net/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/rank_net/dense/kernel/v
?
0Adam/rank_net/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense/kernel/v*
_output_shapes

:*
dtype0
?
Adam/rank_net/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/rank_net/dense/bias/v
?
.Adam/rank_net/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/rank_net/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/rank_net/dense_1/kernel/v
?
2Adam/rank_net/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_1/kernel/v*
_output_shapes

:*
dtype0
?
Adam/rank_net/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/rank_net/dense_1/bias/v
?
0Adam/rank_net/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/rank_net/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?$
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
trainable_variables
	variables
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
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
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
metrics
 layer_regularization_losses
!non_trainable_variables
regularization_losses
trainable_variables
	variables

"layers
#layer_metrics
 
h

kernel
bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

kernel
bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
PN
VARIABLE_VALUErank_net/dense_2/kernel#o/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUErank_net/dense_2/bias!o/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
,metrics
-non_trainable_variables
.layer_regularization_losses
regularization_losses
trainable_variables
	variables

/layers
0layer_metrics
 
 
 
?
1metrics
2non_trainable_variables
3layer_regularization_losses
regularization_losses
trainable_variables
	variables

4layers
5layer_metrics
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
[Y
VARIABLE_VALUErank_net/dense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErank_net/dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUErank_net/dense_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErank_net/dense_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

60
 
 


0
1
2
3
 
 

0
1

0
1
?
7metrics
8non_trainable_variables
9layer_regularization_losses
$regularization_losses
%trainable_variables
&	variables

:layers
;layer_metrics
 

0
1

0
1
?
<metrics
=non_trainable_variables
>layer_regularization_losses
(regularization_losses
)trainable_variables
*	variables

?layers
@layer_metrics
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
sq
VARIABLE_VALUEAdam/rank_net/dense_2/kernel/m?o/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/rank_net/dense_2/bias/m=o/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rank_net/dense/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/rank_net/dense/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/rank_net/dense_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rank_net/dense_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/rank_net/dense_2/kernel/v?o/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/rank_net/dense_2/bias/v=o/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rank_net/dense/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/rank_net/dense/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/rank_net/dense_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rank_net/dense_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2rank_net/dense/kernelrank_net/dense/biasrank_net/dense_1/kernelrank_net/dense_1/biasrank_net/dense_2/kernelrank_net/dense_2/bias*
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
$__inference_signature_wrapper_179323
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+rank_net/dense_2/kernel/Read/ReadVariableOp)rank_net/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)rank_net/dense/kernel/Read/ReadVariableOp'rank_net/dense/bias/Read/ReadVariableOp+rank_net/dense_1/kernel/Read/ReadVariableOp)rank_net/dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/rank_net/dense_2/kernel/m/Read/ReadVariableOp0Adam/rank_net/dense_2/bias/m/Read/ReadVariableOp0Adam/rank_net/dense/kernel/m/Read/ReadVariableOp.Adam/rank_net/dense/bias/m/Read/ReadVariableOp2Adam/rank_net/dense_1/kernel/m/Read/ReadVariableOp0Adam/rank_net/dense_1/bias/m/Read/ReadVariableOp2Adam/rank_net/dense_2/kernel/v/Read/ReadVariableOp0Adam/rank_net/dense_2/bias/v/Read/ReadVariableOp0Adam/rank_net/dense/kernel/v/Read/ReadVariableOp.Adam/rank_net/dense/bias/v/Read/ReadVariableOp2Adam/rank_net/dense_1/kernel/v/Read/ReadVariableOp0Adam/rank_net/dense_1/bias/v/Read/ReadVariableOpConst*&
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
__inference__traced_save_179493
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerank_net/dense_2/kernelrank_net/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_raterank_net/dense/kernelrank_net/dense/biasrank_net/dense_1/kernelrank_net/dense_1/biastotalcountAdam/rank_net/dense_2/kernel/mAdam/rank_net/dense_2/bias/mAdam/rank_net/dense/kernel/mAdam/rank_net/dense/bias/mAdam/rank_net/dense_1/kernel/mAdam/rank_net/dense_1/bias/mAdam/rank_net/dense_2/kernel/vAdam/rank_net/dense_2/bias/vAdam/rank_net/dense/kernel/vAdam/rank_net/dense/bias/vAdam/rank_net/dense_1/kernel/vAdam/rank_net/dense_1/bias/v*%
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
"__inference__traced_restore_179578??
?
?
A__inference_dense_layer_call_and_return_conditional_losses_179181

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_1_layer_call_fn_179394

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
C__inference_dense_1_layer_call_and_return_conditional_losses_1792112
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
?2
?
!__inference__wrapped_model_179165
input_1
input_21
-rank_net_dense_matmul_readvariableop_resource2
.rank_net_dense_biasadd_readvariableop_resource3
/rank_net_dense_1_matmul_readvariableop_resource4
0rank_net_dense_1_biasadd_readvariableop_resource3
/rank_net_dense_2_matmul_readvariableop_resource4
0rank_net_dense_2_biasadd_readvariableop_resource
identity??
$rank_net/dense/MatMul/ReadVariableOpReadVariableOp-rank_net_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$rank_net/dense/MatMul/ReadVariableOp?
rank_net/dense/MatMulMatMulinput_1,rank_net/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/MatMul?
%rank_net/dense/BiasAdd/ReadVariableOpReadVariableOp.rank_net_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%rank_net/dense/BiasAdd/ReadVariableOp?
rank_net/dense/BiasAddBiasAddrank_net/dense/MatMul:product:0-rank_net/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/BiasAdd?
rank_net/dense/LeakyRelu	LeakyRelurank_net/dense/BiasAdd:output:0*'
_output_shapes
:?????????2
rank_net/dense/LeakyRelu?
&rank_net/dense/MatMul_1/ReadVariableOpReadVariableOp-rank_net_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&rank_net/dense/MatMul_1/ReadVariableOp?
rank_net/dense/MatMul_1MatMulinput_2.rank_net/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/MatMul_1?
'rank_net/dense/BiasAdd_1/ReadVariableOpReadVariableOp.rank_net_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rank_net/dense/BiasAdd_1/ReadVariableOp?
rank_net/dense/BiasAdd_1BiasAdd!rank_net/dense/MatMul_1:product:0/rank_net/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/BiasAdd_1?
rank_net/dense/LeakyRelu_1	LeakyRelu!rank_net/dense/BiasAdd_1:output:0*'
_output_shapes
:?????????2
rank_net/dense/LeakyRelu_1?
&rank_net/dense_1/MatMul/ReadVariableOpReadVariableOp/rank_net_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&rank_net/dense_1/MatMul/ReadVariableOp?
rank_net/dense_1/MatMulMatMul&rank_net/dense/LeakyRelu:activations:0.rank_net/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/MatMul?
'rank_net/dense_1/BiasAdd/ReadVariableOpReadVariableOp0rank_net_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rank_net/dense_1/BiasAdd/ReadVariableOp?
rank_net/dense_1/BiasAddBiasAdd!rank_net/dense_1/MatMul:product:0/rank_net/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/BiasAdd?
rank_net/dense_1/LeakyRelu	LeakyRelu!rank_net/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????2
rank_net/dense_1/LeakyRelu?
(rank_net/dense_1/MatMul_1/ReadVariableOpReadVariableOp/rank_net_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net/dense_1/MatMul_1/ReadVariableOp?
rank_net/dense_1/MatMul_1MatMul(rank_net/dense/LeakyRelu_1:activations:00rank_net/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/MatMul_1?
)rank_net/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp0rank_net_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net/dense_1/BiasAdd_1/ReadVariableOp?
rank_net/dense_1/BiasAdd_1BiasAdd#rank_net/dense_1/MatMul_1:product:01rank_net/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/BiasAdd_1?
rank_net/dense_1/LeakyRelu_1	LeakyRelu#rank_net/dense_1/BiasAdd_1:output:0*'
_output_shapes
:?????????2
rank_net/dense_1/LeakyRelu_1?
&rank_net/dense_2/MatMul/ReadVariableOpReadVariableOp/rank_net_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&rank_net/dense_2/MatMul/ReadVariableOp?
rank_net/dense_2/MatMulMatMul(rank_net/dense_1/LeakyRelu:activations:0.rank_net/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/MatMul?
'rank_net/dense_2/BiasAdd/ReadVariableOpReadVariableOp0rank_net_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rank_net/dense_2/BiasAdd/ReadVariableOp?
rank_net/dense_2/BiasAddBiasAdd!rank_net/dense_2/MatMul:product:0/rank_net/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/BiasAdd?
(rank_net/dense_2/MatMul_1/ReadVariableOpReadVariableOp/rank_net_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net/dense_2/MatMul_1/ReadVariableOp?
rank_net/dense_2/MatMul_1MatMul*rank_net/dense_1/LeakyRelu_1:activations:00rank_net/dense_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/MatMul_1?
)rank_net/dense_2/BiasAdd_1/ReadVariableOpReadVariableOp0rank_net_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net/dense_2/BiasAdd_1/ReadVariableOp?
rank_net/dense_2/BiasAdd_1BiasAdd#rank_net/dense_2/MatMul_1:product:01rank_net/dense_2/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/BiasAdd_1?
rank_net/subtract/subSub!rank_net/dense_2/BiasAdd:output:0#rank_net/dense_2/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
rank_net/subtract/sub?
rank_net/activation/SigmoidSigmoidrank_net/subtract/sub:z:0*
T0*'
_output_shapes
:?????????2
rank_net/activation/Sigmoids
IdentityIdentityrank_net/activation/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
p
D__inference_subtract_layer_call_and_return_conditional_losses_179348
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
?
?
$__inference_signature_wrapper_179323
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
!__inference__wrapped_model_1791652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_179333

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
C__inference_dense_1_layer_call_and_return_conditional_losses_179385

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
U
)__inference_subtract_layer_call_fn_179354
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
GPU2*0J 8? *M
fHRF
D__inference_subtract_layer_call_and_return_conditional_losses_1792652
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
?
}
(__inference_dense_2_layer_call_fn_179342

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
C__inference_dense_2_layer_call_and_return_conditional_losses_1792402
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
 
_user_specified_nameinputs
?
n
D__inference_subtract_layer_call_and_return_conditional_losses_179265

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
?
?
)__inference_rank_net_layer_call_fn_179295
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
GPU2*0J 8? *M
fHRF
D__inference_rank_net_layer_call_and_return_conditional_losses_1792762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?l
?
"__inference__traced_restore_179578
file_prefix,
(assignvariableop_rank_net_dense_2_kernel,
(assignvariableop_1_rank_net_dense_2_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate,
(assignvariableop_7_rank_net_dense_kernel*
&assignvariableop_8_rank_net_dense_bias.
*assignvariableop_9_rank_net_dense_1_kernel-
)assignvariableop_10_rank_net_dense_1_bias
assignvariableop_11_total
assignvariableop_12_count6
2assignvariableop_13_adam_rank_net_dense_2_kernel_m4
0assignvariableop_14_adam_rank_net_dense_2_bias_m4
0assignvariableop_15_adam_rank_net_dense_kernel_m2
.assignvariableop_16_adam_rank_net_dense_bias_m6
2assignvariableop_17_adam_rank_net_dense_1_kernel_m4
0assignvariableop_18_adam_rank_net_dense_1_bias_m6
2assignvariableop_19_adam_rank_net_dense_2_kernel_v4
0assignvariableop_20_adam_rank_net_dense_2_bias_v4
0assignvariableop_21_adam_rank_net_dense_kernel_v2
.assignvariableop_22_adam_rank_net_dense_bias_v6
2assignvariableop_23_adam_rank_net_dense_1_kernel_v4
0assignvariableop_24_adam_rank_net_dense_1_bias_v
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp(assignvariableop_rank_net_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_rank_net_dense_2_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp(assignvariableop_7_rank_net_dense_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_rank_net_dense_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_rank_net_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rank_net_dense_1_biasIdentity_10:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp2assignvariableop_13_adam_rank_net_dense_2_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_adam_rank_net_dense_2_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_adam_rank_net_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_rank_net_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_rank_net_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_rank_net_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_rank_net_dense_2_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_rank_net_dense_2_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_rank_net_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_rank_net_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_rank_net_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_rank_net_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
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
C__inference_dense_1_layer_call_and_return_conditional_losses_179211

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
?
?
A__inference_dense_layer_call_and_return_conditional_losses_179365

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense_layer_call_fn_179374

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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1791812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
__inference__traced_save_179493
file_prefix6
2savev2_rank_net_dense_2_kernel_read_readvariableop4
0savev2_rank_net_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_rank_net_dense_kernel_read_readvariableop2
.savev2_rank_net_dense_bias_read_readvariableop6
2savev2_rank_net_dense_1_kernel_read_readvariableop4
0savev2_rank_net_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_rank_net_dense_2_kernel_m_read_readvariableop;
7savev2_adam_rank_net_dense_2_bias_m_read_readvariableop;
7savev2_adam_rank_net_dense_kernel_m_read_readvariableop9
5savev2_adam_rank_net_dense_bias_m_read_readvariableop=
9savev2_adam_rank_net_dense_1_kernel_m_read_readvariableop;
7savev2_adam_rank_net_dense_1_bias_m_read_readvariableop=
9savev2_adam_rank_net_dense_2_kernel_v_read_readvariableop;
7savev2_adam_rank_net_dense_2_bias_v_read_readvariableop;
7savev2_adam_rank_net_dense_kernel_v_read_readvariableop9
5savev2_adam_rank_net_dense_bias_v_read_readvariableop=
9savev2_adam_rank_net_dense_1_kernel_v_read_readvariableop;
7savev2_adam_rank_net_dense_1_bias_v_read_readvariableop
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
value3B1 B+_temp_06c1b5df2e00404d9d1dcfa22597611f/part2	
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
value?B?B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?o/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=o/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_rank_net_dense_2_kernel_read_readvariableop0savev2_rank_net_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_rank_net_dense_kernel_read_readvariableop.savev2_rank_net_dense_bias_read_readvariableop2savev2_rank_net_dense_1_kernel_read_readvariableop0savev2_rank_net_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_rank_net_dense_2_kernel_m_read_readvariableop7savev2_adam_rank_net_dense_2_bias_m_read_readvariableop7savev2_adam_rank_net_dense_kernel_m_read_readvariableop5savev2_adam_rank_net_dense_bias_m_read_readvariableop9savev2_adam_rank_net_dense_1_kernel_m_read_readvariableop7savev2_adam_rank_net_dense_1_bias_m_read_readvariableop9savev2_adam_rank_net_dense_2_kernel_v_read_readvariableop7savev2_adam_rank_net_dense_2_bias_v_read_readvariableop7savev2_adam_rank_net_dense_kernel_v_read_readvariableop5savev2_adam_rank_net_dense_bias_v_read_readvariableop9savev2_adam_rank_net_dense_1_kernel_v_read_readvariableop7savev2_adam_rank_net_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: ::: : : : : ::::: : ::::::::::::: 2(
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

:: 	
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

:: 
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

:: 
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
?#
?
D__inference_rank_net_layer_call_and_return_conditional_losses_179276
input_1
input_2
dense_179192
dense_179194
dense_1_179222
dense_1_179224
dense_2_179251
dense_2_179253
identity??dense/StatefulPartitionedCall?dense/StatefulPartitionedCall_1?dense_1/StatefulPartitionedCall?!dense_1/StatefulPartitionedCall_1?dense_2/StatefulPartitionedCall?!dense_2/StatefulPartitionedCall_1?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_179192dense_179194*
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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1791812
dense/StatefulPartitionedCall?
dense/StatefulPartitionedCall_1StatefulPartitionedCallinput_2dense_179192dense_179194*
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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1791812!
dense/StatefulPartitionedCall_1?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_179222dense_1_179224*
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
C__inference_dense_1_layer_call_and_return_conditional_losses_1792112!
dense_1/StatefulPartitionedCall?
!dense_1/StatefulPartitionedCall_1StatefulPartitionedCall(dense/StatefulPartitionedCall_1:output:0dense_1_179222dense_1_179224*
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
C__inference_dense_1_layer_call_and_return_conditional_losses_1792112#
!dense_1/StatefulPartitionedCall_1?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_179251dense_2_179253*
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
C__inference_dense_2_layer_call_and_return_conditional_losses_1792402!
dense_2/StatefulPartitionedCall?
!dense_2/StatefulPartitionedCall_1StatefulPartitionedCall*dense_1/StatefulPartitionedCall_1:output:0dense_2_179251dense_2_179253*
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
C__inference_dense_2_layer_call_and_return_conditional_losses_1792402#
!dense_2/StatefulPartitionedCall_1?
subtract/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*dense_2/StatefulPartitionedCall_1:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_subtract_layer_call_and_return_conditional_losses_1792652
subtract/PartitionedCall?
activation/SigmoidSigmoid!subtract/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
activation/Sigmoid?
IdentityIdentityactivation/Sigmoid:y:0^dense/StatefulPartitionedCall ^dense/StatefulPartitionedCall_1 ^dense_1/StatefulPartitionedCall"^dense_1/StatefulPartitionedCall_1 ^dense_2/StatefulPartitionedCall"^dense_2/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense/StatefulPartitionedCall_1dense/StatefulPartitionedCall_12B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dense_1/StatefulPartitionedCall_1!dense_1/StatefulPartitionedCall_12B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dense_2/StatefulPartitionedCall_1!dense_2/StatefulPartitionedCall_1:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_179240

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
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?l
?
	dense
o
oi_minus_oj
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "RankNet", "name": "rank_net", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "RankNet"}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
.

0
1"
trackable_list_wrapper
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Subtract", "name": "subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
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
metrics
 layer_regularization_losses
!non_trainable_variables
regularization_losses
trainable_variables
	variables

"layers
#layer_metrics
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
Xserving_default"
signature_map
?

kernel
bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?

kernel
bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
[__call__
*\&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
):'2rank_net/dense_2/kernel
#:!2rank_net/dense_2/bias
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
,metrics
-non_trainable_variables
.layer_regularization_losses
regularization_losses
trainable_variables
	variables

/layers
0layer_metrics
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1metrics
2non_trainable_variables
3layer_regularization_losses
regularization_losses
trainable_variables
	variables

4layers
5layer_metrics
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%2rank_net/dense/kernel
!:2rank_net/dense/bias
):'2rank_net/dense_1/kernel
#:!2rank_net/dense_1/bias
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
7metrics
8non_trainable_variables
9layer_regularization_losses
$regularization_losses
%trainable_variables
&	variables

:layers
;layer_metrics
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
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
<metrics
=non_trainable_variables
>layer_regularization_losses
(regularization_losses
)trainable_variables
*	variables

?layers
@layer_metrics
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
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
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
.:,2Adam/rank_net/dense_2/kernel/m
(:&2Adam/rank_net/dense_2/bias/m
,:*2Adam/rank_net/dense/kernel/m
&:$2Adam/rank_net/dense/bias/m
.:,2Adam/rank_net/dense_1/kernel/m
(:&2Adam/rank_net/dense_1/bias/m
.:,2Adam/rank_net/dense_2/kernel/v
(:&2Adam/rank_net/dense_2/bias/v
,:*2Adam/rank_net/dense/kernel/v
&:$2Adam/rank_net/dense/bias/v
.:,2Adam/rank_net/dense_1/kernel/v
(:&2Adam/rank_net/dense_1/bias/v
?2?
)__inference_rank_net_layer_call_fn_179295?
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
input_1?????????
!?
input_2?????????
?2?
!__inference__wrapped_model_179165?
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
input_1?????????
!?
input_2?????????
?2?
D__inference_rank_net_layer_call_and_return_conditional_losses_179276?
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
input_1?????????
!?
input_2?????????
?2?
(__inference_dense_2_layer_call_fn_179342?
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
C__inference_dense_2_layer_call_and_return_conditional_losses_179333?
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
)__inference_subtract_layer_call_fn_179354?
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
D__inference_subtract_layer_call_and_return_conditional_losses_179348?
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
$__inference_signature_wrapper_179323input_1input_2
?2?
&__inference_dense_layer_call_fn_179374?
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
A__inference_dense_layer_call_and_return_conditional_losses_179365?
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
(__inference_dense_1_layer_call_fn_179394?
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
C__inference_dense_1_layer_call_and_return_conditional_losses_179385?
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
!__inference__wrapped_model_179165?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "3?0
.
output_1"?
output_1??????????
C__inference_dense_1_layer_call_and_return_conditional_losses_179385\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_1_layer_call_fn_179394O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_2_layer_call_and_return_conditional_losses_179333\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_2_layer_call_fn_179342O/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_179365\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense_layer_call_fn_179374O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_rank_net_layer_call_and_return_conditional_losses_179276?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "%?"
?
0?????????
? ?
)__inference_rank_net_layer_call_fn_179295|X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "???????????
$__inference_signature_wrapper_179323?i?f
? 
_?\
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????"3?0
.
output_1"?
output_1??????????
D__inference_subtract_layer_call_and_return_conditional_losses_179348?Z?W
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
)__inference_subtract_layer_call_fn_179354vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????