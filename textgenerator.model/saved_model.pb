¡Ü
É
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.9.12v2.9.0-18-gd8ce9f9c3018¹
µ
.RMSprop/module_wrapper/lstm/lstm_cell/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.RMSprop/module_wrapper/lstm/lstm_cell/bias/rms
®
BRMSprop/module_wrapper/lstm/lstm_cell/bias/rms/Read/ReadVariableOpReadVariableOp.RMSprop/module_wrapper/lstm/lstm_cell/bias/rms*
_output_shapes	
:*
dtype0
Ò
:RMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*K
shared_name<:RMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rms
Ë
NRMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp:RMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rms* 
_output_shapes
:
*
dtype0
½
0RMSprop/module_wrapper/lstm/lstm_cell/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'*A
shared_name20RMSprop/module_wrapper/lstm/lstm_cell/kernel/rms
¶
DRMSprop/module_wrapper/lstm/lstm_cell/kernel/rms/Read/ReadVariableOpReadVariableOp0RMSprop/module_wrapper/lstm/lstm_cell/kernel/rms*
_output_shapes
:	'*
dtype0

RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:'*
dtype0

RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'*)
shared_nameRMSprop/dense/kernel/rms

,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes
:	'*
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
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	

"module_wrapper/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"module_wrapper/lstm/lstm_cell/bias

6module_wrapper/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOp"module_wrapper/lstm/lstm_cell/bias*
_output_shapes	
:*
dtype0
º
.module_wrapper/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.module_wrapper/lstm/lstm_cell/recurrent_kernel
³
Bmodule_wrapper/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp.module_wrapper/lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0
¥
$module_wrapper/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'*5
shared_name&$module_wrapper/lstm/lstm_cell/kernel

8module_wrapper/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp$module_wrapper/lstm/lstm_cell/kernel*
_output_shapes
:	'*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:'*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	'*
dtype0

NoOpNoOp
Í.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*.
valueþ-Bû- Bô-
§
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
'
"0
#1
$2
3
4*
'
"0
#1
$2
3
4*
* 
°
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 
z
2iter
	3decay
4learning_rate
5momentum
6rho	rmsv	rmsw	"rmsx	#rmsy	$rmsz*

7serving_default* 

"0
#1
$2*

"0
#1
$2*
* 

8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

=trace_0
>trace_1* 

?trace_0
@trace_1* 
ª
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
*E&call_and_return_all_conditional_losses
F__call__
Gcell
H
state_spec*

0
1*

0
1*
* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

Utrace_0* 

Vtrace_0* 
d^
VARIABLE_VALUE$module_wrapper/lstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.module_wrapper/lstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"module_wrapper/lstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

W0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

"0
#1
$2*

"0
#1
$2*
* 


Xlayers
Atrainable_variables
B	variables
Ymetrics
Cregularization_losses

Zstates
[non_trainable_variables
\layer_regularization_losses
]layer_metrics
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

^trace_0
_trace_1* 

`trace_0
atrace_1* 
Ì
btrainable_variables
c	variables
dregularization_losses
e	keras_api
*f&call_and_return_all_conditional_losses
g__call__
h
state_size

"kernel
#recurrent_kernel
$bias*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
i	variables
j	keras_api
	ktotal
	lcount*

G0*
* 
* 
* 
* 
* 
* 
* 
* 
* 

"0
#1
$2*

"0
#1
$2*
* 


mlayers
btrainable_variables
c	variables
nmetrics
dregularization_losses
onon_trainable_variables
player_regularization_losses
qlayer_metrics
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

rtrace_0
strace_1* 

ttrace_0
utrace_1* 
* 

k0
l1*

i	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0RMSprop/module_wrapper/lstm/lstm_cell/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:RMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.RMSprop/module_wrapper/lstm/lstm_cell/bias/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

$serving_default_module_wrapper_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ('
Ü
StatefulPartitionedCallStatefulPartitionedCall$serving_default_module_wrapper_input$module_wrapper/lstm/lstm_cell/kernel"module_wrapper/lstm/lstm_cell/bias.module_wrapper/lstm/lstm_cell/recurrent_kerneldense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_12481
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ä
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp8module_wrapper/lstm/lstm_cell/kernel/Read/ReadVariableOpBmodule_wrapper/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp6module_wrapper/lstm/lstm_cell/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOpDRMSprop/module_wrapper/lstm/lstm_cell/kernel/rms/Read/ReadVariableOpNRMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rms/Read/ReadVariableOpBRMSprop/module_wrapper/lstm/lstm_cell/bias/rms/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_14883
ë
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias$module_wrapper/lstm/lstm_cell/kernel.module_wrapper/lstm/lstm_cell/recurrent_kernel"module_wrapper/lstm/lstm_cell/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/dense/kernel/rmsRMSprop/dense/bias/rms0RMSprop/module_wrapper/lstm/lstm_cell/kernel/rms:RMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rms.RMSprop/module_wrapper/lstm/lstm_cell/bias/rms*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_14944µ©
¥
½
$module_wrapper_lstm_while_body_12890D
@module_wrapper_lstm_while_module_wrapper_lstm_while_loop_counterJ
Fmodule_wrapper_lstm_while_module_wrapper_lstm_while_maximum_iterations)
%module_wrapper_lstm_while_placeholder+
'module_wrapper_lstm_while_placeholder_1+
'module_wrapper_lstm_while_placeholder_2+
'module_wrapper_lstm_while_placeholder_3C
?module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1_0
{module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0V
Cmodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0:	'T
Emodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0:	Q
=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0:
&
"module_wrapper_lstm_while_identity(
$module_wrapper_lstm_while_identity_1(
$module_wrapper_lstm_while_identity_2(
$module_wrapper_lstm_while_identity_3(
$module_wrapper_lstm_while_identity_4(
$module_wrapper_lstm_while_identity_5A
=module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1}
ymodule_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensorT
Amodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource:	'R
Cmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource:	O
;module_wrapper_lstm_while_lstm_cell_readvariableop_resource:
¢2module_wrapper/lstm/while/lstm_cell/ReadVariableOp¢4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1¢4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2¢4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3¢8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp¢:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp
Kmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   
=module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0%module_wrapper_lstm_while_placeholderTmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0u
3module_wrapper/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :½
8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOpCmodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0
)module_wrapper/lstm/while/lstm_cell/splitSplit<module_wrapper/lstm/while/lstm_cell/split/split_dim:output:0@module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_splitá
*module_wrapper/lstm/while/lstm_cell/MatMulMatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
,module_wrapper/lstm/while/lstm_cell/MatMul_1MatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
,module_wrapper/lstm/while/lstm_cell/MatMul_2MatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
,module_wrapper/lstm/while/lstm_cell/MatMul_3MatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5module_wrapper/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ½
:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpEmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0
+module_wrapper/lstm/while/lstm_cell/split_1Split>module_wrapper/lstm/while/lstm_cell/split_1/split_dim:output:0Bmodule_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitÕ
+module_wrapper/lstm/while/lstm_cell/BiasAddBiasAdd4module_wrapper/lstm/while/lstm_cell/MatMul:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
-module_wrapper/lstm/while/lstm_cell/BiasAdd_1BiasAdd6module_wrapper/lstm/while/lstm_cell/MatMul_1:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
-module_wrapper/lstm/while/lstm_cell/BiasAdd_2BiasAdd6module_wrapper/lstm/while/lstm_cell/MatMul_2:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
-module_wrapper/lstm/while/lstm_cell/BiasAdd_3BiasAdd6module_wrapper/lstm/while/lstm_cell/MatMul_3:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2module_wrapper/lstm/while/lstm_cell/ReadVariableOpReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
7module_wrapper/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9module_wrapper/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
9module_wrapper/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¡
1module_wrapper/lstm/while/lstm_cell/strided_sliceStridedSlice:module_wrapper/lstm/while/lstm_cell/ReadVariableOp:value:0@module_wrapper/lstm/while/lstm_cell/strided_slice/stack:output:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice/stack_1:output:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÎ
,module_wrapper/lstm/while/lstm_cell/MatMul_4MatMul'module_wrapper_lstm_while_placeholder_2:module_wrapper/lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
'module_wrapper/lstm/while/lstm_cell/addAddV24module_wrapper/lstm/while/lstm_cell/BiasAdd:output:06module_wrapper/lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)module_wrapper/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+module_wrapper/lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Â
'module_wrapper/lstm/while/lstm_cell/MulMul+module_wrapper/lstm/while/lstm_cell/add:z:02module_wrapper/lstm/while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
)module_wrapper/lstm/while/lstm_cell/Add_1AddV2+module_wrapper/lstm/while/lstm_cell/Mul:z:04module_wrapper/lstm/while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;module_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ì
9module_wrapper/lstm/while/lstm_cell/clip_by_value/MinimumMinimum-module_wrapper/lstm/while/lstm_cell/Add_1:z:0Dmodule_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
3module_wrapper/lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ì
1module_wrapper/lstm/while/lstm_cell/clip_by_valueMaximum=module_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum:z:0<module_wrapper/lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9module_wrapper/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
;module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      «
3module_wrapper/lstm/while/lstm_cell/strided_slice_1StridedSlice<module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1:value:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice_1/stack:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÐ
,module_wrapper/lstm/while/lstm_cell/MatMul_5MatMul'module_wrapper_lstm_while_placeholder_2<module_wrapper/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
)module_wrapper/lstm/while/lstm_cell/add_2AddV26module_wrapper/lstm/while/lstm_cell/BiasAdd_1:output:06module_wrapper/lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+module_wrapper/lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+module_wrapper/lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?È
)module_wrapper/lstm/while/lstm_cell/Mul_1Mul-module_wrapper/lstm/while/lstm_cell/add_2:z:04module_wrapper/lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
)module_wrapper/lstm/while/lstm_cell/Add_3AddV2-module_wrapper/lstm/while/lstm_cell/Mul_1:z:04module_wrapper/lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=module_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ð
;module_wrapper/lstm/while/lstm_cell/clip_by_value_1/MinimumMinimum-module_wrapper/lstm/while/lstm_cell/Add_3:z:0Fmodule_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5module_wrapper/lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ò
3module_wrapper/lstm/while/lstm_cell/clip_by_value_1Maximum?module_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0>module_wrapper/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
)module_wrapper/lstm/while/lstm_cell/mul_2Mul7module_wrapper/lstm/while/lstm_cell/clip_by_value_1:z:0'module_wrapper_lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9module_wrapper/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
;module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      «
3module_wrapper/lstm/while/lstm_cell/strided_slice_2StridedSlice<module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2:value:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice_2/stack:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÐ
,module_wrapper/lstm/while/lstm_cell/MatMul_6MatMul'module_wrapper_lstm_while_placeholder_2<module_wrapper/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
)module_wrapper/lstm/while/lstm_cell/add_4AddV26module_wrapper/lstm/while/lstm_cell/BiasAdd_2:output:06module_wrapper/lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(module_wrapper/lstm/while/lstm_cell/TanhTanh-module_wrapper/lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
)module_wrapper/lstm/while/lstm_cell/mul_3Mul5module_wrapper/lstm/while/lstm_cell/clip_by_value:z:0,module_wrapper/lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
)module_wrapper/lstm/while/lstm_cell/add_5AddV2-module_wrapper/lstm/while/lstm_cell/mul_2:z:0-module_wrapper/lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9module_wrapper/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
;module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
;module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      «
3module_wrapper/lstm/while/lstm_cell/strided_slice_3StridedSlice<module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3:value:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice_3/stack:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÐ
,module_wrapper/lstm/while/lstm_cell/MatMul_7MatMul'module_wrapper_lstm_while_placeholder_2<module_wrapper/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
)module_wrapper/lstm/while/lstm_cell/add_6AddV26module_wrapper/lstm/while/lstm_cell/BiasAdd_3:output:06module_wrapper/lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+module_wrapper/lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+module_wrapper/lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?È
)module_wrapper/lstm/while/lstm_cell/Mul_4Mul-module_wrapper/lstm/while/lstm_cell/add_6:z:04module_wrapper/lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
)module_wrapper/lstm/while/lstm_cell/Add_7AddV2-module_wrapper/lstm/while/lstm_cell/Mul_4:z:04module_wrapper/lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=module_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ð
;module_wrapper/lstm/while/lstm_cell/clip_by_value_2/MinimumMinimum-module_wrapper/lstm/while/lstm_cell/Add_7:z:0Fmodule_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5module_wrapper/lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ò
3module_wrapper/lstm/while/lstm_cell/clip_by_value_2Maximum?module_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0>module_wrapper/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*module_wrapper/lstm/while/lstm_cell/Tanh_1Tanh-module_wrapper/lstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)module_wrapper/lstm/while/lstm_cell/mul_5Mul7module_wrapper/lstm/while/lstm_cell/clip_by_value_2:z:0.module_wrapper/lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>module_wrapper/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'module_wrapper_lstm_while_placeholder_1%module_wrapper_lstm_while_placeholder-module_wrapper/lstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒa
module_wrapper/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
module_wrapper/lstm/while/addAddV2%module_wrapper_lstm_while_placeholder(module_wrapper/lstm/while/add/y:output:0*
T0*
_output_shapes
: c
!module_wrapper/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
module_wrapper/lstm/while/add_1AddV2@module_wrapper_lstm_while_module_wrapper_lstm_while_loop_counter*module_wrapper/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
"module_wrapper/lstm/while/IdentityIdentity#module_wrapper/lstm/while/add_1:z:0^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: º
$module_wrapper/lstm/while/Identity_1IdentityFmodule_wrapper_lstm_while_module_wrapper_lstm_while_maximum_iterations^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: 
$module_wrapper/lstm/while/Identity_2Identity!module_wrapper/lstm/while/add:z:0^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: Â
$module_wrapper/lstm/while/Identity_3IdentityNmodule_wrapper/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: ³
$module_wrapper/lstm/while/Identity_4Identity-module_wrapper/lstm/while/lstm_cell/mul_5:z:0^module_wrapper/lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
$module_wrapper/lstm/while/Identity_5Identity-module_wrapper/lstm/while/lstm_cell/add_5:z:0^module_wrapper/lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
module_wrapper/lstm/while/NoOpNoOp3^module_wrapper/lstm/while/lstm_cell/ReadVariableOp5^module_wrapper/lstm/while/lstm_cell/ReadVariableOp_15^module_wrapper/lstm/while/lstm_cell/ReadVariableOp_25^module_wrapper/lstm/while/lstm_cell/ReadVariableOp_39^module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp;^module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"module_wrapper_lstm_while_identity+module_wrapper/lstm/while/Identity:output:0"U
$module_wrapper_lstm_while_identity_1-module_wrapper/lstm/while/Identity_1:output:0"U
$module_wrapper_lstm_while_identity_2-module_wrapper/lstm/while/Identity_2:output:0"U
$module_wrapper_lstm_while_identity_3-module_wrapper/lstm/while/Identity_3:output:0"U
$module_wrapper_lstm_while_identity_4-module_wrapper/lstm/while/Identity_4:output:0"U
$module_wrapper_lstm_while_identity_5-module_wrapper/lstm/while/Identity_5:output:0"|
;module_wrapper_lstm_while_lstm_cell_readvariableop_resource=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0"
Cmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resourceEmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0"
Amodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resourceCmodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0"
=module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1?module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1_0"ø
ymodule_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor{module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2h
2module_wrapper/lstm/while/lstm_cell/ReadVariableOp2module_wrapper/lstm/while/lstm_cell/ReadVariableOp2l
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_14module_wrapper/lstm/while/lstm_cell/ReadVariableOp_12l
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_24module_wrapper/lstm/while/lstm_cell/ReadVariableOp_22l
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_34module_wrapper/lstm/while/lstm_cell/ReadVariableOp_32t
8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp2x
:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
î
ó
)__inference_lstm_cell_layer_call_fn_14631

inputs
states_0
states_1
unknown:	'
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13898p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Û
à
E__inference_sequential_layer_call_and_return_conditional_losses_12054

inputs'
module_wrapper_12023:	'#
module_wrapper_12025:	(
module_wrapper_12027:

dense_12041:	'
dense_12043:'
identity¢dense/StatefulPartitionedCall¢&module_wrapper/StatefulPartitionedCall
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_12023module_wrapper_12025module_wrapper_12027*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_12022
dense/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0dense_12041dense_12043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_12040Û
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12051r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
NoOpNoOp^dense/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameinputs
è
°

lstm_while_body_12213&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	'E
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	B
.lstm_while_lstm_cell_readvariableop_resource_0:

lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	'C
4lstm_while_lstm_cell_split_1_readvariableop_resource:	@
,lstm_while_lstm_cell_readvariableop_resource:
¢#lstm/while/lstm_cell/ReadVariableOp¢%lstm/while/lstm_cell/ReadVariableOp_1¢%lstm/while/lstm_cell/ReadVariableOp_2¢%lstm/while/lstm_cell/ReadVariableOp_3¢)lstm/while/lstm_cell/split/ReadVariableOp¢+lstm/while/lstm_cell/split_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¿
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0á
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split´
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0×
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split¨
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¡
lstm/while/lstm_cell/MatMul_4MatMullstm_while_placeholder_2+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/MulMullstm/while/lstm_cell/add:z:0#lstm/while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_1AddV2lstm/while/lstm_cell/Mul:z:0%lstm/while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¿
*lstm/while/lstm_cell/clip_by_value/MinimumMinimumlstm/while/lstm_cell/Add_1:z:05lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¿
"lstm/while/lstm_cell/clip_by_valueMaximum.lstm/while/lstm_cell/clip_by_value/Minimum:z:0-lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_5MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_1Mullstm/while/lstm_cell/add_2:z:0%lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_3AddV2lstm/while/lstm_cell/Mul_1:z:0%lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_1/MinimumMinimumlstm/while/lstm_cell/Add_3:z:07lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_1Maximum0lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_2Mul(lstm/while/lstm_cell/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_6MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_3Mul&lstm/while/lstm_cell/clip_by_value:z:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/add_5AddV2lstm/while/lstm_cell/mul_2:z:0lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_7MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_6AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_4Mullstm/while/lstm_cell/add_6:z:0%lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_7AddV2lstm/while/lstm_cell/Mul_4:z:0%lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_2/MinimumMinimumlstm/while/lstm_cell/Add_7:z:07lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_2Maximum0lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_5Mul(lstm/while/lstm_cell/clip_by_value_2:z:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
«

I__inference_module_wrapper_layer_call_and_return_conditional_losses_13315

args_0?
,lstm_lstm_cell_split_readvariableop_resource:	'=
.lstm_lstm_cell_split_1_readvariableop_resource:	:
&lstm_lstm_cell_readvariableop_resource:

identity¢lstm/lstm_cell/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp_1¢lstm/lstm_cell/ReadVariableOp_2¢lstm/lstm_cell/ReadVariableOp_3¢#lstm/lstm_cell/split/ReadVariableOp¢%lstm/lstm_cell/split_1/ReadVariableOp¢
lstm/while@

lstm/ShapeShapeargs_0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeargs_0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ'N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ï
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_mask`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0Ï
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_4MatMullstm/zeros:output:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/MulMullstm/lstm_cell/add:z:0lstm/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_1AddV2lstm/lstm_cell/Mul:z:0lstm/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
$lstm/lstm_cell/clip_by_value/MinimumMinimumlstm/lstm_cell/Add_1:z:0/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm/lstm_cell/clip_by_valueMaximum(lstm/lstm_cell/clip_by_value/Minimum:z:0'lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_5MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_1Mullstm/lstm_cell/add_2:z:0lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_3AddV2lstm/lstm_cell/Mul_1:z:0lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_1/MinimumMinimumlstm/lstm_cell/Add_3:z:01lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_1Maximum*lstm/lstm_cell/clip_by_value_1/Minimum:z:0)lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_2Mul"lstm/lstm_cell/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_6MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_3Mul lstm/lstm_cell/clip_by_value:z:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_5AddV2lstm/lstm_cell/mul_2:z:0lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_7MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_4Mullstm/lstm_cell/add_6:z:0lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_7AddV2lstm/lstm_cell/Mul_4:z:0lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_2/MinimumMinimumlstm/lstm_cell/Add_7:z:01lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_2Maximum*lstm/lstm_cell/clip_by_value_2/Minimum:z:0)lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_5Mul"lstm/lstm_cell/clip_by_value_2:z:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ç
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿY
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_13175*!
condR
lstm_while_cond_13174*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ò
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
IdentityIdentitylstm/strided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(': : : 2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameargs_0
Ñ
ï
*__inference_sequential_layer_call_fn_12511

inputs
unknown:	'
	unknown_0:	
	unknown_1:

	unknown_2:	'
	unknown_3:'
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12396o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameinputs
·½
Â
/sequential_module_wrapper_lstm_while_body_11612Z
Vsequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_loop_counter`
\sequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_maximum_iterations4
0sequential_module_wrapper_lstm_while_placeholder6
2sequential_module_wrapper_lstm_while_placeholder_16
2sequential_module_wrapper_lstm_while_placeholder_26
2sequential_module_wrapper_lstm_while_placeholder_3Y
Usequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_strided_slice_1_0
sequential_module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0a
Nsequential_module_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0:	'_
Psequential_module_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0:	\
Hsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0:
1
-sequential_module_wrapper_lstm_while_identity3
/sequential_module_wrapper_lstm_while_identity_13
/sequential_module_wrapper_lstm_while_identity_23
/sequential_module_wrapper_lstm_while_identity_33
/sequential_module_wrapper_lstm_while_identity_43
/sequential_module_wrapper_lstm_while_identity_5W
Ssequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_strided_slice_1
sequential_module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_
Lsequential_module_wrapper_lstm_while_lstm_cell_split_readvariableop_resource:	']
Nsequential_module_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource:	Z
Fsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resource:
¢=sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp¢?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1¢?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2¢?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3¢Csequential/module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp¢Esequential/module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp§
Vsequential/module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   Â
Hsequential/module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_00sequential_module_wrapper_lstm_while_placeholder_sequential/module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0
>sequential/module_wrapper/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ó
Csequential/module_wrapper/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOpNsequential_module_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0¯
4sequential/module_wrapper/lstm/while/lstm_cell/splitSplitGsequential/module_wrapper/lstm/while/lstm_cell/split/split_dim:output:0Ksequential/module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split
5sequential/module_wrapper/lstm/while/lstm_cell/MatMulMatMulOsequential/module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential/module_wrapper/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7sequential/module_wrapper/lstm/while/lstm_cell/MatMul_1MatMulOsequential/module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential/module_wrapper/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7sequential/module_wrapper/lstm/while/lstm_cell/MatMul_2MatMulOsequential/module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential/module_wrapper/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7sequential/module_wrapper/lstm/while/lstm_cell/MatMul_3MatMulOsequential/module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential/module_wrapper/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential/module_wrapper/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ó
Esequential/module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpPsequential_module_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0¥
6sequential/module_wrapper/lstm/while/lstm_cell/split_1SplitIsequential/module_wrapper/lstm/while/lstm_cell/split_1/split_dim:output:0Msequential/module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitö
6sequential/module_wrapper/lstm/while/lstm_cell/BiasAddBiasAdd?sequential/module_wrapper/lstm/while/lstm_cell/MatMul:product:0?sequential/module_wrapper/lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
8sequential/module_wrapper/lstm/while/lstm_cell/BiasAdd_1BiasAddAsequential/module_wrapper/lstm/while/lstm_cell/MatMul_1:product:0?sequential/module_wrapper/lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
8sequential/module_wrapper/lstm/while/lstm_cell/BiasAdd_2BiasAddAsequential/module_wrapper/lstm/while/lstm_cell/MatMul_2:product:0?sequential/module_wrapper/lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
8sequential/module_wrapper/lstm/while/lstm_cell/BiasAdd_3BiasAddAsequential/module_wrapper/lstm/while/lstm_cell/MatMul_3:product:0?sequential/module_wrapper/lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
=sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOpReadVariableOpHsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
Bsequential/module_wrapper/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Dsequential/module_wrapper/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Dsequential/module_wrapper/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
<sequential/module_wrapper/lstm/while/lstm_cell/strided_sliceStridedSliceEsequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp:value:0Ksequential/module_wrapper/lstm/while/lstm_cell/strided_slice/stack:output:0Msequential/module_wrapper/lstm/while/lstm_cell/strided_slice/stack_1:output:0Msequential/module_wrapper/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskï
7sequential/module_wrapper/lstm/while/lstm_cell/MatMul_4MatMul2sequential_module_wrapper_lstm_while_placeholder_2Esequential/module_wrapper/lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
2sequential/module_wrapper/lstm/while/lstm_cell/addAddV2?sequential/module_wrapper/lstm/while/lstm_cell/BiasAdd:output:0Asequential/module_wrapper/lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
4sequential/module_wrapper/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>{
6sequential/module_wrapper/lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?ã
2sequential/module_wrapper/lstm/while/lstm_cell/MulMul6sequential/module_wrapper/lstm/while/lstm_cell/add:z:0=sequential/module_wrapper/lstm/while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
4sequential/module_wrapper/lstm/while/lstm_cell/Add_1AddV26sequential/module_wrapper/lstm/while/lstm_cell/Mul:z:0?sequential/module_wrapper/lstm/while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Dsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value/MinimumMinimum8sequential/module_wrapper/lstm/while/lstm_cell/Add_1:z:0Osequential/module_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential/module_wrapper/lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
<sequential/module_wrapper/lstm/while/lstm_cell/clip_by_valueMaximumHsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum:z:0Gsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOpHsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
Dsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
Fsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Fsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      â
>sequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1StridedSliceGsequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1:value:0Msequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack:output:0Osequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0Osequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskñ
7sequential/module_wrapper/lstm/while/lstm_cell/MatMul_5MatMul2sequential_module_wrapper_lstm_while_placeholder_2Gsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
4sequential/module_wrapper/lstm/while/lstm_cell/add_2AddV2Asequential/module_wrapper/lstm/while/lstm_cell/BiasAdd_1:output:0Asequential/module_wrapper/lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
6sequential/module_wrapper/lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>{
6sequential/module_wrapper/lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?é
4sequential/module_wrapper/lstm/while/lstm_cell/Mul_1Mul8sequential/module_wrapper/lstm/while/lstm_cell/add_2:z:0?sequential/module_wrapper/lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
4sequential/module_wrapper/lstm/while/lstm_cell/Add_3AddV28sequential/module_wrapper/lstm/while/lstm_cell/Mul_1:z:0?sequential/module_wrapper/lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Hsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Fsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1/MinimumMinimum8sequential/module_wrapper/lstm/while/lstm_cell/Add_3:z:0Qsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
>sequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1MaximumJsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0Isequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
4sequential/module_wrapper/lstm/while/lstm_cell/mul_2MulBsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_1:z:02sequential_module_wrapper_lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOpHsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
Dsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
Fsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
Fsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      â
>sequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2StridedSliceGsequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2:value:0Msequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack:output:0Osequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0Osequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskñ
7sequential/module_wrapper/lstm/while/lstm_cell/MatMul_6MatMul2sequential_module_wrapper_lstm_while_placeholder_2Gsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
4sequential/module_wrapper/lstm/while/lstm_cell/add_4AddV2Asequential/module_wrapper/lstm/while/lstm_cell/BiasAdd_2:output:0Asequential/module_wrapper/lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
3sequential/module_wrapper/lstm/while/lstm_cell/TanhTanh8sequential/module_wrapper/lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
4sequential/module_wrapper/lstm/while/lstm_cell/mul_3Mul@sequential/module_wrapper/lstm/while/lstm_cell/clip_by_value:z:07sequential/module_wrapper/lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
4sequential/module_wrapper/lstm/while/lstm_cell/add_5AddV28sequential/module_wrapper/lstm/while/lstm_cell/mul_2:z:08sequential/module_wrapper/lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOpHsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
Dsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
Fsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Fsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      â
>sequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3StridedSliceGsequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3:value:0Msequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack:output:0Osequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0Osequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskñ
7sequential/module_wrapper/lstm/while/lstm_cell/MatMul_7MatMul2sequential_module_wrapper_lstm_while_placeholder_2Gsequential/module_wrapper/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
4sequential/module_wrapper/lstm/while/lstm_cell/add_6AddV2Asequential/module_wrapper/lstm/while/lstm_cell/BiasAdd_3:output:0Asequential/module_wrapper/lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
6sequential/module_wrapper/lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>{
6sequential/module_wrapper/lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?é
4sequential/module_wrapper/lstm/while/lstm_cell/Mul_4Mul8sequential/module_wrapper/lstm/while/lstm_cell/add_6:z:0?sequential/module_wrapper/lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
4sequential/module_wrapper/lstm/while/lstm_cell/Add_7AddV28sequential/module_wrapper/lstm/while/lstm_cell/Mul_4:z:0?sequential/module_wrapper/lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Hsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Fsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2/MinimumMinimum8sequential/module_wrapper/lstm/while/lstm_cell/Add_7:z:0Qsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
>sequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2MaximumJsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0Isequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
5sequential/module_wrapper/lstm/while/lstm_cell/Tanh_1Tanh8sequential/module_wrapper/lstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
4sequential/module_wrapper/lstm/while/lstm_cell/mul_5MulBsequential/module_wrapper/lstm/while/lstm_cell/clip_by_value_2:z:09sequential/module_wrapper/lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
Isequential/module_wrapper/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2sequential_module_wrapper_lstm_while_placeholder_10sequential_module_wrapper_lstm_while_placeholder8sequential/module_wrapper/lstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒl
*sequential/module_wrapper/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(sequential/module_wrapper/lstm/while/addAddV20sequential_module_wrapper_lstm_while_placeholder3sequential/module_wrapper/lstm/while/add/y:output:0*
T0*
_output_shapes
: n
,sequential/module_wrapper/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*sequential/module_wrapper/lstm/while/add_1AddV2Vsequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_loop_counter5sequential/module_wrapper/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ¶
-sequential/module_wrapper/lstm/while/IdentityIdentity.sequential/module_wrapper/lstm/while/add_1:z:0*^sequential/module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: æ
/sequential/module_wrapper/lstm/while/Identity_1Identity\sequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_maximum_iterations*^sequential/module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: ¶
/sequential/module_wrapper/lstm/while/Identity_2Identity,sequential/module_wrapper/lstm/while/add:z:0*^sequential/module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: ã
/sequential/module_wrapper/lstm/while/Identity_3IdentityYsequential/module_wrapper/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^sequential/module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: Ô
/sequential/module_wrapper/lstm/while/Identity_4Identity8sequential/module_wrapper/lstm/while/lstm_cell/mul_5:z:0*^sequential/module_wrapper/lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
/sequential/module_wrapper/lstm/while/Identity_5Identity8sequential/module_wrapper/lstm/while/lstm_cell/add_5:z:0*^sequential/module_wrapper/lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
)sequential/module_wrapper/lstm/while/NoOpNoOp>^sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp@^sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1@^sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2@^sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3D^sequential/module_wrapper/lstm/while/lstm_cell/split/ReadVariableOpF^sequential/module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "g
-sequential_module_wrapper_lstm_while_identity6sequential/module_wrapper/lstm/while/Identity:output:0"k
/sequential_module_wrapper_lstm_while_identity_18sequential/module_wrapper/lstm/while/Identity_1:output:0"k
/sequential_module_wrapper_lstm_while_identity_28sequential/module_wrapper/lstm/while/Identity_2:output:0"k
/sequential_module_wrapper_lstm_while_identity_38sequential/module_wrapper/lstm/while/Identity_3:output:0"k
/sequential_module_wrapper_lstm_while_identity_48sequential/module_wrapper/lstm/while/Identity_4:output:0"k
/sequential_module_wrapper_lstm_while_identity_58sequential/module_wrapper/lstm/while/Identity_5:output:0"
Fsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resourceHsequential_module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0"¢
Nsequential_module_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resourcePsequential_module_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0"
Lsequential_module_wrapper_lstm_while_lstm_cell_split_readvariableop_resourceNsequential_module_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0"¬
Ssequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_strided_slice_1Usequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_strided_slice_1_0"¦
sequential_module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensorsequential_module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2~
=sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp=sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp2
?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_12
?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_22
?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3?sequential/module_wrapper/lstm/while/lstm_cell/ReadVariableOp_32
Csequential/module_wrapper/lstm/while/lstm_cell/split/ReadVariableOpCsequential/module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp2
Esequential/module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOpEsequential/module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ôJ
¦
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13696

inputs

states
states_10
split_readvariableop_resource:	'.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	'*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates

î
E__inference_sequential_layer_call_and_return_conditional_losses_12458
module_wrapper_input'
module_wrapper_12444:	'#
module_wrapper_12446:	(
module_wrapper_12448:

dense_12451:	'
dense_12453:'
identity¢dense/StatefulPartitionedCall¢&module_wrapper/StatefulPartitionedCall¬
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_12444module_wrapper_12446module_wrapper_12448*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_12353
dense/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0dense_12451dense_12453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_12040Û
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12051r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
NoOpNoOp^dense/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:a ]
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
.
_user_specified_namemodule_wrapper_input
"
É
while_body_13710
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_13734_0:	'&
while_lstm_cell_13736_0:	+
while_lstm_cell_13738_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_13734:	'$
while_lstm_cell_13736:	)
while_lstm_cell_13738:
¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0£
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13734_0while_lstm_cell_13736_0while_lstm_cell_13738_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13696Ù
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_13734while_lstm_cell_13734_0"0
while_lstm_cell_13736while_lstm_cell_13736_0"0
while_lstm_cell_13738while_lstm_cell_13738_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
äx
	
while_body_14201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	'@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	'>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split¥
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0È
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    °
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
äx
	
while_body_14457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	'@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	'>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split¥
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0È
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    °
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
«

I__inference_module_wrapper_layer_call_and_return_conditional_losses_12353

args_0?
,lstm_lstm_cell_split_readvariableop_resource:	'=
.lstm_lstm_cell_split_1_readvariableop_resource:	:
&lstm_lstm_cell_readvariableop_resource:

identity¢lstm/lstm_cell/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp_1¢lstm/lstm_cell/ReadVariableOp_2¢lstm/lstm_cell/ReadVariableOp_3¢#lstm/lstm_cell/split/ReadVariableOp¢%lstm/lstm_cell/split_1/ReadVariableOp¢
lstm/while@

lstm/ShapeShapeargs_0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeargs_0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ'N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ï
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_mask`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0Ï
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_4MatMullstm/zeros:output:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/MulMullstm/lstm_cell/add:z:0lstm/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_1AddV2lstm/lstm_cell/Mul:z:0lstm/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
$lstm/lstm_cell/clip_by_value/MinimumMinimumlstm/lstm_cell/Add_1:z:0/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm/lstm_cell/clip_by_valueMaximum(lstm/lstm_cell/clip_by_value/Minimum:z:0'lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_5MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_1Mullstm/lstm_cell/add_2:z:0lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_3AddV2lstm/lstm_cell/Mul_1:z:0lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_1/MinimumMinimumlstm/lstm_cell/Add_3:z:01lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_1Maximum*lstm/lstm_cell/clip_by_value_1/Minimum:z:0)lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_2Mul"lstm/lstm_cell/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_6MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_3Mul lstm/lstm_cell/clip_by_value:z:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_5AddV2lstm/lstm_cell/mul_2:z:0lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_7MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_4Mullstm/lstm_cell/add_6:z:0lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_7AddV2lstm/lstm_cell/Mul_4:z:0lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_2/MinimumMinimumlstm/lstm_cell/Add_7:z:01lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_2Maximum*lstm/lstm_cell/clip_by_value_2/Minimum:z:0)lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_5Mul"lstm/lstm_cell/clip_by_value_2:z:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ç
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿY
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_12213*!
condR
lstm_while_cond_12212*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ò
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
IdentityIdentitylstm/strided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(': : : 2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameargs_0
Î
a
E__inference_activation_layer_call_and_return_conditional_losses_14063

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ':O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs
ôJ
¦
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13898

inputs

states
states_10
split_readvariableop_resource:	'.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	'*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ç	
ò
@__inference_dense_layer_call_and_return_conditional_losses_14053

inputs1
matmul_readvariableop_resource:	'-
biasadd_readvariableop_resource:'
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	'*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À	
¢
lstm_while_cond_12212&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_12212___redundant_placeholder0=
9lstm_while_lstm_while_cond_12212___redundant_placeholder1=
9lstm_while_lstm_while_cond_12212___redundant_placeholder2=
9lstm_while_lstm_while_cond_12212___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

F
*__inference_activation_layer_call_fn_14058

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12051`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ':O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs
ßÓ
Å
 __inference__wrapped_model_11759
module_wrapper_inputY
Fsequential_module_wrapper_lstm_lstm_cell_split_readvariableop_resource:	'W
Hsequential_module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource:	T
@sequential_module_wrapper_lstm_lstm_cell_readvariableop_resource:
B
/sequential_dense_matmul_readvariableop_resource:	'>
0sequential_dense_biasadd_readvariableop_resource:'
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢7sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp¢9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_1¢9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_2¢9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_3¢=sequential/module_wrapper/lstm/lstm_cell/split/ReadVariableOp¢?sequential/module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp¢$sequential/module_wrapper/lstm/whileh
$sequential/module_wrapper/lstm/ShapeShapemodule_wrapper_input*
T0*
_output_shapes
:|
2sequential/module_wrapper/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4sequential/module_wrapper/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4sequential/module_wrapper/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,sequential/module_wrapper/lstm/strided_sliceStridedSlice-sequential/module_wrapper/lstm/Shape:output:0;sequential/module_wrapper/lstm/strided_slice/stack:output:0=sequential/module_wrapper/lstm/strided_slice/stack_1:output:0=sequential/module_wrapper/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
-sequential/module_wrapper/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ð
+sequential/module_wrapper/lstm/zeros/packedPack5sequential/module_wrapper/lstm/strided_slice:output:06sequential/module_wrapper/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:o
*sequential/module_wrapper/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ê
$sequential/module_wrapper/lstm/zerosFill4sequential/module_wrapper/lstm/zeros/packed:output:03sequential/module_wrapper/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
/sequential/module_wrapper/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ô
-sequential/module_wrapper/lstm/zeros_1/packedPack5sequential/module_wrapper/lstm/strided_slice:output:08sequential/module_wrapper/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:q
,sequential/module_wrapper/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ð
&sequential/module_wrapper/lstm/zeros_1Fill6sequential/module_wrapper/lstm/zeros_1/packed:output:05sequential/module_wrapper/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-sequential/module_wrapper/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¹
(sequential/module_wrapper/lstm/transpose	Transposemodule_wrapper_input6sequential/module_wrapper/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ'
&sequential/module_wrapper/lstm/Shape_1Shape,sequential/module_wrapper/lstm/transpose:y:0*
T0*
_output_shapes
:~
4sequential/module_wrapper/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6sequential/module_wrapper/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6sequential/module_wrapper/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.sequential/module_wrapper/lstm/strided_slice_1StridedSlice/sequential/module_wrapper/lstm/Shape_1:output:0=sequential/module_wrapper/lstm/strided_slice_1/stack:output:0?sequential/module_wrapper/lstm/strided_slice_1/stack_1:output:0?sequential/module_wrapper/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:sequential/module_wrapper/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,sequential/module_wrapper/lstm/TensorArrayV2TensorListReserveCsequential/module_wrapper/lstm/TensorArrayV2/element_shape:output:07sequential/module_wrapper/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¥
Tsequential/module_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ½
Fsequential/module_wrapper/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor,sequential/module_wrapper/lstm/transpose:y:0]sequential/module_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ~
4sequential/module_wrapper/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6sequential/module_wrapper/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6sequential/module_wrapper/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
.sequential/module_wrapper/lstm/strided_slice_2StridedSlice,sequential/module_wrapper/lstm/transpose:y:0=sequential/module_wrapper/lstm/strided_slice_2/stack:output:0?sequential/module_wrapper/lstm/strided_slice_2/stack_1:output:0?sequential/module_wrapper/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_maskz
8sequential/module_wrapper/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Å
=sequential/module_wrapper/lstm/lstm_cell/split/ReadVariableOpReadVariableOpFsequential_module_wrapper_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0
.sequential/module_wrapper/lstm/lstm_cell/splitSplitAsequential/module_wrapper/lstm/lstm_cell/split/split_dim:output:0Esequential/module_wrapper/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_splitÞ
/sequential/module_wrapper/lstm/lstm_cell/MatMulMatMul7sequential/module_wrapper/lstm/strided_slice_2:output:07sequential/module_wrapper/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1sequential/module_wrapper/lstm/lstm_cell/MatMul_1MatMul7sequential/module_wrapper/lstm/strided_slice_2:output:07sequential/module_wrapper/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1sequential/module_wrapper/lstm/lstm_cell/MatMul_2MatMul7sequential/module_wrapper/lstm/strided_slice_2:output:07sequential/module_wrapper/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1sequential/module_wrapper/lstm/lstm_cell/MatMul_3MatMul7sequential/module_wrapper/lstm/strided_slice_2:output:07sequential/module_wrapper/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
:sequential/module_wrapper/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Å
?sequential/module_wrapper/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOpHsequential_module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0
0sequential/module_wrapper/lstm/lstm_cell/split_1SplitCsequential/module_wrapper/lstm/lstm_cell/split_1/split_dim:output:0Gsequential/module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitä
0sequential/module_wrapper/lstm/lstm_cell/BiasAddBiasAdd9sequential/module_wrapper/lstm/lstm_cell/MatMul:product:09sequential/module_wrapper/lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2sequential/module_wrapper/lstm/lstm_cell/BiasAdd_1BiasAdd;sequential/module_wrapper/lstm/lstm_cell/MatMul_1:product:09sequential/module_wrapper/lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2sequential/module_wrapper/lstm/lstm_cell/BiasAdd_2BiasAdd;sequential/module_wrapper/lstm/lstm_cell/MatMul_2:product:09sequential/module_wrapper/lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2sequential/module_wrapper/lstm/lstm_cell/BiasAdd_3BiasAdd;sequential/module_wrapper/lstm/lstm_cell/MatMul_3:product:09sequential/module_wrapper/lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
7sequential/module_wrapper/lstm/lstm_cell/ReadVariableOpReadVariableOp@sequential_module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
<sequential/module_wrapper/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
>sequential/module_wrapper/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
>sequential/module_wrapper/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      º
6sequential/module_wrapper/lstm/lstm_cell/strided_sliceStridedSlice?sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp:value:0Esequential/module_wrapper/lstm/lstm_cell/strided_slice/stack:output:0Gsequential/module_wrapper/lstm/lstm_cell/strided_slice/stack_1:output:0Gsequential/module_wrapper/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÞ
1sequential/module_wrapper/lstm/lstm_cell/MatMul_4MatMul-sequential/module_wrapper/lstm/zeros:output:0?sequential/module_wrapper/lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
,sequential/module_wrapper/lstm/lstm_cell/addAddV29sequential/module_wrapper/lstm/lstm_cell/BiasAdd:output:0;sequential/module_wrapper/lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.sequential/module_wrapper/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>u
0sequential/module_wrapper/lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ñ
,sequential/module_wrapper/lstm/lstm_cell/MulMul0sequential/module_wrapper/lstm/lstm_cell/add:z:07sequential/module_wrapper/lstm/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
.sequential/module_wrapper/lstm/lstm_cell/Add_1AddV20sequential/module_wrapper/lstm/lstm_cell/Mul:z:09sequential/module_wrapper/lstm/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential/module_wrapper/lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?û
>sequential/module_wrapper/lstm/lstm_cell/clip_by_value/MinimumMinimum2sequential/module_wrapper/lstm/lstm_cell/Add_1:z:0Isequential/module_wrapper/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
8sequential/module_wrapper/lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    û
6sequential/module_wrapper/lstm/lstm_cell/clip_by_valueMaximumBsequential/module_wrapper/lstm/lstm_cell/clip_by_value/Minimum:z:0Asequential/module_wrapper/lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp@sequential_module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
>sequential/module_wrapper/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
@sequential/module_wrapper/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
@sequential/module_wrapper/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ä
8sequential/module_wrapper/lstm/lstm_cell/strided_slice_1StridedSliceAsequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_1:value:0Gsequential/module_wrapper/lstm/lstm_cell/strided_slice_1/stack:output:0Isequential/module_wrapper/lstm/lstm_cell/strided_slice_1/stack_1:output:0Isequential/module_wrapper/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskà
1sequential/module_wrapper/lstm/lstm_cell/MatMul_5MatMul-sequential/module_wrapper/lstm/zeros:output:0Asequential/module_wrapper/lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
.sequential/module_wrapper/lstm/lstm_cell/add_2AddV2;sequential/module_wrapper/lstm/lstm_cell/BiasAdd_1:output:0;sequential/module_wrapper/lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
0sequential/module_wrapper/lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>u
0sequential/module_wrapper/lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?×
.sequential/module_wrapper/lstm/lstm_cell/Mul_1Mul2sequential/module_wrapper/lstm/lstm_cell/add_2:z:09sequential/module_wrapper/lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
.sequential/module_wrapper/lstm/lstm_cell/Add_3AddV22sequential/module_wrapper/lstm/lstm_cell/Mul_1:z:09sequential/module_wrapper/lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bsequential/module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ÿ
@sequential/module_wrapper/lstm/lstm_cell/clip_by_value_1/MinimumMinimum2sequential/module_wrapper/lstm/lstm_cell/Add_3:z:0Ksequential/module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:sequential/module_wrapper/lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
8sequential/module_wrapper/lstm/lstm_cell/clip_by_value_1MaximumDsequential/module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum:z:0Csequential/module_wrapper/lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
.sequential/module_wrapper/lstm/lstm_cell/mul_2Mul<sequential/module_wrapper/lstm/lstm_cell/clip_by_value_1:z:0/sequential/module_wrapper/lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp@sequential_module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
>sequential/module_wrapper/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
@sequential/module_wrapper/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
@sequential/module_wrapper/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ä
8sequential/module_wrapper/lstm/lstm_cell/strided_slice_2StridedSliceAsequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_2:value:0Gsequential/module_wrapper/lstm/lstm_cell/strided_slice_2/stack:output:0Isequential/module_wrapper/lstm/lstm_cell/strided_slice_2/stack_1:output:0Isequential/module_wrapper/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskà
1sequential/module_wrapper/lstm/lstm_cell/MatMul_6MatMul-sequential/module_wrapper/lstm/zeros:output:0Asequential/module_wrapper/lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
.sequential/module_wrapper/lstm/lstm_cell/add_4AddV2;sequential/module_wrapper/lstm/lstm_cell/BiasAdd_2:output:0;sequential/module_wrapper/lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-sequential/module_wrapper/lstm/lstm_cell/TanhTanh2sequential/module_wrapper/lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
.sequential/module_wrapper/lstm/lstm_cell/mul_3Mul:sequential/module_wrapper/lstm/lstm_cell/clip_by_value:z:01sequential/module_wrapper/lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
.sequential/module_wrapper/lstm/lstm_cell/add_5AddV22sequential/module_wrapper/lstm/lstm_cell/mul_2:z:02sequential/module_wrapper/lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp@sequential_module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
>sequential/module_wrapper/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
@sequential/module_wrapper/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
@sequential/module_wrapper/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ä
8sequential/module_wrapper/lstm/lstm_cell/strided_slice_3StridedSliceAsequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_3:value:0Gsequential/module_wrapper/lstm/lstm_cell/strided_slice_3/stack:output:0Isequential/module_wrapper/lstm/lstm_cell/strided_slice_3/stack_1:output:0Isequential/module_wrapper/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskà
1sequential/module_wrapper/lstm/lstm_cell/MatMul_7MatMul-sequential/module_wrapper/lstm/zeros:output:0Asequential/module_wrapper/lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
.sequential/module_wrapper/lstm/lstm_cell/add_6AddV2;sequential/module_wrapper/lstm/lstm_cell/BiasAdd_3:output:0;sequential/module_wrapper/lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
0sequential/module_wrapper/lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>u
0sequential/module_wrapper/lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?×
.sequential/module_wrapper/lstm/lstm_cell/Mul_4Mul2sequential/module_wrapper/lstm/lstm_cell/add_6:z:09sequential/module_wrapper/lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
.sequential/module_wrapper/lstm/lstm_cell/Add_7AddV22sequential/module_wrapper/lstm/lstm_cell/Mul_4:z:09sequential/module_wrapper/lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bsequential/module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ÿ
@sequential/module_wrapper/lstm/lstm_cell/clip_by_value_2/MinimumMinimum2sequential/module_wrapper/lstm/lstm_cell/Add_7:z:0Ksequential/module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:sequential/module_wrapper/lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
8sequential/module_wrapper/lstm/lstm_cell/clip_by_value_2MaximumDsequential/module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum:z:0Csequential/module_wrapper/lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/sequential/module_wrapper/lstm/lstm_cell/Tanh_1Tanh2sequential/module_wrapper/lstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
.sequential/module_wrapper/lstm/lstm_cell/mul_5Mul<sequential/module_wrapper/lstm/lstm_cell/clip_by_value_2:z:03sequential/module_wrapper/lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<sequential/module_wrapper/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
.sequential/module_wrapper/lstm/TensorArrayV2_1TensorListReserveEsequential/module_wrapper/lstm/TensorArrayV2_1/element_shape:output:07sequential/module_wrapper/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
#sequential/module_wrapper/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 
7sequential/module_wrapper/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1sequential/module_wrapper/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : £	
$sequential/module_wrapper/lstm/whileWhile:sequential/module_wrapper/lstm/while/loop_counter:output:0@sequential/module_wrapper/lstm/while/maximum_iterations:output:0,sequential/module_wrapper/lstm/time:output:07sequential/module_wrapper/lstm/TensorArrayV2_1:handle:0-sequential/module_wrapper/lstm/zeros:output:0/sequential/module_wrapper/lstm/zeros_1:output:07sequential/module_wrapper/lstm/strided_slice_1:output:0Vsequential/module_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fsequential_module_wrapper_lstm_lstm_cell_split_readvariableop_resourceHsequential_module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource@sequential_module_wrapper_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *;
body3R1
/sequential_module_wrapper_lstm_while_body_11612*;
cond3R1
/sequential_module_wrapper_lstm_while_cond_11611*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations  
Osequential/module_wrapper/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
Asequential/module_wrapper/lstm/TensorArrayV2Stack/TensorListStackTensorListStack-sequential/module_wrapper/lstm/while:output:3Xsequential/module_wrapper/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ*
element_dtype0
4sequential/module_wrapper/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
6sequential/module_wrapper/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6sequential/module_wrapper/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
.sequential/module_wrapper/lstm/strided_slice_3StridedSliceJsequential/module_wrapper/lstm/TensorArrayV2Stack/TensorListStack:tensor:0=sequential/module_wrapper/lstm/strided_slice_3/stack:output:0?sequential/module_wrapper/lstm/strided_slice_3/stack_1:output:0?sequential/module_wrapper/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
/sequential/module_wrapper/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ô
*sequential/module_wrapper/lstm/transpose_1	TransposeJsequential/module_wrapper/lstm/TensorArrayV2Stack/TensorListStack:tensor:08sequential/module_wrapper/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	'*
dtype0¼
sequential/dense/MatMulMatMul7sequential/module_wrapper/lstm/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'}
sequential/activation/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'v
IdentityIdentity'sequential/activation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'°
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp8^sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp:^sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_1:^sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_2:^sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_3>^sequential/module_wrapper/lstm/lstm_cell/split/ReadVariableOp@^sequential/module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp%^sequential/module_wrapper/lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2r
7sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp7sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp2v
9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_19sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_12v
9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_29sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_22v
9sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_39sequential/module_wrapper/lstm/lstm_cell/ReadVariableOp_32~
=sequential/module_wrapper/lstm/lstm_cell/split/ReadVariableOp=sequential/module_wrapper/lstm/lstm_cell/split/ReadVariableOp2
?sequential/module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp?sequential/module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp2L
$sequential/module_wrapper/lstm/while$sequential/module_wrapper/lstm/while:a ]
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
.
_user_specified_namemodule_wrapper_input
¥
½
$module_wrapper_lstm_while_body_12627D
@module_wrapper_lstm_while_module_wrapper_lstm_while_loop_counterJ
Fmodule_wrapper_lstm_while_module_wrapper_lstm_while_maximum_iterations)
%module_wrapper_lstm_while_placeholder+
'module_wrapper_lstm_while_placeholder_1+
'module_wrapper_lstm_while_placeholder_2+
'module_wrapper_lstm_while_placeholder_3C
?module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1_0
{module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0V
Cmodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0:	'T
Emodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0:	Q
=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0:
&
"module_wrapper_lstm_while_identity(
$module_wrapper_lstm_while_identity_1(
$module_wrapper_lstm_while_identity_2(
$module_wrapper_lstm_while_identity_3(
$module_wrapper_lstm_while_identity_4(
$module_wrapper_lstm_while_identity_5A
=module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1}
ymodule_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensorT
Amodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource:	'R
Cmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource:	O
;module_wrapper_lstm_while_lstm_cell_readvariableop_resource:
¢2module_wrapper/lstm/while/lstm_cell/ReadVariableOp¢4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1¢4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2¢4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3¢8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp¢:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp
Kmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   
=module_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0%module_wrapper_lstm_while_placeholderTmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0u
3module_wrapper/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :½
8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOpCmodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0
)module_wrapper/lstm/while/lstm_cell/splitSplit<module_wrapper/lstm/while/lstm_cell/split/split_dim:output:0@module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_splitá
*module_wrapper/lstm/while/lstm_cell/MatMulMatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
,module_wrapper/lstm/while/lstm_cell/MatMul_1MatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
,module_wrapper/lstm/while/lstm_cell/MatMul_2MatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
,module_wrapper/lstm/while/lstm_cell/MatMul_3MatMulDmodule_wrapper/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02module_wrapper/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5module_wrapper/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ½
:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpEmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0
+module_wrapper/lstm/while/lstm_cell/split_1Split>module_wrapper/lstm/while/lstm_cell/split_1/split_dim:output:0Bmodule_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitÕ
+module_wrapper/lstm/while/lstm_cell/BiasAddBiasAdd4module_wrapper/lstm/while/lstm_cell/MatMul:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
-module_wrapper/lstm/while/lstm_cell/BiasAdd_1BiasAdd6module_wrapper/lstm/while/lstm_cell/MatMul_1:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
-module_wrapper/lstm/while/lstm_cell/BiasAdd_2BiasAdd6module_wrapper/lstm/while/lstm_cell/MatMul_2:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
-module_wrapper/lstm/while/lstm_cell/BiasAdd_3BiasAdd6module_wrapper/lstm/while/lstm_cell/MatMul_3:product:04module_wrapper/lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2module_wrapper/lstm/while/lstm_cell/ReadVariableOpReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
7module_wrapper/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9module_wrapper/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
9module_wrapper/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¡
1module_wrapper/lstm/while/lstm_cell/strided_sliceStridedSlice:module_wrapper/lstm/while/lstm_cell/ReadVariableOp:value:0@module_wrapper/lstm/while/lstm_cell/strided_slice/stack:output:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice/stack_1:output:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÎ
,module_wrapper/lstm/while/lstm_cell/MatMul_4MatMul'module_wrapper_lstm_while_placeholder_2:module_wrapper/lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
'module_wrapper/lstm/while/lstm_cell/addAddV24module_wrapper/lstm/while/lstm_cell/BiasAdd:output:06module_wrapper/lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)module_wrapper/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+module_wrapper/lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Â
'module_wrapper/lstm/while/lstm_cell/MulMul+module_wrapper/lstm/while/lstm_cell/add:z:02module_wrapper/lstm/while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
)module_wrapper/lstm/while/lstm_cell/Add_1AddV2+module_wrapper/lstm/while/lstm_cell/Mul:z:04module_wrapper/lstm/while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;module_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ì
9module_wrapper/lstm/while/lstm_cell/clip_by_value/MinimumMinimum-module_wrapper/lstm/while/lstm_cell/Add_1:z:0Dmodule_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
3module_wrapper/lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ì
1module_wrapper/lstm/while/lstm_cell/clip_by_valueMaximum=module_wrapper/lstm/while/lstm_cell/clip_by_value/Minimum:z:0<module_wrapper/lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9module_wrapper/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
;module_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      «
3module_wrapper/lstm/while/lstm_cell/strided_slice_1StridedSlice<module_wrapper/lstm/while/lstm_cell/ReadVariableOp_1:value:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice_1/stack:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÐ
,module_wrapper/lstm/while/lstm_cell/MatMul_5MatMul'module_wrapper_lstm_while_placeholder_2<module_wrapper/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
)module_wrapper/lstm/while/lstm_cell/add_2AddV26module_wrapper/lstm/while/lstm_cell/BiasAdd_1:output:06module_wrapper/lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+module_wrapper/lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+module_wrapper/lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?È
)module_wrapper/lstm/while/lstm_cell/Mul_1Mul-module_wrapper/lstm/while/lstm_cell/add_2:z:04module_wrapper/lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
)module_wrapper/lstm/while/lstm_cell/Add_3AddV2-module_wrapper/lstm/while/lstm_cell/Mul_1:z:04module_wrapper/lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=module_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ð
;module_wrapper/lstm/while/lstm_cell/clip_by_value_1/MinimumMinimum-module_wrapper/lstm/while/lstm_cell/Add_3:z:0Fmodule_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5module_wrapper/lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ò
3module_wrapper/lstm/while/lstm_cell/clip_by_value_1Maximum?module_wrapper/lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0>module_wrapper/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
)module_wrapper/lstm/while/lstm_cell/mul_2Mul7module_wrapper/lstm/while/lstm_cell/clip_by_value_1:z:0'module_wrapper_lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9module_wrapper/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
;module_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      «
3module_wrapper/lstm/while/lstm_cell/strided_slice_2StridedSlice<module_wrapper/lstm/while/lstm_cell/ReadVariableOp_2:value:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice_2/stack:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÐ
,module_wrapper/lstm/while/lstm_cell/MatMul_6MatMul'module_wrapper_lstm_while_placeholder_2<module_wrapper/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
)module_wrapper/lstm/while/lstm_cell/add_4AddV26module_wrapper/lstm/while/lstm_cell/BiasAdd_2:output:06module_wrapper/lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(module_wrapper/lstm/while/lstm_cell/TanhTanh-module_wrapper/lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
)module_wrapper/lstm/while/lstm_cell/mul_3Mul5module_wrapper/lstm/while/lstm_cell/clip_by_value:z:0,module_wrapper/lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
)module_wrapper/lstm/while/lstm_cell/add_5AddV2-module_wrapper/lstm/while/lstm_cell/mul_2:z:0-module_wrapper/lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9module_wrapper/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
;module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
;module_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      «
3module_wrapper/lstm/while/lstm_cell/strided_slice_3StridedSlice<module_wrapper/lstm/while/lstm_cell/ReadVariableOp_3:value:0Bmodule_wrapper/lstm/while/lstm_cell/strided_slice_3/stack:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0Dmodule_wrapper/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÐ
,module_wrapper/lstm/while/lstm_cell/MatMul_7MatMul'module_wrapper_lstm_while_placeholder_2<module_wrapper/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
)module_wrapper/lstm/while/lstm_cell/add_6AddV26module_wrapper/lstm/while/lstm_cell/BiasAdd_3:output:06module_wrapper/lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+module_wrapper/lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+module_wrapper/lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?È
)module_wrapper/lstm/while/lstm_cell/Mul_4Mul-module_wrapper/lstm/while/lstm_cell/add_6:z:04module_wrapper/lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
)module_wrapper/lstm/while/lstm_cell/Add_7AddV2-module_wrapper/lstm/while/lstm_cell/Mul_4:z:04module_wrapper/lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=module_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ð
;module_wrapper/lstm/while/lstm_cell/clip_by_value_2/MinimumMinimum-module_wrapper/lstm/while/lstm_cell/Add_7:z:0Fmodule_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5module_wrapper/lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ò
3module_wrapper/lstm/while/lstm_cell/clip_by_value_2Maximum?module_wrapper/lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0>module_wrapper/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*module_wrapper/lstm/while/lstm_cell/Tanh_1Tanh-module_wrapper/lstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)module_wrapper/lstm/while/lstm_cell/mul_5Mul7module_wrapper/lstm/while/lstm_cell/clip_by_value_2:z:0.module_wrapper/lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>module_wrapper/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'module_wrapper_lstm_while_placeholder_1%module_wrapper_lstm_while_placeholder-module_wrapper/lstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒa
module_wrapper/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
module_wrapper/lstm/while/addAddV2%module_wrapper_lstm_while_placeholder(module_wrapper/lstm/while/add/y:output:0*
T0*
_output_shapes
: c
!module_wrapper/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
module_wrapper/lstm/while/add_1AddV2@module_wrapper_lstm_while_module_wrapper_lstm_while_loop_counter*module_wrapper/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
"module_wrapper/lstm/while/IdentityIdentity#module_wrapper/lstm/while/add_1:z:0^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: º
$module_wrapper/lstm/while/Identity_1IdentityFmodule_wrapper_lstm_while_module_wrapper_lstm_while_maximum_iterations^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: 
$module_wrapper/lstm/while/Identity_2Identity!module_wrapper/lstm/while/add:z:0^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: Â
$module_wrapper/lstm/while/Identity_3IdentityNmodule_wrapper/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^module_wrapper/lstm/while/NoOp*
T0*
_output_shapes
: ³
$module_wrapper/lstm/while/Identity_4Identity-module_wrapper/lstm/while/lstm_cell/mul_5:z:0^module_wrapper/lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
$module_wrapper/lstm/while/Identity_5Identity-module_wrapper/lstm/while/lstm_cell/add_5:z:0^module_wrapper/lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
module_wrapper/lstm/while/NoOpNoOp3^module_wrapper/lstm/while/lstm_cell/ReadVariableOp5^module_wrapper/lstm/while/lstm_cell/ReadVariableOp_15^module_wrapper/lstm/while/lstm_cell/ReadVariableOp_25^module_wrapper/lstm/while/lstm_cell/ReadVariableOp_39^module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp;^module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"module_wrapper_lstm_while_identity+module_wrapper/lstm/while/Identity:output:0"U
$module_wrapper_lstm_while_identity_1-module_wrapper/lstm/while/Identity_1:output:0"U
$module_wrapper_lstm_while_identity_2-module_wrapper/lstm/while/Identity_2:output:0"U
$module_wrapper_lstm_while_identity_3-module_wrapper/lstm/while/Identity_3:output:0"U
$module_wrapper_lstm_while_identity_4-module_wrapper/lstm/while/Identity_4:output:0"U
$module_wrapper_lstm_while_identity_5-module_wrapper/lstm/while/Identity_5:output:0"|
;module_wrapper_lstm_while_lstm_cell_readvariableop_resource=module_wrapper_lstm_while_lstm_cell_readvariableop_resource_0"
Cmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resourceEmodule_wrapper_lstm_while_lstm_cell_split_1_readvariableop_resource_0"
Amodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resourceCmodule_wrapper_lstm_while_lstm_cell_split_readvariableop_resource_0"
=module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1?module_wrapper_lstm_while_module_wrapper_lstm_strided_slice_1_0"ø
ymodule_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor{module_wrapper_lstm_while_tensorarrayv2read_tensorlistgetitem_module_wrapper_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2h
2module_wrapper/lstm/while/lstm_cell/ReadVariableOp2module_wrapper/lstm/while/lstm_cell/ReadVariableOp2l
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_14module_wrapper/lstm/while/lstm_cell/ReadVariableOp_12l
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_24module_wrapper/lstm/while/lstm_cell/ReadVariableOp_22l
4module_wrapper/lstm/while/lstm_cell/ReadVariableOp_34module_wrapper/lstm/while/lstm_cell/ReadVariableOp_32t
8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp8module_wrapper/lstm/while/lstm_cell/split/ReadVariableOp2x
:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp:module_wrapper/lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
´
¾
while_cond_14456
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_14456___redundant_placeholder03
/while_while_cond_14456___redundant_placeholder13
/while_while_cond_14456___redundant_placeholder23
/while_while_cond_14456___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ï
ö
#__inference_signature_wrapper_12481
module_wrapper_input
unknown:	'
	unknown_0:	
	unknown_1:

	unknown_2:	'
	unknown_3:'
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_11759o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
.
_user_specified_namemodule_wrapper_input
å
Î
$module_wrapper_lstm_while_cond_12889D
@module_wrapper_lstm_while_module_wrapper_lstm_while_loop_counterJ
Fmodule_wrapper_lstm_while_module_wrapper_lstm_while_maximum_iterations)
%module_wrapper_lstm_while_placeholder+
'module_wrapper_lstm_while_placeholder_1+
'module_wrapper_lstm_while_placeholder_2+
'module_wrapper_lstm_while_placeholder_3F
Bmodule_wrapper_lstm_while_less_module_wrapper_lstm_strided_slice_1[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12889___redundant_placeholder0[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12889___redundant_placeholder1[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12889___redundant_placeholder2[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12889___redundant_placeholder3&
"module_wrapper_lstm_while_identity
²
module_wrapper/lstm/while/LessLess%module_wrapper_lstm_while_placeholderBmodule_wrapper_lstm_while_less_module_wrapper_lstm_strided_slice_1*
T0*
_output_shapes
: s
"module_wrapper/lstm/while/IdentityIdentity"module_wrapper/lstm/while/Less:z:0*
T0
*
_output_shapes
: "Q
"module_wrapper_lstm_while_identity+module_wrapper/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
û
ý
*__inference_sequential_layer_call_fn_12067
module_wrapper_input
unknown:	'
	unknown_0:	
	unknown_1:

	unknown_2:	'
	unknown_3:'
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12054o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
.
_user_specified_namemodule_wrapper_input
î
ó
)__inference_lstm_cell_layer_call_fn_14614

inputs
states_0
states_1
unknown:	'
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13696p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ç	
ò
@__inference_dense_layer_call_and_return_conditional_losses_12040

inputs1
matmul_readvariableop_resource:	'-
biasadd_readvariableop_resource:'
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	'*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
¾
while_cond_13956
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_13956___redundant_placeholder03
/while_while_cond_13956___redundant_placeholder13
/while_while_cond_13956___redundant_placeholder23
/while_while_cond_13956___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
è
°

lstm_while_body_11882&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	'E
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	B
.lstm_while_lstm_cell_readvariableop_resource_0:

lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	'C
4lstm_while_lstm_cell_split_1_readvariableop_resource:	@
,lstm_while_lstm_cell_readvariableop_resource:
¢#lstm/while/lstm_cell/ReadVariableOp¢%lstm/while/lstm_cell/ReadVariableOp_1¢%lstm/while/lstm_cell/ReadVariableOp_2¢%lstm/while/lstm_cell/ReadVariableOp_3¢)lstm/while/lstm_cell/split/ReadVariableOp¢+lstm/while/lstm_cell/split_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¿
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0á
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split´
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0×
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split¨
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¡
lstm/while/lstm_cell/MatMul_4MatMullstm_while_placeholder_2+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/MulMullstm/while/lstm_cell/add:z:0#lstm/while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_1AddV2lstm/while/lstm_cell/Mul:z:0%lstm/while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¿
*lstm/while/lstm_cell/clip_by_value/MinimumMinimumlstm/while/lstm_cell/Add_1:z:05lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¿
"lstm/while/lstm_cell/clip_by_valueMaximum.lstm/while/lstm_cell/clip_by_value/Minimum:z:0-lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_5MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_1Mullstm/while/lstm_cell/add_2:z:0%lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_3AddV2lstm/while/lstm_cell/Mul_1:z:0%lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_1/MinimumMinimumlstm/while/lstm_cell/Add_3:z:07lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_1Maximum0lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_2Mul(lstm/while/lstm_cell/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_6MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_3Mul&lstm/while/lstm_cell/clip_by_value:z:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/add_5AddV2lstm/while/lstm_cell/mul_2:z:0lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_7MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_6AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_4Mullstm/while/lstm_cell/add_6:z:0%lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_7AddV2lstm/while/lstm_cell/Mul_4:z:0%lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_2/MinimumMinimumlstm/while/lstm_cell/Add_7:z:07lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_2Maximum0lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_5Mul(lstm/while/lstm_cell/clip_by_value_2:z:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
«

I__inference_module_wrapper_layer_call_and_return_conditional_losses_13571

args_0?
,lstm_lstm_cell_split_readvariableop_resource:	'=
.lstm_lstm_cell_split_1_readvariableop_resource:	:
&lstm_lstm_cell_readvariableop_resource:

identity¢lstm/lstm_cell/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp_1¢lstm/lstm_cell/ReadVariableOp_2¢lstm/lstm_cell/ReadVariableOp_3¢#lstm/lstm_cell/split/ReadVariableOp¢%lstm/lstm_cell/split_1/ReadVariableOp¢
lstm/while@

lstm/ShapeShapeargs_0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeargs_0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ'N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ï
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_mask`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0Ï
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_4MatMullstm/zeros:output:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/MulMullstm/lstm_cell/add:z:0lstm/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_1AddV2lstm/lstm_cell/Mul:z:0lstm/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
$lstm/lstm_cell/clip_by_value/MinimumMinimumlstm/lstm_cell/Add_1:z:0/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm/lstm_cell/clip_by_valueMaximum(lstm/lstm_cell/clip_by_value/Minimum:z:0'lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_5MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_1Mullstm/lstm_cell/add_2:z:0lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_3AddV2lstm/lstm_cell/Mul_1:z:0lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_1/MinimumMinimumlstm/lstm_cell/Add_3:z:01lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_1Maximum*lstm/lstm_cell/clip_by_value_1/Minimum:z:0)lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_2Mul"lstm/lstm_cell/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_6MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_3Mul lstm/lstm_cell/clip_by_value:z:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_5AddV2lstm/lstm_cell/mul_2:z:0lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_7MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_4Mullstm/lstm_cell/add_6:z:0lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_7AddV2lstm/lstm_cell/Mul_4:z:0lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_2/MinimumMinimumlstm/lstm_cell/Add_7:z:01lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_2Maximum*lstm/lstm_cell/clip_by_value_2/Minimum:z:0)lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_5Mul"lstm/lstm_cell/clip_by_value_2:z:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ç
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿY
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_13431*!
condR
lstm_while_cond_13430*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ò
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
IdentityIdentitylstm/strided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(': : : 2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameargs_0

´
$__inference_lstm_layer_call_fn_14074
inputs_0
unknown:	'
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_13778p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ': : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'
"
_user_specified_name
inputs/0
ìG
³
!__inference__traced_restore_14944
file_prefix0
assignvariableop_dense_kernel:	'+
assignvariableop_1_dense_bias:'J
7assignvariableop_2_module_wrapper_lstm_lstm_cell_kernel:	'U
Aassignvariableop_3_module_wrapper_lstm_lstm_cell_recurrent_kernel:
D
5assignvariableop_4_module_wrapper_lstm_lstm_cell_bias:	)
assignvariableop_5_rmsprop_iter:	 *
 assignvariableop_6_rmsprop_decay: 2
(assignvariableop_7_rmsprop_learning_rate: -
#assignvariableop_8_rmsprop_momentum: (
assignvariableop_9_rmsprop_rho: #
assignvariableop_10_total: #
assignvariableop_11_count: ?
,assignvariableop_12_rmsprop_dense_kernel_rms:	'8
*assignvariableop_13_rmsprop_dense_bias_rms:'W
Dassignvariableop_14_rmsprop_module_wrapper_lstm_lstm_cell_kernel_rms:	'b
Nassignvariableop_15_rmsprop_module_wrapper_lstm_lstm_cell_recurrent_kernel_rms:
Q
Bassignvariableop_16_rmsprop_module_wrapper_lstm_lstm_cell_bias_rms:	
identity_18¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¿
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*å
valueÛBØB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B ø
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_2AssignVariableOp7assignvariableop_2_module_wrapper_lstm_lstm_cell_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_3AssignVariableOpAassignvariableop_3_module_wrapper_lstm_lstm_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_4AssignVariableOp5assignvariableop_4_module_wrapper_lstm_lstm_cell_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_rmsprop_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_rmsprop_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp(assignvariableop_7_rmsprop_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_rmsprop_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_rmsprop_rhoIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp,assignvariableop_12_rmsprop_dense_kernel_rmsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp*assignvariableop_13_rmsprop_dense_bias_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_14AssignVariableOpDassignvariableop_14_rmsprop_module_wrapper_lstm_lstm_cell_kernel_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_15AssignVariableOpNassignvariableop_15_rmsprop_module_wrapper_lstm_lstm_cell_recurrent_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_16AssignVariableOpBassignvariableop_16_rmsprop_module_wrapper_lstm_lstm_cell_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Å
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: ²
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_18Identity_18:output:0*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
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

¼
.__inference_module_wrapper_layer_call_fn_13059

args_0
unknown:	'
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_12353p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(': : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameargs_0
À	
¢
lstm_while_cond_11881&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_11881___redundant_placeholder0=
9lstm_while_lstm_while_cond_11881___redundant_placeholder1=
9lstm_while_lstm_while_cond_11881___redundant_placeholder2=
9lstm_while_lstm_while_cond_11881___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
´
¾
while_cond_14200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_14200___redundant_placeholder03
/while_while_cond_14200___redundant_placeholder13
/while_while_cond_14200___redundant_placeholder23
/while_while_cond_14200___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
7
ó
?__inference_lstm_layer_call_and_return_conditional_losses_14025

inputs"
lstm_cell_13944:	'
lstm_cell_13946:	#
lstm_cell_13948:

identity¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_maskå
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13944lstm_cell_13946lstm_cell_13948*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13898n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13944lstm_cell_13946lstm_cell_13948*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_13957*
condR
while_cond_13956*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ': : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs
Û
à
E__inference_sequential_layer_call_and_return_conditional_losses_12396

inputs'
module_wrapper_12382:	'#
module_wrapper_12384:	(
module_wrapper_12386:

dense_12389:	'
dense_12391:'
identity¢dense/StatefulPartitionedCall¢&module_wrapper/StatefulPartitionedCall
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_12382module_wrapper_12384module_wrapper_12386*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_12353
dense/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0dense_12389dense_12391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_12040Û
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12051r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
NoOpNoOp^dense/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameinputs
Î
a
E__inference_activation_layer_call_and_return_conditional_losses_12051

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ':O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs
«

I__inference_module_wrapper_layer_call_and_return_conditional_losses_12022

args_0?
,lstm_lstm_cell_split_readvariableop_resource:	'=
.lstm_lstm_cell_split_1_readvariableop_resource:	:
&lstm_lstm_cell_readvariableop_resource:

identity¢lstm/lstm_cell/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp_1¢lstm/lstm_cell/ReadVariableOp_2¢lstm/lstm_cell/ReadVariableOp_3¢#lstm/lstm_cell/split/ReadVariableOp¢%lstm/lstm_cell/split_1/ReadVariableOp¢
lstm/while@

lstm/ShapeShapeargs_0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeargs_0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ'N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ï
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_mask`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0Ï
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_4MatMullstm/zeros:output:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/MulMullstm/lstm_cell/add:z:0lstm/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_1AddV2lstm/lstm_cell/Mul:z:0lstm/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
$lstm/lstm_cell/clip_by_value/MinimumMinimumlstm/lstm_cell/Add_1:z:0/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm/lstm_cell/clip_by_valueMaximum(lstm/lstm_cell/clip_by_value/Minimum:z:0'lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_5MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_1Mullstm/lstm_cell/add_2:z:0lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_3AddV2lstm/lstm_cell/Mul_1:z:0lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_1/MinimumMinimumlstm/lstm_cell/Add_3:z:01lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_1Maximum*lstm/lstm_cell/clip_by_value_1/Minimum:z:0)lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_2Mul"lstm/lstm_cell/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_6MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_3Mul lstm/lstm_cell/clip_by_value:z:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_5AddV2lstm/lstm_cell/mul_2:z:0lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_7MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>[
lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/Mul_4Mullstm/lstm_cell/add_6:z:0lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/Add_7AddV2lstm/lstm_cell/Mul_4:z:0lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
&lstm/lstm_cell/clip_by_value_2/MinimumMinimumlstm/lstm_cell/Add_7:z:01lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
lstm/lstm_cell/clip_by_value_2Maximum*lstm/lstm_cell/clip_by_value_2/Minimum:z:0)lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/lstm_cell/mul_5Mul"lstm/lstm_cell/clip_by_value_2:z:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ç
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿY
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_11882*!
condR
lstm_while_cond_11881*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ò
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
IdentityIdentitylstm/strided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(': : : 2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameargs_0

¼
.__inference_module_wrapper_layer_call_fn_13048

args_0
unknown:	'
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_12022p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(': : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameargs_0
"
É
while_body_13957
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_13981_0:	'&
while_lstm_cell_13983_0:	+
while_lstm_cell_13985_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_13981:	'$
while_lstm_cell_13983:	)
while_lstm_cell_13985:
¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0£
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13981_0while_lstm_cell_13983_0while_lstm_cell_13985_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13898Ù
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_13981while_lstm_cell_13981_0"0
while_lstm_cell_13983while_lstm_cell_13983_0"0
while_lstm_cell_13985while_lstm_cell_13985_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
è
°

lstm_while_body_13175&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	'E
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	B
.lstm_while_lstm_cell_readvariableop_resource_0:

lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	'C
4lstm_while_lstm_cell_split_1_readvariableop_resource:	@
,lstm_while_lstm_cell_readvariableop_resource:
¢#lstm/while/lstm_cell/ReadVariableOp¢%lstm/while/lstm_cell/ReadVariableOp_1¢%lstm/while/lstm_cell/ReadVariableOp_2¢%lstm/while/lstm_cell/ReadVariableOp_3¢)lstm/while/lstm_cell/split/ReadVariableOp¢+lstm/while/lstm_cell/split_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¿
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0á
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split´
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0×
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split¨
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¡
lstm/while/lstm_cell/MatMul_4MatMullstm_while_placeholder_2+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/MulMullstm/while/lstm_cell/add:z:0#lstm/while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_1AddV2lstm/while/lstm_cell/Mul:z:0%lstm/while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¿
*lstm/while/lstm_cell/clip_by_value/MinimumMinimumlstm/while/lstm_cell/Add_1:z:05lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¿
"lstm/while/lstm_cell/clip_by_valueMaximum.lstm/while/lstm_cell/clip_by_value/Minimum:z:0-lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_5MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_1Mullstm/while/lstm_cell/add_2:z:0%lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_3AddV2lstm/while/lstm_cell/Mul_1:z:0%lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_1/MinimumMinimumlstm/while/lstm_cell/Add_3:z:07lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_1Maximum0lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_2Mul(lstm/while/lstm_cell/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_6MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_3Mul&lstm/while/lstm_cell/clip_by_value:z:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/add_5AddV2lstm/while/lstm_cell/mul_2:z:0lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_7MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_6AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_4Mullstm/while/lstm_cell/add_6:z:0%lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_7AddV2lstm/while/lstm_cell/Mul_4:z:0%lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_2/MinimumMinimumlstm/while/lstm_cell/Add_7:z:07lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_2Maximum0lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_5Mul(lstm/while/lstm_cell/clip_by_value_2:z:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

ª
/sequential_module_wrapper_lstm_while_cond_11611Z
Vsequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_loop_counter`
\sequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_maximum_iterations4
0sequential_module_wrapper_lstm_while_placeholder6
2sequential_module_wrapper_lstm_while_placeholder_16
2sequential_module_wrapper_lstm_while_placeholder_26
2sequential_module_wrapper_lstm_while_placeholder_3\
Xsequential_module_wrapper_lstm_while_less_sequential_module_wrapper_lstm_strided_slice_1q
msequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_cond_11611___redundant_placeholder0q
msequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_cond_11611___redundant_placeholder1q
msequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_cond_11611___redundant_placeholder2q
msequential_module_wrapper_lstm_while_sequential_module_wrapper_lstm_while_cond_11611___redundant_placeholder31
-sequential_module_wrapper_lstm_while_identity
Þ
)sequential/module_wrapper/lstm/while/LessLess0sequential_module_wrapper_lstm_while_placeholderXsequential_module_wrapper_lstm_while_less_sequential_module_wrapper_lstm_strided_slice_1*
T0*
_output_shapes
: 
-sequential/module_wrapper/lstm/while/IdentityIdentity-sequential/module_wrapper/lstm/while/Less:z:0*
T0
*
_output_shapes
: "g
-sequential_module_wrapper_lstm_while_identity6sequential/module_wrapper/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ô¸
Â
E__inference_sequential_layer_call_and_return_conditional_losses_12774

inputsN
;module_wrapper_lstm_lstm_cell_split_readvariableop_resource:	'L
=module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource:	I
5module_wrapper_lstm_lstm_cell_readvariableop_resource:
7
$dense_matmul_readvariableop_resource:	'3
%dense_biasadd_readvariableop_resource:'
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢,module_wrapper/lstm/lstm_cell/ReadVariableOp¢.module_wrapper/lstm/lstm_cell/ReadVariableOp_1¢.module_wrapper/lstm/lstm_cell/ReadVariableOp_2¢.module_wrapper/lstm/lstm_cell/ReadVariableOp_3¢2module_wrapper/lstm/lstm_cell/split/ReadVariableOp¢4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp¢module_wrapper/lstm/whileO
module_wrapper/lstm/ShapeShapeinputs*
T0*
_output_shapes
:q
'module_wrapper/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)module_wrapper/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)module_wrapper/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!module_wrapper/lstm/strided_sliceStridedSlice"module_wrapper/lstm/Shape:output:00module_wrapper/lstm/strided_slice/stack:output:02module_wrapper/lstm/strided_slice/stack_1:output:02module_wrapper/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"module_wrapper/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¯
 module_wrapper/lstm/zeros/packedPack*module_wrapper/lstm/strided_slice:output:0+module_wrapper/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
module_wrapper/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ©
module_wrapper/lstm/zerosFill)module_wrapper/lstm/zeros/packed:output:0(module_wrapper/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$module_wrapper/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :³
"module_wrapper/lstm/zeros_1/packedPack*module_wrapper/lstm/strided_slice:output:0-module_wrapper/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!module_wrapper/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
module_wrapper/lstm/zeros_1Fill+module_wrapper/lstm/zeros_1/packed:output:0*module_wrapper/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"module_wrapper/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
module_wrapper/lstm/transpose	Transposeinputs+module_wrapper/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ'l
module_wrapper/lstm/Shape_1Shape!module_wrapper/lstm/transpose:y:0*
T0*
_output_shapes
:s
)module_wrapper/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+module_wrapper/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+module_wrapper/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#module_wrapper/lstm/strided_slice_1StridedSlice$module_wrapper/lstm/Shape_1:output:02module_wrapper/lstm/strided_slice_1/stack:output:04module_wrapper/lstm/strided_slice_1/stack_1:output:04module_wrapper/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/module_wrapper/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!module_wrapper/lstm/TensorArrayV2TensorListReserve8module_wrapper/lstm/TensorArrayV2/element_shape:output:0,module_wrapper/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Imodule_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   
;module_wrapper/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!module_wrapper/lstm/transpose:y:0Rmodule_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)module_wrapper/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+module_wrapper/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+module_wrapper/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#module_wrapper/lstm/strided_slice_2StridedSlice!module_wrapper/lstm/transpose:y:02module_wrapper/lstm/strided_slice_2/stack:output:04module_wrapper/lstm/strided_slice_2/stack_1:output:04module_wrapper/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_masko
-module_wrapper/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¯
2module_wrapper/lstm/lstm_cell/split/ReadVariableOpReadVariableOp;module_wrapper_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0ü
#module_wrapper/lstm/lstm_cell/splitSplit6module_wrapper/lstm/lstm_cell/split/split_dim:output:0:module_wrapper/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split½
$module_wrapper/lstm/lstm_cell/MatMulMatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
&module_wrapper/lstm/lstm_cell/MatMul_1MatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
&module_wrapper/lstm/lstm_cell/MatMul_2MatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
&module_wrapper/lstm/lstm_cell/MatMul_3MatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/module_wrapper/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ¯
4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp=module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ò
%module_wrapper/lstm/lstm_cell/split_1Split8module_wrapper/lstm/lstm_cell/split_1/split_dim:output:0<module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitÃ
%module_wrapper/lstm/lstm_cell/BiasAddBiasAdd.module_wrapper/lstm/lstm_cell/MatMul:product:0.module_wrapper/lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
'module_wrapper/lstm/lstm_cell/BiasAdd_1BiasAdd0module_wrapper/lstm/lstm_cell/MatMul_1:product:0.module_wrapper/lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
'module_wrapper/lstm/lstm_cell/BiasAdd_2BiasAdd0module_wrapper/lstm/lstm_cell/MatMul_2:product:0.module_wrapper/lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
'module_wrapper/lstm/lstm_cell/BiasAdd_3BiasAdd0module_wrapper/lstm/lstm_cell/MatMul_3:product:0.module_wrapper/lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,module_wrapper/lstm/lstm_cell/ReadVariableOpReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
1module_wrapper/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
3module_wrapper/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
3module_wrapper/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
+module_wrapper/lstm/lstm_cell/strided_sliceStridedSlice4module_wrapper/lstm/lstm_cell/ReadVariableOp:value:0:module_wrapper/lstm/lstm_cell/strided_slice/stack:output:0<module_wrapper/lstm/lstm_cell/strided_slice/stack_1:output:0<module_wrapper/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask½
&module_wrapper/lstm/lstm_cell/MatMul_4MatMul"module_wrapper/lstm/zeros:output:04module_wrapper/lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
!module_wrapper/lstm/lstm_cell/addAddV2.module_wrapper/lstm/lstm_cell/BiasAdd:output:00module_wrapper/lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#module_wrapper/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%module_wrapper/lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?°
!module_wrapper/lstm/lstm_cell/MulMul%module_wrapper/lstm/lstm_cell/add:z:0,module_wrapper/lstm/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
#module_wrapper/lstm/lstm_cell/Add_1AddV2%module_wrapper/lstm/lstm_cell/Mul:z:0.module_wrapper/lstm/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5module_wrapper/lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ú
3module_wrapper/lstm/lstm_cell/clip_by_value/MinimumMinimum'module_wrapper/lstm/lstm_cell/Add_1:z:0>module_wrapper/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
-module_wrapper/lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ú
+module_wrapper/lstm/lstm_cell/clip_by_valueMaximum7module_wrapper/lstm/lstm_cell/clip_by_value/Minimum:z:06module_wrapper/lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
3module_wrapper/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5module_wrapper/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5module_wrapper/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-module_wrapper/lstm/lstm_cell/strided_slice_1StridedSlice6module_wrapper/lstm/lstm_cell/ReadVariableOp_1:value:0<module_wrapper/lstm/lstm_cell/strided_slice_1/stack:output:0>module_wrapper/lstm/lstm_cell/strided_slice_1/stack_1:output:0>module_wrapper/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¿
&module_wrapper/lstm/lstm_cell/MatMul_5MatMul"module_wrapper/lstm/zeros:output:06module_wrapper/lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
#module_wrapper/lstm/lstm_cell/add_2AddV20module_wrapper/lstm/lstm_cell/BiasAdd_1:output:00module_wrapper/lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%module_wrapper/lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%module_wrapper/lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
#module_wrapper/lstm/lstm_cell/Mul_1Mul'module_wrapper/lstm/lstm_cell/add_2:z:0.module_wrapper/lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
#module_wrapper/lstm/lstm_cell/Add_3AddV2'module_wrapper/lstm/lstm_cell/Mul_1:z:0.module_wrapper/lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Þ
5module_wrapper/lstm/lstm_cell/clip_by_value_1/MinimumMinimum'module_wrapper/lstm/lstm_cell/Add_3:z:0@module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
/module_wrapper/lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    à
-module_wrapper/lstm/lstm_cell/clip_by_value_1Maximum9module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum:z:08module_wrapper/lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
#module_wrapper/lstm/lstm_cell/mul_2Mul1module_wrapper/lstm/lstm_cell/clip_by_value_1:z:0$module_wrapper/lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
3module_wrapper/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5module_wrapper/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
5module_wrapper/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-module_wrapper/lstm/lstm_cell/strided_slice_2StridedSlice6module_wrapper/lstm/lstm_cell/ReadVariableOp_2:value:0<module_wrapper/lstm/lstm_cell/strided_slice_2/stack:output:0>module_wrapper/lstm/lstm_cell/strided_slice_2/stack_1:output:0>module_wrapper/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¿
&module_wrapper/lstm/lstm_cell/MatMul_6MatMul"module_wrapper/lstm/zeros:output:06module_wrapper/lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
#module_wrapper/lstm/lstm_cell/add_4AddV20module_wrapper/lstm/lstm_cell/BiasAdd_2:output:00module_wrapper/lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"module_wrapper/lstm/lstm_cell/TanhTanh'module_wrapper/lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
#module_wrapper/lstm/lstm_cell/mul_3Mul/module_wrapper/lstm/lstm_cell/clip_by_value:z:0&module_wrapper/lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
#module_wrapper/lstm/lstm_cell/add_5AddV2'module_wrapper/lstm/lstm_cell/mul_2:z:0'module_wrapper/lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
3module_wrapper/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
5module_wrapper/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
5module_wrapper/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-module_wrapper/lstm/lstm_cell/strided_slice_3StridedSlice6module_wrapper/lstm/lstm_cell/ReadVariableOp_3:value:0<module_wrapper/lstm/lstm_cell/strided_slice_3/stack:output:0>module_wrapper/lstm/lstm_cell/strided_slice_3/stack_1:output:0>module_wrapper/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¿
&module_wrapper/lstm/lstm_cell/MatMul_7MatMul"module_wrapper/lstm/zeros:output:06module_wrapper/lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
#module_wrapper/lstm/lstm_cell/add_6AddV20module_wrapper/lstm/lstm_cell/BiasAdd_3:output:00module_wrapper/lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%module_wrapper/lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%module_wrapper/lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
#module_wrapper/lstm/lstm_cell/Mul_4Mul'module_wrapper/lstm/lstm_cell/add_6:z:0.module_wrapper/lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
#module_wrapper/lstm/lstm_cell/Add_7AddV2'module_wrapper/lstm/lstm_cell/Mul_4:z:0.module_wrapper/lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Þ
5module_wrapper/lstm/lstm_cell/clip_by_value_2/MinimumMinimum'module_wrapper/lstm/lstm_cell/Add_7:z:0@module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
/module_wrapper/lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    à
-module_wrapper/lstm/lstm_cell/clip_by_value_2Maximum9module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum:z:08module_wrapper/lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$module_wrapper/lstm/lstm_cell/Tanh_1Tanh'module_wrapper/lstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#module_wrapper/lstm/lstm_cell/mul_5Mul1module_wrapper/lstm/lstm_cell/clip_by_value_2:z:0(module_wrapper/lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1module_wrapper/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
#module_wrapper/lstm/TensorArrayV2_1TensorListReserve:module_wrapper/lstm/TensorArrayV2_1/element_shape:output:0,module_wrapper/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
module_wrapper/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,module_wrapper/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&module_wrapper/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
module_wrapper/lstm/whileWhile/module_wrapper/lstm/while/loop_counter:output:05module_wrapper/lstm/while/maximum_iterations:output:0!module_wrapper/lstm/time:output:0,module_wrapper/lstm/TensorArrayV2_1:handle:0"module_wrapper/lstm/zeros:output:0$module_wrapper/lstm/zeros_1:output:0,module_wrapper/lstm/strided_slice_1:output:0Kmodule_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0;module_wrapper_lstm_lstm_cell_split_readvariableop_resource=module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource5module_wrapper_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$module_wrapper_lstm_while_body_12627*0
cond(R&
$module_wrapper_lstm_while_cond_12626*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Dmodule_wrapper/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ÿ
6module_wrapper/lstm/TensorArrayV2Stack/TensorListStackTensorListStack"module_wrapper/lstm/while:output:3Mmodule_wrapper/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ*
element_dtype0|
)module_wrapper/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+module_wrapper/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+module_wrapper/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
#module_wrapper/lstm/strided_slice_3StridedSlice?module_wrapper/lstm/TensorArrayV2Stack/TensorListStack:tensor:02module_wrapper/lstm/strided_slice_3/stack:output:04module_wrapper/lstm/strided_slice_3/stack_1:output:04module_wrapper/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
$module_wrapper/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ó
module_wrapper/lstm/transpose_1	Transpose?module_wrapper/lstm/TensorArrayV2Stack/TensorListStack:tensor:0-module_wrapper/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	'*
dtype0
dense/MatMulMatMul,module_wrapper/lstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'g
activation/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'k
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'Í
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^module_wrapper/lstm/lstm_cell/ReadVariableOp/^module_wrapper/lstm/lstm_cell/ReadVariableOp_1/^module_wrapper/lstm/lstm_cell/ReadVariableOp_2/^module_wrapper/lstm/lstm_cell/ReadVariableOp_33^module_wrapper/lstm/lstm_cell/split/ReadVariableOp5^module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp^module_wrapper/lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,module_wrapper/lstm/lstm_cell/ReadVariableOp,module_wrapper/lstm/lstm_cell/ReadVariableOp2`
.module_wrapper/lstm/lstm_cell/ReadVariableOp_1.module_wrapper/lstm/lstm_cell/ReadVariableOp_12`
.module_wrapper/lstm/lstm_cell/ReadVariableOp_2.module_wrapper/lstm/lstm_cell/ReadVariableOp_22`
.module_wrapper/lstm/lstm_cell/ReadVariableOp_3.module_wrapper/lstm/lstm_cell/ReadVariableOp_32h
2module_wrapper/lstm/lstm_cell/split/ReadVariableOp2module_wrapper/lstm/lstm_cell/split/ReadVariableOp2l
4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp26
module_wrapper/lstm/whilemodule_wrapper/lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameinputs
ª
Ë
?__inference_lstm_layer_call_and_return_conditional_losses_14597
inputs_0:
'lstm_cell_split_readvariableop_resource:	'8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?t
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¤
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¤
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ñ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_14457*
condR
while_cond_14456*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ': : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'
"
_user_specified_name
inputs/0
´
¾
while_cond_13709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_13709___redundant_placeholder03
/while_while_cond_13709___redundant_placeholder13
/while_while_cond_13709___redundant_placeholder23
/while_while_cond_13709___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
K
¨
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14809

inputs
states_0
states_10
split_readvariableop_resource:	'.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	'*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1

´
$__inference_lstm_layer_call_fn_14085
inputs_0
unknown:	'
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_14025p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ': : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'
"
_user_specified_name
inputs/0
K
¨
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14720

inputs
states_0
states_10
split_readvariableop_resource:	'.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	'*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1

î
E__inference_sequential_layer_call_and_return_conditional_losses_12441
module_wrapper_input'
module_wrapper_12427:	'#
module_wrapper_12429:	(
module_wrapper_12431:

dense_12434:	'
dense_12436:'
identity¢dense/StatefulPartitionedCall¢&module_wrapper/StatefulPartitionedCall¬
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_12427module_wrapper_12429module_wrapper_12431*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_12022
dense/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0dense_12434dense_12436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_12040Û
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12051r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
NoOpNoOp^dense/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:a ]
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
.
_user_specified_namemodule_wrapper_input
Ñ
ï
*__inference_sequential_layer_call_fn_12496

inputs
unknown:	'
	unknown_0:	
	unknown_1:

	unknown_2:	'
	unknown_3:'
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12054o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameinputs
è
°

lstm_while_body_13431&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	'E
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	B
.lstm_while_lstm_cell_readvariableop_resource_0:

lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	'C
4lstm_while_lstm_cell_split_1_readvariableop_resource:	@
,lstm_while_lstm_cell_readvariableop_resource:
¢#lstm/while/lstm_cell/ReadVariableOp¢%lstm/while/lstm_cell/ReadVariableOp_1¢%lstm/while/lstm_cell/ReadVariableOp_2¢%lstm/while/lstm_cell/ReadVariableOp_3¢)lstm/while/lstm_cell/split/ReadVariableOp¢+lstm/while/lstm_cell/split_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   ¿
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
element_dtype0f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	'*
dtype0á
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split´
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0×
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split¨
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¡
lstm/while/lstm_cell/MatMul_4MatMullstm_while_placeholder_2+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/MulMullstm/while/lstm_cell/add:z:0#lstm/while/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_1AddV2lstm/while/lstm_cell/Mul:z:0%lstm/while/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¿
*lstm/while/lstm_cell/clip_by_value/MinimumMinimumlstm/while/lstm_cell/Add_1:z:05lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¿
"lstm/while/lstm_cell/clip_by_valueMaximum.lstm/while/lstm_cell/clip_by_value/Minimum:z:0-lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_5MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_1Mullstm/while/lstm_cell/add_2:z:0%lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_3AddV2lstm/while/lstm_cell/Mul_1:z:0%lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_1/MinimumMinimumlstm/while/lstm_cell/Add_3:z:07lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_1Maximum0lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_2Mul(lstm/while/lstm_cell/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_6MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_3Mul&lstm/while/lstm_cell/clip_by_value:z:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/add_5AddV2lstm/while/lstm_cell/mul_2:z:0lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask£
lstm/while/lstm_cell/MatMul_7MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
lstm/while/lstm_cell/add_6AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>a
lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/while/lstm_cell/Mul_4Mullstm/while/lstm_cell/add_6:z:0%lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/Add_7AddV2lstm/while/lstm_cell/Mul_4:z:0%lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
,lstm/while/lstm_cell/clip_by_value_2/MinimumMinimumlstm/while/lstm_cell/Add_7:z:07lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Å
$lstm/while/lstm_cell/clip_by_value_2Maximum0lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/lstm_cell/mul_5Mul(lstm/while/lstm_cell/clip_by_value_2:z:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ô¸
Â
E__inference_sequential_layer_call_and_return_conditional_losses_13037

inputsN
;module_wrapper_lstm_lstm_cell_split_readvariableop_resource:	'L
=module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource:	I
5module_wrapper_lstm_lstm_cell_readvariableop_resource:
7
$dense_matmul_readvariableop_resource:	'3
%dense_biasadd_readvariableop_resource:'
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢,module_wrapper/lstm/lstm_cell/ReadVariableOp¢.module_wrapper/lstm/lstm_cell/ReadVariableOp_1¢.module_wrapper/lstm/lstm_cell/ReadVariableOp_2¢.module_wrapper/lstm/lstm_cell/ReadVariableOp_3¢2module_wrapper/lstm/lstm_cell/split/ReadVariableOp¢4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp¢module_wrapper/lstm/whileO
module_wrapper/lstm/ShapeShapeinputs*
T0*
_output_shapes
:q
'module_wrapper/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)module_wrapper/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)module_wrapper/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!module_wrapper/lstm/strided_sliceStridedSlice"module_wrapper/lstm/Shape:output:00module_wrapper/lstm/strided_slice/stack:output:02module_wrapper/lstm/strided_slice/stack_1:output:02module_wrapper/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"module_wrapper/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¯
 module_wrapper/lstm/zeros/packedPack*module_wrapper/lstm/strided_slice:output:0+module_wrapper/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
module_wrapper/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ©
module_wrapper/lstm/zerosFill)module_wrapper/lstm/zeros/packed:output:0(module_wrapper/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$module_wrapper/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :³
"module_wrapper/lstm/zeros_1/packedPack*module_wrapper/lstm/strided_slice:output:0-module_wrapper/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!module_wrapper/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
module_wrapper/lstm/zeros_1Fill+module_wrapper/lstm/zeros_1/packed:output:0*module_wrapper/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"module_wrapper/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
module_wrapper/lstm/transpose	Transposeinputs+module_wrapper/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ'l
module_wrapper/lstm/Shape_1Shape!module_wrapper/lstm/transpose:y:0*
T0*
_output_shapes
:s
)module_wrapper/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+module_wrapper/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+module_wrapper/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#module_wrapper/lstm/strided_slice_1StridedSlice$module_wrapper/lstm/Shape_1:output:02module_wrapper/lstm/strided_slice_1/stack:output:04module_wrapper/lstm/strided_slice_1/stack_1:output:04module_wrapper/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/module_wrapper/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!module_wrapper/lstm/TensorArrayV2TensorListReserve8module_wrapper/lstm/TensorArrayV2/element_shape:output:0,module_wrapper/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Imodule_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   
;module_wrapper/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!module_wrapper/lstm/transpose:y:0Rmodule_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)module_wrapper/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+module_wrapper/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+module_wrapper/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#module_wrapper/lstm/strided_slice_2StridedSlice!module_wrapper/lstm/transpose:y:02module_wrapper/lstm/strided_slice_2/stack:output:04module_wrapper/lstm/strided_slice_2/stack_1:output:04module_wrapper/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_masko
-module_wrapper/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¯
2module_wrapper/lstm/lstm_cell/split/ReadVariableOpReadVariableOp;module_wrapper_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0ü
#module_wrapper/lstm/lstm_cell/splitSplit6module_wrapper/lstm/lstm_cell/split/split_dim:output:0:module_wrapper/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split½
$module_wrapper/lstm/lstm_cell/MatMulMatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
&module_wrapper/lstm/lstm_cell/MatMul_1MatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
&module_wrapper/lstm/lstm_cell/MatMul_2MatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
&module_wrapper/lstm/lstm_cell/MatMul_3MatMul,module_wrapper/lstm/strided_slice_2:output:0,module_wrapper/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/module_wrapper/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ¯
4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp=module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ò
%module_wrapper/lstm/lstm_cell/split_1Split8module_wrapper/lstm/lstm_cell/split_1/split_dim:output:0<module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitÃ
%module_wrapper/lstm/lstm_cell/BiasAddBiasAdd.module_wrapper/lstm/lstm_cell/MatMul:product:0.module_wrapper/lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
'module_wrapper/lstm/lstm_cell/BiasAdd_1BiasAdd0module_wrapper/lstm/lstm_cell/MatMul_1:product:0.module_wrapper/lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
'module_wrapper/lstm/lstm_cell/BiasAdd_2BiasAdd0module_wrapper/lstm/lstm_cell/MatMul_2:product:0.module_wrapper/lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
'module_wrapper/lstm/lstm_cell/BiasAdd_3BiasAdd0module_wrapper/lstm/lstm_cell/MatMul_3:product:0.module_wrapper/lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,module_wrapper/lstm/lstm_cell/ReadVariableOpReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
1module_wrapper/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
3module_wrapper/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
3module_wrapper/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
+module_wrapper/lstm/lstm_cell/strided_sliceStridedSlice4module_wrapper/lstm/lstm_cell/ReadVariableOp:value:0:module_wrapper/lstm/lstm_cell/strided_slice/stack:output:0<module_wrapper/lstm/lstm_cell/strided_slice/stack_1:output:0<module_wrapper/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask½
&module_wrapper/lstm/lstm_cell/MatMul_4MatMul"module_wrapper/lstm/zeros:output:04module_wrapper/lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
!module_wrapper/lstm/lstm_cell/addAddV2.module_wrapper/lstm/lstm_cell/BiasAdd:output:00module_wrapper/lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#module_wrapper/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%module_wrapper/lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?°
!module_wrapper/lstm/lstm_cell/MulMul%module_wrapper/lstm/lstm_cell/add:z:0,module_wrapper/lstm/lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
#module_wrapper/lstm/lstm_cell/Add_1AddV2%module_wrapper/lstm/lstm_cell/Mul:z:0.module_wrapper/lstm/lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5module_wrapper/lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ú
3module_wrapper/lstm/lstm_cell/clip_by_value/MinimumMinimum'module_wrapper/lstm/lstm_cell/Add_1:z:0>module_wrapper/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
-module_wrapper/lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ú
+module_wrapper/lstm/lstm_cell/clip_by_valueMaximum7module_wrapper/lstm/lstm_cell/clip_by_value/Minimum:z:06module_wrapper/lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
3module_wrapper/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5module_wrapper/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5module_wrapper/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-module_wrapper/lstm/lstm_cell/strided_slice_1StridedSlice6module_wrapper/lstm/lstm_cell/ReadVariableOp_1:value:0<module_wrapper/lstm/lstm_cell/strided_slice_1/stack:output:0>module_wrapper/lstm/lstm_cell/strided_slice_1/stack_1:output:0>module_wrapper/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¿
&module_wrapper/lstm/lstm_cell/MatMul_5MatMul"module_wrapper/lstm/zeros:output:06module_wrapper/lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
#module_wrapper/lstm/lstm_cell/add_2AddV20module_wrapper/lstm/lstm_cell/BiasAdd_1:output:00module_wrapper/lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%module_wrapper/lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%module_wrapper/lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
#module_wrapper/lstm/lstm_cell/Mul_1Mul'module_wrapper/lstm/lstm_cell/add_2:z:0.module_wrapper/lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
#module_wrapper/lstm/lstm_cell/Add_3AddV2'module_wrapper/lstm/lstm_cell/Mul_1:z:0.module_wrapper/lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Þ
5module_wrapper/lstm/lstm_cell/clip_by_value_1/MinimumMinimum'module_wrapper/lstm/lstm_cell/Add_3:z:0@module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
/module_wrapper/lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    à
-module_wrapper/lstm/lstm_cell/clip_by_value_1Maximum9module_wrapper/lstm/lstm_cell/clip_by_value_1/Minimum:z:08module_wrapper/lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
#module_wrapper/lstm/lstm_cell/mul_2Mul1module_wrapper/lstm/lstm_cell/clip_by_value_1:z:0$module_wrapper/lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
3module_wrapper/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5module_wrapper/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
5module_wrapper/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-module_wrapper/lstm/lstm_cell/strided_slice_2StridedSlice6module_wrapper/lstm/lstm_cell/ReadVariableOp_2:value:0<module_wrapper/lstm/lstm_cell/strided_slice_2/stack:output:0>module_wrapper/lstm/lstm_cell/strided_slice_2/stack_1:output:0>module_wrapper/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¿
&module_wrapper/lstm/lstm_cell/MatMul_6MatMul"module_wrapper/lstm/zeros:output:06module_wrapper/lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
#module_wrapper/lstm/lstm_cell/add_4AddV20module_wrapper/lstm/lstm_cell/BiasAdd_2:output:00module_wrapper/lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"module_wrapper/lstm/lstm_cell/TanhTanh'module_wrapper/lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
#module_wrapper/lstm/lstm_cell/mul_3Mul/module_wrapper/lstm/lstm_cell/clip_by_value:z:0&module_wrapper/lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
#module_wrapper/lstm/lstm_cell/add_5AddV2'module_wrapper/lstm/lstm_cell/mul_2:z:0'module_wrapper/lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp5module_wrapper_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
3module_wrapper/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
5module_wrapper/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
5module_wrapper/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-module_wrapper/lstm/lstm_cell/strided_slice_3StridedSlice6module_wrapper/lstm/lstm_cell/ReadVariableOp_3:value:0<module_wrapper/lstm/lstm_cell/strided_slice_3/stack:output:0>module_wrapper/lstm/lstm_cell/strided_slice_3/stack_1:output:0>module_wrapper/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¿
&module_wrapper/lstm/lstm_cell/MatMul_7MatMul"module_wrapper/lstm/zeros:output:06module_wrapper/lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
#module_wrapper/lstm/lstm_cell/add_6AddV20module_wrapper/lstm/lstm_cell/BiasAdd_3:output:00module_wrapper/lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%module_wrapper/lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%module_wrapper/lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
#module_wrapper/lstm/lstm_cell/Mul_4Mul'module_wrapper/lstm/lstm_cell/add_6:z:0.module_wrapper/lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
#module_wrapper/lstm/lstm_cell/Add_7AddV2'module_wrapper/lstm/lstm_cell/Mul_4:z:0.module_wrapper/lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Þ
5module_wrapper/lstm/lstm_cell/clip_by_value_2/MinimumMinimum'module_wrapper/lstm/lstm_cell/Add_7:z:0@module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
/module_wrapper/lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    à
-module_wrapper/lstm/lstm_cell/clip_by_value_2Maximum9module_wrapper/lstm/lstm_cell/clip_by_value_2/Minimum:z:08module_wrapper/lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$module_wrapper/lstm/lstm_cell/Tanh_1Tanh'module_wrapper/lstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#module_wrapper/lstm/lstm_cell/mul_5Mul1module_wrapper/lstm/lstm_cell/clip_by_value_2:z:0(module_wrapper/lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1module_wrapper/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
#module_wrapper/lstm/TensorArrayV2_1TensorListReserve:module_wrapper/lstm/TensorArrayV2_1/element_shape:output:0,module_wrapper/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
module_wrapper/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,module_wrapper/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&module_wrapper/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
module_wrapper/lstm/whileWhile/module_wrapper/lstm/while/loop_counter:output:05module_wrapper/lstm/while/maximum_iterations:output:0!module_wrapper/lstm/time:output:0,module_wrapper/lstm/TensorArrayV2_1:handle:0"module_wrapper/lstm/zeros:output:0$module_wrapper/lstm/zeros_1:output:0,module_wrapper/lstm/strided_slice_1:output:0Kmodule_wrapper/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0;module_wrapper_lstm_lstm_cell_split_readvariableop_resource=module_wrapper_lstm_lstm_cell_split_1_readvariableop_resource5module_wrapper_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$module_wrapper_lstm_while_body_12890*0
cond(R&
$module_wrapper_lstm_while_cond_12889*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Dmodule_wrapper/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ÿ
6module_wrapper/lstm/TensorArrayV2Stack/TensorListStackTensorListStack"module_wrapper/lstm/while:output:3Mmodule_wrapper/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:(ÿÿÿÿÿÿÿÿÿ*
element_dtype0|
)module_wrapper/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+module_wrapper/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+module_wrapper/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
#module_wrapper/lstm/strided_slice_3StridedSlice?module_wrapper/lstm/TensorArrayV2Stack/TensorListStack:tensor:02module_wrapper/lstm/strided_slice_3/stack:output:04module_wrapper/lstm/strided_slice_3/stack_1:output:04module_wrapper/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
$module_wrapper/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ó
module_wrapper/lstm/transpose_1	Transpose?module_wrapper/lstm/TensorArrayV2Stack/TensorListStack:tensor:0-module_wrapper/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	'*
dtype0
dense/MatMulMatMul,module_wrapper/lstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'g
activation/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'k
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'Í
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^module_wrapper/lstm/lstm_cell/ReadVariableOp/^module_wrapper/lstm/lstm_cell/ReadVariableOp_1/^module_wrapper/lstm/lstm_cell/ReadVariableOp_2/^module_wrapper/lstm/lstm_cell/ReadVariableOp_33^module_wrapper/lstm/lstm_cell/split/ReadVariableOp5^module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp^module_wrapper/lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,module_wrapper/lstm/lstm_cell/ReadVariableOp,module_wrapper/lstm/lstm_cell/ReadVariableOp2`
.module_wrapper/lstm/lstm_cell/ReadVariableOp_1.module_wrapper/lstm/lstm_cell/ReadVariableOp_12`
.module_wrapper/lstm/lstm_cell/ReadVariableOp_2.module_wrapper/lstm/lstm_cell/ReadVariableOp_22`
.module_wrapper/lstm/lstm_cell/ReadVariableOp_3.module_wrapper/lstm/lstm_cell/ReadVariableOp_32h
2module_wrapper/lstm/lstm_cell/split/ReadVariableOp2module_wrapper/lstm/lstm_cell/split/ReadVariableOp2l
4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp4module_wrapper/lstm/lstm_cell/split_1/ReadVariableOp26
module_wrapper/lstm/whilemodule_wrapper/lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
 
_user_specified_nameinputs
å
Î
$module_wrapper_lstm_while_cond_12626D
@module_wrapper_lstm_while_module_wrapper_lstm_while_loop_counterJ
Fmodule_wrapper_lstm_while_module_wrapper_lstm_while_maximum_iterations)
%module_wrapper_lstm_while_placeholder+
'module_wrapper_lstm_while_placeholder_1+
'module_wrapper_lstm_while_placeholder_2+
'module_wrapper_lstm_while_placeholder_3F
Bmodule_wrapper_lstm_while_less_module_wrapper_lstm_strided_slice_1[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12626___redundant_placeholder0[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12626___redundant_placeholder1[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12626___redundant_placeholder2[
Wmodule_wrapper_lstm_while_module_wrapper_lstm_while_cond_12626___redundant_placeholder3&
"module_wrapper_lstm_while_identity
²
module_wrapper/lstm/while/LessLess%module_wrapper_lstm_while_placeholderBmodule_wrapper_lstm_while_less_module_wrapper_lstm_strided_slice_1*
T0*
_output_shapes
: s
"module_wrapper/lstm/while/IdentityIdentity"module_wrapper/lstm/while/Less:z:0*
T0
*
_output_shapes
: "Q
"module_wrapper_lstm_while_identity+module_wrapper/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
À	
¢
lstm_while_cond_13174&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_13174___redundant_placeholder0=
9lstm_while_lstm_while_cond_13174___redundant_placeholder1=
9lstm_while_lstm_while_cond_13174___redundant_placeholder2=
9lstm_while_lstm_while_cond_13174___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ª
Ë
?__inference_lstm_layer_call_and_return_conditional_losses_14341
inputs_0:
'lstm_cell_split_readvariableop_resource:	'8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	'*
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	':	':	':	'*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?t
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¤
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¤
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ñ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_14201*
condR
while_cond_14200*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ': : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'
"
_user_specified_name
inputs/0
À	
¢
lstm_while_cond_13430&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_13430___redundant_placeholder0=
9lstm_while_lstm_while_cond_13430___redundant_placeholder1=
9lstm_while_lstm_while_cond_13430___redundant_placeholder2=
9lstm_while_lstm_while_cond_13430___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
7
ó
?__inference_lstm_layer_call_and_return_conditional_losses_13778

inputs"
lstm_cell_13697:	'
lstm_cell_13699:	#
lstm_cell_13701:

identity¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ'   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
shrink_axis_maskå
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13697lstm_cell_13699lstm_cell_13701*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13696n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13697lstm_cell_13699lstm_cell_13701*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_13710*
condR
while_cond_13709*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ': : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'
 
_user_specified_nameinputs
-
¿
__inference__traced_save_14883
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableopC
?savev2_module_wrapper_lstm_lstm_cell_kernel_read_readvariableopM
Isavev2_module_wrapper_lstm_lstm_cell_recurrent_kernel_read_readvariableopA
=savev2_module_wrapper_lstm_lstm_cell_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableopO
Ksavev2_rmsprop_module_wrapper_lstm_lstm_cell_kernel_rms_read_readvariableopY
Usavev2_rmsprop_module_wrapper_lstm_lstm_cell_recurrent_kernel_rms_read_readvariableopM
Isavev2_rmsprop_module_wrapper_lstm_lstm_cell_bias_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¼
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*å
valueÛBØB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B Ñ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop?savev2_module_wrapper_lstm_lstm_cell_kernel_read_readvariableopIsavev2_module_wrapper_lstm_lstm_cell_recurrent_kernel_read_readvariableop=savev2_module_wrapper_lstm_lstm_cell_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableopKsavev2_rmsprop_module_wrapper_lstm_lstm_cell_kernel_rms_read_readvariableopUsavev2_rmsprop_module_wrapper_lstm_lstm_cell_recurrent_kernel_rms_read_readvariableopIsavev2_rmsprop_module_wrapper_lstm_lstm_cell_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesr
p: :	':':	':
:: : : : : : : :	':':	':
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	': 

_output_shapes
:':%!

_output_shapes
:	':&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	': 

_output_shapes
:':%!

_output_shapes
:	':&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
½

%__inference_dense_layer_call_fn_14043

inputs
unknown:	'
	unknown_0:'
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_12040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
ý
*__inference_sequential_layer_call_fn_12424
module_wrapper_input
unknown:	'
	unknown_0:	
	unknown_1:

	unknown_2:	'
	unknown_3:'
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12396o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(': : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ('
.
_user_specified_namemodule_wrapper_input"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ë
serving_default·
Y
module_wrapper_inputA
&serving_default_module_wrapper_input:0ÿÿÿÿÿÿÿÿÿ('>

activation0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ'tensorflow/serving/predict:¯
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
²
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
C
"0
#1
$2
3
4"
trackable_list_wrapper
C
"0
#1
$2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
Þ
*trace_0
+trace_1
,trace_2
-trace_32ó
*__inference_sequential_layer_call_fn_12067
*__inference_sequential_layer_call_fn_12496
*__inference_sequential_layer_call_fn_12511
*__inference_sequential_layer_call_fn_12424À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z*trace_0z+trace_1z,trace_2z-trace_3
Ê
.trace_0
/trace_1
0trace_2
1trace_32ß
E__inference_sequential_layer_call_and_return_conditional_losses_12774
E__inference_sequential_layer_call_and_return_conditional_losses_13037
E__inference_sequential_layer_call_and_return_conditional_losses_12441
E__inference_sequential_layer_call_and_return_conditional_losses_12458À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z.trace_0z/trace_1z0trace_2z1trace_3
ØBÕ
 __inference__wrapped_model_11759module_wrapper_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

2iter
	3decay
4learning_rate
5momentum
6rho	rmsv	rmsw	"rmsx	#rmsy	$rmsz"
	optimizer
,
7serving_default"
signature_map
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ú
=trace_0
>trace_12£
.__inference_module_wrapper_layer_call_fn_13048
.__inference_module_wrapper_layer_call_fn_13059À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 z=trace_0z>trace_1

?trace_0
@trace_12Ù
I__inference_module_wrapper_layer_call_and_return_conditional_losses_13315
I__inference_module_wrapper_layer_call_and_return_conditional_losses_13571À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 z?trace_0z@trace_1
Ã
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
*E&call_and_return_all_conditional_losses
F__call__
Gcell
H
state_spec"
_tf_keras_rnn_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
é
Ntrace_02Ì
%__inference_dense_layer_call_fn_14043¢
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
 zNtrace_0

Otrace_02ç
@__inference_dense_layer_call_and_return_conditional_losses_14053¢
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
 zOtrace_0
:	'2dense/kernel
:'2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
î
Utrace_02Ñ
*__inference_activation_layer_call_fn_14058¢
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
 zUtrace_0

Vtrace_02ì
E__inference_activation_layer_call_and_return_conditional_losses_14063¢
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
 zVtrace_0
7:5	'2$module_wrapper/lstm/lstm_cell/kernel
B:@
2.module_wrapper/lstm/lstm_cell/recurrent_kernel
1:/2"module_wrapper/lstm/lstm_cell/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
W0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
*__inference_sequential_layer_call_fn_12067module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
*__inference_sequential_layer_call_fn_12496inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
*__inference_sequential_layer_call_fn_12511inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
*__inference_sequential_layer_call_fn_12424module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_12774inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_13037inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¥B¢
E__inference_sequential_layer_call_and_return_conditional_losses_12441module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¥B¢
E__inference_sequential_layer_call_and_return_conditional_losses_12458module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
×BÔ
#__inference_signature_wrapper_12481module_wrapper_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
Bý
.__inference_module_wrapper_layer_call_fn_13048args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Bý
.__inference_module_wrapper_layer_call_fn_13059args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_module_wrapper_layer_call_and_return_conditional_losses_13315args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_module_wrapper_layer_call_and_return_conditional_losses_13571args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Xlayers
Atrainable_variables
B	variables
Ymetrics
Cregularization_losses

Zstates
[non_trainable_variables
\layer_regularization_losses
]layer_metrics
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object

^trace_0
_trace_12Ú
?__inference_lstm_layer_call_and_return_conditional_losses_14341
?__inference_lstm_layer_call_and_return_conditional_losses_14597Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z^trace_0z_trace_1
Û
`trace_0
atrace_12¤
$__inference_lstm_layer_call_fn_14074
$__inference_lstm_layer_call_fn_14085Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z`trace_0zatrace_1
á
btrainable_variables
c	variables
dregularization_losses
e	keras_api
*f&call_and_return_all_conditional_losses
g__call__
h
state_size

"kernel
#recurrent_kernel
$bias"
_tf_keras_layer
 "
trackable_list_wrapper
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
ÙBÖ
%__inference_dense_layer_call_fn_14043inputs"¢
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
ôBñ
@__inference_dense_layer_call_and_return_conditional_losses_14053inputs"¢
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
ÞBÛ
*__inference_activation_layer_call_fn_14058inputs"¢
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
ùBö
E__inference_activation_layer_call_and_return_conditional_losses_14063inputs"¢
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
N
i	variables
j	keras_api
	ktotal
	lcount"
_tf_keras_metric
'
G0"
trackable_list_wrapper
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
¨B¥
?__inference_lstm_layer_call_and_return_conditional_losses_14341inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨B¥
?__inference_lstm_layer_call_and_return_conditional_losses_14597inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
$__inference_lstm_layer_call_fn_14074inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
$__inference_lstm_layer_call_fn_14085inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
­

mlayers
btrainable_variables
c	variables
nmetrics
dregularization_losses
onon_trainable_variables
player_regularization_losses
qlayer_metrics
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object

rtrace_0
strace_12Í
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14720
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14809¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zrtrace_0zstrace_1
Î
ttrace_0
utrace_12
)__inference_lstm_cell_layer_call_fn_14614
)__inference_lstm_cell_layer_call_fn_14631¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zttrace_0zutrace_1
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
:  (2total
:  (2count
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
¨B¥
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14720inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨B¥
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14809inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
)__inference_lstm_cell_layer_call_fn_14614inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
)__inference_lstm_cell_layer_call_fn_14631inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
):'	'2RMSprop/dense/kernel/rms
": '2RMSprop/dense/bias/rms
A:?	'20RMSprop/module_wrapper/lstm/lstm_cell/kernel/rms
L:J
2:RMSprop/module_wrapper/lstm/lstm_cell/recurrent_kernel/rms
;:92.RMSprop/module_wrapper/lstm/lstm_cell/bias/rms¨
 __inference__wrapped_model_11759"$#A¢>
7¢4
2/
module_wrapper_inputÿÿÿÿÿÿÿÿÿ('
ª "7ª4
2

activation$!

activationÿÿÿÿÿÿÿÿÿ'¡
E__inference_activation_layer_call_and_return_conditional_losses_14063X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ'
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ'
 y
*__inference_activation_layer_call_fn_14058K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ'
ª "ÿÿÿÿÿÿÿÿÿ'¡
@__inference_dense_layer_call_and_return_conditional_losses_14053]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ'
 y
%__inference_dense_layer_call_fn_14043P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ'Ë
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14720"$#¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ'
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ë
D__inference_lstm_cell_layer_call_and_return_conditional_losses_14809"$#¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ'
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
  
)__inference_lstm_cell_layer_call_fn_14614ò"$#¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ'
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ 
)__inference_lstm_cell_layer_call_fn_14631ò"$#¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ'
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÁ
?__inference_lstm_layer_call_and_return_conditional_losses_14341~"$#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Á
?__inference_lstm_layer_call_and_return_conditional_losses_14597~"$#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
$__inference_lstm_layer_call_fn_14074q"$#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_lstm_layer_call_fn_14085q"$#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ'

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
I__inference_module_wrapper_layer_call_and_return_conditional_losses_13315r"$#C¢@
)¢&
$!
args_0ÿÿÿÿÿÿÿÿÿ('
ª

trainingp "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¿
I__inference_module_wrapper_layer_call_and_return_conditional_losses_13571r"$#C¢@
)¢&
$!
args_0ÿÿÿÿÿÿÿÿÿ('
ª

trainingp"&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_module_wrapper_layer_call_fn_13048e"$#C¢@
)¢&
$!
args_0ÿÿÿÿÿÿÿÿÿ('
ª

trainingp "ÿÿÿÿÿÿÿÿÿ
.__inference_module_wrapper_layer_call_fn_13059e"$#C¢@
)¢&
$!
args_0ÿÿÿÿÿÿÿÿÿ('
ª

trainingp"ÿÿÿÿÿÿÿÿÿÂ
E__inference_sequential_layer_call_and_return_conditional_losses_12441y"$#I¢F
?¢<
2/
module_wrapper_inputÿÿÿÿÿÿÿÿÿ('
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ'
 Â
E__inference_sequential_layer_call_and_return_conditional_losses_12458y"$#I¢F
?¢<
2/
module_wrapper_inputÿÿÿÿÿÿÿÿÿ('
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ'
 ´
E__inference_sequential_layer_call_and_return_conditional_losses_12774k"$#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ('
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ'
 ´
E__inference_sequential_layer_call_and_return_conditional_losses_13037k"$#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ('
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ'
 
*__inference_sequential_layer_call_fn_12067l"$#I¢F
?¢<
2/
module_wrapper_inputÿÿÿÿÿÿÿÿÿ('
p 

 
ª "ÿÿÿÿÿÿÿÿÿ'
*__inference_sequential_layer_call_fn_12424l"$#I¢F
?¢<
2/
module_wrapper_inputÿÿÿÿÿÿÿÿÿ('
p

 
ª "ÿÿÿÿÿÿÿÿÿ'
*__inference_sequential_layer_call_fn_12496^"$#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ('
p 

 
ª "ÿÿÿÿÿÿÿÿÿ'
*__inference_sequential_layer_call_fn_12511^"$#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ('
p

 
ª "ÿÿÿÿÿÿÿÿÿ'Ã
#__inference_signature_wrapper_12481"$#Y¢V
¢ 
OªL
J
module_wrapper_input2/
module_wrapper_inputÿÿÿÿÿÿÿÿÿ('"7ª4
2

activation$!

activationÿÿÿÿÿÿÿÿÿ'