��
��
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
dtypetype�
�
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
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.2.0-dev202003302v1.12.1-28298-g31cb6df2c28��	
�
my_model_9/conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namemy_model_9/conv2d_27/kernel
�
/my_model_9/conv2d_27/kernel/Read/ReadVariableOpReadVariableOpmy_model_9/conv2d_27/kernel*&
_output_shapes
: *
dtype0
�
my_model_9/conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namemy_model_9/conv2d_27/bias
�
-my_model_9/conv2d_27/bias/Read/ReadVariableOpReadVariableOpmy_model_9/conv2d_27/bias*
_output_shapes
: *
dtype0
�
my_model_9/conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*,
shared_namemy_model_9/conv2d_28/kernel
�
/my_model_9/conv2d_28/kernel/Read/ReadVariableOpReadVariableOpmy_model_9/conv2d_28/kernel*&
_output_shapes
: @*
dtype0
�
my_model_9/conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namemy_model_9/conv2d_28/bias
�
-my_model_9/conv2d_28/bias/Read/ReadVariableOpReadVariableOpmy_model_9/conv2d_28/bias*
_output_shapes
:@*
dtype0
�
my_model_9/conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*,
shared_namemy_model_9/conv2d_29/kernel
�
/my_model_9/conv2d_29/kernel/Read/ReadVariableOpReadVariableOpmy_model_9/conv2d_29/kernel*'
_output_shapes
:@�*
dtype0
�
my_model_9/conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namemy_model_9/conv2d_29/bias
�
-my_model_9/conv2d_29/bias/Read/ReadVariableOpReadVariableOpmy_model_9/conv2d_29/bias*
_output_shapes	
:�*
dtype0
�
my_model_9/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*+
shared_namemy_model_9/dense_27/kernel
�
.my_model_9/dense_27/kernel/Read/ReadVariableOpReadVariableOpmy_model_9/dense_27/kernel* 
_output_shapes
:
��@*
dtype0
�
my_model_9/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namemy_model_9/dense_27/bias
�
,my_model_9/dense_27/bias/Read/ReadVariableOpReadVariableOpmy_model_9/dense_27/bias*
_output_shapes
:@*
dtype0
�
my_model_9/dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*+
shared_namemy_model_9/dense_28/kernel
�
.my_model_9/dense_28/kernel/Read/ReadVariableOpReadVariableOpmy_model_9/dense_28/kernel*
_output_shapes

:@@*
dtype0
�
my_model_9/dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namemy_model_9/dense_28/bias
�
,my_model_9/dense_28/bias/Read/ReadVariableOpReadVariableOpmy_model_9/dense_28/bias*
_output_shapes
:@*
dtype0
�
my_model_9/dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*+
shared_namemy_model_9/dense_29/kernel
�
.my_model_9/dense_29/kernel/Read/ReadVariableOpReadVariableOpmy_model_9/dense_29/kernel*
_output_shapes

:@
*
dtype0
�
my_model_9/dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namemy_model_9/dense_29/bias
�
,my_model_9/dense_29/bias/Read/ReadVariableOpReadVariableOpmy_model_9/dense_29/bias*
_output_shapes
:
*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
"Adam/my_model_9/conv2d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/my_model_9/conv2d_27/kernel/m
�
6Adam/my_model_9/conv2d_27/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/my_model_9/conv2d_27/kernel/m*&
_output_shapes
: *
dtype0
�
 Adam/my_model_9/conv2d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/my_model_9/conv2d_27/bias/m
�
4Adam/my_model_9/conv2d_27/bias/m/Read/ReadVariableOpReadVariableOp Adam/my_model_9/conv2d_27/bias/m*
_output_shapes
: *
dtype0
�
"Adam/my_model_9/conv2d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"Adam/my_model_9/conv2d_28/kernel/m
�
6Adam/my_model_9/conv2d_28/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/my_model_9/conv2d_28/kernel/m*&
_output_shapes
: @*
dtype0
�
 Adam/my_model_9/conv2d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/my_model_9/conv2d_28/bias/m
�
4Adam/my_model_9/conv2d_28/bias/m/Read/ReadVariableOpReadVariableOp Adam/my_model_9/conv2d_28/bias/m*
_output_shapes
:@*
dtype0
�
"Adam/my_model_9/conv2d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*3
shared_name$"Adam/my_model_9/conv2d_29/kernel/m
�
6Adam/my_model_9/conv2d_29/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/my_model_9/conv2d_29/kernel/m*'
_output_shapes
:@�*
dtype0
�
 Adam/my_model_9/conv2d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/my_model_9/conv2d_29/bias/m
�
4Adam/my_model_9/conv2d_29/bias/m/Read/ReadVariableOpReadVariableOp Adam/my_model_9/conv2d_29/bias/m*
_output_shapes	
:�*
dtype0
�
!Adam/my_model_9/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*2
shared_name#!Adam/my_model_9/dense_27/kernel/m
�
5Adam/my_model_9/dense_27/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/my_model_9/dense_27/kernel/m* 
_output_shapes
:
��@*
dtype0
�
Adam/my_model_9/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/my_model_9/dense_27/bias/m
�
3Adam/my_model_9/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/my_model_9/dense_27/bias/m*
_output_shapes
:@*
dtype0
�
!Adam/my_model_9/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*2
shared_name#!Adam/my_model_9/dense_28/kernel/m
�
5Adam/my_model_9/dense_28/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/my_model_9/dense_28/kernel/m*
_output_shapes

:@@*
dtype0
�
Adam/my_model_9/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/my_model_9/dense_28/bias/m
�
3Adam/my_model_9/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/my_model_9/dense_28/bias/m*
_output_shapes
:@*
dtype0
�
!Adam/my_model_9/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*2
shared_name#!Adam/my_model_9/dense_29/kernel/m
�
5Adam/my_model_9/dense_29/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/my_model_9/dense_29/kernel/m*
_output_shapes

:@
*
dtype0
�
Adam/my_model_9/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/my_model_9/dense_29/bias/m
�
3Adam/my_model_9/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/my_model_9/dense_29/bias/m*
_output_shapes
:
*
dtype0
�
"Adam/my_model_9/conv2d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/my_model_9/conv2d_27/kernel/v
�
6Adam/my_model_9/conv2d_27/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/my_model_9/conv2d_27/kernel/v*&
_output_shapes
: *
dtype0
�
 Adam/my_model_9/conv2d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/my_model_9/conv2d_27/bias/v
�
4Adam/my_model_9/conv2d_27/bias/v/Read/ReadVariableOpReadVariableOp Adam/my_model_9/conv2d_27/bias/v*
_output_shapes
: *
dtype0
�
"Adam/my_model_9/conv2d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"Adam/my_model_9/conv2d_28/kernel/v
�
6Adam/my_model_9/conv2d_28/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/my_model_9/conv2d_28/kernel/v*&
_output_shapes
: @*
dtype0
�
 Adam/my_model_9/conv2d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/my_model_9/conv2d_28/bias/v
�
4Adam/my_model_9/conv2d_28/bias/v/Read/ReadVariableOpReadVariableOp Adam/my_model_9/conv2d_28/bias/v*
_output_shapes
:@*
dtype0
�
"Adam/my_model_9/conv2d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*3
shared_name$"Adam/my_model_9/conv2d_29/kernel/v
�
6Adam/my_model_9/conv2d_29/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/my_model_9/conv2d_29/kernel/v*'
_output_shapes
:@�*
dtype0
�
 Adam/my_model_9/conv2d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/my_model_9/conv2d_29/bias/v
�
4Adam/my_model_9/conv2d_29/bias/v/Read/ReadVariableOpReadVariableOp Adam/my_model_9/conv2d_29/bias/v*
_output_shapes	
:�*
dtype0
�
!Adam/my_model_9/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��@*2
shared_name#!Adam/my_model_9/dense_27/kernel/v
�
5Adam/my_model_9/dense_27/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/my_model_9/dense_27/kernel/v* 
_output_shapes
:
��@*
dtype0
�
Adam/my_model_9/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/my_model_9/dense_27/bias/v
�
3Adam/my_model_9/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/my_model_9/dense_27/bias/v*
_output_shapes
:@*
dtype0
�
!Adam/my_model_9/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*2
shared_name#!Adam/my_model_9/dense_28/kernel/v
�
5Adam/my_model_9/dense_28/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/my_model_9/dense_28/kernel/v*
_output_shapes

:@@*
dtype0
�
Adam/my_model_9/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/my_model_9/dense_28/bias/v
�
3Adam/my_model_9/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/my_model_9/dense_28/bias/v*
_output_shapes
:@*
dtype0
�
!Adam/my_model_9/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*2
shared_name#!Adam/my_model_9/dense_29/kernel/v
�
5Adam/my_model_9/dense_29/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/my_model_9/dense_29/kernel/v*
_output_shapes

:@
*
dtype0
�
Adam/my_model_9/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/my_model_9/dense_29/bias/v
�
3Adam/my_model_9/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/my_model_9/dense_29/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
�M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�L
value�LB�L B�L
�
	conv1
	relu1
	drop1
	conv2
	relu2
	drop2
	conv3
	relu3
		drop3

flatten
fc1
fc2
fc3
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
R
(trainable_variables
)	variables
*regularization_losses
+	keras_api
R
,trainable_variables
-	variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
R
6trainable_variables
7	variables
8regularization_losses
9	keras_api
R
:trainable_variables
;	variables
<regularization_losses
=	keras_api
R
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
h

Hkernel
Ibias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
h

Nkernel
Obias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
�

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate
Xiterm�m�"m�#m�0m�1m�Bm�Cm�Hm�Im�Nm�Om�v�v�"v�#v�0v�1v�Bv�Cv�Hv�Iv�Nv�Ov�
V
0
1
"2
#3
04
15
B6
C7
H8
I9
N10
O11
V
0
1
"2
#3
04
15
B6
C7
H8
I9
N10
O11
 
�
Ylayer_regularization_losses
Zmetrics
trainable_variables
	variables
[non_trainable_variables
regularization_losses
\layer_metrics

]layers
 
XV
VARIABLE_VALUEmy_model_9/conv2d_27/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_model_9/conv2d_27/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
^layer_regularization_losses
_metrics
trainable_variables
	variables
`non_trainable_variables
regularization_losses
alayer_metrics

blayers
 
 
 
�
clayer_regularization_losses
dmetrics
trainable_variables
	variables
enon_trainable_variables
regularization_losses
flayer_metrics

glayers
 
 
 
�
hlayer_regularization_losses
imetrics
trainable_variables
	variables
jnon_trainable_variables
 regularization_losses
klayer_metrics

llayers
XV
VARIABLE_VALUEmy_model_9/conv2d_28/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_model_9/conv2d_28/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
mlayer_regularization_losses
nmetrics
$trainable_variables
%	variables
onon_trainable_variables
&regularization_losses
player_metrics

qlayers
 
 
 
�
rlayer_regularization_losses
smetrics
(trainable_variables
)	variables
tnon_trainable_variables
*regularization_losses
ulayer_metrics

vlayers
 
 
 
�
wlayer_regularization_losses
xmetrics
,trainable_variables
-	variables
ynon_trainable_variables
.regularization_losses
zlayer_metrics

{layers
XV
VARIABLE_VALUEmy_model_9/conv2d_29/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_model_9/conv2d_29/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
�
|layer_regularization_losses
}metrics
2trainable_variables
3	variables
~non_trainable_variables
4regularization_losses
layer_metrics
�layers
 
 
 
�
 �layer_regularization_losses
�metrics
6trainable_variables
7	variables
�non_trainable_variables
8regularization_losses
�layer_metrics
�layers
 
 
 
�
 �layer_regularization_losses
�metrics
:trainable_variables
;	variables
�non_trainable_variables
<regularization_losses
�layer_metrics
�layers
 
 
 
�
 �layer_regularization_losses
�metrics
>trainable_variables
?	variables
�non_trainable_variables
@regularization_losses
�layer_metrics
�layers
US
VARIABLE_VALUEmy_model_9/dense_27/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmy_model_9/dense_27/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
�
 �layer_regularization_losses
�metrics
Dtrainable_variables
E	variables
�non_trainable_variables
Fregularization_losses
�layer_metrics
�layers
US
VARIABLE_VALUEmy_model_9/dense_28/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmy_model_9/dense_28/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1

H0
I1
 
�
 �layer_regularization_losses
�metrics
Jtrainable_variables
K	variables
�non_trainable_variables
Lregularization_losses
�layer_metrics
�layers
US
VARIABLE_VALUEmy_model_9/dense_29/kernel%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmy_model_9/dense_29/bias#fc3/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
 
�
 �layer_regularization_losses
�metrics
Ptrainable_variables
Q	variables
�non_trainable_variables
Rregularization_losses
�layer_metrics
�layers
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
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
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
{y
VARIABLE_VALUE"Adam/my_model_9/conv2d_27/kernel/mCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE Adam/my_model_9/conv2d_27/bias/mAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/my_model_9/conv2d_28/kernel/mCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE Adam/my_model_9/conv2d_28/bias/mAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/my_model_9/conv2d_29/kernel/mCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE Adam/my_model_9/conv2d_29/bias/mAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/my_model_9/dense_27/kernel/mAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model_9/dense_27/bias/m?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/my_model_9/dense_28/kernel/mAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model_9/dense_28/bias/m?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/my_model_9/dense_29/kernel/mAfc3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model_9/dense_29/bias/m?fc3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/my_model_9/conv2d_27/kernel/vCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE Adam/my_model_9/conv2d_27/bias/vAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/my_model_9/conv2d_28/kernel/vCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE Adam/my_model_9/conv2d_28/bias/vAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/my_model_9/conv2d_29/kernel/vCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE Adam/my_model_9/conv2d_29/bias/vAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/my_model_9/dense_27/kernel/vAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model_9/dense_27/bias/v?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/my_model_9/dense_28/kernel/vAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model_9/dense_28/bias/v?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/my_model_9/dense_29/kernel/vAfc3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model_9/dense_29/bias/v?fc3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model_9/conv2d_27/kernelmy_model_9/conv2d_27/biasmy_model_9/conv2d_28/kernelmy_model_9/conv2d_28/biasmy_model_9/conv2d_29/kernelmy_model_9/conv2d_29/biasmy_model_9/dense_27/kernelmy_model_9/dense_27/biasmy_model_9/dense_28/kernelmy_model_9/dense_28/biasmy_model_9/dense_29/kernelmy_model_9/dense_29/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_6011
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/my_model_9/conv2d_27/kernel/Read/ReadVariableOp-my_model_9/conv2d_27/bias/Read/ReadVariableOp/my_model_9/conv2d_28/kernel/Read/ReadVariableOp-my_model_9/conv2d_28/bias/Read/ReadVariableOp/my_model_9/conv2d_29/kernel/Read/ReadVariableOp-my_model_9/conv2d_29/bias/Read/ReadVariableOp.my_model_9/dense_27/kernel/Read/ReadVariableOp,my_model_9/dense_27/bias/Read/ReadVariableOp.my_model_9/dense_28/kernel/Read/ReadVariableOp,my_model_9/dense_28/bias/Read/ReadVariableOp.my_model_9/dense_29/kernel/Read/ReadVariableOp,my_model_9/dense_29/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/my_model_9/conv2d_27/kernel/m/Read/ReadVariableOp4Adam/my_model_9/conv2d_27/bias/m/Read/ReadVariableOp6Adam/my_model_9/conv2d_28/kernel/m/Read/ReadVariableOp4Adam/my_model_9/conv2d_28/bias/m/Read/ReadVariableOp6Adam/my_model_9/conv2d_29/kernel/m/Read/ReadVariableOp4Adam/my_model_9/conv2d_29/bias/m/Read/ReadVariableOp5Adam/my_model_9/dense_27/kernel/m/Read/ReadVariableOp3Adam/my_model_9/dense_27/bias/m/Read/ReadVariableOp5Adam/my_model_9/dense_28/kernel/m/Read/ReadVariableOp3Adam/my_model_9/dense_28/bias/m/Read/ReadVariableOp5Adam/my_model_9/dense_29/kernel/m/Read/ReadVariableOp3Adam/my_model_9/dense_29/bias/m/Read/ReadVariableOp6Adam/my_model_9/conv2d_27/kernel/v/Read/ReadVariableOp4Adam/my_model_9/conv2d_27/bias/v/Read/ReadVariableOp6Adam/my_model_9/conv2d_28/kernel/v/Read/ReadVariableOp4Adam/my_model_9/conv2d_28/bias/v/Read/ReadVariableOp6Adam/my_model_9/conv2d_29/kernel/v/Read/ReadVariableOp4Adam/my_model_9/conv2d_29/bias/v/Read/ReadVariableOp5Adam/my_model_9/dense_27/kernel/v/Read/ReadVariableOp3Adam/my_model_9/dense_27/bias/v/Read/ReadVariableOp5Adam/my_model_9/dense_28/kernel/v/Read/ReadVariableOp3Adam/my_model_9/dense_28/bias/v/Read/ReadVariableOp5Adam/my_model_9/dense_29/kernel/v/Read/ReadVariableOp3Adam/my_model_9/dense_29/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_6533
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model_9/conv2d_27/kernelmy_model_9/conv2d_27/biasmy_model_9/conv2d_28/kernelmy_model_9/conv2d_28/biasmy_model_9/conv2d_29/kernelmy_model_9/conv2d_29/biasmy_model_9/dense_27/kernelmy_model_9/dense_27/biasmy_model_9/dense_28/kernelmy_model_9/dense_28/biasmy_model_9/dense_29/kernelmy_model_9/dense_29/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1"Adam/my_model_9/conv2d_27/kernel/m Adam/my_model_9/conv2d_27/bias/m"Adam/my_model_9/conv2d_28/kernel/m Adam/my_model_9/conv2d_28/bias/m"Adam/my_model_9/conv2d_29/kernel/m Adam/my_model_9/conv2d_29/bias/m!Adam/my_model_9/dense_27/kernel/mAdam/my_model_9/dense_27/bias/m!Adam/my_model_9/dense_28/kernel/mAdam/my_model_9/dense_28/bias/m!Adam/my_model_9/dense_29/kernel/mAdam/my_model_9/dense_29/bias/m"Adam/my_model_9/conv2d_27/kernel/v Adam/my_model_9/conv2d_27/bias/v"Adam/my_model_9/conv2d_28/kernel/v Adam/my_model_9/conv2d_28/bias/v"Adam/my_model_9/conv2d_29/kernel/v Adam/my_model_9/conv2d_29/bias/v!Adam/my_model_9/dense_27/kernel/vAdam/my_model_9/dense_27/bias/v!Adam/my_model_9/dense_28/kernel/vAdam/my_model_9/dense_28/bias/v!Adam/my_model_9/dense_29/kernel/vAdam/my_model_9/dense_29/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_6680Ɏ
�
b
D__inference_dropout_27_layer_call_and_return_conditional_losses_5627

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�7
�
D__inference_my_model_9_layer_call_and_return_conditional_losses_5916
x
conv2d_27_5878
conv2d_27_5880
conv2d_28_5885
conv2d_28_5887
conv2d_29_5892
conv2d_29_5894
dense_27_5900
dense_27_5902
dense_28_5905
dense_28_5907
dense_29_5910
dense_29_5912
identity��!conv2d_27/StatefulPartitionedCall�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallxconv2d_27_5878conv2d_27_5880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_27_layer_call_and_return_conditional_losses_55362#
!conv2d_27/StatefulPartitionedCall�
re_lu_27/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_27_layer_call_and_return_conditional_losses_56022
re_lu_27/PartitionedCall�
dropout_27/PartitionedCallPartitionedCall!re_lu_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_27_layer_call_and_return_conditional_losses_56272
dropout_27/PartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0conv2d_28_5885conv2d_28_5887*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_28_layer_call_and_return_conditional_losses_55572#
!conv2d_28/StatefulPartitionedCall�
re_lu_28/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_28_layer_call_and_return_conditional_losses_56502
re_lu_28/PartitionedCall�
dropout_28/PartitionedCallPartitionedCall!re_lu_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_28_layer_call_and_return_conditional_losses_56752
dropout_28/PartitionedCall�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_29_5892conv2d_29_5894*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_29_layer_call_and_return_conditional_losses_55782#
!conv2d_29/StatefulPartitionedCall�
re_lu_29/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_29_layer_call_and_return_conditional_losses_56982
re_lu_29/PartitionedCall�
dropout_29/PartitionedCallPartitionedCall!re_lu_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_29_layer_call_and_return_conditional_losses_57232
dropout_29/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall#dropout_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_9_layer_call_and_return_conditional_losses_57422
flatten_9/PartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_27_5900dense_27_5902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_27_layer_call_and_return_conditional_losses_57612"
 dense_27/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_5905dense_28_5907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_28_layer_call_and_return_conditional_losses_57882"
 dense_28/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_5910dense_29_5912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_29_layer_call_and_return_conditional_losses_58142"
 dense_29/StatefulPartitionedCall�
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex:

_output_shapes
: :

_output_shapes
: :
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
: 
�<
�
D__inference_my_model_9_layer_call_and_return_conditional_losses_5831
input_1
conv2d_27_5592
conv2d_27_5594
conv2d_28_5640
conv2d_28_5642
conv2d_29_5688
conv2d_29_5690
dense_27_5772
dense_27_5774
dense_28_5799
dense_28_5801
dense_29_5825
dense_29_5827
identity��!conv2d_27/StatefulPartitionedCall�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�"dropout_27/StatefulPartitionedCall�"dropout_28/StatefulPartitionedCall�"dropout_29/StatefulPartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_27_5592conv2d_27_5594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_27_layer_call_and_return_conditional_losses_55362#
!conv2d_27/StatefulPartitionedCall�
re_lu_27/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_27_layer_call_and_return_conditional_losses_56022
re_lu_27/PartitionedCall�
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall!re_lu_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_27_layer_call_and_return_conditional_losses_56222$
"dropout_27/StatefulPartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0conv2d_28_5640conv2d_28_5642*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_28_layer_call_and_return_conditional_losses_55572#
!conv2d_28/StatefulPartitionedCall�
re_lu_28/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_28_layer_call_and_return_conditional_losses_56502
re_lu_28/PartitionedCall�
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall!re_lu_28/PartitionedCall:output:0#^dropout_27/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_28_layer_call_and_return_conditional_losses_56702$
"dropout_28/StatefulPartitionedCall�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0conv2d_29_5688conv2d_29_5690*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_29_layer_call_and_return_conditional_losses_55782#
!conv2d_29/StatefulPartitionedCall�
re_lu_29/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_29_layer_call_and_return_conditional_losses_56982
re_lu_29/PartitionedCall�
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall!re_lu_29/PartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_29_layer_call_and_return_conditional_losses_57182$
"dropout_29/StatefulPartitionedCall�
flatten_9/PartitionedCallPartitionedCall+dropout_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_9_layer_call_and_return_conditional_losses_57422
flatten_9/PartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_27_5772dense_27_5774*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_27_layer_call_and_return_conditional_losses_57612"
 dense_27/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_5799dense_28_5801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_28_layer_call_and_return_conditional_losses_57882"
 dense_28/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_5825dense_29_5827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_29_layer_call_and_return_conditional_losses_58142"
 dense_29/StatefulPartitionedCall�
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :
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
: 
�
c
D__inference_dropout_29_layer_call_and_return_conditional_losses_5718

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_my_model_9_layer_call_fn_6190
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_my_model_9_layer_call_and_return_conditional_losses_59162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex:

_output_shapes
: :

_output_shapes
: :
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
: 
�
E
)__inference_dropout_29_layer_call_fn_6301

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_29_layer_call_and_return_conditional_losses_57232
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_dropout_28_layer_call_fn_6264

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_28_layer_call_and_return_conditional_losses_56752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�7
�
D__inference_my_model_9_layer_call_and_return_conditional_losses_5872
input_1
conv2d_27_5834
conv2d_27_5836
conv2d_28_5841
conv2d_28_5843
conv2d_29_5848
conv2d_29_5850
dense_27_5856
dense_27_5858
dense_28_5861
dense_28_5863
dense_29_5866
dense_29_5868
identity��!conv2d_27/StatefulPartitionedCall�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_27_5834conv2d_27_5836*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_27_layer_call_and_return_conditional_losses_55362#
!conv2d_27/StatefulPartitionedCall�
re_lu_27/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_27_layer_call_and_return_conditional_losses_56022
re_lu_27/PartitionedCall�
dropout_27/PartitionedCallPartitionedCall!re_lu_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_27_layer_call_and_return_conditional_losses_56272
dropout_27/PartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0conv2d_28_5841conv2d_28_5843*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_28_layer_call_and_return_conditional_losses_55572#
!conv2d_28/StatefulPartitionedCall�
re_lu_28/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_28_layer_call_and_return_conditional_losses_56502
re_lu_28/PartitionedCall�
dropout_28/PartitionedCallPartitionedCall!re_lu_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_28_layer_call_and_return_conditional_losses_56752
dropout_28/PartitionedCall�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_29_5848conv2d_29_5850*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_29_layer_call_and_return_conditional_losses_55782#
!conv2d_29/StatefulPartitionedCall�
re_lu_29/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_29_layer_call_and_return_conditional_losses_56982
re_lu_29/PartitionedCall�
dropout_29/PartitionedCallPartitionedCall!re_lu_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_29_layer_call_and_return_conditional_losses_57232
dropout_29/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall#dropout_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_9_layer_call_and_return_conditional_losses_57422
flatten_9/PartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_27_5856dense_27_5858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_27_layer_call_and_return_conditional_losses_57612"
 dense_27/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_5861dense_28_5863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_28_layer_call_and_return_conditional_losses_57882"
 dense_28/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_5866dense_29_5868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_29_layer_call_and_return_conditional_losses_58142"
 dense_29/StatefulPartitionedCall�
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :
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
: 
�	
�
C__inference_conv2d_28_layer_call_and_return_conditional_losses_5557

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
B__inference_dense_28_layer_call_and_return_conditional_losses_5788

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
"__inference_signature_wrapper_6011
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_55252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :
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
: 
�
C
'__inference_re_lu_29_layer_call_fn_6274

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_29_layer_call_and_return_conditional_losses_56982
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
)__inference_dropout_27_layer_call_fn_6222

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_27_layer_call_and_return_conditional_losses_56222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
B__inference_dense_27_layer_call_and_return_conditional_losses_5761

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:::Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�f
�
__inference__traced_save_6533
file_prefix:
6savev2_my_model_9_conv2d_27_kernel_read_readvariableop8
4savev2_my_model_9_conv2d_27_bias_read_readvariableop:
6savev2_my_model_9_conv2d_28_kernel_read_readvariableop8
4savev2_my_model_9_conv2d_28_bias_read_readvariableop:
6savev2_my_model_9_conv2d_29_kernel_read_readvariableop8
4savev2_my_model_9_conv2d_29_bias_read_readvariableop9
5savev2_my_model_9_dense_27_kernel_read_readvariableop7
3savev2_my_model_9_dense_27_bias_read_readvariableop9
5savev2_my_model_9_dense_28_kernel_read_readvariableop7
3savev2_my_model_9_dense_28_bias_read_readvariableop9
5savev2_my_model_9_dense_29_kernel_read_readvariableop7
3savev2_my_model_9_dense_29_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_my_model_9_conv2d_27_kernel_m_read_readvariableop?
;savev2_adam_my_model_9_conv2d_27_bias_m_read_readvariableopA
=savev2_adam_my_model_9_conv2d_28_kernel_m_read_readvariableop?
;savev2_adam_my_model_9_conv2d_28_bias_m_read_readvariableopA
=savev2_adam_my_model_9_conv2d_29_kernel_m_read_readvariableop?
;savev2_adam_my_model_9_conv2d_29_bias_m_read_readvariableop@
<savev2_adam_my_model_9_dense_27_kernel_m_read_readvariableop>
:savev2_adam_my_model_9_dense_27_bias_m_read_readvariableop@
<savev2_adam_my_model_9_dense_28_kernel_m_read_readvariableop>
:savev2_adam_my_model_9_dense_28_bias_m_read_readvariableop@
<savev2_adam_my_model_9_dense_29_kernel_m_read_readvariableop>
:savev2_adam_my_model_9_dense_29_bias_m_read_readvariableopA
=savev2_adam_my_model_9_conv2d_27_kernel_v_read_readvariableop?
;savev2_adam_my_model_9_conv2d_27_bias_v_read_readvariableopA
=savev2_adam_my_model_9_conv2d_28_kernel_v_read_readvariableop?
;savev2_adam_my_model_9_conv2d_28_bias_v_read_readvariableopA
=savev2_adam_my_model_9_conv2d_29_kernel_v_read_readvariableop?
;savev2_adam_my_model_9_conv2d_29_bias_v_read_readvariableop@
<savev2_adam_my_model_9_dense_27_kernel_v_read_readvariableop>
:savev2_adam_my_model_9_dense_27_bias_v_read_readvariableop@
<savev2_adam_my_model_9_dense_28_kernel_v_read_readvariableop>
:savev2_adam_my_model_9_dense_28_bias_v_read_readvariableop@
<savev2_adam_my_model_9_dense_29_kernel_v_read_readvariableop>
:savev2_adam_my_model_9_dense_29_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_94ad4fc7f0aa46f7bf150d363d8804cc/part2	
Const_1�
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_my_model_9_conv2d_27_kernel_read_readvariableop4savev2_my_model_9_conv2d_27_bias_read_readvariableop6savev2_my_model_9_conv2d_28_kernel_read_readvariableop4savev2_my_model_9_conv2d_28_bias_read_readvariableop6savev2_my_model_9_conv2d_29_kernel_read_readvariableop4savev2_my_model_9_conv2d_29_bias_read_readvariableop5savev2_my_model_9_dense_27_kernel_read_readvariableop3savev2_my_model_9_dense_27_bias_read_readvariableop5savev2_my_model_9_dense_28_kernel_read_readvariableop3savev2_my_model_9_dense_28_bias_read_readvariableop5savev2_my_model_9_dense_29_kernel_read_readvariableop3savev2_my_model_9_dense_29_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_my_model_9_conv2d_27_kernel_m_read_readvariableop;savev2_adam_my_model_9_conv2d_27_bias_m_read_readvariableop=savev2_adam_my_model_9_conv2d_28_kernel_m_read_readvariableop;savev2_adam_my_model_9_conv2d_28_bias_m_read_readvariableop=savev2_adam_my_model_9_conv2d_29_kernel_m_read_readvariableop;savev2_adam_my_model_9_conv2d_29_bias_m_read_readvariableop<savev2_adam_my_model_9_dense_27_kernel_m_read_readvariableop:savev2_adam_my_model_9_dense_27_bias_m_read_readvariableop<savev2_adam_my_model_9_dense_28_kernel_m_read_readvariableop:savev2_adam_my_model_9_dense_28_bias_m_read_readvariableop<savev2_adam_my_model_9_dense_29_kernel_m_read_readvariableop:savev2_adam_my_model_9_dense_29_bias_m_read_readvariableop=savev2_adam_my_model_9_conv2d_27_kernel_v_read_readvariableop;savev2_adam_my_model_9_conv2d_27_bias_v_read_readvariableop=savev2_adam_my_model_9_conv2d_28_kernel_v_read_readvariableop;savev2_adam_my_model_9_conv2d_28_bias_v_read_readvariableop=savev2_adam_my_model_9_conv2d_29_kernel_v_read_readvariableop;savev2_adam_my_model_9_conv2d_29_bias_v_read_readvariableop<savev2_adam_my_model_9_dense_27_kernel_v_read_readvariableop:savev2_adam_my_model_9_dense_27_bias_v_read_readvariableop<savev2_adam_my_model_9_dense_28_kernel_v_read_readvariableop:savev2_adam_my_model_9_dense_28_bias_v_read_readvariableop<savev2_adam_my_model_9_dense_29_kernel_v_read_readvariableop:savev2_adam_my_model_9_dense_29_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : @:@:@�:�:
��@:@:@@:@:@
:
: : : : : : : : : : : : @:@:@�:�:
��@:@:@@:@:@
:
: : : @:@:@�:�:
��@:@:@@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$  

_output_shapes

:@
: !

_output_shapes
:
:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@�:!'

_output_shapes	
:�:&("
 
_output_shapes
:
��@: )

_output_shapes
:@:$* 

_output_shapes

:@@: +

_output_shapes
:@:$, 

_output_shapes

:@
: -

_output_shapes
:
:.

_output_shapes
: 
�
�
)__inference_my_model_9_layer_call_fn_5972
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_my_model_9_layer_call_and_return_conditional_losses_59162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :
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
: 
�
E
)__inference_dropout_27_layer_call_fn_6227

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_27_layer_call_and_return_conditional_losses_56272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
_
C__inference_flatten_9_layer_call_and_return_conditional_losses_5742

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� R 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_dense_29_layer_call_and_return_conditional_losses_5814

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�	
�
C__inference_conv2d_27_layer_call_and_return_conditional_losses_5536

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�6
�
D__inference_my_model_9_layer_call_and_return_conditional_losses_6132
x,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identity��
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2Dx'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_27/BiasAdd|
re_lu_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
re_lu_27/Relu�
dropout_27/IdentityIdentityre_lu_27/Relu:activations:0*
T0*/
_output_shapes
:��������� 2
dropout_27/Identity�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2DConv2Ddropout_27/Identity:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d_28/Conv2D�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_28/BiasAdd|
re_lu_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
re_lu_28/Relu�
dropout_28/IdentityIdentityre_lu_28/Relu:activations:0*
T0*/
_output_shapes
:���������@2
dropout_28/Identity�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2DConv2Ddropout_28/Identity:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_29/Conv2D�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_29/BiasAdd}
re_lu_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
re_lu_29/Relu�
dropout_29/IdentityIdentityre_lu_29/Relu:activations:0*
T0*0
_output_shapes
:����������2
dropout_29/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� R 2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_29/Identity:output:0flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_9/Reshape�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_27/Relu�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_28/Relu�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_29/BiasAddm
IdentityIdentitydense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  :::::::::::::R N
/
_output_shapes
:���������  

_user_specified_namex:

_output_shapes
: :

_output_shapes
: :
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
: 
�
}
(__inference_conv2d_27_layer_call_fn_5546

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_27_layer_call_and_return_conditional_losses_55362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
C
'__inference_re_lu_28_layer_call_fn_6237

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_28_layer_call_and_return_conditional_losses_56502
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_my_model_9_layer_call_fn_6161
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_my_model_9_layer_call_and_return_conditional_losses_59162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex:

_output_shapes
: :

_output_shapes
: :
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
: 
�
b
D__inference_dropout_28_layer_call_and_return_conditional_losses_5675

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
^
B__inference_re_lu_27_layer_call_and_return_conditional_losses_5602

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:��������� 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
}
(__inference_conv2d_29_layer_call_fn_5588

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_29_layer_call_and_return_conditional_losses_55782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
^
B__inference_re_lu_29_layer_call_and_return_conditional_losses_5698

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:����������2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_dense_28_layer_call_and_return_conditional_losses_6343

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
^
B__inference_re_lu_29_layer_call_and_return_conditional_losses_6269

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:����������2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_re_lu_28_layer_call_and_return_conditional_losses_5650

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
 __inference__traced_restore_6680
file_prefix0
,assignvariableop_my_model_9_conv2d_27_kernel0
,assignvariableop_1_my_model_9_conv2d_27_bias2
.assignvariableop_2_my_model_9_conv2d_28_kernel0
,assignvariableop_3_my_model_9_conv2d_28_bias2
.assignvariableop_4_my_model_9_conv2d_29_kernel0
,assignvariableop_5_my_model_9_conv2d_29_bias1
-assignvariableop_6_my_model_9_dense_27_kernel/
+assignvariableop_7_my_model_9_dense_27_bias1
-assignvariableop_8_my_model_9_dense_28_kernel/
+assignvariableop_9_my_model_9_dense_28_bias2
.assignvariableop_10_my_model_9_dense_29_kernel0
,assignvariableop_11_my_model_9_dense_29_bias
assignvariableop_12_beta_1
assignvariableop_13_beta_2
assignvariableop_14_decay%
!assignvariableop_15_learning_rate!
assignvariableop_16_adam_iter
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1:
6assignvariableop_21_adam_my_model_9_conv2d_27_kernel_m8
4assignvariableop_22_adam_my_model_9_conv2d_27_bias_m:
6assignvariableop_23_adam_my_model_9_conv2d_28_kernel_m8
4assignvariableop_24_adam_my_model_9_conv2d_28_bias_m:
6assignvariableop_25_adam_my_model_9_conv2d_29_kernel_m8
4assignvariableop_26_adam_my_model_9_conv2d_29_bias_m9
5assignvariableop_27_adam_my_model_9_dense_27_kernel_m7
3assignvariableop_28_adam_my_model_9_dense_27_bias_m9
5assignvariableop_29_adam_my_model_9_dense_28_kernel_m7
3assignvariableop_30_adam_my_model_9_dense_28_bias_m9
5assignvariableop_31_adam_my_model_9_dense_29_kernel_m7
3assignvariableop_32_adam_my_model_9_dense_29_bias_m:
6assignvariableop_33_adam_my_model_9_conv2d_27_kernel_v8
4assignvariableop_34_adam_my_model_9_conv2d_27_bias_v:
6assignvariableop_35_adam_my_model_9_conv2d_28_kernel_v8
4assignvariableop_36_adam_my_model_9_conv2d_28_bias_v:
6assignvariableop_37_adam_my_model_9_conv2d_29_kernel_v8
4assignvariableop_38_adam_my_model_9_conv2d_29_bias_v9
5assignvariableop_39_adam_my_model_9_dense_27_kernel_v7
3assignvariableop_40_adam_my_model_9_dense_27_bias_v9
5assignvariableop_41_adam_my_model_9_dense_28_kernel_v7
3assignvariableop_42_adam_my_model_9_dense_28_bias_v9
5assignvariableop_43_adam_my_model_9_dense_29_kernel_v7
3assignvariableop_44_adam_my_model_9_dense_29_bias_v
identity_46��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp,assignvariableop_my_model_9_conv2d_27_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp,assignvariableop_1_my_model_9_conv2d_27_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_my_model_9_conv2d_28_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_my_model_9_conv2d_28_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_my_model_9_conv2d_29_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_my_model_9_conv2d_29_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_my_model_9_dense_27_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_my_model_9_dense_27_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_my_model_9_dense_28_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_my_model_9_dense_28_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp.assignvariableop_10_my_model_9_dense_29_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp,assignvariableop_11_my_model_9_dense_29_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_beta_1Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_beta_2Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_decayIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0	*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_my_model_9_conv2d_27_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_my_model_9_conv2d_27_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_my_model_9_conv2d_28_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_my_model_9_conv2d_28_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_my_model_9_conv2d_29_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_my_model_9_conv2d_29_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_my_model_9_dense_27_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_my_model_9_dense_27_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp5assignvariableop_29_adam_my_model_9_dense_28_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_my_model_9_dense_28_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp5assignvariableop_31_adam_my_model_9_dense_29_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_my_model_9_dense_29_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_my_model_9_conv2d_27_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_my_model_9_conv2d_27_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_my_model_9_conv2d_28_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_my_model_9_conv2d_28_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_my_model_9_conv2d_29_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adam_my_model_9_conv2d_29_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_my_model_9_dense_27_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_my_model_9_dense_27_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adam_my_model_9_dense_28_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp3assignvariableop_42_adam_my_model_9_dense_28_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp5assignvariableop_43_adam_my_model_9_dense_29_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp3assignvariableop_44_adam_my_model_9_dense_29_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45�
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: 
�
b
D__inference_dropout_29_layer_call_and_return_conditional_losses_5723

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
|
'__inference_dense_29_layer_call_fn_6371

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_29_layer_call_and_return_conditional_losses_58142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
D
(__inference_flatten_9_layer_call_fn_6312

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_9_layer_call_and_return_conditional_losses_57422
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
D__inference_dropout_27_layer_call_and_return_conditional_losses_5622

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
c
D__inference_dropout_29_layer_call_and_return_conditional_losses_6286

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
D__inference_dropout_27_layer_call_and_return_conditional_losses_6212

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
^
B__inference_re_lu_27_layer_call_and_return_conditional_losses_6195

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:��������� 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
c
D__inference_dropout_28_layer_call_and_return_conditional_losses_5670

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
|
'__inference_dense_27_layer_call_fn_6332

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_27_layer_call_and_return_conditional_losses_57612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
b
D__inference_dropout_29_layer_call_and_return_conditional_losses_6291

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_27_layer_call_and_return_conditional_losses_6217

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
}
(__inference_conv2d_28_layer_call_fn_5567

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_28_layer_call_and_return_conditional_losses_55572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
_
C__inference_flatten_9_layer_call_and_return_conditional_losses_6307

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� R 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_dense_29_layer_call_and_return_conditional_losses_6362

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_my_model_9_layer_call_fn_5943
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_my_model_9_layer_call_and_return_conditional_losses_59162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :
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
: 
�
c
D__inference_dropout_28_layer_call_and_return_conditional_losses_6249

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_conv2d_29_layer_call_and_return_conditional_losses_5578

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@:::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
B__inference_dense_27_layer_call_and_return_conditional_losses_6323

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:::Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�S
�
D__inference_my_model_9_layer_call_and_return_conditional_losses_6082
x,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identity��
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2Dx'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_27/BiasAdd|
re_lu_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
re_lu_27/Reluy
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_27/dropout/Const�
dropout_27/dropout/MulMulre_lu_27/Relu:activations:0!dropout_27/dropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout_27/dropout/Mul
dropout_27/dropout/ShapeShapere_lu_27/Relu:activations:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape�
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype021
/dropout_27/dropout/random_uniform/RandomUniform�
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_27/dropout/GreaterEqual/y�
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2!
dropout_27/dropout/GreaterEqual�
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout_27/dropout/Cast�
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout_27/dropout/Mul_1�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2DConv2Ddropout_27/dropout/Mul_1:z:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d_28/Conv2D�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_28/BiasAdd|
re_lu_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
re_lu_28/Reluy
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_28/dropout/Const�
dropout_28/dropout/MulMulre_lu_28/Relu:activations:0!dropout_28/dropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShapere_lu_28/Relu:activations:0*
T0*
_output_shapes
:2
dropout_28/dropout/Shape�
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype021
/dropout_28/dropout/random_uniform/RandomUniform�
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_28/dropout/GreaterEqual/y�
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2!
dropout_28/dropout/GreaterEqual�
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout_28/dropout/Cast�
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout_28/dropout/Mul_1�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2DConv2Ddropout_28/dropout/Mul_1:z:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_29/Conv2D�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_29/BiasAdd}
re_lu_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
re_lu_29/Reluy
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_29/dropout/Const�
dropout_29/dropout/MulMulre_lu_29/Relu:activations:0!dropout_29/dropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout_29/dropout/Mul
dropout_29/dropout/ShapeShapere_lu_29/Relu:activations:0*
T0*
_output_shapes
:2
dropout_29/dropout/Shape�
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype021
/dropout_29/dropout/random_uniform/RandomUniform�
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_29/dropout/GreaterEqual/y�
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2!
dropout_29/dropout/GreaterEqual�
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout_29/dropout/Cast�
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout_29/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� R 2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_29/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_9/Reshape�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_27/Relu�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_28/Relu�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_29/BiasAddm
IdentityIdentitydense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  :::::::::::::R N
/
_output_shapes
:���������  

_user_specified_namex:

_output_shapes
: :

_output_shapes
: :
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
: 
�
b
)__inference_dropout_29_layer_call_fn_6296

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_29_layer_call_and_return_conditional_losses_57182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_re_lu_28_layer_call_and_return_conditional_losses_6232

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
C
'__inference_re_lu_27_layer_call_fn_6200

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_re_lu_27_layer_call_and_return_conditional_losses_56022
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
|
'__inference_dense_28_layer_call_fn_6352

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_28_layer_call_and_return_conditional_losses_57882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�@
�
__inference__wrapped_model_5525
input_17
3my_model_9_conv2d_27_conv2d_readvariableop_resource8
4my_model_9_conv2d_27_biasadd_readvariableop_resource7
3my_model_9_conv2d_28_conv2d_readvariableop_resource8
4my_model_9_conv2d_28_biasadd_readvariableop_resource7
3my_model_9_conv2d_29_conv2d_readvariableop_resource8
4my_model_9_conv2d_29_biasadd_readvariableop_resource6
2my_model_9_dense_27_matmul_readvariableop_resource7
3my_model_9_dense_27_biasadd_readvariableop_resource6
2my_model_9_dense_28_matmul_readvariableop_resource7
3my_model_9_dense_28_biasadd_readvariableop_resource6
2my_model_9_dense_29_matmul_readvariableop_resource7
3my_model_9_dense_29_biasadd_readvariableop_resource
identity��
*my_model_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp3my_model_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*my_model_9/conv2d_27/Conv2D/ReadVariableOp�
my_model_9/conv2d_27/Conv2DConv2Dinput_12my_model_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
my_model_9/conv2d_27/Conv2D�
+my_model_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp4my_model_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+my_model_9/conv2d_27/BiasAdd/ReadVariableOp�
my_model_9/conv2d_27/BiasAddBiasAdd$my_model_9/conv2d_27/Conv2D:output:03my_model_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
my_model_9/conv2d_27/BiasAdd�
my_model_9/re_lu_27/ReluRelu%my_model_9/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
my_model_9/re_lu_27/Relu�
my_model_9/dropout_27/IdentityIdentity&my_model_9/re_lu_27/Relu:activations:0*
T0*/
_output_shapes
:��������� 2 
my_model_9/dropout_27/Identity�
*my_model_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp3my_model_9_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*my_model_9/conv2d_28/Conv2D/ReadVariableOp�
my_model_9/conv2d_28/Conv2DConv2D'my_model_9/dropout_27/Identity:output:02my_model_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
my_model_9/conv2d_28/Conv2D�
+my_model_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp4my_model_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+my_model_9/conv2d_28/BiasAdd/ReadVariableOp�
my_model_9/conv2d_28/BiasAddBiasAdd$my_model_9/conv2d_28/Conv2D:output:03my_model_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
my_model_9/conv2d_28/BiasAdd�
my_model_9/re_lu_28/ReluRelu%my_model_9/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
my_model_9/re_lu_28/Relu�
my_model_9/dropout_28/IdentityIdentity&my_model_9/re_lu_28/Relu:activations:0*
T0*/
_output_shapes
:���������@2 
my_model_9/dropout_28/Identity�
*my_model_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp3my_model_9_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02,
*my_model_9/conv2d_29/Conv2D/ReadVariableOp�
my_model_9/conv2d_29/Conv2DConv2D'my_model_9/dropout_28/Identity:output:02my_model_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
my_model_9/conv2d_29/Conv2D�
+my_model_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp4my_model_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+my_model_9/conv2d_29/BiasAdd/ReadVariableOp�
my_model_9/conv2d_29/BiasAddBiasAdd$my_model_9/conv2d_29/Conv2D:output:03my_model_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
my_model_9/conv2d_29/BiasAdd�
my_model_9/re_lu_29/ReluRelu%my_model_9/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
my_model_9/re_lu_29/Relu�
my_model_9/dropout_29/IdentityIdentity&my_model_9/re_lu_29/Relu:activations:0*
T0*0
_output_shapes
:����������2 
my_model_9/dropout_29/Identity�
my_model_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� R 2
my_model_9/flatten_9/Const�
my_model_9/flatten_9/ReshapeReshape'my_model_9/dropout_29/Identity:output:0#my_model_9/flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2
my_model_9/flatten_9/Reshape�
)my_model_9/dense_27/MatMul/ReadVariableOpReadVariableOp2my_model_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
��@*
dtype02+
)my_model_9/dense_27/MatMul/ReadVariableOp�
my_model_9/dense_27/MatMulMatMul%my_model_9/flatten_9/Reshape:output:01my_model_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
my_model_9/dense_27/MatMul�
*my_model_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp3my_model_9_dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*my_model_9/dense_27/BiasAdd/ReadVariableOp�
my_model_9/dense_27/BiasAddBiasAdd$my_model_9/dense_27/MatMul:product:02my_model_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
my_model_9/dense_27/BiasAdd�
my_model_9/dense_27/ReluRelu$my_model_9/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
my_model_9/dense_27/Relu�
)my_model_9/dense_28/MatMul/ReadVariableOpReadVariableOp2my_model_9_dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02+
)my_model_9/dense_28/MatMul/ReadVariableOp�
my_model_9/dense_28/MatMulMatMul&my_model_9/dense_27/Relu:activations:01my_model_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
my_model_9/dense_28/MatMul�
*my_model_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp3my_model_9_dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*my_model_9/dense_28/BiasAdd/ReadVariableOp�
my_model_9/dense_28/BiasAddBiasAdd$my_model_9/dense_28/MatMul:product:02my_model_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
my_model_9/dense_28/BiasAdd�
my_model_9/dense_28/ReluRelu$my_model_9/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
my_model_9/dense_28/Relu�
)my_model_9/dense_29/MatMul/ReadVariableOpReadVariableOp2my_model_9_dense_29_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02+
)my_model_9/dense_29/MatMul/ReadVariableOp�
my_model_9/dense_29/MatMulMatMul&my_model_9/dense_28/Relu:activations:01my_model_9/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
my_model_9/dense_29/MatMul�
*my_model_9/dense_29/BiasAdd/ReadVariableOpReadVariableOp3my_model_9_dense_29_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02,
*my_model_9/dense_29/BiasAdd/ReadVariableOp�
my_model_9/dense_29/BiasAddBiasAdd$my_model_9/dense_29/MatMul:product:02my_model_9/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
my_model_9/dense_29/BiasAddx
IdentityIdentity$my_model_9/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������  :::::::::::::X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :
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
: 
�
b
)__inference_dropout_28_layer_call_fn_6259

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_28_layer_call_and_return_conditional_losses_56702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
b
D__inference_dropout_28_layer_call_and_return_conditional_losses_6254

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������  <
output_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�	
	conv1
	relu1
	drop1
	conv2
	relu2
	drop2
	conv3
	relu3
		drop3

flatten
fc1
fc2
fc3
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�
_tf_keras_model�{"class_name": "MyModel", "name": "my_model_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "config": {"name": "conv2d_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
�
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "re_lu_27", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
trainable_variables
	variables
 regularization_losses
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�	

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 32]}}
�
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "re_lu_28", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
,trainable_variables
-	variables
.regularization_losses
/	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�	

0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
�
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "re_lu_29", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_29", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 86528}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 86528]}}
�

Hkernel
Ibias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�

Nkernel
Obias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate
Xiterm�m�"m�#m�0m�1m�Bm�Cm�Hm�Im�Nm�Om�v�v�"v�#v�0v�1v�Bv�Cv�Hv�Iv�Nv�Ov�"
	optimizer
v
0
1
"2
#3
04
15
B6
C7
H8
I9
N10
O11"
trackable_list_wrapper
v
0
1
"2
#3
04
15
B6
C7
H8
I9
N10
O11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ylayer_regularization_losses
Zmetrics
trainable_variables
	variables
[non_trainable_variables
regularization_losses
\layer_metrics

]layers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
5:3 2my_model_9/conv2d_27/kernel
':% 2my_model_9/conv2d_27/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
^layer_regularization_losses
_metrics
trainable_variables
	variables
`non_trainable_variables
regularization_losses
alayer_metrics

blayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
clayer_regularization_losses
dmetrics
trainable_variables
	variables
enon_trainable_variables
regularization_losses
flayer_metrics

glayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
hlayer_regularization_losses
imetrics
trainable_variables
	variables
jnon_trainable_variables
 regularization_losses
klayer_metrics

llayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
5:3 @2my_model_9/conv2d_28/kernel
':%@2my_model_9/conv2d_28/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mlayer_regularization_losses
nmetrics
$trainable_variables
%	variables
onon_trainable_variables
&regularization_losses
player_metrics

qlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rlayer_regularization_losses
smetrics
(trainable_variables
)	variables
tnon_trainable_variables
*regularization_losses
ulayer_metrics

vlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wlayer_regularization_losses
xmetrics
,trainable_variables
-	variables
ynon_trainable_variables
.regularization_losses
zlayer_metrics

{layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
6:4@�2my_model_9/conv2d_29/kernel
(:&�2my_model_9/conv2d_29/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|layer_regularization_losses
}metrics
2trainable_variables
3	variables
~non_trainable_variables
4regularization_losses
layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�metrics
6trainable_variables
7	variables
�non_trainable_variables
8regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�metrics
:trainable_variables
;	variables
�non_trainable_variables
<regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�metrics
>trainable_variables
?	variables
�non_trainable_variables
@regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,
��@2my_model_9/dense_27/kernel
&:$@2my_model_9/dense_27/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�metrics
Dtrainable_variables
E	variables
�non_trainable_variables
Fregularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*@@2my_model_9/dense_28/kernel
&:$@2my_model_9/dense_28/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�metrics
Jtrainable_variables
K	variables
�non_trainable_variables
Lregularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*@
2my_model_9/dense_29/kernel
&:$
2my_model_9/dense_29/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�metrics
Ptrainable_variables
Q	variables
�non_trainable_variables
Rregularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
::8 2"Adam/my_model_9/conv2d_27/kernel/m
,:* 2 Adam/my_model_9/conv2d_27/bias/m
::8 @2"Adam/my_model_9/conv2d_28/kernel/m
,:*@2 Adam/my_model_9/conv2d_28/bias/m
;:9@�2"Adam/my_model_9/conv2d_29/kernel/m
-:+�2 Adam/my_model_9/conv2d_29/bias/m
3:1
��@2!Adam/my_model_9/dense_27/kernel/m
+:)@2Adam/my_model_9/dense_27/bias/m
1:/@@2!Adam/my_model_9/dense_28/kernel/m
+:)@2Adam/my_model_9/dense_28/bias/m
1:/@
2!Adam/my_model_9/dense_29/kernel/m
+:)
2Adam/my_model_9/dense_29/bias/m
::8 2"Adam/my_model_9/conv2d_27/kernel/v
,:* 2 Adam/my_model_9/conv2d_27/bias/v
::8 @2"Adam/my_model_9/conv2d_28/kernel/v
,:*@2 Adam/my_model_9/conv2d_28/bias/v
;:9@�2"Adam/my_model_9/conv2d_29/kernel/v
-:+�2 Adam/my_model_9/conv2d_29/bias/v
3:1
��@2!Adam/my_model_9/dense_27/kernel/v
+:)@2Adam/my_model_9/dense_27/bias/v
1:/@@2!Adam/my_model_9/dense_28/kernel/v
+:)@2Adam/my_model_9/dense_28/bias/v
1:/@
2!Adam/my_model_9/dense_29/kernel/v
+:)
2Adam/my_model_9/dense_29/bias/v
�2�
D__inference_my_model_9_layer_call_and_return_conditional_losses_6132
D__inference_my_model_9_layer_call_and_return_conditional_losses_5872
D__inference_my_model_9_layer_call_and_return_conditional_losses_6082
D__inference_my_model_9_layer_call_and_return_conditional_losses_5831�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_my_model_9_layer_call_fn_5943
)__inference_my_model_9_layer_call_fn_5972
)__inference_my_model_9_layer_call_fn_6161
)__inference_my_model_9_layer_call_fn_6190�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference__wrapped_model_5525�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������  
�2�
C__inference_conv2d_27_layer_call_and_return_conditional_losses_5536�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv2d_27_layer_call_fn_5546�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
B__inference_re_lu_27_layer_call_and_return_conditional_losses_6195�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_re_lu_27_layer_call_fn_6200�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dropout_27_layer_call_and_return_conditional_losses_6212
D__inference_dropout_27_layer_call_and_return_conditional_losses_6217�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_27_layer_call_fn_6227
)__inference_dropout_27_layer_call_fn_6222�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv2d_28_layer_call_and_return_conditional_losses_5557�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
(__inference_conv2d_28_layer_call_fn_5567�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
B__inference_re_lu_28_layer_call_and_return_conditional_losses_6232�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_re_lu_28_layer_call_fn_6237�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dropout_28_layer_call_and_return_conditional_losses_6249
D__inference_dropout_28_layer_call_and_return_conditional_losses_6254�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_28_layer_call_fn_6264
)__inference_dropout_28_layer_call_fn_6259�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv2d_29_layer_call_and_return_conditional_losses_5578�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
(__inference_conv2d_29_layer_call_fn_5588�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
B__inference_re_lu_29_layer_call_and_return_conditional_losses_6269�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_re_lu_29_layer_call_fn_6274�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dropout_29_layer_call_and_return_conditional_losses_6291
D__inference_dropout_29_layer_call_and_return_conditional_losses_6286�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_29_layer_call_fn_6296
)__inference_dropout_29_layer_call_fn_6301�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_flatten_9_layer_call_and_return_conditional_losses_6307�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_flatten_9_layer_call_fn_6312�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_27_layer_call_and_return_conditional_losses_6323�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_27_layer_call_fn_6332�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_28_layer_call_and_return_conditional_losses_6343�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_28_layer_call_fn_6352�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_29_layer_call_and_return_conditional_losses_6362�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_29_layer_call_fn_6371�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
1B/
"__inference_signature_wrapper_6011input_1�
__inference__wrapped_model_5525}"#01BCHINO8�5
.�+
)�&
input_1���������  
� "3�0
.
output_1"�
output_1���������
�
C__inference_conv2d_27_layer_call_and_return_conditional_losses_5536�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
(__inference_conv2d_27_layer_call_fn_5546�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
C__inference_conv2d_28_layer_call_and_return_conditional_losses_5557�"#I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
(__inference_conv2d_28_layer_call_fn_5567�"#I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
C__inference_conv2d_29_layer_call_and_return_conditional_losses_5578�01I�F
?�<
:�7
inputs+���������������������������@
� "@�=
6�3
0,����������������������������
� �
(__inference_conv2d_29_layer_call_fn_5588�01I�F
?�<
:�7
inputs+���������������������������@
� "3�0,�����������������������������
B__inference_dense_27_layer_call_and_return_conditional_losses_6323^BC1�.
'�$
"�
inputs�����������
� "%�"
�
0���������@
� |
'__inference_dense_27_layer_call_fn_6332QBC1�.
'�$
"�
inputs�����������
� "����������@�
B__inference_dense_28_layer_call_and_return_conditional_losses_6343\HI/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� z
'__inference_dense_28_layer_call_fn_6352OHI/�,
%�"
 �
inputs���������@
� "����������@�
B__inference_dense_29_layer_call_and_return_conditional_losses_6362\NO/�,
%�"
 �
inputs���������@
� "%�"
�
0���������

� z
'__inference_dense_29_layer_call_fn_6371ONO/�,
%�"
 �
inputs���������@
� "����������
�
D__inference_dropout_27_layer_call_and_return_conditional_losses_6212l;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
D__inference_dropout_27_layer_call_and_return_conditional_losses_6217l;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
)__inference_dropout_27_layer_call_fn_6222_;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
)__inference_dropout_27_layer_call_fn_6227_;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
D__inference_dropout_28_layer_call_and_return_conditional_losses_6249l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
D__inference_dropout_28_layer_call_and_return_conditional_losses_6254l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
)__inference_dropout_28_layer_call_fn_6259_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
)__inference_dropout_28_layer_call_fn_6264_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
D__inference_dropout_29_layer_call_and_return_conditional_losses_6286n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
D__inference_dropout_29_layer_call_and_return_conditional_losses_6291n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
)__inference_dropout_29_layer_call_fn_6296a<�9
2�/
)�&
inputs����������
p
� "!������������
)__inference_dropout_29_layer_call_fn_6301a<�9
2�/
)�&
inputs����������
p 
� "!������������
C__inference_flatten_9_layer_call_and_return_conditional_losses_6307c8�5
.�+
)�&
inputs����������
� "'�$
�
0�����������
� �
(__inference_flatten_9_layer_call_fn_6312V8�5
.�+
)�&
inputs����������
� "�������������
D__inference_my_model_9_layer_call_and_return_conditional_losses_5831s"#01BCHINO<�9
2�/
)�&
input_1���������  
p
� "%�"
�
0���������

� �
D__inference_my_model_9_layer_call_and_return_conditional_losses_5872s"#01BCHINO<�9
2�/
)�&
input_1���������  
p 
� "%�"
�
0���������

� �
D__inference_my_model_9_layer_call_and_return_conditional_losses_6082m"#01BCHINO6�3
,�)
#� 
x���������  
p
� "%�"
�
0���������

� �
D__inference_my_model_9_layer_call_and_return_conditional_losses_6132m"#01BCHINO6�3
,�)
#� 
x���������  
p 
� "%�"
�
0���������

� �
)__inference_my_model_9_layer_call_fn_5943f"#01BCHINO<�9
2�/
)�&
input_1���������  
p
� "����������
�
)__inference_my_model_9_layer_call_fn_5972f"#01BCHINO<�9
2�/
)�&
input_1���������  
p 
� "����������
�
)__inference_my_model_9_layer_call_fn_6161`"#01BCHINO6�3
,�)
#� 
x���������  
p
� "����������
�
)__inference_my_model_9_layer_call_fn_6190`"#01BCHINO6�3
,�)
#� 
x���������  
p 
� "����������
�
B__inference_re_lu_27_layer_call_and_return_conditional_losses_6195h7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
'__inference_re_lu_27_layer_call_fn_6200[7�4
-�*
(�%
inputs��������� 
� " ���������� �
B__inference_re_lu_28_layer_call_and_return_conditional_losses_6232h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
'__inference_re_lu_28_layer_call_fn_6237[7�4
-�*
(�%
inputs���������@
� " ����������@�
B__inference_re_lu_29_layer_call_and_return_conditional_losses_6269j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
'__inference_re_lu_29_layer_call_fn_6274]8�5
.�+
)�&
inputs����������
� "!������������
"__inference_signature_wrapper_6011�"#01BCHINOC�@
� 
9�6
4
input_1)�&
input_1���������  "3�0
.
output_1"�
output_1���������
