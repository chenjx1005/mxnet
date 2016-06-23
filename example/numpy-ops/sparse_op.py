# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np
import logging
from scipy.sparse import csr_matrix

class SparseInnerProduct(mx.operator.CustomOp):
    def __init__(self, m, num_hidden, no_bias):
        self.m = m
        self.num_hidden = num_hidden
        self.no_bias = no_bias
    
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0].asnumpy()
        indices = in_data[1].asnumpy().astype(np.int32)
        indptr = in_data[2].asnumpy().astype(np.int32)
        shape = (len(indptr)-1, self.m)
        x = csr_matrix((data,indices,indptr), shape=shape)        
        w = in_data[3].asnumpy()
        y = x.dot(w.T)
        if not self.no_bias:
            bias = in_data[4].asnumpy()
            y += bias
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = out_grad[0].asnumpy()
        data = in_data[0].asnumpy()
        indices = in_data[1].asnumpy()
        indptr = in_data[2].asnumpy()
        shape = (len(indptr)-1, self.m)
        x = csr_matrix((data,indices,indptr), shape=shape)
        gweight = x.transpose().dot(grad).T
        self.assign(in_grad[3], req[3], mx.nd.array(gweight))
        if not self.no_bias:
            self.assign(in_grad[4], req[4], mx.nd.sum(grad, axis=0))

@mx.operator.register("sparse_inner_product")
class SparseInnerProductProp(mx.operator.CustomOpProp):
    def __init__(self, m, num_hidden, no_bias):
        super(SparseInnerProductProp, self).__init__(need_top_grad=True)
        self.m = int(m)
        self.num_hidden = int(num_hidden)
        self.no_bias = no_bias in ('True',)
    
    def list_arguments(self):
        return ['data', 'indices', 'indptr', 'weight', 'bias']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        indices_shape = in_shape[1]
        indptr_shape = in_shape[2]
        weight_shape = (self.num_hidden, self.m)
        bias_shape = (self.num_hidden, )
        n = indptr_shape[0] - 1
        output_shape = (n, self.num_hidden)
        return [data_shape, indices_shape, indptr_shape, weight_shape, bias_shape],\
                [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return SparseInnerProduct(self.m, self.num_hidden, self.no_bias)

# define mlp

data = mx.symbol.Variable('data')
indices = mx.symbol.Variable('indices')
indptr = mx.symbol.Variable('indptr')
#fc1 = mx.symbol.SparseInnerProduct(data = data, name='fc1', num_hidden=128)
sfc = mx.symbol.Custom(data=data, indices=indices, indptr=indptr,
                       op_type='sparse_inner_product', name='sfc',
                       num_hidden=2, m=3, no_bias=False)

a_indptr = np.array([0,2,3,6])
a_indices = np.array([0,2,2,0,1,2])
a_data = np.array([1,2,3,4,5,6])
weight = mx.nd.array(np.random.randn(2,3))
bias = mx.nd.array(np.random.randn(2))
a = csr_matrix( (a_data,a_indices,a_indptr), shape=(3,3) ).todense()
exe = sfc.bind(mx.cpu(), [mx.nd.array(a_data), mx.nd.array(a_indices),
               mx.nd.array(a_indptr), weight, bias])

#check forward               
forward = exe.outputs[0].asnumpy()
compute = a.dot(exe.arg_dict['sfc_weight'].asnumpy().T) + bias.asnumpy()

#check backward