#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from zhusuan.utils import add_name_scope
import numpy as np

__all__ = [
    'InvertibleTransform',
]


class InvertibleTransform(object):
    _batch_shape = None
    _get_batch_shape = None

    def get_batch_shape(self):
        return self._get_batch_shape
    
    def batch_shape(self):
        return self._batch_shape
    
    def prepare_shape(self, latents, **kwargs):
        x, _ = self.inverse(latents, **kwargs)
        self._get_batch_shape = x.get_shape()
        self._batch_shape = tf.shape(x)

    def _forward(self, x, **kwargs):
        """
        Forward computation.

        :param x: an input batch.
        :param scope: the scope of the variable scope.
        :param reuse: whether or not to reuse the var scope.

        :return: output, the output transformed by the transformation.
            May be empty.
        :return: log_det, a batch of log-determinants. May be None.
        """
        raise NotImplementedError

    def _inverse(self, y, **kwargs):
        """
        Inverse computation.

        :param y: the outputs from the transformation.
        :param scope: the scope of the variable scope.
        :param reuse: whether or not to reuse the var scope.

        :return: the reconstructed inputs.
        :return: a batch of log-determinants. May be None.
        """
        raise NotImplementedError

    def _sample(self, latents, **kwargs):
        x, _ = self._inverse(latents, **kwargs)
        return x

    @add_name_scope
    def forward(self, x, **kwargs):
        """
        Forward computation.

        :param x: an input batch.
        :param scope: the scope of the variable scope.
        :param reuse: whether or not to reuse the var scope.
        :param custom_grad: whether or not to use custom gradient computation.

        :return: the output transformed by the transformation.
        :return: a batch of log-determinants. May be None.
        """
        return self._forward(x, **kwargs)

    @add_name_scope
    def inverse(self, y, **kwargs):
        """
        Inverse computation.

        :param y: an input batch.
        :param scope: the scope of the variable scope.
        :param reuse: whether or not to reuse the var scope.
        :param custom_grad: whether or not to use custom gradient computation.

        :return: the reconstructed inputs.
        :return: a batch of log-determinants. May be None.
        """
        return self._inverse(y, **kwargs)
    
    @add_name_scope
    def sample(self, latents, **kwargs):
        return self._sample(latents, **kwargs)

    @add_name_scope
    def gradients(self, outputs, log_det, loss, *, var_list=None, **kwargs):
        """
        :param var_list: the list of variables to differentiate. If None, use
            trainable variables.
        :return: A list of (gradient, variable) for variable in var_list.

        """
        outputs_grad = tf.gradients(loss, outputs)[0]
        if outputs_grad is None:
            outputs_grad = tf.zeros_like(outputs)
        
        if log_det is not None:
            log_det_grad = tf.gradients(loss, log_det)[0]
            if log_det_grad is None:
                log_det_grad = tf.zeros_like(log_det)
        else:
            log_det_grad = None
        return self.backward(outputs, outputs_grad, log_det_grad, var_list=var_list, **kwargs)[2]
        
    @add_name_scope
    def backward(self, outputs, outputs_grad, log_det_grad, *, var_list=None, **kwargs):
        """
        Compute the gradients without storing intermediate tensors.

        :param outputs: the output of the transformation.
        :param outputs_grad: the gradient of output.
        :param log_det_grad: the gradient of log-determinant.
        :param var_list: the list of variables to differentiate. If None, use
            all trainable variables.

        :return: inputs, the reconstructed inputs.
        :return: inputs_grad, the gradient of the inputs.
        :return: grads, a list of (gradient, variable) pairs for the paraameters
            of the layer.
        """
        inputs = tf.stop_gradient(self.inverse(outputs, **kwargs))
        new_outputs, new_log_det = self.forward(inputs, **kwargs)
        if new_log_det is None:
            return inputs, outputs_grad, []
        #TODO
        pass
