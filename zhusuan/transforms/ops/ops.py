
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.distributions import Normal

from zhusuan.transforms.ops.logistic import (
    mixlogistic_logpdf, mixlogistic_logcdf, mixlogistic_invcdf
)

__all__ = [
    'VarConfig',
    'get_var',
    'dense',
    'conv2d',
    'nin',
    'init_normalization',
    'layernorm',
    'gated_conv',
    'gated_attn',
    'gated_nin',
    'sumflat',
    'gaussian_sample_logp',
    'inverse_sigmoid',
]

VarConfig = namedtuple('VarConfig', ['init', 'ema', 'dtype', 'use_resource'])
VarConfig.__new__.__defaults__ = (False, None, tf.float32, False)


def get_var(var_name, *, shape, initializer, vcfg: VarConfig, trainable=True):
    assert vcfg is not None and isinstance(vcfg, VarConfig)
    if isinstance(initializer, np.ndarray):
        initializer = initializer.astype(vcfg.dtype.as_numpy_dtype)
    v = tf.get_variable(var_name, shape=shape, dtype=vcfg.dtype, initializer=initializer, trainable=trainable,
                        use_resource=vcfg.use_resource)
    if vcfg.ema is not None:
        assert isinstance(vcfg.ema, tf.train.ExponentialMovingAverage)
        v = vcfg.ema.average(v)
    return v


def dense(x, *, name, num_units, init_scale=1., vcfg: VarConfig):
    with tf.variable_scope(name):
        _, in_dim = x.shape
        W = get_var('W', shape=[in_dim, num_units], initializer=tf.random_normal_initializer(0, 0.05), vcfg=vcfg)
        b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), vcfg=vcfg)

        if vcfg.init:
            y = tf.matmul(x, W)
            m_init, v_init = tf.nn.moments(y, [0])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            new_W = W * scale_init[None, :]
            new_b = -m_init * scale_init
            with tf.control_dependencies([W.assign(new_W), b.assign(new_b)]):
                if vcfg.use_resource:
                    return tf.nn.bias_add(tf.matmul(x, new_W), new_b)
                else:
                    x = tf.identity(x)

        return tf.nn.bias_add(tf.matmul(x, W), b)


def conv2d(x, *, name, num_units, filter_size=(3, 3), stride=(1, 1), pad='SAME', init_scale=1., vcfg: VarConfig):
    with tf.variable_scope(name):
        assert x.shape.ndims == 4
        W = get_var('W', shape=[*filter_size, int(x.shape[-1]), num_units],
                    initializer=tf.random_normal_initializer(0, 0.05), vcfg=vcfg)
        b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), vcfg=vcfg)

        if vcfg.init:
            y = tf.nn.conv2d(x, W, [1, *stride, 1], pad)
            m_init, v_init = tf.nn.moments(y, [0, 1, 2])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            new_W = W * scale_init[None, None, None, :]
            new_b = -m_init * scale_init
            with tf.control_dependencies([W.assign(new_W), b.assign(new_b)]):
                if vcfg.use_resource:
                    return tf.nn.bias_add(tf.nn.conv2d(x, new_W, [1, *stride, 1], pad), new_b)
                else:
                    x = tf.identity(x)
        return tf.nn.bias_add(tf.nn.conv2d(x, W, [1, *stride, 1], pad), b)


def nin(x, *, num_units, **kwargs):
    assert 'num_units' not in kwargs
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units=num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])


def init_normalization(x, *, name, init_scale=1., vcfg: VarConfig):
    with tf.variable_scope(name):
        g = get_var('g', shape=(1, 1, 1, x.shape[-1]), initializer=tf.constant_initializer(1.), vcfg=vcfg)
        b = get_var('b', shape=(1, 1, 1, x.shape[-1]), initializer=tf.constant_initializer(0.), vcfg=vcfg)
        if vcfg.init:
            # data based normalization
            m_init, v_init = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            assert m_init.shape == v_init.shape == scale_init.shape == g.shape == b.shape
            with tf.control_dependencies([
                g.assign(scale_init),
                b.assign(-m_init * scale_init)
            ]):
                g, b = tf.identity_n([g, b])
        return g, b


def concat_elu(x):
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def gate(x, *, axis):
    a, b = tf.split(x, 2, axis=axis)
    return a * tf.sigmoid(b)


def layernorm(x, *, name, vcfg: VarConfig, e=1e-5):
    """Layer norm over last axis"""
    with tf.variable_scope(name):
        shape = [1] * (x.shape.ndims - 1) + [int(x.shape[-1])]
        g = get_var('g', shape=shape, initializer=tf.constant_initializer(1), vcfg=vcfg)
        b = get_var('b', shape=shape, initializer=tf.constant_initializer(0), vcfg=vcfg)
        u = tf.reduce_mean(x, axis=-1, keepdims=True)
        s = tf.reduce_mean(tf.squared_difference(x, u), axis=-1, keepdims=True)
        return (x - u) * tf.rsqrt(s + e) * g + b


def gated_conv(x, *, name, a, nonlinearity=concat_elu, conv=conv2d, use_nin, dropout_p, vcfg: VarConfig):
    with tf.variable_scope(name):
        num_filters = int(x.shape[-1])

        c1 = conv(nonlinearity(x), name='c1', num_units=num_filters, vcfg=vcfg)
        if a is not None:  # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), name='a_proj', num_units=num_filters, vcfg=vcfg)
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)

        c2 = (nin if use_nin else conv)(c1, name='c2', num_units=num_filters * 2, init_scale=0.1, vcfg=vcfg)
        return x + gate(c2, axis=3)


def gated_attn(x, *, name, pos_emb, heads, dropout_p, vcfg: VarConfig):
    with tf.variable_scope(name):
        bs, height, width, ch = x.shape.as_list()
        assert pos_emb.shape == [height, width, ch]
        assert ch % heads == 0
        timesteps = height * width
        dim = ch // heads
        # Position embeddings
        c = x + pos_emb[None, :, :, :]
        # b, h, t, d == batch, num heads, num timesteps, per-head dim (C // heads)
        c = nin(c, name='proj1', num_units=3 * ch, vcfg=vcfg)
        assert c.shape == [bs, height, width, 3 * ch]
        # Split into heads / Q / K / V
        c = tf.reshape(c, [bs, timesteps, 3, heads, dim])  # b, t, 3, h, d
        c = tf.transpose(c, [2, 0, 3, 1, 4])  # 3, b, h, t, d
        q_bhtd, k_bhtd, v_bhtd = tf.unstack(c, axis=0)
        assert q_bhtd.shape == k_bhtd.shape == v_bhtd.shape == [bs, heads, timesteps, dim]
        # Attention
        w_bhtt = tf.matmul(q_bhtd, k_bhtd, transpose_b=True) / np.sqrt(float(dim))
        w_bhtt = tf.nn.softmax(w_bhtt)
        assert w_bhtt.shape == [bs, heads, timesteps, timesteps]
        a_bhtd = tf.matmul(w_bhtt, v_bhtd)
        # Merge heads
        a_bthd = tf.transpose(a_bhtd, [0, 2, 1, 3])
        assert a_bthd.shape == [bs, timesteps, heads, dim]
        a_btc = tf.reshape(a_bthd, [bs, timesteps, ch])
        # Project
        c1 = tf.reshape(a_btc, [bs, height, width, ch])
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = nin(c1, name='proj2', num_units=ch * 2, init_scale=0.1, vcfg=vcfg)
        return x + gate(c2, axis=3)


def gated_nin(x, *, name, pos_emb, dropout_p, vcfg: VarConfig):
    with tf.variable_scope(name):
        bs, height, width, ch = x.shape.as_list()
        assert pos_emb.shape == [height, width, ch]
        # Position embeddings
        c = x + pos_emb[None, :, :, :]
        c = nin(c, name='proj1', num_units=3 * ch, vcfg=vcfg)
        assert c.shape == [bs, height, width, 3 * ch]
        c = tf.reshape(c, [bs, height, width, ch, 3])
        c1 = tf.reduce_max(c, axis=4)
        assert c1.shape == [bs, height, width, ch]
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = nin(c1, name='proj2', num_units=ch * 2, init_scale=0.1, vcfg=vcfg)
        return x + gate(c2, axis=3)


def sumflat(x):
    return tf.reduce_sum(tf.reshape(x, [x.shape[0], -1]), axis=1)


def gaussian_sample_logp(shape):
    eps = tf.random_normal(shape)
    logp = Normal(0., 1.).log_prob(eps)
    assert logp.shape == eps.shape
    return eps, sumflat(logp)


def inverse_sigmoid(x):
    return -tf.log(tf.reciprocal(x) - 1.)
