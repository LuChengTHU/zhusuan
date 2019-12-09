#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from zhusuan.utils import add_name_scope
from zhusuan.transforms.ops.ops import (
    dense, conv2d, nin, layernorm, init_normalization, inverse_sigmoid,
    gated_conv, gated_attn, gated_nin, sumflat, gaussian_sample_logp
)


__all__ = [
    'Sequential',
    'Inverse',
    'SpaceToDepth',
    'CheckerboardSplit',
    'ChannelSplit',
    'TupleFlip',
    'Sigmoid',
    'ImgProc',
    'ActNorm',
    'Pointwise',
    'MixLogisticAttnCoupling',
    'AffineCoupling',
]


class Sequential(InvertibleTransform):
    def __init__(self, layers, layer_names):
        self.layers = layers
        
    def _forward(self, x, *, **kwargs):
        bs = int((x[0] if isinstance(x, tuple) else x).shape[0])
        logd_terms = []
        for f in self.layers:
            assert isinstance(f, InvertibleTransform)
            x, l = f.forward(x, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return x, tf.add_n(logd_terms) if logd_terms else tf.constant(0.) 

    def _inverse(self, y, *, **kwargs):
        bs = int((y[0] if isinstance(y, tuple) else y).shape[0])
        logd_terms = []
        for f in reversed(self.layers[::-1]):
            assert isinstance(f, InvertibleTransform)
            y, l = f.inverse(y, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return y, tf.add_n(logd_terms) if logd_terms else tf.constant(0.)

    @add_name_scope
    def backward(self, outputs, outputs_grad, log_det_grad, *, var_list=None, **kwargs):
        #TODO
        pass


class Inverse(InvertibleTransform):
    def __init__(self, base_flow):
        self.base_flow = base_flow

    def _forward(self, x, *, **kwargs):
        return self.base_flow.inverse(x, **kwargs)

    def _inverse(self, y, *, **kwargs):
        return self.base_flow.forward(y, **kwargs)


class SpaceToDepth(InvertibleTransform):
    def __init__(self, block_size=2):
        self.block_size = block_size

    def _forward(self, x, *, **kwargs):
        return tf.space_to_depth(x, self.block_size), None

    def _inverse(self, y, **kwargs):
        return tf.depth_to_space(y, self.block_size), None


class CheckerboardSplit(InvertibleTransform):
    def _forward(self, x, *, **kwargs):
        assert isinstance(x, tf.Tensor)
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H, W // 2, 2, C])
        a, b = tf.unstack(x, axis=3)
        assert a.shape == b.shape == [B, H, W // 2, C]
        return (a, b), None

    def _inverse(self, y, *, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        assert a.shape == b.shape
        B, H, W_half, C = a.shape
        x = tf.stack([a, b], axis=3)
        assert x.shape == [B, H, W_half, 2, C]
        return tf.reshape(x, [B, H, W_half * 2, C]), None


class ChannelSplit(InvertibleTransform):
    def _forward(self, x, *, **kwargs):
        assert isinstance(x, tf.Tensor)
        assert len(x.shape) == 4 and x.shape[3] % 2 == 0
        return tuple(tf.split(x, 2, axis=3)), None

    def _inverse(self, y, *, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        return tf.concat([a, b], axis=3), None


class TupleFlip(InvertibleTransform):
    def _forward(self, x, *, **kwargs):
        assert isinstance(x, tuple)
        a, b = x
        return (b, a), None

    def _inverse(self, y, *, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        return (b, a), None


class Sigmoid(InvertibleTransform):
    def _forward(self, x, *, **kwargs):
        y = tf.sigmoid(x)
        logd = -tf.nn.softplus(x) - tf.nn.softplus(-x)
        return y, sumflat(logd)

    def _inverse(self, y, *, **kwargs):
        x = inverse_sigmoid(y)
        logd = -tf.log(y) - tf.log(1. - y)
        return x, sumflat(logd)


class ImgProc(InvertibleTransform):
    def __init__(self, max_val=256):
        self.max_val = max_val

    def _forward(self, x, *, **kwargs):
        x = x * (.9 / self.max_val) + .05  # [0, self.max_val] -> [.05, .95]
        x, logd = Sigmoid().inverse(x)
        logd += np.log(.9 / self.max_val) * int(np.prod(x.shape.as_list()[1:]))
        return x, logd

    def _inverse(self, y, *, **kwargs):
        y, logd = Sigmoid().forward(y)
        y = (y - .05) / (.9 / self.max_val)  # [.05, .95] -> [0, self.max_val]
        logd -= np.log(.9 / self.max_val) * int(np.prod(y.shape.as_list()[1:]))
        return y, logd


class ActNorm(InvertibleTransform):
    def __init__(self, init_scale=1.):
        def f(input_, forward, *, vcfg, **kwargs):
            assert not isinstance(input_, list)
            if isinstance(input_, tuple):
                is_tuple = True
            else:
                assert isinstance(input_, tf.Tensor)
                input_ = [input_]
                is_tuple = False

            bs = int(input_[0].shape[0])
            g_and_b = []
            for (i, x) in enumerate(input_):
                g, b = init_normalization(x, name='norm{}'.format(i), init_scale=init_scale, vcfg=vcfg)
                g = tf.maximum(g, 1e-10)
                assert x.shape[0] == bs and g.shape == b.shape == x.shape[1:]
                g_and_b.append((g, b))

            logd = tf.fill([bs], tf.add_n([tf.reduce_sum(tf.log(g)) for (g, _) in g_and_b]))
            if forward:
                out = [x * g[None] + b[None] for (x, (g, b)) in zip(input_, g_and_b)]
            else:
                out = [(x - b[None]) / g[None] for (x, (g, b)) in zip(input_, g_and_b)]
                logd = -logd

            if not is_tuple:
                assert len(out) == 1
                return out[0], logd
            return tuple(out), logd

        self.template = tf.make_template(self.__class__.__name__, f)

    def _forward(self, x, *, **kwargs):
        return self.template(x, forward=True, **kwargs)

    def _inverse(self, y, *, **kwargs):
        return self.template(y, forward=False, **kwargs)


class Pointwise(InvertibleTransform):
    def __init__(self, noisy_identity_init=0.001):
        def f(input_, forward, *, vcfg, **kwargs):
            assert not isinstance(input_, list)
            if isinstance(input_, tuple):
                is_tuple = True
            else:
                assert isinstance(input_, tf.Tensor)
                input_ = [input_]
                is_tuple = False

            out, logds = [], []
            for i, x in enumerate(input_):
                _, img_h, img_w, img_c = x.shape.as_list()
                if noisy_identity_init:
                    # identity + gaussian noise
                    initializer = (
                            np.eye(img_c) + noisy_identity_init * np.random.randn(img_c, img_c)
                    ).astype(np.float32)
                else:
                    init_w = np.random.randn(img_c, img_c)
                    initializer = np.linalg.qr(init_w)[0].astype(np.float32)
                W = get_var('W{}'.format(i), shape=None, initializer=initializer, vcfg=vcfg)
                out.append(self._nin(x, W if forward else tf.matrix_inverse(W)))
                logds.append(
                    (1 if forward else -1) * img_h * img_w *
                    tf.to_float(tf.log(tf.abs(tf.matrix_determinant(tf.to_double(W)))))
                )
            logd = tf.fill([input_[0].shape[0]], tf.add_n(logds))

            if not is_tuple:
                assert len(out) == 1
                return out[0], logd
            return tuple(out), logd

        self.template = tf.make_template(self.__class__.__name__, f)

    @staticmethod
    def _nin(x, w, b=None):
        _, out_dim = w.shape
        s = x.shape.as_list()
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = tf.matmul(x, w)
        if b is not None:
            assert b.shape.ndims == 1
            x = x + b[None, :]
        return tf.reshape(x, s[:-1] + [out_dim])

    def _forward(self, x, *, **kwargs):
        return self.template(x, forward=True, **kwargs)

    def inverse(self, y, *, **kwargs):
        return self.template(y, forward=False, **kwargs)


class ElemwiseAffine(InvertibleTransform):
    def __init__(self, *, scales, biases, logscales=None):
        self.scales = scales
        self.biases = biases
        self.logscales = logscales

    def _get_logscales(self):
        return tf.log(self.scales) if (self.logscales is None) else self.logscales

    def _forward(self, x, *, **kwargs):
        logscales = self._get_logscales()
        assert logscales.shape == x.shape
        return (x * self.scales + self.biases), sumflat(logscales)

    def _inverse(self, y, *, **kwargs):
        logscales = self._get_logscales()
        assert logscales.shape == y.shape
        return ((y - self.biases) / self.scales), sumflat(-logscales)


class MixLogisticCDF(InvertibleTransform):
    """
    Elementwise transformation by the CDF of a mixture of logistics
    """

    def __init__(self, *, logits, means, logscales, min_logscale=-7.):
        self.logits = logits
        self.means = means
        self.logscales = logscales
        self.min_logscale = min_logscale

    def _get_logistic_kwargs(self):
        return dict(
            prior_logits=self.logits,
            means=self.means,
            logscales=tf.maximum(self.logscales, self.min_logscale)
        )

    def _forward(self, x, *, **kwargs):
        logistic_kwargs = self._get_logistic_kwargs()
        out = tf.exp(mixlogistic_logcdf(x=x, **logistic_kwargs))
        logd = mixlogistic_logpdf(x=x, **logistic_kwargs)
        return out, sumflat(logd)

    def _inverse(self, y, *, **kwargs):
        logistic_kwargs = self._get_logistic_kwargs()
        out = mixlogistic_invcdf(y=tf.clip_by_value(y, 0., 1.), **logistic_kwargs)
        logd = -mixlogistic_logpdf(x=out, **logistic_kwargs)
        return out, sumflat(logd)


class MixLogisticAttnCoupling(InvertibleTransform):
    """
    CDF of mixture of logistics, followed by affine
    """

    def __init__(self, filters, blocks, components, heads=4, init_scale=0.1, enable_print=True):
        def f(x, *, vcfg: VarConfig, context=None, dropout_p=0., verbose=True):
            if vcfg.init and verbose and enable_print:
                # debug stuff
                xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.shape))))
                x = tf.Print(
                    x, [tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x)],
                    message='{} (shape/mean/std/min/max) '.format(self.template.variable_scope.name), summarize=10
                )
            B, H, W, C = x.shape.as_list()
            pos_emb = get_var('pos_emb', shape=[H, W, filters], initializer=tf.random_normal_initializer(stddev=0.01),
                              vcfg=vcfg)
            x = conv2d(x, name='proj_in', num_units=filters, vcfg=vcfg)
            for i_block in range(blocks):
                with tf.variable_scope('block{}'.format(i_block)):
                    if context is not None:
                        ctx = context[f'{x.shape.as_list()[1]}_{x.shape.as_list()[2]}']
                    else:
                        ctx = None
                    x = gated_conv(x, name='conv', a=ctx, use_nin=True, dropout_p=dropout_p, vcfg=vcfg)
                    x = layernorm(x, name='ln1', vcfg=vcfg)
                    x = gated_attn(x, name='attn', pos_emb=pos_emb, heads=heads, dropout_p=dropout_p, vcfg=vcfg)
                    x = layernorm(x, name='ln2', vcfg=vcfg)
            x = conv2d(x, name='proj_out', num_units=C * (2 + 3 * components), init_scale=init_scale, vcfg=vcfg)
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])

            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            ml_logits, ml_means, ml_logscales = tf.split(x[:, :, :, :, 2:], 3, axis=4)
            assert s.shape == t.shape == [B, H, W, C]
            assert ml_logits.shape == ml_means.shape == ml_logscales.shape == [B, H, W, C, components]

            return Compose([
                MixLogisticCDF(logits=ml_logits, means=ml_means, logscales=ml_logscales),
                Inverse(Sigmoid()),
                ElemwiseAffine(scales=tf.exp(s), logscales=s, biases=t),
            ])

        self.template = tf.make_template(self.__class__.__name__, f)

    def _forward(self, x, *, **kwargs):
        assert isinstance(x, tuple)
        cf, ef = x
        flow = self.template(cf, **kwargs)
        out, logd = flow.forward(ef)
        assert out.shape == ef.shape == cf.shape
        return (cf, out), logd

    def _inverse(self, y, *, **kwargs):
        assert isinstance(y, tuple)
        cf, ef = y
        flow = self.template(cf, **kwargs)
        out, logd = flow.inverse(ef)
        assert out.shape == ef.shape == cf.shape
        return (cf, out), logd


class AffineCoupling(Flow):
    """
    pure flow, affine couping
    """

    def __init__(self, filters, blocks, init_scale=0.1):
        def f(x, *, vcfg: VarConfig, context=None, dropout_p=0., verbose=True):
            if vcfg.init and verbose:
                # debug stuff
                xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.shape))))
                x = tf.Print(
                    x, [tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x)],
                    message='{} (shape/mean/std/min/max) '.format(self.template.variable_scope.name), summarize=10
                )
            B, H, W, C = x.shape.as_list()
            # pos_emb = get_var('pos_emb', shape=[H, W, filters], initializer=tf.random_normal_initializer(stddev=0.01), vcfg=vcfg)
            x = conv2d(x, name='proj_in', num_units=filters, vcfg=vcfg)
            for i_block in range(blocks):
                with tf.variable_scope(f'block{i_block}'):
                    if context is not None:
                        ctx = context[f'{x.shape.as_list()[1]}_{x.shape.as_list()[2]}']
                    else:
                        ctx = None
                    x = gated_conv(x, name='conv', a=ctx, use_nin=True, dropout_p=dropout_p, vcfg=vcfg)
                    x = layernorm(x, name='ln1', vcfg=vcfg)
                    # x = gated_nin(x, name='attn', pos_emb=pos_emb, dropout_p=dropout_p, vcfg=vcfg)
                    # x = layernorm(x, name='ln2', vcfg=vcfg)
            components = 0  # no mixture of logistics
            x = conv2d(x, name='proj_out', num_units=C * (2 + 3 * components), init_scale=init_scale, vcfg=vcfg)
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])

            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            assert s.shape == t.shape == [B, H, W, C]
            return ElemwiseAffine(scales=tf.exp(s), logscales=s, biases=t)

        self.template = tf.make_template(self.__class__.__name__, f)

    def _forward(self, x, *, **kwargs):
        assert isinstance(x, tuple)
        cf, ef = x
        flow = self.template(cf, **kwargs)
        out, logd = flow.forward(ef)
        assert out.shape == ef.shape == cf.shape
        return (cf, out), logd

    def _inverse(self, y, *, **kwargs):
        assert isinstance(y, tuple)
        cf, ef = y
        flow = self.template(cf, **kwargs)
        out, logd = flow.inverse(ef)
        assert out.shape == ef.shape == cf.shape
        return (cf, out), logd

