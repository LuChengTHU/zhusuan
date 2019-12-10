#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
from collections import deque
from pprint import pprint

import tensorflow as tf
from six.moves import range
import numpy as np
from tqdm import trange

import zhusuan as zs
from zhusuan.transforms.invertible import *
from zhusuan.transforms.ops.ops import (
    VarConfig, conv2d, gated_conv, gaussian_sample_logp,
)

from utils import (
    iterbatches, seed_all, TensorBoardOutput, tile_imgs, load_data,
    setup_horovod, print_params,
)


def construct(*, filters, dequant_filters, components, blocks):
    # see MixLogisticAttnCoupling constructor
    dequant_coupling_kwargs = dict(filters=dequant_filters, blocks=2, components=components)
    coupling_kwargs = dict(filters=filters, blocks=blocks, components=components)


    class Generative(InvertibleTransform):
        def __init__(self):
            self.flow = Sequential([
                CheckerboardSplit(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Inverse(CheckerboardSplit()),

                SpaceToDepth(),

                ChannelSplit(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Inverse(ChannelSplit()),

                CheckerboardSplit(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Inverse(CheckerboardSplit()),
            ])
        
        def _forward(self, x, **kwargs):
            return self.flow.forward(x, **kwargs)
        
        def _inverse(self, y, **kwargs):
            return self.flow.inverse(y, **kwargs)
        
        def _sample(self, latents, **kwargs):
            x, _ = self._inverse(latents, **kwargs)
            x, _ = ImgProc().inverse(x, **kwargs)
            return x


    class Dequant(InvertibleTransform):
        def __init__(self):
            self.dequant_flow = Sequential([
                CheckerboardSplit(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                ActNorm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Inverse(CheckerboardSplit()),
                Sigmoid(),
            ])

            def shallow_processor(x, *, dropout_p, vcfg, **kwargs):
                x = x / 256.0 - 0.5
                context = {}

                (this, that), _ = CheckerboardSplit().forward(x)
                x = conv2d(tf.concat([this, that], 3), name='proj', num_units=32, vcfg=vcfg)
                for i in range(3):
                    x = gated_conv(x, name=f'c{i}', vcfg=vcfg, dropout_p=dropout_p, use_nin=False, a=None)
                context[f'{x.shape.as_list()[1]}_{x.shape.as_list()[2]}'] = x

                return context

            self.context_proc = tf.make_template("context_proc", shallow_processor)

        def _forward(self, x, **kwargs):
            if 'context' in kwargs.keys():
                assert kwargs['context'] is None
            dequant_flow_kwargs = dict(kwargs)
            dequant_flow_kwargs['context'] = self.context_proc(x, **kwargs)
            eps, eps_logp = gaussian_sample_logp(x.shape.as_list())
            xd, logd = self.dequant_flow.forward(eps, **dequant_flow_kwargs)
            assert eps.shape == x.shape and logd.shape == eps_logp.shape == [x.shape[0]]
            x = x + xd
            dequant_logd = logd - eps_logp
            x, logd = ImgProc().forward(x, **kwargs)
            return x, dequant_logd + logd
        

    dequant_flow = Dequant()
    flow = Generative()
    return dequant_flow, flow


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(*, full_batch_shape, flow, flow_kwargs, n_particles=None):
    bn = zs.BayesianNet()
    #TODO: auto compute the batch_shape of latents
    ux_mean = tf.zeros([full_batch_shape[0], 16, 16, 12])
    group_ndims = len(full_batch_shape) - 1
    ux = bn.normal("ux", ux_mean, std=1., group_ndims=group_ndims, n_samples=n_particles)
    bn.flow("x", ux, flow, flow_kwargs, n_samples=n_particles, group_ndims=group_ndims)
    return bn


def build_forward(*, x, dequant_flow, model, flow_kwargs):
    dequant_x, dequant_logd = dequant_flow.forward(x, **flow_kwargs)
    main_logd = model.observe()["x"].dist.log_prob(dequant_x)
    assert dequant_logd.shape == main_logd.shape == [dequant_x.shape[0]] == [x.shape[0]]
    total_logp = dequant_logd + main_logd
    loss = -tf.reduce_mean(total_logp)
    return loss, total_logp


def train(
        *,
        flow_constructor,
        logdir,
        lr_schedule,
        dropout_p,
        seed,
        init_bs,
        total_bs,
        ema_decay,
        steps_per_log,
        epochs_per_val,
        max_grad_norm,
        dtype=tf.float32,
        scale_loss=None,
        restore_checkpoint=None,
        scale_grad=None,
        dataset='cifar10',
        steps_per_extra_samples=None
):
    hvd, MPI, is_root, mpi_average = setup_horovod()

    # Seeding and logging setup
    seed_all(hvd.rank() + hvd.size() * seed)
    assert total_bs % hvd.size() == 0
    local_bs = total_bs // hvd.size()

    logger = None
    logdir = '{}_mpi{}_{}'.format(os.path.expanduser(logdir), hvd.size(), time.time())
    checkpointdir = os.path.join(logdir, 'checkpoints')
    if is_root:
        print('Floating point format:', dtype)
        pprint(locals())
        os.makedirs(logdir)
        os.makedirs(checkpointdir)
        logger = TensorBoardOutput(logdir)

    # Load data
    if is_root:
        # Load once on root first to prevent downloading conflicts
        print('Loading data')
        load_data(dataset=dataset, dtype=dtype.as_numpy_dtype)
    MPI.COMM_WORLD.Barrier()
    data_train, data_val = load_data(dataset=dataset, dtype=dtype.as_numpy_dtype)
    img_shp = list(data_train.shape[1:])
    if is_root:
        print('Training data: {}, Validation data: {}'.format(data_train.shape[0], data_val.shape[0]))
        print('Image shape:', img_shp)
    bpd_scale_factor = 1. / (np.log(2) * np.prod(img_shp))

    # Build graph
    if is_root: print('Building graph')
    dequant_flow, flow = flow_constructor()
    # Data-dependent init
    if is_root: print('===== Init graph =====')
    x_init_full_batch_shape = [init_bs] + img_shp
    x_init_sym = tf.placeholder(dtype, x_init_full_batch_shape)
    init_flow_kwargs=dict(vcfg=VarConfig(init=True, ema=None, dtype=dtype), dropout_p=dropout_p, verbose=is_root)
    model_init = build_gen(
        full_batch_shape=x_init_full_batch_shape, flow=flow, flow_kwargs=init_flow_kwargs
    )
    init_loss_sym, _ = build_forward(
        x=x_init_sym, dequant_flow=dequant_flow, model=model_init, flow_kwargs=init_flow_kwargs
    )
    # Training
    if is_root: print('===== Training graph =====')
    x_full_batch_shape = [local_bs] + img_shp
    x_sym = tf.placeholder(dtype, x_full_batch_shape)
    train_flow_kwargs=dict(vcfg=VarConfig(init=False, ema=None, dtype=dtype), dropout_p=dropout_p, verbose=is_root)
    model_train = build_gen(
        full_batch_shape=x_full_batch_shape, flow=flow, flow_kwargs=train_flow_kwargs
    )
    loss_sym, _ = build_forward(
        x=x_sym, dequant_flow=dequant_flow, model=model_train, flow_kwargs=train_flow_kwargs
    )

    # EMA
    params = tf.trainable_variables()
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    maintain_averages_op = tf.group(ema.apply(params))
    # Op for setting the ema params to the current non-ema params (for use after data-dependent init)
    name2var = {v.name: v for v in tf.global_variables()}
    copy_params_to_ema = tf.group([
        name2var[p.name.replace(':0', '') + '/ExponentialMovingAverage:0'].assign(p) for p in params
    ])

    # Validation and sampling (with EMA)
    if is_root: print('===== Validation graph =====')
    val_flow_kwargs = dict(vcfg=VarConfig(init=False, ema=ema, dtype=dtype), dropout_p=0, verbose=is_root)
    model_val = build_gen(
        full_batch_shape=x_full_batch_shape, flow=flow, flow_kwargs=val_flow_kwargs
    )
    val_loss_sym, _ = build_forward(
        x=x_sym, dequant_flow=dequant_flow, model=model_val, flow_kwargs=val_flow_kwargs
    )

    if is_root: print('===== Sampling graph =====')
    samples_sym = model_val.observe()["x"].sample(None)
    allgathered_samples_sym = hvd.allgather(tf.to_float(samples_sym))
    assert len(tf.trainable_variables()) == len(params)

    def run_validation(sess, i_step):
        data_val_shard = np.array_split(data_val, hvd.size(), axis=0)[hvd.rank()]
        shard_losses = np.concatenate([
            sess.run([val_loss_sym], {x_sym: val_batch}) for val_batch, in
            iterbatches([data_val_shard], batch_size=local_bs, include_final_partial_batch=False)
        ])
        val_loss, total_count = mpi_average(shard_losses)
        samples = sess.run(allgathered_samples_sym)
        if is_root:
            logger.writekvs(
                [
                    ('val_bpd', bpd_scale_factor * val_loss),
                    ('num_val_examples', total_count * local_bs),
                    ('samples', tile_imgs(np.clip(samples, 0, 255).astype(np.uint8)))
                ],
                i_step
            )

    def run_sampling_only(sess, i_step):
        samples = sess.run(allgathered_samples_sym)
        if is_root:
            logger.writekvs(
                [
                    ('samples', tile_imgs(np.clip(samples, 0, 255).astype(np.uint8)))
                ],
                i_step
            )

    # Optimization
    lr_sym = tf.placeholder(dtype, [], 'lr')
    optimizer = hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr_sym))

    if scale_loss is None:
        grads_and_vars = optimizer.compute_gradients(loss_sym, var_list=params)
    else:
        grads_and_vars = [
            (g / scale_loss, v) for (g, v) in optimizer.compute_gradients(loss_sym * scale_loss, var_list=params)
        ]

    if scale_grad is not None:
        grads_and_vars = [
            (g / scale_grad, v) for (g, v) in grads_and_vars
        ]
    if max_grad_norm is not None:
        clipped_grads, grad_norm_sym = tf.clip_by_global_norm([g for (g, _) in grads_and_vars], max_grad_norm)
        grads_and_vars = [(cg, v) for (cg, (_, v)) in zip(clipped_grads, grads_and_vars)]
    else:
        grad_norm_sym = tf.constant(0.)
    opt_sym = tf.group(optimizer.apply_gradients(grads_and_vars), maintain_averages_op)

    def loop(sess: tf.Session):
        i_step = 0

        if is_root: print('Initializing')
        sess.run(tf.global_variables_initializer())
        if restore_checkpoint is not None:
            # Restore from checkpoint
            if is_root:
                saver = tf.train.Saver()
                print('Restoring checkpoint:', restore_checkpoint)
                restore_step = int(restore_checkpoint.split('-')[-1])
                print('Restoring from step:', restore_step)
                saver.restore(sess, restore_checkpoint)
                i_step = restore_step
            else:
                saver = None
        else:
            # No checkpoint: perform data dependent initialization
            if is_root: print('Data dependent init')
            init_loss = sess.run(init_loss_sym,
                                 {x_init_sym: data_train[np.random.randint(0, data_train.shape[0], init_bs)]})
            if is_root: print('Init loss:', init_loss * bpd_scale_factor)
            sess.run(copy_params_to_ema)
            saver = tf.train.Saver() if is_root else None
        if is_root: print('Broadcasting initial parameters')
        sess.run(hvd.broadcast_global_variables(0))
        sess.graph.finalize()

        if is_root:
            print('Training')
            print_params()

        loss_hist = deque(maxlen=steps_per_log)
        gnorm_hist = deque(maxlen=steps_per_log)
        for i_epoch in range(99999999999):
            if i_epoch % epochs_per_val == 0:
                run_validation(sess, i_step=i_step)
                if saver is not None:
                    saver.save(sess, os.path.join(checkpointdir, 'model'), global_step=i_step)

            epoch_start_t = time.time()
            for i_epoch_step, (batch,) in enumerate(iterbatches(  # non-sharded: each gpu goes through the whole dataset
                    [data_train], batch_size=local_bs, include_final_partial_batch=False,
            )):

                if steps_per_extra_samples is not None and i_step % steps_per_extra_samples == 0:
                    run_sampling_only(sess, i_step)

                lr = lr_schedule(i_step)
                loss, gnorm, _ = sess.run([loss_sym, grad_norm_sym, opt_sym], {x_sym: batch, lr_sym: lr})
                loss_hist.append(loss)
                gnorm_hist.append(gnorm)

                # Skip timing the very first step, which will be unusually slow due to TF initialization
                if i_epoch == i_epoch_step == 0:
                    epoch_start_t = time.time()

                if i_step % steps_per_log == 0:
                    loss_hist_means = MPI.COMM_WORLD.gather(float(np.mean(loss_hist)), root=0)
                    gnorm_hist_means = MPI.COMM_WORLD.gather(float(np.mean(gnorm_hist)), root=0)
                    steps_per_sec = (i_epoch_step + 1) / (time.time() - epoch_start_t)
                    if is_root:
                        kvs = [
                            ('iter', i_step),
                            ('epoch', i_epoch + i_epoch_step * local_bs / data_train.shape[0]),  # epoch for this gpu
                            ('bpd', float(np.mean(loss_hist_means) * bpd_scale_factor)),
                            ('gnorm', float(np.mean(gnorm_hist_means))),
                            ('lr', float(lr)),
                            ('fps', steps_per_sec * total_bs),  # fps calculated over all gpus (this epoch)
                            ('sps', steps_per_sec),
                        ]
                        logger.writekvs(kvs, i_step)
                i_step += 1
            # End of epoch

    # Train
    config = tf.ConfigProto()
    # config.log_device_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())  # Pin GPU to local rank (one GPU per process)
    with tf.Session(config=config) as sess:
        loop(sess)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_checkpoint', type=str, default=None)
    parser.add_argument('--save_samples', type=str, default=None)
    args = parser.parse_args()

    max_lr = 3e-4
    warmup_steps = 2000
    lr_decay = 1

    def lr_schedule(step):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        return max_lr * (lr_decay ** (step - warmup_steps))

    dropout_p = 0.2
    components = 32  # logistic mixture components
    blocks = 10
    filters = dequant_filters = 96
    ema_decay = 0.999

    def flow_constructor():
        return construct(filters=filters, dequant_filters=dequant_filters, components=components, blocks=blocks)

    # if args.eval_checkpoint:
    #     evaluate(
    #         flow_constructor=flow_constructor, seed=0, restore_checkpoint=args.eval_checkpoint,
    #         samples_filename=args.save_samples
    #     )
    #     return

    train(
        flow_constructor=flow_constructor,
        logdir=f'~/workspace/logs/zhusuan_cifar_fbdq{dequant_filters}_mixlog{components}_blocks{blocks}_f{filters}_lr{max_lr}_drop{dropout_p}',
        lr_schedule=lr_schedule,
        dropout_p=dropout_p,
        seed=0,
        init_bs=128,
        total_bs=64,
        ema_decay=ema_decay,
        steps_per_log=100,
        epochs_per_val=1,
        max_grad_norm=1.,
    )


if __name__ == "__main__":
    main()
