"""Checkpoint variable-name remapping, adapted verbatim from Google's
official RepNet colab notebook (google-research/google-research,
repnet/repnet_colab.ipynb, Apache 2.0 license), retrieved 2026-08-20.

The published checkpoint's internal variable names don't match a freshly
constructed `ResnetPeriodEstimator`'s names (a Keras model-naming change
between when the checkpoint was saved and the current notebook code). This
module works around that by remapping old -> new names, zipped positionally
against `model.weights` -- so `MAPPING_NEW_TO_OLD_LAYER_NAMES`'s order must
stay in exact lockstep with `ResnetPeriodEstimator.__init__`'s layer
construction order in `model.py`. Do not reorder either independently.
"""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.training import py_checkpoint_reader

MAPPING_NEW_TO_OLD_LAYER_NAMES = [
    ("pos_encoding/pos_encoding", "pos_encoding/.ATTRIBUTES/VARIABLE_VALUE"),
    ("pos_encoding2/pos_encoding2", "pos_encoding2/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv1_conv/kernel/kernel", "base_model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv1_conv/bias/bias", "base_model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_preact_bn/gamma/gamma", "base_model/layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_preact_bn/beta/beta", "base_model/layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block1_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block1_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block1_1_conv/kernel/kernel", "base_model/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_1_bn/gamma/gamma", "base_model/layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_1_bn/beta/beta", "base_model/layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block1_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block1_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block1_2_conv/kernel/kernel", "base_model/layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_2_bn/gamma/gamma", "base_model/layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_2_bn/beta/beta", "base_model/layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block1_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block1_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block1_0_conv/kernel/kernel", "base_model/layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_0_conv/bias/bias", "base_model/layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_3_conv/kernel/kernel", "base_model/layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_3_conv/bias/bias", "base_model/layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_preact_bn/gamma/gamma", "base_model/layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_preact_bn/beta/", "base_model/layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block2_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block2_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block2_1_conv/kernel/kernel", "base_model/layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_1_bn/gamma/gamma", "base_model/layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_1_bn/beta/beta", "base_model/layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block2_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block2_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block2_2_conv/kernel/kernel", "base_model/layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_2_bn/gamma/gamma", "base_model/layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_2_bn/beta/beta", "base_model/layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block2_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block2_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block2_3_conv/kernel/kernel", "base_model/layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_3_conv/bias/bias", "base_model/layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_preact_bn/gamma/gamma", "base_model/layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_preact_bn/beta/beta", "base_model/layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block3_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block3_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block3_1_conv/kernel/kernel", "base_model/layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_1_bn/gamma/gamma", "base_model/layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_1_bn/beta/beta", "base_model/layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block3_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block3_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block3_2_conv/kernel/kernel", "base_model/layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_2_bn/gamma/gamma", "base_model/layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_2_bn/beta/beta", "base_model/layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv2_block3_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv2_block3_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv2_block3_3_conv/kernel/kernel", "base_model/layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_3_conv/bias/bias", "base_model/layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_preact_bn/gamma/gamma", "base_model/layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_preact_bn/beta/beta", "base_model/layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block1_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block1_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block1_1_conv/kernel/kernel", "base_model/layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_1_bn/gamma/gamma", "base_model/layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_1_bn/beta/beta", "base_model/layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block1_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block1_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block1_2_conv/kernel/kernel", "base_model/layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_2_bn/gamma/gamma", "base_model/layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_2_bn/beta/beta", "base_model/layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block1_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block1_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block1_0_conv/kernel/kernel", "base_model/layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_0_conv/bias/bias", "base_model/layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_3_conv/kernel/kernel", "base_model/layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_3_conv/bias/bias", "base_model/layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_preact_bn/gamma/gamma", "base_model/layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_preact_bn/beta/beta", "base_model/layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block2_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block2_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block2_1_conv/kernel/kernel", "base_model/layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_1_bn/gamma/gamma", "base_model/layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_1_bn/beta/beta", "base_model/layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block2_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block2_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block2_2_conv/kernel/kernel", "base_model/layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_2_bn/gamma/gamma", "base_model/layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_2_bn/beta/beta", "base_model/layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block2_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-31/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block2_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-31/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block2_3_conv/kernel/kernel", "base_model/layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_3_conv/bias/bias", "base_model/layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_preact_bn/gamma/gamma", "base_model/layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_preact_bn/beta/beta", "base_model/layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block3_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-33/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block3_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-33/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block3_1_conv/kernel/kernel", "base_model/layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_1_bn/gamma/gamma", "base_model/layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_1_bn/beta/beta", "base_model/layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block3_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-35/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block3_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-35/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block3_2_conv/kernel/kernel", "base_model/layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_2_bn/gamma/gamma", "base_model/layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_2_bn/beta/beta", "base_model/layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block3_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-37/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block3_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-37/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block3_3_conv/kernel/kernel", "base_model/layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_3_conv/bias/bias", "base_model/layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_preact_bn/gamma/gamma", "base_model/layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_preact_bn/beta/beta", "base_model/layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block4_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-39/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block4_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-39/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block4_1_conv/kernel/kernel", "base_model/layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_1_bn/gamma/gamma", "base_model/layer_with_weights-41/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_1_bn/beta/beta", "base_model/layer_with_weights-41/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block4_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-41/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block4_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-41/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block4_2_conv/kernel/kernel", "base_model/layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_2_bn/gamma/gamma", "base_model/layer_with_weights-43/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_2_bn/beta/beta", "base_model/layer_with_weights-43/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv3_block4_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-43/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv3_block4_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-43/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv3_block4_3_conv/kernel/kernel", "base_model/layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_3_conv/bias/bias", "base_model/layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_preact_bn/gamma/gamma", "base_model/layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_preact_bn/beta/beta", "base_model/layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block1_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-45/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block1_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-45/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block1_1_conv/kernel/kernel", "base_model/layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_1_bn/gamma/gamma", "base_model/layer_with_weights-47/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_1_bn/beta/beta", "base_model/layer_with_weights-47/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block1_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-47/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block1_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-47/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block1_2_conv/kernel/kernel", "base_model/layer_with_weights-48/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_2_bn/gamma/gamma", "base_model/layer_with_weights-49/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_2_bn/beta/beta", "base_model/layer_with_weights-49/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block1_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-49/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block1_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-49/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block1_0_conv/kernel/kernel", "base_model/layer_with_weights-50/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_0_conv/bias/bias", "base_model/layer_with_weights-50/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_3_conv/kernel/kernel", "base_model/layer_with_weights-51/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_3_conv/bias/bias", "base_model/layer_with_weights-51/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_preact_bn/gamma/gamma", "base_model/layer_with_weights-52/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_preact_bn/beta/beta", "base_model/layer_with_weights-52/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block2_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-52/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block2_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-52/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block2_1_conv/kernel/kernel", "base_model/layer_with_weights-53/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_1_bn/gamma/gamma", "base_model/layer_with_weights-54/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_1_bn/beta/beta", "base_model/layer_with_weights-54/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block2_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-54/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block2_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-54/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block2_2_conv/kernel/kernel", "base_model/layer_with_weights-55/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_2_bn/gamma/gamma", "base_model/layer_with_weights-56/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_2_bn/beta/beta", "base_model/layer_with_weights-56/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block2_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-56/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block2_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-56/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block2_3_conv/kernel/kernel", "base_model/layer_with_weights-57/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_3_conv/bias/bias", "base_model/layer_with_weights-57/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_preact_bn/gamma/gamma", "base_model/layer_with_weights-58/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_preact_bn/beta/beta", "base_model/layer_with_weights-58/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block3_preact_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-58/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block3_preact_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-58/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block3_1_conv/kernel/kernel", "base_model/layer_with_weights-59/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_1_bn/gamma/gamma", "base_model/layer_with_weights-60/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_1_bn/beta/beta", "base_model/layer_with_weights-60/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block3_1_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-60/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block3_1_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-60/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block3_2_conv/kernel/kernel", "base_model/layer_with_weights-61/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_2_bn/gamma/gamma", "base_model/layer_with_weights-62/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_2_bn/beta/beta", "base_model/layer_with_weights-62/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "conv4_block3_2_bn/moving_mean/moving_mean",
        "base_model/layer_with_weights-62/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "conv4_block3_2_bn/moving_variance/moving_variance",
        "base_model/layer_with_weights-62/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("conv4_block3_3_conv/kernel/kernel", "base_model/layer_with_weights-63/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_3_conv/bias/bias", "base_model/layer_with_weights-63/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "resnet_period_estimator_6/conv3d_6/kernel/kernel",
        "temporal_conv_layers/0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("resnet_period_estimator_6/conv3d_6/bias/bias", "temporal_conv_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "resnet_period_estimator_6/batch_normalization_6/gamma/gamma",
        "temporal_bn_layers/0/gamma/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/batch_normalization_6/beta/beta",
        "temporal_bn_layers/0/beta/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/batch_normalization_6/moving_mean/moving_mean",
        "temporal_bn_layers/0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/batch_normalization_6/moving_variance/moving_variance",
        "temporal_bn_layers/0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("resnet_period_estimator_6/conv2d_6/kernel/kernel", "conv_3x3_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/conv2d_6/bias/bias", "conv_3x3_layer/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_66/kernel/kernel", "input_projection/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_66/bias/bias", "input_projection/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_67/kernel/kernel", "input_projection2/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_67/bias/bias", "input_projection2/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_68/kernel/kernel",
        "transformer_layers/0/mha/wq/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_68/bias/bias",
        "transformer_layers/0/mha/wq/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_69/kernel/kernel",
        "transformer_layers/0/mha/wk/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_69/bias/bias",
        "transformer_layers/0/mha/wk/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_70/kernel/kernel",
        "transformer_layers/0/mha/wv/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_70/bias/bias",
        "transformer_layers/0/mha/wv/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_71/kernel/kernel",
        "transformer_layers/0/mha/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_71/bias/bias",
        "transformer_layers/0/mha/dense/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_72/kernel/kernel",
        "transformer_layers/0/ffn/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_72/bias/bias",
        "transformer_layers/0/ffn/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_73/kernel/kernel",
        "transformer_layers/0/ffn/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_73/bias/bias",
        "transformer_layers/0/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/layer_normalization_12/gamma/gamma",
        "transformer_layers/0/layernorm1/gamma/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/layer_normalization_12/beta/beta",
        "transformer_layers/0/layernorm1/beta/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/layer_normalization_13/gamma/gamma",
        "transformer_layers/0/layernorm2/gamma/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_6/layer_normalization_13/beta/beta",
        "transformer_layers/0/layernorm2/beta/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_74/kernel/kernel",
        "transformer_layers2/0/mha/wq/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_74/bias/bias",
        "transformer_layers2/0/mha/wq/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_75/kernel/kernel",
        "transformer_layers2/0/mha/wk/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_75/bias/bias",
        "transformer_layers2/0/mha/wk/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_76/kernel/kernel",
        "transformer_layers2/0/mha/wv/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_76/bias/bias",
        "transformer_layers2/0/mha/wv/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_77/kernel/kernel",
        "transformer_layers2/0/mha/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_77/bias/bias",
        "transformer_layers2/0/mha/dense/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_78/kernel/kernel",
        "transformer_layers2/0/ffn/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_78/bias/bias",
        "transformer_layers2/0/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_79/kernel/kernel",
        "transformer_layers2/0/ffn/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_79/bias/bias",
        "transformer_layers2/0/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/layer_normalization_14/gamma/gamma",
        "transformer_layers2/0/layernorm1/gamma/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/layer_normalization_14/beta/beta",
        "transformer_layers2/0/layernorm1/beta/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/layer_normalization_15/gamma/gamma",
        "transformer_layers2/0/layernorm2/gamma/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    (
        "resnet_period_estimator_6/transformer_layer_7/layer_normalization_15/beta/beta",
        "transformer_layers2/0/layernorm2/beta/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("resnet_period_estimator_6/dense_80/kernel/kernel", "fc_layers/0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_80/bias/bias", "fc_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_81/kernel/kernel", "fc_layers/1/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_81/bias/bias", "fc_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_82/kernel/kernel", "fc_layers/2/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_82/bias/bias", "fc_layers/2/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "resnet_period_estimator_6/dense_83/kernel/kernel",
        "within_period_fc_layers/0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("resnet_period_estimator_6/dense_83/bias/bias", "within_period_fc_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "resnet_period_estimator_6/dense_84/kernel/kernel",
        "within_period_fc_layers/1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("resnet_period_estimator_6/dense_84/bias/bias", "within_period_fc_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    (
        "resnet_period_estimator_6/dense_85/kernel/kernel",
        "within_period_fc_layers/2/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    ),
    ("resnet_period_estimator_6/dense_85/bias/bias", "within_period_fc_layers/2/bias/.ATTRIBUTES/VARIABLE_VALUE"),
]


def load_ckpt_with_custom_layer_mapping(model, logdir, custom_mappings):
    """Remap checkpoint variables to handle Keras model name changes correctly."""
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=logdir, max_to_keep=10)
    latest_ckpt = ckpt_manager.latest_checkpoint
    checkpoint_reader = py_checkpoint_reader.NewCheckpointReader(latest_ckpt)
    for (new_layer_name, old_layer_name), weight in zip(custom_mappings, model.weights):
        old_layer_name = "model/" + old_layer_name
        old_weight = checkpoint_reader.get_tensor(old_layer_name)
        if weight.value.shape.num_elements() != np.prod(old_weight.shape):
            raise ValueError(f"Shape mismatch in layer ({new_layer_name}, {old_layer_name})")
        weight.assign(old_weight)
