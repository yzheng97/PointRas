import os
import sys
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
import tensorflow as tf
import numpy as np
import tf_util
from tf_sampling import gather_point
from PointConv import feature_encoding_layer_rend, feature_decoding_layer, interpolate_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, sigma, bn_decay=None, weight_decay=None, mode="train"):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = point_cloud
    l0_idx = None

    # Feature encoding layers
    l1_xyz, l1_points, l1_idx = feature_encoding_layer_rend(l0_xyz, l0_points, l0_idx, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points, l2_idx = feature_encoding_layer_rend(l1_xyz, l1_points, l1_idx, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points, l3_idx = feature_encoding_layer_rend(l2_xyz, l2_points, l2_idx, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l4_xyz, l4_points, l4_idx = feature_encoding_layer_rend(l3_xyz, l3_points, l3_idx, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')

    # Feature decoding layers
    l3_points = feature_decoding_layer(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    final_mask = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')

    if mode == "train":
        net = tf_util.conv1d(l3_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
        l3_mask = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc4')
        l3_final_mask = gather_point(final_mask, l3_idx)
        end_points['l3_idx'] = l3_idx
        l3_combined_feats = tf.concat(axis=2, values=[l3_points, l3_mask, l3_final_mask])
        l3_rend = tf_util.conv1d(l3_combined_feats, num_class, 1, padding='VALID', activation_fn=None, scope='rend_mlp')
        end_points['l3_rend'] = l3_rend
        end_points['l3_mask'] = l3_mask

    else:
        final_uncertain_idx = get_uncertainty_idx(final_mask, final_mask.shape[1]//30)
        final_mask = gather_point(point_cloud, final_uncertain_idx)

    return final_mask, end_points

def batch_scatter_add(ori, idx, delta):
    #return ori
    res = []
    B = idx.shape[0]
    L = idx.shape[1]
    for i in range(B):
        res.append(tf.scatter_nd(tf.reshape(idx[i], (L,-1)), delta[i], ori[i].shape))
    return ori + tf.convert_to_tensor(res)


def get_idx_in_all(idx_in_all, idx):
    new_idx = []
    B = idx.shape[0]
    for i in range(B):
        new_idx.append(tf.gather_nd(idx_in_all[i], tf.reshape(idx[i], (idx[i].shape[0],-1))))
    return tf.convert_to_tensor(new_idx)


def get_uncertainty_idx(mask, k):
    B = mask.shape[0]
    mask = tf.sort(mask, axis=2, direction='DESCENDING')
    uncertainty_map = -1 * (mask[:,:,0] - mask[:,:,1])
    _, idx = tf.nn.top_k(tf.reshape(uncertainty_map, (B,-1)), k)
    return idx

def get_loss(pred, label, smpw, end_points, mode="train"):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """

    if mode == "train":
        B, N, C = pred.shape[0], pred.shape[1], pred.shape[2]
        classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
        weight_reg = tf.add_n(tf.get_collection('losses'))
        classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')

        l3_rend, l3_mask, l3_idx = end_points['l3_rend'], end_points['l3_mask'], end_points['l3_idx']
        l3_label = tf.to_int32(tf.reshape(gather_point(tf.reshape(tf.to_float(label), (B, N, 1)), l3_idx), (B, -1)))
        l3_smpw = tf.reshape(gather_point(tf.reshape(smpw, (B, N, 1)), l3_idx), (B, -1))
        rend_loss = tf.losses.sparse_softmax_cross_entropy(labels=l3_label, logits=l3_rend, weights=l3_smpw)
        mask_loss = tf.losses.sparse_softmax_cross_entropy(labels=l3_label, logits=l3_mask, weights=l3_smpw)
        rend_loss_mean = tf.reduce_mean(rend_loss, name='rend_loss_mean')
        mask_loss_mean = tf.reduce_mean(mask_loss, name='mask_loss_mean')

        total_loss = (classify_loss_mean + weight_reg) + (rend_loss_mean + mask_loss_mean)
        tf.summary.scalar('total loss', total_loss)
        return total_loss
    else:
        classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
        weight_reg = tf.add_n(tf.get_collection('losses'))
        classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
        total_loss = classify_loss_mean + weight_reg
        tf.summary.scalar('classify loss', classify_loss)
        tf.summary.scalar('total loss', total_loss)
        return total_loss

if __name__=='__main__':
    import pdb
    pdb.set_trace()

    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10, 1.0)
        print(net)

