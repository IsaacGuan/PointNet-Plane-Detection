import argparse
import tensorflow as tf
import json
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='train_results/trained_models/epoch_100.ckpt', help='Model checkpoint path')
parser.add_argument('--model', choices=["1", "2"], default="1", help='Model to use [1/2]')
FLAGS = parser.parse_args()


# DEFAULT SETTINGS
pretrained_model_path = FLAGS.model_path # os.path.join(BASE_DIR, './pretrained_model/model.ckpt')
hdf5_data_dir = os.path.join(BASE_DIR, './data/hdf5_data')
ply_data_dir = os.path.join(BASE_DIR, './data')
gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_results')
output_verbose = True   # If true, output all color-coded part segmentation obj files

if FLAGS.model == "1":
	model = __import__('pointnet_plane_detection')
if FLAGS.model == "2":
	model = __import__('pointnet_plane_detection2')

# MAIN SCRIPT
point_num = 2048            # the max number of points in the all testing data shapes
batch_size = 1

test_file_list = os.path.join(BASE_DIR, 'testing_ply_file_list')

color_map_file = os.path.join(hdf5_data_dir, 'plane_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))

NUM_PART_CATS = 2

def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    return pointclouds_ph

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def load_pts_seg_files(pts_file, seg_file):
    with open(pts_file, 'r') as f:
        pts_str = [item.rstrip() for item in f.readlines()]
        pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
    with open(seg_file, 'r') as f:
        part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
        seg = np.array([x for x in part_ids])
    return pts, seg

def pc_augment_to_point_num(pts, pn):
    assert(pts.shape[0] <= pn)
    cur_len = pts.shape[0]
    res = np.array(pts)
    while cur_len < pn:
        res = np.concatenate((res, pts))
        cur_len += pts.shape[0]
    return res[:pn, :]

def predict():
    is_training = False
    with tf.device('/gpu:'+str(gpu_to_use)):
        pointclouds_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # simple model
        seg_pred, end_points = model.get_model(pointclouds_ph, \
        	part_num=NUM_PART_CATS, is_training=is_training_ph, \
        	batch_size=batch_size, num_point=point_num, weight_decay=0.0, bn_decay=None)
        
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        flog = open(os.path.join(output_dir, 'log.txt'), 'w')

        # Restore variables from disk.
        printout(flog, 'Loading model %s' % pretrained_model_path)
        saver.restore(sess, pretrained_model_path)
        printout(flog, 'Model restored.')
        
        # Note: the evaluation for the model with BN has to have some statistics
        # Using some test datas as the statistics
        batch_data = np.zeros([batch_size, point_num, 3]).astype(np.float32)

        total_acc = 0.0
        total_seen = 0
        total_acc_iou = 0.0

        ffiles = open(test_file_list, 'r')
        lines = [line.rstrip() for line in ffiles.readlines()]
        pts_files = [line.split()[0] for line in lines]
        seg_files = [line.split()[1] for line in lines]
        ffiles.close()

        len_pts_files = len(pts_files)
        for shape_idx in range(len_pts_files):
            if shape_idx % 100 == 0:
                printout(flog, '%d/%d ...' % (shape_idx, len_pts_files))

            pts_file_to_load = os.path.join(ply_data_dir, pts_files[shape_idx])
            seg_file_to_load = os.path.join(ply_data_dir, seg_files[shape_idx])

            pts, seg = load_pts_seg_files(pts_file_to_load, seg_file_to_load)
            ori_point_num = len(seg)

            batch_data[0, ...] = pc_augment_to_point_num(pc_normalize(pts), point_num)

            seg_pred_res = sess.run([seg_pred], feed_dict={
                        pointclouds_ph: batch_data,
                        is_training_ph: is_training,
                    })
            
            seg_pred_res = np.array(seg_pred_res)[0, ...][0, ...]

            iou_oids = [0, 1]

            seg_pred_val = np.argmax(seg_pred_res, axis=1)[:ori_point_num]

            print seg_pred_val

            seg_acc = np.mean(seg_pred_val == seg)

            total_acc += seg_acc
            total_seen += 1

            mask = np.int32(seg_pred_val == seg)

            total_iou = 0.0
            iou_log = ''
            for oid in iou_oids:
                n_pred = np.sum(seg_pred_val == oid)
                n_gt = np.sum(seg == oid)
                n_intersect = np.sum(np.int32(seg == oid) * mask)
                n_union = n_pred + n_gt - n_intersect
                iou_log += '_' + str(n_pred)+'_'+str(n_gt)+'_'+str(n_intersect)+'_'+str(n_union)+'_'
                if n_union == 0:
                    total_iou += 1
                    iou_log += '_1\n'
                else:
                    total_iou += n_intersect * 1.0 / n_union
                    iou_log += '_'+str(n_intersect * 1.0 / n_union)+'\n'

            avg_iou = total_iou / len(iou_oids)
            total_acc_iou += avg_iou
            
            if output_verbose:
                output_color_point_cloud(pts, seg, os.path.join(output_dir, str(shape_idx)+'_gt.obj'))
                output_color_point_cloud(pts, seg_pred_val, os.path.join(output_dir, str(shape_idx)+'_pred.obj'))
                output_color_point_cloud_red_blue(pts, np.int32(seg == seg_pred_val), 
                        os.path.join(output_dir, str(shape_idx)+'_diff.obj'))

                with open(os.path.join(output_dir, str(shape_idx)+'.log'), 'w') as fout:
                    fout.write('Total Point: %d\n\n' % ori_point_num)
                    fout.write('Accuracy: %f\n' % seg_acc)
                    fout.write('IoU: %f\n\n' % avg_iou)
                    fout.write('IoU details: %s\n' % iou_log)

        printout(flog, 'Accuracy: %f' % (total_acc / total_seen))
        printout(flog, 'IoU: %f' % (total_acc_iou / total_seen))

                
with tf.Graph().as_default():
    predict()
