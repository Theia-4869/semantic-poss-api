#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# developed by Shiji Xin

import numpy as np
import cv2 as cv
import os


def read_points(bin_file):
    points = np.fromfile(bin_file, dtype=np.float32)
    points = np.reshape(points, (-1, 4))  # x,y,z,intensity
    return points


def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

    Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


basepath = 'D:\\课程PPT\\智能机器人概论\\作业\\Final\\dataset\\sequences'
bin_path = basepath + '\\02\\velodyne\\'
scene_bin_path = basepath + '\\02\\scene\\velodyne\\'
label_path = basepath + '\\02\\labels\\'
scene_label_path = basepath + '\\02\\scene\\labels\\'
calib_path = basepath + '\\02\\calib.txt'
pose_path = basepath + '\\02\\poses.txt'
# timestep = 2

# points = read_points(os.path.join(bin_path, "%06d.bin" % (timestep)))
# calib = parse_calibration(calib_path)

# b_o = parse_poses(pose_path, calib)
# b = []
# b = [pose.astype(np.float32) for pose in b_o]
# poses = np.stack(b)

# pose = poses[timestep]
# hpoints = np.hstack(
#     (points[:, :3], np.ones_like(points[:, :1])))
# new_points = np.sum(np.expand_dims(
#     hpoints, 2) * pose.T, axis=1)

new_points_all = []
labels_all = []

l = len(os.listdir(bin_path))

for timestep in range(l):
    if timestep % 16 != 4:
        continue
    old_path = os.path.join(bin_path, "%06d.bin" % (timestep))
    points = read_points(os.path.join(bin_path, "%06d.bin" % (timestep)))
    calib = parse_calibration(calib_path)

    b_o = parse_poses(pose_path, calib)
    b = []
    b = [pose.astype(np.float32) for pose in b_o]
    poses = np.stack(b)

    pose = poses[timestep]
    hpoints = np.hstack(
        (points[:, :3], np.ones_like(points[:, :1])))
    new_points = np.sum(np.expand_dims(
        hpoints, 2) * pose.T, axis=1)
    # new_path = os.path.join(scene_bin_path, "%06d.bin" % (timestep))
    # new_points.tofile(new_path)
    new_points_all.append(new_points)

    old_path_label = os.path.join(label_path, "%06d.label" % (timestep))
    labels = np.fromfile(old_path_label, dtype=np.uint32).reshape((-1, 1))
    labels_all.append(labels)
    
new_points_all = np.vstack(new_points_all)
print(new_points_all.shape)
new_path2 = os.path.join(scene_bin_path, "000001.bin")
new_points_all.tofile(new_path2)

labels_all = np.vstack(labels_all)
print(labels_all.shape)
labels_all = labels_all.reshape(-1)
scene_label_path = os.path.join(scene_label_path, "000001.label")
labels_all.tofile(scene_label_path)
