import numpy as np
import cv2 as cv
import os

W = 1800 # width of range image
H = 40   # height of range image
LABEL_DICT = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1", # standing sign
    11: "traffic sign 2", # hanging sign
    12: "traffic sign 3", # high/big hanging sign
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground"} # class definition
SEM_COLOR = np.array([
    [0, 0, 0],                       # 0: "unlabeled"
    [0, 0, 0], [0, 0, 0], [0, 0, 0], # don't care
    [255, 30, 30],                   # 4: "1 person"
    [255, 30, 30],                   # 5: "2+ person"
    [255, 40, 200],                  # 6: "rider"
    [100, 150, 245],                 # 7: "car"
    [135,60,0],                      # 8: "trunk"
    [0, 175, 0],                     # 9: "plants"
    [255, 0, 0],                     # 10: "traffic sign 1"
    [255, 0, 0],                     # 11: "traffic sign 2"
    [255, 0, 0],                     # 12: "traffic sign 3"
    [255, 240, 150],                 # 13: "pole"
    [125, 255, 0],                   # 14: "trashcan"
    [255, 200, 0],                   # 15: "building"
    [50, 255, 255],                  # 16: "cone/stone"
    [255, 120, 50],                  # 17: "fence"
    [0,0,0],[0,0,0],[0,0,0],         # don't care
    [100, 230, 245],                 # 21: "bike"
    [128, 128, 128]],                # 22: "ground"
    dtype = np.uint8) # color definition

def read_points(bin_file):
    points = np.fromfile(bin_file, dtype = np.float32)
    points = np.reshape(points,(-1,4)) # x,y,z,intensity
    return points

def read_semlabels(label_file):
    semlabels = np.fromfile(label_file, dtype = np.uint32) & 0xffff
    return semlabels

def read_inslabels(label_file):
    inslabels = np.fromfile(label_file, dtype = np.uint32) >> 16
    return inslabels

def get_rangeimage(bin_file, tag_file):
    points = read_points(bin_file)
    tags = np.fromfile(tag_file, dtype = np.bool)
    dis = np.linalg.norm(points[:,0:3], axis = 1) * 5
    dis = np.minimum(dis, 255)
    dis = dis.astype(np.uint8)
    dis_vec = np.zeros((H*W), dtype = np.uint8)
    dis_vec[tags] = dis
    dis_mat = np.reshape(dis_vec, (H,W))
    rangeimage = cv.cvtColor(dis_mat, cv.COLOR_GRAY2BGR)
    return rangeimage

def get_semimage(label_file, tag_file):
    semlabels = read_semlabels(label_file)
    tags = np.fromfile(tag_file, dtype = np.bool)
    label_vec = np.zeros((H*W), dtype = np.uint32)
    label_vec[tags] = semlabels
    image_vec = SEM_COLOR[label_vec]
    semimage = np.reshape(image_vec, (H,W,3))
    semimage = semimage[:,:,::-1] # RGB
    return semimage

def get_insimage(label_file, tag_file):
    inslabels = read_inslabels(label_file)
    tags = np.fromfile(tag_file, dtype = np.bool)
    label_vec = np.zeros((H*W), dtype = np.uint32)
    label_vec[tags] = inslabels
    image_vec = SEM_COLOR[label_vec%23]
    insimage = np.reshape(image_vec, (H,W,3))
    return insimage

if __name__=='__main__': # an example for generating video of sequence 00
    data_path = '../dataset/sequences/02/'
    videoWriter = cv.VideoWriter('../demo/demo_img.avi',cv.VideoWriter_fourcc('M','J','P','G'),25,(1024,384),True)
    for bin_file,label_file,tag_file in zip(sorted(os.listdir(data_path+'velodyne/')),
            sorted(os.listdir(data_path+'predictions/')),sorted(os.listdir(data_path+'tag/'))):
        rangeimage = get_rangeimage(data_path+'velodyne/'+bin_file, data_path+'tag/'+tag_file)
        rangeimage = cv.resize(rangeimage,(1024,128))
        semimage = get_semimage(data_path+'predictions/'+label_file, data_path+'tag/'+tag_file)
        semimage = cv.resize(semimage,(1024,128))
        insimage = get_insimage(data_path+'predictions/'+label_file, data_path+'tag/'+tag_file)
        insimage = cv.resize(insimage,(1024,128))
        mergeimage = np.concatenate((rangeimage,semimage,insimage),axis = 0)
        videoWriter.write(mergeimage)
    videoWriter.release()