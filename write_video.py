import os
import moviepy
import moviepy.video.io.ImageSequenceClip

if __name__=='__main__':
    data_path = '../demo/frames_pcd_label_large'
    fps = 10
    image_files = []
    for data_idx in os.listdir(data_path):
        filename = os.path.join(data_path, data_idx)
        image_files.append(filename)
    image_files = sorted(image_files,key=lambda x:x[-8:-4])
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile("../demo/demo_pcd_label_large.mp4", fps=fps)