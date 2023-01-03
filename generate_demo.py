from moviepy.editor import *
from moviepy.video.VideoClip import TextClip

# clip1 = VideoFileClip("../demo/demo_pcd_prediction_large.mp4")
# clip2 = VideoFileClip("../demo/demo_pcd_label_large.mp4")
# # clip3 = VideoFileClip("../demo/demo_instance.mp4")
# # clip4 = VideoFileClip("../demo/demo_panoptic.mp4")

# final_clip = clips_array([[clip1],
#                         [clip2]]) 

# final_clip.write_videofile("../demo/demo_pred_gt_large.mp4")

# print("demo video generated")

# # subclip视频截取开始时间和结束时间
video = VideoFileClip("../demo/demo_pred_gt_large.mp4")

# 制作文字，指定文字大小和颜色
txt_clip1 = (TextClip("Semantic Prediction", fontsize=20, color='red')
            .set_position((0+5, 400-25))
            .set_duration(video.duration))
txt_clip2 = (TextClip("Instance Prediction", fontsize=20, color='green')
            .set_position((400+5, 400-25))
            .set_duration(video.duration))
txt_clip3 = (TextClip("Panoptic Prediction", fontsize=20, color='blue')
            .set_position((800+5, 400-25))
            .set_duration(video.duration))
txt_clip4 = (TextClip("Semantic Ground Truth", fontsize=20, color='red')
            .set_position((0+5, 800-25))
            .set_duration(video.duration))
txt_clip5 = (TextClip("Instance Ground Truth", fontsize=20, color='green')
            .set_position((400+5, 800-25))
            .set_duration(video.duration))
txt_clip6 = (TextClip("Panoptic Ground Truth", fontsize=20, color='blue')
            .set_position((800+5, 800-25))
            .set_duration(video.duration))

result = CompositeVideoClip([video, txt_clip1, txt_clip2, txt_clip3, txt_clip4, txt_clip5, txt_clip6])
result.write_videofile("../demo/demo_pred_gt_large_mark.mp4")


