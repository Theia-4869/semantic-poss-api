import os
import numpy as np
from PIL import Image

semantic_dir = '../demo/frames_semantic'
instence_dir = '../demo/frames_instance'
panoptic_dir = '../demo/frames_panoptic'

for k in range(1, 501):
    semantic_file = os.path.join(semantic_dir, 'scan_{:04}.png'.format(k))
    semantic_frame= Image.open(semantic_file)
    semantic_frame = semantic_frame.convert("RGB")
    semantic_array = semantic_frame.load()

    instance_file = os.path.join(instence_dir, 'scan_{:04}.png'.format(k))
    instance_frame= Image.open(instance_file)
    instance_frame = instance_frame.convert("RGB")
    instance_array = instance_frame.load()

    width, height = semantic_frame.size
    for i in range(width):
        for j in range(height):
            if instance_array[i, j][0] == instance_array[i, j][1] and instance_array[i, j][1] == instance_array[i, j][2]:
                continue
            semantic_array[i, j] = instance_array[i, j]
    panoptic_file = os.path.join(panoptic_dir, 'scan_{:04}.png'.format(k))
    semantic_frame.save(panoptic_file)
