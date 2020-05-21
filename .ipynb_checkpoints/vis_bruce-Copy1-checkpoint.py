import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

from processor.io import IO
import tools
import tools.utils as utils
import tools.utils.visualization as visualization
import time
tt = time.time()
p = IO()

p.model.eval()

# window = p.arg.window
exp = p.arg.exp
# step = p.arg.stp
# batch_ = p.arg.bat
### 1/5
file_json = '../../data/testing/skeleton_result/02_u_10.json'
filename = (file_json.split('/')[-1]).split('.')[0]
with open(file_json, encoding='utf-8') as data_file:
    data = json.loads(data_file.read())
sequence_info = []
weight = data.get('## description').get('image_width')
height = data.get('## description').get('image_height')
height0 = data.get('## description').get('image_height')
for frame_number in data.keys():
    try:
        frame_number = int(frame_number)
    except Exception as e:
        continue
    #Get highest possibilites human confiden score --> usualy at detection no 0
    # print(frame_number)
    id_name = list(data.get(str(frame_number)).keys())[2]

    frame_id = frame_number
    frame_data = {'frame_index': frame_id}
    skeletons = []
    
    score, coordinates = [], []
    skeleton = {}
    
    
    for part in data.get(str(frame_number)).get(id_name):
        #wu tiao jian jin wei
        coordinates += [int(part.get('position')[0]*weight+0.5),int(part.get('position')[1]*height+0.5)]

        # coordinates += [(part.get('position')[0]),(part.get('position')[1])]

        score += [part.get('score')]
    skeleton['pose'] = coordinates
    skeleton['score'] = score
    skeletons +=[skeleton]
    frame_data['skeleton'] = skeletons
    sequence_info += [frame_data]

    

    # set the lenght of video
    # if frame_number == 1000:
    #     break

video_info = dict()
video_info['data'] = sequence_info
video_info['label'] = 'unknowns'
video_info['label_index'] = -1

pose, _ = utils.video.video_info_parsing(video_info, num_person_out=1)
data = torch.from_numpy(pose)
data = data.unsqueeze(0)
data = data.float().to(p.dev).detach()
# data = data.cpu()
# torch.cuda.empty_cache()
print(data.shape)
output, feature, x1, x2, x3, xx = p.model.extract_feature(data)

# extract feature
output = output[0]
feature = feature[0]
intensity = (feature*feature).sum(dim=0)**0.5
intensity = intensity.cpu().detach().numpy()
print('outtttttttttttttttttttttttttttttttttttt')
print(type(output))
torch.save(output,'output.pt')
label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
print(label)

### 2/5
label_name_path = './resource/drilling_skeleton/label_name_'+exp+'.txt'
with open(label_name_path) as f:
    label_name = f.readlines()
    label_name = [line.rstrip() for line in label_name]
print('Prediction result: {}'.format(label_name[label]))
print('Done.')
# visualization


### 3/5
video = utils.video.get_video_frames('../../data/testing/roi_result/'+ filename + '_original.mp4')
print('\nVisualization...')
label_sequence = output.sum(dim=2).argmax(dim=0)
# print('0')
# print(output.shape)
# print('1')
# print(output.sum(dim=2))
# print('2')
# print(output.sum(dim=2).argmax(dim=0))
# print('3')
# print(label_name)
label_name_sequence = [[label_name[p] for p in l ]for l in label_sequence]

# print(label_name_sequence)

### 4/5
# output_result_dir = './visual_result/' + exp + '/window_' + window + '/' + 'step_80_' + step + '/batch_12/'
output_result_dir = p.arg.visual_dir
edge = p.model.graph.edge
print('start visualization')
images = visualization_0425.stgcn_visualize(
    pose, edge, intensity, video, weight, height0, output_result_dir, filename, label_name[label] , label_name_sequence)
# height=152)
print('Done.')
# import cv2
# cv2.putText(images,'abcd',(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# cv2.imshow('result',images)
# save video
print('\nSaving...')


### 4/5
# output_result_dir = './visual_result/' + exp + '/window_' + window + '/' +'step_80_' + step +'/'

### 5/5
output_result_path = output_result_dir + filename + '_output.mp4'
if not os.path.exists(output_result_dir):
    os.makedirs(output_result_dir)
torch.save(output, output_result_dir+'output.pt')
writer = skvideo.io.FFmpegWriter(output_result_path,outputdict={'-b': '300000000'})
print('start writting')
print(output_result_path)
for img in images:
    writer.writeFrame(img)
writer.close()
print(time.time() - tt)