import numpy as np
import cv2
#from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#from keras.models import load_model
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from PIL import Image
import pdb
import torchvision
import torchvision.transforms as transforms

model=torch.load('model/lanenet.pkl')
model.to('cpu')
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []
def road_lines_test(image):
#     print(image.shape)
    image_new=Image.fromarray(image,mode='RGB')
    #image_new.show()
    #注释掉的一直出问题是因为少了mode，依然是tuple？
    #small_img=np.array(Image.fromarray(image).resize(80.160))
    small_img=image_new.resize((80,160),Image.ANTIALIAS)
    transform=transforms.Compose([transforms.ToTensor()])
    images_tensor = transform(small_img) #转为Tensor
    images_tensor=images_tensor.resize(1,3,80,160)
    prediction=model(images_tensor)
    
    prediction=prediction.detach().numpy()
    p = np.squeeze(prediction)
    p*=255
    p=np.expand_dims(p,axis=2)
#     print(p.shape)
    p_RGB = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)
    lane_image=cv2.resize(p_RGB,(852,480),interpolation=cv2.INTER_CUBIC)
    return lane_image
lanes=Lanes()
vid_output='project_video_hy.mp4'
clip1=VideoFileClip('project_video.mp4')
vid_clip=clip1.fl_image(road_lines_test)
vid_clip.write_videofile(vid_output,audio=False)