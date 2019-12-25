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
    #print(images_tensor)
#     print(images_tensor.shape)
    #images_tensor=images_tensor.to(device)
    prediction=model(images_tensor)
    
    prediction=prediction.detach().numpy()
    p = np.squeeze(prediction)
    p*=255
    
#     im4=Image.fromarray(p)
#     im4.show()
    #因为忘了乘255，全是黑的
#     prediction*=255
#     p = np.squeeze(prediction)
#     pdb.set_trace()
#     im3=Image.fromarray(p)
#     im3.show()
    p=np.expand_dims(p,axis=2)
#     print(p.shape)
    p_RGB = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)
#     print(p_RGB.shape)
#     print(type(p_RGB))
#     cv2.imshow("output",p_RGB)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     im3=Image.fromarray(p_RGB)
#     im3.show()
    #print(type(prediction))
    
    
#     lanes.recent_fit.append(p_RGB)
#     if len(lanes.recent_fit)>5:
#         lanes.recent_fit=lanes.recent_fit[1:]
#     lanes.avg_fit=np.mean(np.array([i for i in lanes.recent_fit]),axis=0)
#     blanks=np.zeros_like(lanes.recent_fit).astype(np.uint8)
#     pdb.set_trace()
#     lanes_drawn=np.dstack((blanks,lanes.recent_fit,blanks)) 

#     lanes_drawn=np.array(lanes_drawn)
#     cv2.imshow("output",lanes_drawn)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     lanes_drawn=Image.fromarray(lanes_drawn,mode='RGB')
#     lanes_drawn.show()
#     lane_image=lanes_drawn.resize((852,480),Image.ANTIALIAS)
#     lanes_drawn=np.squeeze(lanes_drawn) #shape(80,480,3)
#     lanes_drawn=Image.fromarray(lanes_drawn,mode='RGB')
#     lanes_drawn.show()
#     lane_image=lanes_drawn.resize((852,480),Image.ANTIALIAS)
    lane_image=cv2.resize(p_RGB,(852,480),interpolation=cv2.INTER_CUBIC)
    
#     pdb.set_trace()
#     result=cv2.addWeighted(image,160,lane_image,1,0.)

#     cv2.imshow("output",lane_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()  
#     im = Image.fromarray(result)
#     im.show()
    return lane_image
lanes=Lanes()
vid_output='project_video_hy.mp4'
clip1=VideoFileClip('project_video.mp4')
vid_clip=clip1.fl_image(road_lines_test)
vid_clip.write_videofile(vid_output,audio=False)