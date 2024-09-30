import torch
#import torch.nn.functional as F
import torchvision.transforms.v2 as T
import numpy as np
import comfy.utils
import comfy.model_management as mm
import gc
from tqdm import tqdm
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
import folder_paths
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
script_directory = os.path.dirname(os.path.abspath(__file__))


def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)
    #return T.ToTensor()(Image.fromarray(image)).permute(1, 2, 0)

  
class FFTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE","FFTData")
    RETURN_NAMES = ("image","FFTData")
    FUNCTION = "toFFT"
    CATEGORY = "FFT"
    def toFFT(self, image):
        FFTImageList=[]
        FFT_Channel_Data=[]    
        channelCount=3
        imageCount=image.shape[0]
        # print("image.shape:",image.shape) 
        # ##   torch.Size([1, 64, 64, 3])  表示来了1张图，64*64，3通道     loadImage节点过来就是这个           
        # print("type of image:",type(image))  ##  type of image: <class 'torch.Tensor'>
        # pixel = image[0][0][0]
        # print("pixel:",pixel)  ##  pixel: tensor([0., 0., 0.])
        # tValue=pixel[0]
        # print("Type of tValue:",type(tValue))  ##  type of tensors_out: <class 'torch.Tensor'>

        for i in range(image.shape[0]):
            sourceImg = image[i]
            cv2Image = (sourceImg.contiguous() * 255).byte()
        #image1=image[0]
        #print(image1[0][0])
        ## tensor([0., 0., 0.])
        ## 是0-1之间的浮点数

            dim = sourceImg.dim()
            if dim == 3:

                #print(cv2Image.shape)
                ##torch.Size([64, 64, 3])  变成0-255的byte, 颜色值, 3通道
                R_channel = cv2Image[:, :,0]
                G_channel = cv2Image[:, :,1]
                B_channel = cv2Image[:, :,2]

                fshiftData=[]
                #傅里叶变换
                R_fft = np.fft.fft2(R_channel)
                R_fshift = np.fft.fftshift(R_fft)
                fshiftData.append(R_fshift)
                G_fft = np.fft.fft2(G_channel)
                G_fshift = np.fft.fftshift(G_fft)
                fshiftData.append(G_fshift)
                B_fft = np.fft.fft2(B_channel)
                B_fshift = np.fft.fftshift(B_fft)
                fshiftData.append(B_fshift)

                R_img = np.log(np.abs(R_fshift))
                B_img = np.log(np.abs(B_fshift))
                G_img = np.log(np.abs(G_fshift))

                R_img = R_img / np.max(R_img)
                B_img = B_img / np.max(B_img)
                G_img = G_img / np.max(G_img)

                fftImg = np.dstack((R_img, G_img, B_img))
                fftImg = fftImg.astype(np.float32)
                #image = image.astype(np.uint8)
                FFT_Channel_Data.append(fshiftData)
                # # # one_channel = cv2Image[:, :,2]
                #print(red_channel.shape)
                ##  torch.Size([64, 64])  拿到其中一个颜色通道
                
                # # # f = np.fft.fft2(one_channel)
                # # # #dft = cv2.dft(np.float32(one_channel), flags = cv2.DFT_COMPLEX_OUTPUT)
                # # # FFT_Data.append(f)
                # # # #print(f.shape)
                # # # ## (64, 64)  现在不是torch, tensor了，是个二维数组

                # # # #默认结果中心点位置是在左上角,
                # # # #调用fftshift()函数转移到中间位置
                # # # fshift = np.fft.fftshift(f)       

                # # # #fft结果是复数, 其绝对值结果是振幅
                # # # fimg = np.log(np.abs(fshift))
                # # # #print(fimg.shape)
                # # # ##  (64, 64)
                # # # ## 取复数的模，然后取对数。仅用于显示，原始数据还在f里
                # # # ##  此时是取对数后的float结果，范围不一定是0-255，也不一定是0-1，可能很大也可能很小
                # # # max_v = np.max(fimg)
                # # # fimg = fimg / max_v
                # # # print(max_v)

                # # # img_3_channels = np.repeat(fimg[:, :, np.newaxis], 3, axis=2)
                #print(img_3_channels.shape)
                ##  (64, 64, 3)  复制到其余两个通道。让它变成灰度图    
                #print(img_3_channels[0][0])
                FFTImageList.append(fftImg)
            else:
                channelCount=1
                fshiftData=[]
                #傅里叶变换
                R_fft = np.fft.fft2(cv2Image)
                R_fshift = np.fft.fftshift(R_fft)
                fshiftData.append(R_fshift)
                fftImg = np.log(np.abs(R_fshift))
                fftImg = fftImg / np.max(fftImg)
                fftImg = fftImg.astype(np.float32)
                FFTImageList.append(fftImg)
                FFT_Channel_Data.append(fshiftData)

        tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in FFTImageList])
        )
        FFT_Data={'channelCount':channelCount,'FFT_Channel_Data':FFT_Channel_Data,'imageCount':imageCount}

        # print("in FFTNode, before output")
        # print("type of tensors_out:",type(tensors_out))
        # print("tensors_out shape:",tensors_out.shape)
        # print("type of tensors_out[0]:",type(tensors_out[0]))
        # print("tensors_out shape[0]:",tensors_out[0].shape)
        # pixel = tensors_out[0][0][0]
        # print("pixel:",pixel)  ##  pixel: tensor([0., 0., 0.])
        # tValue=pixel[0]
        # print("Type of tValue:",type(tValue))  ##  Type of tValue: <class 'torch.Tensor'>
        return (tensors_out,FFT_Data)
    
def ApplyMaskAndInvert(l_fshift,l_mask):
    # print("in fucntion, mask shape:",l_mask.shape)
    # print("in fucntion, fshift shape:",l_fshift.shape)
    f = l_fshift * l_mask
    ishift = np.fft.ifftshift(f)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)      
    return f,iimg

class InvertFFTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ff": ("FFTData", ),
                "filterRadius": ("INT", {
                    "default": 100,
                })
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("image","IMAGE","IMAGE",)
    FUNCTION = "fromFFT"
    CATEGORY = "FFT"
    def DoOneChannel(self,fshift,filterRadius):
        FFTImageList=[]
        #f=ff[0][0]
        #fshift = np.fft.fftshift(f)
#傅里叶逆变换
        #ishift = np.fft.ifftshift(fshift)
        iimg = np.fft.ifft2(fshift)
        iimg = np.abs(iimg)
        FFTImageList.append(iimg)

#设置高通滤波器
        rows, cols = fshift.shape
        crow,ccol = int(rows/2), int(cols/2)
        mask = np.ones((rows, cols))
        mask[crow-filterRadius:crow+filterRadius, ccol-filterRadius:ccol+filterRadius] = 0
        #fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

        mask2 = np.zeros((rows, cols))
        mask2[crow-filterRadius:crow+filterRadius, ccol-filterRadius:ccol+filterRadius] = 1

#傅里叶逆变换
        fshiftmask,hiPassImg = ApplyMaskAndInvert(fshift,mask)
        FFTImageList.append(mask*255)
        # # ishift = np.fft.ifftshift(fshift)
        # # iimg = np.fft.ifft2(ishift)
        # # iimg = np.abs(iimg)
        # print("np.max(hiPass):",np.max(hiPassImg))
        # print("np.min(hiPass):",np.min(hiPassImg))
        FFTImageList.append(hiPassImg)

        fimg = np.log(np.abs(fshiftmask))
        #print(fimg.shape)
        ##  (64, 64)
        ## 取复数的模，然后取对数。仅用于显示，原始数据还在f里
        ##  此时是取对数后的float结果，范围不一定是0-255，也不一定是0-1，可能很大也可能很小
        max_v = np.max(fimg)
        fimg = fimg / max_v * 254
        # print("max_v of fimg:",max_v)
        # print("max_v of after fimg:",np.max(fimg))
             
        FFTImageList.append(fimg)        
        fshiftmask,lowPassImg = ApplyMaskAndInvert(fshift,mask2)
        FFTImageList.append(mask2*255)
        # print("np.max(lowPassImg):",np.max(lowPassImg))
        # print("np.min(lowPassImg):",np.min(lowPassImg))
        fimg = np.log(np.abs(fshiftmask))
        #print(fimg.shape)
        ##  (64, 64)
        ## 取复数的模，然后取对数。仅用于显示，原始数据还在f里
        ##  此时是取对数后的float结果，范围不一定是0-255，也不一定是0-1，可能很大也可能很小
        max_v = np.max(fimg)
        fimg = fimg / max_v * 254
        # print("max_v of fimg:",max_v)
        # print("max_v of after fimg:",np.max(fimg))
        #img_3_channels = np.repeat(fimg[:, :, np.newaxis], 3, axis=2)        
        FFTImageList.append(fimg)     
        FFTImageList.append(lowPassImg)

        tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in FFTImageList])/255
        )
        return tensors_out
    def fromFFT(self, ff,filterRadius):
        f0=ff[0][0]
        #f1=ff[0][1]
        #f2=ff[0][2]
        out1=self.DoOneChannel(f0,filterRadius)
        #print("C1")
        #out2=self.DoOneChannel(f1,filterRadius)
        #print("C2")
        #out3=self.DoOneChannel(f2,filterRadius)
        return (out1,out1,out1,)
    
    
class InvertFFTWithMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ff": ("FFTData", ),
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fromFFT"
    CATEGORY = "FFT"
    def DoOneChannel(self,fshift,mask):
        fshiftmask,hiPassImg = ApplyMaskAndInvert(fshift,mask)
        return hiPassImg/255
    def fromFFT(self, ff,mask):
        mask0=mask[0]
        maskNP=np.array(mask0)
        channelCount = ff['channelCount']
        imageCount = ff['imageCount']
        ### ff['FFT_Channel_Data'] 是数组，每个元素对应一张图片。每个成员也是数组。
        ### ff['FFT_Channel_Data'][0] 是一个数组，对应一张图片。成员数量是3或者1，对应3个通道或者1个通道
        res=[]
        for i in range(imageCount):
            if channelCount==3:
                f0 = ff['FFT_Channel_Data'][i][0]
                #print("type of f0:",type(f0))
                out0=self.DoOneChannel(f0,maskNP)
                #print("type of out0:",type(out0))
                #print("out0 shape:",out0.shape)
                f1 = ff['FFT_Channel_Data'][i][1]
                out1=self.DoOneChannel(f1,maskNP)
                #print("type of out1:",type(out1))
                #print("out1 shape:",out1.shape)
                f2 = ff['FFT_Channel_Data'][i][2]
                out2=self.DoOneChannel(f2,maskNP)
                #print("type of out2:",type(out2))
                #print("out2 shape:",out2.shape)
                doneImg = np.dstack((out0, out1, out2))
                doneImg = doneImg.astype(np.float32)
                #print("doneImg shape:",doneImg.shape)
                res.append(doneImg)
            else:
                f0 = ff['FFT_Channel_Data'][i]
                doneImg=self.DoOneChannel(f0,maskNP)
                doneImg = doneImg.astype(np.float32)
                #doneImg = out0
                res.append(doneImg)
        #print("C1")
        #out2=self.DoOneChannel(f1,filterRadius)
        #print("C2")
        #out3=self.DoOneChannel(f2,filterRadius)
        tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in res])
        )
        # print("in InvertFFTWitMask, before output")
        # print("type of tensors_out:",type(tensors_out))
        # print("tensors_out shape:",tensors_out.shape)

        # print("type of tensors_out[0]:",type(tensors_out[0]))
        # print("tensors_out shape[0]:",tensors_out[0].shape)

        # pixel = tensors_out[0][0][0]
        # print("pixel shape:",pixel.shape)
        # print("pixel:",pixel)
        # tvalue=pixel[0]
        # print("tvalue:",tvalue)
        # print("tvalue type:",type(tvalue))
        
        return (tensors_out,)
    
class FindFFTSpot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"FFTImage": ("IMAGE", ),
                "ff": ("FFTData", ),
                "hiPassFactor": ("INT", {"default": 10, "min": 1, "max": 1024}),
                "centerRange": ("INT", {"default": 30, "min": 1, "max": 1024}),
                "crossWidth": ("INT", {"default": 30, "min": 0, "max": 100}),
                "threshold": ("INT", {"default": 100, "min": 0, "max": 254}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "FindSpot"
    CATEGORY = "FFT"
    def FindSpot(self, ff,hiPassFactor,centerRange,crossWidth,threshold):
        FFTImageList=[]
        #imageCount=FFTImage.shape[0]
        
        channelCount = ff['channelCount']
        imageCount = ff['imageCount']

        for i in range(imageCount):

            if(channelCount==1):
                ffData = ff['FFT_Channel_Data'][i]
            else:
                ffData = ff['FFT_Channel_Data'][i][0]

            rows, cols = ffData.shape
            crow,ccol = int(rows/2), int(cols/2) #中心位置
            mask = np.ones((rows,cols),np.uint8)
            mask[crow-hiPassFactor:crow+hiPassFactor, ccol-hiPassFactor:ccol+hiPassFactor] = 0

            _,R_channel = ApplyMaskAndInvert(ffData,mask)




            # sourceImg=FFTImage[i]
            # cv2Image = (sourceImg.contiguous() * 255).byte()
            # dim = channelCount            
            # if dim == 3:
            #     R_channel = cv2Image[:, :,0]
            # else:
            #     R_channel = cv2Image
            # #print("R_channel shape:",R_channel.shape)
            # R_channel = np.asanyarray(R_channel)
            # #print("R_channel AS NP shape:",R_channel.shape)
            
            min_val, max_val = np.min(R_channel), np.max(R_channel)
            out1 = (R_channel - min_val)*(255.0/(max_val - min_val))

            rows, cols = R_channel.shape
            crow,ccol = int(rows/2), int(cols/2) #中心位置
            
            out1[crow-centerRange:crow+centerRange, ccol-centerRange:ccol+centerRange] = 0
            out1[:, ccol-crossWidth:ccol+crossWidth] = 0
            out1[crow-crossWidth:crow+crossWidth, :] = 0
            if(threshold>0):
                out1[out1<threshold]=0
                out1[out1>threshold]=255
            fftImg = np.dstack((out1, out1, out1))  
            fftImg = fftImg.astype(np.float32)         
            FFTImageList.append(fftImg)

        tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in FFTImageList])/255
        )
        # print("find spot")
        # print("type of tensors_out:",type(tensors_out))
        # print("tensors_out shape:",tensors_out.shape)

        # print("type of tensors_out[0]:",type(tensors_out[0]))
        # print("tensors_out shape[0]:",tensors_out[0].shape)
        
        # pixel = tensors_out[0][0][0]
        # print("pixel:",pixel)  ##  pixel: tensor([0., 0., 0.])
        # tValue=pixel[0]
        # print("Type of tValue:",type(tValue))  ##  Type of tValue: <class 'float'>

        return (tensors_out,)

   

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "InvertFFTNode":InvertFFTNode,
    "FFTNode":FFTNode,    
    "InvertFFTWithMask":InvertFFTWithMask,
    "FindFFTSpot":FindFFTSpot,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     "InvertFFTNode":"InvertFFTNode",
     "FFTNode":"FFTNode",
     "InvertFFTWithMask":"InvertFFTWithMask",
     "FindFFTSpot":"FindFFTSpot",
     
}
