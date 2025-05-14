import numpy as np
import cv2
import time
from os.path import splitext, basename, join

class Colorizer:
    def __init__(self,height=400,width=600):
        (self.height,self.width) = height, width
        
        self.colorModel = cv2.dnn.readNetFromCaffe("model/colorization_deploy_v2.prototxt", caffeModel="model/colorization_release_v2.caffemodel")  #stores the weights of the above model
        
        clusterCenters = np.load("model/pts_in_hull.npy")  #stores the black and white and their corresponding RGB values
        clusterCenters = clusterCenters.transpose().reshape(2,313,1,1)
        
        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1,313], 2.606, np.float32)]
        
        #These two layers are responsible for producing the colorized output image
    
    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        self.img = cv2.resize(self.img, (self.width, self.height))
        
        self.processFrame() #method to process the image
        cv2.imwrite(join("output",basename(imgName)), self.imgFinal, [cv2.IMWRITE_JPEG_QUALITY, 90]) #Saves the processed image to a file in the "output" directory using OpenCV's imwrite function with JPEG compression quality set to 90.
        
        cv2.imshow("output",self.imgFinal) #shows the output in a separate window
    
    def processFrame(self):
        imgNormalized = (self.img[:,:,[2,1,0]] * 1.0/255).astype(np.float32)
        # Normalize the input image by dividing each pixel value by 255 to bring them in the range [0,1]. It also swaps the color channels to RGB using self.img[:,:,[2,1,0]].
        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        # Convert the input image from RGB to LAB color space using cv2.cvtColor(). The LAB color space separates the image into L (luminance), A (green-magenta), and B (blue-yellow) channels.
        channelL = imgLab[:,:,0]
        
        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized,(224,224)),cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:,:,0]
        channelLResized -=  50
        
        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0,:,:,:].transpose((1,2,0))
        
        resultResized = cv2.resize(result, (self.width,self.height))
        
        self.imgOut = np.concatenate((channelL[:,:,np.newaxis], resultResized),axis=2)
        #Concatenate the original L channel with the resized predicted AB channels using np.concatenate()
        self.imgOut = np.clip(cv2.cvtColor(self.imgOut, cv2.COLOR_LAB2BGR), 0,1)
        #Clip the pixel values of the resulting image to the range [0,1] using np.clip()
        self.imgOut = np.array((self.imgOut)*255, dtype = np.uint8)
        
        self.imgFinal = np.hstack((self.img, self.imgOut))
        
    def colorize(self):
        imgNormalized = (self.img[:,:,[2,1,0]] * 1.0/255).astype(np.float32)
        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        channelL = imgLab[:,:,0]
    
        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized,(224,224)),cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:,:,0]
        channelLResized -=  50
    
        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0,:,:,:].transpose((1,2,0))
    
        resultResized = cv2.resize(result, (self.width,self.height))
    
        self.result = np.concatenate((channelL[:,:,np.newaxis], resultResized),axis=2)
        self.result = np.clip(cv2.cvtColor(self.result, cv2.COLOR_LAB2BGR), 0,1)
        self.result = np.array((self.result)*255, dtype = np.uint8)
    
    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)
        success, self.img = cap.read()
        nextFrameTime = 0
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(join("output",splitext(basename(videoName))[0] + '.avi'), fourcc, cap.get(cv2.CAP_PROP_FPS), (self.width * 2, self.height), isColor=True)
        while success:
            self.img = cv2.resize(self.img, (self.width,self.height))
            self.colorize()
            out.write(cv2.hconcat([self.img, self.result]))
            success, self.img = cap.read()
        cap.release()
        out.release()
        cv2.destroyAllWindows()




