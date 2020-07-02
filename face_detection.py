
#Import the necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_img(path):
    #read image and copy 
    img = cv2.imread(path)
    img_copy=img.copy()
    return img,img_copy
    
def convertToGRAY(img):
    #convert img to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img
    
def convertToRGB(gray_img):
    #img=cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
    
def Display_GrayImg(gray_img):
    # Displaying the grayscale image
    cv2.imshow('image',gray_img)

def load_cascadeClassifier():
    #load Cascade Classifier
    haar_cascade_face=cv2.CascadeClassifier('./classifier/haarcascade_frontalface_default.xml')
    return haar_cascade_face

def Display_facedetection(RGB_image):
   imgplot = plt.imshow(RGB_image)
   plt.show()

def Detect_Face(CascadeClassifier,img,gray_img):
    #detect faces and return rectangle coordinates 
    faces_rects = CascadeClassifier.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5);
    # Let us print the no. of faces found
    print('Faces found: ', len(faces_rects))
    #draw rectangle 
    for (x,y,w,h) in faces_rects:
         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    RGB_img=convertToRGB(img)
    print(RGB_img.shape)
    return RGB_img
    

if __name__ == '__main__':
  #path for the image
  img_path="./images/faces.jpeg"
  #load image
  img,img_copy=load_img(img_path)
  #Convert image to gray scale
  gray_img=convertToGRAY(img_copy)
  #Display gray image
  Display_GrayImg(gray_img)
  #load CaseCase classifier
  CascadeClassifier=load_cascadeClassifier()
  #Detect faces
  RGB_image=Detect_Face(CascadeClassifier,img,gray_img)
  #Display image
  Display_facedetection(RGB_image)
  
  
  
  
  
  
  
  

    


