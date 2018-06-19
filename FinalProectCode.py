from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import imutils
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import argparse as ap
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler



################################################# BORDER THE IMAGE   #####################################################################################
def borders(image,color):
    #WHITE=[255,255,255]
    image=cv2.copyMakeBorder(image,50,50,20,20,cv2.BORDER_CONSTANT,value=color)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    #cv2.imshow('bordered image',image)
    return image
################################################  BORDER THE IMAGE ENDS  ###################################################################################



################################################# DETECT HUMANS AND MERGE ALL INNER BOXES #################################################
def human_detection(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    #image = cv2.imread('as.jpg')WH
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    #print(pick)
    # draw the final bounding boxes
    #print (pick[0])
    if (len(pick)!=0):
        xA, yA, xB, yB= pick[0]
        #cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        temp=image[yA:yB,xA:xB]
        #cv2.imshow("ROI", temp)
        # show the output images
        #cv2.imshow("Before NMS", orig)
        #cv2.imshow("After NMS", image)
        return temp
    return []
################################################# detect HUMANS AND MERGE ALL INNER BOXES ends #################################################


#----- Get ALL HUMANS in DIFFERENT IMAGES  -------- will be done in future ----#


#################################### FACE DETECTION - SILVER JEWELLER DETECTION  ---- KASHMIRI CLASSIFIED #################################################
def find_face(img):
    img=borders(img,[0,0,0])
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    lower_silver = np.array([0,0,125])
    upper_silver = np.array([180,21,255])
    #120  53  24
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #print (img_hsv)
    imgy=cv2.inRange(img_hsv, lower_silver, upper_silver)
    blackcount=0
    whitecount=0
    v=0
    #print (y)
    noface = -1
    for (x, y, w, h) in faces:
        noface =0
        blackcount=1
        whitecount=0
        xnew=x+np.int(w/8)
        ynew=y-np.int(h/3.5)
        wnew=x+w - np.int(w/8)
        hnew=y+np.int(h/10)
        temp1=xnew
        temp2=ynew
        while (temp1 <= wnew):
            temp2=ynew
            while(temp2 <= hnew):
                v=imgy[temp2,temp1]
                if(v==0):
                    blackcount=blackcount+1
                else:
                    whitecount+=1
                temp2=temp2+1
            temp1=temp1+1
        #print (blackcount)
        #print (whitecount)
        if(whitecount / blackcount != 0.0):
            return (whitecount / blackcount)
    return noface
################################### FACE DETECTION SILVER JEWELLER DETECTION  ---- KASHMIRI CLASSIFIED ends#################################################


#################################################COLAR DETECTION --------- HARYANA CLASSIFIED ###################################################################
def colar_detection(img):
    # Get the training classes names and store them in a list
    cascadePath = "haarcascade_frontalface_default.xml"
    train_path = "D:\\t\\"
    training_names= os.listdir(train_path)

    image_paths = []  # Inilialising the list
    image_classes = []  # Inilialising the list
    class_id = 0
    for (i, training_name) in enumerate(training_names):
            label = training_name.split(os.path.sep)[-1].split("_")[0]
            temp='b'
            if label=='haryana':
                temp='Collar'
            else:
                temp='Non Collar'   
            image_paths.append(training_name)
            image_classes.append(temp)
            class_id+=1
              
    sift=cv2.xfeatures2d.SIFT_create()
    # List where all the descriptors are stored
    des_list = []
    # Reading the image and calculating the features and corresponding descriptors
    for image_pat in image_paths:
        image_path=train_path+image_pat
        im = cv2.imread(image_path)
        #print (im)
        #======================face detection and ROI==================#
        faceCascade = cv2.CascadeClassifier(cascadePath)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        # Draw a rectangle around the faces
        W=0
        H=0
        X=0
        Y=0
        for (x, y, w, h) in faces:
            W=w
            H=h
            X=x
            Y=y
        if(W==0 & H==0 & X==0 & Y==0):
            temp_train=im
        else:
            temp_train=im[Y+H:Y+H+40,X-20:X+W+20]
        temp2_train=cv2.cvtColor(temp_train,cv2.COLOR_BGR2GRAY)
        ret_train,thresh_train = cv2.threshold(temp2_train,225,255,0)
        #======================End face detection and ROI==================#

        kpts, des = sift.detectAndCompute(temp2_train,None)
        des_list.append((image_path, des))  # Appending all the descriptors into the single list

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  # Stacking the descriptors

    # Perform k-means clustering
    k = 50  # Number of clusters
    voc, variance = kmeans(descriptors, k, 1)  # Perform Kmeans with default values

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    # Calculating the number of occurrences
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
    # Giving weight to one that occurs more frequently

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features) # Scaling the visual words for better Prediction

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    samples = im_features
    responses = np.array(image_classes) 
    classes_names = training_names 
    voc = voc
    clf = KNeighborsClassifier()
    #Use - rawImages and labels for traing the model.
    clf.fit(samples,responses)
    #args = vars(parser.parse_args())
    image_paths1 = [img]
    #cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    W=0
    H=0
    X=0
    Y=0
    for (x, y, w, h) in faces:
        W=w
        H=h
        X=x
        Y=y
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.imshow('faces',img)
    ##---------END FACE DETECTION----------##

    ##---------COLAR DETECTION----------##
    temp=img[Y+H:Y+H+40,X-20:X+W+20]
    cv2.imshow('ROI',temp)
    temp2=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(temp2,225,255,0)

    # List where all the descriptors are stored
    des_list1 = []
    kpts1, des1 = sift.detectAndCompute(thresh,None)
    des_list1.append((image_paths1, des1))   # Appending the descriptors to a single list
       
    # Stack all the descriptors vertically in a numpy array
    descriptors1 = des_list1[0][1]
    test_features1 = np.zeros((len(image_paths1), k), "float32")
    for i in range(len(image_paths1)):
        words1, distance1 = vq(des_list1[i][1],voc)
        for w in words1:
            test_features1[i][w] += 1  # Calculating the histogram of features

    # Perform Tf-Idf vectorization
    nbr_occurences1 = np.sum( (test_features1 > 0) * 1, axis = 0)  # Getting the number of occurrences of each word
    idf1 = np.array(np.log((1.0*len(image_paths1)+1) / (1.0*nbr_occurences1 + 1)), 'float32')

    op=clf.predict(test_features1)
    return op
###########################################################COLAR DETECTION ----- HARYANA CLASSIFIED ends #####################################################



################################################# COLOR DETECTION --------- LUCKNOW CLASSIFIED ##########################################################
def color_range(img):
    #range one
    #H - full range     S - 0% to 15%      V - 75% to 100%
    lower1 = np.array([0,0,190])
    upper1 = np.array([179,40,255])

    #range two
    #H - full range     S - 0% to 23%      V - 80% to 100%
    lower2 = np.array([0,0,205])
    upper2 = np.array([179,60,255])

    #range three
    #H - full range     S - 0% to 38%      V - 95% to 100%
    lower3 = np.array([0,0,242])
    upper3 = np.array([179,95,255])

    #120  53  24
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #print (img_hsv)
    img_range1=cv2.inRange(img_hsv, lower1, upper1)
    img_range2=cv2.inRange(img_hsv, lower2, upper2)
    img_range3=cv2.inRange(img_hsv, lower3, upper3)

    height, width = img.shape[:2]
    midh=np.int(height/2)
    midw=np.int(width/2)
    temph= np.int(height/4)
    tempw= np.int(width/4)
    x= midw - np.int(tempw/2)
    y= midh - np.int(temph/2)
    w=tempw
    h=temph
##    print (x)
##    print (y)
##    print (w)
##    print (h)
    blackcount=1
    whitecount=0
    tempi=x
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.imshow('image',img)
##    cv2.imshow('1',img_range1)
##    cv2.imshow('2',img_range2)
##    cv2.imshow('3',img_range3)
##    #print (x, y, x+w, y+h)
    while(tempi < (x+w)):
        tempj=y
        while(tempj < (y+h)):
            v1=img_range1[tempj,tempi]
            v2=img_range2[tempj,tempi]
            v3=img_range3[tempj,tempi]
            #print (v1 , v2  , v3, tempi, tempj)
            if(v1==255 or v2==255 or v3==255):
                #print ("YES")
                whitecount+=1
            else:
                blackcount=blackcount+1
            tempj=tempj+1
        tempi=tempi+1
##    print (blackcount)
##    print (whitecount)
##    print (whitecount / blackcount)
    if ((whitecount/blackcount)>1):
        return 1
    else:
        return 0
################################################# COLOR DETECTION --------- LUCKNOW CLASSIFIED ENDS #########################################################

############################################################### PUNJABI PATIYALA Vs LEHENGA ###############################################################
def patila_vs_lehenga(img):
    # Get the training classes names and store them in a list
    #cascadePath = "haarcascade_frontalface_default.xml"
    train_path = "D:\\tt\\"
    training_names= os.listdir(train_path)

    image_paths = []  # Inilialising the list
    image_classes = []  # Inilialising the list
    class_id = 0
    for (i, training_name) in enumerate(training_names):
            label = training_name.split(os.path.sep)[-1].split(" ")[0]
            temp='b'
            if label=='lehenga':
                temp='legenga'
            else:
                temp='patiyala'   
            image_paths.append(training_name)
            image_classes.append(temp)
            class_id+=1
              
    sift=cv2.xfeatures2d.SIFT_create()
    # List where all the descriptors are stored
    des_list = []
    # Reading the image and calculating the features and corresponding descriptors
    for image_pat in image_paths:
        image_path=train_path+image_pat
        im = cv2.imread(image_path)
        #print (image_path)
        dst=edges_detected(im)
        ret_train,thresh_train = cv2.threshold(dst,225,255,0)
        kpts, des = sift.detectAndCompute(dst,None)
        des_list.append((image_path, des))  # Appending all the descriptors into the single list

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  # Stacking the descriptors

    # Perform k-means clustering
    k = 50  # Number of clusters
    voc, variance = kmeans(descriptors, k, 1)  # Perform Kmeans with default values

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    # Calculating the number of occurrences
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
    # Giving weight to one that occurs more frequently

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features) # Scaling the visual words for better Prediction

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    samples = im_features
    responses = np.array(image_classes) 
    classes_names = training_names 
    voc = voc
    clf = LinearRegression()
    #Use - rawImages and labels for traing the model.
    clf.fit(samples,responses)
    #args = vars(parser.parse_args())
    image_paths1 = [img]
    height, width = img.shape[:2]
    img=img[np.int(height/2):height,0:width]
    #cv2.imshow('in image',img)
    # List where all the descriptors are stored
    des_list1 = []
    dst=edges_detected(img)
    ret_train,thresh_train = cv2.threshold(dst,225,255,0)
    kpts1, des1 = sift.detectAndCompute(dst,None)
    des_list1.append((image_paths1, des1))   # Appending the descriptors to a single list
       
    # Stack all the descriptors vertically in a numpy array
    descriptors1 = des_list1[0][1]
    test_features1 = np.zeros((len(image_paths1), k), "float32")
    for i in range(len(image_paths1)):
        words1, distance1 = vq(des_list1[i][1],voc)
        for w in words1:
            test_features1[i][w] += 1  # Calculating the histogram of features

    # Perform Tf-Idf vectorization
    nbr_occurences1 = np.sum( (test_features1 > 0) * 1, axis = 0)  # Getting the number of occurrences of each word
    idf1 = np.array(np.log((1.0*len(image_paths1)+1) / (1.0*nbr_occurences1 + 1)), 'float32')

    op=clf.predict(test_features1)
    #print (op)
    return op
    
############################################################### PUNJABI PATIYALA Vs LEHENGA ENDS ###############################################################


def edges_detected(img):
    img3 = cv2.resize(img, (200,300))
    img2 = cv2.cvtColor(img3,cv2.COLOR_RGB2GRAY)
    im = cv2.medianBlur(img2,3)
    #cv2.imshow('black and white',img)
    Z = im.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((im.shape))
    #cv2.imshow('res2',res2)

    #creating a kernel matrix
    kernel=[[1,1,1],
            [1,-8,1],
            [1,1,1]]
    kernel = np.array(kernel)
    #applying filter2D to gray image for getting Laplacian edges
    dst = cv2.filter2D(res2,-1,kernel)
    #cv2.imshow('shw',dst)
    return dst

#################################################DRESS LENGTH DETECTION ------------ PUNJABI CLASSIFIED #################################################
#################################################DRESS LENGTH DETECTION ------------ PUNJABI CLASSIFIED ENDS #################################################


#################################################REMAiNING IS GUJARATI #################################################



i=1
j=35
while (i<6):
    t='final ('+str(i)+').jpg'
    print (t)
    image = cv2.imread(t)
    #cv2.imshow('original', image)
    image_bordered = borders(image,[255,255,255])
    humandetected_image=human_detection(image_bordered)
    if(len(humandetected_image) == 0):
        #print ("NO HUMAN DETECTED")
        #-- probable that there is only face in the image or only half image is present--#
        silver_raito=find_face(image_bordered)
        if (silver_raito==-1):
            print ("---REGRET---")
        elif (silver_raito>=0.3):
            print ("IS KASHMIRI")
        #--proceed for further inspection --#
        else:
            output=colar_detection(image_bordered)
            if(output[0]=='Collar'):
                print ("IS HARYANA")
            else:
                print("--REGRET--")
            
    else:
        #print ("HUMAN DETECTED")
        #image_bordered = borders(humandetected_image)
##        cv2.imshow("humans detected", humandetected_image)
        isLuck = color_range(humandetected_image)
        if(isLuck==1):
            print("IS LUCKNOWI")
        else:
            op=patila_vs_lehenga(humandetected_image)
            if(op[0]=='patiyala'):
                print("IS PUNJABI")
            else:
                silver_raito=find_face(humandetected_image)    
                if (silver_raito==-1):
                    print ("IS GUJARATI")
                elif (silver_raito>=0.3):
                    print ("IS KASHMIRI-")
                #--proceed for further inspection --#
                else:
                    output=colar_detection(image_bordered)
                    if(output[0]=='Collar'):
                        print ("IS HARYANA")
                    else:
                        print("IS GUJARATI")
    i=i+1
cv2.waitKey(0)
cv2.destroyAllWindows()


    

