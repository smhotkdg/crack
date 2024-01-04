from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from config import Config as cfg
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torch
import os
import linecache
import time
import math
import heatmap
import matplotlib.pylab as plt
import glob
import cv2 as cv
from scipy.spatial import distance as dist

import math
import pandas as pd

from scipy import ndimage
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
import imutils
# Define midpoint coordinate operation

#import isect_segments_bentley_ottmann.poly_point_isect as bot
#import bentley_ottmann as bot
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

count =0
maxcount =0
width = 0
height = 0
strSave_resultPath = ''
def test(test_data_path='data/test_dataPath.txt',
         save_path='deepcrack_results/result',
         pretrained_model='checkpoints/DeepCrack_CT260_FT1.pth',postion =0):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = 'deepcrack_results/result'
    test_pipline = dataReadPip(transforms=None)

    test_list = readIndex(test_data_path)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))

    model.eval()
    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(
                device)
            test_pred = trainer.val_op(test_data, test_target)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            #save_pred = torch.zeros((512 * 2, 512))
            save_pred = torch.zeros((512, 512))
            #save_pred[:512, :] = test_pred
            save_pred= test_pred
            #save_pred[512:, :] = lab.cpu().squeeze()
            ori_save = lab.cpu().squeeze()
            #saveOri_Path = os.path.join(save_path, 'ori.jpg')            
            save_name = os.path.join(save_path, os.path.split(names[1])[1])
            save_pred = save_pred.numpy() * 255            
            resultIamge = save_pred.astype(np.uint8)
            resultIamge = cv2.resize(save_pred,dsize=(512,512))
            cv2.imwrite(save_name, resultIamge)

def testInit():    
    strTestPath = cfg.classification_path
    print('#######')
    print(strTestPath)
    print('#######')
    #p = str(Path(strTestPath))  # os-agnostic
    p = os.path.abspath(strTestPath)  # absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # 
    myCheckFile = open('data/check.txt', 'r')
    save_path= cfg.detect_save_path
    save_path_2= cfg.detect_path
    detect_path = cfg.detect_path
    
    list_Pos =[]
    # 7 20
    resultPath = 'data/test_dataPath.txt'
    global count
    global maxcount
    for i in range(len(files)):
        global strSave_resultPath
        print(cfg.detect_image_save_path+str(i))        
        if not os.path.exists(cfg.detect_image_save_path+str(i)):            
            os.makedirs(cfg.detect_image_save_path+str(i))   
        
        strSave_resultPath = cfg.detect_image_save_path+str(i)+'/'
        print("#################")
        print(strSave_resultPath)
        strOriPath = files[i]
        #strOriPath = 'D:/sample/23.jpg"
        for fileCount in range(3):
      
            if(count ==3):                
                count =0
            image = splitImage(strOriPath)
            
           
            #print(image)
            print("#################")
            #image = cv2.imread(strOriPath)         
            image = cv2.GaussianBlur(image,(5,5),0)           
            #image = cv2.resize(image, dsize=(3584,10240))       
            image = cv2.resize(image, dsize=(4096,4096))       
            save_Resultimage = image
            save_Resultimage_bin = image
            myFile_result = open(resultPath, 'w') 
            for w in range(8):
                for h in range(8):                
                    #save_pred = torch.zeros((512, 512))                
                    
                    save_pred = image[512*h:(512)+512*h,512*w:(512)+512*w]    
                    #print(save_pred)
                    #scaleCheck 
                    save_pred = cv2.resize(save_pred,dsize=(512,512))
                    saveResultPath = save_path +str(w)+'_'+str(h) +'.jpg'                                
                    cv2.imwrite(saveResultPath,save_pred)
                    myFile_result.writelines(saveResultPath)
                    myFile_result.writelines(' ')
                    myFile_result.writelines(saveResultPath)
                    myFile_result.writelines('\n')
                    #print(saveResultPath)
            myFile_result.close()
            ##여기 주석 풀어야함
            test('data/test_dataPath.txt','deepcrack_results/result','checkpoints/DeepCrack_CT260_FT1.pth',i)           
            #####################################################################
            
            #다시 재 조합
            #
            CheckMinCrack(i)
            
            #strOriPath = splitImage(files[i])
            image = splitImage(files[i])
            image = cv2.resize(image, dsize=(4096,4096))       
            save_Resultimage = image
            save_Resultimage_bin = image
            
            for w in range(8):
                for h in range(8):                
                    #save_pred = torch.zeros((512, 512))                
                    
                    save_pred = image[512*h:(512)+512*h,512*w:(512)+512*w]    
                    #print(save_pred)
                    #scaleCheck 
                    save_pred = cv2.resize(save_pred,dsize=(512,512))
                    saveResultPath = save_path +str(w)+'_'+str(h) +'.jpg'                                
                    cv2.imwrite(saveResultPath,save_pred)        

            binMake(fileCount)
            myFile_result = open(resultPath, 'r')      
            iindex =0
            for w in range(8):
                for h in range(8):          
                    list_Pos.append(2)
                    #deepRresultImage = testImageSet(save_path +str(w)+'_'+str(h) +'.jpg',detect_path +str(w)+'_'+str(h) +'.jpg',list_Pos[iindex])                                   
                    iindex+=1
                    saveResultPath = save_path +str(w)+'_'+str(h) +'.jpg'                                
                    #cv2.imwrite(saveResultPath,deepRresultImage)
                    deepRresultImage = cv2.imread(saveResultPath)  
                    save_Resultimage[512*h:(512)+512*h,512*w:(512)+512*w] = deepRresultImage                
            save_Resultimage = cv2.resize(save_Resultimage, dsize=(3872,3872))          
            print('########### make bin ############')     
            print(strSave_resultPath+str(fileCount)+'result.jpg')
            print('########### make bin ############')     
            cv2.imwrite(strSave_resultPath+str(fileCount)+'result.jpg',save_Resultimage)

            #time.sleep(2)
            drawBin(strOriPath,fileCount)
            count+=1
        #재조합
        maekOrigin()
        maxcount+=1
def maekOrigin():
    global maxcount
    oriPath = cfg.detect_image_save_path+str(maxcount)+'/'

    print("#############merge#############")
    print(width)
    if(width != 18592):
        bin_result = np.zeros((11616,3872,3),np.uint8)
        result_result = np.zeros((11616,3872,3),np.uint8)
        deep_result = np.zeros((11616,3872,3),np.uint8)
        deepBin_result = np.zeros((11616,3872,3),np.uint8)
        deepMergeBin_result = np.zeros((11616,3872,3),np.uint8)
        for h in range(3):               
            binImg = cv2.imread(oriPath+str(h)+'binresult.jpg')              
            reulstImg = cv2.imread(oriPath+str(h)+'result.jpg')  
            deepImage = cv2.imread(oriPath+'Deepresult_'+str(h)+'.jpg')  
            deepBinImag = cv2.imread(oriPath+'Deepresult_'+str(h)+'_Bin.jpg')  
            deepMergeImag = cv2.imread(oriPath+'Deepresult_result_'+str(h)+'.jpg')  

           
            print(binImg.shape)
            print(bin_result.shape)
            
            bin_result[3872*h :(3872)+3872*h, 0: 3872]=binImg     
            result_result[3872*h :(3872)+3872*h, 0: 3872]=reulstImg     
            deep_result[3872*h :(3872)+3872*h, 0: 3872]=deepImage     
            deepBin_result[3872*h :(3872)+3872*h, 0: 3872]=deepBinImag     
            deepMergeBin_result[3872*h :(3872)+3872*h, 0: 3872]=deepMergeImag     
        print("##########################")
        
        #print(source_path+str(maxcount)+'.jpg')
        bin_result = cv2.resize(bin_result, dsize=(width,10000))    
        result_result = cv2.resize(result_result, dsize=(width,10000))    
        deep_result = cv2.resize(deep_result, dsize=(width,10000))    
        deepBin_result = cv2.resize(deepBin_result, dsize=(width,10000))    
        deepMergeBin_result = cv2.resize(deepMergeBin_result, dsize=(width,10000))    
        cv2.imwrite(oriPath+'mergeBin.jpg',bin_result)
        cv2.imwrite(oriPath+'ori.jpg',result_result)
        cv2.imwrite(oriPath+'deep.jpg',deep_result)
        cv2.imwrite(oriPath+'mergeResult.jpg',deepBin_result)
        cv2.imwrite(oriPath+'mergeMergeResult.jpg',deepMergeBin_result)
    else :        
        bin_result = np.zeros((10000,width,3),np.uint8)
        result_result = np.zeros((10000,width,3),np.uint8)
        deep_result = np.zeros((10000,width,3),np.uint8)
        deepBin_result = np.zeros((10000,width,3),np.uint8)
        deepMergeBin_result = np.zeros((10000,width,3),np.uint8)
        for h in range(3):               
            binImg = cv2.imread(oriPath+str(h)+'binresult.jpg')              
            reulstImg = cv2.imread(oriPath+str(h)+'result.jpg')  
            deepImage = cv2.imread(oriPath+'Deepresult_'+str(h)+'.jpg')  
            deepBinImag = cv2.imread(oriPath+'Deepresult_'+str(h)+'_Bin.jpg')  
            deepMergeImag = cv2.imread(oriPath+'Deepresult_result_'+str(h)+'.jpg')  

           
            print(binImg.shape)
            print(bin_result.shape)
            
            bin_result[ 0:10000, 6198*h:(6198)+6198*h] =binImg     
            result_result[ 0:10000, 6198*h:(6198)+6198*h] =reulstImg     
            deep_result[ 0:10000, 6198*h:(6198)+6198*h] =deepImage     
            deepBin_result[ 0:10000, 6198*h:(6198)+6198*h] =deepBinImag     
            deepMergeBin_result[ 0:10000, 6198*h:(6198)+6198*h] =deepMergeImag     
        print("##########################")
        
        #print(source_path+str(maxcount)+'.jpg')
        bin_result = cv2.resize(bin_result, dsize=(width,10000))    
        result_result = cv2.resize(result_result, dsize=(width,10000))    
        deep_result = cv2.resize(deep_result, dsize=(width,10000))    
        deepBin_result = cv2.resize(deepBin_result, dsize=(width,10000))    
        deepMergeBin_result = cv2.resize(deepMergeBin_result, dsize=(width,10000))    
        cv2.imwrite(oriPath+'mergeBin.jpg',bin_result)
        cv2.imwrite(oriPath+'ori.jpg',result_result)
        cv2.imwrite(oriPath+'deep.jpg',deep_result)
        cv2.imwrite(oriPath+'mergeResult.jpg',deepBin_result)
        cv2.imwrite(oriPath+'mergeMergeResult.jpg',deepMergeBin_result)
def splitImage(instrOriPath):
    global width
    global height
    image = cv2.imread(instrOriPath)
    if(image.shape[1]== 18592):
        width = 18592
        height = 10000
    else:
        width = image.shape[1]
    if(width != 18592):         
        image = cv2.resize(image, dsize=(width,11616))    
        save_pred = image[3872*count :(3872)+3872*count, 0: 3872]                       
        return save_pred
    else :        
        image = cv2.resize(image, dsize=(width,10000))                                     
        #6198
        save_pred = image[ 0:10000, 6198*count:(6198)+6198*count]                       
        return save_pred   
def binMake(i): 
    detect_path = cfg.detect_path
    save_Resultimage = np.zeros((4096,4096,3),np.uint8)
    for w in range(8):
        for h in range(8):              
            deepRresultImage = detect_path +str(w)+'_'+str(h) +'.jpg'      
            openImage = cv2.imread(deepRresultImage)                                                    
            save_Resultimage[512*h:(512)+512*h,512*w:(512)+512*w] = openImage                
            #return
    save_Resultimage = cv2.resize(save_Resultimage, dsize=(3872,3872))        
    global strSave_resultPath       
    cv2.imwrite(strSave_resultPath+str(i)+'Binresult.jpg',save_Resultimage)
def CheckMinCrack(pos):    
    resultPath = 'data/test_dataPath.txt'    
    save_path= cfg.detect_save_path

    detect_path = cfg.detect_save_path


    myFile_result = open(resultPath, 'w')     
    for w_main in range(8):
        for h_main in range(8): 
            mixDetectPath = detect_path+str(w_main)+'_'+str(h_main)+'.jpg'                         
            mixImg = cv2.imread(mixDetectPath)         
                    
            gray_mix = cv2.cvtColor(mixImg, cv2.COLOR_BGR2GRAY)
            gray_mix = cv2.resize(gray_mix,dsize=(512,512))     

            ret_mix, dst_mix =cv2.threshold(gray_mix, 0, 255, cv2.THRESH_BINARY)    
            
            bSave =True
            count_mix =0
            for h_mix in range(512):
                for w_mix in range(512):
                    if(dst_mix[h_mix,w_mix] > 0):                    
                        count_mix +=1   
                    dst_mix[h_mix,w_mix] =0
            if(count_mix  > 3000):
                bSave =False                              
            else:
                cv2.imwrite(mixDetectPath,dst_mix)    
            if(bSave == False):                
                saveResultPath = save_path +str(w_main)+'_'+str(h_main) +'.jpg'    
                myFile_result.writelines(saveResultPath)
                myFile_result.writelines(' ')
                myFile_result.writelines(saveResultPath)
                myFile_result.writelines('\n') 
                print(saveResultPath)
            #ret_mix, dst_mix =cv2.threshold(gray_mix, 200, 255, cv2.THRESH_BINARY)    
            #cv2.imwrite(mixDetectPath,dst_mix)
    myFile_result.close()
def drawBin(strPath,pos):
    global strSave_resultPath
    #binFilePath = cfg.detect_image_save_path
    binFilePath = strSave_resultPath
    strTestPath = cfg.classification_Data_path
    #p = str(Path(strTestPath))  # os-agnostic
    p = os.path.abspath(strTestPath)  # absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files

    # 0 line 1 fatiue

    #for i in range(len(test_list)):        
    #for i in range(1):        

    strOriPath = strPath
    strBinPath = binFilePath+str(pos)+'Binresult.jpg'
    
    image = cv2.imread(binFilePath+str(pos)+'result.jpg')         
    #image = splitImage(strOriPath)
    imgBin = cv2.imread(strBinPath)
    strOriFilePath =files[pos]
    
    myFile = open(strOriFilePath, 'r')

    lineCheckImage_gray = cv2.cvtColor(imgBin, cv2.COLOR_BGR2GRAY)            
    ret, dst =cv2.threshold(lineCheckImage_gray, 50, 255, cv2.THRESH_BINARY)   
    #result_Bin = np.zeros(imgBin.shape[0],imgBin.shape[1])
    result_Bin = np.zeros((3872,3872,3),np.uint8)
    # for x_pos in range(3872):
    #         for y_pos in range(3872):
    #             if(dst[y_pos,x_pos] >0):
    #                 image[y_pos,x_pos] = (255,0,0)

    print('detect level 1')
    
    while True:
        line = myFile.readline()
        if not line : break
        
        crackType =  int(line.split(' ')[0])
        TopPos = int(line.split(' ')[1].replace('(','').replace(',',''))            
        TopLeftPos = int(line.split(' ')[2].replace(')',''))            
        bottomPos = int(line.split(' ')[3].replace('(','').replace(',',''))            
        bottomRightPos = int(line.split(' ')[4].replace(')','').replace('\n',''))
        
        
        if(crackType ==0):
            #x
            for x in range(abs(bottomPos - TopPos)):
                for y in range(abs(bottomRightPos - TopLeftPos)):
                    if(dst[TopLeftPos+y,TopPos+x] >0):
                        image[TopLeftPos+y,TopPos+x]= (0,255,0)          
                        result_Bin[TopLeftPos+y,TopPos+x] = (255,255,255)
        else:
            for x in range(abs(bottomPos - TopPos)):
                for y in range(abs(bottomRightPos - TopLeftPos)):
                    if(dst[TopLeftPos+y,TopPos+x] >0):
                        image[TopLeftPos+y,TopPos+x]= (0,0,255)        
                        result_Bin[TopLeftPos+y,TopPos+x] = (255,255,255)        
    
    cv2.imwrite(strSave_resultPath+'Deepresult_'+str(pos)+'.jpg',image)
    cv2.imwrite(strSave_resultPath+'Deepresult_'+str(pos)+'_Bin.jpg',result_Bin)
    measure(result_Bin,image,pos)
    print(pos)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def measure(img,image_ori,pos):  
    global strSave_resultPath 
    
    image = image_ori.copy()
    binFilePath = cfg.detect_image_save_path
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Binary image 
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    # Calculate the coordinates of the four corner points of the black square
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    for cnt in contours:
        M = cv.moments(cnt)        
        x, y, w, h = cv.boundingRect(cnt)    	
        rect = cv.minAreaRect(cnt)        
        box = cv.boxPoints(rect)        
        box = np.int0(box)
       

        if M['m00'] != 0:
        	# print(M)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #Based on the center point obtained by the geometric distance, draw the center circle, blocked by the blue line, so you can't see it.
            cv.circle(image_ori,(np.int(cx),np.int(cy)),2,(0,255,255),-1) 
            # 
            cv.rectangle(image_ori, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # 4 
            cv.drawContours(image_ori, [box], 0, (0, 0, 255), 1)

            roi = image_ori[y:y + h,x:x + w]    
            if(roi.shape[0] ==0 or roi.shape[1] ==0 or roi.shape[2] ==0):
                    print("empty")
            else:  
                gray_roi = cv.cvtColor(roi, cv.COLOR_RGB2GRAY)
                ret_roi,thresh_roi=cv2.threshold(gray_roi,50,255,cv2.THRESH_BINARY_INV)
            
                areaCount = cv2.countNonZero(thresh_roi)
                #count += 1 
            for (x, y) in box:
                cv2.circle(image_ori, (int(x), int(y)), 1, (0, 0, 255), -1)
                # tl upper left corner image coordinate, tr upper right corner image coordinate, br lower right corner image coordinate, bl lower left corner image coordinate
                (tl, tr, br, bl) = box
                # Calculate the center point of the 4 sides of the red frame
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                # 
                cv2.circle(image_ori, (int(tltrX), int(tltrY)), 1, (255, 0, 0), -1)
                cv2.circle(image_ori, (int(blbrX), int(blbrY)), 1, (255, 0, 0), -1)
                cv2.circle(image_ori, (int(tlblX), int(tlblY)), 1, (255, 0, 0), -1)
                cv2.circle(image_ori, (int(trbrX), int(trbrY)), 1, (255, 0, 0), -1)
                #  4 points, that is, 2 blue lines in the picture

                cv2.line(image_ori, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 0), 1)
                cv2.line(image_ori, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 0), 1)
                # Calculate the coordinates of the center point
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                # Convert the image length to the actual length, 1 is equivalent to the scale, I use the mm unit, that is, 1mm is equivalent to 1 images

                dimA = dA 
                dimB = dB 
                # Print the calculation result on the original image, which is the yellow content.

                #cv2.putText(image_ori, "Area {:.1f}".format(areaCount),
                    #(int(tltrX - 25), int(tltrY - 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    #0.25, (0, 0, 0), 1)
                cv2.putText(image_ori, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (0, 0, 0), 1)
                cv2.putText(image_ori, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (0, 0, 0), 1)  
                Totalarea = dimA*dimB
                if(Totalarea >0):
                    if(areaCount >Totalarea):
                        percent = (Totalarea/areaCount) *100
                    else:
                        percent = (areaCount/Totalarea) *100
                    cv2.putText(image_ori, "Area {:.1f} %".format(percent),
                        (int(tltrX - 30), int(tltrY - 30)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (0, 0, 0), 1)
    cv2.imwrite(strSave_resultPath+'Deepresult_result_'+str(pos)+'.jpg',image_ori)

if __name__ == '__main__':    
    #test()
    testInit()
    
    
    
    