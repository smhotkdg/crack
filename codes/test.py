from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from config import Config as cfg
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import linecache
import time
import math
import heatmap
import matplotlib.pylab as plt
#import isect_segments_bentley_ottmann.poly_point_isect as bot
#import bentley_ottmann as bot
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def test(test_data_path='data/test_dataPath.txt',
         save_path='deepcrack_results/result',
         pretrained_model='checkpoints/DeepCrack_CT260_FT1.pth',postion =0):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = 'deepcrack_results/result/'+str(postion)
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
    # with torch.no_grad():
    #     for names, (img, lab) in tqdm(zip(test_list, test_loader)):
    #         test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(
    #             device)
    #         test_pred = trainer.val_op(test_data, test_target)
    #         test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
    #         save_pred = torch.zeros((512 * 2, 512))
    #         save_pred[:512, :] = test_pred
    #         save_pred[512:, :] = lab.cpu().squeeze()
    #         save_name = os.path.join(save_path, os.path.split(names[1])[1])
    #         save_pred = save_pred.numpy() * 255
    #         cv2.imwrite(save_name, save_pred.astype(np.uint8))
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
            
def VeriticalCheck(test_data_path='data/test_dataPath.txt',
         save_path='deepcrack_results/vertical',
         pretrained_model='checkpoints/vertical.pth',postion =0):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = 'deepcrack_results/vertical/'+str(postion)
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

def HorizontalCheck(test_data_path='data/test_dataPath.txt',
         save_path='deepcrack_results/horizontal',
         pretrained_model='checkpoints/horizontal.pth',postion =0):

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = 'deepcrack_results/horizontal/'+str(postion)
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
def testOne(test_data_path='data/test_dataPath.txt',
         save_path='deepcrack_results/result',
         pretrained_model='checkpoints/roadTeck.pth'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

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

            
def testInitOne():
    test_data_path = 'data/test_example.txt'
    test_list = readIndex(test_data_path)
    myFile = open(test_data_path, 'r')
    save_path='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'
    save_path_2='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    
    # 7 20
    resultPath = 'data/test_dataPath.txt'
    
    for i in range(len(test_list)):
        strOriPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/1.jpg'
        image = cv2.imread(strOriPath)                    
        image = cv2.resize(image, dsize=(512,512))       
        save_Resultimage = image
        save_Resultimage_bin = image
        myFile_result = open(resultPath, 'w') 
                    
        #save_pred = torch.zeros((512, 512))                
        
        save_pred = image[0:(512),0:(512)]    
        #print(save_pred)
        save_pred = cv2.resize(save_pred,dsize=(512,512))
        saveResultPath = save_path +'result.jpg'                                
        cv2.imwrite(saveResultPath,save_pred)
        myFile_result.writelines(saveResultPath)
        myFile_result.writelines(' ')
        myFile_result.writelines(saveResultPath)
        myFile_result.writelines('\n')
                #print(saveResultPath)
        myFile_result.close()
    
        #다시 재 조합
        
     
        VeriticalCheck()                
        HorizontalCheck()
        test()   

        CheckMinCrackOne()
        resultPath = 'data/test_dataPath.txt'

        save_path='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'
        detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
        
        myFile_result = open(resultPath, 'r')      
        iindex =0
        #image = cv2.resize(image, dsize=(3584,10240))    
        save_Resultimage = np.zeros((512,512,3),np.uint8)
        
        
        minMaxCount =0       
        for k in range(3):                
            if(k ==0):
                detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
            if(k ==1):
                detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/horizontal/'
            if(k ==2):
                detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/vertical/'
            deepRresultImage,maxCount = testImageSet(save_path +'result.jpg',detect_path+'result.jpg',k)                       
            if(maxCount > minMaxCount):                                                              
                minMaxCount = maxCount
                deepRresultImage = checkRect(deepRresultImage,k)        
            iindex+=1                
            saveResultPath = save_path +'result.jpg'                                
            cv2.imwrite(saveResultPath,deepRresultImage)
            save_Resultimage[:512,:] = deepRresultImage                
                #return
        save_Resultimage = cv2.resize(save_Resultimage, dsize=(512,512))   
        cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/results/'+'0result.jpg',save_Resultimage)

def testInit():
    test_data_path = 'data/test_example.txt'
    test_list = readIndex(test_data_path)
    myFile = open(test_data_path, 'r')
    myCheckFile = open('data/check.txt', 'r')
    save_path='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'
    save_path_2='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    list_Pos =[]
    # 7 20
    resultPath = 'data/test_dataPath.txt'
    
    for i in range(len(test_list)):
        #####################################################################
        strOriPath = 'D:/sample/deepTest/'+str(i)+".jpg"
        #strOriPath = 'D:/sample/23.jpg"
        image = cv2.imread(strOriPath)                    
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
                save_pred = cv2.resize(save_pred,dsize=(256,256))
                saveResultPath = save_path +str(w)+'_'+str(h) +'.jpg'                                
                cv2.imwrite(saveResultPath,save_pred)
                myFile_result.writelines(saveResultPath)
                myFile_result.writelines(' ')
                myFile_result.writelines(saveResultPath)
                myFile_result.writelines('\n')
                #print(saveResultPath)
        myFile_result.close()

        #VeriticalCheck('data/test_dataPath.txt','deepcrack_results/horizontal','checkpoints/horizontal.pth',i)                
        #HorizontalCheck('data/test_dataPath.txt','deepcrack_results/vertical','checkpoints/vertical.pth',i)
        test('data/test_dataPath.txt','deepcrack_results/result','checkpoints/DeepCrack_CT260_FT1.pth',i)           
        #####################################################################
        
        #다시 재 조합
        #
        CheckMinCrack(i)
        
        strOriPath = 'D:/sample/deepTest/'+str(i)+".jpg"
        image = cv2.imread(strOriPath)                    
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

          
        # VeriticalCheck()                
        # HorizontalCheck()
        # test()   
        # #비쥬얼 검출
        # testtest(i)

        binMake(i)
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
        cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/results/'+str(i)+'result.jpg',save_Resultimage)
        # for w in range(8):
        #     for h in range(8):     
        #         binPath = save_path_2 +str(w)+'_'+str(h) +'.jpg'                    
        #         deepRresultImage_bin = cv2.imread(binPath)          
        #         save_Resultimage_bin[512*h:(512)+512*h,512*w:(512)+512*w] = deepRresultImage_bin        
        # deepRresultImage_bin = cv2.resize(save_Resultimage_bin, dsize=(3872,3872))                       
        # cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/results/'+str(i)+'result_bin.jpg',deepRresultImage_bin)
def testPng():
    strOriPath = 'D:/workspace/road-crack-detection/module-test/256/CRKWH100_GT/1000.bmp'
    img = cv2.imread(strOriPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    height = dst.shape[0]
    width = dst.shape[1]
    for y in range(0,height):
        for x in range(0,width):                       
            if(dst[y,x] == 0):                
                dst[y,x] = 100
                
    
    cv2.imwrite('D:/sample/test2/test2.png',dst)

def Horizontal_Image(oriImage,detectImage):
     #detectPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/0.jpg'    
    save_pred = cv2.imread(detectImage)                    
    gray = cv2.cvtColor(save_pred, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,dsize=(512,512))
    gray_sample = cv2.resize(gray,dsize=(512,512))
    #ret, dst = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU)    
    #oriPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/ori.jpg'
    ori_save = cv2.imread(oriImage)
    ori_default = cv2.imread(oriImage)

    ori_save = cv2.resize(ori_save,dsize=(512,512))
    ori_default = cv2.resize(ori_default,dsize=(512,512))
    
    lineCheckImage_gray = ori_default
    lineCheckImage_gray = cv2.cvtColor(lineCheckImage_gray, cv2.COLOR_BGR2GRAY)    
    ret, dst =cv2.threshold(lineCheckImage_gray, 250, 255, cv2.THRESH_BINARY)     
    lineCount =0
    for h in range(512):
        for w in range(512):
            if(dst[h,w] >0):
                lineCount+= 1    
    
    if(lineCount > 10000):
        return 0    
    count =0
    for h in range(512):
        for w in range(512):
            if(gray[h,w]>10):     
                gray_sample[h,w] = 255
                ori_save[h,w,0] =255
                ori_save[h,w,1] =0
                ori_save[h,w,2] =0
                count = count+1    
                cfg.bCheck = True
                return 2       


    return 0
def Vertical_Image(oriImage,detectImage):
     #detectPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/0.jpg'    
    save_pred = cv2.imread(detectImage)                    
    gray = cv2.cvtColor(save_pred, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,dsize=(512,512))
    gray_sample = cv2.resize(gray,dsize=(512,512))
    #ret, dst = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU)    
    #oriPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/ori.jpg'
    ori_save = cv2.imread(oriImage)
    ori_default = cv2.imread(oriImage)

    ori_save = cv2.resize(ori_save,dsize=(512,512))
    ori_default = cv2.resize(ori_default,dsize=(512,512))
    
    lineCheckImage_gray = ori_default
    lineCheckImage_gray = cv2.cvtColor(lineCheckImage_gray, cv2.COLOR_BGR2GRAY)    
    ret, dst =cv2.threshold(lineCheckImage_gray, 250, 255, cv2.THRESH_BINARY)     
    lineCount =0
    for h in range(512):
        for w in range(512):
            if(dst[h,w] >0):
                lineCount+= 1    
    
    if(lineCount > 10000):
        return 0    
    count =0
    for h in range(512):
        for w in range(512):
            if(gray[h,w]>10):     
                gray_sample[h,w] = 255
                ori_save[h,w,0] =0
                ori_save[h,w,1] =255
                ori_save[h,w,2] =0
                count = count+1    
                cfg.bCheck = True
                return 2
         


    return 0
def CheckMixModel():
        #다시 재 조합
    resultPath = 'data/test_dataPath.txt'
    save_path='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result_mix/'
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    detect_path_horizontal = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/horizontal/'
    detect_path_vertical = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/vertical/'
    
    myFile_result = open(resultPath, 'r')      
    iindex =0
    #image = cv2.resize(image, dsize=(3584,10240))    
    save_Resultimage = np.zeros((10240,3584,3),np.uint8)
    for w in range(7):
        for h in range(20):          
            for k in range(3):
                mixResult = cv2.imread(detect_path +str(w)+'_'+str(h) +'.jpg')                
                horizontal_result = cv2.imread(detect_path_horizontal +str(w)+'_'+str(h) +'.jpg')
                vertical_result = cv2.imread(detect_path_vertical +str(w)+'_'+str(h) +'.jpg')
                gray_mix = cv2.cvtColor(mixResult, cv2.COLOR_BGR2GRAY)
                gray_horizontal = cv2.cvtColor(horizontal_result, cv2.COLOR_BGR2GRAY)
                gray_vertical = cv2.cvtColor(vertical_result, cv2.COLOR_BGR2GRAY)

                ret_mix, dst_mix =cv2.threshold(gray_mix, 0, 255, cv2.THRESH_BINARY)     
                ret_hori, dst_hori =cv2.threshold(gray_horizontal, 0, 255, cv2.THRESH_BINARY)     
                ret_ver, dst_ver =cv2.threshold(gray_vertical, 0, 255, cv2.THRESH_BINARY)     
                for h_local in range(512):
                    for w_local in range(512):
                        if(dst_mix[h_local,w_local] == dst_hori[h_local,w_local] or dst_mix[h_local,w_local] == dst_ver[h_local,w_local]):
                            mixResult[h_local,w_local] =0
                cv2.imwrite(save_path +str(w)+'_'+str(h)+'.jpg',mixResult)

def testImageSet(oriImage,detectImage,iindex):
    #detectPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/0.jpg'        
    save_pred = cv2.imread(detectImage)                    
    gray = cv2.cvtColor(save_pred, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,dsize=(512,512))
    gray_sample = cv2.resize(gray,dsize=(512,512))
    #ret, dst = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU)    
    #oriPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/ori.jpg'
    ori_save = cv2.imread(oriImage)
    ori_save_temp = cv2.imread(oriImage)
    ori_default = cv2.imread(oriImage)

    ori_save = cv2.resize(ori_save,dsize=(512,512))
    ori_default = cv2.resize(ori_default,dsize=(512,512))
    
    lineCheckImage_gray = ori_default
    lineCheckImage_gray = cv2.cvtColor(lineCheckImage_gray, cv2.COLOR_BGR2GRAY)    
    ret, dst =cv2.threshold(lineCheckImage_gray, 250, 255, cv2.THRESH_BINARY)     
    lineCount =0
    for h in range(512):
        for w in range(512):
            if(dst[h,w] >0):
                lineCount+= 1    
    
    if(lineCount > 10000):
        return ori_save,-1  
    count =0
    for h in range(512):
        for w in range(512):
            if(gray[h,w]>10):     
                gray_sample[h,w] = 255      
                if(iindex ==0):
                    ori_save[h,w,0] =0
                    ori_save[h,w,1] =0
                    ori_save[h,w,2] =255
                if(iindex ==1):
                    ori_save[h,w,0] =255
                    ori_save[h,w,1] =0
                    ori_save[h,w,2] =0
                if(iindex ==2):
                    ori_save[h,w,0] =255
                    ori_save[h,w,1] =0
                    ori_save[h,w,2] =0
                count = count+1    
            else:
                gray_sample[h,w] =0

    #outlineDetection
    # if(count < 500 and iindex != 0):
    #     return ori_save_temp,-1
    # elif(iindex == 0 and count < 5000):
    #     return ori_save_temp,-1
    if(iindex ==0):
        img1 = gray_sample.copy()    
        ret, thresh = cv2.threshold(img1,10,255,0)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
        cv2.drawContours(ori_save, contours, -1,(0,255,0), 0)
        
    if(iindex ==1):
        img1 = gray_sample.copy()    
        ret, thresh = cv2.threshold(img1,10,255,0)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
        cv2.drawContours(ori_save, contours, -1,(0,255,255), 0)
        
        
    if(iindex ==2):
        img1 = gray_sample.copy()    
        ret, thresh = cv2.threshold(img1,10,255,0)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
        cv2.drawContours(ori_save, contours, -1,(0,255,255), 0)        
        
    if(iindex ==0):
        count = count*4
    return ori_save,count
def checkRect(ori_saveImage,iIndex):
    if(iIndex == -1):
        return
    if(iIndex ==0):        
        cv2.rectangle(ori_saveImage,(0,0),(512,512),(0,0,255),2)
        return ori_saveImage
    else:        
        cv2.rectangle(ori_saveImage,(0,0),(512,512),(0,255,255),2)
        return ori_saveImage
def CheckMinCrack(pos):    
    resultPath = 'data/test_dataPath.txt'    
    save_path='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'

    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'+str(pos)+'/'
    #detect_path_horizontal = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/horizontal/'+str(pos)+'/'
    #detect_path_vertical = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/vertical/'+str(pos)+'/'

    myFile_result = open(resultPath, 'w')     
    for w_main in range(7):
        for h_main in range(20): 
            mixDetectPath = detect_path+str(w_main)+'_'+str(h_main)+'.jpg'
            #horizontalPath = detect_path_horizontal+str(w_main)+'_'+str(h_main)+'.jpg'
            #verticalPath = detect_path_vertical + str(w_main)+'_'+str(h_main)+'.jpg'
            #save_pred = cv2.imread(detectImage)                        
            mixImg = cv2.imread(mixDetectPath)
            #HorizontalImg = cv2.imread(horizontalPath)
            #VerticalImg = cv2.imread(verticalPath)
                    
            gray_mix = cv2.cvtColor(mixImg, cv2.COLOR_BGR2GRAY)
            #gray_Horizontal = cv2.cvtColor(HorizontalImg, cv2.COLOR_BGR2GRAY)
            #gray_Vertical = cv2.cvtColor(VerticalImg, cv2.COLOR_BGR2GRAY)

            gray_mix = cv2.resize(gray_mix,dsize=(512,512))
            #gray_Horizontal = cv2.resize(gray_Horizontal,dsize=(512,512))
            #gray_Vertical = cv2.resize(gray_Vertical,dsize=(512,512))

            ret_mix, dst_mix =cv2.threshold(gray_mix, 0, 255, cv2.THRESH_BINARY)     
            #ret_hori, dst_hori =cv2.threshold(gray_Horizontal, 0, 255, cv2.THRESH_BINARY)     
            #ret_ver, dst_ver =cv2.threshold(gray_Vertical, 0, 255, cv2.THRESH_BINARY)     
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
                #print(str(count_mix) +'mix'+str(h_main)+'_'+str(w_main))
                
            # count_horizontal =0
            # for h_hori in range(512):
            #     for w_hori in range(512):
            #         if(dst_hori[h_hori,w_hori]>0):                    
            #             count_horizontal +=1   
            #         dst_hori[h_hori,w_hori] =0
            # if(count_horizontal  > 3000):
            #     bSave =False                
            # else:
            #     cv2.imwrite(horizontalPath,dst_hori)
            #     #print(str(count_horizontal) +'horizontal'+str(h_main)+'_'+str(w_main))

            # count_vertical =0
            # for h_ver in range(512):
            #     for w_ver in range(512):
            #         if(dst_ver[h_ver,w_ver]>0):                    
            #             count_vertical +=1   
            #         dst_ver[h_ver,w_ver] =0
            # if(count_vertical  > 3000):
            #     bSave = False                
            # else:
            #     cv2.imwrite(verticalPath,dst_ver)
            #     #print(str(count_vertical) +'vertical'+str(h_main)+'_'+str(w_main))

            if(bSave == False):                
                saveResultPath = save_path +str(w_main)+'_'+str(h_main) +'.jpg'    
                myFile_result.writelines(saveResultPath)
                myFile_result.writelines(' ')
                myFile_result.writelines(saveResultPath)
                myFile_result.writelines('\n') 
                print(saveResultPath)
    myFile_result.close()
def testtest(index):    
      #다시 재 조합
    resultPath = 'data/test_dataPath.txt'
    save_path='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    
    myFile_result = open(resultPath, 'r')      
    iindex =0
    #image = cv2.resize(image, dsize=(3584,10240))    
    save_Resultimage = np.zeros((10240,3584,3),np.uint8)
    
    for w in range(7):
        for h in range(20):   
            minMaxCount =0       
            for k in range(3):                
                if(k ==0):
                    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'+str(index)+'/'
                if(k ==1):
                    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/horizontal/'+str(index)+'/'
                if(k ==2):
                    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/vertical/'+str(index)+'/'
                deepRresultImage,maxCount = testImageSet(save_path +str(w)+'_'+str(h) +'.jpg',detect_path+str(w)+'_'+str(h) +'.jpg',k)                       
                if(maxCount > minMaxCount):                                                              
                    minMaxCount = maxCount
                    deepRresultImage = checkRect(deepRresultImage,k)        
                iindex+=1                
                saveResultPath = save_path +str(w)+'_'+str(h) +'.jpg'                                
                cv2.imwrite(saveResultPath,deepRresultImage)
                save_Resultimage[512*h:(512)+512*h,512*w:(512)+512*w] = deepRresultImage                
            #return
    save_Resultimage = cv2.resize(save_Resultimage, dsize=(3550,10000))               
    cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/results/'+str(index)+'result.jpg',save_Resultimage)
def CheckMinCrackOne():    
    resultPath = 'data/test_dataPath.txt'    
    save_path='D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'

    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    detect_path_horizontal = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/horizontal/'
    detect_path_vertical = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/vertical/'

    myFile_result = open(resultPath, 'w')     
  
    mixDetectPath = detect_path+'result.jpg'
    horizontalPath = detect_path_horizontal+'result.jpg'
    verticalPath = detect_path_vertical + 'result.jpg'
    #save_pred = cv2.imread(detectImage)                        
    mixImg = cv2.imread(mixDetectPath)
    HorizontalImg = cv2.imread(horizontalPath)
    VerticalImg = cv2.imread(verticalPath)
            
    gray_mix = cv2.cvtColor(mixImg, cv2.COLOR_BGR2GRAY)
    gray_Horizontal = cv2.cvtColor(HorizontalImg, cv2.COLOR_BGR2GRAY)
    gray_Vertical = cv2.cvtColor(VerticalImg, cv2.COLOR_BGR2GRAY)

    gray_mix = cv2.resize(gray_mix,dsize=(512,512))
    gray_Horizontal = cv2.resize(gray_Horizontal,dsize=(512,512))
    gray_Vertical = cv2.resize(gray_Vertical,dsize=(512,512))

    ret_mix, dst_mix =cv2.threshold(gray_mix, 0, 255, cv2.THRESH_BINARY)     
    ret_hori, dst_hori =cv2.threshold(gray_Horizontal, 0, 255, cv2.THRESH_BINARY)     
    ret_ver, dst_ver =cv2.threshold(gray_Vertical, 0, 255, cv2.THRESH_BINARY)     
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
        #print(str(count_mix) +'mix'+str(h_main)+'_'+str(w_main))
        
    count_horizontal =0
    for h_hori in range(512):
        for w_hori in range(512):
            if(dst_hori[h_hori,w_hori]>0):                    
                count_horizontal +=1   
            dst_hori[h_hori,w_hori] =0
    if(count_horizontal  > 3000):
        bSave =False                
    else:
        cv2.imwrite(horizontalPath,dst_hori)
        #print(str(count_horizontal) +'horizontal'+str(h_main)+'_'+str(w_main))

    count_vertical =0
    for h_ver in range(512):
        for w_ver in range(512):
            if(dst_ver[h_ver,w_ver]>0):                    
                count_vertical +=1   
            dst_ver[h_ver,w_ver] =0
    if(count_vertical  > 3000):
        bSave = False                
    else:
        cv2.imwrite(verticalPath,dst_ver)
        #print(str(count_vertical) +'vertical'+str(h_main)+'_'+str(w_main))

    if(bSave == False):                
        saveResultPath = save_path  +'result.jpg'    
        myFile_result.writelines(saveResultPath)
        myFile_result.writelines(' ')
        myFile_result.writelines(saveResultPath)
        myFile_result.writelines('\n') 
        print(saveResultPath)
    myFile_result.close()
def invertColor():    
    str_invertPath='D:/invert/'    
    for i in range(42):
        strOriPath = 'D:/invert/'+str(i+1)+'.jpg'
        image = cv2.imread(strOriPath)                 
        image = cv2.bitwise_not(image)
        cv2.imwrite('D:/invert/result2/'+str(i+1)+'.jpg',image)

def binMake(i): 
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'+str(i)+'/'
    save_Resultimage = np.zeros((4096,4096,3),np.uint8)
    for w in range(8):
        for h in range(8):              
            deepRresultImage = detect_path +str(w)+'_'+str(h) +'.jpg'      
            openImage = cv2.imread(deepRresultImage)                                                    
            save_Resultimage[512*h:(512)+512*h,512*w:(512)+512*w] = openImage                
            #return
    save_Resultimage = cv2.resize(save_Resultimage, dsize=(3872,3872))               
    cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/results/'+str(i)+'Binresult.jpg',save_Resultimage)
def lineDetection():
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/test.jpg'                
    img = cv2.imread(detect_path)     
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    kernel_size_row = 5
    kernel_size_col = 5
    kernel = np.ones((5, 5), np.uint8)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 150  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    points = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 )
            if(dist > 250):
                #points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'+str(0)+'edges.jpg',lines_edges)   
def contoursTest():
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/0_0.jpg'                
    image = cv2.imread(detect_path)  
       
    #hm = heatmap.Heatmap()             
    #hm_img_1ch = hm.heatmap(100, dotsize=120, opacity=50, scheme ='classic', size=(image.shape[1],image.shape[0]),area=((0, 0), (image.shape[1],image.shape[0])))

    cv_img = cv2.cvtColor(np.asarray(hm_img_1ch), cv2.COLOR_RGBA2BGR)
    cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'+str(0)+'hm_img_1ch.jpg',cv_img)   

#512x512 이미지 저장 위치
strDivisionPath = 'D:/sample/save/'
#원본 이미지 위치
strDivisionOriPath = 'D:/sample/0.jpg'
#병합된 이미지 저장 
strAbsorptionPath = 'D:/sample/Absorption.jpg'

#이미지를 512x512로 자름
def ImageDivision():
    image = cv2.imread(strDivisionOriPath)
    image = cv2.resize(image, dsize=(3584,10240))               
    for w in range(7):
        for h in range(20):                
            save_pred = image[512*h:(512)+512*h,512*w:(512)+512*w]           
            save_pred = cv2.resize(save_pred,dsize=(512,512))
            saveResultPath = strDivisionPath +str(w)+'_'+str(h) +'.jpg'                                
            cv2.imwrite(saveResultPath,save_pred)

#512x512로 된 이미지를 병합함
def Imageabsorption():       
    
    absorptionImage = np.zeros((10240,3584,3),np.uint8)
    for w in range(7):
        for h in range(20):              
            deepRresultImage = strDivisionPath +str(w)+'_'+str(h) +'.jpg'      
            openImage = cv2.imread(deepRresultImage)                                                    
            absorptionImage[512*h:(512)+512*h,512*w:(512)+512*w] = openImage                
            #return
    absorptionImage = cv2.resize(absorptionImage, dsize=(3550,10000))               
    cv2.imwrite(strAbsorptionPath,absorptionImage)

def yoloFileMaker():
    strhoitFilePath = 'H:/crack_train/10m/hoit'
    strSavePath = 'D:/yolov5-master/yolov5-master/data/images/'
    arr = os.listdir(strhoitFilePath)
    
    
    
    for i in range(len(arr)):
        if(os.path.splitext(arr[i])[1] == '.jpg'):   
            strImagePath = strhoitFilePath+'/'+arr[i]
            image = cv2.imread(strImagePath)    
            image = cv2.resize(image, dsize=(3872,11616))     
            for h in range(3):                        
                save_pred = image[3872*h :(3872)+3872*h, 0: 3872]      
                #save_pred = cv2.resize(save_pred,dsize=(1500,1500))           
                saveResultPath = strSavePath +os.path.splitext(arr[i])[0]+'_'+str(h) +'.jpg'                                
                cv2.imwrite(saveResultPath,save_pred)                   
def makeEmptyFile():
    strhoitFilePath = 'H:/crack_train/10m/empty/images'
    strsavehoitFilePath = 'H:/crack_train/10m/empty/labels/'
    arr = os.listdir(strhoitFilePath)
    for i in range(len(arr)):
        if(os.path.splitext(arr[i])[1] == '.jpg'):                            
            saveResultPath = strsavehoitFilePath +os.path.splitext(arr[i])[0]+'.txt'      
            f = open(saveResultPath,'w+')
            f.close()
def checkTrainFile():
    strhoitFilePath = 'H:/crack_train/10m/crackImage/labels'    
    strsavehoitFilePath = 'H:/crack_train/10m/crackImage/temp.txt'
    arr = os.listdir(strhoitFilePath)
    
    f = open(strsavehoitFilePath,'w+')
    for i in range(len(arr)):
        if(os.path.splitext(arr[i])[1] == '.txt'): 
            openf = open(strhoitFilePath+'/'+arr[i],'r')                                       
            #print(strhoitFilePath+'/'+arr[i])
            while True:
                line = openf.readline()                
                if not line: break
                f.write(line)
            f.write
    f.close()
def findFiles():
    strFilePath = 'H:/crack_train/10m/split2/'
    arr = os.listdir(strFilePath)
    print(arr[0])
def saveResultTest():
    strFilePath = 'D:/yolov5-master/yolov5-master/data/images/'
    arr = os.listdir(strFilePath)

    ResultFilePath = 'E:/result/result/'
    savePath = 'E:/result/ori/'
    arr_result = os.listdir(ResultFilePath)

    for i in range(len(arr)):
        for j in range(len(arr_result)):                        
            if(arr[i] == arr_result[j]):
                #print(arr_result[j])
                pathstr = strFilePath + arr_result[j]
                
                openstr = strFilePath + arr[i]
                image = cv2.imread(openstr)    
                print(openstr)
                pathSaveStr = savePath+arr_result[j]
                print(pathSaveStr)
                cv2.imwrite(pathSaveStr,image)    

if __name__ == '__main__':    
    #saveResultTest()
    #findFiles()
    #checkTrainFile()
    #yoloFileMaker()
    #ImageDivision()
    #Imageabsorption()
    #contoursTest()
    #lineDetection()
    #binMake()
    testInit()
    #testInitOne()
    #testtest(0)
    #CheckMixModel()
    #testOne()
    #testPng()
    #testImageSet()
    #invertColor()
    