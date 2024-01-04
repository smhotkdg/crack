import cv2 as cv
import numpy as np
import cv2

from scipy.spatial import distance as dist
# Define midpoint coordinate operation
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def measure(img):    
    count =0
    areaList = []
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
        
        print("###########################")     
       
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
            print("###########################")     
            #print(box)
            print("###########################")     
            print("###########################")     
            roi = image[y:y + h,x:x + w]      
            #print(roi.shape)      
            if(roi.shape[0] ==0 or roi.shape[1] ==0 or roi.shape[2] ==0):
                print("empty")
            else:            
               #cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/hoit'+str(count)+'.jpg',roi)

                gray_roi = cv.cvtColor(roi, cv.COLOR_RGB2GRAY)
                ret_roi,thresh_roi=cv2.threshold(gray_roi,50,255,cv2.THRESH_BINARY_INV)
            
                areaCount = cv2.countNonZero(thresh_roi)
                count += 1                
           

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
                # #  4 points, that is, 2 blue lines in the picture

                cv2.line(image_ori, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 0), 1)
                cv2.line(image_ori, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 0), 1)
                # Calculate the coordinates of the center point
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                # # Convert the image length to the actual length, 6.5 is equivalent to the scale, I use the mm unit, that is, 1mm is equivalent to 6.5 images

                dimA = dA #/ 6.5
                dimB = dB #/ 6.5
                # Print the calculation result on the original image, which is the yellow content.
                #cv2.countNonZero(image_ori) 
                
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
                percent = (areaCount/Totalarea) *100
                cv2.putText(image_ori, "Area {:.1f} %".format(percent),
                    (int(tltrX + 10), int(tltrY + 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (0, 0, 0), 1)
                print(percent)
    cv2.imwrite('D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result.bmp',image_ori)
    

#Start the camera and set the resolution
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
strFilePath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/test.jpg'
strOriPath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/test.jpg'
image = cv2.imread(strFilePath)
image_ori =cv2.imread(strOriPath)
#cv2.imshow("input image", image)
measure(image)
