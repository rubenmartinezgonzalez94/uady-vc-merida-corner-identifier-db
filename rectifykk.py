import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from corners_descriptors_db import dbManager

def rectify(coords, img):
  pts1 = np.float32(coords)
  pts2 = np.float32([[0, 0], [500, 0],
                           [0, 500], [500, 500]])
  matrix = cv.getPerspectiveTransform(pts1, pts2)
  result = cv.warpPerspective(img, matrix, (500, 500))
  return result

def matchSearch(db_manager, path, cutMethod, searchMethod, threshold, umbral = None):
    if cutMethod == 1:
        img = cv.imread(path, cv.COLOR_RGB2Lab)
        res = cv.medianBlur(img[:,:,0],7)
        res = cv.Canny(res,100,200)
        res = cv.boxFilter(res,-1,(3,3), normalize = True)
        (thresh, res) = cv.threshold(res, 70, 255, cv.THRESH_BINARY)
        contours,_ = cv.findContours(res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = tuple(sorted(contours, key=len, reverse=True))
        i = 0
        cosas1 = []
        cosas2 = []
        while i<len(contours) and i <= 10:
            x, y, w, h = cv.boundingRect(contours[i])
      
            rect = rectify([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], img)
#            plt.figure()
#            plt.imshow(rect)
#            plt.show()
            if searchMethod == 'sift':
                best_match, num = db_manager.recognize_image_sift(rect, 1500)
                cosas1.append(best_match)
                cosas2.append(num)
                #print(best_match.address)
            else:
                best_match, num = db_manager.recognize_image_orb(rect, 1500, umbral)
                cosas1.append(best_match)
                cosas2.append(num)
            #correct = threshold < num
            #print(num)
            #if correct:
            #    return best_match
          
            i = i+1
            
        idx = cosas2.index(max(cosas2))    
        return cosas1[idx]
        
    elif cutMethod == 2:
        image = cv.imread(path)
        lab =  cv.cvtColor(image, cv.COLOR_RGB2Lab)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
        imgs = [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2], image[:, :, 0], image[:, :, 1], image[:, :, 2], gray]
        for img in imgs:
            blur = cv.medianBlur(img, 189)
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv.filter2D(blur, -1, sharpen_kernel)

            re, thresh = cv.threshold(sharpen,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            thresh = np.array(thresh, np.uint8)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,3))
            close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

            # Find contours and filter using threshold area
            cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            min_area = 8000
            max_area = 20000000
            image_number = 0
            cosas1 = []
            cosas2 = []
            for c in cnts:
                area = cv.contourArea(c)
                if area > min_area and area < max_area:
                    x,y,w,h = cv.boundingRect(c)
                    
                    ROI = image[y:y+h, x:x+w]
                    #x1, y1, z1 = np.shape(ROI)
                    rect = rectify([[0, 0], [0+w, 0], [0, 0+h], [0+w, 0+h]], ROI)
#                    plt.figure()
#                    plt.imshow(ROI)
#                    plt.show()
                    if searchMethod == 'sift':
                        best_match, number = db_manager.recognize_image(rect,searchMethod, 1500)
                        cosas1.append(best_match)
                        cosas2.append(number)
                        #print(best_match.address)
                    else:
                        best_match, number = db_manager.recognize_image(rect,searchMethod, 1500, umbral)
                        cosas1.append(best_match)
                        cosas2.append(number)  
                    #print(number)                      
                    #if number > threshold:
                    #    return best_match                                     
        idx = cosas2.index(max(cosas2))    
        return cosas1[idx]
                    
