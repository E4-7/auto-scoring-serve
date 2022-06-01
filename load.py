import pandas as pd
import numpy as np
import cv2
import glob
import os
import shutil
#import zbar
import openpyxl
from openpyxl import Workbook
import pandas as pd
import xlsxwriter
import copy
from PIL import Image
import base64
import io
import requests
from io import BytesIO
import tempfile
import requests

#변수들
omr = 15 # 5지선다 문항 수
short = 15 # 단답형 문항 수
essay = 3 # 서술형 문항 수 

# omr 변수
heightImg = 1470
widthImg  = 580
questions= 15
choices= 5
global ans, imgQ, orb, kp1, des1, impKp1, per, template_version,template_link, answer,students
ans = []
# 변수
roi1_1 = [[(70,198), (336,777), 'omr','omr_all'],
          [(328,209),(549,243), 'short', '1'],
          [(328,242),(549,281), 'short', '2'],
          [(328,281),(549,318), 'short', '3'],
          [(328,317),(549,355), 'short', '4'],
          [(328,354),(549,391), 'short', '5'],
          [(328,389),(549,429), 'short', '6'],
          [(328,426),(549,465), 'short', '7'],
          [(328,464),(549,502), 'short', '8'],
          [(328,502),(549,539), 'short', '9'],
          [(328,538),(549,575), 'short', '10'],
          [(328,575),(549,613), 'short', '11'],
          [(328,612),(549,649), 'short', '12'],
          [(328,648),(549,684), 'short', '13'],
          [(328,684),(549,721), 'short', '14'],
          [(328,720),(549,758), 'short', '15'],
          [(560,39), (1082,278), 'essay','1'],
          [(560,278), (1082,519), 'essay','2'],
          [(560,519), (1082,757), 'essay','3']]

roi = {'1.1' : roi1_1}


def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew

def rectContour(contours):
    #import pdb; pdb.set_trace()
    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)

    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def splitBoxes(img):
    rows = np.vsplit(img,15)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes

def drawGrid(img,questions=15,choices=5):  ##
    secW = int(img.shape[1]/choices)
    secH = int(img.shape[0]/questions)
    for i in range (0,19):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)

    return img

def showAnswers(img,myIndex,grading,ans,questions,choices): ##
     secW = int(img.shape[1]/choices)
     secH = int(img.shape[0]/questions)

     for x in range(0,questions):
         myAns= myIndex[x]
         cX = (myAns * secW) + secW // 2
         cY = (x * secH) + secH // 2
         
         
        
         if grading[x]==1:
            myColor = (0,255,0)
            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img,(cX,cY),30,myColor,cv2.FILLED)
         else:
            myColor = (0,0,255)
            #cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            #cv2.circle(img, (cX, cY), 30, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0,0,255) #(0, 255, 0)#
            correctAns = ans[x]
            cv2.circle(img,((correctAns * secW)+secW//2, (x * secH)+secH//2),
            30,myColor,cv2.FILLED)

# omr 채점 --- 이미지, 점수 반환
def omrGrading(img):
  #img = cv2.imread(pathImage)
  img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
  imgFinal = img.copy()
  imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
  imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
  imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 

  ## FIND ALL COUNTOURS
  imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
  imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
  #_, contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
  contours = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
  contours = contours[0] if len(contours) == 2 else contours[1]

  cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS

  rectCon = rectContour(contours) # FILTER FOR RECTANGLE CONTOURS
  biggestPoints= getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE



  if biggestPoints.size != 0 :

    # BIGGEST RECTANGLE WARPING
    biggestPoints=reorder(biggestPoints) # REORDER FOR WARPING
    cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggestPoints) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE


    # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
    #print(imgThresh.shape)
    boxes = splitBoxes(imgThresh) # GET INDIVIDUAL BOXES
    countR=0
    countC=0
    myPixelVal = np.zeros((questions,choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC]= totalPixels
        countC += 1
        if (countC==choices):countC=0;countR +=1
    #print(myPixelVal)
    # FIND THE USER ANSWERS AND PUT THEM IN A LIST
    myIndex=[]
    for x in range (0,questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    #print("USER ANSWERS",myIndex) 

    # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
    grading=[]
    #print("myin",myIndex)
    for x in range(0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:grading.append(0)
    #print("GRADING",grading)

    # DISPLAYING ANSWERS
    showAnswers(imgWarpColored,myIndex,grading,ans,questions,choices) # DRAW DETECTED ANSWERS
    drawGrid(imgWarpColored) # DRAW GRID
    imgRawDrawings = np.zeros_like(imgWarpColored) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
    showAnswers(imgRawDrawings, myIndex, grading, ans, questions,choices) # DRAW ON NEW IMAGE
    invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # INVERSE TRANSFORMATION MATRIX
    imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # INV IMAGE WARP

  
    # SHOW ANSWERS AND GRADE ON FINAL IMAGE
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
  

    # IMAGE ARRAY FOR DISPLAY
    imageArray = ([img,imgGray,imgCanny,imgContours],
                  [imgBigContour,imgThresh,imgWarpColored,imgFinal])


    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Edges","Contours"],
              ["Biggest Contour","Threshold","Warpped","Final"]]

    #stackedImage = stackImages(imageArray,0.5,lables)
    #cv2_imshow(stackedImage)
    return imgFinal, myIndex

def set_omrGrading(img):
  global ans
  # import pdb;
  # pdb.set_trace()
  img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
  imgFinal = img.copy()
  imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
  imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
  imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 

  ## FIND ALL COUNTOURS
  imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
  imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
  #contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
  contours = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
  contours = contours[0] if len(contours) == 2 else contours[1]
  cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
  rectCon = rectContour(contours) # FILTER FOR RECTANGLE CONTOURS

  biggestPoints= getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE



  if biggestPoints.size != 0 :

    # BIGGEST RECTANGLE WARPING
    biggestPoints=reorder(biggestPoints) # REORDER FOR WARPING
    cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggestPoints) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE


    # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
    #print(imgThresh.shape)
    boxes = splitBoxes(imgThresh) # GET INDIVIDUAL BOXES
    countR=0
    countC=0
    myPixelVal = np.zeros((questions,choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC]= totalPixels
        countC += 1
        if (countC==choices):countC=0;countR +=1
    #print(myPixelVal)
    # FIND THE USER ANSWERS AND PUT THEM IN A LIST
    myIndex=[]
    for x in range (0,questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
        ans.append(myIndexVal[0][0])
    #print("USER ANSWERS",ans)
    #ans = myIndex
    # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
    grading=[]
    for x in range(0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:grading.append(0)
    
    # DISPLAYING ANSWERS
    showAnswers(imgWarpColored,myIndex,grading,ans,questions,choices) # DRAW DETECTED ANSWERS
    drawGrid(imgWarpColored) # DRAW GRID
    imgRawDrawings = np.zeros_like(imgWarpColored) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
    showAnswers(imgRawDrawings, myIndex, grading, ans, questions,choices) # DRAW ON NEW IMAGE
    invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # INVERSE TRANSFORMATION MATRIX
    imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # INV IMAGE WARP

  
    # SHOW ANSWERS AND GRADE ON FINAL IMAGE
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
  

    # IMAGE ARRAY FOR DISPLAY
    imageArray = ([img,imgGray,imgCanny,imgContours],
                  [imgBigContour,imgThresh,imgWarpColored,imgFinal])
    #cv2_imshow(imgFinal)

    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Edges","Contours"],
              ["Biggest Contour","Threshold","Warpped","Final"]]

    #stackedImage = stackImages(imageArray,0.5,lables)
    #cv2_imshow(stackedImage)
    return imgFinal, ans
def make_imgMask(img_link, orb, des1, per, imgQ, kp1, w, h) : #평평하게 펴기
  
  image_nparray = np.asarray(bytearray(requests.get(img_link).content), dtype=np.uint8)
  img = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)

  kp2, des2 = orb.detectAndCompute(img,None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.match(des2,des1)

  matches = list(matches)
  #print(len(matches))
  matches.sort(key = lambda x:x.distance)
  good = matches[:int(len(matches)*(per/100))]
  imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
  srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
  dstPoints = np.float32([kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

  M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
  imgScan = cv2.warpPerspective(img,M,(w,h))

  #cv2.imshow(y, imgScan)
  imgShow = imgScan.copy()
  imgMask = np.zeros_like(imgShow)
  return imgScan, imgShow, imgMask

def crop_omr_answers(imgScan, imgShow, imgMask, template_version, i):

  r = roi[template_version][0]

  cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
  imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)  ##바꿀수 있음
  imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
  #print(imgCrop)
  #cv2_imshow(imgCrop)
  if i==0:
    #import pdb;pdb.set_trace()
    imgFinal, stdans = set_omrGrading(imgCrop)
    #print(stdans)
  else : 
    imgFinal, stdans = omrGrading(imgCrop)
  return imgFinal, stdans

def crop_answers(type,col,imgScan, imgShow, imgMask,template_version):
  if type =='short':
    i = col
  else:
    i = 15+col
  r = roi[template_version][i]
  cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
  imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)  ##바꿀수 있음
  imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
  return imgCrop

###############################################################################
# main
def main(students, wbook):
  #students = pd.read_csv("/content/drive/Shareddrives/data/E47/list2.csv")
  template_version = students['name'][0]#양식 이름 1.1 전역변수
  template_link = students['AnswerId'][0] #양식 이미지 링크
  answer_link = students['AnswerId'][1] # 정답 이미지 링크
  
  image_nparray = np.asarray(bytearray(requests.get(template_link).content), dtype=np.uint8)
  imgQ = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
  image_nparray = np.asarray(bytearray(requests.get(answer_link).content), dtype=np.uint8)
  answer = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)

  students=students.iloc[1:]
  students = students.reset_index().drop('index',axis = 1)
  h,w,c = imgQ.shape
  orb = cv2.ORB_create(10000)   
  kp1, des1 = orb.detectAndCompute(imgQ,None)
  impKp1 = cv2.drawKeypoints(imgQ,kp1,None)
  per = 25

  workbook  = xlsxwriter.Workbook(wbook)
  worksheet1 = workbook.add_worksheet('객관식')   
  worksheet1_ = workbook.add_worksheet('객관식 view')      
  worksheet2 = workbook.add_worksheet('단답형')  
  worksheet3 = workbook.add_worksheet('서술형') 

  head = workbook.add_format({'align' : 'center','valign':'vcenter', 'bold':True, 'font_size':11,'text_wrap': True})
  normal = workbook.add_format({'align' : 'center','valign':'vcenter','text_wrap': True})
  red = workbook.add_format({'align' : 'center','valign':'vcenter', 'bold':True, 'font_color':'red','text_wrap': True})
  green = workbook.add_format({'align' : 'center','valign':'vcenter', 'bold':True, 'font_color':'green','text_wrap': True})

  # 객관식
  worksheet1.set_column(0, 2, 13)
  worksheet1.set_column(4, 4, 25)
  worksheet1.set_default_row(25)
  worksheet1.write('A1', '학번', head)
  worksheet1.write('B1', '이름', head)
  worksheet1.write('C1', '신분증 인증', head)
  worksheet1.write('D1', '점수', head)
  worksheet1.write('E1', '채점 이미지', head)
  for i in range(15):
    worksheet1.write(0,i+5, str(i+1)+"번", head)

  omrcrops=[]

  # 객관식 view
  worksheet1_.set_column(0, 0, 25.5)
  worksheet1_.set_default_row(25)


  write_row =0

  # 단답형
  image_width = 125
  image_height = 50

  cell_width =130
  cell_height = 50
  x_scale = cell_width/image_width
  y_scale = cell_height/image_height
  worksheet2.set_column(0, 2, 18)
  worksheet2.set_column(3, 18, 32)
  worksheet2.set_default_row(25)
  worksheet2.write('A1', '학번', head)
  worksheet2.write('B1', '이름', head)
  worksheet2.write('C1', '신분증 인증', head)

  for i in range(15):
    ch = chr(ord('D') + i)
    ch = ch+'1'
    worksheet2.write(ch, str(i+1), head)


  worksheet3.set_column(0, 2, 13)
  worksheet3.set_column(3, 18, 73)
  worksheet3.set_default_row(25)
  worksheet3.write('A1', '학번', head)
  worksheet3.write('B1', '이름', head)
  worksheet3.write('C1', '신분증 인증', head)
  for i in range(3):
    ch = chr(ord('D') + i)
    ch = ch+'1'
    worksheet3.write(ch, str(i+1), head)

  for i in range(len(students)):
    row = i+1
    worksheet1.write(row,0,students['studentID'][i],normal)
    worksheet1.write(row,1,students['name'][i],normal)

    imgScan, imgShow, imgMask = make_imgMask(students['AnswerId'][i], orb, des1, per, imgQ, kp1, w, h)


    imgCrop,stdans= crop_omr_answers(imgScan, imgShow, imgMask, template_version, i)

    omrcrops.append(imgCrop)
    #print(imgCrop)
    #cv2.imwrite("s.jpg", imgCrop)
    if(students['name'][i] != '정답'):  
      if students['is_certified'][i] == 1 :
        worksheet1.write(row,2,'O',green)
      else :
        worksheet1.write(row,2,'X',red)
      #imgCrop, stdans= omrGrading(imgCrop)
      omrcrops.append(imgCrop)
      for col in range(15):
        if ans[col] == stdans[col]:
          worksheet1.write(row, col+5, stdans[col]+1,green)
        else:
          worksheet1.write(row, col+5, stdans[col]+1,red)
    else: 
      for col in range(15):
        worksheet1.write(row, col+5, ans[col]+1, head)

    # 단답
    worksheet2.write(row,0,students['studentID'][i],normal)
    worksheet2.write(row,1,students['name'][i],normal)
    if students['is_certified'][i] == 1 :
      worksheet2.write(row,2,'O',green)
    else :
      worksheet2.write(row,2,'X',red)
    for col in range(1,16):
      
      imgCrop= crop_answers('short',col,imgScan, imgShow, imgMask, template_version)
      fd,path = tempfile.mkstemp(prefix='short'+str(i), suffix = '.jpg')
      cv2.imwrite(path,imgCrop)
      worksheet2.insert_image(row, col+2, path,{'x_scale': x_scale, 'y_scale': y_scale,'object_position': 3})

    # 서술형
    worksheet3.write(row,0,students['studentID'][i],normal)
    worksheet3.write(row,1,students['name'][i],normal)
    if students['is_certified'][i] == 1 :
      worksheet3.write(row,2,'O',green)
    else :
      worksheet3.write(row,2,'X',red)
    for col in range(1,4): 
      imgCrop= crop_answers('essay',col,imgScan, imgShow, imgMask, template_version)
      fd,path = tempfile.mkstemp(prefix='essay'+str(i), suffix = '.jpg')
      #print('temp path:', path)
      cv2.imwrite(path,imgCrop)
      worksheet3.set_row(row, 168)
      worksheet3.insert_image(row, col+2, path,{'x_scale': 0.9, 'y_scale': 0.9,'object_position': 3})
  for i in range(0,len(students),1):
    row_sheet1 = i+1
    row = i+2
    tmp1 = "객관식" + "!A" + str(row)
    tmp2 = "객관식" + "!B" + str(row)
    tmp3 = "객관식" + "!D" + str(row)
    worksheet1_.write(write_row,0,f'=CONCATENATE({tmp1}," / ",{tmp2}," / ", {tmp3},"점")',head)
    write_row+=1
    worksheet1_.set_row(write_row,346)

    fd,path = tempfile.mkstemp(prefix='short'+str(i), suffix = '.jpg')
    

    cv2.imwrite(path,omrcrops[i])

    worksheet1_.insert_image(write_row, 0, path,{'x_scale': 0.3, 'y_scale':0.3,'object_position': 3})
    worksheet1.write_url(row_sheet1, 4, f"internal:'{'객관식 view'}'!A"+str(write_row),string='이미지 바로가기')
    write_row +=1
  workbook.close()
  os.unlink(path)


  

