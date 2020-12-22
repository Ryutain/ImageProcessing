import datetime
from tkinter import filedialog
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import cv2
import numpy
import pymysql
import random
import os
import numpy as np
import tempfile
import xlrd
import xlsxwriter
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import winsound
## 함수 선언부
def malloc(h, w, value=0) :
    retMemory = [ [ value for _ in range(w)]  for _ in range(h) ]
    return retMemory

def mallocNumpy(t, h, w):
    retMemory = np.zeros((t, h, w), dtype=np.int16)
    return retMemory

def cvOut2outImage() :  # OvenCV의 결과 --> OutImage 메모리에 넣기
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    ## 결과 메모리의 크기
    outH = cvOutImage.shape[0]
    outW = cvOutImage.shape[1]
    ## 입력이미지용 메모리 할당
    outImage = mallocNumpy(RGB, outH, outW)
    ## cvOut --> 메모리
    for i in range(outH):
        for k in range(outW):
            if (cvOutImage.ndim == 2) : # 그레이, 흑백
                outImage[R][i][k] = cvOutImage.item(i, k)
                outImage[G][i][k] = cvOutImage.item(i, k)
                outImage[B][i][k] = cvOutImage.item(i, k)
            else :
                outImage[R][i][k] = cvOutImage.item(i, k ,B)
                outImage[G][i][k] = cvOutImage.item(i, k, G)
                outImage[B][i][k] = cvOutImage.item(i, k, R)

def openFile() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, OriH, OriW

    ## 파일 선택하기
    filename = askopenfilename(parent=window,
           filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))

    if filename == '' or filename == None:
        return
    ## (중요!) 입력이미지의 높이와 폭 알아내기
    cvInImage = cv2.imread(filename)
    cvInImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2RGB)
    OriH = cvInImage.shape[0]
    OriW = cvInImage.shape[1]
    inH = OriH
    inW = OriW

    OriImage = np.transpose(cvInImage, (2, 0, 1))
    OriImage = np.array(OriImage)
    OriImage = OriImage.astype(np.int32)
    inImage = OriImage.copy()
    #
    equalColor()

def createOri() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, OriH, OriW

    OriH = cvInImage.shape[0]
    OriW = cvInImage.shape[1]
    inH = OriH
    inW = OriW

    cvInImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2RGB)

    OriImage = np.transpose(cvInImage, (2, 0, 1))
    OriImage = np.array(OriImage)
    OriImage = OriImage.astype(np.int32)
    inImage = OriImage.copy()

def saveFile() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, OriH, OriW
    if filename == '' or filename == None:
        return
    saveCvPhoto = np.zeros((outH, outW, 3), np.uint8)
    for i in range(outH):
        for k in range(outW):
            tup = tuple((outImage[B][i][k], outImage[G][i][k], outImage[R][i][k]))
            saveCvPhoto[i,k] = tup

    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='.',
                           filetypes=(("그림 파일", "*.png;*.jpg;*.bmp;*.tif"), ("모든 파일", "*.*")))

    if saveFp == '' or saveFp == None:
        return

    cv2.imwrite(saveFp.name, saveCvPhoto)

def displayImageColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, iCopyImage
    global cvInImage, cvOutImage, OriImage, OriH, OriW

    VX, VY = 512,512 #최대 화면 크기
    ##크기가 512 보다 크면, 최대 512로 보이기...
    if outH <= VY or outW <= VX:
        VY = outH;VX = outW; step = 1
    else:
        if outH > outW:
            step = outH / VY
            VX = int(VY * outW / outH)
        else:
            step = outW / VX
            VY = int(VX * outH / outW)

    window.geometry(str(int(VX * 1.2)) + 'x' + str(int(VY * 1.2)))
    # 기존 이미지 삭제
    if canvas != None:
        canvas.destroy()
    # 이미지 출력 창 프레임 설정
    canvas = Canvas(window, height=VY, width=VX)
    paper = PhotoImage(height=VY, width=VX)
    canvas.create_image((VX // 2, VY // 2), image=paper, state='normal')
    # 메모리에서 처리한 후, 한방에 화면에 보이기 --> 완전 빠름

    rgbString =""
    for i in numpy.arange(0, outH, step) :
        tmpString = "" # 각 줄
        for k in numpy.arange(0, outW, step) :
            i = int(i); k=int(k);
            r = outImage[R][i][k]
            g = outImage[G][i][k]
            b = outImage[B][i][k]
            tmpString += "#%02x%02x%02x " % (r, g, b)
        rgbString += '{' + tmpString + '} '
    paper.put(rgbString)
    canvas.pack(expand=1, anchor=CENTER)
    status.configure(text='이미지정보:' + str(outH) + 'x' + str(outW)+'      '+filename)

def overwrite() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    inH = outH
    inW = outW

    inImage = mallocNumpy(RGB, inH, inW)
    inImage = outImage.copy()


    cvInImage = np.zeros((outH, outW, 3), np.uint8)
    for i in range(outH):
        for k in range(outW):
            tup = tuple((outImage[B][i][k], outImage[G][i][k], outImage[R][i][k]))
            cvInImage[i, k] = tup

def makecopy() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, icH, icW, ocH, ocW
    global filename, iCopyImage, oCopyImage
    if filename == '' or filename == None:
        return
    icH = inH;    icW = inW;    ocH = outH;    ocW = outW;

    iCopyImage = mallocNumpy(RGB, icH, icW)
    oCopyImage = mallocNumpy(RGB, ocH, ocW)

    iCopyImage = inImage.copy()
    oCopyImage = outImage.copy()

def undo() :
    global window, canvas, paper, inImage, outImage, inH, inW, icH, icW, outH, outW, filename, iCopyImage
    if filename == '' or filename == None:
        return

    outH = icH
    outW = icW

    outImage = mallocNumpy(RGB, outH, outW)

    outImage = iCopyImage.copy()

    displayImageColor()
    overwrite()

def redo() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, iCopyImage
    global ocW, oCopyImage, ocH
    if filename == '' or filename == None:
        return

    outH = ocH
    outW = ocW
    outImage = mallocNumpy(RGB, outH, outW)
    ### 진짜 영상처리 알고리즘 ###
    outImage = oCopyImage.copy()

    displayImageColor()
    overwrite()

###### 화소점 처리 함수 ##########

def equalColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, OriH, OriW
    if filename == '' or filename == None:
        return

    outH = OriH;    outW = OriW
    outImage = OriImage.copy()

    displayImageColor()
    overwrite()

def addColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    value = askinteger("밝게하기", "값")
    if value == None:
        return
    outImage = value + inImage
    outImage[outImage > 255] = 255

    displayImageColor()
    makecopy()
    overwrite()

def minusColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    value = askinteger("어둡게하기", "값")
    if value == None:
        return
    outImage = inImage - value
    outImage[outImage < 0] = 0

    displayImageColor()
    makecopy()
    overwrite()

def grayColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outImage[0] = (inImage[0] + inImage[1] + inImage[2]) // 3
    outImage = np.array([outImage[0], outImage[0], outImage[0]])

    displayImageColor()
    makecopy()
    overwrite()

def binaryColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outImage[0] = (inImage[0] + inImage[1] + inImage[2]) // 3
    outImage = np.array([outImage[0], outImage[0], outImage[0]])

    outImage[outImage > 127] = 255
    outImage[outImage <= 127] = 0

    displayImageColor()
    makecopy()
    overwrite()

def avgbinColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outImage[0] = (inImage[0] + inImage[1] + inImage[2]) // 3
    outImage = np.array([outImage[0], outImage[0], outImage[0]])

    outImage[outImage > inImage.mean()] = 255
    outImage[outImage <= inImage.mean()] = 0

    displayImageColor()

def cenbinColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outImage[0] = (inImage[0] + inImage[1] + inImage[2]) // 3
    outImage = np.array([outImage[0], outImage[0], outImage[0]])

    outImage[outImage > np.median(inImage)] = 255
    outImage[outImage <= np.median(inImage)] = 0

    displayImageColor()
    makecopy()
    overwrite()

def concenColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    p1 = askinteger("", "영역 시작 값 : ")
    p2 = askinteger("", "영역 끝 값 : ")
    p3 = askinteger("", "영역에 더해줄 값 : ")

    outImage = np.where((p1 <= inImage) & (inImage <= p2), inImage + p3, inImage)
    outImage = outImage.astype(np.int16)
    outImage = np.where(outImage > 255, 255, outImage)
    outImage = np.where(outImage < 0, 0, outImage)

    displayImageColor()
    makecopy()
    overwrite()

def exColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outImage = 255 - inImage

    displayImageColor()
    makecopy()
    overwrite()

def posterColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    num = askinteger("포스터라이징", "값-->", minvalue=2, maxvalue=128)
    num1 = 256 // num
    outImage = inImage.copy()
    for i in range(num) :
        outImage = np.where((num1 * i <= inImage) & (inImage < num1 * (i+1)), num1 * i, outImage)
    outImage = outImage.astype(np.int16)
    displayImageColor()
    makecopy()
    overwrite()

def gammaColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    value = askfloat("감마 값", "값-->", minvalue=0.01, maxvalue=100)
    if value == None:
        return
    outImage = inImage ** (1.0/value)
    outImage = outImage.astype(np.int16)
    outImage = np.where(outImage > 255, 255, outImage)
    outImage = np.where(outImage < 0, 0, outImage)

    displayImageColor()
    makecopy()
    overwrite()

def paraCapColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outImage = 255 * (1 - (inImage / 128.0 - 1) ** 2)
    outImage = outImage.astype(np.int16)
    outImage = np.where(outImage > 255, 255, outImage)
    outImage = np.where(outImage < 0, 0, outImage)

    displayImageColor()
    makecopy()
    overwrite()

def paraCupColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outImage = 255 * ((inImage / 128.0 - 1) ** 2)
    outImage = outImage.astype(np.int16)
    outImage = np.where(outImage > 255, 255, outImage)
    outImage = np.where(outImage < 0, 0, outImage)
    displayImageColor()
    makecopy()
    overwrite()

### 기하학 함수 ###

def mirrorLRColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    cvOutImage = cv2.flip(cvInImage, 2)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def mirrorUDColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    cvOutImage = cv2.flip(cvInImage, 0)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def moveColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outH = inH; outW=inW
    outImage = mallocNumpy(RGB,outH,outW)
    valuex = askinteger("x축 이동", "값-->", minvalue=0, maxvalue=300)
    valuey = askinteger("y축 이동", "값-->", minvalue=0, maxvalue=300)
    if valuex == None or valuey == None:
        return
    for rgb in range(RGB):
        for i in range(valuey, inH):
            for k in range(valuex, inW):
                outImage[rgb][i][k] = inImage[rgb][i - valuey][k - valuex]
            for k in range(0, valuex) :
                outImage[rgb][i][k] = 255
        for i in np.arange(valuey):
            for k in range(inW) :
                outImage[rgb][i][k] = 255
    displayImageColor()
    makecopy()
    overwrite()

def rotateColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    angle = askinteger("회전", "각도", minvalue=0, maxvalue=360)
    if angle == None:
        return
    radian = angle * math.pi / 180
    outH = int(abs(inH* math.cos(radian)) + abs(inW * math.sin(radian)))
    outW = int(abs(inW*math.cos(radian)) + abs(inH * math.sin(radian)))
    cx = inH / 2
    cy = inW / 2

    outImage = mallocNumpy(RGB, outH, outW)

    for rgb in range(RGB) :
        for i in range(outH):
            for k in range(outW):
                xs = i
                ys = k
                xd = int(math.cos(radian) * (xs - outH/2) - math.sin(radian) * (ys - outW/2) + cx)
                yd = int(math.sin(radian) * (xs - outH/2) + math.cos(radian) * (ys - outW/2) + cy)
                if 0 <= xd < inH and 0 <= yd < inW:
                    outImage[rgb][xs][ys] = inImage[rgb][xd][yd]
                else :
                    outImage[rgb][xs][ys] = 255

    displayImageColor()
    makecopy()
    overwrite()

def sizeUpColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    scale = askinteger("확대", "배율 : ")
    if scale == None:
        return
    cvOutImage = cv2.resize(cvInImage, None,  None, scale, scale, cv2.INTER_CUBIC)
    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def sizeDownColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    scale = askinteger("축소", "배율 : ")
    if scale == None:
        return
    cvOutImage = cv2.resize(cvInImage, None, None, 1/scale, 1/scale, cv2.INTER_CUBIC)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

### 화소 영역 처리 ###

def blurColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    value = askinteger("블러링 값", "값 입력")
    if value == None:
        return
    mask = np.full((value,value), 1/(value*value))
    cvOutImage = cv2.filter2D(cvInImage, -1, mask)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def shaftColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    mask = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])
    cvOutImage = cv2.filter2D(cvInImage,-1,mask)
    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def emboColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    mask = np.zeros((3, 3), np.float32)
    mask[0][0] = -1.0
    mask[2][2] = 1.0
    cvInImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    cvOutImage = cv2.filter2D(cvInImage,-1,mask)
    cvOutImage += 127

    cvOut2outImage()
    displayImageColor()
    makecopy()

def cannyColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    cvOutImage = cv2.Canny(cvInImage, 100, 200)

    cvOut2outImage()
    displayImageColor()
    makecopy()

def gaussColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    mask = np.array([
        [1/16.0, 1/8.0, 1/16.0],
        [1/8.0, 1/4.0, 1/8.0],
        [1/16.0, 1/8.0, 1/16.0]])

    cvOutImage = cv2.filter2D(cvInImage, -1, mask)
    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def gozupaColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    mask = np.array([
        [-1 / 9.0, -1 / 9.0, -1 / 9.0],
        [-1 / 9.0, 8 / 9.0, -1 / 9.0],
        [-1 / 9.0, -1 / 9.0, -1 / 9.0]])

    cvOutImage = cv2.filter2D(cvInImage, -1, mask)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def yusaColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outH = inH
    outW = inW
    tmpInImage = mallocNumpy(RGB, outH + 2, outW + 2)
    tmpOutImage = mallocNumpy(RGB, outH + 2, outW + 2)

    for rgb in range(RGB) :
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])

    for rgb in range(RGB) :
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = max(abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i - 1][k - 1]),
                        abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i - 1][k]),
                        abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i - 1][k + 1]),
                        abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i][k - 1]),
                        abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i][k + 1]),
                        abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i + 1][k - 1]),
                        abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i + 1][k]),
                        abs(tmpInImage[rgb][i][k] - tmpInImage[rgb][i + 1][k + 1]))
                tmpOutImage[rgb][i - 1][k - 1] = S

    # 마스크에 따라 127 더할지 결정
    for rgb in range(RGB) :
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])

    ########################

    displayImageColor()
    makecopy()
    overwrite()

def chaColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    outH = inH
    outW = inW
    tmpInImage = mallocNumpy(RGB, outH + 2, outW + 2)
    tmpOutImage = mallocNumpy(RGB, outH + 2, outW + 2)
    outImage = mallocNumpy(RGB, outH, outW)
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])

    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = max(abs(tmpInImage[rgb][1+i][1+k] - tmpInImage[rgb][i - 1][k - 1]),
                        abs(tmpInImage[rgb][1+i][k] - tmpInImage[rgb][i - 1][k]),
                        abs(tmpInImage[rgb][1+i][k-1] - tmpInImage[rgb][i - 1][k + 1]),
                        abs(tmpInImage[rgb][i][k+1] - tmpInImage[rgb][i][k - 1]),
                       )
                tmpOutImage[rgb][i - 1][k - 1] = S

    # 마스크에 따라 127 더할지 결정
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])

    displayImageColor()
    makecopy()
    overwrite()

def sobelColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    img_sobelx = cv2.Sobel(cvInImage, cv2.CV_8U, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(cvInImage, cv2.CV_8U, 0, 1, ksize=3)
    cvOutImage = img_sobelx + img_sobely
    cvOut2outImage()
    ########################

    displayImageColor()
    makecopy()

def robertsColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    mask1 = np.array([
            [-1, 0, 0],
             [0, 1, 0],
             [0, 0, 0]])
    mask2 = np.array([
            [0, 0, -1],
             [0, 1, 0],
             [0, 0, 0]])

    img_robertsx = cv2.filter2D(cvInImage, -1, mask1)
    img_robertsy = cv2.filter2D(cvInImage, -1, mask2)
    cvOutImage = img_robertsx + img_robertsy

    cvOut2outImage()
    displayImageColor()
    makecopy()

def prewittColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    mask1 = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]])
    mask2 = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]])

    img_prewittx = cv2.filter2D(cvInImage, -1, mask1)
    img_prewitty = cv2.filter2D(cvInImage, -1, mask2)
    cvOutImage = img_prewittx + img_prewitty

    cvOut2outImage()
    displayImageColor()
    makecopy()

def laplaceColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    cvOutImage = cv2.Laplacian(cvInImage, cv2.CV_8U, ksize=3)

    cvOut2outImage()
    displayImageColor()
    makecopy()

def LoGColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    mask = np.array([[0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]])

    cvOutImage = cv2.filter2D(cvInImage,-1,mask)
    cvOut2outImage()

    displayImageColor()
    makecopy()

def DoGColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    mask = np.array([[0, 0, -1, -1, -1, 0, 0],
            [0, -2, -3, -3, -3, -2, 0],
            [-1, -3, 5, 5, 5, -3, -1],
            [-1, -3, 5, 16, 5, -3, -1],
            [-1, -3, 5, 5, 5, -3, -1],
            [0, -2, -3, -3, -3, -2, 0],
            [0, 0, -1, -1, -1, 0, 0]])

    cvOutImage = cv2.filter2D(cvInImage, -1, mask)

    cvOut2outImage()
    displayImageColor()
    makecopy()

def DoG2Color() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    mask = np.array([[0, 0, 0, -1, -1, -1, 0, 0, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [-1, -3, -1, 9, 19, 9, -1, -3, -1],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, 0]])

    cvOutImage = cv2.filter2D(cvInImage, -1, mask)

    cvOut2outImage()
    displayImageColor()
    makecopy()

### 히스토그램 처리 ###

def stretchColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    ##out = (in - low) / (high - low) * 255
    for rgb in range(RGB) :
        low = high = inImage[rgb][0][0]
        for i in range(inH):
            for k in range(inW):
                if low > inImage[rgb][i][k]:
                    low = inImage[rgb][i][k]
                elif high < inImage[rgb][i][k]:
                    high = inImage[rgb][i][k]

    for rgb in range(RGB) :
        for i in range(inH):
            for k in range(inW):
                value = (inImage[rgb][i][k] - low) * 255 / (high - low)
                if value > 255:
                    outImage[rgb][i][k] = 255
                elif value < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(value)

    displayImageColor()
    makecopy()
    overwrite()

def endInColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    ##out = (in - low) / (high - low) * 255

    low = high = inImage[0][0][0]
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                if low > inImage[rgb][i][k]:
                    low = inImage[rgb][i][k]
                elif high < inImage[rgb][i][k]:
                    high = inImage[rgb][i][k]
    value1 = askinteger("low값 보정", "값 입력")
    value2 = askinteger("high값 보정", "값 입력")

    low += value1
    high -= value2


    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                value = (inImage[rgb][i][k] - low) * 255 / (high - low)
                if inImage[rgb][i][k] > high:
                    outImage[rgb][i][k] = 255
                elif inImage[rgb][i][k] < low:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(value)

    displayImageColor()
    makecopy()
    overwrite()

def equalizedColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    ### 1단계 : 히스토그램 만들기
    histo = [[0 for _ in range(256)] for _ in range(3)]
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                histo[rgb][inImage[rgb][i][k]] += 1


    ### 2단계 : 누적 히스토그램 만들기
    sumhisto = [[0 for _ in range(256)] for _ in range(3)]
    sumhisto[0][0] = histo[0][0]
    sumhisto[1][0] = histo[1][0]
    sumhisto[2][0] = histo[2][0]
    for rgb in range(RGB) :
        for i in range(1, 256):
            sumhisto[rgb][i] = histo[rgb][i] + sumhisto[rgb][i - 1]

    ### 3단계 : 정규화 히스토그램
    ## n = sumhisto[255] * ( 1/ 크기) * 컬러 최대값(255)
    for rgb in range(RGB) :
        for i in range(outH):
            for k in range(outW):
                value = 255 * sumhisto[rgb][inImage[rgb][i][k]] / (inH * inW)
                if value > 255:
                    outImage[rgb][i][k] = 255
                elif value < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(value)

    displayImageColor()
    makecopy()
    overwrite()

### SQL 처리 ###

def upMySQL():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == None or filename == '':
        return
    saveCvPhoto = np.zeros((outH, outW, 3), np.uint8)
    for i in range(outH):
        for k in range(outW):
            tup = tuple(([outImage[B][i][k], outImage[G][i][k], outImage[R][i][k]]))
            saveCvPhoto[i, k] = tup

    saveFname = tempfile.gettempdir() + '/' + os.path.basename(filename)
    cv2.imwrite(saveFname, saveCvPhoto)

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비
    p_id = random.randint(-2100000000, 2100000000)
    tmpName = os.path.basename(saveFname)
    p_fname, p_ext = tmpName.split('.')
    p_size = os.path.getsize(saveFname)
    tmpImage = cv2.imread(saveFname)
    p_width = tmpImage.shape[0]
    p_height = tmpImage.shape[1]
    now = datetime.datetime.now()
    p_upDate = now.strftime('%Y-%m-%d %H:%M:%S')
    p_upUser = loginID  # 로그인한 사용자

    fp = open(saveFname, 'rb')
    blobData = fp.read()
    fp.close()

    sql = "INSERT INTO photo_table(p_id, p_fname, p_ext, p_size, p_height, p_width, p_upDate, p_upUser, p_photo)"
    sql += "VALUES (" + str(p_id) + ",'" + p_fname + "','" + p_ext + "'," + str(p_size) + "," + str(p_height) + ","
    sql += str(p_width) + ",'" + p_upDate + "','" + p_upUser + "', %s)"
    tupleData = (blobData,)
    cur.execute(sql, tupleData)

    conn.commit()
    cur.close()
    conn.close()
    messagebox.showinfo('성공', filename + '  잘 입력됨.')

def downMySQL() : #따로 열기 개념
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, fileList, filename, OriH, OriW
    ##########################
    #준비
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비

    sql = "SELECT p_id, p_fname, p_ext, p_size FROM photo_table"
    cur.execute(sql)

    fileList = cur.fetchall()

    cur.close()
    conn.close()


    def downLoad():
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
        global cvInImage, cvOutImage, fileList, filename, OriH, OriW
        selectIndex = listData.curselection()[0]

        conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
        cur = conn.cursor()  # 빈 트럭 준비

        sql = "SELECT p_fname, p_ext, p_photo FROM photo_table WHERE p_id="
        sql += str(fileList[selectIndex][0])
        cur.execute(sql)
        p_fname, p_ext, p_photo = cur.fetchone()

        fullPath = tempfile.gettempdir() + '/' + p_fname + '.' + p_ext
        fp = open(fullPath, 'wb')
        fp.write(p_photo)
        fp.close()

        cur.close()
        conn.close()
        filename = fullPath
        subWindow.destroy()
        ## (중요!) 입력이미지의 높이와 폭 알아내기
        cvInImage = cv2.imread(filename)
        OriH = cvInImage.shape[0]
        OriW = cvInImage.shape[1]
        inH = OriH
        inW = OriW

        ## 입력이미지용 메모리 할당
        inImage = []
        for _ in range(RGB):
            inImage.append(malloc(OriH, OriW))
        OriImage = []
        for _ in range(RGB):
            OriImage.append(malloc(OriH, OriW))

        ## 파일 --> 메모리 로딩
        for i in range(OriH):
            for k in range(OriW):
                OriImage[R][i][k] = cvInImage.item(i, k, B)
                OriImage[G][i][k] = cvInImage.item(i, k, G)
                OriImage[B][i][k] = cvInImage.item(i, k, R)

        for rgb in range(RGB):
            for i in range(OriH):
                for k in range(OriW):
                    inImage[rgb][i][k] = OriImage[rgb][i][k]

        equalColor()

    # 서브 윈도창 나오기.
    subWindow = Toplevel(window)
    subWindow.geometry('200x350')
    listData = Listbox(subWindow); listData.pack(fill=BOTH, expand=1)
    for fileTup in fileList :
        listData.insert(END, fileTup[1:])

    btnDownLoad = Button(subWindow, text='다운로드',command=downLoad);btnDownLoad.pack(side=LEFT, padx=10, pady=10)

def folderupMySQL():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    root = Tk()
    root.dirName = filedialog.askdirectory()
    file_list = os.listdir(root.dirName)


    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비
    for file in file_list:
        p_id = random.randint(-2100000000, 2100000000)
        saveFname = root.dirName + '/' + os.path.basename(file)
        p_fname, p_ext = file.split('.')
        if p_ext == "jpg" or p_ext == "png" or p_ext == "bmp" or p_ext == "tif" :
            p_size = os.path.getsize(saveFname)
            tmpImage = cv2.imread(saveFname)
            p_width = tmpImage.shape[0]
            p_height = tmpImage.shape[1]
            now = datetime.datetime.now()
            p_upDate = now.strftime('%Y-%m-%d %H:%M:%S')
            p_upUser = loginID

            fp = open(saveFname, 'rb')
            blobData = fp.read()
            fp.close()

            sql = "INSERT INTO photo_table(p_id, p_fname, p_ext, p_size, p_height, p_width, p_upDate, p_upUser, p_photo)"
            sql += "VALUES (" + str(p_id) + ",'" + p_fname + "','" + p_ext + "'," + str(p_size) + "," + str(p_height) + ","
            sql += str(p_width) + ",'" + p_upDate + "','" + p_upUser + "', %s)"
            tupleData = (blobData,)
            cur.execute(sql, tupleData)

    root.destroy()
    conn.commit()
    cur.close()
    conn.close()
    messagebox.showinfo('성공', root.dirName +  '  잘 입력됨.')

### Excel 연산 ###

def saveExcel():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, OriH, OriW
    if filename == '' or filename == None:
        return
    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='xlsx',
                           filetypes=(("엑셀 파일", "*.xls"), ("모든 파일", "*.*")))

    if saveFp == '' or saveFp == None:
        return
    xlsxName = saveFp.name
    # wb = xlwt.Workbook()
    wb = xlsxwriter.Workbook(xlsxName)

    ws_R = wb.add_worksheet("RED")
    ws_G = wb.add_worksheet("GREEN")
    ws_B = wb.add_worksheet("BLUE")

    for i in range(outH):
        for k in range(outW):
            ws_R.write(i,k,outImage[R][i][k])
            ws_G.write(i,k,outImage[G][i][k])
            ws_B.write(i,k,outImage[B][i][k])

    wb.close()
    print('Excel. save ok')

def openExcel():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, OriH, OriW

    filename = askopenfilename(parent=window,
                               filetypes=(('Excel 파일', '*.xls'), ('All File', '*.*')))

    if filename == '' or filename == None:
        return

    workbook = xlrd.open_workbook(filename)
    wsList = workbook.sheets() # 3장 워크 시트
    OriH = wsList[0].nrows
    OriW = wsList[0].ncols
    inH = OriH
    inW = OriW



    ## 입력이미지용 메모리 할당
    inImage = []
    for _ in range(RGB):
        inImage.append(malloc(OriH, OriW))
    OriImage = []
    for _ in range(RGB):
        OriImage.append(malloc(OriH, OriW))

    ## 파일 --> 메모리 로딩
    for i in range(OriH):
        for k in range(OriW):
            OriImage[R][i][k] = int(wsList[R].cell_value(i,k))
            OriImage[G][i][k] = int(wsList[G].cell_value(i,k))
            OriImage[B][i][k] = int(wsList[B].cell_value(i,k))


    for rgb in range(RGB):
        for i in range(OriH):
            for k in range(OriW):
                inImage[rgb][i][k] = OriImage[rgb][i][k]

    equalColor()

def drawExcel() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, OriImage
    global cvInImage, cvOutImage, OriH, OriW
    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='xlsx',
                           filetypes=(("엑셀 파일", "*.xls"), ("모든 파일", "*.*")))
    if saveFp == '' or saveFp == None:
        return
    xlsxName = saveFp.name
    wb = xlsxwriter.Workbook(xlsxName)
    ws_R = wb.add_worksheet("RED")
    ws_G = wb.add_worksheet("GREEN")
    ws_B = wb.add_worksheet("BLUE")
    ws_A = wb.add_worksheet("ALL")

    # 셀 크기를 조절
    ws_R.set_column(0, outW - 1, 1.0)  # 엑셀에서 0.34
    for i in range(outH):
        ws_R.set_row(i, 9.5)  # 엑셀에서 약 0.35
    ws_G.set_column(0, outW - 1, 1.0)  # 엑셀에서 0.34
    for i in range(outH):
        ws_G.set_row(i, 9.5)  # 엑셀에서 약 0.35
    ws_B.set_column(0, outW - 1, 1.0)  # 엑셀에서 0.34
    for i in range(outH):
        ws_B.set_row(i, 9.5)  # 엑셀에서 약 0.35
    ws_A.set_column(0, outW - 1, 1.0)  # 엑셀에서 0.34
    for i in range(outH):
        ws_A.set_row(i, 9.5)  # 엑셀에서 약 0.35

    # 메모리 --> 엑셀 파일
    for i in range(outH):
        for k in range(outW):
            ## Red 시트
            data = outImage[R][i][k]
            if data <= 15:
                hexStr = '#' + ('0' + hex(data)[2:]) + '0000'
            else:
                hexStr = '#' + hex(data)[2:] + '0000'
            # 셀 속성 변경
            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_R.write(i, k, '', cell_format)

            ## Green 시트
            data = outImage[G][i][k]
            if data <= 15:
                hexStr = '#' + ('000' + hex(data)[2:]) + '00'
            else:
                hexStr = '#' + '00' + hex(data)[2:] + '00'
            # 셀 속성 변경
            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_G.write(i, k, '', cell_format)

            ## Blue 시트
            data = outImage[B][i][k]
            if data <= 15:
                hexStr = '#' + ('00000' + hex(data)[2:])
            else:
                hexStr = '#' + '0000' +  hex(data)[2:]
            # 셀 속성 변경
            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_B.write(i, k, '', cell_format)

            ##All 시트
            data = outImage[R][i][k]
            if data <= 15:
                hexStr = '#' + ('0' + hex(data)[2:])
            else:
                hexStr = '#' + hex(data)[2:]
            data = outImage[G][i][k]
            if data <= 15:
                hexStr += '0' + hex(data)[2:]
            else:
                hexStr += hex(data)[2:]

            data = outImage[B][i][k]
            if data <= 15:
                hexStr += '0' + hex(data)[2:]
            else:
                hexStr += hex(data)[2:]

            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_A.write(i, k, '', cell_format)

    wb.close()
    print('Excel Art. save ok...')

### OpenCV ###

def erodeCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    cvOutImage = np.ones((5,5),np.uint8)
    cvOutImage = cv2.erode(cvInImage,cvOutImage,iterations=1)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def dilationCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    cvOutImage = np.ones((5,5),np.uint8)
    cvOutImage = cv2.dilate(cvInImage,cvOutImage,iterations=1)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def cartoonGray() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return

    cvOutImage = cv2.cvtColor(cvInImage, cv2.COLOR_RGB2GRAY)
    cvOutImage = cv2.medianBlur(cvOutImage, 7)
    edges = cv2.Laplacian(cvOutImage, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    cvOutImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def cartoonColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return

    cvOutImage = cv2.stylization(cvInImage, sigma_s=150, sigma_r=0.5)

    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def videoCartoon():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    movieName = askopenfilename(parent=window,
                                filetypes=(('동영상 파일', '*.avi;*.mp4'), ('All File', '*.*')))
    s_factor = 0.4  # 화면 크기 비율

    capture = cv2.VideoCapture(movieName)
    while True:

        ret, frame = capture.read()
        frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
        if ret:
            cartoon = cv2.stylization(frame, sigma_s=150, sigma_r=0.5)
            cv2.imshow('Cartoon', cartoon)

            key = cv2.waitKey(1)  # 화면 속도 조절
            if key == 27:  # esc키
                cv2.destroyWindow('Cartoon')
                cv2.destroyWindow('Video')
                break
            elif key == ord('c') or key == ord('C'):
                # 키보드 외의 조건 처리도 가능함 (사람이 나오거나 개가 나오거나 등등)
                cvInImage = cvOutImage = cartoon
                filename = movieName
                cvOut2outImage()
                createOri()

        else:
            break
    capture.release()
    displayImageColor()
    makecopy()
    overwrite()

def blendingCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList, filename2

    if filename == '' or filename == None:
        return

    filename2 = askopenfilename(parent=window,
                                filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))

    a = 0.0

    while (a <= 1.0):
        cvInImage = cv2.imread(filename)
        img2 = cv2.imread(filename2)

        # 블렌딩하는 두 이미지의 크기가 같아야함
        width = cvInImage.shape[1]
        height = cvInImage.shape[0]
        img2 = cv2.resize(img2, (width, height))

        # img1 사진은 점점 투명해지고 img2 사진은 점점 불투명해짐
        b = 1.0 - a
        cvOutImage = cv2.addWeighted(cvInImage, a, img2, b, 0)
        cv2.imshow('dst', cvOutImage)
        key = cv2.waitKey(0)
        if key == 122 :
            a += 0.05
        elif key == 120 and a > 0:
            a -= 0.05
        elif key == 32 :
            break
        cvOut2outImage()

    cv2.destroyAllWindows()
    displayImageColor()
    makecopy()
    overwrite()

def mosaicCV():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return

    rate = 15  # 모자이크에 사용할 축소 비율 (1/rate)
    win_title = 'mosaic'  # 창 제목
    while True:
        x, y, w, h = cv2.selectROI(win_title, cvInImage, False)  # 관심영역 선택
        if w and h:
            roi = cvInImage[y:y + h, x:x + w]  # 관심영역 지정
            roi = cv2.resize(roi, (w // rate, h // rate))  # 1/rate 비율로 축소
            # 원래 크기로 확대
            roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
            cvInImage[y:y + h, x:x + w] = roi  # 원본 이미지에 적용
            cvOutImage = cvInImage
            cvOut2outImage()
        else:
            break

    cv2.destroyAllWindows()
    displayImageColor()
    makecopy()
    overwrite()

def mosaicCV2():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return

    ksize = 30  # 블러 처리에 사용할 커널 크기
    win_title = 'mosaic'  # 창 제목
    while True:
        x, y, w, h = cv2.selectROI(win_title, cvInImage, False)  # 관심영역 선택
        if w > 0 and h > 0:  # 폭과 높이가 음수이면 드래그 방향이 옳음
            roi = cvInImage[y:y + h, x:x + w]  # 관심영역 지정
            roi = cv2.blur(roi, (ksize, ksize))  # 블러(모자이크) 처리
            cvInImage[y:y + h, x:x + w] = roi  # 원본 이미지에 적용
            cvOutImage = cvInImage
            cvOut2outImage()
        else:
            break

    cv2.destroyAllWindows()
    displayImageColor()
    makecopy()
    overwrite()

## 러닝 ##
def ssdNet(image) :
    CONF_VALUE = 0.4
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image

def faceNet(image) :
    CONF_VALUE = 0.5
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return image

def deepStop():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    if filename == '' or filename == None:
        return
    cvOutImage = ssdNet(cvInImage)
    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def deepMove():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    movieName = askopenfilename(parent=window,
           filetypes=(('동영상 파일', '*.avi;*.mp4'), ('All File', '*.*')))
    s_factor = 0.5 #화면 크기 비율

    capture = cv2.VideoCapture(movieName)
    frameCount = 0 # 처리할 프레임의 숫자 (자동증가)

    while True:
        ret, frame = capture.read()
        if not ret:  # 동영상 읽기 실패
            break
        frameCount += 1
        if frameCount % 3 == 0 : #10프레임당 하나
            frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
            ## 1장짜리 SSD 딥러닝 ##
            retImage = ssdNet(frame)

            cv2.imshow('Video', retImage)

        key = cv2.waitKey(1) #화면 속도 조절
        if key == 27:  # esc키
            break
        elif key == ord('c') or key == ord('C'):
            #키보드 외의 조건 처리도 가능함 (사람이 나오거나 개가 나오거나 등등)
            cvInImage = cvOutImage = retImage
            filename = movieName
            cvOut2outImage()
            createOri()
            capture.release()
            displayImageColor()

    cv2.destroyWindow('Video')
    makecopy()
    overwrite()

def deepStopface():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    if filename == '' or filename == None:
        return
    cvOutImage = faceNet(cvInImage)
    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def deepMoveface():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    movieName = askopenfilename(parent=window,
           filetypes=(('동영상 파일', '*.avi;*.mp4'), ('All File', '*.*')))
    s_factor = 0.8 #화면 크기 비율

    capture = cv2.VideoCapture(movieName)
    frameCount = 0 # 처리할 프레임의 숫자 (자동증가)

    while True:
        ret, frame = capture.read()
        if not ret:  # 동영상 읽기 실패
            break
        frameCount += 1
        if frameCount % 3 == 0 : #10프레임당 하나
            frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
            ## 1장짜리 SSD 딥러닝 ##
            retImage = faceNet(frame)

            cv2.imshow('Video', retImage)

        key = cv2.waitKey(10) #화면 속도 조절
        if key == 27:  # esc키
            cv2.destroyWindow('Video')
            break
        elif key == ord('c') or key == ord('C'):
            #키보드 외의 조건 처리도 가능함 (사람이 나오거나 개가 나오거나 등등)
            cvInImage = cvOutImage = retImage
            filename = movieName
            cvOut2outImage()
            createOri()

    capture.release()
    displayImageColor()
    makecopy()
    overwrite()

def Emotion(image) :
    face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('emotion_model.hdf5', compile=False)
    EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canva = np.zeros((250, 300, 3), dtype="uint8")
    # Face detection in frame
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))
    if len(faces) > 0:
        # For the largest image
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Resize the image to 48x48 for neural network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Emotion predict
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Assign labeling
        cv2.putText(image, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

        #Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canva, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 255, 0), -1)
            cv2.putText(canva, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        cv2.imshow('emotion', canva)
    return image

def deepEmo():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    if filename == '' or filename == None:
        return
    cvOutImage = Emotion(cvInImage)
    cvOut2outImage()
    displayImageColor()
    makecopy()
    overwrite()

def deepMoveeMo():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    movieName = askopenfilename(parent=window,
                                filetypes=(('동영상 파일', '*.avi;*.mp4'), ('All File', '*.*')))
    s_factor = 0.5 #화면 크기 비율

    capture = cv2.VideoCapture(movieName)
    frameCount = 0 # 처리할 프레임의 숫자 (자동증가)

    while True:
        # Capture image from camera
        ret, frame = capture.read()
        frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)

        if not ret:  # 동영상 읽기 실패
            break
        frameCount += 1
        if frameCount % 5 == 0:  # 10프레임당 하나

            retImage = Emotion(frame)

            cv2.imshow('Video', retImage)

        key = cv2.waitKey(10) #화면 속도 조절

        if key == ord('c') or key == ord('C'):
            #키보드 외의 조건 처리도 가능함 (사람이 나오거나 개가 나오거나 등등)
            cvInImage = cvOutImage = retImage
            filename = movieName
            cvOut2outImage()
            createOri()
        elif key == ord('q') or key == ord("Q"):  # esc키
            break

    cv2.destroyWindow('emotion')
    cv2.destroyWindow('Video')
    capture.release()
    displayImageColor()
    makecopy()
    overwrite()

def deepMask():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    movieName = askopenfilename(parent=window,
                                filetypes=(('동영상 파일', '*.avi;*.mp4'), ('All File', '*.*')))
    capture = cv2.VideoCapture(movieName)

    # facenet : 얼굴을 찾는 모델
    facenet = cv2.dnn.readNet('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
    # model : 마스크 검출 모델
    model = load_model('mask_detector.model')
    frameCount = 0
    s_factor = 0.5 #화면 크기 비율

    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)

        if not ret:
            break
        frameCount += 1
        if frameCount % 10 == 0:  # 10프레임당 하나

            # 이미지의 높이와 너비 추출
            h, w = frame.shape[:2]

            # 이미지 전처리
            # ref. https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))

            # facenet의 input으로 blob을 설정
            facenet.setInput(blob)
            # facenet 결과 추론, 얼굴 추출 결과가 dets의 저장
            dets = facenet.forward()

            # 한 프레임 내의 여러 얼굴들을 받음
            result_frame = frame.copy()

            # 마스크를 착용했는지 확인
            for i in range(dets.shape[2]):

                # 검출한 결과가 신뢰도
                confidence = dets[0, 0, i, 2]
                # 신뢰도를 0.5로 임계치 지정
                if confidence < 0.5:
                    continue

                # 바운딩 박스를 구함
                x1 = int(dets[0, 0, i, 3] * w)
                y1 = int(dets[0, 0, i, 4] * h)
                x2 = int(dets[0, 0, i, 5] * w)
                y2 = int(dets[0, 0, i, 6] * h)

                # 원본 이미지에서 얼굴영역 추출
                face = frame[y1:y2, x1:x2]

                # 추출한 얼굴영역을 전처리
                face_input = cv2.resize(face, dsize=(224, 224))
                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(face_input)
                face_input = np.expand_dims(face_input, axis=0)

                # 마스크 검출 모델로 결과값 return
                mask, nomask = model.predict(face_input).squeeze()

                # 마스크를 꼈는지 안겼는지에 따라 라벨링해줌
                if mask > nomask:
                    color = (0, 255, 0)
                    label = 'Mask %d%%' % (mask * 100)
                else:
                    color = (0, 0, 255)
                    label = 'No Mask %d%%' % (nomask * 100)
                    frequency = 1500  # Set Frequency To 2500 Hertz
                    duration = 300  # Set Duration To 1000 ms == 1 second
                    winsound.Beep(frequency, duration)

                # 화면에 얼굴부분과 마스크 유무를 출력해해줌
                cv2.rectangle(result_frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
                cv2.putText(result_frame, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=color, thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow('Mask', result_frame)

        key = cv2.waitKey(10)  # 화면 속도 조절
        # q를 누르면 종료
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            # 키보드 외의 조건 처리도 가능함 (사람이 나오거나 개가 나오거나 등등)
            cvInImage = cvOutImage = result_frame
            filename = movieName
            cvOut2outImage()
            createOri()

    cv2.destroyWindow('Mask')
    capture.release()
    displayImageColor()
    makecopy()
    overwrite()

## 전역 변수부
window, canvas, paper = None, None, None
inImage, outImage = [], [];  inH, inW, outH, outW = [0] * 4
cvInImage, cvOutImage = None, None
filename = ''
RGB,R, G, B= 3, 0, 1, 2
status = None
loginID = ""

#DB 관련
conn, cur = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
fileList = None
## 메인코드부
def main(login) :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, status
    global cvInImage, cvOutImage, fileList, filename,loginID
    window = Tk()
    window.title('영상 편집 Ver 0.9')
    window.geometry('512x512')
    window.resizable(height=False, width=False)
    status = Label(window, text='이미지정보:', bd=1, relief=SUNKEN, anchor=W)
    status.pack(side=BOTTOM, fill=X)
    loginID = login


    ### 메뉴 만들기 ###
    mainMenu = Menu(window)
    window.configure(menu=mainMenu)

    fileMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="파일", menu=fileMenu)
    fileMenu.add_command(label="열기(Open)", command=openFile)
    fileMenu.add_command(label="저장(Save)", command=saveFile)
    fileMenu.add_separator()
    fileMenu.add_command(label="취소", command=undo)
    fileMenu.add_command(label="다시실행", command=redo)
    fileMenu.add_separator()
    fileMenu.add_command(label="닫기(Close)", command=quit)

    pixelMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="화소점 처리", menu=pixelMenu)
    pixelMenu.add_command(label="동일영상", command=equalColor)
    pixelMenu.add_command(label="밝게하기(합)", command=addColor)
    pixelMenu.add_command(label="어둡게하기(차)", command=minusColor)

    binaryMenu = Menu(pixelMenu)
    pixelMenu.add_cascade(label="이진화", menu=binaryMenu)
    binaryMenu.add_command(label="이진화(중간)", command=binaryColor)
    binaryMenu.add_command(label="이진화(평균)", command=avgbinColor)
    binaryMenu.add_command(label="이진화(중위)", command=cenbinColor)

    transMenu = Menu(pixelMenu)
    pixelMenu.add_cascade(label="색변환", menu=transMenu)
    transMenu.add_command(label="범위강조", command=concenColor)
    transMenu.add_command(label="그레이필터", command=grayColor)
    transMenu.add_command(label="색반전", command=exColor)
    transMenu.add_command(label="포스터라이징", command=posterColor)

    pixelMenu.add_separator()
    pixelMenu.add_command(label="감마", command=gammaColor)
    pixelMenu.add_command(label="파라볼라(캡)", command=paraCapColor)
    pixelMenu.add_command(label="파라볼라(컵)", command=paraCupColor)

    geoMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="기하학 처리", menu=geoMenu)
    geoMenu.add_command(label="좌우반전", command=mirrorLRColor)
    geoMenu.add_command(label="상하반전", command=mirrorUDColor)
    geoMenu.add_command(label="이동", command=moveColor)
    geoMenu.add_command(label="회전", command=rotateColor)
    geoMenu.add_command(label="확대", command=sizeUpColor)
    geoMenu.add_command(label="축소", command=sizeDownColor)

    areaMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="화소영역 처리", menu=areaMenu)

    effectMenu = Menu(areaMenu)
    areaMenu.add_cascade(label="필터 효과", menu=effectMenu)
    effectMenu.add_command(label="블러링", command=blurColor)
    effectMenu.add_command(label="샤프팅", command=shaftColor)
    effectMenu.add_command(label="엠보싱", command=emboColor)
    effectMenu.add_command(label="가우시안", command=gaussColor)

    edgeMenu = Menu(areaMenu)
    areaMenu.add_cascade(label="경계추출", menu=edgeMenu)
    edgeMenu.add_command(label="경계추출", command=cannyColor)
    edgeMenu.add_command(label="고주파", command=gozupaColor)
    edgeMenu.add_command(label="유사연산자", command=yusaColor)
    edgeMenu.add_command(label="차연산자", command=chaColor)
    edgeMenu.add_command(label="소벨", command=sobelColor)
    edgeMenu.add_command(label="로버츠", command=robertsColor)
    edgeMenu.add_command(label="프리윗", command=prewittColor)
    edgeMenu.add_command(label="라플라시안", command=laplaceColor)
    areaMenu.add_separator()
    areaMenu.add_command(label="LoG", command=LoGColor)
    areaMenu.add_command(label="DoG", command=DoGColor)
    areaMenu.add_command(label="DoG2", command=DoG2Color)

    histoMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="히스토그램 처리", menu=histoMenu)
    histoMenu.add_command(label="스트레칭", command=stretchColor)
    histoMenu.add_command(label="엔드 인 탐색", command=endInColor)
    histoMenu.add_command(label="이퀄라이즈드", command=equalizedColor)

    mysqlMenu = Menu(window)
    mainMenu.add_cascade(label="MySQL", menu=mysqlMenu)
    mysqlMenu.add_command(label="MySQL에 저장", command=upMySQL)
    mysqlMenu.add_command(label="MySQL에서 열기", command=downMySQL)
    mysqlMenu.add_command(label="폴더 업로드", command=folderupMySQL)

    excelMenu = Menu(window)
    mainMenu.add_cascade(label="Excel", menu=excelMenu)
    excelMenu.add_command(label="Excel에 저장", command=saveExcel)
    excelMenu.add_command(label="Excel에서 열기", command=openExcel)
    excelMenu.add_separator()
    excelMenu.add_command(label="Excel 아트", command=drawExcel)

    openCVMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="openCV", menu=openCVMenu)
    openCVMenu.add_command(label="침식", command=erodeCV)
    openCVMenu.add_command(label="팽창", command=dilationCV)
    openCVMenu.add_command(label="카툰화(흑백)", command=cartoonGray)
    openCVMenu.add_command(label="카툰화(칼라)", command=cartoonColor)
    openCVMenu.add_command(label="영상 카툰화", command=videoCartoon)
    openCVMenu.add_command(label="두가지 사진 섞기", command=blendingCV)
    openCVMenu.add_command(label="모자이크", command=mosaicCV)
    openCVMenu.add_command(label="모자이크2", command=mosaicCV2)

    DeepMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="딥러닝", menu=DeepMenu)
    DeepMenu.add_command(label="사물인식(정지)", command=deepStop)
    DeepMenu.add_command(label="사물인식(영상)", command=deepMove)
    DeepMenu.add_command(label="안면인식(정지)", command=deepStopface)
    DeepMenu.add_command(label="안면인식(영상)", command=deepMoveface)
    DeepMenu.add_command(label="표정인식(정지)", command=deepEmo)
    DeepMenu.add_command(label="표정인식(영상)", command=deepMoveeMo)
    DeepMenu.add_command(label="마스크(영상)", command=deepMask)

    window.mainloop()