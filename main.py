from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Process,Manager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def read_img_gray(imageName):
    cur_dir = '/'.join(os.path.abspath(__file__).split('\\')[:-1])
    dir = cur_dir + imageName
    img = cv2.imread(dir,0)
    return img
    
def read_img_rgb(imageName):
    cur_dir = '/'.join(os.path.abspath(__file__).split('\\')[:-1])
    dir = cur_dir + imageName
    img = cv2.imread(dir,1)
    # 将图像通道分离开。
    b, g, r = cv2.split(img)
    # 以RGB的形式重新组合。
    rgb_image = cv2.merge([r, g, b])
    return rgb_image,r,g,b

def get_ICH(img,z=1):
    H=np.zeros(shape=(256, 256))
    for i in range(z,len(img)-z,2*z+1):
        for j in range(z,len(img[0])-z,2*z+1):
            area = img[i-z:i+z+1,j-z:j+z+1].ravel()
            for k in list(area):
                H[int(img[i][j]),int(k)] += 1
                H[int(k),int(img[i][j])] += 1 
    return H

def get_PMF(H):
    P = H/np.sum(H)
    return P

def get_U(P):
    return 1/np.sum(P!=0)

def get_P_(P,U):
    return np.where((P == 0) | (P > U), 0, U-P)

def get_S(P_,X,z):
    S=np.zeros(shape=(len(X),len(X[0])))
    for i in range(z,len(X)-z):
        for j in range(z,len(X[0])-z):
            area = X[i-z:i+z+1,j-z:j+z+1].ravel()
            for k in list(area):
                S[i][j] +=  P_[X[i][j]][k]
    return S

def liner_map(array):
    #映射到0~255且四舍五入取整
    minvalue = np.min(array)
    maxvalue = np.max(array) - minvalue
    new_array = np.around((array - minvalue)*255/maxvalue,0)
    return np.uint8(new_array)

def binary_map(array):
    #映射到0,255
    return np.uint8(np.where((array == 0) , 0, 255))

def fun(X,z,c,rgb_result):
    #多进程同时执行计算
    print(c,"begin")
    H = get_ICH(X,z)
    P = get_PMF(H)
    U = get_U(P)
    P_ = get_P_(P,U)
    S = get_S(P_,X,z)
    rgb_result[c]=S
    return S

def set_form(r,g,b):
    #对RGB读取格式进行加和
    rgb = r + g + b
    rgb = np.where((rgb > 255) , 255, rgb)
    return rgb
def Gauss_pro(img):
    Gauss = cv2.GaussianBlur(img, (231, 231), 0)
    Gauss = np.where((Gauss<40), 255 , 0)    
    return Gauss

def get_result(X,Y):
    result = cv2.add(X,Y)
    return result

def show_ICH(pic):
    #构造需要显示的值
    X=np.arange(0, 256, step=1)#X轴的坐标
    Y=np.arange(0, 256, step=1)#Y轴的坐标
    #设置每一个（X，Y）坐标所对应的Z轴的值，在这边Z（X，Y）=X+Y
    # Z=np.zeros(shape=(10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         Z[i, j]=i+j

    xx, yy=np.meshgrid(X, Y)#网格化坐标
    X, Y=xx.ravel(), yy.ravel()#矩阵扁平化
    bottom=np.zeros_like(X)#设置柱状图的底端位值
    # Z=Z.ravel()#扁平化矩阵

    width=height=1#每一个柱子的长和宽

    #绘图设置
    fig=plt.figure()
    ax=fig.gca(projection='3d')#三维坐标轴
    ax.bar3d(X, Y, bottom, width, height, pic.ravel(), shade=True)#
    #坐标轴设置
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z(value)')
    plt.show()
    

if __name__ == '__main__':

    #model为0为灰色模式，为1为rgb模式
    model = 1
    z = 5
    imageName = "/test.jpg"
    # X = read_img_gray(imageName)
    X,r,g,b = read_img_rgb(imageName)
    X.shape
    r.shape
    g.shape
    b.shape
    result1 = Image.fromarray(X)
    result1.show()


    #灰色图像
    if model == 0:
        X = read_img_gray(imageName)
        H = get_ICH(X,z)
        
        print("H:",H.shape)
        P = get_PMF(H)
        # show_ICH(P)
        print("P:",type(P))
        U = get_U(P)
        print("U:",U)
        P_ = get_P_(P,U)
        
        print("Get P_ Over")
        S = get_S(P_,X,z)
        print("Get S Over")
        liner_S = liner_map(S)
        binary_S = binary_map(S)
        result1 = Image.fromarray(liner_S)
        result1.show()
        result2 = Image.fromarray(binary_S)
        result2.show()
    elif model == 1:
        X,r,g,b = read_img_rgb(imageName)
        
        rgb_result = Manager().dict()
        p1 = Process(target = fun,args=(r,z,'r',rgb_result))
        p2 = Process(target = fun,args=(g,z,'g',rgb_result))
        p3 = Process(target = fun,args=(b,z,'b',rgb_result))
        begin = time.time()
        p1.start()
        p2.start()
        p3.start()
        p1.join()
        p2.join()
        p3.join()
        end = time.time()
        print("运行时间为" + str(end - begin) + "s")
        
        
        '''先加起来再做映射
        rgb = set_form(rgb_result['r'], rgb_result['g'], rgb_result['b'])

        liner_rgb = liner_map(rgb)
        binary_rgb = binary_map(rgb)
        '''
        '''先做映射再加起来'''
        liner_r = liner_map(rgb_result['r'])
        liner_g = liner_map(rgb_result['g'])
        liner_b = liner_map(rgb_result['b'])
        liner_rgb = set_form(liner_r,liner_g,liner_b)
        
        # binary_r = binary_map(rgb_result['r'])
        # binary_g = binary_map(rgb_result['g'])
        # binary_b = binary_map(rgb_result['b'])
        # binary_rgb = set_form(binary_r,binary_g,binary_b)
        

        temp_liner_rgb = np.uint8(np.where((liner_rgb < 150) , 0, 255))
        result_rgb_liner = Image.fromarray(temp_liner_rgb)
        # result_rgb_liner.show()
        # result_rgb_bin = Image.fromarray(binary_rgb)
        
        
        # result_rgb_liner.save("result_rgb_liner.jpg")
        # result_rgb_bin.show()
        # result_rgb_bin.save("result_rgb_bin.jpg")
        
        Gauss = cv2.GaussianBlur(temp_liner_rgb, (231, 231), 0)
        Gauss = Gauss_pro(temp_liner_rgb)  
        Gauss_img = Image.fromarray(Gauss)
        # Gauss_img.show()
            

        gauss = np.uint8(cv2.merge([Gauss, Gauss, Gauss]))
        result = get_result(X, gauss)

        blend_img = Image.fromarray(result)
        blend_img.show()
        
