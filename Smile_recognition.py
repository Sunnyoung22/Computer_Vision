# 检测笑不笑
import numpy as np
import cv2

PICTURE_FOLDER=".\\result\\faceImages\\nonsmiling"

def image2array(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high, width = image_gray.shape
    # 转换为行向量
    image_array = image_gray.reshape(high*width)
    return image_array

def get_image_matrix(path, number):
    image_matrix = []
    for i in range(1, number+1):
        image = cv2.imread(path+"\\"+str(i)+".png")
        image_array = image2array(image)
        image_matrix.append(image_array)
    # 转换为二维矩阵，一列是一张图
    image_matrix = np.array(image_matrix)

    return image_matrix

# 零均值化
def zero_mean(data_matrix):
    mean_val = np.mean(data_matrix, axis=0) # 按列求均值
    resutl_matrix = data_matrix - mean_val
    return resutl_matrix, mean_val

def pca(data_matrix, k):
    data_new, mean_val = zero_mean(data_matrix)
    """计算协方差矩阵C的替代L"""
    
    diffMatrix = numpy.mat(data_new)# 创建矩阵类型的数据
    cov_mat = diffMatrix * diffMatrix.T     # 采用SVD奇异特征值法可以减小计算量
    D, V = np.linalg.eig(cov_mat) # 特征向量V[:,i]对应特征值D[i]
    V1 = V[:,0:k] # 取前K个特征向量
    V1 = diffMatrix.T * V1

    for i in range(k):
        


if __name__ =='__main__':
    print("Haha")