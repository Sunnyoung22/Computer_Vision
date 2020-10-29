# 检测笑不笑
import numpy as np
import cv2

def image2vector(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high, width = image_gray.shape
    # 转换为行向量
    image_vector = image_gray.reshape(high*width)
    return image_vector

def get_image_matrix(path, number):
    image_matrix = []
    for i in range(1, number+1):
        image = cv2.imread(path+"\\"+str(i)+".png")
        image_vector = image2vector(image)
        image_matrix.append(image_vector)
    # 转换为二维矩阵，一列是一张图
    image_matrix = np.array(image_matrix)

    return image_matrix

# 零均值化
def zero_mean(data_matrix):
    mean_val = np.mean(data_matrix, axis=0) # 按列求均值
    resutl_matrix = data_matrix - mean_val
    return resutl_matrix, mean_val

def pca(data_matrix, k):
    matrix_zero_mean, mean_val = zero_mean(data_matrix)
    matrix_zero_mean = np.mat(matrix_zero_mean)# 创建矩阵类型的数据
    """计算协方差矩阵C的替代L"""
    cov_mat = matrix_zero_mean * matrix_zero_mean.T     # 采用SVD奇异特征值法可以减小计算量
    D, V = np.linalg.eig(cov_mat) # 特征向量V[:,i]对应特征值D[i]
    V1 = V[:,0:k] # 取前K个特征向量
    V1 = matrix_zero_mean.T * V1 # 得到协方差矩阵(covMatT')的特征向量

    for i in range(k):                                 # 特征向量归一化
        V1[:,i] /= np.linalg.norm(V1[:,i])

    #low_D_data_mat = np.array(Z*V1)
    low_D_data_mat = matrix_zero_mean * V1
    return low_D_data_mat, mean_val, V1
        

if __name__ =='__main__':
    print("Haha")
    PICTURE_FOLDER=".\\data\\faceImages\\nonsmiling"
    TRAIN_NUMBER = 14
    nonsmiling_image_matrix = get_image_matrix(PICTURE_FOLDER, TRAIN_NUMBER)
    eigenfaces_nonsmilling, mean_val, V = pca(nonsmiling_image_matrix, 10) #???

