## 【计算机视觉】基于PCA的笑脸判断
##            2020.10.4 Sunnyoung
import numpy as np
import cv2

IMAGE_SIZE = (60,60)    # 将读入图片尺寸固定为60*60

def image2vector(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray,IMAGE_SIZE,interpolation=cv2.INTER_CUBIC)
    high, width = image_gray.shape
    # 转换为行向量
    image_vector = image_gray.reshape(high*width)
    return image_vector

def get_image_matrix(path, total_num):
    image_matrix = []
    for i in range(1, total_num+1):
        image = cv2.imread(path + str(i) + ".png")
        image_vector = image2vector(image)
        image_matrix.append(image_vector)
    # 转换为二维矩阵，一行是一张图
    image_matrix = np.array(image_matrix)
    return image_matrix

# 零均值化
def zero_mean(data_matrix):
    mean_val = np.mean(data_matrix, axis=0) # 按列求均值
    result_matrix = data_matrix - mean_val  # 零均值化
    return result_matrix, mean_val

def pca(data_matrix, k):
    matrix_zero_mean, mean_val = zero_mean(data_matrix)
    matrix_zero_mean = np.mat(matrix_zero_mean)# 创建矩阵类型的数据
    
    """计算协方差矩阵"""
    cov_mat = matrix_zero_mean * matrix_zero_mean.T
    # cov_mat = np.cov(matrix_zero_mean, rowvar=1)
    egvalue, egvector_ = np.linalg.eig(cov_mat)
    egval_ascending = np.argsort(egvalue)       # 对特征值从小到大排序
    index = egval_ascending[-1:-(k+1):-1]       # 最大的k个特征值的下标
    egvector = egvector_[:,index]               # 最大的k个特征值对应的特征向量

    egvector = matrix_zero_mean.T * egvector    # 得到协方差矩阵(cov_mat)的特征向量
    for i in range(k):                          # 特征向量归一化
        egvector[:,i] /= np.linalg.norm(egvector[:,i])
    
    #降维后的数据矩阵
    low_D_data_mat = np.array(matrix_zero_mean * egvector)
    return low_D_data_mat, mean_val, egvector
    
def smiling_face_judgment(test_image, pca_mat, mean_val, egvector):
    train_total_number = pca_mat.shape[0]
    image_vector = image2vector(test_image)
    test_image_zero_mean = image_vector - mean_val
    pca_mat_new = np.array(test_image_zero_mean * egvector) # 得到测试脸在特征向量下的数据

    distance = []
    for i in range(0, train_total_number):
        temp = pca_mat[i,:]
        dis = np.linalg.norm(pca_mat_new - temp) # 计算欧几里得范数
        distance.append(dis)
    minDistance = min(distance) # 取得最小距离
    index = distance.index(minDistance) # 取得最小距离图片的下标
    if index + 1 <= 14:
        return True
    else:
        return False

if __name__ =='__main__':

    TRAIN_FOLDER = ".\\data\\face_images\\Train\\"
    TEST_FOLDER = ".\\data\\face_images\\Test\\"
    TRAIN_TOTAL_NUMBER = 28   # 训练图片数量28张
    TEST_TOTAL_NUMBER = 14    # 测试图片数量14张
    image_matrix = get_image_matrix(TRAIN_FOLDER, TRAIN_TOTAL_NUMBER)
    pca_mat, mean_val, egvector = pca(image_matrix, 20)   # 得到降维矩阵，均值与eigenface

    correct_num = 0
    for i in range(1, TEST_TOTAL_NUMBER+1):
        test_image = cv2.imread(TEST_FOLDER + str(i) +".png")
        if i <= 7 and smiling_face_judgment(test_image, pca_mat, mean_val, egvector): 
            correct_num += 1
        if i > 7 and smiling_face_judgment(test_image, pca_mat, mean_val, egvector) == False: 
            correct_num += 1
        accuracy = float(correct_num)/TEST_TOTAL_NUMBER
    print("The recognition accuracy is: %.2f%%" %(accuracy * 100))

    print("-----------------------------------------")
    test_num = input("| Please enter a picture number(1~14)：")
    test_image = cv2.imread(TEST_FOLDER + test_num +".png")
    result = smiling_face_judgment(test_image, pca_mat, mean_val, egvector)
    if result:
        print("\n| This picture is a smiley face.")
        print("-----------------------------------------")
    else:
        print("\n| This picture is not a smiley face.")
        print("-----------------------------------------")
    cv2.namedWindow("Test picture", 0)
    cv2.resizeWindow("Test picture", 180, 180)
    cv2.imshow("Test picture",test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()