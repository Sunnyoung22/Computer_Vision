## 【计算机视觉】基于SIFT的图像拼接
##            2020.10.4 Sunnyoung

import cv2
import numpy as np

# 计算亮度
def get_brightness(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = np.median(hsv_image[:,:,2])
    return brightness

# 递归裁剪函数，将拼接后图片多余的黑边除去
def crop(frame):
    if not np.sum(frame[0]):#顶部一行
        return crop(frame[1:])
    if not np.sum(frame[-1]):#底部
        return crop(frame[-2:])
    if not np.sum(frame[:,0]):#左边界
        return crop(frame[:,1])
    if not np.sum(frame[:,-1]):#右边界
        return crop(frame[:,:-15])
    return frame

picl_name = ".\\data\\Boat_a.jpg"
picr_name = ".\\data\\Boat_b.jpg"

# 读取需要进行拼接的两幅图片，进行预处理（尺寸与灰度的变换）
origin_img_l = cv2.imread(picl_name)
img_left = cv2.resize(origin_img_l, (0,0), fx=0.2, fy=0.2)
img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

origin_img_r = cv2.imread(picr_name)
img_right = cv2.resize(origin_img_r, (0,0), fx=0.2, fy=0.2)
img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
# print(origin_img_l.shape)
# print(img_left.shape)

cv2.imshow("Boat_left", img_left)
cv2.imshow("Boat_right", img_right)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# 使用SIFT算法找出图像的关键点，返回关键点和特征描述子
sift = cv2.xfeatures2d.SIFT_create()
kp_a, des_a = sift.detectAndCompute(img_left_gray, None)
kp_b, des_b = sift.detectAndCompute(img_right_gray, None)

# 在图像上绘制关键点
cv2.imshow("Feature point", cv2.drawKeypoints(img_left, kp_a, None))
cv2.waitKey(2000)
cv2.destroyAllWindows()

# 构造BFMatcher()蛮力匹配，匹配sift特征向量距离最近对应组分
match = cv2.BFMatcher()
matches = match.knnMatch(des_a, des_b, k=2)

# 取一幅图像中的一个SIFT关键点，并找出其与另一幅图像中欧式距离最近的前两个关键点，在这两个关键点中，
# 如果最近的距离除以次近的距离得到的比率ratio少于某个阈值T，则接受这一对匹配点。
# d1:最近邻，d2:次近邻。即d1<k*d2。
# 我们知道距离越近匹配度越高，但是，当所有点的距离都比较近时，匹配的可靠性不高。反之，如果只有点一个距离比较近，
# 其它点距离都相对较远时，该点匹配的可靠度增加。d1<k*d2就是为了说明这一点。
good_points = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

# imageA和imageB表示图片，kpsA和kpsB表示关键点，matches表示经过cv2.BFMatcher获得的匹配的索引值，flags表示有几个图像
img3 = cv2.drawMatches(img_left, kp_a, img_right, kp_b, good_points, None, flags = 2)
cv2.imshow("Feature point matching",img3)
cv2.waitKey(3000)
cv2.destroyAllWindows()

MIN_MATCH_COUNT = 10
if len(good_points) > MIN_MATCH_COUNT:
    dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_points])
    src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_points])
    # 在这个函数参数中，输入的m1和m2是两个对应的序列，这两组序列的每一对数据一一匹配，其中既有正确的匹配，也有错误的匹配，
    # 正确的可以称为内点，错误的称为外点，RANSAC方法就是从这些包含错误匹配的数据中，分离出正确的匹配，并且求得单应矩阵。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches are found - %d/%d", (len(good_points)/MIN_MATCH_COUNT))
# 获得根据单应性矩阵变化后的图像
transformed_image = cv2.warpPerspective(img_right,M,(img_right.shape[1] + img_left.shape[1], img_left.shape[0]))
cv2.imshow("Transformed image", transformed_image)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# 将左侧图片与变换后的右侧图片连接
transformed_image[0:img_left.shape[0],0:img_left.shape[1]] = img_left
cv2.imshow("Image stitching", transformed_image)
cv2.waitKey(2000)
cv2.destroyAllWindows()

result=crop(transformed_image)
cv2.imshow("The final result of image stitching", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
