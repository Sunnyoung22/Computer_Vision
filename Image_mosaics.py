## 2020.10.4 Sunnyoung
import cv2
import numpy as np

picl_name = ".\\data\\Boat_a.jpg"
picr_name = ".\\data\\Boat_b.jpg"
# picl_name = ".\\data\\City_a.jpg"
# picr_name = ".\\data\\City_b.jpg"

# Read the pictures that needed to solve
origin_img_l = cv2.imread(picl_name) 
img_left = cv2.resize(origin_img_l, (0,0), fx=0.2, fy=0.2)
img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)

origin_img_r = cv2.imread(picr_name)
img_right = cv2.resize(origin_img_r, (0,0), fx=0.2, fy=0.2)
img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)

# print(origin_img_l.shape)
# print(img_left.shape)
cv2.imshow("Boat_left", img_left)
cv2.imshow("Boat_right", img_right)
cv2.waitKey(0)

sift = cv2.xfeatures2d.SIFT_create()
# find the key points and descriptors with SIFT
kp_a, dist_a = sift.detectAndCompute(img_left_gray, None)
kp_b, dist_b = sift.detectAndCompute(img_right_gray, None)

cv2.imshow("Feature point", cv2.drawKeypoints(img_left, kp_a, None))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 构造BFMatcher()蛮力匹配，匹配sift特征向量距离最近对应组分
match = cv2.BFMatcher()
matches = match.knnMatch(dist_a, dist_b, k=2)

# 取一幅图像中的一个SIFT关键点，并找出其与另一幅图像中欧式距离最近的前两个关键点，在这两个关键点中，
# 如果最近的距离除以次近的距离得到的比率ratio少于某个阈值T，则接受这一对匹配点。
# d1:最近邻，d2:次近邻。即d1<k*d2。
# 我们知道距离越近匹配度越高，但是，当所有点的距离都比较近时，匹配的可靠性不高。反之，如果只有点一个距离比较近，
# 其它点距离都相对较远时，该点匹配的可靠度增加。d1<k*d2就是为了说明这一点。
good_points = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

# imageA和imageB表示图片，kpsA和kpsB表示关键点， matches表示进过cv2.BFMatcher获得的匹配的索引值，也有距离， flags表示有几个图像
img3 = cv2.drawMatches(img_left, kp_a, img_right, kp_b, good_points, None, flags = 2)
cv2.imshow("Feature point matching",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

MIN_MATCH_COUNT = 15
if len(good_points) > MIN_MATCH_COUNT:
    dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
    src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
    # 在这个函数参数中，输入的m1和m2是两个对应的序列，这两组序列的每一对数据一一匹配，其中既有正确的匹配，也有错误的匹配，
    # 正确的可以称为内点，错误的称为外点，RANSAC方法就是从这些包含错误匹配的数据中，分离出正确的匹配，并且求得单应矩阵。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = img_right_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img_left_gray = cv2.polylines(img_left_gray,[np.int32(dst)],True,0,3, cv2.LINE_AA)
    cv2.imshow("original_image_overlapping.jpg", img_left_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

dst = cv2.warpPerspective(img_right,M,(img_right.shape[1] + img_left.shape[1], img_left.shape[0]))
cv2.imshow("original_image_stitched.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
dst[0:img_left.shape[0],0:img_left.shape[1]] = img_left
cv2.imshow("original_image_stitched.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.destroyAllWindows()
