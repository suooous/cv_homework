import cv2
import numpy as np
import os

# 创建输出目录（如果不存在）
output_dir = "processing_step66s"
os.makedirs(output_dir, exist_ok=True)

def save_image(image, step_name):
    """保存图像到指定目录"""
    filename = f"{output_dir}/{step_name}.jpg"
    cv2.imwrite(filename, image)
    print(f"已保存: {filename}")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
# 读取图像
image = cv2.imread("img6.jpg")
orig = image.copy()
save_image(image, "01_original")  # 保存原始图像

# 1. 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
save_image(gray, "02_gray")

# 2. 高斯模糊
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
save_image(blurred, "03_blurred")

# 3. 边缘检测
edged = cv2.Canny(blurred, 50, 150)
save_image(edged, "04_edged")

# 4. 形态学操作
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edged, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)
save_image(dilated, "05_dilated")
save_image(eroded, "06_eroded")

# 5. 轮廓检测
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
save_image(contour_img, "07_all_contours")

# 6. 筛选四边形轮廓
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.7 < aspect_ratio < 1.4 and cv2.contourArea(c) > 5000:
            screenCnt = approx
            # 绘制最终轮廓
            final_contour_img = cv2.drawContours(image.copy(), [screenCnt], -1, (0, 0, 255), 3)
            save_image(final_contour_img, "08_final_contour")
            break

if 'screenCnt' not in locals():
    raise ValueError("未检测到四边形轮廓！")

# 7. 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2))
save_image(warped, "09_warped_final_result")



