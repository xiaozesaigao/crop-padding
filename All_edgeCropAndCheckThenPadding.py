import os
import cv2
import numpy as np
from PIL import Image

# 设置文件夹路径
input_folder = '../changshi'  # 要处理的图像集路径
output_folder = '../out'  # 保存路径
os.makedirs(output_folder, exist_ok=True)


# 边缘检测并裁剪图像函数
def edge_crop(image):
	# 转换为灰度图像
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# 应用 Canny 边缘检测
	edges = cv2.Canny(gray, threshold1=100, threshold2=200)
	
	# 找到边缘的非零像素
	coords = np.column_stack(np.where(edges > 0))
	
	if len(coords) == 0:
		return None  # 如果没有边缘，返回 None
	
	# 获取最小边界框
	x_min, y_min = coords.min(axis=0)
	x_max, y_max = coords.max(axis=0)
	
	# 裁剪图像
	cropped_image = image[x_min:x_max, y_min:y_max]
	
	return cropped_image


# 填充图像函数
def pad_to_400(image):
	h, w, _ = image.shape
	# 创建一个400x400的白色背景
	background = np.full((400, 400, 3), 255, dtype=np.uint8)
	
	# 将原始图像居中放置到白色背景上
	y_offset = (400 - h) // 2
	x_offset = (400 - w) // 2
	background[y_offset:y_offset + h, x_offset:x_offset + w] = image
	
	return background


# 遍历子文件夹并处理图像
for subdir, _, files in os.walk(input_folder):
	for file in files:
		if file.endswith(('.jpg', '.png', '.jpeg')):
			image_path = os.path.join(subdir, file)
			image = cv2.imread(image_path)
			
			# 执行边缘检测并裁剪
			cropped_image = edge_crop(image)
			
			if cropped_image is None:
				print(f"跳过没有边缘的图像: {file}")
				continue
			
			h, w, _ = cropped_image.shape
			
			# 丢弃宽或高小于100的图像
			if h < 100 or w < 100:
				print(f"丢弃图像: {file}, 尺寸 ({h}, {w}) 太小")
				continue
			
			# 如果图像宽或高大于400，保持不变
			if h > 400 or w > 400:
				# output_image = cropped_image
				print(f"丢弃图像: {file}, 尺寸 ({h}, {w}) 太大")
				continue
			else:
				# 否则进行白色填充至400x400
				output_image = pad_to_400(cropped_image)
			
			# 保存处理后的图像
			output_path = os.path.join(output_folder, file)
			cv2.imwrite(output_path, output_image)

print("over！")
