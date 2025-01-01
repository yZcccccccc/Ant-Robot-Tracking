import cv2
import numpy as np
import os

# 路径定义
base_path = r'D:\result\mobile\wx_mobile'
rgb_path = os.path.join(base_path, 'rgb')              # absolutely moving objects' mask
depth_path = os.path.join(base_path, 'depth')
chair_path = os.path.join(base_path, 'chair')          # movable objects' mask
ws_file = os.path.join(base_path, 'wx.txt')
output_path = os.path.join(base_path, 'output')

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)

# 读取关联文件
with open(ws_file, 'r') as file:
    associations = file.readlines()


def get_connected_components(mask):
    """ 获取mask中的连通区域 """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return num_labels, labels, stats


# 处理每个关联文件
for line in associations:
    # 解析文件名
    parts = line.strip().split()
    timestamp = parts[0]
    rgb_file = os.path.join(base_path, parts[1])
    depth_file = os.path.join(base_path, parts[3])

    # 读取mask和深度图像
    person_mask = cv2.imread(rgb_file, cv2.IMREAD_GRAYSCALE)
    depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    chair_mask = cv2.imread(os.path.join(chair_path, os.path.basename(rgb_file)), cv2.IMREAD_GRAYSCALE)

    # 检查person mask是否为空
    if cv2.countNonZero(person_mask) == 0:
        output_image = np.zeros_like(person_mask)
        cv2.imwrite(os.path.join(output_path, os.path.basename(rgb_file)), output_image)
        continue

    # 获取person mask中白色区域对应的深度值
    person_mask_indices = (person_mask == 255)
    depth_values_person = depth_map[person_mask_indices]

    # 忽略掉深度值为0的区域
    depth_values_person = depth_values_person[depth_values_person > 0]

    # 如果没有有效的深度值，输出全黑图像
    if len(depth_values_person) == 0:
        output_image = np.zeros_like(person_mask)
        cv2.imwrite(os.path.join(output_path, os.path.basename(rgb_file)), output_image)
        continue

    # 获取深度值的范围R
    R_min, R_max = np.min(depth_values_person), np.max(depth_values_person)

    # 获取chair mask中白色区域对应的深度值
    chair_mask_indices = (chair_mask == 255)
    depth_values_chair = depth_map[chair_mask_indices]

    # 扩展person mask用于邻近检测（3x3邻域）
    kernel = np.ones((3, 3), np.uint8)
    person_mask_dilated = cv2.dilate(person_mask, kernel)

    # 判断chair mask中白色区域对应的深度值是否在R范围内
    if np.any((depth_values_chair >= R_min) & (depth_values_chair <= R_max)):
        # 获取chair mask中的连通区域
        num_labels, labels, stats = get_connected_components(chair_mask)

        # 遍历所有连通区域
        chair_combined = np.zeros_like(chair_mask)
        for label in range(1, num_labels):
            chair_component = (labels == label).astype(np.uint8) * 255

            # 检查当前连通区域是否与person mask邻近
            if np.any(cv2.bitwise_and(person_mask_dilated, chair_component)):
                # 如果邻近，将连通区域加入到结果中
                chair_combined = cv2.bitwise_or(chair_combined, chair_component)

        # 组合person mask的白色区域和邻近部分的chair mask的白色区域
        output_image = cv2.bitwise_or(person_mask, chair_combined)
    else:
        # 如果深度值不在范围内，只输出person mask的白色区域
        output_image = person_mask

    # 保存输出图像，名称与person mask名称一致
    cv2.imwrite(os.path.join(output_path, os.path.basename(rgb_file)), output_image)
