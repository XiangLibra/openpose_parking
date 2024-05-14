import os
import xml.etree.ElementTree as ET
import json

# 設定 XML 檔案所在目錄
xml_dir = r'D:\Dataset\parking_data\park_webcam\0331train\Annotation'
json_dir = '../data/annotations'
json_name='parktrain0331'
# 構建輸出 JSON 檔案的完整路徑
json_output_path = os.path.join(json_dir, "parking_keypoints_{}.json".format(json_name))

def xml_to_json(xml_path, json_data):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 取得檔案名稱（不含副檔名）
    base_name = os.path.splitext(os.path.basename(xml_path))[0]

    # 計算 image_id
    current_image_id = len(json_data["images"]) + 1
     # 更新 JSON 對象
    json_data["images"].append({
            "id": current_image_id,
            "width": int(root.find(".//size/width").text),
            "height": int(root.find(".//size/height").text),
            "file_name": f"{base_name}.jpg"
        })

    # 遍歷每個物件
    for obj in root.findall(".//object"):
        # 計算 annotation id
        current_id = len(json_data["annotations"]) + 1

        # 提取bbox的(x0, y0, x1, y1, x2, y2, x3, y3)的值
        x0 = int(obj.find("bndbox/x0").text)
        y0 = int(obj.find("bndbox/y0").text)
        x1 = int(obj.find("bndbox/x1").text)
        y1 = int(obj.find("bndbox/y1").text)
        x2 = int(obj.find("bndbox/x2").text)
        y2 = int(obj.find("bndbox/y2").text)
        x3 = int(obj.find("bndbox/x3").text)
        y3 = int(obj.find("bndbox/y3").text)

        # 計算bbox的中心點和寬度、高度
        center_x = (x0 + x1 + x2 + x3) / 4
        center_y = (y0 + y1 + y2 + y3) / 4
        width = max(x0, x1, x2, x3) - min(x0, x1, x2, x3)
        height = max(y0, y1, y2, y3) - min(y0, y1, y2, y3)

        # 更新 JSON 對象，添加bbox屬性
        json_data["annotations"].append({
            "id": current_id,
            "image_id": current_image_id,
            "category_id": 1,
            "bbox": [center_x, center_y, width, height],
            "segmentation": [
                [x0, y0, x1, y1, x2, y2, x3, y3]
            ],
            "keypoints": [
                x0, y0, 2,
                x1, y1, 2,
                x2, y2, 2,
                x3, y3, 2
            ],
            "num_keypoints": 4,
            "iscrowd": 0
        })

# 初始化 JSON 對象
json_data = {"categories": [{"id": 1, "name": "1", "keypoints": ["0", "1", "2", "3"]}], "images": [], "annotations": []}

# 遍歷 XML 檔案
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, xml_file)
        xml_to_json(xml_path, json_data)



# 將 JSON 寫入檔案
with open(json_output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=2)
