# YOLOv8_detection_streak
เก็บรวบรวมโค้ดของ yolov8n พร้อมคำอธิบาย 

บทความที่วิเคราะห์และรีวิวสถาปัตยกรรมของ YOLOv8 - [A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS](https://arxiv.org/abs/2304.00501?utm_source=chatgpt.com)

Ultralytics YOLOv8 Docs - [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)

[![arXiv](https://img.shields.io/badge/arXiv-2304.00501-red)](https://arxiv.org/abs/2304.00501)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics%20YOLOv8-Docs-1E90FF)](https://docs.ultralytics.com/models/yolov8/)


# ขั้นตอนที่ 1: การติดตั้ง Dependencies

## 1.1. ติดตั้ง Python 3.8 - 3.10 - [Download python](https://www.python.org/downloads/)

## 1.2. ติดตั้ง Dependencies อื่นๆ
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install -U numpy opencv-python tqdm pandas matplotlib seaborn scipy
```

# ขั้นตอนที่ 2: การทำ Labels
โปรแกรมที่ใช้ทำ Labels - [labelme](https://github.com/wkentaro/labelme)

[![Labels - labelme](https://img.shields.io/badge/Labels%20-%20labelme-FFD700)](https://github.com/wkentaro/labelme)

## 2.1. Install

```bash
pip install labelme
pip install pyqt5
```

## 2.2. เรียกใช้งาน

```bash
cd <path โฟลเดอร์ labelImg-master> #Ex. cd C:\Users\SETTHANUN
labelme
```

จะขึ้นหน้านี้

![image](https://github.com/user-attachments/assets/ffe40574-ceda-4870-84bc-d0053f014920)

## 2.3. การใช้งาน

### 2.3.1. กด Open Dir แล้วเลือกโฟลเดอร์ Dataset ที่ต้องการทำ Labels

![image](https://github.com/user-attachments/assets/515f5dc4-2f8e-47ec-acf2-4045bb20c3b0)

### 2.3.2. กด เลือก Create Polygons

![image](https://github.com/user-attachments/assets/65037548-a030-4ea2-a39d-6d1bc6ae0e6b)

### 2.3.3. ลากจุดเชื่อมกันสี่จุดให้เป็นกรอบตรงวัตถุแล้วใส่คำว่า Object ตามด้วยกด OK

![image](https://github.com/user-attachments/assets/3e372b6f-f0eb-4d70-bcb8-8ea974f922c7)

### 2.3.4. กด Save 

![image](https://github.com/user-attachments/assets/311ef943-e7aa-4b3a-b1f5-76ee46d113f0)

### 2.3.5. กดไปรูปถัดไป

![image](https://github.com/user-attachments/assets/4897caf4-8b73-444c-acba-14ef219362b5)


### 2.3.6. ทำแบบนี้จนกว่าจะครบทุกภาพในโฟลเดอร์

# ขั้นตอนที่ 3: Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data=r"C:\Users\SETTHANUN\Desktop\Split\dataset.yaml",  
    epochs=50, 
    imgsz=640,  
    batch=8,  
    device="cuda"  # ใช้ GPU เทรน
)

```
ตัวอย่างไฟล์ dataset.yaml - [dataset.yaml](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/dataset.yaml)

## 3.1. ในกรณีที่เริ่มเทรนใหม่
ตัวอย่าง: model.train(data=r"C:\Users\SETTHANUN\Desktop\Dear\Dataset\dataset.yaml", epochs=5, imgsz=640, project=r"C:\Users\SETTHANUN\Desktop\results", name="train")

```python
model.train(data=r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>", epochs=5, imgsz=640, project=r"<ใส่ path สำหรับเก็บไฟล์โมเดล>", name="<ใส่ชื่อโฟลเดอร์สำหรับเก็บไฟล์โมเดล>")
```

## 3.2. ในกรณีที่เทรนต่อจากโมเดลที่มีอยู่แล้ว
ตัวอย่าง: model.train(data=r"C:\Users\SETTHANUN\Desktop\Dear\Dataset\dataset.yaml", epochs=5, imgsz=640, project=r"C:\Users\SETTHANUN\Desktop\results", name="train", weights=r"C:\Users\SETTHANUN\runs\train\best.pt")

```python
model.train(data=r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>", epochs=5, imgsz=640, project=r"<ใส่ path สำหรับเก็บไฟล์โมเดล>", name="<ใส่ชื่อโฟลเดอร์สำหรับเก็บไฟล์โมเดล>", weights=r"<ใส่ path ที่เก็บไฟล์โมเดลที่เคยเทรนไว้แล้ว.pt>")
```

# ขั้นตอนที่ 4: Test

## 4.1. การ Test ปกติ
```python
import supervision as sv
from ultralytics import YOLO
import numpy as np  
import matplotlib.pyplot as plt
 
data_yaml_path = r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>" #C:\Users\SETTHANUN\Downloads\dataset.yaml
 
annotations_directory_path = r"<ใส่ path ที่มีโฟลเดอร์ mask อยู่>" #C:\Users\SETTHANUN\Desktop\Split\Test\Labels
 
images_directory_path = r"<ใส่ path ที่มีโฟลเดอร์รูปภาพ อยู่>" #C:\Users\SETTHANUN\Desktop\Split\Test\Image
 
with open(data_yaml_path, 'r', encoding='utf-8') as file:
    print(file.read())  
 
dataset = sv.DetectionDataset.from_yolo(
    annotations_directory_path=annotations_directory_path,
    data_yaml_path=data_yaml_path,
    images_directory_path=images_directory_path
)
 
model = YOLO(r"<ใส่ path ที่มีโมเดล.pt>") #C:\Users\SETTHANUN\Desktop\Dear\Model\segment\weight_streak_yolov8_seg.pt
 
def callback(image: np.ndarray) -> sv.Detections:
    result = model.predict(image, save=True)[0]
   
    if result.probs is not None and len(result.probs) > 0:
        print("Detected class names:", result.names)
        print("Detected class indices:", result.probs.argmax(axis=-1))
        predicted_classes = [result.names[int(i)] for i in result.probs.argmax(axis=-1)]
        print("Predicted class names:", predicted_classes)
    else:
        print("No objects detected in the image.")
   
    return sv.Detections.from_ultralytics(result)
 
 
mean_average_precision = sv.MeanAveragePrecision.benchmark(
    dataset=dataset,
    callback=callback
)
 
print("mAP50: ", mean_average_precision.map50)
print("mAP50_95: ", mean_average_precision.map50_95)
```


## 4.2. การ Test และวาดจุด
```python
import cv2
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

input_folder = r"<ใส่ path ที่มีโฟลเดอร์รูปภาพ อยู่>" #E:\41552_processed
output_folder = r"<ใส่ path ที่มีโฟลเดอร์ Output อยู่>" #C:\Users\SETTHANUN\Desktop\Dear\detect_segment\Continuous_frames
output_csv = os.path.join(output_folder, "segmentation_results.csv")
output_image_folder = os.path.join(output_folder, "annotated_images")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

model = YOLO(r"<ใส่ path ที่มีโมเดล.pt>") #C:\Users\SETTHANUN\Desktop\Dear\Model\segment\weight_streak_yolov8_seg.pt

image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".png"))]

results_list = []

for img_path in image_paths:
    image = cv2.imread(img_path)

    if image is None:
        print(f"Error loading image: {img_path}")
        continue

    results = model(img_path, save=True, imgsz=4096)

    if not results or len(results[0].masks) == 0:
        print(f"No detections for: {img_path}")
        results_list.append([os.path.basename(img_path), 0, None, None, None, None, None, None, None])
        continue

    for obj_id, (mask, box, conf) in enumerate(zip(results[0].masks.data, results[0].boxes.data, results[0].boxes.conf), start=1):
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        all_contours = np.vstack(contours)
        x1, y1, w, h = cv2.boundingRect(all_contours)
        x2, y2 = x1 + w, y1 + h

        M = cv2.moments(all_contours)
        x0 = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x1 + w // 2
        y0 = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y1 + h // 2

        conf = float(conf.cpu().numpy())

        results_list.append([os.path.basename(img_path), obj_id, x1, y1, x2, y2, x0, y0, conf])

df = pd.DataFrame(results_list, columns=["File name", "Object ID", "x1", "y1", "x2", "y2", "x0", "y0", "Confidence"])
df.to_csv(output_csv, index=False)
print(f"Results saved at: {output_csv}")

for img_path in image_paths:
    image = cv2.imread(img_path)
    if image is None:
        continue

    file_name = os.path.basename(img_path)
    df_image = df[df['File name'] == file_name]

    for _, row in df_image.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        x0, y0 = int(row['x0']), int(row['y0'])
        conf = row['Confidence']

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(image, (x0, y0), 5, (255, 0, 0), -1)
        cv2.putText(image, f"Centroid: ({x0}, {y0})", (x0 + 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        corners = [(x1, y1), (x2, y2), (x1, y2), (x2, y1)]
        for (cx, cy) in corners:
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(image, f"({cx}, {cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_img_path = os.path.join(output_image_folder, file_name)
    cv2.imwrite(output_img_path, image)
    print(f"Saved annotated image: {output_img_path}")

print("Final segmentation results:")
print(df.head(20))
print(f"Annotated images saved at: {output_image_folder}")
```

# ขั้นตอนที่ 5: Result
## 5.1. การ Segmet ธรรมดา
ภาพนี้แสดงผลลัพธ์ของการแบ่งส่วนภาพ (segmentation) ซึ่งมีการเน้นวัตถุในภาพโดยใช้สีเหลืองสดเพื่อล้อมรอบบริเวณที่ตรวจจับได้ วัตถุที่ถูกแบ่งส่วนนั้นอยู่ในอากาศเหนือแม่น้ำและมีรูปร่างที่ซับซ้อน โดยพื้นหลังของภาพประกอบไปด้วยท้องฟ้า เมฆ แม่น้ำ และสิ่งปลูกสร้าง ซึ่งช่วยให้สามารถแยกวัตถุออกจากบริบทของภาพได้อย่างชัดเจน เทคนิคที่ใช้ในการแบ่งส่วนภาพนี้สามารถช่วยในการระบุและแยกวัตถุจากพื้นหลังเพื่อการวิเคราะห์และประมวลผลเพิ่มเติม เช่น การตรวจจับและติดตามวัตถุในระบบคอมพิวเตอร์วิทัศน์ หรือการนำไปใช้ในแอปพลิเคชันที่เกี่ยวข้องกับยานพาหนะทางอากาศและภูมิศาสตร์
![image](https://github.com/user-attachments/assets/806b0911-3748-4a65-b569-aad11f0f0b8e)

## 5.2. วาดจุด x,y,center
ภาพนี้แสดงผลลัพธ์ของการระบุตำแหน่งและขอบเขตของวัตถุในภาพ โดยใช้จุดพิกัดที่สำคัญ ได้แก่ x1,y1,x2,y2 ซึ่งแทนตำแหน่งมุมซ้ายบนและมุมขวาล่างของกรอบสี่เหลี่ยมที่ล้อมรอบวัตถุ (bounding box) และ x,y center ซึ่งเป็นจุดศูนย์กลางของกรอบสี่เหลี่ยม คำนวณจากค่าเฉลี่ยของพิกัดมุมทั้งสอง การแสดงผลนี้ช่วยในการวิเคราะห์ตำแหน่งของวัตถุในภาพ ซึ่งสามารถนำไปใช้ในงานด้านการตรวจจับวัตถุ (object detection) การติดตามวัตถุ (object tracking) หรือการวิเคราะห์ภาพอื่น ๆ ได้
![2024_11_25_15_40_06_exp_10_000000_THAICOM6_processed](https://github.com/user-attachments/assets/dad573d0-44cf-4fcd-a847-f621d3f260af)

## 5.3. mAP 
mAP (mean Average Precision) เป็นค่าที่ใช้ในการประเมินประสิทธิภาพของโมเดลในงานที่เกี่ยวข้องกับการตรวจจับ (object detection) และการแยกส่วน (segmentation) ซึ่ง mAP จะใช้ในการวัดความแม่นยำของการทำนายของโมเดลที่ทำการตรวจจับวัตถุหรือแยกส่วนวัตถุในภาพ โดยทั่วไปแล้ว mAP ถูกคำนวณจากค่า Average Precision (AP) ที่ได้จากการประเมินผลในแต่ละคลาสหรือประเภทของวัตถุ ซึ่ง AP จะเป็นการคำนวณความแม่นยำของโมเดลในแต่ละกรณี เช่น คะแนน Precision-Recall (PR Curve) และคำนวณค่าเฉลี่ยจากแต่ละคลาสที่มีการทำนาย
![image](https://github.com/user-attachments/assets/3a0ed6ff-d9a6-4dcb-b61e-88deed747ad4)


# ขั้นตอนที่ 6: เมื่อเกิดเหตุขัดข้อง
## 6.1. กรณีดาวน์โหลด Ultralytics ไม่ได้

โฟลเดอร์ใน Google drive - [Ultralytics drive](https://drive.google.com/file/d/1JaNYy7bcdA9FnZMclFmockTiUT2IHGE7/view?usp=sharing)

[![DRIVE - Ultralytics](https://img.shields.io/badge/DRIVE-Ultralytics-006400)](https://drive.google.com/file/d/1JaNYy7bcdA9FnZMclFmockTiUT2IHGE7/view?usp=sharing)

## 6.2. กรณีดาวน์โหลด labelImg ไม่ได้

โฟลเดอร์ใน Google drive - [labelImg drive](https://drive.google.com/file/d/1sQ2g4o0fdcOSwqGdM01ZhoKLkwvsYdpV/view?usp=sharing)

[![DRIVE - labelme](https://img.shields.io/badge/DRIVE-labelImg-32CD32)](https://drive.google.com/file/d/1sQ2g4o0fdcOSwqGdM01ZhoKLkwvsYdpV/view?usp=sharing)

# เพิ่มเติม

ไฟล์ Jupyter notebook ที่เป็นโค้ดสำเร็จรูปแล้ว - [Fit yolo](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/Fit_yolo.ipynb)

ไฟล์โมเดลที่เทรนแล้ว - [Model](https://drive.google.com/file/d/1veqK1fydOkwu1toAbM0Fy2QSQefl48v0/view?usp=sharing)

ข้อมูลสำหรับการเทส - [Test](https://drive.google.com/file/d/1E1ZifJ56DEVDnBdfYzICRzc7HSHdTqY4/view?usp=sharing)

[![Fit yolo](https://img.shields.io/badge/Fit%20yolo-YOLOv8-90EE90)](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/Fit_yolo.ipynb) [![DRIVE - Model](https://img.shields.io/badge/DRIVE-Model-59ed17)](https://drive.google.com/file/d/1veqK1fydOkwu1toAbM0Fy2QSQefl48v0/view?usp=sharing)


