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
 
data_yaml_path = r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>"
 
annotations_directory_path = r"<ใส่ path ที่มีโฟลเดอร์ mask อยู่>"
 
images_directory_path = r"<ใส่ path ที่มีโฟลเดอร์รูปภาพ อยู่>"
 
with open(data_yaml_path, 'r', encoding='utf-8') as file:
    print(file.read())  
 
dataset = sv.DetectionDataset.from_yolo(
    annotations_directory_path=annotations_directory_path,
    data_yaml_path=data_yaml_path,
    images_directory_path=images_directory_path
)
 
model = YOLO(r"<ใส่ path ที่มีโมเดล.pt>")  
 
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

## 4.1. การ Test ปกติ
```python
import supervision as sv
from ultralytics import YOLO
import numpy as np  
import matplotlib.pyplot as plt
 
data_yaml_path = r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>"
 
annotations_directory_path = r"<ใส่ path ที่มีโฟลเดอร์ mask อยู่>"
 
images_directory_path = r"<ใส่ path ที่มีโฟลเดอร์รูปภาพ อยู่>"
 
with open(data_yaml_path, 'r', encoding='utf-8') as file:
    print(file.read())  
 
dataset = sv.DetectionDataset.from_yolo(
    annotations_directory_path=annotations_directory_path,
    data_yaml_path=data_yaml_path,
    images_directory_path=images_directory_path
)
 
model = YOLO(r"<ใส่ path ที่มีโมเดล.pt>")  
 
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

# ขั้นตอนที่ 5: Result
## 5.1. การ Segmet ธรรมดา
![image](https://github.com/user-attachments/assets/806b0911-3748-4a65-b569-aad11f0f0b8e)

## 5.2. วาดจุด x,y,center

![2024_11_25_15_40_06_exp_10_000000_THAICOM6_processed](https://github.com/user-attachments/assets/dad573d0-44cf-4fcd-a847-f621d3f260af)

## 5.3. mAP 

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


