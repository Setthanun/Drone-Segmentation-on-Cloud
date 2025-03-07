# YOLOv8_segmentation_streak
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

input_folder = r"E:\41552_processed2"
output_folder = r"D:\result9"
output_csv = os.path.join(output_folder, "segmentation_results.csv")
output_image_folder = os.path.join(output_folder, "annotated_images")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

model = YOLO(r"C:\Users\SETTHANUN\Desktop\Dear\Model\segment\weight_streak_yolov8_seg.pt")

image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".png"))]

results_list = []

for img_path in image_paths:
    image = cv2.imread(img_path)

    if image is None:
        print(f"Error loading image: {img_path}")
        continue

    results = model(img_path, save=False, imgsz=4096)

    if not results or len(results[0].masks) == 0:
        print(f"No detections for: {img_path}")
        results_list.append([os.path.basename(img_path), 0, None, None, None, None, None, None, None])
        continue

    for obj_id, (mask, box, conf) in enumerate(zip(results[0].masks.data, results[0].boxes.data, results[0].boxes.conf), start=1):
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(f"Object {obj_id} Mask Contours: {contours}")
        
        #mask_size = mask_np.shape  # (height, width)
        #print(f"Object {obj_id} Mask Size: {mask_size}")
        
        #cv2.imshow(f"Mask {obj_id}", mask_np)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if contours:
            all_contours = np.vstack(contours)
            x1, y1, w, h = cv2.boundingRect(all_contours)
            x2, y2 = x1 + w, y1 + h

            M = cv2.moments(mask_np)
            x0 = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x1 + w // 2
            y0 = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y1 + h // 2

            conf = float(conf.cpu().numpy())

            results_list.append([os.path.basename(img_path), obj_id, x1, y1, x2, y2, x0, y0, conf])

    df = pd.DataFrame(results_list, columns=["File name", "Object ID", "x1", "y1", "x2", "y2", "x0", "y0", "Confidence"])
    df.to_csv(output_csv, index=False)
    print(f"Results saved at: {output_csv}")

    color_mask = np.zeros_like(image, dtype=np.uint8)
    for obj_id, mask in enumerate(results[0].masks.data, start=1):
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(image, [c], -1, (0, 0, 255), thickness=cv2.FILLED)  # สีแดงเข้ม

    segmented_image = cv2.addWeighted(image, 1, color_mask, 0.7, 0)

    df_image = df[df['File name'] == os.path.basename(img_path)]
    for _, row in df_image.iterrows():
        obj_id = int(row['Object ID'])
        x0, y0 = int(row['x0']), int(row['y0'])
        x1, y1 = int(row['x1']), int(row['y1'])

        cv2.circle(segmented_image, (x0, y0), 5, (255, 50, 50), -1)
        cv2.putText(segmented_image, f"Obj {obj_id} (Conf {row['Confidence']:.2f})", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(segmented_image, f"Centroid: ({x0}, {y0})", (x0 + 10, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)  

    output_segmented_path = os.path.join(output_image_folder, f"segmented_{os.path.basename(img_path)}")
    cv2.imwrite(output_segmented_path, segmented_image)
    print(f"Saved segmented image: {output_segmented_path}")

print("Final segmentation results:")
print(df.head(20))
print(f"Annotated images saved at: {output_image_folder}")
)
```

# ขั้นตอนที่ 5: Result
## 5.1. ผลลัพธ์จากการ Segmet แบบธรรมดา

เป็นผลลัพธ์จากการทดสอบ Image Segmentation ซึ่งเป็นกระบวนการแบ่งส่วนภาพออกเป็นกลุ่มของพิกเซลที่มีความหมาย โดยในที่นี้ใช้สำหรับการตรวจจับขอบเขตและพิกัดในภาพท้องฟ้าที่มีดวงดาว ระบบสามารถระบุพื้นที่ของวัตถุที่ตรวจจับได้ และทำการไฮไลต์บริเวณที่เป็นวัตถุเป้าหมาย การใช้กรอบสีน้ำเงิน (bounding box) ทำหน้าที่แสดงขอบเขตของวัตถุที่ถูกตรวจพบ อย่างไรก็ตาม ในการแบ่งส่วนจริงๆ ระบบจะพิจารณาพื้นที่ของวัตถุที่ตรวจจับได้ โดยไม่จำกัดอยู่แค่กรอบสี่เหลี่ยม แต่จะเป็นการแบ่งพื้นที่ตามลักษณะของวัตถุจริง

ตัวเลข 0.85 และ 0.89 ที่ปรากฏในภาพ หมายถึงค่าความมั่นใจของโมเดลในการแบ่งส่วนของวัตถุแต่ละรายการ ค่าที่สูงกว่า 0.8 แสดงให้เห็นว่าโมเดลมีความมั่นใจสูงว่าพื้นที่ที่ถูกตรวจจับนั้นเป็นวัตถุที่ต้องการ

![2024_11_26_13_24_02_exp_5_000000_THAICOM8_processed](https://github.com/user-attachments/assets/d83e4610-f91a-4957-b7a4-f81ec1f4ec02)

![image](https://github.com/user-attachments/assets/3aad0334-d25d-4c37-92c6-0fd31a0aefc3)


## 5.2. วาดจุด x,y,center

ผลลัพธ์จากการทดสอบการแบ่งส่วน (Segmentation) ของวัตถุในภาพถ่ายดาราศาสตร์ วัตถุแต่ละชิ้นที่ถูกตรวจจับจะถูกกำหนดหมายเลข เช่น Obj 1 และ Obj 2 ซึ่งหมายถึงลำดับของวัตถุที่โมเดลสามารถตรวจพบได้ในภาพ, สำหรับแต่ละวัตถุ จะมีค่า Confidence (Conf) กำกับอยู่ เพื่อแสดงระดับความมั่นใจของโมเดลที่มีต่อการตรวจจับวัตถุนั้น ค่า Conf นี้เป็นตัวเลขที่อยู่ระหว่าง 0 ถึง 1 โดยค่าใกล้ 1 หมายถึงความมั่นใจสูง, จุดสีน้ำเงิน ในภาพแสดงถึง Centroid หรือจุดศูนย์กลางของวัตถุที่ถูกตรวจจับ พิกัดของ Centroid จะแสดงเป็นข้อความสีน้ำเงินข้างวัตถุนั้น จุดศูนย์กลางนี้เป็นค่ากลางของตำแหน่งพิกเซลทั้งหมดที่เป็นส่วนหนึ่งของวัตถุ โดยจะแสดงเป็นพิกัด x,y และ เส้นสีแดง เป็น Segment Mask ซึ่งแสดงรูปร่างของวัตถุที่ถูกตรวจจับในภาพ เส้นนี้แสดงให้เห็นว่า วัตถุที่ตรวจพบนั้นไม่ได้เป็นจุดเดี่ยว แต่มีลักษณะเป็นเส้นหรือมีการเคลื่อนที่ในช่วงเวลาที่บันทึกภาพ

![segmented_2024_11_26_13_24_02_exp_5_000000_THAICOM8_processed](https://github.com/user-attachments/assets/4917a641-7720-4516-ae89-f51b60952271)

![image](https://github.com/user-attachments/assets/c09dfbf3-09dd-4033-978c-b3c63af74c79)


## 5.3. mAP 
mAP (mean Average Precision) เป็นค่าที่ใช้ในการประเมินประสิทธิภาพของโมเดลในงานที่เกี่ยวข้องกับการตรวจจับ (object detection) และการแยกส่วน (segmentation) ซึ่ง mAP จะใช้ในการวัดความแม่นยำของการทำนายของโมเดลที่ทำการตรวจจับวัตถุหรือแยกส่วนวัตถุในภาพ โดยทั่วไปแล้ว mAP ถูกคำนวณจากค่า Average Precision (AP) ที่ได้จากการประเมินผลในแต่ละคลาสหรือประเภทของวัตถุ ซึ่ง AP จะเป็นการคำนวณความแม่นยำของโมเดลในแต่ละกรณี เช่น คะแนน Precision-Recall (PR Curve) และคำนวณค่าเฉลี่ยจากแต่ละคลาสที่มีการทำนาย

![image](https://github.com/user-attachments/assets/3a0ed6ff-d9a6-4dcb-b61e-88deed747ad4)


# ขั้นตอนที่ 6: เมื่อเกิดเหตุขัดข้อง
## 6.1. กรณีดาวน์โหลด Ultralytics ไม่ได้

โฟลเดอร์ใน Google drive - [Ultralytics drive](https://drive.google.com/file/d/1JaNYy7bcdA9FnZMclFmockTiUT2IHGE7/view?usp=sharing)

[![DRIVE - Ultralytics](https://img.shields.io/badge/DRIVE-Ultralytics-006400)](https://drive.google.com/file/d/1JaNYy7bcdA9FnZMclFmockTiUT2IHGE7/view?usp=sharing)

## 6.2. กรณีดาวน์โหลด labelme ไม่ได้

โฟลเดอร์ใน Google drive - [labelImg drive](https://drive.google.com/file/d/1sQ2g4o0fdcOSwqGdM01ZhoKLkwvsYdpV/view?usp=sharing)

[![DRIVE - labelme](https://img.shields.io/badge/DRIVE-labelImg-32CD32)](https://drive.google.com/file/d/1sQ2g4o0fdcOSwqGdM01ZhoKLkwvsYdpV/view?usp=sharing)

# เพิ่มเติม

ไฟล์ Jupyter notebook ที่เป็นโค้ดสำเร็จรูปแล้ว - [Fit yolo](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/Fit_yolo.ipynb)

ไฟล์โมเดลที่เทรนแล้ว - [Model](https://drive.google.com/file/d/1veqK1fydOkwu1toAbM0Fy2QSQefl48v0/view?usp=sharing)

ข้อมูลสำหรับการเทส - [Test](https://drive.google.com/file/d/1E1ZifJ56DEVDnBdfYzICRzc7HSHdTqY4/view?usp=sharing)

[![Fit yolo](https://img.shields.io/badge/Fit%20yolo-YOLOv8-90EE90)](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/Fit_yolo.ipynb) [![DRIVE - Model](https://img.shields.io/badge/DRIVE-Model-59ed17)](https://drive.google.com/file/d/1veqK1fydOkwu1toAbM0Fy2QSQefl48v0/view?usp=sharing)


