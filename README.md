# Drone Segmentation on Cloud

## Step 1: Data Preparation

### 1.1. Download Dataset 
This dataset consists of high-resolution images capturing drones in various environments and conditions. The images are specifically collected for tasks such as object detection, tracking, and classification, enabling the development and evaluation of computer vision models for drone-related applications.

[![Dataset](https://img.shields.io/badge/Dataset-Drone-red)](https://drive.google.com/file/d/1EYZkrOq_FYzLHuo12X-vbcBQhEuoWPFc/view?usp=sharing)
### 1.2. Data Splitting
To provide flexibility in managing your dataset, you are encouraged to manually divide the data into three distinct sets: Training, Testing, and Validation. Please follow the steps below to properly organize your data
- Training Set: This set is used to train the model. It should contain the majority of the data and be representative of the overall dataset.
- Testing Set: The testing set is reserved for evaluating the model's performance after training. It should be kept separate from the training data to ensure a fair assessment of the model’s generalization ability.
- Validation Set: The validation set is used during the training process to tune hyperparameters and monitor the model’s performance. It helps in adjusting model parameters to avoid overfitting.

#### 1.2.1. Recommended Steps
- Randomization: Shuffle the data before splitting to ensure that the subsets are representative and free from any ordering bias.
- Manual Division: Allocate the appropriate number of samples to each of the three subsets, ensuring that each set is balanced and reflects the overall distribution of the data.

#### 1.2.2. Code for Data Splitting
The provided code is intended for data splitting tasks and is to be executed within the Command Prompt (CMD) interface.

- In order to run the code, please ensure that Python is called in the Command Prompt by executing the following command:
```bash
python
```

- Once Python is invoked, use the following code to perform the data splitting:
```bash
import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_size, val_size, test_size):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found.")
        return
    files = []
    for root, dirs, file_names in os.walk(source_dir):
        for file_name in file_names:
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                files.append(os.path.join(root, file_name))
    if not files:
        print(f"No image files found in the source directory.")
        return
    random.shuffle(files)
    total_files = len(files)
    train_count = int(total_files * train_size)
    val_count = int(total_files * val_size)
    test_count = total_files - train_count - val_count
    print(f'Total files: {total_files}, Train: {train_count}, Validation: {val_count}, Test: {test_count}')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for i, file_path in enumerate(files):
        file_name = os.path.basename(file_path)
        if i < train_count:
            target_dir = train_dir
        elif i < train_count + val_count:
            target_dir = val_dir
        else:
            target_dir = test_dir
        target_path = os.path.join(target_dir, file_name)
        shutil.copy(file_path, target_path)
        print(f"Copied file {file_name} to {target_path}")

source_dir = 'path/to/your/dataset'
train_dir = 'path/to/save/train'
val_dir = 'path/to/save/val'
test_dir = 'path/to/save/test'

train_size =  #Specify the number you want to split for train
val_size =  #Specify the number you want to split for validation
test_size =  #Specify the number you want to split for test

split_dataset(source_dir, train_dir, val_dir, test_dir, train_size, val_size, test_size)
```

- Example of how to specify a path: 

```bash
source_dir = r'C:\Users\SETTHANUN\Downloads\archive\Database1\Dataset'
train_dir = r'C:\Users\SETTHANUN\Desktop\Split\Train\Image'
val_dir = r'C:\Users\SETTHANUN\Desktop\Split\Validation\Image'
test_dir = r'C:\Users\SETTHANUN\Desktop\Split\Test\Image'
```

### 1.3. mask
For the annotation process, we will use Roboflow, a powerful tool designed for labeling and annotating images efficiently. Roboflow allows us to create and manage labeled datasets for various machine learning tasks, including image classification, object detection, and segmentation. By leveraging its intuitive interface, we can easily annotate our images and export them in the required formats for model training and evaluation.

#### 1.3.1. Go to the Roboflow website.
[![Roboflow](https://img.shields.io/badge/roboflow-labels-purple)](https://app.roboflow.com/)

#### 1.3.2. On the Roboflow page, select Create my own workspace.
#### 1.3.3. Enter the desired workspace name in the 'Name Your Workspace' field.
#### 1.3.4. Choose Public Plan.
#### 1.3.5. Click Continue.

![image](https://github.com/user-attachments/assets/0b090fce-2aee-4874-a74c-7bc1990a5edc)

#### 1.3.6. On the Invite teammates page, you can add other users to collaborate on labeling.
#### 1.3.7. Click Create Workspace.

![image](https://github.com/user-attachments/assets/01cda051-d18a-4ff7-a3b0-6c144a10f087)

#### 1.3.8. Go to the Projects page and click New Project.
![image](https://github.com/user-attachments/assets/0fb03c02-4935-42be-aff3-083d5aae129a)

#### 1.3.9. Enter the Project Name and Annotation Group.
#### 1.3.10. Select Object Detection, then click Create Public Project.

![image](https://github.com/user-attachments/assets/5e4232b5-04f1-470b-8000-6d389755549c)

#### 1.3.11. On the Upload page, select 'Select Folder,' then choose the folder containing the images to be labeled.

![image](https://github.com/user-attachments/assets/b1bed0c5-7abc-4f49-984f-7e0bdc89aef9)

#### 1.3.12. Enter the Batch Name and then click Save and Continue

![image](https://github.com/user-attachments/assets/a9e840de-b077-454f-b167-59cf991cfdfb)

#### 1.3.13. Click Start Manual Labeling. 
![image](https://github.com/user-attachments/assets/e8752296-f373-4d92-91c6-53bcb1bbae30)

#### 1.3.14. You can add other users and then click 'Assign to Myself'.

![image](https://github.com/user-attachments/assets/faa46488-2913-4db5-ae02-6c2952b62660)

#### 1.3.15. Select Start Annotating

![image](https://github.com/user-attachments/assets/db8241e6-a443-459f-bbd3-ffa435e46ccf)

#### 1.3.16. Select the Bounding Box Tool, then perform the annotation by dragging a rectangle to surround the drone.
#### 1.3.17. There will be a field to enter the class name, type 'Drone', and then click Save. Repeat this process until all images are done. **The class name must be the same for all images.

![image](https://github.com/user-attachments/assets/cb10a487-069f-4fe6-a457-5cf21c5e65ea)

#### !!! Don't forget to specify whether it's training, validation, or test data.
![image](https://github.com/user-attachments/assets/2513afa2-becd-4500-8789-f8d0364026e9)

#### 1.3.18. After annotating, click + New Version.

![image](https://github.com/user-attachments/assets/7d552e4e-8292-43d0-b073-90397edb01a2)

#### 1.3.19. Check the accuracy of the data, such as the total number of images, the number of classes, and the number of images in each split.

![image](https://github.com/user-attachments/assets/2bb176b6-a349-4f28-ba33-dbaf3bb2726e)

#### 1.3.20. Preprocessing: Remove Auto-Orient and Resize, then click Continue.
![image](https://github.com/user-attachments/assets/eb651c41-b553-4035-8186-d009cbed2092)

#### 1.3.21. Augmentation: Don't select anything, click Continue.

![image](https://github.com/user-attachments/assets/31fb6faa-9a03-4d85-a300-b5314c76dc3d)

#### 1.3.22. Create: Click Create.

![image](https://github.com/user-attachments/assets/b118f76f-07d9-44b9-a0d9-ed5b7a1f0d97)

#### 1.3.23. On the Dataset Versions page, select Download Dataset.

![image](https://github.com/user-attachments/assets/34cbcb65-173b-49b0-a1d3-2270d1b4e1e4)

#### 1.3.24 On the Download page, select Download zip to computer. For Format, choose TXT and select the YOLOv8 model name, then click Continue.

![image](https://github.com/user-attachments/assets/c4c48a03-44e5-40fd-b3a4-7c5023c0d13a)

### 1.4. Upload data to the cloud in Azure AI | Machine Learning Studio

#### 1.4.1. Go to the Azure AI | Machine Learning Studio
[![Azure](https://img.shields.io/badge/Azure-ML-blue)](https://ml.azure.com/)

#### 1.4.2. Create a new folder by merging the images and their corresponding annotation files (.txt) into the same folder. Do this for Train, Test, and Validation datasets.
![image](https://github.com/user-attachments/assets/4a86b6b7-b0f7-4af2-8573-2b0e8c38944e)

#### 1.4.3. In the Data page, click Create.
![image](https://github.com/user-attachments/assets/3c850fe9-99a0-444f-9a34-6adad7654692)

#### 1.4.4. In the Create data asset section, enter the dataset name and select the type as Folder, then click Next.

![image](https://github.com/user-attachments/assets/37166773-c349-4a39-8d30-7c4629b17efc)

#### 1.4.5. In the Create data asset tab, under Data source, select From local file, then click Next.

![image](https://github.com/user-attachments/assets/93b0ac3d-e4cf-4a1a-a4b7-890661189ca7)

#### 1.4.6. In the Create data asset tab, under Destination storage type, select workspaceblobstore, then click Next.

![image](https://github.com/user-attachments/assets/cb594e08-45ee-4c1c-bb38-dc2d99624ed6)

#### 1.4.7. In the Create data asset tab, under Folder selection, click Upload file or folder, then select Upload folder. Next, upload the prepared dataset folders for train, validation, and test, which include both images and annotation .txt files. Then, click Next.

![image](https://github.com/user-attachments/assets/734a4779-b486-41aa-a015-16537c54dcd4)

#### 1.4.8. On the Review page, click Create.
Result

![image](https://github.com/user-attachments/assets/ac851b53-64e2-460f-b889-14fe79c2f252)

## Step 2: Training and Testing YOLOv8 Model on the Cloud
### 2.1. Go to the Azure AI | Machine Learning Studio
[![Azure](https://img.shields.io/badge/Azure-ML-blue)](https://ml.azure.com/)

### 2.2. Install Dependencies
```bash
pip install ultralytics azureml-sdk roboflow supervision
```

### 2.3. On the left menu bar, scroll up to the top and click on "Notebooks." In the Notebooks page, click on "+ Files," then select "Create new folder" and name the folder before clicking "Create."
![image](https://github.com/user-attachments/assets/3c36d4b3-2282-4989-b944-ec0bb9f52c1a)
![image](https://github.com/user-attachments/assets/868ce472-fffb-4207-8202-26662fb79a6a)

### 2.4. Go to the folder you created, click on the "..." button at the back, then select "Create new file." Enter the file name and choose "File type" as Notebook (*.ipynb), then click "Create."
![image](https://github.com/user-attachments/assets/65092cc9-7555-4d82-84e3-08609156e490)

![image](https://github.com/user-attachments/assets/4ca13ab8-b0e2-46a7-8818-0b467c74128c)

### 2.5. In the Notebook.ipynb page that was created, under Compute, select the created Compute and click Run (if it's green, no need to click) until it changes from a black circle to green.
![image](https://github.com/user-attachments/assets/361eb4e0-bad5-4cd3-a0b0-ab411d42b052)

### 2.6. To load and mount a dataset from Azure Machine Learning (Azure ML) to a local directory using the Azure ML SDK

```python
from azureml.core import Workspace, Dataset
import os

ws = Workspace(
    subscription_id="", 
    resource_group="", 
    workspace_name=""
) #'Workspace' changes to the 'Workspace' data of the user's account.

dataset = Dataset.get_by_name(ws, name="", version="1") #Change 'name' to the user's own dataset name.

local_path = "./datasets/DroneDataset" #Change from 'DroneDataset' to the user's own dataset name.
os.makedirs(local_path, exist_ok=True)

mount_context = dataset.mount(local_path)
mount_context.start()

train_data = os.path.join(local_path, "Train_data")
val_data = os.path.join(local_path, "Validation_data")
test_data = os.path.join(local_path, "Test_data")
#Change from 'Train_data', 'Validation_data', 'Test_data' to the folder names Train, Validation, Test that the user has uploaded in their own Dataset.

print("Mounted dataset at:", local_path)
print("Train data:", train_data)
print("Validation data:", val_data)
print("Test data:", test_data)

```
- How to view Workspace: User can find the 'workspace_name' from the 'Current workspace' mentioned in parentheses.

![image](https://github.com/user-attachments/assets/519bcfc8-41e0-4ce1-a924-d00431f6431c)

- Example of how to specify a Workspace: 

```python
ws = Workspace(
    subscription_id="fcbf128f-aa80-4469-99a5-706b320e4401",
    resource_group="ml-resource-group", 
    workspace_name="ml-workspace"
)
```

### 2.7. Create dataset.yaml
- Go to the folder you created, click on the "..." button at the back, then select "Create new file." Enter the file name and choose "File type" as Notebook (*.ipynb), then click "Create."
![image](https://github.com/user-attachments/assets/65092cc9-7555-4d82-84e3-08609156e490)

- To create a .yaml file

![image](https://github.com/user-attachments/assets/0b3e9f4e-dd7c-4e8a-9192-eb42aaef0294)

- Insert the code and save it.
```bash
train: /mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Train_data
val: /mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Validation_data
test: /mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Test_data

nc: 1  # จำนวน classes
names: ['drone']  # classes name
```
- How to insert a path into a .yaml file: 
```bash
train: /mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Train_data
val: /mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Validation_data
test: /mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Test_data
```
Do not make changes to the section of /mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/. In the Users/kewalee.sr/Object_detection/datasets/DroneDataset, choose 'Copy folder path' from the Datasets folder you created and replace it (with DroneDataset being the user's dataset name). In the Train_data section, replace it with the name of the folder you set for Train, Validation, and Test.


### 2.8. Train YOLOv8 model
#### 2.8.1.In the case of starting training from scratch
```python
import time

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

start_time = time.time()

model.train(data='./dataset.yaml', epochs=, imgsz=640) #Set the epochs yourself.

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")
```

#### 2.8.2.In the case of fine-tuning from an already existing model
```python
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.ptt")  

start_time = time.time()

model.train(data='./dataset.yaml', epochs=, imgsz=640) #Set the epochs yourself.

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")
```

### 2.9. Test YOLOv8 model
#### 2.9.1. Testing to see the results.
```python
import numpy as np  
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("./runs/detect/train/weights/best.pt")  

results = model.predict(source='./datasets/DroneDataset/Test_data', save=True) #Change DroneDataset to the user's own dataset name and change Test_data to the uploaded test folder.
```

#### 2.9.2. Testing to determine the mAP value.

```python
import supervision as sv
from ultralytics import YOLO
import numpy as np  
import matplotlib.pyplot as plt

data_yaml_path = r"./dataset.yaml"

annotations_directory_path = r"/mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Test_data"
#Enter the path of the test dataset in the data.yaml file.

images_directory_path = r"/mnt/batch/tasks/shared/LS_root/mounts/clusters/compute-target/code/Users/kewalee.sr/Object_detection/datasets/DroneDataset/Test_data"
#Enter the path of the test dataset in the data.yaml file.

with open(data_yaml_path, 'r', encoding='utf-8') as file:
    print(file.read())  

dataset = sv.DetectionDataset.from_yolo(
    annotations_directory_path=annotations_directory_path,
    data_yaml_path=data_yaml_path,
    images_directory_path=images_directory_path
)

model = YOLO("./runs/detect/train/weights/best.pt")  

def callback(image: np.ndarray) -> sv.Detections:
    result = model.predict(source=images_directory_path, save=True)[0]
    
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


## 3. Result

### 3.1. Predicted Image
This image appears to be the result of testing a trained YOLOv8 model on a drone detection dataset. The image showcases an aerial drone, which has been detected and annotated with a bounding box labeled "drone" and a confidence score. The bounding box and label indicate that the object detection model has identified the drone with moderate confidence. The presence of the bounding box suggests that the YOLOv8 model has been trained on a dataset containing drone images, enabling it to recognize and classify drones in real-world scenarios.

![image](https://github.com/user-attachments/assets/8be0533b-b557-457e-8f05-29a967a4df42)

![image](https://github.com/user-attachments/assets/6d74b5af-2a64-42b2-be84-59f2eab9fd22)


### 3.1. mAP Value Obtained from Prediction
This image presents the mean Average Precision (mAP) results obtained from testing a YOLOv8 model trained on a drone detection dataset. The results indicate the model's performance metrics, including inference speed, preprocessing time, and postprocessing time for each image of shape (1, 3, 384, 640). The key evaluation metrics include mAP@50, which is reported as 0.9341, and mAP@50-95, which is recorded as 0.5546. These values represent the model’s detection accuracy at different Intersection over Union (IoU) thresholds, demonstrating its effectiveness in detecting drones within the dataset.

![image](https://github.com/user-attachments/assets/f6526415-fad0-48c8-a8f6-8512a0ca868a)







