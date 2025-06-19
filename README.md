# Sunspot Detection and Classification

## 1. Introduction

This project explores a hybrid approach to sunspot analysis, combining the power of the YOLOv8 object detection model with ResNet classification, and information extracted from Solar Region Summary (SRS) reports. The primary goal is to automatically identify, locate, and classify sunspots in solar images, leveraging both image data and structured report information. The ultimate aim is to predict sunspot characteristics detailed in NOAA SRS reports.

## 2. Proposed Solution

The project aims to develop a system capable of:

1.  **Sunspot Detection:** Utilizing YOLOv8 to accurately locate sunspots within solar images.
2.  **Sunspot Classification:** Employing a ResNet model to classify detected sunspot regions based on characteristics described in SRS reports (e.g., McIntosh classification).
3.  **Automated Dataset Generation:** Creating a pipeline to generate training data by linking SRS reports with corresponding solar images.

## 3. Methodology

### 3.1. Data Acquisition

*   **Solar Images:** The primary source of image data is solar images, ideally from the Solar Dynamics Observatory (SDO).  These images are downloaded and converted to JPG format.
*   **Solar Region Summaries (SRS):** SRS reports provide structured data about active regions on the sun, including location, classification, and magnetic field characteristics. These reports are crucial for generating training annotations and classification labels. SRS files for 2014 are stored in the `SRS/2014_SRS` directory.

### 3.2. Data Preprocessing

Preprocessing steps are essential to enhance image quality and prepare the data for model training. This may include:

*   **Image Resizing:** Resizing images to a consistent size.
*   **Grayscale Conversion:** Converting color images to grayscale.
*   **Data Augmentation:** Applying transformations such as flips, rotations, and scaling to increase the diversity of the training data.

### 3.3. Model Training

The training process involves two main stages: sunspot detection and sunspot classification.

*   **YOLOv8 (Detection):** A pre-trained YOLOv8 model (`yolov8n.pt`) is used as a starting point for sunspot detection. The model is fine-tuned to accurately locate sunspots within solar images.
*   **ResNet (Classification):** A ResNet model is intended to classify the sunspot regions detected by YOLOv8. The classification targets will be based on the information available in the SRS reports (e.g., McIntosh classification).

The training process is managed within the [notebook.ipynb](notebook.ipynb) Jupyter Notebook.

### 3.4. Automated Dataset Generation (In Progress)

The [srs_notebook.ipynb](srs_notebook.ipynb) notebook outlines the steps for automatically generating a training dataset by linking SRS reports to solar images. The intended workflow is as follows:

1.  **SRS Report Parsing:** Extract sunspot region data from NOAA SRS reports using the `parse_srs_file` function.
2.  **Image Matching:** Match SRS data with corresponding solar images from SDO based on date and location.
3.  **Annotation Generation:** Convert the SRS data into annotation files in YOLO format, indicating the location of sunspots in the images. This is the **critical missing piece** of the current implementation.
4.  **Dataset Creation:** Combine the annotated images to create a training dataset for YOLOv8 and ResNet. This dataset will include bounding box annotations for YOLOv8 and classification labels (derived from SRS data) for ResNet.

**Current Status:** Due to ongoing development, the automated dataset generation pipeline is not yet fully implemented. The current YOLOv8 training process relies on an external dataset from Roboflow: [Sunspot Detection using YOLOv5](https://universe.roboflow.com/internship-projects/sunspot-detection-using-yolov5). 

**Important:** This Roboflow dataset is *not* included in this Git repository and must be downloaded separately. A sample NOAA region file will be included as an example to demonstrate the image acquisition process within the

## 4. Results (Preliminary)

The trained YOLOv8 model outputs the location of detected sunspots in solar images. The model weights and training results are saved in the `sunspot_model` directory. ResNet training and evaluation are planned for future development.

## 5. Code Description

*   **[notebook.ipynb](notebook.ipynb):** Jupyter Notebook for training the YOLOv8 model. Key steps include data loading, model configuration, training execution, and result saving.
*   **[srs_notebook.ipynb](srs_notebook.ipynb):** Jupyter Notebook for parsing SRS files and, in the future, generating training annotations and classification labels. The `parse_srs_file` function extracts relevant data from SRS reports.
*   **[datasets/data.yaml](datasets/data.yaml):** Configuration file for the YOLOv8 training process, specifying the location of training, validation, and testing images, as well as class names. *Note: This file is configured for the Roboflow dataset, which is not included in this repository.*
*   **[yolov8n.pt](yolov8n.pt):** Pre-trained YOLOv8 model.
*   **sunspot\_model/:** This directory contains the output from the YOLOv8 training run. Key files and subdirectories include:
    *   `test_run/args.yaml`: Configuration file detailing the training parameters used (e.g., epochs, image size, batch size).
    *   `test_run/weights/`: Contains the trained model weights. `best.pt` typically represents the best-performing weights, while `last.pt` represents the weights from the final epoch.
    *   `test_run/results.csv`: CSV file containing metrics from the training run.
    *   `test_run/results.png`: A plot summarizing the training results.
    *   `test_run/confusion_matrix.png`: Confusion matrix visualizing the model's performance.
    > In this specific instance, the training run ("test_run") was configured for only one epoch to verify the execution and behavior.*
## 6. Dependencies

The project utilizes the following key Python libraries:

```
cv2
numpy
pandas
matplotlib
ultralytics
IPython
astropy
sunpy
PIL
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```
> Note: The requirements.txt file may include additional packages beyond those listed above to ensure comprehensive environment compatibility.


## 7. Future Work

*   **Complete Automated Dataset Generation:** Finalize the srs_notebook.ipynb notebook to fully automate the process of generating training data and classification labels from SRS reports and solar images. This includes implementing the conversion of SRS data into YOLO annotation files.
*   **ResNet Implementation and Training:** Implement and train a ResNet model for sunspot classification based on SRS data. This will involve training the model to predict McIntosh classes.
*   **Model Evaluation:** Conduct a thorough evaluation of both the YOLOv8 and ResNet models using held-out test datasets.
*   **Explicability with LLM:** Investigate the use of Large Language Models (LLMs) to provide explanations for the classification decisions, linking model outputs to relevant sunspot characteristics.
