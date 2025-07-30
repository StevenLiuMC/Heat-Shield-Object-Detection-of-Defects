#  AI-Based Industrial Defect Detection System

A powerful and easy-to-use **metal part defect detection software** powered by AI and designed for seamless integration into **industrial PLCs** and automated production lines.

---

## Key Benefits

- **Automated Quality Control**: Instantly detects surface defects on metal parts using AI vision.
- **Zero Expertise Required**: No deep learning knowledge needed—load your model, start the shift, and begin inspection.
- **Real-Time Monitoring**: Tracks defect rates, production totals, and shift performance in real time.
- **Seamless Integration**: Designed for future deployment into real-world production lines via industrial cameras and PLC control.
- **Flexible Reporting**: Export data in CSV, JSON, or HTML formats for quality tracking and reporting.

---

## How It Works

### 1. Load Your Trained AI Model
- Supported formats: `.pt` (YOLO) and `.pth` (RF-DETR)
- Drag your model into the `models/` folder
- Click the **"Load Model"** button in the interface

### 2. Start a New Production Shift
- Enter shift number, operator count, and target cycle time
- The system automatically logs all activity under this shift

### 3. Simulate or Capture Images
- Use the **offline camera simulator** to test the system using stored images (`test images/` folder)
- In the future, switch to a real camera for live detection (planned `camera_online` module)
- If you choose our product, we will continue to develop and customize camera integration modules to support industrial-grade live detection, tailored to your actual production needs.

### 4. Trigger Detection
- Click **"Trigger Detection"** to simulate a capture
- The software will analyze the image and display:
  - Detected defects (if any)
  - Annotated image with bounding boxes
  - Confidence levels
  - Summary of the result (Good / Defect Detected)

### 5. Monitor Real-Time OEE Metrics

This system automatically calculates and displays your live production performance using the OEE standard:

- **Availability**: Tracks equipment uptime
- **Performance**: Compares actual vs expected production speed
- **Quality**: Measures how many parts are defect-free
- **OEE Score**: Combined score of the above, shown live on the dashboard

 For a full explanation of how OEE is calculated and interpreted, see the [What is OEE?](#what-is-oee) section below.

### 6. Log Downtime
- Record the reason and duration for any unexpected stoppages
- Downtime is included in availability calculations

### 7. Export Reports
- At any point, export production data as:
  - `.csv` — for spreadsheets
  - `.json` — for integration with ERP/MES
  - `.html` — for executive reporting with charts and summaries

## Screenshots
- Defect detected on the metal part.
![Defect detection result](https://github.com/StevenLiuMC/Heat-Shield-Object-Detection-of-Defects/blob/main/Detection%20Software%20v1/software%20image%20screenshot/running%20test%20defect%20sample.png)
- Good metal part
![Good result](https://github.com/StevenLiuMC/Heat-Shield-Object-Detection-of-Defects/blob/main/Detection%20Software%20v1/software%20image%20screenshot/running%20test%20good.png)

---

## What is OEE?
- **OEE (Overall Equipment Effectiveness)** is a gold standard metric used by manufacturers to measure how effectively a production line is utilized. It combines **Availability**, **Performance**, and **Quality** into a single, easy-to-understand percentage.

### OEE Formula
```text
OEE = Availability × Performance × Quality
```

### Components Explained
- **Availability**
  - Measures how much of the scheduled time the equipment is actually running.
  - Availability = (Planned Time - Downtime) ÷ Planned Time
- **Performance**
  - Compares the speed of production to the ideal target speed.
  - Performance = Actual Output ÷ Expected Output
- **Quality**
  - Measures the proportion of good parts produced.
  - Quality = Good Parts ÷ Total Parts Produced

### Example
**If in an 8-hour shift:**
- Machine ran for 7 hours (1 hour downtime)
- Produced 400 parts (ideal is 500)
- 380 of them were good

**Then:**
- Availability = 7 ÷ 8 = 87.5%
- Performance = 400 ÷ 500 = 80%
- Quality = 380 ÷ 400 = 95%

**OEE** = 0.875 × 0.80 × 0.95 ≈ 66.5%

### What Does It Mean?
| **OEE Score**           | **Interpretation**            |
|:------------------------|:------------------------------|
| 85% and above           | World-class efficiency        |
| 60–85%                  | Room for improvement          |
| Below 60%               | Significant losses present    |

---

## Program Structure
```
[run.py]
  └── [main.py]
        ├── [camera_offline.py]        # Camera simulator, load test image
        ├── [model_manager.py]         # Model loader (supports YOLO and RF-DETR)
        │     └── [logger_config.py] # Log configurator, providing a unified logging interface for all modules
        ├── [inference_engine.py]      # Inference Pipeline
        │     └── [logger_config.py]
        ├── [database_manager.py]      # Database operator, recording production cycle, OEE, downtime information, etc.
        │     └── [logger_config.py]
        ├── [export_manager.py]        # Data exporter, export CSV / JSON / HTML reports
        │     └── [database_manager.py]
        └── [logger_config.py]         # The GUI itself also records logs
```

## System Requirements
```
Python >= 3.8
PyQt6 >= 6.4.0
torch >= 2.0.0
torchvision >= 0.15.0
opencv-python >= 4.8.0
numpy >= 1.24.0
Pillow >= 9.5.0
ultralytics >= 8.0.0
sqlite3
```
Recommended GPU with CUDA for real-time performance (CPU also supported)

To install the required dependencies, run the following command in your terminal:
```
pip install -r requirements.txt
```

## Getting Started
1. Install dependencies
2. Place your trained model into models/
3. Add test images into test images/
4. Run the application:
```
python run.py
```

## Designed For
- **Factory managers** looking to modernize QA processes
- **Manufacturers** aiming to reduce manual inspection cost
- **Industrial engineers** integrating AI into production workflows

## Behind the Scenes (No Expertise Needed!)
- Powered by state-of-the-art deep learning models (YOLO / RF-DETR)
- Models are pre-trained and then fine-tuned for your specific parts
- Smart logic automatically classifies each part as:
  -  Good
  -  Defect Detected (based on detection confidence)

### Academic Validation Results
- Impact of Data Augmentation
![Impact of Data Augmentation](https://github.com/StevenLiuMC/Heat-Shield-Object-Detection-of-Defects/blob/main/Impact%20of%20Data%20Augmentation%20on%20mAP%2C%20Average%20Precision%20and%20Recall.png)
- Model Performance Comparison
![Model Performance Comparison](https://github.com/StevenLiuMC/Heat-Shield-Object-Detection-of-Defects/blob/main/Model%20Performance%20Comparison%20Based%20on%20mAP%2C%20Precision%20and%20Recall.png)

# Ready to transform your production line?
- Get started with **smart**, **automated** defect detection system!
- **If you know how to click a mouse, you already know how to use this software!**
