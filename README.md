# 🚦 Traffic Surveillance System

A real-time traffic surveillance system built using **YOLO (You Only Look Once)** for object detection and **Streamlit** for an interactive user interface. This application allows users to analyze traffic data from images, videos, or a live webcam feed with adjustable detection sensitivity and model selection.

---

## 📌 Features

* 🔍 **Object Detection with YOLO**

  * Detect vehicles, pedestrians, and other objects in real time.

* 🎥 **Multiple Input Sources**

  * **Image Upload** – Analyze static images.
  * **Video Upload** – Process pre-recorded traffic footage.
  * **Webcam Feed** – Perform real-time detection.

* 🎛️ **Confidence Threshold Slider**

  * Adjust detection sensitivity dynamically.

* 🧠 **Multiple YOLO Models**

  * Choose from **5 different YOLO models** for varying performance and accuracy trade-offs.

* 📊 **Object Counting Module**

  * Separate module for counting objects (vehicles, etc.) in a scene.

---

## 🗂️ Project Structure

```
├── streamlit.py        # Streamlit app (UI & detection interface)
├── main.py             # Core logic for object detection & counting (non-Streamlit)
├── models_dir/         # YOLO model weights
├── core/               # Model Loader, Hyperparameters, etc.
├── utils/              # Helper functions (preprocessing, visualization, etc.)
├── requirements.txt    # Dependencies
├── packages            # Dependencies 
└── README.md           # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/traffic-surveillance.git
   cd traffic-surveillance
   ```

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### 1. Run Streamlit App

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

### 2. Using the Interface

* Select input type:

  * 📷 Image
  * 🎞️ Video
  * 📹 Webcam

* Adjust:

  * **Confidence Slider** to filter detections
  * **YOLO Model** from dropdown (5 options available)

* View:

  * Real-time detection results with bounding boxes

---

### 3. Run Object Counting (Standalone)

The `main.py` module provides functionality beyond the Streamlit interface, including:

* 🚗 Vehicle counting
* 📈 Traffic flow analysis
* 🧮 Custom detection pipelines

Run it using:

```bash
python main.py
```

---

## 🧠 YOLO Models

The system supports multiple YOLO variants to balance speed and accuracy:

| Model                         | Description                                                                  |
|-------------------------------|------------------------------------------------------------------------------|
| yolo11n.pt                    | Fastest, lower accuracy                                                      |
| yolo11n-pretrained-traffic.pt | Fine Tuned for Traffic Surveillance, Best for traffic surveillance from afar |
| yolo11s.pt                    | Improved accuracy                                                            |
| yolo26n.pt                    | High precision                                                               |
| yolo26s.pt                    | Slightly slower than 's' models but better overall                           |

---

## 📸 Example Outputs

* Bounding boxes on vehicles and pedestrians
* Real-time webcam detection
* Object counts in video streams

---

## 🔧 Technologies Used

* **YOLO** – Object Detection
* **Streamlit** – Web Interface
* **OpenCV** – Image & Video Processing
* **Python** – Core Programming Language

---

## 🚀 Future Improvements

* Traffic density estimation
* Lane detection integration
* Speed estimation of vehicles
* Cloud deployment support

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Fork the repo
* Create a new branch
* Submit a pull request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

For questions or suggestions, feel free to reach out.

---

⭐ *If you found this project useful, consider giving it a star!*
