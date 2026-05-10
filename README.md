# Speed Estimation & Vehicle Tracking 🚗💨

A high-performance vehicle tracking and speed estimation system built with **YOLOv11**, **Supervision**, and **OpenCV**. This project utilizes perspective transformation to map camera-view coordinates to real-world coordinates, allowing for accurate speed calculation of vehicles in video streams.

## 🚀 Key Features

*   **Vehicle Detection**: Leverages the state-of-the-art **YOLOv11x** model for robust object detection.
*   **Object Tracking**: Implements **ByteTrack** for consistent ID tracking across frames.
*   **Perspective Transformation**: Uses a Bird's-Eye View (BEV) mapping to convert pixel displacements into real-world distances.
*   **Speed Estimation**: Calculates real-time speed in km/h based on temporal displacement.
*   **Rich Annotations**: Visualizes tracking IDs, bounding boxes, speed labels, and historical movement traces.
*   **Region of Interest (ROI)**: Focuses processing on a specific polygon zone to filter out irrelevant background activity.

## 🛠️ Technology Stack

*   **Python**: Core programming language.
*   **Ultralytics YOLOv11**: Object detection framework.
*   **Supervision**: Comprehensive library for computer vision tasks (tracking, zones, annotations).
*   **OpenCV**: Image processing and real-time visualization.
*   **NumPy**: Numerical operations and coordinate transformations.

## 📁 Project Structure

```text
.
├── data/                   # Directory for input/output video assets
├── main.py                 # Core logic for tracking and speed estimation
├── video_downloader.py     # Utility script to download sample video assets
├── yolo11x.pt             # YOLOv11 model weights (automatically downloaded)
└── README.md               # Project documentation
```

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/speed-estimation-vehicle-tracking.git
    cd speed-estimation-vehicle-tracking
    ```

2.  **Install dependencies**:
    ```bash
    pip install ultralytics supervision opencv-python numpy
    ```

## 🚦 Usage

### 1. Download Sample Data
Run the downloader script to fetch the sample video asset:
```bash
python video_downloader.py
```

### 2. Run the Application
Execute the main script to start tracking and speed estimation:
```bash
python main.py
```
*Press `q` to quit the visualization window.*

## 🧠 How It Works

1.  **Detection & ROI**: The system detects vehicles using YOLOv11. A `PolygonZone` filters detections to only include those within the defined road section.
2.  **Tracking**: ByteTrack assigns unique IDs to each vehicle, maintaining continuity across frames.
3.  **Coordinate Mapping**: The `ViewTransformer` class applies a perspective transform matrix (calculated via `cv2.getPerspectiveTransform`) to map the trapezoidal road view to a rectangular grid.
4.  **Speed Calculation**:
    *   The system tracks the bottom-center anchor of each vehicle.
    *   Distance is measured between the current position and the position one second (FPS frames) ago in the transformed space.
    *   The speed is calculated as `distance / time` and converted to km/h.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---
*Built with ❤️ for Computer Vision enthusiasts.*
