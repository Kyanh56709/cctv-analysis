import argparse
from utils import calculate_iou, export_to_csv, run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Traffic Analysis from CCTV Footage")
    parser.add_argument('--video-path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--model-name', type=str, default='yolov8m.pt', help="YOLO model to use (e.g., 'yolov8n.pt', 'yolov8m.pt').")
    parser.add_argument('--output-csv', type=str, default=None, help="Path to save the output CSV report. Defaults to 'report_[videoname].csv'.")
    parser.add_argument('--confidence-threshold', type=float, default=0.4, help="Object detection confidence threshold.")
    parser.add_argument('--merge-iou-threshold', type=float, default=0.3, help="IoU threshold for merging person/motorcycle detections.")
    parser.add_argument('--line-y', type=int, default=500, help="Y-coordinate of the counting line.")
    parser.add_argument('--no-display', action='store_true', help="Run in headless mode without displaying the video feed.")
    
    args = parser.parse_args()
    run(args)