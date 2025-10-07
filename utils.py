import cv2
from ultralytics import YOLO
from collections import deque, Counter
import torch
import numpy as np
import csv
import os
from pathlib import Path

def calculate_iou(boxA, boxB):
    """
    Tính chỉ số Intersection over Union (IoU) giữa hai bounding box.
    IoU là thước đo mức độ chồng lấn giữa hai hộp.

    Args:
        boxA (list or tuple): Tọa độ hộp thứ nhất [x1, y1, x2, y2].
        boxB (list or tuple): Tọa độ hộp thứ hai [x1, y1, x2, y2].

    Returns:
        float: Giá trị IoU, từ 0 (không chồng lấn) đến 1 (chồng lấn hoàn toàn).
    """
    # Xác định tọa độ (x, y) của vùng giao nhau (intersection)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích vùng giao nhau
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Tính diện tích của từng hộp
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính IoU theo công thức: diện tích giao nhau / diện tích hợp nhất
    # Diện tích hợp nhất = Diện tích A + Diện tích B - Diện tích giao nhau
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def export_to_csv(video_path, in_counts, out_counts):
    video_filename = os.path.basename(video_path)
    report_filename = f'analysis_report_{os.path.splitext(video_filename)[0]}.csv'
    
    print(f"\nĐang xuất báo cáo phân tích ra file: {report_filename}")

    header = ['Metric', 'Value']
    all_classes = sorted(list(set(in_counts.keys()) | set(out_counts.keys()) - {'total'}))
    
    with open(report_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(['--- SUMMARY ---', ''])
        writer.writerow(['Total Objects Entering', in_counts.get('total', 0)])
        writer.writerow(['Total Objects Exiting', out_counts.get('total', 0)])
        writer.writerow(['--- ENTERING DETAILS ---', ''])
        for cls in all_classes: writer.writerow([f'Entering: {cls}', in_counts.get(cls, 0)])
        writer.writerow(['--- EXITING DETAILS ---', ''])
        for cls in all_classes: writer.writerow([f'Exiting: {cls}', out_counts.get(cls, 0)])
    
    print("Xuất báo cáo thành công!")

def calculate_ioa(box_small, box_large):
    """
    Tính toán tỷ lệ diện tích giao nhau trên diện tích của hộp nhỏ hơn (IoA).
    Dùng để kiểm tra xem box_small có bị chứa trong box_large hay không.
    """
    # Tọa độ của vùng giao nhau
    x1_inter = max(box_small[0], box_large[0])
    y1_inter = max(box_small[1], box_large[1])
    x2_inter = min(box_small[2], box_large[2])
    y2_inter = min(box_small[3], box_large[3])

    # Tính diện tích vùng giao nhau
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Tính diện tích của box nhỏ
    small_box_area = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])

    # Tính IoA
    ioa = inter_area / small_box_area if small_box_area > 0 else 0
    return ioa

def run(config):
    """Main function to run the traffic analysis."""
    # --- CONFIGURATION from args ---
    VIDEO_PATH = config.video_path
    MODEL_NAME = config.model_name
    CONFIDENCE_THRESHOLD = config.confidence_threshold
    TARGET_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
    WINDOW_NAME = 'CCTV Traffic Analysis'
    SHOW_DISPLAY = not config.no_display

    # Logic for merging Person/Motorcycle
    MERGE_IOU_THRESHOLD = config.merge_iou_threshold

    # Trajectory and line configuration
    TRAJECTORY_MAX_LEN = 30
    LINE_Y = config.line_y

    # Congestion detection configuration
    CONGESTION_DISTANCE_THRESHOLD = 70
    CONGESTION_NEIGHBOR_THRESHOLD = 4
    CONGESTION_GLOBAL_ALERT_THRESHOLD = 8

    # --- GLOBAL VARIABLE INITIALIZATION ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    track_history = {}
    track_class_history = {}
    track_colors = {}
    in_counts_by_class = Counter()
    out_counts_by_class = Counter()

    # --- MODEL AND VIDEO INITIALIZATION ---
    print(f"Using device: {device}")
    print(f"Loading YOLO model: {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}\nPlease ensure the model file '{MODEL_NAME}' is accessible.")
        return

    PERSON_CLASS_ID = list(model.names.keys())[list(model.names.values()).index('person')]
    MOTORCYCLE_CLASS_ID = list(model.names.keys())[list(model.names.values()).index('motorcycle')]
    print("Model loaded successfully!")

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.2f}, Total Frames: {total_frames}")

    if SHOW_DISPLAY:
        cv2.namedWindow(WINDOW_NAME)
    is_paused = False
    process_in_background = not SHOW_DISPLAY

    # --- MAIN LOOP ---
    while cap.isOpened():
        if SHOW_DISPLAY and is_paused and not process_in_background:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '): is_paused = False
            elif key == ord('q'): break
            continue

        ret, frame = cap.read()
        if not ret:
            print("\nFinished processing the video.")
            break

        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # --- TRACKING AND SMOOTHING ---
        results = model.track(source=frame, persist=True, tracker='bytetrack.yaml', device=device, verbose=False)
        
        detections = []
        if results[0].boxes.id is not None:
            boxes, ids, confs, clss = (d.cpu().numpy() for d in (results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.conf, results[0].boxes.cls))
            for box, track_id, conf, cls in zip(boxes.astype(int), ids.astype(int), confs, clss.astype(int)):
                if cls in TARGET_CLASSES and conf > CONFIDENCE_THRESHOLD:
                    detections.append({'box': list(box), 'cls': int(cls), 'conf': conf, 'id': int(track_id)})

        # Class smoothing
        smoothed_detections = []
        for det in detections:
            track_id = det['id']
            if track_id not in track_class_history: track_class_history[track_id] = deque(maxlen=20)
            track_class_history[track_id].append(det['cls'])
            det['cls'] = Counter(track_class_history[track_id]).most_common(1)[0][0]
            smoothed_detections.append(det)

        # Merge person-motorcycle logic
        indices_to_remove = set()
        temp_detections = list(smoothed_detections)
        for i in range(len(temp_detections)):
            for j in range(i + 1, len(temp_detections)):
                if i in indices_to_remove or j in indices_to_remove: continue
                det1, det2 = temp_detections[i], temp_detections[j]
                is_pair = (det1['cls'] == PERSON_CLASS_ID and det2['cls'] == MOTORCYCLE_CLASS_ID) or \
                          (det1['cls'] == MOTORCYCLE_CLASS_ID and det2['cls'] == PERSON_CLASS_ID)
                if is_pair and calculate_iou(det1['box'], det2['box']) > MERGE_IOU_THRESHOLD:
                    if det1['cls'] == PERSON_CLASS_ID: indices_to_remove.add(i)
                    else: indices_to_remove.add(j)
        final_detections = [det for idx, det in enumerate(temp_detections) if idx not in indices_to_remove]
        
        # --- UPDATE TRAJECTORIES AND DETECT CONGESTION ---
        current_ids = {det['id'] for det in final_detections}
        for det in final_detections:
            track_id, box = det['id'], det['box']
            anchor_point = ((box[0] + box[2]) // 2, box[3])
            if track_id not in track_history: track_history[track_id] = deque(maxlen=TRAJECTORY_MAX_LEN)
            track_history[track_id].append(anchor_point)

        # Clean up old tracks
        inactive_ids = set(track_history.keys()) - current_ids
        for inactive_id in inactive_ids:
            for history_dict in [track_history, track_class_history, track_colors]:
                if inactive_id in history_dict: del history_dict[inactive_id]

        # Congestion Logic
        congested_vehicle_ids = set()
        vehicles_only_detections = [det for det in final_detections if det['cls'] != PERSON_CLASS_ID]
        vehicle_positions = {det['id']: track_history.get(det['id'])[-1] for det in vehicles_only_detections if det.get('id') and track_history.get(det['id'])}
        if len(vehicle_positions) > 1:
            for track_id_1, pos_1 in vehicle_positions.items():
                neighbor_count = 0
                for track_id_2, pos_2 in vehicle_positions.items():
                    if track_id_1 != track_id_2 and np.linalg.norm(np.array(pos_1) - np.array(pos_2)) < CONGESTION_DISTANCE_THRESHOLD:
                        neighbor_count += 1
                if neighbor_count >= CONGESTION_NEIGHBOR_THRESHOLD:
                    congested_vehicle_ids.add(track_id_1)

        # --- COUNTING ---
        for det in final_detections:
            track_id, cls = det['id'], det['cls']
            trajectory = track_history.get(track_id, [])
            if len(trajectory) >= 2:
                current_y, prev_y = trajectory[-1][1], trajectory[-2][1]
                cls_name = model.names[cls]
                if prev_y <= LINE_Y < current_y and (track_id, 'in') not in in_counts_by_class:
                    in_counts_by_class[cls_name] += 1
                    in_counts_by_class[(track_id, 'in')] = True
                    if cls_name != 'person': in_counts_by_class['total_vehicles'] += 1
                elif prev_y >= LINE_Y > current_y and (track_id, 'out') not in out_counts_by_class:
                    out_counts_by_class[cls_name] += 1
                    out_counts_by_class[(track_id, 'out')] = True
                    if cls_name != 'person': out_counts_by_class['total_vehicles'] += 1

        # --- VISUALIZATION / PROGRESS UPDATE ---
        if not process_in_background:
            cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 0), 2)
            for det in final_detections:
                track_id, box, cls = det['id'], det['box'], det['cls']
                if track_id not in track_colors: track_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                is_congested = track_id in congested_vehicle_ids
                color = (0, 0, 255) if is_congested else track_colors[track_id]
                
                # Draw trajectory
                trajectory = track_history.get(track_id, [])
                if len(trajectory) > 1:
                    pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
                
                label = f'ID:{track_id} {model.names[cls]}'
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display counts and alerts
            count_in_text = f"Entering Vehicles: {in_counts_by_class.get('total_vehicles', 0)}"
            count_out_text = f"Exiting Vehicles: {out_counts_by_class.get('total_vehicles', 0)}"
            cv2.putText(frame, count_in_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, count_out_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if len(congested_vehicle_ids) > CONGESTION_GLOBAL_ALERT_THRESHOLD:
                cv2.putText(frame, "TRAFFIC CONGESTION ALERT!", (frame.shape[1] // 2 - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                choice = input("Viewer closed. Process remaining frames for full report? (y/n): ").lower()
                if choice == 'y':
                    process_in_background = True
                    print("Processing remaining frames in background... Please wait.")
                else:
                    break
            elif key == ord(' '):
                is_paused = not is_paused
        else: # Background processing
            if current_frame_num % 100 == 0 or current_frame_num == total_frames:
                progress = (current_frame_num / total_frames) * 100
                print(f"Processing... {progress:.2f}% complete", end='\r')

    # --- CLEANUP AND REPORTING ---
    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    export_to_csv(VIDEO_PATH, in_counts_by_class, out_counts_by_class, config.output_csv)
    print("Program finished.")