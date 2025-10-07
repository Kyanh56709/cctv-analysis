# Phân Tích Lưu Lượng Giao Thông và Phát Hiện Tắc Nghẽn bằng YOLOv8

<!-- TODO: Thay thế link trên bằng GIF demo dự án của bạn -->

Dự án này sử dụng mô hình YOLOv8 để phân tích video từ camera giao thông, thực hiện các tác vụ theo thời gian thực bao gồm: theo dõi đối tượng, đếm phương tiện ra/vào một khu vực xác định, và cảnh báo khi có dấu hiệu tắc nghẽn giao thông.

## Tính năng chính

-   **Theo dõi đa đối tượng (Multi-Object Tracking)**: Sử dụng YOLOv8 kết hợp với thuật toán ByteTrack để phát hiện và gán một ID duy nhất cho mỗi đối tượng (người, ô tô, xe máy, xe buýt, xe tải).
-   **Đếm phương tiện**: Đếm chính xác số lượng phương tiện khi chúng đi qua một đường ranh ảo, phân loại theo chiều vào/ra.
-   **Phát hiện tắc nghẽn**: Phân tích mật độ và khoảng cách giữa các phương tiện để đưa ra cảnh báo khi có nguy cơ xảy ra tắc nghẽn.
-   **Gộp đối tượng thông minh**: Tự động gộp đối tượng 'người' (person) và 'xe máy' (motorcycle) khi chúng có độ chồng chéo (IoU) cao, tránh việc đếm nhầm người lái xe là một đối tượng riêng biệt.
-   **Giao diện dòng lệnh (CLI)**: Toàn bộ quá trình có thể được cấu hình và chạy qua các tham số terminal, dễ dàng tích hợp và xử lý hàng loạt.
-   **Chế độ không giao diện (Headless Mode)**: Hỗ trợ chạy mà không cần hiển thị cửa sổ video, phù hợp để triển khai trên máy chủ hoặc các tác vụ tự động.
-   **Xuất báo cáo CSV**: Tự động tạo file CSV sạch sẽ, tổng hợp kết quả đếm được sau khi xử lý.

## Cấu trúc dự án
```
yolo-traffic-analysis/
│
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
│
├── videos/
│   └── (Nơi chứa các video mẫu)
│
└── reports/
    └── (Thư mục được tạo tự động để lưu báo cáo CSV)
```

## Cài đặt

#### 1. Clone repository
```bash
git clone https://github.com/your-username/yolo-traffic-analysis.git
cd yolo-traffic-analysis
```

#### 2. Tạo môi trường ảo (Khuyến khích)
Điều này giúp quản lý các gói phụ thuộc của dự án một cách độc lập.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Cài đặt các gói phụ thuộc
Tất cả các thư viện cần thiết đều được liệt kê trong file `requirements.txt`.
```bash
pip install -r requirements.txt
```
*Lưu ý: Các trọng số của mô hình YOLOv8 sẽ được thư viện `ultralytics` tự động tải về trong lần chạy đầu tiên.*

## Sử dụng

Chương trình được thiết kế để chạy từ terminal với các tham số tùy chỉnh.

#### Chạy với giao diện hiển thị
Để phân tích một video và xem trực tiếp kết quả xử lý:
```bash
python main.py --video-path "path/to/your/video.mp4"
```
-   Nhấn phím **`Space`** để Tạm dừng/Tiếp tục.
-   Nhấn phím **`q`** để đóng cửa sổ. Bạn sẽ được hỏi có muốn tiếp tục xử lý ở chế độ nền để hoàn thành báo cáo không.

#### Chạy ở chế độ nền (Headless)
Để xử lý video mà không cần hiển thị cửa sổ, và lưu báo cáo vào một đường dẫn cụ thể:
```bash
python main.py --video-path "videos/sample_video.mp4" --output-csv "reports/custom_report.csv" --no-display
```

### Các tham số dòng lệnh

| Tham số | Mô tả | Mặc định |
| :--- | :--- | :--- |
| `--video-path` | **(Bắt buộc)** Đường dẫn đến file video đầu vào. | `None` |
| `--model-name` | Mô hình YOLO để sử dụng (ví dụ: `yolov8n.pt`). | `yolov8m.pt` |
| `--output-csv` | Đường dẫn để lưu file báo cáo CSV. | `report_[tên_video].csv` |
| `--confidence-threshold` | Ngưỡng tin cậy để xác định một đối tượng (0.0 - 1.0). | `0.4` |
| `--line-y` | Tọa độ y của đường ranh ảo dùng để đếm. | `500` |
| `--no-display` | Cờ để chạy chương trình mà không hiển thị cửa sổ OpenCV. | `False` |


## Luồng hoạt động

1.  **Khởi tạo**: Tải mô hình YOLOv8, đọc video từ đường dẫn được cung cấp, và khởi tạo các biến.
2.  **Vòng lặp chính**: Xử lý từng khung hình của video.
    a. **Phát hiện & Theo dõi**: Sử dụng `model.track()` để phát hiện và gán ID cho các đối tượng.
    b. **Lọc & Gộp đối tượng**: Lọc các đối tượng có độ tin cậy thấp và thực hiện logic gộp 'người' và 'xe máy'.
    c. **Cập nhật Quỹ đạo & Phát hiện Tắc nghẽn**: Lưu lại lịch sử vị trí của đối tượng và phân tích mật độ phương tiện.
    d. **Đếm Phương tiện**: Kiểm tra quỹ đạo của đối tượng có cắt qua đường ranh ảo (`LINE_Y`) hay không để cập nhật bộ đếm.
    e. **Hiển thị / Cập nhật tiến trình**: Vẽ các thông tin lên khung hình (nếu không ở chế độ headless) hoặc in tiến trình xử lý ra console.
3.  **Kết thúc & Báo cáo**: Sau khi xử lý xong video, chương trình sẽ giải phóng tài nguyên và gọi hàm `export_to_csv()` để lưu kết quả vào file CSV.

## Đầu ra

Khi chương trình kết thúc, một file báo cáo CSV sẽ được tạo ra (ví dụ: `report_Road_2.csv`). File này chứa thống kê chi tiết số lượng của từng loại phương tiện đi vào và đi ra.

**Ví dụ nội dung file báo cáo:**

| Class | Entering_Count | Exiting_Count |
| :--- | :--- |:--- |
| car | 15 | 12 |
| motorcycle | 25 | 22 |
| bus | 2 | 0 |
| truck | 0 | 1 |
| --- | --- | --- |
| **Total Vehicles** | **42** | **35** |
