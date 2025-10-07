# Phân Tích Lưu Lượng Giao Thông và Phát Hiện Tắc Nghẽn bằng YOLOv8

Dự án này sử dụng mô hình YOLOv8 để phân tích video từ camera giao thông, thực hiện các tác vụ theo thời gian thực bao gồm: theo dõi đối tượng (tracking), đếm phương tiện ra/vào một khu vực xác định, và cảnh báo khi có dấu hiệu tắc nghẽn giao thông.

## Mục lục
- [Tính năng chính](#tính-năng-chính)
- [Demo](#demo)
- [Yêu cầu](#yêu-cầu)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Sử dụng](#sử-dụng)
- [Luồng hoạt động](#luồng-hoạt-động)
- [Đầu ra](#đầu-ra)

## Tính năng chính

- **Theo dõi đa đối tượng (Multi-Object Tracking)**: Sử dụng YOLOv8 kết hợp với thuật toán ByteTrack để phát hiện và gán một ID duy nhất cho mỗi đối tượng (người, ô tô, xe máy, xe buýt, xe tải) di chuyển trong khung hình.
- **Đếm phương tiện**: Đếm số lượng phương tiện đi vào và đi ra khỏi một làn ranh ảo được xác định trước trong video.
- **Phát hiện tắc nghẽn**: Phân tích mật độ và khoảng cách giữa các phương tiện để đưa ra cảnh báo khi có nguy cơ xảy ra tắc nghẽn.
- **Gộp đối tượng thông minh**: Tự động gộp đối tượng 'người' (person) và 'xe máy' (motorcycle) khi chúng có độ chồng chéo (IoU) cao, tránh việc đếm nhầm người lái xe là một đối tượng riêng biệt.
- **Làm mịn phân loại**: Giảm thiểu hiện tượng "nhấp nháy" khi mô hình phân loại sai một đối tượng trong vài khung hình bằng cách sử dụng lịch sử phân loại gần nhất.
- **Trình xem tương tác**:
    - Tạm dừng/Tiếp tục xử lý video bằng phím `Space`.
    - Thoát trình xem nhưng vẫn tiếp tục xử lý video ở chế độ nền để xuất báo cáo đầy đủ.
- **Xuất báo cáo**: Tự động tạo file CSV tổng hợp kết quả đếm được sau khi xử lý xong video.

## Demo


## Yêu cầu

- Python 3.8+
- PyTorch (khuyến nghị phiên bản có hỗ trợ CUDA để tăng tốc GPU)
- OpenCV
- Ultralytics (YOLOv8)


## Cấu hình

Tất cả các tham số chính có thể được tùy chỉnh trong phần `--- CẤU HÌNH ---` của file script.

-   `VIDEO_PATH`: Đường dẫn đến file video đầu vào.
-   `MODEL_NAME`: Tên file mô hình YOLOv8 cần sử dụng (ví dụ: `yolov8n.pt`, `yolov8m.pt`, `yolov8l.pt`).
-   `CONFIDENCE_THRESHOLD`: Ngưỡng tin cậy để xem xét một phát hiện là hợp lệ (0.0 - 1.0).
-   `TARGET_CLASSES`: Danh sách ID của các lớp đối tượng cần theo dõi.
    - `0`: person, `2`: car, `3`: motorcycle, `5`: bus, `7`: truck
-   `MERGE_IOU_THRESHOLD`: Ngưỡng IoU (Intersection over Union) để gộp một đối tượng 'person' vào một 'motorcycle'.
-   `LINE_Y`: Tọa độ y của đường ranh ảo dùng để đếm phương tiện.
-   **Cấu hình phát hiện tắc nghẽn**:
    -   `CONGESTION_DISTANCE_THRESHOLD`: Khoảng cách tối đa (pixel) để coi hai phương tiện là "hàng xóm" của nhau.
    -   `CONGESTION_NEIGHBOR_THRESHOLD`: Số lượng "hàng xóm" tối thiểu để một phương tiện bị coi là đang trong vùng tắc nghẽn.
    -   `CONGESTION_GLOBAL_ALERT_THRESHOLD`: Số lượng phương tiện tắc nghẽn tối thiểu để hiển thị cảnh báo tắc nghẽn toàn cục.

## Sử dụng

1.  Mở file script và chỉnh sửa các thông số trong phần `--- CẤU HÌNH ---` cho phù hợp.
2.  Chạy script từ terminal:
    ```bash
    python main.py
    ```
3.  Một cửa sổ OpenCV sẽ hiện lên để hiển thị video đang được xử lý.
    -   Nhấn phím **`Space`** để Tạm dừng/Tiếp tục.
    -   Nhấn phím **`q`** để đóng cửa sổ xem. Sau khi đóng, chương trình sẽ hỏi bạn có muốn tiếp tục xử lý các khung hình còn lại ở chế độ nền để có báo cáo đầy đủ hay không.

## Luồng hoạt động

1.  **Khởi tạo**: Tải mô hình YOLOv8, mở file video và khởi tạo các biến cần thiết.
2.  **Vòng lặp chính**: Xử lý từng khung hình của video.
    a. **Phát hiện & Theo dõi**: Sử dụng `model.track()` để phát hiện và gán ID cho các đối tượng.
    b. **Lọc & Làm mịn**: Lọc các đối tượng không mong muốn và có độ tin cậy thấp. Làm mịn lớp của đối tượng dựa trên lịch sử để tránh thay đổi đột ngột.
    c. **Gộp Người-Xe máy**: Kiểm tra các cặp đối tượng 'person' và 'motorcycle' gần nhau. Nếu IoU vượt ngưỡng, đối tượng 'person' sẽ được loại bỏ để tránh đếm trùng.
    d. **Cập nhật Quỹ đạo**: Lưu lại vị trí của mỗi đối tượng để vẽ đường đi.
    e. **Phát hiện Tắc nghẽn**: Dựa trên vị trí hiện tại của các phương tiện, tính toán mật độ để xác định các phương tiện đang bị kẹt.
    f. **Đếm Phương tiện**: Kiểm tra xem quỹ đạo của đối tượng có cắt qua đường ranh ảo (`LINE_Y`) hay không để cập nhật bộ đếm vào/ra.
    g. **Hiển thị**: Vẽ các bounding box, ID, quỹ đạo, đường đếm và thông tin thống kê lên khung hình.
3.  **Kết thúc & Báo cáo**: Sau khi xử lý xong video (hoặc người dùng chọn xử lý nền), chương trình sẽ giải phóng tài nguyên và gọi hàm `export_to_csv()` để lưu kết quả đếm vào một file CSV.

## Đầu ra

Khi chương trình kết thúc, một file báo cáo có tên `report_TEN_VIDEO.csv` sẽ được tạo trong cùng thư mục. File này chứa các thông tin sau:

-   Tên file video đã phân tích.
-   Thời gian xuất báo cáo.
-   Thống kê chi tiết số lượng của từng loại phương tiện đi vào.
-   Thống kê chi tiết số lượng của từng loại phương tiện đi ra.
-   Tổng số phương tiện (không bao gồm người đi bộ) đi vào và đi ra.

Ví dụ nội dung file `report_Road_2.csv`:

```csv
Key,Value
Video Path,C:\Users\lekya\...\Road_2.mp4
Report Time,2023-10-27 15:30:00
--- Entering Counts ---,
car,15
motorcycle,25
bus,2
total_vehicles,42
--- Exiting Counts ---,
car,12
motorcycle,22
truck,1
total_vehicles,35
```