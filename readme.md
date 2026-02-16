# AI Audio Book Converter

Ứng dụng Streamlit giúp chuyển đổi sách điện tử (PDF) thành sách nói (Audiobook - MP3) sử dụng công nghệ Text-to-Speech (TTS) chất lượng cao từ Microsoft Edge (Edge TTS).

## Chính sách & Tính năng

- **Chuyển đổi PDF sang Text**: Sử dụng `pdfplumber` để trích xuất văn bản, hỗ trợ loại bỏ Header/Footer bằng crop-box.
- **Làm sạch văn bản thông minh**:
  - Tự động nối các từ bị ngắt dòng (smart de-hyphenation).
  - Loại bỏ số trang, ký tự rác.
  - Nhận diện và giữ nguyên cấu trúc đoạn văn bản.
- **Chỉnh sửa nội dung**: Cho phép xem trước và chỉnh sửa văn bản sau khi trích xuất trước khi chuyển thành giọng nói.
- **Text-to-Speech (TTS)**:
  - Sử dụng giọng đọc Neural tự nhiên (Hỗ trợ Tiếng Việt & Tiếng Anh).
  - Tùy chỉnh tốc độ đọc.
- **Quản lý file đầu ra**:
  - Chia nhỏ file audio theo từng phần (Chunking) để tránh lỗi khi xử lý văn bản dài.
  - Hỗ trợ **Resume** (tiếp tục tạo từ đoạn đang dang dở) và **Retry** (thử lại khi lỗi mạng).
  - Tải xuống từng phần hoặc nén toàn bộ thành file ZIP.

## Cài đặt

Yêu cầu: Python 3.8 trở lên.

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd convert_audio
   ```

2. **Tạo môi trường ảo (khuyên dùng):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Trên macOS/Linux
   # hoặc
   .venv\Scripts\activate     # Trên Windows
   ```

3. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

## Sử dụng

1. **Chạy ứng dụng:**
   ```bash
   streamlit run app.py
   ```

2. **Giao diện Web sẽ mở ra:**
   - **Bước 1**: Upload file PDF cần chuyển đổi.
   - **Bước 2**: Cấu hình biên (margin) để loại bỏ header/footer thừa. Kiểm tra và chỉnh sửa nội dung văn bản nếu cần.
   - **Bước 3**: Chọn giọng đọc (Voice) và tốc độ (Rate). Nhấn **Tạo Audio**.
   - **Bước 4**: Tải xuống các file MP3 sau khi hoàn tất.

## Cấu trúc thư mục

```
convert_audio/
├── app.py                # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt      # Danh sách thư viện phụ thuộc
├── output/               # Thư mục chứa các file MP3 đầu ra (được gitignore)
├── .gitignore            # Cấu hình file cần bỏ qua của Git
└── ...
```

## Thư viện chính

- [Streamlit](https://streamlit.io/): Framework giao diện Web.
- [pdfplumber](https://github.com/jsvine/pdfplumber): Trích xuất văn bản từ PDF.
- [edge-tts](https://github.com/rany2/edge-tts): API Python cho Microsoft Edge TTS.

## Lưu ý

- File PDF nên là dạng text-based (có thể bôi đen chữ được). File dạng ảnh (scan) sẽ không hoạt động tốt nếu không có lớp text ẩn (OCR).
- Tốc độ chuyển đổi phụ thuộc vào kết nối mạng (do gọi API Edge TTS).
