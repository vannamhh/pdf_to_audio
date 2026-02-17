# AI Audio Book Converter

Ứng dụng Streamlit giúp chuyển đổi sách điện tử (PDF) thành sách nói (Audiobook - MP3) sử dụng công nghệ Text-to-Speech (TTS) chất lượng cao từ nhiều nguồn (Edge TTS, Google Cloud TTS, OpenAI TTS).

## Tính năng nổi bật

- **Chuyển đổi PDF sang Text**: Sử dụng `pdfplumber` để trích xuất văn bản, hỗ trợ loại bỏ Header/Footer bằng crop-box.
- **✨ AI Biên tập viên (Mới)**:
  - Tích hợp **Google Gemini** để tự động sửa lỗi chính tả, lỗi OCR, nối từ bị ngắt dòng.
  - Xử lý thông minh với tài liệu chuyên ngành (Y tế, Kỹ thuật...).
- **Đa nguồn giọng đọc (Multi-Provider TTS)**:
  - **Edge TTS**: Miễn phí, giọng đọc tự nhiên.
  - **Google Cloud TTS**: Giọng đọc chuẩn, chất lượng cao (cần JSON Key).
  - **OpenAI TTS**: Giọng đọc cực kỳ tự nhiên và cảm xúc (cần API Key).
- **Quản lý & Bảo mật API Key**:
  - Hỗ trợ file `.env` để lưu trữ API Key an toàn, không cần nhập lại mỗi lần sử dụng.
- **Chỉnh sửa & Xuất văn bản**:
  - Sửa trực tiếp nội dung từng đoạn trước khi tạo audio.
  - Tải xuống hoặc lưu file text đã chỉnh sửa (`.txt`) để sử dụng lại.
- **Quản lý file đầu ra**:
  - Chia nhỏ file audio theo từng phần (Chunking).
  - Hỗ trợ **Resume** (tiếp tục tạo) và **Retry** (thử lại).
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

4. **Cấu hình API Key (Tùy chọn):**
   - Copy file `.env.example` thành `.env`.
   - Điền API Key của bạn vào file `.env` nếu muốn dùng tính năng nâng cao (Gemini, OpenAI, Google TTS).
   - *Lưu ý: File `.env` đã được gitignore bảo vệ.*

## Sử dụng

1. **Chạy ứng dụng:**
   ```bash
   streamlit run app.py
   ```

2. **Quy trình xử lý:**
   - **Bước 1**: Upload file PDF.
   - **Bước 2 (Editor)**:
     - Xem và sửa văn bản.
     - Bấm **"✨ AI sửa trang này"** hoặc **"✨ AI sửa toàn bộ"** để tự động sửa lỗi.
     - Bấm **"💾 Xuất văn bản"** để tải về file text hoàn chỉnh.
   - **Bước 3 (TTS Config)**: Chọn nguồn giọng đọc (Edge / Google / OpenAI) và cấu hình giọng.
   - **Bước 4 (Processing)**: Bấm **Tạo Audio** và tải xuống kết quả.

## Cấu trúc thư mục

```
convert_audio/
├── app.py                # Mã nguồn chính
├── requirements.txt      # Danh sách thư viện
├── output/               # Chứa file MP3 đầu ra (được gitignore)
├── .env                  # Lưu API Key (được gitignore)
├── usage_log.json        # Log hạn ngạch Google TTS Free Tier
└── ...
```

## Thư viện chính

- [Streamlit](https://streamlit.io/): Framework UI.
- [pdfplumber](https://github.com/jsvine/pdfplumber): Trích xuất PDF.
- [edge-tts](https://github.com/rany2/edge-tts): Microsoft Edge TTS (Free).
- [google-genai](https://pypi.org/project/google-genai/): Google Gemini SDK (Mới).
- [google-cloud-texttospeech](https://cloud.google.com/text-to-speech): Google TTS API.
- [openai](https://github.com/openai/openai-python): OpenAI API.

## Lưu ý

- File PDF nên là dạng text-based. File scan ảnh cần OCR trước.
- **Google Cloud TTS Free Tier**: 1 triệu ký tự/tháng. App có bộ đếm quota tích hợp để theo dõi.
