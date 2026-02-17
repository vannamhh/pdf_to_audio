"""
AI Audio Book Converter — v3 (PyMuPDF Engine)
==============================================
Streamlit app chuyển đổi file PDF → MP3 audiobook.

v3 Cải tiến:
- PyMuPDF (fitz) block-based extraction → nhận diện đoạn văn chính xác.
- Bounding-box header/footer removal (configurable margin).
- Smart de-hyphenation (chỉ nối khi từ tiếp theo viết thường).
- Paragraph-aware chunking (ưu tiên \n\n, tránh cắt viết tắt).
- Giữ nguyên: Resume, Retry, Stop/Pause, Partial Download, Editor.

Chạy:  streamlit run app.py
"""

import asyncio
import io
import os
import re
import time
import hashlib
import zipfile

import edge_tts
import pdfplumber
import streamlit as st
from docx import Document

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

VOICE_OPTIONS = {
    "🇻🇳 Hoài My (Nữ - VN)": "vi-VN-HoaiMyNeural",
    "🇻🇳 Nam Minh (Nam - VN)": "vi-VN-NamMinhNeural",
    "🇺🇸 Aria (Female - US)": "en-US-AriaNeural",
    "🇺🇸 Guy (Male - US)": "en-US-GuyNeural",
    "🇬🇧 Sonia (Female - UK)": "en-GB-SoniaNeural",
}

DEFAULT_CHUNK_SIZE = 3000
DEFAULT_MARGIN_PX = 50
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Regex: dấu chấm kết thúc câu thật (tránh viết tắt)
# Negative lookbehind cho các viết tắt phổ biến VN & EN
_SENTENCE_END_RE = re.compile(
    r'(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!St)(?<!vs)'
    r'(?<!Tp)(?<!PGS)(?<!TS)(?<!GS)(?<!Ths)(?<!KS)'
    r'(?<!\d)'
    r'[.!?]["\u201D»)\]]*'
    r'(?=\s|$)',
)

# ──────────────────────────────────────────────
# HELPERS — File & Folder
# ──────────────────────────────────────────────


def get_output_folder(file_name: str) -> str:
    """
    Tạo và trả về thư mục output riêng cho mỗi file PDF.

    Cấu trúc: output/<tên_file_không_dấu>/
    """
    base = os.path.splitext(file_name)[0]
    # Sanitize: chỉ giữ chữ cái, số, dấu gạch, khoảng trắng → gạch dưới
    safe_name = re.sub(r"[^\w\s\-]", "", base).strip()
    safe_name = re.sub(r"\s+", "_", safe_name)
    if not safe_name:
        safe_name = hashlib.md5(file_name.encode()).hexdigest()[:12]

    folder = os.path.join(OUTPUT_ROOT, safe_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def chunk_filepath(output_folder: str, base_name: str, index: int) -> str:
    """Trả về đường dẫn file MP3 cho chunk thứ index (1-indexed)."""
    return os.path.join(output_folder, f"{base_name}_Part{index:03d}.mp3")


def list_existing_mp3s(output_folder: str) -> list[str]:
    """Liệt kê tất cả file MP3 đã tồn tại trong folder, sắp xếp theo tên."""
    if not os.path.isdir(output_folder):
        return []
    files = [
        os.path.join(output_folder, f)
        for f in sorted(os.listdir(output_folder))
        if f.lower().endswith(".mp3") and os.path.getsize(os.path.join(output_folder, f)) > 0
    ]
    return files


# ──────────────────────────────────────────────
# STEP 1: EXTRACT — Trích xuất text bằng pdfplumber (crop-based)
# ──────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def extract_text_with_cache(
    file_bytes: bytes,
    file_name: str,
    margin_top: int = DEFAULT_MARGIN_PX,
    margin_bottom: int = DEFAULT_MARGIN_PX,
) -> list[str]:
    """
    Trích xuất text từ PDF bằng pdfplumber với crop bounding-box.

    Fault-tolerant: Bỏ qua các trang lỗi (ảnh scan, kích thước lạ) thay vì crash.
    Fallback: Nếu margin quá lớn, trích xuất toàn bộ trang thay vì bỏ qua.

    Args:
        file_bytes: Nội dung file PDF (bytes).
        file_name: Tên file gốc.
        margin_top: Vùng loại bỏ phía trên trang (px).
        margin_bottom: Vùng loại bỏ phía dưới trang (px).

    Returns:
        Danh sách text từng trang (đã loại header/footer).
    """
    pages_text: list[str] = []
    total_pages = 0
    skipped_pages = 0

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    # Kiểm tra margin có vượt quá kích thước trang không
                    use_crop = margin_top + margin_bottom < page.height
                    
                    if use_crop:
                        # Cắt bỏ header/footer theo margin
                        bbox = (
                            0,                                # x0 (left)
                            margin_top,                       # y0 (top, cắt header)
                            page.width,                       # x1 (right)
                            page.height - margin_bottom,      # y1 (bottom, cắt footer)
                        )
                        cropped = page.crop(bbox)
                        text = cropped.extract_text()
                    else:
                        # Fallback: margin quá lớn → lấy toàn bộ trang
                        text = page.extract_text()
                    
                    # Xử lý None/empty text
                    if text and text.strip():
                        pages_text.append(text)
                    else:
                        # Trang không có text (có thể là ảnh scan)
                        skipped_pages += 1
                        
                except Exception as e:
                    # Lỗi trang cụ thể → bỏ qua, không crash toàn bộ
                    skipped_pages += 1
                    # Log cảnh báo nhưng không hiển thị error popup
                    print(f"⚠️ Trang {page_num}/{total_pages}: Bỏ qua do lỗi - {e}")
                    continue
            
            # Thông báo tổng kết nếu có trang bị skip
            if skipped_pages > 0:
                st.warning(
                    f"ℹ️ Đã bỏ qua {skipped_pages}/{total_pages} trang "
                    f"(có thể là ảnh scan hoặc không có text)"
                )
                
    except Exception as e:
        # Lỗi nghiêm trọng (file corrupt, không mở được)
        st.error(f"❌ Lỗi khi đọc PDF **{file_name}**: `{e}`")
        return []
    
    return pages_text


@st.cache_data(show_spinner=False)
def extract_docx(file_bytes: bytes, file_name: str) -> list[str]:
    """
    Trích xuất text từ file Microsoft Word (.docx).

    Args:
        file_bytes: Nội dung file DOCX (bytes).
        file_name: Tên file gốc.

    Returns:
        Danh sách đoạn văn (paragraphs).
    """
    paragraphs_list: list[str] = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs_list.append(text)
        
        if not paragraphs_list:
            st.warning("⚠️ File DOCX không chứa văn bản.")
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc DOCX **{file_name}**: `{e}`")
    
    return paragraphs_list


@st.cache_data(show_spinner=False)
def extract_text_file(file_bytes: bytes, file_name: str) -> list[str]:
    """
    Trích xuất text từ file văn bản thuần (.txt, .md).

    Args:
        file_bytes: Nội dung file text (bytes).
        file_name: Tên file gốc.

    Returns:
        Danh sách đoạn văn (split by \n\n).
    """
    paragraphs_list: list[str] = []
    try:
        # Thử decode UTF-8 (tiêng Việt)
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Fallback: Windows-1252 hoặc Latin-1
            text = file_bytes.decode('latin-1')
            st.warning("⚠️ File không phải UTF-8, dùng Latin-1 encoding.")
        except Exception as e:
            st.error(f"❌ Không thể decode file **{file_name}**: `{e}`")
            return []
    
    # Tách theo đoạn văn (double newline)
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        cleaned = para.strip()
        if cleaned:
            paragraphs_list.append(cleaned)
    
    if not paragraphs_list:
        st.warning("⚠️ File text trống hoặc không có nội dung.")
    
    return paragraphs_list


# ──────────────────────────────────────────────
# STEP 2: CLEAN — Làm sạch văn bản
# ──────────────────────────────────────────────


def _is_page_number(text: str) -> bool:
    """
    Kiểm tra block có phải chỉ là số trang hay không.

    Nhận diện: "42", "- 7 -", "— 12 —", "page 5", "trang 10".
    """
    s = text.strip()
    if not s:
        return False
    if re.fullmatch(r"\d{1,4}", s):
        return True
    if re.fullmatch(r"[-–—]\s*\d{1,4}\s*[-–—]", s):
        return True
    if re.fullmatch(r"(?:page|trang|p\.?)\s*\d{1,4}", s, re.IGNORECASE):
        return True
    return False


def _is_section_number(line: str) -> bool:
    """
    Kiểm tra dòng bắt đầu bằng số mục/section.

    Nhận diện: "1)", "1.2.", "1.2.1.", "a)", "- item", "• item".
    """
    s = line.strip()
    if not s:
        return False
    if re.match(r"^\d+(\.\d+)+\.?\s", s):
        return True
    if re.match(r"^(\d{1,3}|[a-zA-Z]|[ivxIVX]{1,4})[.)]\s", s):
        return True
    if re.match(r"^[-•*▪◦‣►–—]\s", s):
        return True
    return False


def _is_heading_line(line: str) -> bool:
    """
    Heuristic nhận diện dòng tiêu đề.

    Tiêu đề: ngắn (< 100 ký tự), không kết thúc bằng dấu câu nội dung,
    bắt đầu bằng chữ hoa, section number, hoặc keyword đặc biệt.
    """
    s = line.strip()
    if not s or len(s) > 100:
        return False
    # Kết thúc bằng dấu câu nội dung → không phải heading
    if re.search(r"[!?;,]\s*$", s):
        return False
    if re.search(r"[a-zA-ZÀ-ỹ]\.\s*$", s) and len(s) > 60:
        return False
    # Toàn chữ hoa (>= 3 ký tự chữ)
    alpha_chars = re.findall(r"[a-zA-ZÀ-ỹ]", s)
    if len(alpha_chars) >= 3 and s == s.upper():
        return True
    # Keyword tiêu đề (VN + EN)
    if re.match(
        r"^(Chương|CHƯƠNG|Phần|PHẦN|Bài|BÀI|Mục|MỤC|"
        r"Chapter|CHAPTER|Part|PART|Section|SECTION)\b",
        s,
    ):
        return True
    # Dotted section number
    if re.match(r"^\d+(\.\d+)+\.?\s", s):
        return True
    # Ngắn + chữ hoa đầu
    if len(s) < 60 and len(s.split()) <= 10:
        first_alpha = re.search(r"[a-zA-ZÀ-ỹ]", s)
        if first_alpha and first_alpha.group().isupper():
            return True
    return False


def _buffer_ends_complete(buffer: str) -> bool:
    """Kiểm tra buffer kết thúc bằng câu hoàn chỉnh (dấu câu, không phải viết tắt)."""
    if not buffer:
        return True
    # Nếu kết thúc bằng viết tắt + dấu chấm → chưa hết câu
    if re.search(
        r'(?:Mr|Mrs|Ms|Dr|St|vs|Tp|PGS|TS|GS|Ths|KS|ThS|Q|P|Tr)\.' r'\s*$',
        buffer,
    ):
        return False
    return bool(re.search(r'[.!?:;"\u201D»)\]]\s*$', buffer))


def _reflow_paragraphs(text: str) -> str:
    """
    Lightweight reflow cho text đã trích xuất bằng PyMuPDF blocks.

    PyMuPDF blocks đã nhận diện paragraph khá tốt, nhưng vẫn có thể
    chứa hard line-breaks bên trong block. Hàm này nối các dòng bị
    ngắt giữa câu và giữ nguyên cấu trúc heading/list.
    """
    lines = text.split("\n")
    result: list[str] = []
    buffer = ""
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            blank_count += 1
            continue

        # Xử lý dòng trống tích lũy
        if blank_count > 0:
            if buffer and not _buffer_ends_complete(buffer):
                pass  # Cross-page/block break giữa câu → nối tiếp
            else:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append("")
            blank_count = 0

        # Heading → đứng riêng
        if _is_heading_line(stripped):
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append(stripped)
            continue

        # List item → bắt đầu block mới
        if _is_section_number(stripped):
            if buffer:
                result.append(buffer)
                buffer = ""
            buffer = stripped
            continue

        # Nối dòng thường
        if buffer:
            buffer += " " + stripped
        else:
            buffer = stripped

        # Kết thúc câu → flush
        if _buffer_ends_complete(buffer):
            result.append(buffer)
            buffer = ""

    if buffer:
        result.append(buffer)

    return "\n".join(result)


def clean_text(pages_text: list[str]) -> str:
    """
    Pipeline làm sạch văn bản từ PyMuPDF pages:
    1. Xóa số trang.
    2. Smart de-hyphenation (chỉ nối khi từ tiếp viết thường).
    3. Xóa ký tự control.
    4. Chuẩn hóa khoảng trắng.
    5. Reflow: nối dòng bị ngắt giữa câu.
    """
    cleaned_pages: list[str] = []

    for page in pages_text:
        lines = page.strip().split("\n")

        # Xóa dòng chỉ là số trang
        lines = [l for l in lines if not _is_page_number(l)]

        text = "\n".join(lines)

        # Smart de-hyphenation: chỉ nối nếu từ tiếp theo viết thường
        # "thông-\nbáo" → "thôngbáo" nhưng "Bắc-\nKinh" giữ nguyên
        text = re.sub(r"(\w+)-\n([a-zà-ỹ])", r"\1\2", text)

        # Xóa ký tự control (giữ newline, tab, space)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Chuẩn hóa khoảng trắng
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = "\n".join(l.strip() for l in text.split("\n"))

        text = text.strip()
        if text:
            cleaned_pages.append(text)

    # Ghép tất cả trang, sau đó reflow paragraph
    merged = "\n\n".join(cleaned_pages)
    return _reflow_paragraphs(merged)


# ──────────────────────────────────────────────
# STEP 2b: CHUNKING
# ──────────────────────────────────────────────


def split_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    Chia text thành đoạn ≤ chunk_size ký tự.

    Quy tắc ưu tiên cắt:
    1. Double newline (\n\n) — ranh giới đoạn văn.
    2. Dấu chấm câu thật (tránh viết tắt Tp., Mr., Dr..).
    3. Single newline.
    4. Hard-cut tại chunk_size.
    """
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining.strip())
            break

        segment = remaining[:chunk_size]
        best_cut = -1

        # Ưu tiên 1: Cắt tại paragraph boundary (\n\n)
        para_cut = segment.rfind("\n\n")
        if para_cut > chunk_size * 0.3:  # Chỉ dùng nếu ≥ 30% chunk
            best_cut = para_cut + 2

        # Ưu tiên 2: Cắt tại dấu câu thật (tránh viết tắt)
        if best_cut <= 0:
            # Tìm tất cả vị trí kết thúc câu thật trong segment
            for m in _SENTENCE_END_RE.finditer(segment):
                pos = m.end()
                if pos > best_cut:
                    best_cut = pos

        # Ưu tiên 3: Cắt tại newline đơn
        if best_cut <= 0:
            nl_cut = segment.rfind("\n")
            if nl_cut > 0:
                best_cut = nl_cut + 1

        # Fallback: hard-cut
        if best_cut <= 0:
            best_cut = chunk_size

        chunk = remaining[:best_cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[best_cut:]

    return chunks


# ──────────────────────────────────────────────
# STEP 3: SYNTHESIZE — TTS với Retry & Resume
# ──────────────────────────────────────────────


async def synthesize_chunk_with_retry(
    text: str,
    output_path: str,
    voice: str,
    rate: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[bool, str]:
    """
    Chuyển 1 chunk text → MP3 với cơ chế retry.

    Returns:
        (success: bool, message: str)
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Đảm bảo thư mục output tồn tại (phòng trường hợp user xóa sau khi upload)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(output_path)
            # Kiểm tra file thực sự được ghi
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True, f"✅ OK (attempt {attempt})"
            else:
                raise RuntimeError("File rỗng sau khi save")
        except Exception as e:
            msg = f"Attempt {attempt}/{max_retries} failed: {e}"
            if attempt < max_retries:
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            else:
                return False, f"❌ {msg}"
    return False, "❌ Unknown error"


def run_synthesis_pipeline(
    chunks: list[str],
    voice: str,
    rate: str,
    output_folder: str,
    base_name: str,
    progress_bar,
    status_text,
    log_container,
) -> list[str]:
    """
    Chạy pipeline TTS cho tất cả chunks.

    Đặc điểm:
    - Resume: bỏ qua chunk đã có file MP3 hợp lệ.
    - Stop: kiểm tra st.session_state["stop_requested"] mỗi vòng lặp.
    - Real-time log: ghi log vào log_container.
    - Lưu ngay: mỗi chunk xong → file trên đĩa.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    total = len(chunks)
    completed_files: list[str] = []
    log_lines: list[str] = []

    def append_log(msg: str):
        """Thêm 1 dòng log và cập nhật UI."""
        log_lines.append(msg)
        # Hiển thị 20 dòng gần nhất
        log_container.code("\n".join(log_lines[-20:]), language=None)

    try:
        for i, chunk_text in enumerate(chunks):
            part_num = i + 1
            filepath = chunk_filepath(output_folder, base_name, part_num)

            # ── Check STOP flag ──
            if st.session_state.get("stop_requested", False):
                append_log(f"⏸️ Dừng tại phần {part_num}/{total} theo yêu cầu.")
                status_text.text(f"⏸️ Đã dừng. Hoàn thành {len(completed_files)}/{total} phần.")
                break

            # ── RESUME: Skip nếu file đã tồn tại ──
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                completed_files.append(filepath)
                append_log(f"⏭️ Part {part_num:03d}/{total} — đã tồn tại, bỏ qua.")
                progress_bar.progress(part_num / total)
                continue

            # ── Gọi TTS ──
            status_text.text(f"🔊 Đang tạo phần {part_num}/{total}...")
            success, msg = loop.run_until_complete(
                synthesize_chunk_with_retry(chunk_text, filepath, voice, rate)
            )

            if success:
                completed_files.append(filepath)
                append_log(f"🔊 Part {part_num:03d}/{total} — {msg}")
            else:
                append_log(f"⚠️ Part {part_num:03d}/{total} — {msg}")

            progress_bar.progress(part_num / total)

    except Exception as e:
        append_log(f"💥 Lỗi hệ thống: {e}")
        status_text.text(f"❌ Lỗi hệ thống. Đã lưu {len(completed_files)} phần.")
    finally:
        loop.close()

    return completed_files


# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────


def create_zip_from_files(mp3_files: list[str]) -> bytes:
    """Đóng gói danh sách file MP3 vào ZIP (in-memory)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in mp3_files:
            zf.write(path, os.path.basename(path))
    buf.seek(0)
    return buf.read()


def format_rate(val: int) -> str:
    """Chuyển slider value → chuỗi rate cho edge-tts (vd: +20%, -10%)."""
    return f"+{val}%" if val >= 0 else f"{val}%"


# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────


def init_session_state():
    """Khởi tạo session_state mặc định."""
    defaults = {
        "full_text": "",
        "chunks": [],
        "output_folder": "",
        "base_name": "",
        "processing_done": False,
        "stop_requested": False,
        "current_file_name": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar() -> tuple[str, str, int, int]:
    """Render sidebar cấu hình. Trả về (voice_id, rate_str, chunk_size, margin_px)."""
    with st.sidebar:
        st.header("⚙️ Cấu hình")

        st.subheader("🎤 Giọng đọc")
        voice_label = st.selectbox(
            "Chọn giọng đọc",
            list(VOICE_OPTIONS.keys()),
            index=0,
            help="Giọng Việt Nam phù hợp cho sách tiếng Việt.",
        )
        voice_id = VOICE_OPTIONS[voice_label]

        st.subheader("⏩ Tốc độ đọc")
        rate_val = st.slider(
            "Tốc độ (%)", -50, 50, 0, 5,
            help="0% = bình thường.",
        )
        rate_str = format_rate(rate_val)
        st.caption(f"Rate: `{rate_str}`")

        st.subheader("✂️ Chunk size")
        chunk_size = st.slider(
            "Ký tự / chunk", 1000, 10000, DEFAULT_CHUNK_SIZE, 500,
            help="Số ký tự tối đa mỗi đoạn gửi TTS.",
        )

        st.subheader("📐 Header/Footer margin")
        margin_px = st.slider(
            "Margin (px)", 0, 150, DEFAULT_MARGIN_PX, 10,
            help="Bỏ text trong vùng X px đầu/cuối trang (header/footer).",
        )

        st.divider()
        st.caption("🛠️ Powered by pdfplumber + Edge TTS")

    return voice_id, rate_str, chunk_size, margin_px


def render_step1_upload():
    """Step 1: Upload & Extract Document."""
    st.markdown("### 📤 Bước 1 — Upload Tài liệu")

    uploaded = st.file_uploader(
        "Chọn file PDF, DOCX, TXT, hoặc MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=False,
        help="Hỗ trợ PDF, Word, Text, và Markdown.",
    )

    if uploaded is None:
        st.info("👆 Upload file để bắt đầu.")
        # Reset state khi không có file
        st.session_state["full_text"] = ""
        st.session_state["chunks"] = []
        st.session_state["current_file_name"] = ""
        return False

    # Nếu file mới khác file cũ → reset state
    if uploaded.name != st.session_state.get("current_file_name", ""):
        st.session_state["current_file_name"] = uploaded.name
        st.session_state["full_text"] = ""
        st.session_state["chunks"] = []
        st.session_state["processing_done"] = False
        st.session_state["stop_requested"] = False

    # Extract (cached)
    if not st.session_state["full_text"]:
        file_ext = os.path.splitext(uploaded.name)[1].lower()
        margin_px = st.session_state.get("margin_px", DEFAULT_MARGIN_PX)
        
        with st.spinner("📖 Đang trích xuất văn bản..."):
            file_bytes = uploaded.read()
            
            # Dispatcher: chọn hàm trích xuất theo loại file
            if file_ext == ".pdf":
                pages = extract_text_with_cache(
                    file_bytes, uploaded.name, margin_px, margin_px,
                )
            elif file_ext == ".docx":
                pages = extract_docx(file_bytes, uploaded.name)
            elif file_ext in [".txt", ".md"]:
                pages = extract_text_file(file_bytes, uploaded.name)
            else:
                st.error(f"❌ Định dạng file không được hỗ trợ: {file_ext}")
                return False

        if not pages:
            st.warning("⚠️ Không trích xuất được văn bản từ file này.")
            return False

        cleaned = clean_text(pages)
        if not cleaned.strip():
            st.warning("⚠️ Văn bản sau khi làm sạch trống.")
            return False

        st.session_state["full_text"] = cleaned
        st.session_state["base_name"] = os.path.splitext(uploaded.name)[0]
        st.session_state["output_folder"] = get_output_folder(uploaded.name)

    # Thống kê
    text = st.session_state["full_text"]
    st.success(
        f"✅ Đã trích xuất: **{len(text):,}** ký tự · "
        f"**{len(text.split()):,}** từ"
    )
    return True


def render_step2_editor(chunk_size: int):
    """Step 2: Pagination Editor — sửa text theo từng chunk."""
    st.markdown("### ✏️ Bước 2 — Xem & Sửa Văn bản")

    text = st.session_state["full_text"]
    if not text:
        return

    # Chia chunks (hoặc dùng chunks đã lưu nếu chunk_size không đổi)
    chunks = split_into_chunks(text, chunk_size)
    st.session_state["chunks"] = chunks
    total_chunks = len(chunks)

    st.info(f"📦 Văn bản chia thành **{total_chunks} phần** (chunk size: {chunk_size:,} ký tự)")

    if total_chunks == 0:
        return

    # ── Pagination controls ──
    col_nav1, col_nav2 = st.columns([1, 3])
    with col_nav1:
        page = st.number_input(
            "Chọn phần để xem/sửa",
            min_value=1,
            max_value=total_chunks,
            value=1,
            step=1,
            key="editor_page",
        )
    with col_nav2:
        st.caption(f"Phần {page}/{total_chunks} · {len(chunks[page - 1]):,} ký tự")

    # ── Text editor cho chunk hiện tại ──
    chunk_idx = page - 1
    edited = st.text_area(
        f"Nội dung phần {page}",
        value=chunks[chunk_idx],
        height=350,
        key=f"chunk_editor_{page}",
        label_visibility="collapsed",
    )

    # ── Nút Save ──
    if st.button("💾 Lưu chỉnh sửa", key="save_edits"):
        if edited != chunks[chunk_idx]:
            chunks[chunk_idx] = edited
            st.session_state["chunks"] = chunks
            # Rebuild full_text từ chunks đã sửa
            st.session_state["full_text"] = "\n\n".join(chunks)
            st.success(f"✅ Đã lưu chỉnh sửa phần {page}.")
            # Xóa file MP3 cũ của chunk này (vì nội dung đã thay đổi)
            old_mp3 = chunk_filepath(
                st.session_state["output_folder"],
                st.session_state["base_name"],
                page,
            )
            if os.path.exists(old_mp3):
                os.remove(old_mp3)
                st.info(f"🗑️ Đã xóa file audio cũ Part {page:03d} (text đã thay đổi).")
        else:
            st.info("ℹ️ Không có thay đổi.")


def render_step4_processing(voice_id: str, rate_str: str):
    """Step 4: TTS Processing với Progress, Resume, Stop."""
    st.markdown("### 🔊 Bước 3 — Tạo Audio")

    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.warning("⚠️ Chưa có chunks. Hãy hoàn thành Bước 1 & 2.")
        return

    output_folder = st.session_state["output_folder"]
    base_name = st.session_state["base_name"]
    total = len(chunks)

    # Đếm file đã có
    existing = list_existing_mp3s(output_folder)
    existing_count = len(existing)

    if existing_count > 0:
        st.info(
            f"📁 Đã có **{existing_count}/{total}** file MP3 trong "
            f"`{os.path.basename(output_folder)}/`. "
            f"Các phần này sẽ được **bỏ qua** khi tạo audio."
        )

    # ── Buttons: Generate / Stop ──
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        generate = st.button(
            f"🎧 Tạo Audio ({total - existing_count} phần còn lại)",
            key="btn_generate",
            disabled=(existing_count == total),
        )
    with col_btn2:
        stop = st.button("⏹️ Dừng xử lý", key="btn_stop")

    if stop:
        st.session_state["stop_requested"] = True
        st.warning("⏸️ Đã gửi lệnh dừng. Tiến trình sẽ dừng sau chunk hiện tại.")

    if generate:
        st.session_state["stop_requested"] = False
        st.session_state["processing_done"] = False

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.empty()

        completed = run_synthesis_pipeline(
            chunks=chunks,
            voice=voice_id,
            rate=rate_str,
            output_folder=output_folder,
            base_name=base_name,
            progress_bar=progress_bar,
            status_text=status_text,
            log_container=log_container,
        )

        total_done = len(list_existing_mp3s(output_folder))
        if total_done == total:
            st.session_state["processing_done"] = True
            status_text.text(f"✅ Hoàn tất! {total_done}/{total} phần.")
            st.balloons()
        else:
            status_text.text(
                f"⚠️ Đã xử lý {total_done}/{total} phần. "
                f"Bấm 'Tạo Audio' để tiếp tục các phần còn lại."
            )


def render_step5_download():
    """Step 5: Download Zone — luôn hiển thị nếu có ≥ 1 file MP3."""
    st.markdown("### 📥 Bước 4 — Tải về")

    output_folder = st.session_state.get("output_folder", "")
    if not output_folder:
        return

    mp3_files = list_existing_mp3s(output_folder)
    if not mp3_files:
        st.caption("Chưa có file MP3 nào. Hãy chạy Bước 3.")
        return

    total_chunks = len(st.session_state.get("chunks", []))
    done = len(mp3_files)

    if done < total_chunks:
        st.warning(f"⚠️ Mới hoàn thành **{done}/{total_chunks}** phần.")
    else:
        st.success(f"✅ Đã hoàn thành tất cả **{done}/{total_chunks}** phần!")

    # ── Audio players (collapsed, chỉ load khi mở) ──
    st.markdown("##### 🎵 Nghe thử")
    for mp3_path in mp3_files:
        name = os.path.basename(mp3_path)
        with st.expander(f"▶️ {name}", expanded=False):
            with open(mp3_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                f"⬇️ Tải {name}",
                data=audio_bytes,
                file_name=name,
                mime="audio/mpeg",
                key=f"dl_{name}",
            )

    # ── ZIP download (luôn khả dụng) ──
    st.markdown("##### 📦 Tải tất cả (ZIP)")
    zip_bytes = create_zip_from_files(mp3_files)
    base = st.session_state.get("base_name", "audiobook")
    st.download_button(
        f"⬇️ Download ZIP ({done} phần)",
        data=zip_bytes,
        file_name=f"{base}_AudioBook.zip",
        mime="application/zip",
        key="dl_zip",
    )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────


def main():
    """Entry point."""

    st.set_page_config(
        page_title="📖 AI Audio Book Converter",
        page_icon="🎧",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📖 AI Audio Book Converter")
    st.markdown(
        "Chuyển đổi **PDF** → **MP3 Audiobook** với giọng AI chất lượng cao. "
        "Hỗ trợ **Resume**, **Pause**, và **Partial Download**."
    )
    st.divider()

    # Khởi tạo state
    init_session_state()

    # Sidebar
    voice_id, rate_str, chunk_size, margin_px = render_sidebar()
    st.session_state["margin_px"] = margin_px

    # ── Step 1: Upload & Extract ──
    has_text = render_step1_upload()
    if not has_text:
        return

    st.divider()

    # ── Step 2: Edit (Pagination) ──
    render_step2_editor(chunk_size)

    st.divider()

    # ── Step 3 (previously 4): Processing ──
    render_step4_processing(voice_id, rate_str)

    st.divider()

    # ── Step 4 (previously 5): Download ──
    render_step5_download()


if __name__ == "__main__":
    main()
