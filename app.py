"""
AI Audio Book Converter — v4 (Multi-Provider TTS)
==================================================
Streamlit app chuyển đổi file PDF/DOCX/TXT → MP3 audiobook.

v4 Cải tiến:
- Multi-Provider TTS Architecture (Strategy Pattern).
  + Edge TTS  (Miễn phí — default)
  + Google Cloud TTS (cần Service Account JSON)
  + OpenAI TTS (cần API Key)
- Dynamic Sidebar UI theo provider được chọn.
- Giữ nguyên: Resume, Retry, Stop/Pause, Partial Download, Pagination Editor.

Chạy:  streamlit run app.py
"""

import asyncio
import io
import json
import os
import re
import tempfile
import time
import hashlib
import zipfile

from dotenv import load_dotenv
from abc import ABC, abstractmethod

import edge_tts
import pdfplumber
import streamlit as st
from docx import Document

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

PROVIDER_EDGE = "Edge TTS (Miễn phí)"
PROVIDER_GOOGLE = "Google Cloud TTS"
PROVIDER_OPENAI = "OpenAI TTS"
PROVIDER_LIST = [PROVIDER_EDGE, PROVIDER_GOOGLE, PROVIDER_OPENAI]

# --- Edge TTS voices ---
EDGE_VOICE_OPTIONS = {
    "🇻🇳 Hoài My (Nữ - VN)": "vi-VN-HoaiMyNeural",
    "🇻🇳 Nam Minh (Nam - VN)": "vi-VN-NamMinhNeural",
    "🇺🇸 Aria (Female - US)": "en-US-AriaNeural",
    "🇺🇸 Guy (Male - US)": "en-US-GuyNeural",
    "🇬🇧 Sonia (Female - UK)": "en-GB-SoniaNeural",
}

# --- Google Cloud TTS voices ---
GOOGLE_VOICE_OPTIONS = {
    "vi-VN-Neural2-A (Female)": {"name": "vi-VN-Neural2-A", "language_code": "vi-VN", "ssml_gender": "FEMALE"},
    "vi-VN-Neural2-D (Male)": {"name": "vi-VN-Neural2-D", "language_code": "vi-VN", "ssml_gender": "MALE"},
    "vi-VN-Wavenet-A (Female)": {"name": "vi-VN-Wavenet-A", "language_code": "vi-VN", "ssml_gender": "FEMALE"},
    "vi-VN-Wavenet-C (Male)": {"name": "vi-VN-Wavenet-C", "language_code": "vi-VN", "ssml_gender": "MALE"},
    "vi-VN-Wavenet-D (Male)": {"name": "vi-VN-Wavenet-D", "language_code": "vi-VN", "ssml_gender": "MALE"},
    "en-US-Neural2-A (Female)": {"name": "en-US-Neural2-A", "language_code": "en-US", "ssml_gender": "FEMALE"},
    "en-US-Neural2-D (Male)": {"name": "en-US-Neural2-D", "language_code": "en-US", "ssml_gender": "MALE"},
}

# --- OpenAI TTS voices & models ---
OPENAI_VOICE_OPTIONS = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_TTS_MODELS = ["tts-1", "tts-1-hd"]

DEFAULT_CHUNK_SIZE = 3000
DEFAULT_MARGIN_PX = 50
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
GOOGLE_FREE_TIER_LIMIT = 1_000_000  # 1 triệu ký tự / tháng
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
USAGE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage_log.json")
_DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Load .env từ thư mục project
load_dotenv(_DOTENV_PATH)

# Regex: dấu chấm kết thúc câu thật (tránh viết tắt)
_SENTENCE_END_RE = re.compile(
    r'(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!St)(?<!vs)'
    r'(?<!Tp)(?<!PGS)(?<!TS)(?<!GS)(?<!Ths)(?<!KS)'
    r'(?<!\d)'
    r'[.!?]["\u201D»)\]]*'
    r'(?=\s|$)',
)


# ══════════════════════════════════════════════
# QUOTA MANAGER — Google Cloud Free Tier Guard
# ══════════════════════════════════════════════


class QuotaManager:
    """
    Quản lý hạn ngạch miễn phí Google Cloud TTS (1M ký tự/tháng).

    Lưu lịch sử sử dụng vào file JSON cục bộ. Tự động reset khi sang tháng mới.
    """

    def __init__(self, log_path: str = USAGE_LOG_PATH, limit: int = GOOGLE_FREE_TIER_LIMIT):
        self._log_path = log_path
        self._limit = limit
        self._data = self._load()

    def _current_month(self) -> str:
        """Trả về tháng hiện tại dạng YYYY-MM."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m")

    def _load(self) -> dict:
        """Đọc file log. Reset nếu sang tháng mới."""
        default = {"current_month": self._current_month(), "used_chars": 0}
        if not os.path.exists(self._log_path):
            return default
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Reset nếu sang tháng mới
            if data.get("current_month") != self._current_month():
                return default
            return data
        except (json.JSONDecodeError, KeyError, OSError):
            return default

    def _save(self) -> None:
        """Ghi data ra file JSON."""
        try:
            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"⚠️ Không thể lưu usage log: {e}")

    @property
    def used_chars(self) -> int:
        return self._data.get("used_chars", 0)

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def remaining(self) -> int:
        return max(0, self._limit - self.used_chars)

    @property
    def usage_ratio(self) -> float:
        """Tỷ lệ sử dụng (0.0 → 1.0)."""
        return min(1.0, self.used_chars / self._limit) if self._limit > 0 else 1.0

    def check(self, char_count: int) -> tuple[bool, str]:
        """
        Kiểm tra xem có đủ quota cho char_count ký tự không.

        Returns:
            (allowed, message)
        """
        if self.used_chars + char_count > self._limit:
            return False, (
                f"🚫 Đã hết hạn mức miễn phí tháng này! "
                f"Đã dùng: {self.used_chars:,}/{self._limit:,} ký tự. "
                f"Cần thêm: {char_count:,}. "
                f"Vui lòng chuyển sang Edge TTS hoặc chờ tháng sau."
            )
        return True, "OK"

    def record(self, char_count: int) -> None:
        """Ghi nhận số ký tự đã sử dụng sau khi gọi API thành công."""
        self._data["current_month"] = self._current_month()
        self._data["used_chars"] = self.used_chars + char_count
        self._save()


# ══════════════════════════════════════════════
# TTS PROVIDER — Strategy Pattern
# ══════════════════════════════════════════════


class TTSProvider(ABC):
    """Abstract base class cho các TTS providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên hiển thị của provider."""
        ...

    @abstractmethod
    async def generate_audio(
        self, text: str, output_path: str, config: dict
    ) -> tuple[bool, str]:
        """
        Chuyển text → file MP3.

        Args:
            text: Đoạn văn bản cần đọc.
            output_path: Đường dẫn file MP3 output.
            config: Dict chứa voice, rate, và các tùy chọn riêng provider.

        Returns:
            (success, message)
        """
        ...


class EdgeTTSProvider(TTSProvider):
    """Edge TTS — Miễn phí, sử dụng Microsoft Edge Neural Voices."""

    @property
    def name(self) -> str:
        return "Edge TTS"

    async def generate_audio(
        self, text: str, output_path: str, config: dict
    ) -> tuple[bool, str]:
        voice = config["voice"]
        rate = config.get("rate", "+0%")
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, "OK"
        raise RuntimeError("File rỗng sau khi save")


class GoogleCloudTTSProvider(TTSProvider):
    """Google Cloud TTS — cần Service Account JSON key."""

    def __init__(self, credentials_path: str, quota_manager: QuotaManager | None = None):
        self._credentials_path = credentials_path
        self._quota = quota_manager or QuotaManager()

    @property
    def name(self) -> str:
        return "Google Cloud TTS"

    async def generate_audio(
        self, text: str, output_path: str, config: dict
    ) -> tuple[bool, str]:
        # Lazy import — chỉ cần khi dùng Google provider
        try:
            from google.cloud import texttospeech
        except ImportError:
            return False, (
                "❌ Thiếu thư viện google-cloud-texttospeech. "
                "Chạy: pip install google-cloud-texttospeech"
            )

        # ── Quota check ──
        char_count = len(text)
        allowed, quota_msg = self._quota.check(char_count)
        if not allowed:
            return False, quota_msg

        # Thiết lập credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_path

        voice_info = config["voice_info"]  # dict: name, language_code, ssml_gender
        language_code = voice_info["language_code"]
        voice_name = voice_info["name"]
        ssml_gender_str = voice_info["ssml_gender"]

        # Map string → enum
        gender_map = {
            "FEMALE": texttospeech.SsmlVoiceGender.FEMALE,
            "MALE": texttospeech.SsmlVoiceGender.MALE,
        }
        ssml_gender = gender_map.get(
            ssml_gender_str, texttospeech.SsmlVoiceGender.NEUTRAL
        )

        client = texttospeech.TextToSpeechClient()

        # Google Cloud TTS giới hạn 5000 bytes/request.
        # Text tiếng Việt UTF-8 ≈ 2-3 bytes/ký tự → chunk 3000 ký tự ≈ an toàn.
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            ssml_gender=ssml_gender,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=config.get("speaking_rate", 1.0),
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        with open(output_path, "wb") as f:
            f.write(response.audio_content)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Ghi nhận quota sau khi thành công
            self._quota.record(char_count)
            return True, "OK"
        raise RuntimeError("File rỗng sau khi save")


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS — cần API Key (sk-...)."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "OpenAI TTS"

    async def generate_audio(
        self, text: str, output_path: str, config: dict
    ) -> tuple[bool, str]:
        # Lazy import
        try:
            from openai import OpenAI
        except ImportError:
            return False, "❌ Thiếu thư viện openai. Chạy: pip install openai"

        model = config.get("model", "tts-1")
        voice = config.get("voice", "alloy")
        speed = config.get("speed", 1.0)

        client = OpenAI(api_key=self._api_key)

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed,
            response_format="mp3",
        )

        response.stream_to_file(output_path)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, "OK"
        raise RuntimeError("File rỗng sau khi save")


# ──────────────────────────────────────────────
# HELPERS — File & Folder
# ──────────────────────────────────────────────


def get_output_folder(file_name: str) -> str:
    """
    Tạo và trả về thư mục output riêng cho mỗi file PDF.

    Cấu trúc: output/<tên_file_không_dấu>/
    """
    base = os.path.splitext(file_name)[0]
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
                    use_crop = margin_top + margin_bottom < page.height

                    if use_crop:
                        bbox = (
                            0,
                            margin_top,
                            page.width,
                            page.height - margin_bottom,
                        )
                        cropped = page.crop(bbox)
                        text = cropped.extract_text()
                    else:
                        text = page.extract_text()

                    if text and text.strip():
                        pages_text.append(text)
                    else:
                        skipped_pages += 1

                except Exception as e:
                    skipped_pages += 1
                    print(f"⚠️ Trang {page_num}/{total_pages}: Bỏ qua do lỗi - {e}")
                    continue

            if skipped_pages > 0:
                st.warning(
                    f"ℹ️ Đã bỏ qua {skipped_pages}/{total_pages} trang "
                    f"(có thể là ảnh scan hoặc không có text)"
                )

    except Exception as e:
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
        Danh sách đoạn văn (split by \\n\\n).
    """
    paragraphs_list: list[str] = []
    try:
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode('latin-1')
            st.warning("⚠️ File không phải UTF-8, dùng Latin-1 encoding.")
        except Exception as e:
            st.error(f"❌ Không thể decode file **{file_name}**: `{e}`")
            return []

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
    if re.search(r"[!?;,]\s*$", s):
        return False
    if re.search(r"[a-zA-ZÀ-ỹ]\.\s*$", s) and len(s) > 60:
        return False
    alpha_chars = re.findall(r"[a-zA-ZÀ-ỹ]", s)
    if len(alpha_chars) >= 3 and s == s.upper():
        return True
    if re.match(
        r"^(Chương|CHƯƠNG|Phần|PHẦN|Bài|BÀI|Mục|MỤC|"
        r"Chapter|CHAPTER|Part|PART|Section|SECTION)\b",
        s,
    ):
        return True
    if re.match(r"^\d+(\.\d+)+\.?\s", s):
        return True
    if len(s) < 60 and len(s.split()) <= 10:
        first_alpha = re.search(r"[a-zA-ZÀ-ỹ]", s)
        if first_alpha and first_alpha.group().isupper():
            return True
    return False


def _buffer_ends_complete(buffer: str) -> bool:
    """Kiểm tra buffer kết thúc bằng câu hoàn chỉnh (dấu câu, không phải viết tắt)."""
    if not buffer:
        return True
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

        if blank_count > 0:
            if buffer and not _buffer_ends_complete(buffer):
                pass  # Cross-page/block break giữa câu → nối tiếp
            else:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append("")
            blank_count = 0

        if _is_heading_line(stripped):
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append(stripped)
            continue

        if _is_section_number(stripped):
            if buffer:
                result.append(buffer)
                buffer = ""
            buffer = stripped
            continue

        if buffer:
            buffer += " " + stripped
        else:
            buffer = stripped

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
        lines = [l for l in lines if not _is_page_number(l)]
        text = "\n".join(lines)
        text = re.sub(r"(\w+)-\n([a-zà-ỹ])", r"\1\2", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = "\n".join(l.strip() for l in text.split("\n"))
        text = text.strip()
        if text:
            cleaned_pages.append(text)

    merged = "\n\n".join(cleaned_pages)
    return _reflow_paragraphs(merged)


# ──────────────────────────────────────────────
# STEP 2b: CHUNKING
# ──────────────────────────────────────────────


def split_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    Chia text thành đoạn ≤ chunk_size ký tự.

    Quy tắc ưu tiên cắt:
    1. Double newline (\\n\\n) — ranh giới đoạn văn.
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

        para_cut = segment.rfind("\n\n")
        if para_cut > chunk_size * 0.3:
            best_cut = para_cut + 2

        if best_cut <= 0:
            for m in _SENTENCE_END_RE.finditer(segment):
                pos = m.end()
                if pos > best_cut:
                    best_cut = pos

        if best_cut <= 0:
            nl_cut = segment.rfind("\n")
            if nl_cut > 0:
                best_cut = nl_cut + 1

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
    provider: TTSProvider,
    tts_config: dict,
    max_retries: int = MAX_RETRIES,
) -> tuple[bool, str]:
    """
    Chuyển 1 chunk text → MP3 với cơ chế retry (provider-agnostic).

    Returns:
        (success: bool, message: str)
    """
    for attempt in range(1, max_retries + 1):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            success, msg = await provider.generate_audio(text, output_path, tts_config)
            if success:
                return True, f"✅ OK (attempt {attempt})"
            # Provider trả về failure message
            raise RuntimeError(msg)
        except Exception as e:
            msg = f"Attempt {attempt}/{max_retries} failed: {e}"
            if attempt < max_retries:
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            else:
                return False, f"❌ {msg}"
    return False, "❌ Unknown error"


def run_synthesis_pipeline(
    chunks: list[str],
    provider: TTSProvider,
    tts_config: dict,
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
        log_container.code("\n".join(log_lines[-20:]), language=None)

    append_log(f"🔧 Provider: {provider.name}")

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

            # ── Gọi TTS (provider-agnostic) ──
            status_text.text(f"🔊 [{provider.name}] Đang tạo phần {part_num}/{total}...")
            success, msg = loop.run_until_complete(
                synthesize_chunk_with_retry(chunk_text, filepath, provider, tts_config)
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


def _save_uploaded_json_to_temp(uploaded_file) -> str:
    """
    Lưu file JSON upload vào thư mục tạm.

    Returns:
        Đường dẫn tuyệt đối tới file tạm.
    """
    tmp_dir = os.path.join(tempfile.gettempdir(), "tts_credentials")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "google_credentials.json")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tmp_path


# ──────────────────────────────────────────────
# AI PROOFREADING — Gemini
# ──────────────────────────────────────────────

GEMINI_SYSTEM_PROMPT = (
    "Bạn là một biên tập viên tiếng Việt chuyên nghiệp. "
    "Nhiệm vụ của bạn là sửa lại văn bản sau đây được trích xuất từ OCR.\n"
    "Yêu cầu:\n"
    "- Sửa lỗi chính tả (ví dụ: 'chỉ phí' -> 'chi phí', 'phân 1' -> 'phần 1').\n"
    "- Sửa lỗi dính từ, ngắt dòng sai, ký tự rác.\n"
    "- Tuyệt đối KHÔNG thay đổi văn phong, ý nghĩa hoặc tóm tắt nội dung. "
    "Giữ nguyên cấu trúc gốc.\n"
    "- Chỉ trả về văn bản đã sửa, không thêm lời dẫn."
)

GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]


def _strip_markdown_wrapper(text: str) -> str:
    """
    Loại bỏ markdown code-block wrapper mà Gemini đôi khi thêm vào.

    Ví dụ: ```\nNội dung\n``` → Nội dung
    Cũng xử lý: ```text\nNội dung\n``` hoặc ```markdown\n...\n```
    """
    # Bỏ code-block bao ngoài (greedy, dotall)
    stripped = re.sub(
        r'^```(?:\w*)\s*\n(.*?)```\s*$',
        r'\1',
        text.strip(),
        flags=re.DOTALL,
    )
    # Bỏ dòng dẫn nhập kiểu "Dưới đây là văn bản đã sửa:" nếu có
    stripped = re.sub(
        r'^(?:Dưới đây là|Here is|Đây là).*?(?::\s*\n)',
        '',
        stripped.strip(),
        count=1,
        flags=re.IGNORECASE,
    )
    return stripped.strip()


def clean_text_with_ai(text: str, api_key: str, model_name: str = "gemini-2.0-flash") -> tuple[bool, str]:
    """
    Sử dụng Google Gemini để sửa lỗi chính tả / OCR cho đoạn text.

    Sử dụng SDK mới: google.genai (thay thế google.generativeai đã deprecated).

    Args:
        text: Văn bản cần sửa.
        api_key: Google AI API Key.
        model_name: Tên model Gemini.

    Returns:
        (success, result_text_or_error_message)
    """
    if not text.strip():
        return True, text

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return False, (
            "❌ Thiếu thư viện google-genai. "
            "Chạy: pip install google-genai"
        )

    # Tắt tất cả safety filters — cần thiết cho tài liệu chuyên ngành
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_PROMPT,
                safety_settings=safety_settings,
            ),
        )

        # Kiểm tra response bị block
        if not response.candidates:
            block_reason = getattr(
                response.prompt_feedback, 'block_reason', 'Unknown'
            )
            return False, (
                f"🚫 Gemini từ chối xử lý (block_reason: {block_reason}). "
                f"Có thể do nội dung nhạy cảm. Hãy thử lại hoặc sửa thủ công."
            )

        candidate = response.candidates[0]
        if (
            candidate.finish_reason
            and hasattr(candidate.finish_reason, 'name')
            and candidate.finish_reason.name == "SAFETY"
        ):
            return False, (
                "🚫 Gemini chặn kết quả do Safety Filter. "
                "Nội dung chunk này cần sửa thủ công."
            )

        result_text = response.text
        if not result_text or not result_text.strip():
            return False, "⚠️ Gemini trả về kết quả rỗng."

        # Strip markdown wrapper nếu Gemini thêm vào
        cleaned = _strip_markdown_wrapper(result_text)
        return True, cleaned

    except Exception as e:
        return False, f"❌ Lỗi Gemini API: {e}"


# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────


def _save_env_key(key: str, value: str) -> None:
    """
    Ghi hoặc cập nhật một key=value vào file .env.

    Nếu key đã tồn tại, sẽ cập nhật giá trị. Nếu chưa, sẽ thêm dòng mới.
    """
    lines: list[str] = []
    found = False
    if os.path.exists(_DOTENV_PATH):
        with open(_DOTENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_line = f"{key}={value}\n"
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = new_line
            found = True
            break

    if not found:
        lines.append(new_line)

    with open(_DOTENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def init_session_state():
    """Khởi tạo session_state mặc định, load keys từ .env."""
    defaults = {
        "full_text": "",
        "chunks": [],
        "output_folder": "",
        "base_name": "",
        "processing_done": False,
        "stop_requested": False,
        "current_file_name": "",
        # Provider-specific state — load từ .env nếu có
        "google_credentials_path": os.environ.get("GOOGLE_CREDENTIALS_PATH", ""),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        # AI Proofreading state
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        "gemini_model": GEMINI_MODELS[0],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar() -> tuple[TTSProvider | None, dict, int, int]:
    """
    Render sidebar cấu hình với dynamic provider selection.

    Returns:
        (provider, tts_config, chunk_size, margin_px)
        provider = None nếu credentials chưa được cung cấp.
    """
    provider: TTSProvider | None = None
    tts_config: dict = {}

    with st.sidebar:
        st.header("⚙️ Cấu hình")

        # ── Provider Selection ──
        st.subheader("🌐 Nguồn giọng đọc")
        selected_provider = st.selectbox(
            "Chọn TTS Provider",
            PROVIDER_LIST,
            index=0,
            help="Edge TTS miễn phí. Google Cloud và OpenAI cần API key/credentials.",
        )

        st.divider()

        # ═══════════════════════════════════════
        # EDGE TTS — Giữ nguyên logic cũ
        # ═══════════════════════════════════════
        if selected_provider == PROVIDER_EDGE:
            st.subheader("🎤 Giọng đọc (Edge TTS)")
            voice_label = st.selectbox(
                "Chọn giọng đọc",
                list(EDGE_VOICE_OPTIONS.keys()),
                index=0,
                help="Giọng Việt Nam phù hợp cho sách tiếng Việt.",
            )
            voice_id = EDGE_VOICE_OPTIONS[voice_label]

            st.subheader("⏩ Tốc độ đọc")
            rate_val = st.slider(
                "Tốc độ (%)", -50, 50, 0, 5,
                help="0% = bình thường.",
            )
            rate_str = format_rate(rate_val)
            st.caption(f"Rate: `{rate_str}`")

            provider = EdgeTTSProvider()
            tts_config = {"voice": voice_id, "rate": rate_str}

        # ═══════════════════════════════════════
        # GOOGLE CLOUD TTS
        # ═══════════════════════════════════════
        elif selected_provider == PROVIDER_GOOGLE:
            st.subheader("🔑 Google Cloud Credentials")

            # Auto-load từ .env nếu đã lưu trước đó
            saved_cred = st.session_state.get("google_credentials_path", "")
            if saved_cred and os.path.exists(saved_cred):
                st.success(f"✅ Credentials đã lưu: `{os.path.basename(saved_cred)}`")
            else:
                uploaded_json = st.file_uploader(
                    "Upload Service Account JSON",
                    type=["json"],
                    help="File JSON key từ Google Cloud Console (IAM → Service Accounts).",
                    key="google_json_uploader",
                )

                if uploaded_json is not None:
                    try:
                        content = uploaded_json.getvalue()
                        parsed = json.loads(content)
                        if "type" not in parsed or parsed.get("type") != "service_account":
                            st.error("❌ File JSON không phải Service Account key hợp lệ.")
                        else:
                            cred_path = _save_uploaded_json_to_temp(uploaded_json)
                            st.session_state["google_credentials_path"] = cred_path
                            _save_env_key("GOOGLE_CREDENTIALS_PATH", cred_path)
                            st.success("✅ Credentials đã tải lên và lưu.")
                    except json.JSONDecodeError:
                        st.error("❌ File không phải JSON hợp lệ.")

            st.subheader("🎤 Giọng đọc (Google Cloud)")
            google_voice_label = st.selectbox(
                "Chọn giọng Google",
                list(GOOGLE_VOICE_OPTIONS.keys()),
                index=0,
            )
            google_voice_info = GOOGLE_VOICE_OPTIONS[google_voice_label]

            st.subheader("⏩ Tốc độ đọc")
            speaking_rate = st.slider(
                "Speaking Rate", 0.5, 2.0, 1.0, 0.1,
                help="1.0 = bình thường. 0.5 = chậm. 2.0 = nhanh.",
            )

            cred_path = st.session_state.get("google_credentials_path", "")
            if cred_path and os.path.exists(cred_path):
                quota_mgr = QuotaManager()
                provider = GoogleCloudTTSProvider(
                    credentials_path=cred_path, quota_manager=quota_mgr
                )
                tts_config = {
                    "voice_info": google_voice_info,
                    "speaking_rate": speaking_rate,
                }

                # ── Quota display ──
                st.divider()
                st.subheader("📊 Hạn ngạch miễn phí")
                used = quota_mgr.used_chars
                limit = quota_mgr.limit
                pct = quota_mgr.usage_ratio
                st.progress(pct)
                if pct >= 1.0:
                    st.error(f"🚫 Đã hết hạn mức: **{used:,}** / {limit:,} ký tự")
                elif pct >= 0.8:
                    st.warning(
                        f"⚠️ Sắp hết: **{used:,}** / {limit:,} ký tự ({pct:.0%}) "
                        f"· Còn lại: **{quota_mgr.remaining:,}**"
                    )
                else:
                    st.caption(
                        f"Đã dùng: **{used:,}** / {limit:,} ký tự ({pct:.0%}) "
                        f"· Còn lại: **{quota_mgr.remaining:,}**"
                    )
            else:
                st.warning("⚠️ Hãy upload file JSON credentials để sử dụng Google Cloud TTS.")

        # ═══════════════════════════════════════
        # OPENAI TTS
        # ═══════════════════════════════════════
        elif selected_provider == PROVIDER_OPENAI:
            st.subheader("🔑 OpenAI API Key")
            saved_openai = st.session_state.get("openai_api_key", "")
            api_key = st.text_input(
                "Nhập API Key",
                type="password",
                value=saved_openai,
                placeholder="sk-...",
                help="API Key từ https://platform.openai.com/api-keys",
                key="openai_key_input",
            )

            if api_key and api_key != saved_openai:
                st.session_state["openai_api_key"] = api_key
                _save_env_key("OPENAI_API_KEY", api_key)
                st.success("✅ API Key đã lưu.")
            elif api_key:
                st.success("✅ API Key đã lưu từ lần trước.")

            st.subheader("🤖 Model")
            openai_model = st.selectbox(
                "Chọn model",
                OPENAI_TTS_MODELS,
                index=0,
                help="tts-1 = nhanh, rẻ hơn. tts-1-hd = chất lượng cao hơn.",
            )

            st.subheader("🎤 Giọng đọc (OpenAI)")
            openai_voice = st.selectbox(
                "Chọn giọng OpenAI",
                OPENAI_VOICE_OPTIONS,
                index=0,
            )

            st.subheader("⏩ Tốc độ đọc")
            openai_speed = st.slider(
                "Speed", 0.25, 4.0, 1.0, 0.25,
                help="1.0 = bình thường. Khoảng hỗ trợ: 0.25 – 4.0.",
            )

            effective_key = st.session_state.get("openai_api_key", "") or api_key
            if effective_key:
                provider = OpenAITTSProvider(api_key=effective_key)
                tts_config = {
                    "model": openai_model,
                    "voice": openai_voice,
                    "speed": openai_speed,
                }
            else:
                st.warning("⚠️ Hãy nhập OpenAI API Key để sử dụng.")

        # ── Common settings ──
        st.divider()

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

        # ── AI Proofreading config ──
        st.divider()
        st.subheader("✨ AI Biên tập viên")
        st.caption(
            "Dùng Google Gemini để tự động sửa lỗi chính tả, OCR. "
            "[Lấy API Key](https://aistudio.google.com/apikey)"
        )
        saved_gemini = st.session_state.get("gemini_api_key", "")
        gemini_key = st.text_input(
            "Google AI API Key",
            type="password",
            value=saved_gemini,
            placeholder="AIza...",
            key="gemini_key_input",
        )
        if gemini_key and gemini_key != saved_gemini:
            st.session_state["gemini_api_key"] = gemini_key
            _save_env_key("GEMINI_API_KEY", gemini_key)
            st.success("✅ API Key đã lưu.")
        elif gemini_key:
            st.caption("✅ API Key đã lưu từ lần trước.")

        gemini_model = st.selectbox(
            "Model",
            GEMINI_MODELS,
            index=0,
            help="gemini-2.0-flash: nhanh & rẻ (khuyên dùng).",
            key="gemini_model_select",
        )
        st.session_state["gemini_model"] = gemini_model

        st.divider()
        provider_label = selected_provider.split(" (")[0] if provider else "—"
        st.caption(f"🛠️ Powered by pdfplumber + {provider_label}")

    return provider, tts_config, chunk_size, margin_px


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
        st.session_state["full_text"] = ""
        st.session_state["chunks"] = []
        st.session_state["current_file_name"] = ""
        return False

    if uploaded.name != st.session_state.get("current_file_name", ""):
        st.session_state["current_file_name"] = uploaded.name
        st.session_state["full_text"] = ""
        st.session_state["chunks"] = []
        st.session_state["processing_done"] = False
        st.session_state["stop_requested"] = False

    if not st.session_state["full_text"]:
        file_ext = os.path.splitext(uploaded.name)[1].lower()
        margin_px = st.session_state.get("margin_px", DEFAULT_MARGIN_PX)

        with st.spinner("📖 Đang trích xuất văn bản..."):
            file_bytes = uploaded.read()

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

    chunks = split_into_chunks(text, chunk_size)
    st.session_state["chunks"] = chunks
    total_chunks = len(chunks)

    st.info(f"📦 Văn bản chia thành **{total_chunks} phần** (chunk size: {chunk_size:,} ký tự)")

    if total_chunks == 0:
        return

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

    chunk_idx = page - 1
    widget_key = f"chunk_editor_{page}"

    # ── Pending AI result: áp dụng TRƯỚC khi widget được tạo ──
    # Streamlit chỉ cho phép set session_state[widget_key] TRƯỚC khi widget render.
    pending = st.session_state.pop("_ai_pending_chunks", None)
    if pending and isinstance(pending, dict):
        for p_idx_str, p_text in pending.items():
            wk = f"chunk_editor_{p_idx_str}"
            st.session_state[wk] = p_text
        st.toast(f"✅ AI đã sửa xong — nội dung đã cập nhật.", icon="✨")

    # Khởi tạo widget value nếu chưa có
    if widget_key not in st.session_state:
        st.session_state[widget_key] = chunks[chunk_idx]

    edited = st.text_area(
        f"Nội dung phần {page}",
        height=350,
        key=widget_key,
        label_visibility="collapsed",
    )

    # ── Action buttons ──
    col_save, col_ai_one, col_ai_all = st.columns([1, 1, 1])

    with col_save:
        save_clicked = st.button("💾 Lưu chỉnh sửa", key="save_edits")
    with col_ai_one:
        ai_key = st.session_state.get("gemini_api_key", "")
        ai_one_clicked = st.button(
            "✨ AI sửa trang này",
            key="ai_fix_one",
            disabled=not ai_key,
            help="Sửa lỗi chính tả / OCR cho phần hiện tại bằng Gemini." if ai_key
                 else "Cần nhập Google AI API Key ở sidebar.",
        )
    with col_ai_all:
        ai_all_clicked = st.button(
            "✨ AI sửa toàn bộ",
            key="ai_fix_all",
            disabled=not ai_key,
            help="Sửa lỗi tất cả các phần (lần lượt)." if ai_key
                 else "Cần nhập Google AI API Key ở sidebar.",
        )

    # ── Save handler ──
    if save_clicked:
        if edited != chunks[chunk_idx]:
            chunks[chunk_idx] = edited
            st.session_state["chunks"] = chunks
            st.session_state["full_text"] = "\n\n".join(chunks)
            st.success(f"✅ Đã lưu chỉnh sửa phần {page}.")
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

    # ── AI fix ONE chunk ──
    if ai_one_clicked and ai_key:
        model_name = st.session_state.get("gemini_model", GEMINI_MODELS[0])
        with st.spinner(f"✨ Gemini ({model_name}) đang sửa phần {page}..."):
            success, result = clean_text_with_ai(chunks[chunk_idx], ai_key, model_name)
        if success:
            chunks[chunk_idx] = result
            st.session_state["chunks"] = chunks
            st.session_state["full_text"] = "\n\n".join(chunks)
            # Xóa MP3 cũ vì text đã đổi
            old_mp3 = chunk_filepath(
                st.session_state["output_folder"],
                st.session_state["base_name"],
                page,
            )
            if os.path.exists(old_mp3):
                os.remove(old_mp3)
            # Lưu vào pending buffer → rerun → áp dụng TRƯỚC widget
            st.session_state["_ai_pending_chunks"] = {str(page): result}
            st.rerun()
        else:
            st.error(result)

    # ── AI fix ALL chunks ──
    if ai_all_clicked and ai_key:
        model_name = st.session_state.get("gemini_model", GEMINI_MODELS[0])
        progress = st.progress(0)
        status = st.empty()
        fixed_count = 0
        error_count = 0
        pending_updates: dict[str, str] = {}

        for idx in range(total_chunks):
            status.text(f"✨ [{model_name}] Đang sửa phần {idx + 1}/{total_chunks}...")
            success, result = clean_text_with_ai(chunks[idx], ai_key, model_name)
            if success:
                if result != chunks[idx]:
                    chunks[idx] = result
                    fixed_count += 1
                    pending_updates[str(idx + 1)] = result
                    # Xóa MP3 cũ
                    old_mp3 = chunk_filepath(
                        st.session_state["output_folder"],
                        st.session_state["base_name"],
                        idx + 1,
                    )
                    if os.path.exists(old_mp3):
                        os.remove(old_mp3)
            else:
                error_count += 1
                st.warning(f"⚠️ Phần {idx + 1}: {result}")
            progress.progress((idx + 1) / total_chunks)

        st.session_state["chunks"] = chunks
        st.session_state["full_text"] = "\n\n".join(chunks)
        status.text(
            f"✅ Hoàn tất! Đã sửa {fixed_count}/{total_chunks} phần"
            + (f", {error_count} lỗi." if error_count else ".")
        )
        if fixed_count > 0:
            # Lưu vào pending buffer → rerun → áp dụng TRƯỚC widget
            st.session_state["_ai_pending_chunks"] = pending_updates
            st.rerun()

    # ── Export văn bản đã chỉnh sửa ──
    st.divider()
    st.markdown("#### 💾 Xuất văn bản")

    current_text = st.session_state.get("full_text", "")
    if current_text:
        file_name = st.session_state.get("current_file_name", "document")
        base_name = os.path.splitext(file_name)[0]
        txt_filename = f"{base_name}_corrected.txt"

        col_dl, col_save = st.columns(2)

        with col_dl:
            st.download_button(
                "⬇️ Download file .txt",
                data=current_text.encode("utf-8"),
                file_name=txt_filename,
                mime="text/plain",
                key="download_corrected_text",
            )

        with col_save:
            if st.button("📂 Lưu vào thư mục output", key="save_text_to_folder"):
                output_folder = st.session_state.get("output_folder", "")
                if not output_folder:
                    # Tạo output folder nếu chưa có
                    output_folder = os.path.join(OUTPUT_ROOT, base_name)
                    os.makedirs(output_folder, exist_ok=True)
                    st.session_state["output_folder"] = output_folder

                save_path = os.path.join(output_folder, txt_filename)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(current_text)
                st.success(f"✅ Đã lưu: `{save_path}`")


def render_step4_processing(provider: TTSProvider | None, tts_config: dict):
    """Step 3: TTS Processing với Progress, Resume, Stop."""
    st.markdown("### 🔊 Bước 3 — Tạo Audio")

    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.warning("⚠️ Chưa có chunks. Hãy hoàn thành Bước 1 & 2.")
        return

    # Kiểm tra provider sẵn sàng
    if provider is None:
        st.error("❌ Chưa cấu hình TTS Provider. Hãy kiểm tra sidebar.")
        return

    output_folder = st.session_state["output_folder"]
    base_name = st.session_state["base_name"]
    total = len(chunks)

    existing = list_existing_mp3s(output_folder)
    existing_count = len(existing)

    if existing_count > 0:
        st.info(
            f"📁 Đã có **{existing_count}/{total}** file MP3 trong "
            f"`{os.path.basename(output_folder)}/`. "
            f"Các phần này sẽ được **bỏ qua** khi tạo audio."
        )

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
            provider=provider,
            tts_config=tts_config,
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
    """Step 4: Download Zone — luôn hiển thị nếu có ≥ 1 file MP3."""
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
        "Chuyển đổi **PDF / DOCX / TXT** → **MP3 Audiobook** với giọng AI chất lượng cao. "
        "Hỗ trợ **Resume**, **Pause**, và **Partial Download**."
    )
    st.divider()

    # Khởi tạo state
    init_session_state()

    # Sidebar — Dynamic provider selection
    provider, tts_config, chunk_size, margin_px = render_sidebar()
    st.session_state["margin_px"] = margin_px

    # ── Step 1: Upload & Extract ──
    has_text = render_step1_upload()
    if not has_text:
        return

    st.divider()

    # ── Step 2: Edit (Pagination) ──
    render_step2_editor(chunk_size)

    st.divider()

    # ── Step 3: Processing ──
    render_step4_processing(provider, tts_config)

    st.divider()

    # ── Step 4: Download ──
    render_step5_download()


if __name__ == "__main__":
    main()
