import pdfplumber
import os
import re

def clean_basic(text):
    """Làm sạch cơ bản: Nối dòng bị ngắt, xóa khoảng trắng thừa"""
    if not text: return ""
    # Nối các từ bị ngắt dòng (ví dụ: "thông -\nbáo" -> "thông báo")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Xóa ký tự lạ không đọc được (giữ lại tiếng Việt và dấu câu cơ bản)
    # text = re.sub(r'[^\w\s\.,\?\!\-\(\)đĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]', '', text)
    return text

def pdf_to_text(pdf_path):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{base_name}_raw.txt"
    
    print(f"[-] Đang xử lý: {pdf_path}")
    full_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            print(f"    + Tìm thấy {total} trang.")
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Làm sạch cơ bản từng trang
                    cleaned = clean_basic(text)
                    full_text.append(cleaned)
                
                if (i+1) % 10 == 0:
                    print(f"    -> Đã quét {i+1}/{total} trang...")
        
        # Ghi ra file
        final_content = "\n\n".join(full_text)
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_content)
            
        print(f"[OK] Đã xuất xong file: {output_filename}")
        print(f"     -> LỜI KHUYÊN: Hãy mở file này lên và xóa các phần Header/Footer thừa trước khi chạy Giai đoạn 2.")
        
    except Exception as e:
        print(f"[Lỗi] Không đọc được file {pdf_path}: {e}")

def main():
    # Quét tất cả file PDF trong thư mục hiện tại
    files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if not files:
        print("Không tìm thấy file PDF nào.")
        return
        
    for pdf in files:
        pdf_to_text(pdf)

if __name__ == "__main__":
    main()