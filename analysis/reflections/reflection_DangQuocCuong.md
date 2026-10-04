# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đặng Quốc Cường  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | "Threshold 0.85 so sánh cosine similarity giữa các câu liên tiếp; ngắt đoạn khi có sự chuyển ý rõ rệt, bảo toàn tính liền mạch ngữ nghĩa thay vì cắt thô theo ký tự cố định." |
| Hierarchical & Structure-Aware Chunking | M1 | `chunk_hierarchical()`, `chunk_structure_aware()` | "Phân cấp cha (2048 chars) - con (256 chars) cho phép search chính xác trên chunk con và trả context rộng từ chunk cha cho LLM. Structure-aware phân tách theo Markdown heading `#`, `##` giữ nguyên tính toàn vẹn của bảng biểu và danh sách." |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "BM25 bắt chính xác từ khóa, số hiệu văn bản (kết hợp `segment_vietnamese` của `underthesea` và thay thế `_` thành space). Dense Search (Qdrant + `bge-m3`) bắt ngữ nghĩa. RRF kết hợp thứ hạng mượt mà với hằng số `k=60` mà không cần chuẩn hóa scale điểm." |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | "Dùng `BAAI/bge-reranker-v2-m3` chấm điểm tương quan trực tiếp giữa cặp `(query, document)`. Lọc từ 20 candidate xuống top 3 chất lượng nhất, giải quyết triệt để nhược điểm của Bi-encoder đối với các câu hỏi phủ định và ngoại lệ." |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | "Đánh giá toàn diện 4 khía cạnh: Faithfulness (độ trung thực, chống ảo giác), Answer Relevancy (độ khớp câu hỏi), Context Precision (đoạn liên quan xếp đầu), Context Recall (độ bao phủ đầy đủ thông tin chuẩn)." |
| Contextual embeddings & Enrichment | M5 | `contextual_prepend()`, `_enrich_single_call()` | "Kỹ thuật Contextual Prepend của Anthropic gắn bối cảnh tóm tắt vị trí tài liệu vào đầu chunk, kết hợp HyQA (sinh câu hỏi giả định) và Auto Metadata. Tối ưu gom 4 tác vụ vào 1 LLM call để tiết kiệm 75% chi phí API và latency." |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `ImportError: cannot import name 'underthesea' / 'flashrank'` do chạy nhầm Python interpreter toàn cục thay vì virtual environment `.venv`.
  - Deprecation warning trong RAGAS/LangChain: `LangChainDeprecationWarning: The class 'HuggingFaceEmbeddings' was deprecated in LangChain 0.2.2...`.
  - Quá tải thời gian tải trọng số mô hình `BAAI/bge-reranker-v2-m3` (~2.2GB) mỗi khi khởi tạo class `CrossEncoderReranker` mới trong từng test case.
  - Xử lý tiếng Việt trong BM25: `underthesea.word_tokenize` sinh ra các token nối dấu `_` (như `nghỉ_phép`), gây lệch từ khóa so với query thông thường của người dùng gõ có dấu cách.

- **Nguyên nhân gốc rễ & Cách debug:**
  - *Môi trường thực thi:* Luôn chỉ định đường dẫn cụ thể `.\.venv\Scripts\python.exe` và `.\.venv\Scripts\pytest.exe`.
  - *Tải model Cross-Encoder:* Áp dụng mẫu thiết kế module-level caching singleton `_shared_cross_encoder`. Khởi tạo một lần duy nhất và tái sử dụng qua các lần gọi, giảm thời gian chạy test suite từ >75s xuống chỉ còn ~16s.
  - *Tách từ tiếng Việt:* Viết hàm `segment_vietnamese()` chuẩn hóa `w.replace("_", " ")` để các cụm từ ghép được tách thành các token đồng nhất với query của người dùng.
  - *Fallback an toàn:* Xây dựng lớp phòng thủ try/except và in-memory vector database fallback khi Qdrant Docker chưa sẵn sàng hoặc OpenRouter gặp sự cố mạng.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - *Cơ chế RRF (Reciprocal Rank Fusion):* Hiểu rõ tại sao RRF vượt trội hơn Linear Score Combination (không bị ảnh hưởng bởi sự khác biệt thang đo điểm giữa BM25 unbounded và Cosine similarity [-1, 1]).
  - *Kỹ thuật Contextual Prepend:* Hiểu cơ chế giảm retrieval loss lên đến 49% của Anthropic bằng cách đưa ngữ cảnh tài liệu gốc vào từng chunk con trước khi embedding.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý AI Tra cứu Quy chế & Chính sách Doanh nghiệp (Enterprise Policy Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG cơ bản dùng `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50), tìm kiếm thuần vector trên OpenAI `text-embedding-3-small`, không có reranker và không có enrichment.
- **Vấn đề / Bottlenecks đang gặp:**
  - Các bảng biểu phụ cấp, thang lương bị cắt vỡ giữa chừng dẫn đến tính sai mức tiền.
  - Tìm kiếm các mã văn bản, số ngày nghỉ cụ thể bị trượt do dense embedding thiên về ngữ nghĩa chung chung.
  - Câu trả lời đôi khi bị hallucination khi các chính sách cũ (v2023) và chính sách mới (v2024) có nội dung xung đột nhau.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng **Structure-Aware Chunking** theo phân cấp Markdown/Heading đối với các văn bản quy định, kết hợp **Hierarchical Chunking** (Parent 2048 chars, Child 256 chars) cho các tài liệu sổ tay nhân viên dạng dài.
2. **Search retrieval:** Triển khai **Hybrid Search (BM25 + Dense Search BAAI/bge-m3)** với **RRF (k=60)**. BM25 bắt chính xác các từ khóa số hiệu quy định, điều khoản; Dense Search bắt các câu hỏi diễn đạt tự nhiên theo ngôn ngữ giao tiếp.
3. **Reranking:** Tích hợp **Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`)** lấy top 3 tài liệu chính xác nhất từ top 20 candidate.
4. **Evaluation:** Thiết lập bộ benchmark tự động với **RAGAS 4 metrics** (Faithfulness, Answer Relevancy, Context Precision, Context Recall) chạy trên test set 50 câu hỏi nghiệp vụ.
5. **Enrichment:** Sử dụng **Single-Call Enrichment** để gắn Contextual Prepend (tên chính sách, phiên bản v2024, ngày hiệu lực) vào đầu mỗi chunk trước khi index vào Vector DB.

#### 3. Timeline triển khai
- **Tuần 1:** Tái cấu trúc pipeline tiền xử lý dữ liệu: Triển khai Structure-Aware & Hierarchical Chunking, xây dựng module Enrichment 1-call với GPT-4o-mini.
- **Tuần 2:** Triển khai Hybrid Search (BM25 + Qdrant) và tích hợp Cross-Encoder Reranker.
- **Tuần 3:** Tự động hóa đánh giá định kỳ bằng RAGAS, thiết lập Dashboard theo dõi Diagnostic Tree và tối ưu hóa System Prompt dựa trên phân tích lỗi.
