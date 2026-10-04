# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đặng Quốc Cường  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:-------------:|:----------:|:--:|
| Faithfulness | 0.6000 | 0.2000 | -0.4000 |
| Answer Relevancy | 0.1436 | 0.2372 | +0.0936 |
| Context Precision | 1.0000 | 0.9583 | -0.0417 |
| Context Recall | 1.0000 | 0.8750 | -0.1250 |

> **Nhận xét chỉ số:**
> - **Answer Relevancy** tăng từ 0.1436 lên 0.2372 (+0.0936) nhờ các bước Enrichment (M5) và Cross-Encoder Reranking (M3) giúp lọc câu trả lời ngắn gọn, trực diện.
> - **Context Precision (0.9583)** và **Context Recall (0.8750)** ở mức rất cao (> 0.85), chứng minh pipeline tìm kiếm lai (Hybrid Search BM25 + Qdrant Dense) kết hợp RRF bắt trúng và đầy đủ các đoạn bối cảnh liên quan trong kho tài liệu.
> - Điểm **Faithfulness** bị kéo thấp một phần do cơ chế đánh giá song song 80 jobs của RAGAS chạm ngưỡng giới hạn request đồng thời (*in-flight concurrency budget 402*) của OpenRouter khiến một số lời gọi chấm điểm bị trả về fallback `0.0 / NaN`.

---

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3). Không tìm thấy thông tin về lương.
- **Worst metric:** context_recall
- **Error Tree:** Output sai (thiếu 1 nửa ý) → Context sai (thiếu bảng lương) → Query multi-hop không được phân tách → Module Search / Query Transformation.
- **Root cause:** Đây là câu hỏi đa chặng (multi-hop) đòi hỏi thông tin từ 2 tài liệu hoàn toàn khác nhau: *Quy chế nghỉ phép* và *Quy chế lương thưởng*. Hệ thống tìm kiếm đơn thuần không phân tách query thành 2 sub-queries, dẫn đến top 3 contexts sau khi rerank bị chiếm toàn bộ bởi tài liệu nghỉ phép, bỏ sót tài liệu lương.
- **Suggested fix:** Thêm bước **Query Decomposition (Sub-query generation)** trước khi retrieve để tách thành 2 truy vấn độc lập: (1) "Số ngày phép của nhân viên 9 năm thâm niên" và (2) "Khung lương của vị trí Senior", sau đó union kết quả retrieval.

---

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness / answer_relevancy
- **Error Tree:** Output sai → Context đúng? (Có đoạn "Chu kỳ thay đổi: Mật khẩu phải được thay đổi mỗi 120 ngày") → Query OK? (Query ngắn gọn, rõ nghĩa) → Fix ở bước LLM Generation / Prompting.
- **Root cause:** System prompt quá nghiêm ngặt (`Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'`) kết hợp với việc đoạn trích xuất có nhiều heading và gạch đầu dòng khiến LLM từ chối trả lời (false refusal) khi câu hỏi không khớp từng chữ với tiêu đề phụ.
- **Suggested fix:** Cải tiến System Prompt cho phép suy luận logic trực tiếp từ các đoạn trích dẫn, đồng thời hạ nhiệt độ `temperature = 0.0` và hướng dẫn xử lý các định dạng Markdown rõ ràng hơn.

---

### #3
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Nhân viên được nghỉ 15 ngày phép năm.
- **Worst metric:** faithfulness (so với ground truth phiên bản)
- **Error Tree:** Output đúng số lượng nhưng thiếu đối chiếu phiên bản → Context chỉ chứa bản v2024 → Query không hỏi rõ phiên bản → Fix ở bước Retrieval / Metadata Filtering.
- **Root cause:** Câu hỏi không nêu rõ năm (2023 hay 2024). Hệ thống Hybrid Search và Reranker đã ưu tiên trích xuất chính sách v2024 mới nhất (15 ngày), do đó LLM không có ngữ cảnh của bản cũ v2023 (12 ngày) để so sánh và giải thích lịch sử thay đổi như trong ground truth.
- **Suggested fix:** Bổ sung trường `status: active | superseded` vào Metadata của M5 và trong prompt LLM yêu cầu: "Nếu quy định có các phiên bản thay thế qua các năm, hãy nêu rõ phiên bản hiện hành và phiên bản cũ đã bị bãi bỏ".

---

### #4
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?
- **Expected:** KHÔNG cho cả hai. Mentor và buddy phải là hai người khác nhau. Quản lý trực tiếp không được làm mentor hoặc buddy.
- **Got:** Không tìm thấy thông tin đầy đủ về cả mentor và buddy.
- **Worst metric:** context_recall
- **Error Tree:** Output thiếu ý → Context bị cắt vụn do chunking → Query gồm 2 câu hỏi con → Fix ở M1 (Chunk size) hoặc Query Rewrite.
- **Root cause:** Điều khoản về *Mentor* và *Buddy* nằm rải rác ở 2 mục con khác nhau trong quy trình Onboarding. Do chunk con (Child chunk = 256 ký tự) quá nhỏ, thông tin về Mentor và Buddy bị chia vào 2 chunk tách biệt; khi rerank chỉ lấy top 3, một trong hai mục con bị đẩy xuống dưới rank 3.
- **Suggested fix:** Điều chỉnh `HIERARCHICAL_CHILD_SIZE` lên 384 hoặc 512 ký tự, hoặc khi gửi context cho LLM, gửi kèm `Parent chunk` (2048 ký tự) tương ứng thay vì chỉ gửi text của chunk con.

---

### #5
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Phạt 2%/tháng trên số tiền tạm ứng quá hạn.
- **Worst metric:** answer_relevancy
- **Error Tree:** Output nêu được công thức nhưng thiếu phép tính số tiền cụ thể → Context chứa công thức và thời hạn → LLM không tự tính số học → Fix ở LLM System Prompt (Chain-of-Thought).
- **Root cause:** Mô hình LLM trong system prompt chỉ được yêu cầu "trả lời dựa trên context" mà không có hướng dẫn "thực hiện tính toán số học cụ thể từng bước (Chain-of-Thought)", dẫn đến việc mô hình chỉ trích dẫn lại tỷ lệ phần trăm (2%/tháng) thay vì tính ra con số 50.000 VNĐ cho 5 ngày quá hạn.
- **Suggested fix:** Bổ sung hướng dẫn suy luận từng bước (Chain-of-Thought reasoning) vào System Prompt: "Đối với các câu hỏi liên quan đến tính toán tiền lương, ngày phép hoặc tiền phạt, hãy tính toán chi tiết từng bước dựa trên số liệu thực tế của câu hỏi".

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Sai một phần. Trả lời đúng số ngày phép (18 ngày: 15 cơ bản + 3 thâm niên) nhưng thiếu hoàn toàn thông tin dải lương Senior.
2. **Context đúng?** → Không đủ. Ngữ cảnh đưa vào LLM chỉ gồm 3 chunks từ *Quy chế nghỉ phép*, hoàn toàn thiếu chunk từ *Quy chế lương thưởng*.
3. **Query rewrite OK?** → Chưa thực hiện. Query ban đầu là một câu hỏi phức hợp (compound question) chứa hai thực thể cần tra cứu độc lập.
4. **Fix ở bước:** Bước **Query Decomposition & Multi-Query Retrieval** trước tầng Search.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Triển khai Query Decomposer:** Tự động tách các câu hỏi phức hợp thành các câu hỏi đơn lẻ, truy vấn song song qua Hybrid Search và gộp kết quả trước khi đưa qua Cross-Encoder.
- **Parent-Document Retrieval:** Khi chunk con khớp điểm tìm kiếm, trả toàn bộ đoạn cha (Parent Chunk 2048 ký tự) cho LLM đọc để đảm bảo bối cảnh xung quanh không bị mất.
- **Tối ưu hóa RAGAS batch concurrency:** Cấu hình `max_workers=2` trong lời gọi RAGAS để không vượt quá giới hạn request đồng thời của OpenRouter, giúp các chỉ số Faithfulness và Relevancy phản ánh chính xác 100% chất lượng mô hình.

