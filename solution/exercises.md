# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> _Câu trả lời của bạn_
Em nhận thấy khi càng tăng dần temperature thì nó càng trả lời dài và phong phú cũng như tính sáng tạo nhiều hơn so với temperature để thấp. Nhưng quá cao thì em thấy nó bắt đầu dần bị trả lời không còn đúng và liên quan đến câu hỏi nhiều nữa.
### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> _Câu trả lời của bạn_
Em sẽ đặt temperature=0.1 cho chatbot hỗ trợ khách hàng, vì khi hỗ trợ cho khách hàng, chatbot cần nên trả lời đúng và chính xác đối với các chính sách của cửa hàng, tránh hallucination, nhưng cũng cần có thêm 1 ít tính sáng tạo để nên ở đây em có thể kiểm soát thông qua system prompt để có thể generate văn bản mang tính gần gũi với khách hàng hơn thay vì chỉnh temperature.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> _Câu trả lời của bạn_
Đối với workload này, GPT-4o đắt hơn GPT-4o-mini khoảng 16,67 lần (105 USD so với 6,3 USD/ngày); nên dùng GPT-4o khi cần xử lý logic suy luận phức tạp như phân tích pháp lý hay debug code sâu, còn mini hoàn toàn tối ưu cho các tác vụ khối lượng lớn như trích xuất dữ liệu hoặc chatbot trả lời FAQ cơ bản.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> _Câu trả lời của bạn_
Khi đóng vai giáo viên, model đưa ra câu trả lời ngắn gọn, dùng từ vựng giản dị cùng hình ảnh ẩn dụ gần gũi như "cuốn sổ tay chung mà cả lớp cùng giữ và không ai tự ý tẩy xóa được". Ngược lại, với vai chuyên gia tài chính, phản hồi dài hơn, sử dụng dày đặc thuật ngữ kỹ thuật như "sổ cái phân tán (DLT)", "hàm băm mật mã (cryptographic hash)" và "cơ chế đồng thuận". Điều này cho thấy system prompt không chỉ định hình phong cách hành văn và độ sâu kiến thức, mà còn đóng vai trò như một bộ lọc ngữ cảnh giúp model căn chỉnh độ phức tạp phù hợp chính xác với đối tượng tiếp nhận.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> _Câu trả lời của bạn_
Hai con số chênh nhau khoảng 35-50%, vì tiếng việt thường tốn thêm token cho các kí tự có dấu khi trong cùng 1 độ dài so với tiếng anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> _Câu trả lời của bạn_
Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng (chatbot dòng lệnh, web UI) khi model cần sinh câu trả lời dài, bởi nó giúp giảm thiểu đáng kể thời gian chờ phản hồi đầu tiên (Time To First Token - TTFT), mang lại cảm giác phản hồi tức thì và không bị treo máy. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm trong backend (batch jobs, pipeline ETL), các tác vụ yêu cầu trả về định dạng có cấu trúc hoàn chỉnh như JSON để validate schema trước khi dùng, hoặc các bước trung gian cần kiểm duyệt nội dung an toàn trước khi hiển thị cho end user.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> _Câu trả lời của bạn_
Exponential backoff giúp giãn cách thời gian chờ tăng dần theo cấp số nhân ($2^n$), tạo ra khoảng lặng đủ lớn cho hệ thống máy chủ tự phục hồi sau khi nghẽn tải. Nếu hàng nghìn client cùng retry với khoảng delay cố định (ví dụ luôn chờ đúng 1 giây), các yêu cầu sẽ đồng loạt dồn ngược lại máy chủ cùng một thời điểm, gây ra hiện tượng bão yêu cầu khiến server tiếp tục sập và không bao giờ thoát khỏi trạng thái quá tải.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> _Câu trả lời của bạn_
Sysmem prompt chọn: "Bạn là trợ giảng lập trình Python thân thiện, luôn trả lời ngắn gọn dưới 3 câu và bắt buộc phản hồi bằng tiếng Việt."
Khi đưa ra yêu cầu trả lời ngắn gọn dưới 3 câu sẽ giúp tiết kiệm token và giảm latency cho mỗi request. Đồng thời giúp cho model tập trung vô vấn đề chính tránh lan man. Chỉ định ngôn ngữ bằng tiếng việt giúp nhất quán ngôn ngữ giữa các lần trả về response cho end user.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> _Câu trả lời của bạn_
Hạn chế lớn nhất: Bộ nhớ hội thoại (history) bị cắt cứng ở 3 lượt cuối (tối đa 6 message), khiến trợ lý quên sạch ngữ cảnh ban đầu khi trao đổi dài, dẫn đến việc hỏi lại những điều người dùng đã đề cập trước đó.
Cải thiện đề xuất: Thay vì cắt bỏ thẳng tay các tin nhắn cũ, triển khai kỹ thuật Context Summarization (Tóm tắt ngữ cảnh).
Cách triển khai: Khi độ dài history vượt quá ngưỡng quy định (ví dụ sau mỗi 3 lượt), kích hoạt một lời gọi ngầm dùng model nhẹ (gpt-4o-mini) để tóm tắt các điểm chính của đoạn hội thoại cũ thành một đoạn văn ngắn lưu vào trường summary. Sau đó, ghép đoạn summary này vào ngay sau System Prompt ở các lượt hỏi tiếp theo để bảo toàn ngữ cảnh dài hạn mà không làm bùng nổ số lượng token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
