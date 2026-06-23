# 🐶🐱 HỆ THỐNG NHẬN DIỆN CÁ THỂ CHÓ MÈO
### Ứng dụng pipeline Face Recognition vào Pet Identity Recognition

---

## Giới thiệu

Hầu hết các hệ thống phân loại chó/mèo truyền thống chỉ trả lời câu hỏi:

> **"Đây là chó hay mèo?"**

Hệ thống này đặt ra mục tiêu cao hơn — tương tự bài toán nhận diện khuôn mặt người:

> **"Đây là con chó/mèo nào? Tên nó là gì?"**

Thay vì phân loại theo loài, hệ thống **nhận ra từng cá thể cụ thể** dựa trên đặc trưng ngoại hình — giống như nhận diện khuôn mặt người, nhưng áp dụng cho thú cưng.

---

## Ý tưởng cốt lõi

Pipeline được mượn nguyên từ hệ thống nhận diện khuôn mặt người (Face Recognition):

```
[Face Recognition - Người]          [Pet Recognition - Thú cưng]
─────────────────────────           ──────────────────────────────
Ảnh khuôn mặt người                 Ảnh mặt/toàn thân chó/mèo
        │                                       │
        ▼                                       ▼
  Detect khuôn mặt                    Detect vùng đầu/mặt thú
  (MTCNN, RetinaFace)                 (YOLOv8-detect)
        │                                       │
        ▼                                       ▼
  CNN trích xuất                       CNN trích xuất
  embedding 512-D                      embedding 512-D
  (FaceNet, ArcFace)                   (ResNet50, EfficientNet)
        │                                       │
        ▼                                       ▼
  L2 Normalization                    L2 Normalization
        │                                       │
        ▼                                       ▼
  KNN / Cosine Similarity             KNN / Cosine Similarity
        │                                       │
        ▼                                       ▼
  "Đây là Nguyễn Văn A"              "Đây là Milu (mèo của nhà X)"
```

---

## So sánh với bài toán phân loại thông thường

| Tiêu chí | Phân loại chó/mèo (BTL gốc) | Nhận diện cá thể (hệ thống này) |
|---|---|---|
| Câu hỏi | Chó hay mèo? | Con nào? Tên gì? |
| Số lớp đầu ra | 2 (dog/cat) | N (số cá thể đã đăng ký) |
| Kỹ thuật chính | Softmax classifier | Embedding + Similarity Search |
| Thêm cá thể mới | Phải train lại toàn bộ | **Không cần train lại CNN** |
| Dữ liệu cần | 10.000+ ảnh/lớp | 10–50 ảnh/cá thể là đủ |
| Mô hình tham khảo | CNN, YOLOv8-cls | FaceNet, ArcFace (adapted) |

---

## Kiến trúc hệ thống

### Hai giai đoạn chính

#### Giai đoạn 1 — Đăng ký (Enrollment)

Thực hiện một lần cho mỗi cá thể mới:

```
Ảnh thú cưng (10–50 ảnh)
        │
        ▼
[Bước 1] Detect & Crop
  YOLOv8-detect xác định vùng mặt/đầu
  → Cắt và chuẩn hóa về 128×128
        │
        ▼
[Bước 2] Trích xuất embedding
  ResNet50 / EfficientNet (pretrained ImageNet)
  → Vector đặc trưng 512 chiều
        │
        ▼
[Bước 3] L2 Normalization
  Đưa vector về mặt cầu đơn vị
  → Cosine similarity = dot product
        │
        ▼
[Bước 4] Lưu vào Database
  {tên: "Milu", embedding: [...512 giá trị...]}
```

#### Giai đoạn 2 — Nhận diện (Recognition)

Thực hiện mỗi khi cần nhận diện:

```
Ảnh đầu vào (ảnh mới chưa biết)
        │
        ▼
  Detect → Crop → CNN → L2 Norm
  (giống hệt Enrollment)
        │
        ▼
  So sánh với toàn bộ database
  (KNN hoặc Cosine Similarity)
        │
        ▼
  Khoảng cách < ngưỡng?
    ├── Có → Trả về tên cá thể gần nhất
    └── Không → "Unknown" (chưa đăng ký)
```

---

## Tại sao dùng Embedding thay vì Softmax?

Đây là điểm khác biệt then chốt so với bài toán phân loại thông thường.

**Softmax Classifier** (như CNN trong BTL):
- Đầu ra: xác suất thuộc từng lớp cố định
- Thêm 1 cá thể mới → **phải train lại toàn bộ mô hình**
- Không phù hợp khi số lượng cá thể tăng liên tục

**Embedding + Similarity** (như Face Recognition):
- Đầu ra: vector đặc trưng trong không gian metric
- Thêm 1 cá thể mới → **chỉ cần chụp ảnh & lưu embedding**
- CNN không cần train lại, chỉ KNN cập nhật

> Đây chính là lý do hệ thống nhận diện khuôn mặt có thể mở rộng lên hàng triệu người mà không cần retrain.

---

## Thách thức riêng so với nhận diện người

| Thách thức | Nhận diện người | Nhận diện thú cưng |
|---|---|---|
| Đặc trưng nhận diện | Khuôn mặt cố định, ổn định | Lông có thể thay đổi theo mùa, tuổi |
| Góc nhìn | Thường chụp thẳng | Thú cưng ít hợp tác, góc tùy ý |
| Detect vùng quan tâm | Face detector rất tốt | Pet face detector còn hạn chế |
| Dữ liệu đăng ký | Dễ chụp ảnh người | Thú cưng khó giữ yên để chụp |
| Phân biệt cùng giống | Ít gặp | Khó: 2 mèo Anh lông ngắn rất giống nhau |

---

## Công nghệ đề xuất

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Detect vùng mặt | YOLOv8-detect | Nhanh, hỗ trợ nhiều góc độ |
| Backbone CNN | ResNet50 (pretrained) | Đặc trưng phong phú, ổn định |
| Embedding size | 512 chiều | Cân bằng giữa thông tin và tốc độ |
| So khớp | KNN (k=3) + Cosine | Đơn giản, hiệu quả với dataset nhỏ |
| Ngưỡng nhận diện | Cosine distance < 0.4 | Cần thực nghiệm để tinh chỉnh |
| Ngôn ngữ | Python | TensorFlow/Keras + Scikit-learn |

---

## Khả năng ứng dụng thực tế

- **Tìm thú cưng thất lạc**: Chụp ảnh thú cưng trên đường → so với database ảnh thú cưng bị mất
- **Quản lý trại thú cưng**: Tự động điểm danh, theo dõi sức khỏe từng cá thể
- **Cửa thông minh cho thú cưng**: Chỉ mở cửa cho đúng con của nhà
- **Ứng dụng thú cưng cá nhân**: Nhận ra "đây là Milu của tôi" trong ảnh gia đình

---

## Hướng phát triển

- Tích hợp **ArcFace Loss** thay vì Softmax để embedding có tính phân biệt cao hơn
- Xây dựng **pet face detector** chuyên biệt thay vì dùng general object detector
- Mở rộng sang **video real-time** qua webcam với YOLOv8 + tracking
- Kết hợp **đặc trưng toàn thân** (màu lông, hoa văn) bên cạnh đặc trưng khuôn mặt

---

