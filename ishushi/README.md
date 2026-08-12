# ishushi

Nhánh triển khai riêng cho một dự án khác, dùng chung nền tảng dự báo với
`agents/`. Hiện đang phục vụ vận hành nên chưa hợp nhất.

## Nợ kỹ thuật đã ghi nhận

Thư mục này trùng lặp đáng kể với `agents/`:

| Module | `ishushi/` | `agents/` | Ghi chú |
|---|---|---|---|
| `data_agent.py` | 769 dòng | 1.041 dòng | Logic nạp & làm sạch dữ liệu gần giống |
| `master_file_agent.py` | 726 dòng | 515 dòng | Ghi/đọc Excel master, đã phân kỳ |
| `forecast_model.py` | 1.615 dòng | — | Riêng của nhánh này |
| `brain.py` | 662 dòng | 2.677 dòng (`forecast_brain.py`) | Cùng ý tưởng, khác mức độ hoàn thiện |

**Rủi ro:** hai nhánh sẽ tiếp tục phân kỳ theo thời gian. Bug sửa ở một bên
không tự động sang bên kia, và người mới rất dễ sửa nhầm file.

## Hướng xử lý

Hợp nhất thành **một** codebase, đưa khác biệt vào cấu hình thay vì nhân bản code:

```python
# Thay vì hai bản copy của DataAgent:
@dataclass(frozen=True)
class BrandConfig:
    code: str                       # "default" | "ishushi"
    shift_definitions: dict
    master_file_name: str
    volume_thresholds: VolumeThresholds
    holiday_curve: dict

agent = DataAgent(brand=BRANDS["ishushi"])
```

## Vì sao chưa làm ngay

Nhánh này đang chạy production cho một dự án khác. Hợp nhất đòi hỏi:

1. Lập bảng đối chiếu đầy đủ những chỗ hai bản khác nhau
2. Trích xuất khác biệt thành `BrandConfig`
3. Chạy song song hai nhánh trên cùng dữ liệu, xác minh **output không đổi**
4. Chỉ cắt chuyển sau khi (3) đạt

Rewrite một lượt sẽ nhanh hơn nhưng rủi ro không chấp nhận được với hệ thống
đang phục vụ vận hành thật.

**Lộ trình:** [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §4 Giai đoạn 3.

## Quy ước tạm thời

Trong lúc chờ hợp nhất:

- Sửa bug ở `agents/` thì **kiểm tra xem `ishushi/` có lỗi tương tự không**
- Code **mới** viết vào `agents/`, không thêm vào `ishushi/`
- Mọi thay đổi ở đây ghi rõ trong commit message có prefix `ishushi:`
