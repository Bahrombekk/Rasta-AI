"""Class-agnostic detektorni SKU-110K'da fine-tune qilish (real rasmlar uchun).

Umumiy YOLO (COCO) zich javon mahsulotlarini yaxshi topmaydi. SKU-110K —
1.7 mln quti, bitta 'object' klassi — aynan shu vazifa uchun. Ultralytics
bu datasetni avtomatik yuklaydi.

Foydalanish:
  python scripts/train_detector.py --model yolo11m.pt --epochs 40 --imgsz 1280

Eslatma: SKU-110K litsenziyasi akademik — tijorat uchun o'z rasmlaringizda
qo'shimcha fine-tune qiling (hisobotning 17-bo'limi).
"""
import argparse

import _bootstrap  # noqa: F401


def main():
    ap = argparse.ArgumentParser(description="SKU-110K detektor fine-tune")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--data", default="SKU-110K.yaml", help="Ultralytics dataset config")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="models",
        name="sku110k_detector",
    )
    print("[✓] O'qitish tugadi. Og'irliklar: models/sku110k_detector/weights/best.pt")
    print("    config.yaml -> detector.weights ni shu yo'lga o'zgartiring.")


if __name__ == "__main__":
    main()
