from ultralytics import YOLO
import torch, os
from multiprocessing import freeze_support

def main():
    data_dir = r"D:\dataset\UHCTD_datas"
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11s-cls.yaml")
    print("✅ 当前模型结构配置如下:")
    print(model.model.yaml)

    results = model.train(
        data=data_dir,
        imgsz=224,        # 更小图
        batch=16,          # 16batch
        epochs=60,        # 理论上够
        patience=5,      # 提前停止
        cache="ram",
        workers=8,       # 根据 CPU/IO 再调
        cos_lr=True,
        device=device,
        pretrained=False,
        project=f"{data_dir}/runs",
        name="pretrain_224",
        exist_ok=True,
    )

    print(f"✅ Best model path: {results.save_dir / 'best.pt'}")
    print(f"✅ Top-1 accuracy: {results.top1:.2%}")


if __name__ == "__main__":
    freeze_support()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    main()
