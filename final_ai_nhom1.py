# !pip install uv
# !uv pip install tensorflow opencv-python matplotlib scikit-learn tqdm pillow numpy
import os
import sys
import shutil
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')
CONFIG = {
    "img_size"     : (128, 128),  
    "batch_size"   : 32,
    "epochs"       : 20,
    "learning_rate": 0.001,
    "train_ratio"  : 0.70,        
    "test_ratio"   : 0.20,        
    "val_ratio"    : 0.10,        
    "model_path"   : "cat_dog_model.h5",
    "classes"      : ["Cat", "Dog"],
    "random_seed"  : 42,
}

pathname = r'/kaggle/input/datasets/datlee220806/dogcat/PetImages'
random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

def phan_chia_du_lieu(raw_dir: str = "dataset/raw"):
    splits = {"train": CONFIG["train_ratio"],
              "test" : CONFIG["test_ratio"],
              "val"  : CONFIG["val_ratio"]}

    for split in splits:
        for cls in CONFIG["classes"]:
            Path(f"{split}/{cls}").mkdir(parents=True, exist_ok=True)

    raw_dir = Path(raw_dir)
    tong = 0

    for cls in CONFIG["classes"]:
        files = list((raw_dir / cls).glob("*"))
        files = [f for f in files
                 if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * CONFIG["train_ratio"])
        n_test  = int(n * CONFIG["test_ratio"])
        # val = phần còn lại

        plan = {
            "train": files[:n_train],
            "test" : files[n_train:n_train + n_test],
            "val"  : files[n_train + n_test:],
        }

        print(f"\n[{cls.upper()}] Tổng: {n} ảnh")
        for split, lst in plan.items():
            for f in lst:
                dst = Path(f"{split}/{cls}/{f.name}")
                if not dst.exists():
                    shutil.copy(f, dst)
            pct = len(lst) / n * 100 if n else 0
            print(f"  {split:5s}: {len(lst):5d} ảnh ({pct:.1f}%)")
        tong += n

    print(f"\n[✓] Tổng cộng {tong} ảnh đã phân chia")


def tao_pipeline():
    import tensorflow as tf

    IMG  = CONFIG["img_size"]
    BS   = CONFIG["batch_size"]

    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    datagen_eval = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
    )

    train_gen = datagen_train.flow_from_directory(
        f"train",
        target_size=IMG, batch_size=BS,
        class_mode='binary', shuffle=True,
        seed=CONFIG["random_seed"],
    )
    test_gen = datagen_eval.flow_from_directory(
        f"test",
        target_size=IMG, batch_size=BS,
        class_mode='binary', shuffle=False,
    )
    val_gen = datagen_eval.flow_from_directory(
        f"val",
        target_size=IMG, batch_size=BS,
        class_mode='binary', shuffle=False,
    )

    print(f"\n[Pipeline] Class mapping: {train_gen.class_indices}")
    print(f"           Train: {train_gen.samples} | Test: {test_gen.samples} | Val: {val_gen.samples}")
    return train_gen, test_gen, val_gen


def xay_dung_mo_hinh_cnn():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    IMG_H, IMG_W = CONFIG["img_size"]
    
    model = models.Sequential([
        # ── Block 1 ──────────────────────────────────
        layers.Conv2D(32, (3,3), padding='same', input_shape=(IMG_H, IMG_W, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # ── Block 2 ──────────────────────────────────
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # ── Block 3 ──────────────────────────────────
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # ── Classifier ───────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),   # Output [0, 1]
    ], name="CNN_Custom")

    return model


def xay_dung_mo_hinh_mobilenet():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    IMG_H, IMG_W = CONFIG["img_size"]

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_H, IMG_W, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False  

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),   # Output [0, 1]
    ], name="MobileNetV2_Transfer")

    return model


def huan_luyen(model, train_gen, val_gen):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            CONFIG["model_path"], monitor='val_accuracy',
            save_best_only=True, verbose=1,
        ),
    ]

    print(f"\n[*] Bắt đầu huấn luyện — {CONFIG['epochs']} epochs...")
    history = model.fit(
        train_gen,
        epochs=CONFIG["epochs"],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def danh_gia(model, test_gen):
    from sklearn.metrics import classification_report, confusion_matrix
    import tensorflow as tf

    print("\n" + "="*55)
    print("  ĐÁNH GIÁ TRÊN TẬP TEST (20%)")
    print("="*55)

    y_prob = model.predict(test_gen, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes

    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\n  Loss     : {loss:.4f}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=CONFIG["classes"],
        digits=4,
    ))

    cm = confusion_matrix(y_true, y_pred)
    ve_bieu_do_danh_gia(cm, y_prob, y_true)

    return acc


def ve_bieu_do_danh_gia(cm, y_prob, y_true):
    """Vẽ confusion matrix và phân phối xác suất."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=CONFIG["classes"],
           yticklabels=CONFIG["classes"],
           title="Confusion Matrix (Test Set)",
           ylabel="Nhãn thực tế",
           xlabel="Nhãn dự đoán")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i,j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=14, fontweight='bold')

    ax = axes[1]
    cats = y_prob[y_true == 0]
    dogs = y_prob[y_true == 1]
    ax.hist(cats, bins=30, alpha=0.7, color='orange', label='Cat (nhãn thực)')
    ax.hist(dogs, bins=30, alpha=0.7, color='steelblue', label='Dog (nhãn thực)')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Ngưỡng 0.5')
    ax.set(xlabel="Xác suất dự đoán [0 → 1]",
           ylabel="Số lượng ảnh",
           title="Phân phối xác suất đầu ra",
           xlim=[0, 1])
    ax.legend()

    plt.tight_layout()
    plt.savefig("danh_gia_mo_hinh.png", dpi=120, bbox_inches='tight')
    print("\n  [✓] Biểu đồ đánh giá đã lưu: danh_gia_mo_hinh.png")
    plt.show()


def ve_bieu_do_hoc(history):
    """Vẽ Learning Curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in zip(
        axes,
        [('accuracy','val_accuracy'), ('loss','val_loss')],
        ['Accuracy theo Epoch', 'Loss theo Epoch'],
    ):
        ax.plot(history.history[metric[0]], label='Train', linewidth=2)
        ax.plot(history.history[metric[1]], label='Validation', linewidth=2)
        ax.set(title=title, xlabel='Epoch',
               ylabel=metric[0].capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=120, bbox_inches='tight')
    print("  [✓] Learning curve đã lưu: learning_curve.png")
    plt.show()


def du_doan_anh(duong_dan_anh: str, model=None, hien_thi: bool = True):
    import tensorflow as tf
    import cv2

    if model is None:
        if not Path(CONFIG["model_path"]).exists():
            raise FileNotFoundError(
                f"Không tìm thấy model '{CONFIG['model_path']}'. "
                "Hãy huấn luyện trước bằng cách chạy hàm main()."
            )
        model = tf.keras.models.load_model(CONFIG["model_path"])

    img_bgr = cv2.imread(duong_dan_anh)
    if img_bgr is None:
        raise ValueError(f"Không đọc được file: {duong_dan_anh}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, CONFIG["img_size"])
    img_norm = img_res.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)  

    xac_suat = float(model.predict(img_input, verbose=0)[0][0])
    nhan = "dog" if xac_suat >= 0.5 else "cat"
    do_tin_cay = xac_suat if nhan == "dog" else (1.0 - xac_suat)

    ket_qua = {
        "xac_suat"  : round(xac_suat, 4),
        "nhan"      : nhan,
        "do_tin_cay": round(do_tin_cay, 4),
    }

    if hien_thi:
        print("\n" + "─"*45)
        print(f"  Ảnh       : {Path(duong_dan_anh).name}")
        print(f"  Xác suất  : {xac_suat:.4f}  (0=Mèo ← → Chó=1)")
        print(f"  Dự đoán   : {'🐶 CHÓ' if nhan=='dog' else '🐱 MÈO'}")
        print(f"  Tin cậy   : {do_tin_cay*100:.1f}%")
        print("─"*45)

        # Cảnh báo: Lệnh cv2.imshow sẽ gây lỗi treo Kernel nếu chạy trên Kaggle/Google Colab.
        # Nếu chạy trên môi trường đám mây, bạn nên đổi sang dùng matplotlib.pyplot.imshow()
        try:
            mau = (30, 144, 255) if nhan == "dog" else (255, 140, 0)
            ten_hien = f"{'CHO' if nhan=='dog' else 'MEO'}  {do_tin_cay:.2f}"
            h, w = img_rgb.shape[:2]
            scale = min(400 / max(h, w), 1.0)
            img_show = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
            
            img_show_bgr = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(img_show_bgr, (0, 0), (img_show_bgr.shape[1], 40), mau, -1)
            cv2.putText(img_show_bgr, ten_hien, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Ket qua nhan dien", img_show_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print("[!] Không thể hiển thị ảnh pop-up (có thể bạn đang ở môi trường Cloud như Kaggle/Colab).")

    return ket_qua


def du_doan_lo(thu_muc_hoac_danh_sach, model=None):
    import tensorflow as tf

    if model is None and Path(CONFIG["model_path"]).exists():
        model = tf.keras.models.load_model(CONFIG["model_path"])

    if isinstance(thu_muc_hoac_danh_sach, str):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        danh_sach = [
            str(f) for f in Path(thu_muc_hoac_danh_sach).iterdir()
            if f.suffix.lower() in exts
        ]
    else:
        danh_sach = thu_muc_hoac_danh_sach

    if not danh_sach:
        print("[!] Không tìm thấy ảnh nào")
        return []

    ket_qua_lo = []
    print(f"\n[*] Đang dự đoán {len(danh_sach)} ảnh...")
    for anh in tqdm(danh_sach):
        try:
            kq = du_doan_anh(anh, model=model, hien_thi=False)
            kq["file"] = Path(anh).name
            ket_qua_lo.append(kq)
        except Exception as e:
            print(f"  [!] Lỗi {anh}: {e}")

    cats = sum(1 for k in ket_qua_lo if k["nhan"] == "cat")
    dogs = sum(1 for k in ket_qua_lo if k["nhan"] == "dog")
    print(f"\n  Kết quả: 🐱 Mèo = {cats}  |  🐶 Chó = {dogs}  |  Tổng = {len(ket_qua_lo)}")
    print(f"\n  {'File':<30} {'Xác suất':>10} {'Nhãn':>6} {'Tin cậy':>10}")
    print("  " + "─"*60)
    for k in ket_qua_lo:
        print(f"  {k['file']:<30} {k['xac_suat']:>10.4f} {k['nhan']:>6} {k['do_tin_cay']*100:>9.1f}%")

    return ket_qua_lo


def main(su_dung_mobilenet: bool = True, chi_du_doan_path: str = None):
    """
    Hàm chính điều hướng luồng chạy.
    - Nếu có truyền `chi_du_doan_path`, script sẽ bỏ qua huấn luyện và đi thẳng vào dự đoán.
    - Nếu không, script sẽ tự động chia dữ liệu, huấn luyện và đánh giá.
    """
    if chi_du_doan_path:
        print(f"[*] Chế độ dự đoán được bật cho: {chi_du_doan_path}")
        if Path(chi_du_doan_path).is_file():
            du_doan_anh(chi_du_doan_path)
        elif Path(chi_du_doan_path).is_dir():
            du_doan_lo(chi_du_doan_path)
        else:
            print(f"[!] Đường dẫn {chi_du_doan_path} không hợp lệ.")
        return

    # Quy trình huấn luyện tiêu chuẩn
    phan_chia_du_lieu(raw_dir=pathname) 
    train_gen, test_gen, val_gen = tao_pipeline()
    
    if su_dung_mobilenet:
        print("\n[*] Sử dụng kiến trúc MobileNetV2 Transfer Learning...")
        model = xay_dung_mo_hinh_mobilenet()
    else:
        print("\n[*] Sử dụng kiến trúc CNN Tự Xây Dựng...")
        model = xay_dung_mo_hinh_cnn()
        
    history = huan_luyen(model, train_gen, val_gen)
    ve_bieu_do_hoc(history)
    danh_gia(model, test_gen)


if __name__ == "__main__":
    # Để huấn luyện toàn bộ quy trình:
    main(su_dung_mobilenet=True) 
    
    # Nếu muốn dự đoán trực tiếp (đã có model_path rồi), có thể comment lệnh bên trên và dùng:
    # main(chi_du_doan_path="duong_dan_den_thu_muc_hoac_file_anh")
