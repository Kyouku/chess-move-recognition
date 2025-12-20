# %% [markdown]
# # YOLO11 speed and accuracy comparison
#
# - Benchmark YOLO11 s/m/l in PyTorch (with and without fuse)
# - Benchmark exported ONNX models with ONNX Runtime on CPU
# - Evaluate mAP and precision/recall on your chess dataset
# - Plot results and save CSV files under `models/comparison`

import time
# %%
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType  # noqa: F401

    HAS_ORT = True
except Exception:
    HAS_ORT = False

# Project root relative to this notebook (training/notebooks)
PROJECT_ROOT = Path("../..").resolve()
print("Project root:", PROJECT_ROOT)

# %% [markdown]
# ## Model and dataset configuration

# %%
# Models: (Name, path)
YOLO_MODELS = [
    ("YOLO11s", PROJECT_ROOT / "models" / "yolo11s_best.pt"),
    ("YOLO11m", PROJECT_ROOT / "models" / "yolo11m_best.pt"),
    ("YOLO11l", PROJECT_ROOT / "models" / "yolo11l_best.pt"),
]

# Folder for ONNX exports (we keep them directly in models/)
ONNX_DIR = PROJECT_ROOT / "models"

# Ultralytics data config for your chess dataset (for accuracy metrics)
YOLO_DATA_CONFIG = (
        PROJECT_ROOT / "training" / "Automated-Chess-Move-Recognition-17-yolo" / "data.yaml"
)

# Optional: example warped board (used as input for benchmarking if it exists)
SAMPLE_IMAGE = (
        PROJECT_ROOT
        / "data"
        / "processed_images"
        / "game3_img_0003_warp.png"
)

# Image sizes to compare
IMAGE_SIZES = [640, 512, 416]

DEVICE = "cpu"

N_WARMUP = 10
N_ITERS = 50

print("Data config:", YOLO_DATA_CONFIG)
print("Sample image exists:", SAMPLE_IMAGE.exists())
print("Torch device:", DEVICE)
print("ONNX Runtime available:", HAS_ORT)

[(name, path, path.exists()) for name, path in YOLO_MODELS]


# %% [markdown]
# ## Input helpers

# %%
def make_input_tensor(image_size: int, device: str) -> torch.Tensor:
    """Create an input tensor (1,3,H,W) for PyTorch."""
    if SAMPLE_IMAGE is not None and SAMPLE_IMAGE.exists():
        import cv2

        img = cv2.imread(str(SAMPLE_IMAGE))
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        t = torch.from_numpy(arr).unsqueeze(0).to(device)
    else:
        t = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32, device=device)
    return t


def make_input_numpy(image_size: int) -> np.ndarray:
    """Create an input array (1,3,H,W) for ONNX Runtime."""
    if SAMPLE_IMAGE is not None and SAMPLE_IMAGE.exists():
        import cv2

        img = cv2.imread(str(SAMPLE_IMAGE))
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, 0)
    else:
        arr = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    return arr


# %% [markdown]
# ## Speed benchmarking helpers

# %%
def benchmark_yolo_pytorch(
        model_name: str,
        weights_path: Path,
        image_size: int,
        device: str,
        fused: bool,
) -> dict:
    """Benchmark YOLO11 model in PyTorch (with or without fuse)."""
    model = YOLO(str(weights_path))
    if fused:
        model.fuse()

    inp = make_input_tensor(image_size, device)

    # Warmup
    _ = model.predict(
        source=inp,
        imgsz=image_size,
        device=device,
        half=False,
        verbose=False,
    )
    for _ in range(N_WARMUP):
        _ = model.predict(
            source=inp,
            imgsz=image_size,
            device=device,
            half=False,
            verbose=False,
        )

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = model.predict(
            source=inp,
            imgsz=image_size,
            device=device,
            half=False,
            verbose=False,
        )
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / N_ITERS
    fps = 1000.0 / avg_ms

    return {
        "model": model_name,
        "variant": f"{model_name} PyTorch{' fused' if fused else ''} {image_size}",
        "backend": "pytorch",
        "fused": fused,
        "image_size": image_size,
        "latency_ms": avg_ms,
        "fps": fps,
    }


def ensure_onnx_export(
        model_name: str,
        weights_path: Path,
        onnx_dir: Path,
        image_size: int,
) -> Path:
    """Export YOLO11 model to ONNX for a given image size if missing.

    Creates one file per model and size, for example `yolo11s_best_416.onnx`.
    """
    onnx_dir.mkdir(parents=True, exist_ok=True)
    stem = weights_path.stem  # for example yolo11s_best
    onnx_path = onnx_dir / f"{stem}_{image_size}.onnx"

    if onnx_path.exists():
        print(f"ONNX model for {model_name} {image_size} already exists:", onnx_path)
        return onnx_path

    print(f"Export {model_name} to ONNX for {image_size}...")
    model = YOLO(str(weights_path))
    tmp_path = Path(
        model.export(
            format="onnx",
            imgsz=image_size,
            opset=17,
            simplify=True,
        )
    )
    # Ultralytics chooses its own name; rename to our size specific name if needed.
    if tmp_path != onnx_path:
        tmp_path.replace(onnx_path)

    print("ONNX exported to:", onnx_path)
    return onnx_path


def benchmark_onnx_runtime(
        model_name: str,
        onnx_path: Path,
        image_size: int,
) -> dict:
    """Benchmark YOLO11 ONNX model with ONNX Runtime on CPU."""
    if not HAS_ORT:
        raise RuntimeError("onnxruntime is not installed")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # adapt threads to your CPU if you want
    so.intra_op_num_threads = 8
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    inp = make_input_numpy(image_size)

    for _ in range(N_WARMUP):
        _ = sess.run(None, {input_name: inp})

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = sess.run(None, {input_name: inp})
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / N_ITERS
    fps = 1000.0 / avg_ms

    return {
        "model": model_name,
        "variant": f"{model_name} ONNX {image_size}",
        "backend": "onnxruntime",
        "fused": None,
        "image_size": image_size,
        "latency_ms": avg_ms,
        "fps": fps,
    }


# %% [markdown]
# ## Run speed benchmarks

# %%
all_rows: list[dict] = []

for model_name, weights_path in YOLO_MODELS:
    if not weights_path.exists():
        print("Weights for", model_name, "not found, skip.")
        continue

    print("\n==============================")
    print("Model:", model_name)
    print("Path:", weights_path)
    print("==============================")

    for size in IMAGE_SIZES:
        print("\n----- Image size:", size, "-----")

        # PyTorch normal
        all_rows.append(
            benchmark_yolo_pytorch(
                model_name=model_name,
                weights_path=weights_path,
                image_size=size,
                device=DEVICE,
                fused=False,
            )
        )

        # PyTorch fused
        all_rows.append(
            benchmark_yolo_pytorch(
                model_name=model_name,
                weights_path=weights_path,
                image_size=size,
                device=DEVICE,
                fused=True,
            )
        )

        # ONNX Runtime
        if HAS_ORT:
            print("ONNX Runtime benchmark for", model_name, size)
            onnx_path = ensure_onnx_export(
                model_name,
                weights_path,
                ONNX_DIR,
                size,
            )
            all_rows.append(
                benchmark_onnx_runtime(
                    model_name=model_name,
                    onnx_path=onnx_path,
                    image_size=size,
                )
            )
        else:
            print("onnxruntime not available, skip ONNX for", model_name, size)

results_df = pd.DataFrame(all_rows)
if not results_df.empty:
    results_df.sort_values(
        by=["model", "image_size", "latency_ms"],
        inplace=True,
    )

results_df


# %% [markdown]
# ## Accuracy metrics (mAP, precision, recall)

# %%
def eval_yolo_metrics(
        model_name: str,
        weights_path: Path,
        data_config: Path,
        image_size: int,
        device: str,
) -> Optional[dict]:
    """Run model.val for one model and one image size and return mAP and precision/recall."""
    if not data_config.exists():
        print(
            "Data config not found, skip metrics for",
            model_name,
            image_size,
        )
        return None

    print(f"Validation for {model_name} at image size {image_size}...")
    model = YOLO(str(weights_path))
    model.fuse()

    metrics = model.val(
        data=str(data_config),
        imgsz=image_size,
        device=device,
        verbose=False,
    )

    box = metrics.box
    mp, mr, map50, map50_95 = box.mean_results()

    return {
        "model": model_name,
        "image_size": image_size,
        "mAP50": float(map50),
        "mAP50_95": float(map50_95),
        "precision": float(mp),
        "recall": float(mr),
    }


# %%
metric_rows: list[dict] = []

for model_name, weights_path in YOLO_MODELS:
    if not weights_path.exists():
        print("Weights for", model_name, "not found, skip metrics.")
        continue

    print("\n=== Accuracy for", model_name, "===")
    for size in IMAGE_SIZES:
        m = eval_yolo_metrics(
            model_name,
            weights_path,
            YOLO_DATA_CONFIG,
            size,
            DEVICE,
        )
        if m is not None:
            metric_rows.append(m)

metrics_df = pd.DataFrame(metric_rows)
if not metrics_df.empty:
    metrics_df.sort_values(by=["model", "image_size"], inplace=True)

metrics_df

# %% [markdown]
# ## Plots: speed and accuracy

# %%
# Speed plots
if not results_df.empty:
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["variant"], results_df["latency_ms"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Latency in ms")
    plt.title("YOLO11 model CPU latency comparison")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(results_df["variant"], results_df["fps"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("FPS")
    plt.title("YOLO11 model CPU FPS comparison")
    plt.tight_layout()
    plt.show()
else:
    print("No speed results available.")

# Accuracy plots
if not metrics_df.empty:
    plt.figure(figsize=(7, 5))
    for model_name in metrics_df["model"].unique():
        sub = metrics_df[metrics_df["model"] == model_name]
        plt.plot(
            sub["image_size"],
            sub["mAP50_95"],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Image size")
    plt.ylabel("mAP50–95")
    plt.title("Validation mAP50–95 over different input resolutions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No metric results available.")

# %% [markdown]
# ## Save CSVs (speed, accuracy, combined)

# %%
if not metrics_df.empty:
    full_df = pd.merge(
        results_df,
        metrics_df,
        on=["model", "image_size"],
        how="left",
    )
else:
    full_df = results_df.copy()

comparison_dir = PROJECT_ROOT / "models" / "comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)

if not results_df.empty:
    results_df.to_csv(comparison_dir / "yolo11_speed_results.csv", index=False)
if not metrics_df.empty:
    metrics_df.to_csv(comparison_dir / "yolo11_accuracy_results.csv", index=False)
if not full_df.empty:
    full_df.to_csv(comparison_dir / "yolo11_full_results.csv", index=False)

full_df.head()
