import time
import platform
from pathlib import Path

from pandas import DataFrame

from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.utils import LINUX, MACOS, ASSETS, LOGGER, WEIGHTS_DIR
from ultralytics.engine.exporter import export_formats
from ultralytics.utils.files import file_size
from ultralytics.utils.checks import check_yolo
from ultralytics.utils.torch_utils import select_device


def benchmark(
    model=WEIGHTS_DIR / "yolov8n.pt",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
) -> DataFrame:

    import pandas as pd

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    y = []
    t0 = time.time()
    for i, (
        name,
        format,
        suffix,
        cpu,
        gpu,
    ) in export_formats().iterrows():  # index, (name, format, suffix, CPU, GPU)
        emoji, filename = "❌", None  # export defaults
        try:
            assert i != 9 or LINUX, "Edge TPU export only supported on Linux"
            if i == 5:
                assert MACOS or LINUX, "CoreML export only supported on macOS and Linux"
            elif i == 10:
                assert MACOS or LINUX, "TF.js export only supported on macOS and Linux"
            if name in [
                "TensorRT",
                "OpenVINO",
                "CoreML",
                "TensorFlow SavedModel",
                "TensorFlow GraphDef",
                "TensorFlow Lite",
                "TensorFlow Edge TPU",
                "TensorFlow.js",
                "PaddlePaddle",
            ]:
                continue
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if format == "-":
                filename = model.ckpt_path or model.cfg
                exported_model = model  # PyTorch format
            else:
                filename = model.export(
                    imgsz=imgsz,
                    format=format,
                    half=half,
                    int8=int8,
                    device=device,
                    verbose=False,
                )
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "export failed"
            emoji = "❎"  # indicates export succeeded

            # Predict
            assert (
                model.task != "pose" or i != 7
            ), "GraphDef Pose inference is not supported"
            assert i not in (
                9,
                10,
            ), "inference not supported"  # Edge TPU and TF.js are unsupported
            assert (
                i != 5 or platform.system() == "Darwin"
            ), "inference only supported on macOS>=10.13"  # CoreML
            exported_model.predict(
                ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half
            )

            # Validate
            data = (
                data or TASK2DATA[model.task]
            )  # task to dataset, i.e. coco8.yaml for task=detect
            key = TASK2METRIC[
                model.task
            ]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect
            results = exported_model.val(
                data=data,
                batch=1,
                imgsz=imgsz,
                plots=False,
                device=device,
                half=half,
                int8=int8,
                verbose=False,
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            y.append(
                [
                    name,
                    "✅",
                    round(file_size(filename), 1),
                    round(metric, 4),
                    round(speed, 2),
                ]
            )
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.warning(f"ERROR ❌️ Benchmark failure for {name}: {e}")
            y.append(
                [name, emoji, round(file_size(filename), 1), None, None]
            )  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(
        y, columns=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)"]
    )

    name = Path(model.ckpt_path).name
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  # values to compare to floor
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(
            x > floor for x in metrics if pd.notna(x)
        ), f"Benchmark failure: metric(s) < floor {floor}"

    return df
