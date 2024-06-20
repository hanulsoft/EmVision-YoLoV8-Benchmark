import os
from pathlib import Path
from itertools import product

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from bench_helper.names import get_module_name
from bench_helper.benchmark import benchmark


display_module_name = get_module_name()
dir_module_name = display_module_name.replace("NVIDIA Jetson ", "")
dir_module_name = dir_module_name.replace("(", "")
dir_module_name = dir_module_name.replace(")", "")
dir_module_name = dir_module_name.replace(" ", "_")
work_dir = Path(f"./{dir_module_name}_work")
work_dir.mkdir(exist_ok=True)
bench_dir = Path(f"./{dir_module_name}_bench").absolute()
bench_dir.mkdir(exist_ok=True)

print(f"Module name: {display_module_name}")
print(f"Module directory name: ./{dir_module_name}")
print(f"Work directory: {work_dir}")
print(f"Benchmark directory: {bench_dir}")

root_dir = Path(__file__).parent.absolute()

os.chdir(work_dir)


def single_benchmark(model_name: str, dtype: str = "FP32") -> None:
    global bench_dir
    half = False
    int8 = False
    if dtype == "FP16":
        half = True
    elif dtype == "INT8":
        int8 = True
    result = benchmark(
        model=model_name,
        data="coco128.yaml",
        imgsz=640,
        half=half,
        int8=int8,
        verbose=False,
        device="cuda",
    )
    csv_path = bench_dir / f"{model_name}_{dtype}.csv"
    result.to_csv(csv_path.absolute().as_posix(), index=False)
    print(f"Saved benchmark result to benchmark/{model_name}_{dtype}.csv")
    return


def export_benchmark():
    global bench_dir, root_dir, dir_module_name
    bench_files = list(bench_dir.glob("*.csv"))
    bench_dfs = [pd.read_csv(file) for file in bench_files]
    for fname, df in zip(bench_files, bench_dfs):
        model_precision = fname.stem
        df["Model"] = model_precision
    benchmark_df = pd.concat(bench_dfs, ignore_index=True)
    bench_columns = ["Model"] + list(benchmark_df.columns[:-1])
    benchmark_df = benchmark_df[bench_columns]
    benchmark_df = benchmark_df.drop(columns=["Status❔"])
    benchmark_df = benchmark_df.rename(
        columns={
            "metrics/mAP50-95(B)": "mAP50-95",
            "Inference time (ms/im)": "Inference (ms/im)",
        }
    )

    def custom_sort_key(model_precision):
        key_order = {"n": 0, "s": 10, "m": 20, "l": 30, "x": 40}
        precision_order = {"FP32": 0, "FP16": 1, "INT8": 2}
        suffix = model_precision.split("_")[0][-1]
        precision = model_precision.split("_")[1]
        return key_order[suffix] + precision_order[precision]

    benchmark_df = benchmark_df.sort_values(
        by="Model", key=lambda x: x.map(custom_sort_key)
    )
    benchmark_df = benchmark_df.reset_index(drop=True)
    benchmark_df.to_markdown(root_dir / f"benchmark_{dir_module_name}.md", index=False)

    def color_mapping_infer(s):
        idx_max = s.sort_values(ascending=False).index
        min_val = s.min()
        result = []
        log_s = np.log1p(s)
        cm_map = plt.cm.Greens(log_s / log_s.max())
        cm_idx = 0
        for idx in idx_max:
            r, g, b, _ = cm_map[cm_idx]
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            if min_val == s[cm_idx]:
                result.append(
                    f"background-color: rgb({240}, {80}, {80});"
                    f"color: #FFFFFF; font-weight: bold;"
                )
            elif brightness < 90:
                result.append(
                    f"background-color: rgb({r}, {g}, {b});" f"color: #A9A9A9;"
                )
            else:
                result.append(
                    f"background-color: rgb({r}, {g}, {b});" f"color: #000000;"
                )
            cm_idx += 1
        return result

    def color_mapping_map(s):
        idx_max = s.sort_values(ascending=True).index
        max_val = s.max()
        result = []
        cm_map = plt.cm.Greens(s / s.max())
        cm_idx = 0
        for idx in idx_max:
            r, g, b, _ = cm_map[len(s) - 1 - cm_idx]
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            if max_val == s[cm_idx]:
                result.append(
                    f"background-color: rgb({240}, {80}, {80});"
                    f"color: #FFFFFF; font-weight: bold;"
                )
            elif brightness < 90:
                result.append(
                    f"background-color: rgb({r}, {g}, {b});" f"color: #A9A9A9;"
                )
            else:
                result.append(
                    f"background-color: rgb({r}, {g}, {b});" f"color: #000000;"
                )
            cm_idx += 1
        return result

    benchmark_df = benchmark_df.style.apply(
        color_mapping_map, subset=["mAP50-95"]
    ).apply(color_mapping_infer, subset=["Inference (ms/im)"])
    benchmark_df.to_html(
        root_dir / f"benchmark_{dir_module_name}.html", index=False, justify="right"
    )
    print(f"Exported benchmark to {root_dir}/benchmark_{dir_module_name}.html")
    return


def run():
    # yolov8x는 16GB 이상의 메모리가 필요함
    models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
    precisions = ["INT8", "FP16", "FP32"]
    combinations = list(product(models, precisions))
    for model_name, dtype in combinations:
        try:
            single_benchmark(model_name, dtype)
        except Exception as e:
            print(f"Error occured during benchmarking {model_name} {dtype}")
            print(f"Error message: {e}")
    export_benchmark()
    return


if __name__ == "__main__":
    run()
