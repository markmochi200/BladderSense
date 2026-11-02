"""
Benchmark script to compute parameter counts, FLOPs (if thop installed), and runtime for each model variant.
Saves results to CSV and prints a table.
"""
import csv
from model import create_model, count_parameters, try_compute_flops, benchmark_runtime

MODEL_NAMES = ["base", "no_coord", "no_spatial", "no_channel", "no_spatial_no_channel"]

def run_bench(output_csv='benchmarks.csv', device=None, input_size=224):
    rows = []
    for name in MODEL_NAMES:
        print("Benchmarking:", name)
        model = create_model(name, in_channels=1)
        total, trainable = count_parameters(model)
        macs, params_thop = try_compute_flops(model, input_size=(1, 1, input_size, input_size), device=device) if True else (None, None)
        mean_ms, median_ms = benchmark_runtime(model, input_size=(1, 1, input_size, input_size), device=device, n_iters=30, warmup=5)
        row = {
            'model': name,
            'params_total': int(total),
            'params_trainable': int(trainable),
            'flops_macs': int(macs) if macs is not None else None,  # type: ignore
            'flops_params_thop': int(params_thop) if params_thop is not None else None,  # type: ignore
            'runtime_mean_ms': float(mean_ms),
            'runtime_median_ms': float(median_ms)
        }
        rows.append(row)
        print(row)

    # write CSV
    keys = rows[0].keys()
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("Saved benchmark CSV to", output_csv)
    return rows

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='benchmarks.csv')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    run_bench(output_csv=args.out, device=args.device)
