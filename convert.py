#!/usr/bin/env python3
"""
批量转换 stdata/*.tsv → converted/*.csv
========================================
修复:
  - DataFrame 碎片化警告（用 pd.concat 替代逐列赋值）
  - 大图像警告
  - 全零列处理
"""

import os
import numpy as np
import pandas as pd

# ═══════════════ 配置 ═══════════════
INPUT_DIR = r"D:\HuaweiMoveData\Users\王杨琴\Documents\Her2st\Her2st\stdata"
OUTPUT_DIR = "converted"
N_TOP_GENES = 3000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_one(tsv_path, output_dir):
    basename = os.path.splitext(os.path.basename(tsv_path))[0]
    sample_name = basename.replace("_stdata", "")

    print(f"\n{'─'*50}")
    print(f"  📄 {sample_name}")

    # ── 读取 ──
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    n_spots, n_genes_raw = df.shape
    print(f"  Spots: {n_spots} | Genes: {n_genes_raw}")

    # ── barcode → 阵列坐标 ──
    barcodes = df.index.astype(str).tolist()
    rows, cols = [], []
    for b in barcodes:
        parts = b.split('x')
        if len(parts) == 2:
            try:
                rows.append(int(parts[0]))
                cols.append(int(parts[1]))
            except ValueError:
                rows.append(0)
                cols.append(0)
        else:
            rows.append(0)
            cols.append(0)

    rows_arr = np.array(rows)
    cols_arr = np.array(cols)

    # 归一化
    row_norm = (rows_arr - rows_arr.min()) / (rows_arr.max() - rows_arr.min() + 1e-6)
    col_norm = (cols_arr - cols_arr.min()) / (cols_arr.max() - cols_arr.min() + 1e-6)

    # ── 筛选高变基因 ──
    n_keep = min(N_TOP_GENES, df.shape[1])
    gene_vals = df.values.astype(np.float32)

    # 方差计算，排除全零列
    gene_vars = gene_vals.var(axis=0)
    valid = np.where(gene_vars > 0)[0]

    if len(valid) == 0:
        print(f"  ⚠️ 所有基因方差为0，跳过")
        return

    if len(valid) < n_keep:
        n_keep = len(valid)
        top_idx = valid
    else:
        top_idx = valid[np.argsort(gene_vars[valid])[-n_keep:]]

    gene_data = gene_vals[:, top_idx]

    # log1p + 标准化
    if gene_data.max() > 50:
        gene_data = np.log1p(gene_data)
    gene_data = (gene_data - gene_data.mean(axis=0)) / (gene_data.std(axis=0) + 1e-6)

    # ── 组装输出（修复碎片化：pd.concat 一次性拼接）──
    out = pd.DataFrame({
        "X_norm": col_norm,
        "Y_norm": row_norm,
        "array_row": rows_arr,
        "array_col": cols_arr,
        "barcode": barcodes,
        "sample": sample_name,
    })

    gene_df = pd.DataFrame(
        gene_data,
        columns=[f"G{i:04d}" for i in range(gene_data.shape[1])]
    )
    out = pd.concat([out, gene_df], axis=1)

    # ── 保存 ──
    out_path = os.path.join(output_dir, f"{sample_name}.csv")
    out.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  ✅ {os.path.basename(out_path)} ({size_mb:.1f} MB) | {out.shape[0]} spots × {out.shape[1]} cols")


def main():
    tsv_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".tsv")])
    print(f"📂 {len(tsv_files)} 个 TSV → {OUTPUT_DIR}/")
    ok, fail = 0, 0

    for i, f in enumerate(tsv_files):
        print(f"[{i+1}/{len(tsv_files)}]", end="")
        try:
            convert_one(os.path.join(INPUT_DIR, f), OUTPUT_DIR)
            ok += 1
        except Exception as e:
            print(f"  ❌ {e}")
            fail += 1

    print(f"\n✅ 完成: {ok} 成功, {fail} 失败")
    print(f"📁 {OUTPUT_DIR}/ 共 {len(os.listdir(OUTPUT_DIR))} 个 CSV")


if __name__ == "__main__":
    main()