#!/usr/bin/env python3
"""
SpaCRD + mclSTExp 完整训练 — 68 样本联合
==========================================
读取 converted/*.csv + HEimage/*.jpg
三阶段训练: mclSTExp → SpaCRD Stage I → Stage II+III

修复:
  - NearestNeighbors 参数格式
  - 大图像警告消除
"""

import os, glob, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 解除大图像限制

from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from torchvision import transforms, models

# ════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_DIR = "converted"
HE_DIR = r"D:\HuaweiMoveData\Users\王杨琴\Documents\Her2st\Her2st\HEimage"
SAVE_DIR = "spacrd_checkpoints"
N_GENES = 3000
PATCH_SIZE = 112
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🔥 设备: {DEVICE}")

# ════════════════════════════════
# 模型定义
# ════════════════════════════════

class ImageEncoder(nn.Module):
    def __init__(self, in_dim=2048, h_dim=1024, p_dim=512, d=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(d),
            nn.Linear(h_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(d))
        self.proj = nn.Linear(512, p_dim)
    def forward(self, x): return F.normalize(self.proj(self.net(x)), dim=-1)


class GeneEncoder(nn.Module):
    def __init__(self, in_dim=3000, h_dim=1024, p_dim=512, d=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(d),
            nn.Linear(h_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(d))
        self.proj = nn.Linear(512, p_dim)
    def forward(self, x): return F.normalize(self.proj(self.net(x)), dim=-1)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d=512, h=8, dp=0.1):
        super().__init__()
        self.h, self.dk = h, d // h
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.drop = nn.Dropout(dp)

    def forward(self, q, k, v):
        B = q.size(0)
        Q = self.wq(q).view(B, -1, self.h, self.dk).transpose(1, 2)
        K = self.wk(k).view(B, -1, self.h, self.dk).transpose(1, 2)
        V = self.wv(v).view(B, -1, self.h, self.dk).transpose(1, 2)
        a = self.drop(F.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.dk), dim=-1))
        return self.wo((a @ V).transpose(1, 2).contiguous().view(B, -1, self.wq.in_features))


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, d=512, h=8):
        super().__init__()
        self.ca_g = MultiHeadCrossAttention(d, h)
        self.ca_i = MultiHeadCrossAttention(d, h)
        self.f = nn.Sequential(
            nn.Linear(d * 2, d), nn.LayerNorm(d), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(d, d))

    def forward(self, hi, hg):
        return self.f(torch.cat([
            self.ca_i(hi, hg, hg)[:, 0, :],
            self.ca_g(hg, hi, hi)[:, 0, :]
        ], dim=-1))


class RVAE(nn.Module):
    def __init__(self, d=512, ld=64, nc=2):
        super().__init__()
        self.e1 = nn.Linear(d, 256)
        self.e2 = nn.Linear(256, 128)
        self.mu = nn.Linear(128, ld)
        self.lv = nn.Linear(128, ld)
        self.d1 = nn.Linear(ld, 128)
        self.d2 = nn.Linear(128, 256)
        self.do = nn.Linear(256, d)
        self.cm = nn.Parameter(torch.randn(nc, ld) * 0.5)

    def encode(self, x):
        h = F.relu(self.e2(F.relu(self.e1(x))))
        return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        return self.do(F.relu(self.d2(F.relu(self.d1(z))))), mu, lv, z


class CancerClassifier(nn.Module):
    def __init__(self, ld=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ld * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, mu, lv):
        return self.net(torch.cat([mu, lv], -1)).squeeze(-1)


class SpaCRD(nn.Module):
    def __init__(self, id=2048, gd=3000, d=512, ld=64):
        super().__init__()
        self.ie = ImageEncoder(id, p_dim=d)
        self.ge = GeneEncoder(gd, p_dim=d)
        self.bca = BidirectionalCrossAttention(d)
        self.rvae = RVAE(d, ld)
        self.clf = CancerClassifier(ld)

    def fit_gmm(self, s):
        s = np.array(s).reshape(-1, 1)
        g = GaussianMixture(2, random_state=42).fit(s)
        m = g.means_.flatten()
        idx = np.argsort(m)
        m1, m2 = m[idx[0]], m[idx[1]]
        s1 = np.sqrt(g.covariances_.flatten()[idx[0]])
        s2 = np.sqrt(g.covariances_.flatten()[idx[1]])
        p1, p2 = g.weights_[idx[0]], g.weights_[idx[1]]
        a = 1/(2*s2**2) - 1/(2*s1**2)
        b = m2/(s2**2) - m1/(s1**2)
        c = m1**2/(2*s1**2) - m2**2/(2*s2**2) + np.log((s2*p1)/(s1*p2))
        if abs(a) < 1e-12:
            return float((m1 + m2) / 2)
        r = np.roots([a, b, c])
        v = [x.real for x in r if abs(x.imag) < 1e-8 and min(m1, m2) < x.real < max(m1, m2)]
        return float(v[0]) if v else float((m1 + m2) / 2)


class mclSTExp(nn.Module):
    def __init__(self, id=2048, gd=3000, hd=1024, pd=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(id, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hd, pd), nn.BatchNorm1d(pd), nn.ReLU(), nn.Dropout(0.2))
        self.dec = nn.Sequential(
            nn.Linear(pd, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hd, hd // 2), nn.BatchNorm1d(hd // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hd // 2, gd))

    def forward(self, x):
        return self.dec(self.enc(x))


# ════════════════════════════════
# 特征提取器
# ════════════════════════════════

class FeatureExtractor:
    def __init__(self):
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        bb.fc = nn.Identity()
        self.m = bb.to(DEVICE).eval()
        self.pp = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def extract_patches(self, img, coords, ps=112):
        w, h = img.size
        half = ps // 2
        feats = []
        for sx, sy in coords:
            l, t = max(0, int(sx) - half), max(0, int(sy) - half)
            r, b = min(w, int(sx) + half), min(h, int(sy) + half)
            if r - l < 10 or b - t < 10:
                feats.append(np.zeros(2048, np.float32))
            else:
                p = img.crop((l, t, r, b))
                tns = self.pp(p).unsqueeze(0).to(DEVICE)
                feats.append(self.m(tns).cpu().numpy().flatten())
        return np.array(feats, dtype=np.float32)


# ════════════════════════════════
# 工具函数
# ════════════════════════════════

def find_knn(coords, k=6):
    """修复：显式使用 n_neighbors 参数"""
    c = np.array(coords)
    n = NearestNeighbors(n_neighbors=min(k + 1, len(c))).fit(c)
    return n.kneighbors(c)[1][:, 1:]


def build_nb(img_f, gene_f, nb):
    n, k_nb = len(img_f), nb.shape[1]
    hi = np.zeros((n, k_nb + 1, img_f.shape[1]), np.float32)
    hg = np.zeros((n, k_nb + 1, gene_f.shape[1]), np.float32)
    for i in range(n):
        hi[i, 0], hg[i, 0] = img_f[i], gene_f[i]
        for j, nn in enumerate(nb[i]):
            hi[i, j + 1] = img_f[nn]
            hg[i, j + 1] = gene_f[nn]
    return hi, hg


# ════════════════════════════════
# H&E 图像智能匹配
# ════════════════════════════════

def find_he(sample, he_dir):
    """智能匹配 H&E 图像（兼容 BC/BT 混用）"""
    if not os.path.exists(he_dir):
        return None
    parts = sample.split("_")
    if len(parts) < 2:
        return None
    core, region = parts[0], "_".join(parts[1:])
    cn = core[2:]  # 数字编号
    prefixes = [core, f"BT{cn}", f"BC{cn}"]
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        for p in prefixes:
            path = os.path.join(he_dir, f"HE_{p}_{region}{ext}")
            if os.path.exists(path):
                return path
    # 兜底：遍历匹配
    for f in sorted(os.listdir(he_dir)):
        if cn in f and region.lower() in f.lower():
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                return os.path.join(he_dir, f)
    return None


# ════════════════════════════════
# 数据加载
# ════════════════════════════════

def load_all(csv_dir, he_dir):
    # ✅ 这里已修改：只取前 5 个 CSV
    csvs = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))[:5]
    print(f"\n📂 {len(csvs)} 个 CSV")

    extractor = FeatureExtractor()
    all_img, all_gene, all_nb, offsets = [], [], [], [0]
    missed_he = 0

    for csv_path in csvs:
        name = os.path.basename(csv_path).replace(".csv", "")
        df = pd.read_csv(csv_path)

        # ── 基因列（G0000～G2999）──
        gene_cols = [c for c in df.columns if c.startswith("G") and c[1:].isdigit()]
        gene = df[gene_cols].values.astype(np.float32)
        if gene.shape[1] < N_GENES:
            pad = np.zeros((gene.shape[0], N_GENES - gene.shape[1]), np.float32)
            gene = np.concatenate([gene, pad], 1)
        elif gene.shape[1] > N_GENES:
            gene = gene[:, :N_GENES]

        n_spots = len(gene)

        # ── H&E 图像 ──
        he_path = find_he(name, he_dir)
        if he_path:
            img = Image.open(he_path).convert("RGB")
            # 缩放到 max 3000px
            if max(img.size) > 3000:
                r = 3000 / max(img.size)
                img = img.resize((int(img.size[0] * r), int(img.size[1] * r)))
            iw, ih = img.size
            # 归一化坐标 → 像素
            xn = df["X_norm"].values
            yn = df["Y_norm"].values
            margin = 0.05
            px = (xn * iw * (1 - 2 * margin) + iw * margin).astype(int)
            py = (yn * ih * (1 - 2 * margin) + ih * margin).astype(int)
            coords = list(zip(px, py))
            img_f = extractor.extract_patches(img, coords, PATCH_SIZE)
            print(f"  ✅ {name:<25s} {n_spots:4d} spots | H&E: {os.path.basename(he_path)}")
        else:
            img_f = np.random.randn(n_spots, 2048).astype(np.float32) * 0.1
            coords = list(zip(df["X_norm"] * 2000, df["Y_norm"] * 2000))
            missed_he += 1
            print(f"  ⚠️ {name:<25s} {n_spots:4d} spots | (无图像，模拟特征)")

        nb = find_knn(coords, k=6)
        all_img.append(img_f)
        all_gene.append(gene)
        all_nb.append(nb)
        offsets.append(offsets[-1] + n_spots)

    # ── 合并 ──
    img_all = np.concatenate(all_img)
    gene_all = np.concatenate(all_gene)
    nb_all = np.concatenate([n + offsets[i] for i, n in enumerate(all_nb)])

    print(f"\n📊 总计: {len(img_all)} spots | 无图像样本: {missed_he}")
    return img_all, gene_all, nb_all


# ════════════════════════════════
# 主训练
# ════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 55)
    print("  SpaCRD + mclSTExp — 68 样本联合训练")
    print("=" * 55)

    img_f, gene_f, nb_idx = load_all(CSV_DIR, HE_DIR)
    N = len(img_f)

    X = torch.tensor(img_f).to(DEVICE)
    Y = torch.tensor(gene_f).to(DEVICE)

    # ══════════════════════════
    # 1. mclSTExp
    # ══════════════════════════
    print("\n" + "=" * 45)
    print("  1/3: mclSTExp (图像 → 基因)")
    print("=" * 45)
    mcl = mclSTExp(img_f.shape[1], gene_f.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(mcl.parameters(), lr=1e-4)
    dl = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)
    for ep in range(100):
        tl = 0
        for xb, yb in dl:
            loss = F.mse_loss(mcl(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1:4d}/100 | Loss: {tl/N:.6f}")
    mcl.eval()
    print(f"  ✅ mclSTExp 完成 ({time.time()-t0:.0f}s)")

    # ══════════════════════════
    # 2. SpaCRD Stage I
    # ══════════════════════════
    print("\n" + "=" * 45)
    print("  2/3: SpaCRD Stage I (对比学习)")
    print("=" * 45)
    sp = SpaCRD(img_f.shape[1], gene_f.shape[1]).to(DEVICE)
    sp.ie.train()
    sp.ge.train()
    opt = torch.optim.Adam(list(sp.ie.parameters()) + list(sp.ge.parameters()), lr=1e-5)
    temp = 0.07
    for ep in range(150):
        for xb, yb in dl:
            hi, hg = sp.ie(xb), sp.ge(yb)
            sim = hi @ hg.T / temp
            tgt = torch.arange(len(xb)).to(DEVICE)
            loss = (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 30 == 0:
            print(f"  Epoch {ep+1:4d}/150 | Loss: {loss.item():.6f}")
    sp.ie.eval()
    sp.ge.eval()
    print(f"  ✅ Stage I 完成 ({time.time()-t0:.0f}s)")

    # ── 重新编码 ──
    with torch.no_grad():
        hi = sp.ie(X).cpu().numpy()
        hg = sp.ge(Y).cpu().numpy()

    # ══════════════════════════
    # 3. SpaCRD Stage II+III
    # ══════════════════════════
    print("\n" + "=" * 45)
    print("  3/3: SpaCRD Stage II+III (VRBCA+分类)")
    print("=" * 45)
    hi_nb, hg_nb = build_nb(hi, hg, nb_idx)

    # 无监督伪标签
    sp.eval()
    with torch.no_grad():
        H_i = torch.tensor(hi_nb).to(DEVICE)
        H_g = torch.tensor(hg_nb).to(DEVICE)
        hs = sp.bca(H_i, H_g)
        _, mu, lv, _ = sp.rvae(hs)
        scores = torch.sigmoid(sp.clf(mu, lv)).cpu().numpy()
        th = sp.fit_gmm(scores)
        labels = (scores >= th).astype(np.float32)
    print(f"  GMM 阈值: {th:.4f} | 肿瘤: {labels.sum():.0f}/{len(labels)} ({labels.mean()*100:.1f}%)")

    sp.bca.train()
    sp.rvae.train()
    sp.clf.train()
    params = list(sp.bca.parameters()) + list(sp.rvae.parameters()) + list(sp.clf.parameters())
    opt = torch.optim.Adam(params, lr=1e-5)
    L = torch.tensor(labels).to(DEVICE)
    dl2 = DataLoader(TensorDataset(H_i, H_g, L), batch_size=128, shuffle=True)

    for ep in range(80):
        for hi_b, hg_b, yb in dl2:
            hs = sp.bca(hi_b, hg_b)
            recon, mu, lv, z = sp.rvae(hs)
            logit = sp.clf(mu, lv)
            bce = F.binary_cross_entropy_with_logits(logit, yb)
            recon_l = F.mse_loss(recon, hs)
            cm = sp.rvae.cm[yb.long()]
            kl = 0.5 * (lv.exp() + (mu - cm)**2 - lv - 1).sum(-1).mean()
            loss = bce + 0.1 * recon_l + 0.5 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1:4d}/80 | Loss: {loss.item():.6f}")
    sp.eval()
    print(f"  ✅ Stage II+III 完成 ({time.time()-t0:.0f}s)")

    # ══════════════════════════
    # 保存
    # ══════════════════════════
    save_path = os.path.join(SAVE_DIR, "spacrd_trained.pt")
    torch.save({
        "model_state": sp.state_dict(),
        "mclstexp_state": mcl.state_dict(),
        "trained": True,
        "img_dim": img_f.shape[1],
        "gene_dim": gene_f.shape[1],
    }, save_path)
    print(f"\n💾 {save_path}")
    print(f"\n{'='*55}")
    print(f"  ✅ 全部完成！总耗时: {(time.time()-t0)/60:.1f} 分钟")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()