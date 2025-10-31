# frft_discrete.py
# 单精度高效版 Discrete FrFT 2D（BCHW），含缓存与完整测试样例
import torch
import numpy as np
import scipy.io as scio
from pathlib import Path
from typing import Dict, Tuple

# =========================
# 精度/缓存配置
# =========================
# 默认使用单精度（推荐）：大幅减少显存、通常显著提速
DTYPE_REAL    = torch.float32
DTYPE_COMPLEX = torch.complex64

# 如需切回双精度：
# DTYPE_REAL    = torch.float64
# DTYPE_COMPLEX = torch.complex128

USE_PHASE_CACHE = True  # True: 按 (N, a, device, dtype) 缓存相位向量

# ---------------- 缓存：避免重复读取/构造 ----------------
_E_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}
_PHASE_CACHE: Dict[Tuple[int, float, str, torch.dtype], torch.Tensor] = {}


def set_precision(mode: str = "single"):
    """
    运行时切换精度：
      - "single": float32/complex64（默认）
      - "double": float64/complex128
    """
    global DTYPE_REAL, DTYPE_COMPLEX
    if mode == "single":
        DTYPE_REAL, DTYPE_COMPLEX = torch.float32, torch.complex64
    elif mode == "double":
        DTYPE_REAL, DTYPE_COMPLEX = torch.float64, torch.complex128
    else:
        raise ValueError("mode 必须是 'single' 或 'double'。")


def clear_caches():
    """清空所有缓存（E/phase）。"""
    _E_CACHE.clear()
    _PHASE_CACHE.clear()


# ---------------- 工具：安全读取 mat ----------------
def _try_load_mat(path: Path):
    if not path.exists():
        return None
    try:
        data = scio.loadmat(str(path))
        return data
    except Exception:
        return None


# ---------------- 加载 E(N×N) 并缓存 ----------------
def _load_E(N: int, device: torch.device):
    """
    自动按 N 尺寸加载 E：
    - 支持多种常见命名：{N}.mat, E_{N}.mat, {N}_E.mat, frft_{N}.mat
    - 支持变量名 'E'，或 'E_real' + 'E_imag'
    - 读入后统一转为 DTYPE_COMPLEX（默认 complex64），并缓存
    """
    key = (N, device.type, DTYPE_COMPLEX)
    if key in _E_CACHE:
        return _E_CACHE[key]

    candidates = [
        Path(f"./mat/{N}.mat"),
        Path(f"E_{N}.mat"),
        Path(f"{N}_E.mat"),
        Path(f"frft_{N}.mat"),
    ]
    data = None
    used_path = None
    for p in candidates:
        data = _try_load_mat(p)
        if data is not None:
            used_path = p
            break

    if data is None:
        raise FileNotFoundError(
            f"未找到尺寸 {N} 对应的 E 矩阵文件。请提供以下任一文件："
            f" ./mat/{N}.mat 或 E_{N}.mat 或 {N}_E.mat 或 frft_{N}.mat"
        )

    # 变量名兼容
    if "E" in data:
        E_np = np.array(data["E"])
    elif "E_real" in data and "E_imag" in data:
        E_np = np.array(data["E_real"]) + 1j * np.array(data["E_imag"])
    else:
        raise KeyError(
            f"文件 {used_path} 内未找到 'E' 或 ('E_real','E_imag') 变量。"
        )

    if E_np.shape != (N, N):
        raise ValueError(
            f"文件 {used_path} 中 E 形状为 {E_np.shape}，但期望 {(N, N)}。"
        )

    # 先在 CPU 侧降精，减少 H2D 传输
    # 注意：astype(copy=False) 避免不必要复制
    if DTYPE_COMPLEX == torch.complex64:
        E_np = E_np.astype(np.complex64, copy=False)
    else:
        E_np = E_np.astype(np.complex128, copy=False)

    E_t = torch.from_numpy(E_np)  # CPU
    E = E_t.to(device=device, dtype=DTYPE_COMPLEX, non_blocking=True)

    _E_CACHE[key] = E
    return E


# ---------------- 等价于 fftshift 的半幅滚动（0-based 安全索引） ----------------
def _make_shift_indices(N: int, device: torch.device):
    return torch.remainder(torch.arange(N, device=device) + (N // 2), N)


# ---------------- 相位缓存/构造 ----------------
def _get_phase(N: int, a: torch.Tensor, device: torch.device):
    """
    生成长度 N 的相位向量：
        phase[k] = exp(-j * pi/2 * a * k), k=0..N-1
    使用全 torch 常量，保持 dtype/设备一致。
    """
    a_s = a.to(dtype=DTYPE_REAL)
    if USE_PHASE_CACHE:
        # 用 float(a) 作为键（训练中 a 连续变化会增加条目，请按需关闭缓存）
        key = (N, float(a_s.item()), device.type, DTYPE_REAL)
        if key in _PHASE_CACHE:
            return _PHASE_CACHE[key]

    k = torch.arange(N, device=device, dtype=DTYPE_REAL)  # (N,)
    # 注意：(-1j) 常量直接是 complex，后续会自动提升为 DTYPE_COMPLEX
    phase = torch.exp((-1j * (torch.pi / 2) * a_s) * k).to(DTYPE_COMPLEX)  # (N,)

    if USE_PHASE_CACHE:
        _PHASE_CACHE[key] = phase
    return phase


# ---------------- 在“第一维(N)”做一次 Disfrft ----------------
def _disfrft_firstdim(X: torch.Tensor, a: torch.Tensor, E: torch.Tensor):
    """
    X: (N, M) complex
    a: 标量张量（float）
    E: (N, N) complex
    返回: (N, M) complex
    """
    N, M = X.shape
    device = X.device

    shft = _make_shift_indices(N, device)   # (N,)
    Xs = X.index_select(dim=0, index=shft)  # 中心化

    # E^T * Xs
    inner = torch.matmul(E.transpose(0, 1), Xs)  # (N, M)

    # 相位对角
    phase = _get_phase(N, a, device)            # (N,)
    inner = phase[:, None] * inner              # (N, M)

    # E * (...)
    Y_tmp = torch.matmul(E, inner)              # (N, M)

    # 反中心化
    Y = torch.empty_like(X)
    Y.index_copy_(0, shft, Y_tmp)
    return Y


# ---------------- 2D DisFrFT：先沿 H，再沿 W ----------------
def Disfrft2d_bchw(x: torch.Tensor, a: torch.Tensor):
    """
    x: (B, C, H, W) 实数或复数
    a: 标量张量（全局同一阶数）
    返回: (B, C, H, W) complex（默认 complex64）
    """
    assert x.dim() == 4, f"x must be (B,C,H,W), got {tuple(x.shape)}"
    B, C, H, W = x.shape
    device = x.device

    # 统一到目标复数精度
    if not torch.is_complex(x):
        xr = x.to(DTYPE_REAL)
        x = torch.complex(xr, torch.zeros_like(xr))
    else:
        x = x.to(DTYPE_COMPLEX)

    # 加载对应 E 矩阵
    E_H = _load_E(H, device)
    E_W = _load_E(W, device)

    # 沿 H 维（把 H 放到第 0 维）
    X = x.permute(2, 0, 1, 3).reshape(H, B * C * W)   # (H, B*C*W)
    Y = _disfrft_firstdim(X, a, E_H)                  # (H, B*C*W)
    y = Y.reshape(H, B, C, W).permute(1, 2, 0, 3)     # (B, C, H, W)

    # 沿 W 维（把 W 放到第 0 维）
    X2 = y.permute(3, 0, 1, 2).reshape(W, B * C * H)  # (W, B*C*H)
    Y2 = _disfrft_firstdim(X2, a, E_W)                # (W, B*C*H)
    y2 = Y2.reshape(W, B, C, H).permute(1, 2, 3, 0)   # (B, C, H, W)
    return y2


# ---------------- 前/反向封装 ----------------
def frft(obj: torch.Tensor, frft_order: torch.Tensor):
    return Disfrft2d_bchw(obj, frft_order)


def ifrft(propfield: torch.Tensor, frft_order: torch.Tensor):
    return Disfrft2d_bchw(propfield, -frft_order)


# ---------------- 自测 ----------------
def test():
    """
    说明：
    - 需要 ./mat/{H}.mat 和 ./mat/{W}.mat（或候选命名）中存在 E 矩阵。
    - 默认测试尺寸 H=W=8；请按实际 E 的尺寸修改。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[INFO] device={device}, real={DTYPE_REAL}, complex={DTYPE_COMPLEX}")

    B, C, H, W = 1, 512, 8, 8   # 修改为你手头 E 的尺寸
    try:
        x = torch.rand(B, C, H, W, dtype=DTYPE_REAL, device=device)
        a = torch.tensor(0.5, device=device, dtype=DTYPE_REAL)

        Y = frft(x, a)                 # complex
        Mag = torch.abs(Y)             # float
        x_rec = ifrft(Y, a)            # 逆变换（-a）

        print("[OK] forward/backward done",
              f"Y.dtype={Y.dtype}, |Y|.dtype={Mag.dtype}, x_rec.dtype={x_rec.dtype}")
        print("Shapes:", x.shape, Y.shape, x_rec.shape)
    except FileNotFoundError as e:
        print("[WARN] 未找到 E 矩阵文件：", e)
        print("      请在 ./mat/ 下放置对应尺寸的 .mat 文件后再运行 test()。")


# if __name__ == "__main__":
#     # 如需切换到双精度：set_precision("double")
#     # set_precision("double")
#     test()
