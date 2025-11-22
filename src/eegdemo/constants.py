"""定数定義"""
from pathlib import Path

# プロジェクトルート
ROOT = Path(__file__).parent.parent.parent

# データパス
DATA_DIR = ROOT / "data"
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
CACHE_DIR = DATA_DIR / "cache"

# 実験設定
SAMPLING_RATE = 500  # Hz
N_CHANNELS = 32
N_SUBJECTS = 12
N_TRAIN_SERIES = 8
N_TEST_SERIES = 2  # series 9, 10

# 6イベント (常にこの順で発生)
EVENTS = [
    "HandStart",
    "FirstDigitTouch",
    "BothStartLoadPhase",
    "LiftOff",
    "Replace",
    "BothReleased",
]

# 32チャンネル名 (10-20拡張系)
CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]

# 10-20 電極座標 [x, y] — 正規化済み球面投影 (topomap 用)
# 原点 = Cz, 半径 1 = 耳 (T7/T8)
CHANNEL_POS_2D: dict[str, tuple[float, float]] = {
    "Fp1":  (-0.18, 0.85), "Fp2":  ( 0.18, 0.85),
    "F7":   (-0.71, 0.55), "F3":   (-0.37, 0.55), "Fz":   ( 0.00, 0.55), "F4":  ( 0.37, 0.55), "F8":   ( 0.71, 0.55),
    "FC5":  (-0.60, 0.27), "FC1":  (-0.22, 0.27), "FC2":  ( 0.22, 0.27), "FC6": ( 0.60, 0.27),
    "T7":   (-1.00, 0.00), "C3":   (-0.50, 0.00), "Cz":   ( 0.00, 0.00), "C4":  ( 0.50, 0.00), "T8":   ( 1.00, 0.00),
    "TP9":  (-0.95,-0.31), "CP5":  (-0.60,-0.27), "CP1":  (-0.22,-0.27), "CP2": ( 0.22,-0.27), "CP6":  ( 0.60,-0.27), "TP10": ( 0.95,-0.31),
    "P7":   (-0.71,-0.55), "P3":   (-0.37,-0.55), "Pz":   ( 0.00,-0.55), "P4":  ( 0.37,-0.55), "P8":   ( 0.71,-0.55),
    "PO9":  (-0.50,-0.78), "O1":   (-0.18,-0.85), "Oz":   ( 0.00,-0.85), "O2":  ( 0.18,-0.85), "PO10": ( 0.50,-0.78),
}

# 周波数帯域定義
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "mu":    (8.0, 13.0),   # μ帯 (中心部で運動関連)
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# 運動関連チャンネル (ERP/ERD 主解析対象)
MOTOR_CHANNELS = ["C3", "Cz", "C4", "FC1", "FC2", "CP1", "CP2"]

# 前頭チャンネル
FRONTAL_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]
