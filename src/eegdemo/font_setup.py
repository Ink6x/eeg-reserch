"""matplotlib の日本語フォント設定 (Windows)"""
import matplotlib
import matplotlib.pyplot as plt


def setup_japanese_font() -> None:
    """Windows 環境で日本語フォントを設定する"""
    try:
        plt.rcParams["font.family"] = ["MS Gothic", "DejaVu Sans"]
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
