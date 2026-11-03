"""
Module thực hiện kỹ thuật Feature Engineering cho chuỗi thời gian:
- Tạo đặc trưng trễ (Lag Features) phục vụ chuyển đổi bài toán sang Supervised Learning.
- Tạo đặc trưng thống kê trượt (Rolling Mean, Rolling Std).
- Tách ma trận đầu vào X và biến mục tiêu y.
- Phân chia tập Train / Validation / Test theo thứ tự thời gian và xuất file.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, NamedTuple, Optional
import pandas as pd
from .config import Config


class DatasetSplits(NamedTuple):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame


class TimeSeriesFeatureBuilder:
    """
    Class chuyên trách tạo các đặc trưng cho chuỗi thời gian và phân chia tập dữ liệu.
    """

    @staticmethod
    def create_lag_features(series: pd.Series, p: int = Config.DEFAULT_LAGS) -> pd.DataFrame:
        """
        Tạo p bậc trễ liên tục từ lag_1 đến lag_p và loại bỏ các dòng NaN do shift.
        """
        df_feat = pd.DataFrame({"y": series})
        for i in range(1, p + 1):
            df_feat[f"lag_{i}"] = series.shift(i)
        return df_feat.dropna()

    @staticmethod
    def create_selective_lags(series: pd.Series, lags: List[int]) -> pd.DataFrame:
        """
        Tạo các đặc trưng trễ chọn lọc (như chu kỳ ngày, 2-3 ngày, chu kỳ tuần).
        """
        df_feat = pd.DataFrame({"y": series})
        for lag in lags:
            df_feat[f"lag_{lag}"] = series.shift(lag)
        return df_feat.dropna()

    @staticmethod
    def add_rolling_features(
        df_supervised: pd.DataFrame, 
        base_col: str = "lag_1", 
        windows: List[int] = [6, 12, 24]
    ) -> pd.DataFrame:
        """
        Tạo các đặc trưng thống kê trượt (Rolling Mean, Rolling Std) dựa trên biến lag_1
        để tuyệt đối không gây rò rỉ dữ liệu tương lai (No Data Leakage).
        """
        df_res = df_supervised.copy()
        for w in windows:
            df_res[f"rolling_mean_{w}"] = df_res[base_col].rolling(window=w).mean()
            df_res[f"rolling_std_{w}"] = df_res[base_col].rolling(window=w).std()
        return df_res.dropna()

    @staticmethod
    def split_X_y(df_supervised: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Tách DataFrame thành ma trận đặc trưng X và vector mục tiêu y."""
        X = df_supervised.drop(columns=["y"])
        y = df_supervised["y"]
        return X, y

    @staticmethod
    def time_based_split(
        df_supervised: pd.DataFrame,
        ratios: Dict[str, float] = Config.SPLIT_RATIOS,
        export_dir: Optional[Union[str, Path]] = None
    ) -> DatasetSplits:
        """
        Phân chia dữ liệu tuần tự theo dòng thời gian thành 3 phần: Train, Val, Test.
        Tuyệt đối không xáo trộn (shuffle=False) để bảo toàn tính nhân quả thời gian.
        """
        n_total = len(df_supervised)
        train_size = int(n_total * ratios.get("train", 0.75))
        val_size = int(n_total * ratios.get("val", 0.125))
        test_size = n_total - train_size - val_size

        df_train = df_supervised.iloc[:train_size]
        df_val = df_supervised.iloc[train_size:train_size + val_size]
        df_test = df_supervised.iloc[train_size + val_size:]

        X_train = df_train.drop(columns=["y"])
        y_train = df_train["y"]
        X_val = df_val.drop(columns=["y"])
        y_val = df_val["y"]
        X_test = df_test.drop(columns=["y"])
        y_test = df_test["y"]

        # Nếu có thư mục xuất dữ liệu thì tự động lưu file
        if export_dir is not None:
            out_path = Path(export_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            df_train.to_csv(out_path / "aep_train.csv")
            df_val.to_csv(out_path / "aep_val.csv")
            df_test.to_csv(out_path / "aep_test.csv")
            X_train.to_csv(out_path / "X_train.csv")
            y_train.to_csv(out_path / "y_train.csv")
            X_val.to_csv(out_path / "X_val.csv")
            y_val.to_csv(out_path / "y_val.csv")
            X_test.to_csv(out_path / "X_test.csv")
            y_test.to_csv(out_path / "y_test.csv")

        return DatasetSplits(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            df_train=df_train, df_val=df_val, df_test=df_test
        )


# Các hàm tiện ích
def create_lag_features(series: pd.Series, p: int = Config.DEFAULT_LAGS) -> pd.DataFrame:
    return TimeSeriesFeatureBuilder.create_lag_features(series, p=p)


def time_based_split(
    df_supervised: pd.DataFrame,
    ratios: Dict[str, float] = Config.SPLIT_RATIOS,
    export_dir: Optional[Union[str, Path]] = None
) -> DatasetSplits:
    return TimeSeriesFeatureBuilder.time_based_split(df_supervised, ratios=ratios, export_dir=export_dir)

