"""
Module phụ trách nạp dữ liệu thô, xử lý tính liên tục thời gian (DST, missing timestamps)
và chuẩn hóa chuỗi dữ liệu phục vụ phân tích.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
from .config import Config


class DataLoader:
    """
    Class quản lý quá trình nạp và làm sạch dữ liệu chuỗi thời gian phụ tải điện.
    """

    def __init__(self, raw_path: Union[str, Path] = Config.RAW_DATA_PATH):
        self.raw_path = Path(raw_path)

    def load_clean_series(self) -> pd.Series:
        """
        Đọc dữ liệu thô, sắp xếp tăng dần, xử lý trùng giờ DST và nội suy các mốc thiếu.

        Returns:
            pd.Series: Chuỗi phụ tải sạch liên tục với DatetimeIndex (freq='h').
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {self.raw_path}")

        # 1. Đọc file CSV thô
        df_raw = pd.read_csv(self.raw_path)

        # 2. Định dạng thời gian và sắp xếp tăng dần
        df_raw[Config.TIME_COL] = pd.to_datetime(df_raw[Config.TIME_COL])
        df_sorted = df_raw.sort_values(Config.TIME_COL).reset_index(drop=True)

        # 3. Xử lý trùng lặp giờ DST (Fall Back) bằng cách lấy trung bình
        # và thiết lập lưới thời gian liên tục 1 giờ (freq='h')
        series = (
            df_sorted.groupby(Config.TIME_COL)[Config.TARGET_COL]
            .mean()
            .asfreq(Config.FREQUENCY)
        )

        # 4. Nội suy các mốc thời gian thiếu (DST Spring Forward) bằng phép nội suy thời gian
        series = series.interpolate(method="time")
        series.name = Config.TARGET_COL

        return series

    @staticmethod
    def get_summary_statistics(series: pd.Series) -> Dict[str, Any]:
        """
        Trích xuất các chỉ số thống kê mô tả quan trọng của chuỗi thời gian.
        """
        desc = series.describe()
        return {
            "count": int(desc["count"]),
            "mean": float(desc["mean"]),
            "std": float(desc["std"]),
            "min": float(desc["min"]),
            "min_time": str(series.idxmin()),
            "q25": float(desc["25%"]),
            "median": float(desc["50%"]),
            "q75": float(desc["75%"]),
            "max": float(desc["max"]),
            "max_time": str(series.idxmax()),
            "start_time": str(series.index.min()),
            "end_time": str(series.index.max()),
            "nan_count": int(series.isnull().sum())
        }


def load_and_clean_data(raw_path: Union[str, Path] = Config.RAW_DATA_PATH) -> pd.Series:
    """Hàm tiện ích nạp nhanh chuỗi dữ liệu làm sạch."""
    loader = DataLoader(raw_path)
    return loader.load_clean_series()

