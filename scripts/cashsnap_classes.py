from __future__ import annotations

CLASS_NAMES = [
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {index: name for index, name in enumerate(CLASS_NAMES)}
