from __future__ import annotations

import io
import urllib.request
import zipfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sklearn.datasets import fetch_california_housing, fetch_covtype, get_data_home
from sklearn.model_selection import train_test_split


REGRESSION = "regression"
BINARY = "binary"
MULTICLASS = "multiclass"

MAGIC_URL = "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"


@dataclass(frozen=True)
class Dataset:

    name: str
    task: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y_train)


    def headline(self) -> str:
        shape = f"{len(self.x_train):,} train / {len(self.x_test):,} test, {self.n_features} features"

        if self.task == REGRESSION:
            return f"{self.name} | regression ({shape})"

        return f"{self.name} | {self.task} classification ({shape}, {len(self.classes)} classes)"


def load_california_housing(max_rows: int | None = 25_000, seed: int = 4327) -> Dataset:
    bunch = fetch_california_housing()

    return _split(
        name="California Housing",
        task=REGRESSION,
        x=bunch.data,
        y=bunch.target,
        max_rows=max_rows,
        seed=seed,
    )


def load_magic_telescope(max_rows: int | None = 25_000, seed: int = 4327) -> Dataset:
    table = np.loadtxt(_magic_data_file(), delimiter=",", dtype=str)

    return _split(
        name="MAGIC Gamma Telescope",
        task=BINARY, 
        x=table[:, :-1].astype(float),
        y=(table[:, -1] == "g").astype(int),
        max_rows=max_rows,
        seed=seed,
    )


def load_covertype(max_rows: int | None = 25_000, seed: int = 4327) -> Dataset:
    bunch = fetch_covtype()

    return _split(
        name="Forest Covertype",
        task=MULTICLASS,
        x=bunch.data,
        y=bunch.target.astype(int) - 1,
        max_rows=max_rows,
        seed=seed,
    )


LOADERS = {
    "california": load_california_housing,
    "magic": load_magic_telescope,
    "covertype": load_covertype,
}


def _split(
    name: str,
    task: str,
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int | None,
    seed: int,
    test_size: float = 0.25,
) -> Dataset:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    if max_rows is not None and len(x) > max_rows:
        x, _, y, _ = train_test_split(
            x, y, train_size=max_rows, random_state=seed, stratify=_stratify(task, y)
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=_stratify(task, y)
    )

    return Dataset(name, task, x_train, x_test, y_train, y_test)


def _stratify(task: str, y: np.ndarray) -> np.ndarray | None:
    return None if task == REGRESSION else y


def _magic_data_file() -> Path:
    cached = Path(get_data_home()) / "magic04.data"

    if not cached.exists():
        with urllib.request.urlopen(MAGIC_URL, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))

        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(archive.read("magic04.data"))

    return cached
