from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import h5py
import numpy as np
import pandas as pd
import yaml


TEXT_EXTENSIONS = {".md", ".txt", ".html", ".htm"}
TABLE_EXTENSIONS = {".csv"}
JSON_EXTENSIONS = {".json"}
YAML_EXTENSIONS = {".yml", ".yaml"}
FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf"}


@dataclass(frozen=True)
class PackRecord:
    """Metadata describing a file that has been stored in the HDF5 archive."""

    relative_path: str
    dataset_path: str
    file_type: str
    size: int
    sha256: str


class HDF5ReportPackager:
    """Collect report artefacts (tables, configs, figures) into a single HDF5 file."""

    def __init__(
        self,
        report_dir: Path,
        include_figures: bool = True,
        compression: Optional[str] = "gzip",
        chunk_size: int = 1024 * 32,
    ) -> None:
        self.report_dir = report_dir
        self.include_figures = include_figures
        self.compression = compression
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pack(self, output_path: Path) -> List[PackRecord]:
        """Pack ``report_dir`` into ``output_path`` and return stored metadata."""

        records: List[PackRecord] = []

        with h5py.File(output_path, "w") as h5:
            self._write_metadata(h5)
            artefacts_group = h5.create_group("artefacts")

            for file_path in sorted(self._iter_candidate_files()):
                rel_path = file_path.relative_to(self.report_dir)
                group = self._ensure_group(artefacts_group, rel_path.parent)
                record = self._store_file(group, rel_path.name, file_path)
                records.append(record)

            self._write_index(h5, records)

        return records

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _write_metadata(self, h5: h5py.File) -> None:
        h5.attrs["created_utc"] = dt.datetime.utcnow().isoformat() + "Z"
        h5.attrs["report_root"] = str(self.report_dir.resolve())
        h5.attrs["generator"] = "HDF5ReportPackager"
        h5.attrs["generator_version"] = "1.0"

    def _write_index(self, h5: h5py.File, records: Iterable[PackRecord]) -> None:
        data = list(records)
        if not data:
            return

        dtype = np.dtype(
            [
                ("relative_path", h5py.string_dtype("utf-8")),
                ("dataset_path", h5py.string_dtype("utf-8")),
                ("file_type", h5py.string_dtype("utf-8")),
                ("size", "<i8"),
                ("sha256", h5py.string_dtype("utf-8")),
            ]
        )
        array = np.empty(len(data), dtype=dtype)
        for idx, record in enumerate(data):
            array[idx] = (
                record.relative_path,
                record.dataset_path,
                record.file_type,
                record.size,
                record.sha256,
            )

        index_group = h5.create_group("index")
        index_group.create_dataset("files", data=array, compression=self.compression)

    # ------------------------------------------------------------------
    # Core packing logic
    # ------------------------------------------------------------------
    def _iter_candidate_files(self) -> Iterator[Path]:
        if not self.report_dir.exists():
            raise FileNotFoundError(f"Report directory not found: {self.report_dir}")

        for path in self.report_dir.rglob("*"):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix in TABLE_EXTENSIONS | JSON_EXTENSIONS | YAML_EXTENSIONS | TEXT_EXTENSIONS:
                yield path
            elif suffix in FIGURE_EXTENSIONS:
                if self.include_figures:
                    yield path
            else:
                # Skip derived artefacts like checkpoints or logs.
                continue

    def _ensure_group(self, root: h5py.Group, relative_dir: Path) -> h5py.Group:
        group = root
        for part in relative_dir.parts:
            if part:
                group = group.require_group(part)
        return group

    def _store_file(self, group: h5py.Group, filename: str, file_path: Path) -> PackRecord:
        suffix = file_path.suffix.lower()
        file_type = self._infer_file_type(suffix)

        file_group = group.create_group(filename)
        file_group.attrs["source_path"] = str(file_path)
        file_group.attrs["file_type"] = file_type

        if suffix in TABLE_EXTENSIONS:
            dataset_path = self._store_csv(file_group, file_path)
        elif suffix in JSON_EXTENSIONS:
            dataset_path = self._store_json(file_group, file_path)
        elif suffix in YAML_EXTENSIONS:
            dataset_path = self._store_yaml(file_group, file_path)
        elif suffix in TEXT_EXTENSIONS:
            dataset_path = self._store_text(file_group, file_path)
        elif suffix in FIGURE_EXTENSIONS and self.include_figures:
            dataset_path = self._store_binary(file_group, file_path)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

        sha256 = self._sha256(file_path)
        size = file_path.stat().st_size

        return PackRecord(
            relative_path=str(file_path.relative_to(self.report_dir)),
            dataset_path=dataset_path,
            file_type=file_type,
            size=size,
            sha256=sha256,
        )

    def _store_csv(self, file_group: h5py.Group, file_path: Path) -> str:
        df = pd.read_csv(file_path, keep_default_na=False)
        dataset = self._create_dataframe_dataset(file_group, "table", df)
        dataset.attrs["columns"] = list(df.columns)
        dataset.attrs["row_count"] = len(df)
        return dataset.name

    def _store_json(self, file_group: h5py.Group, file_path: Path) -> str:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        text = json.dumps(payload, indent=2, sort_keys=True)
        dataset = self._create_text_dataset(file_group, "json", text)
        dataset.attrs["keys"] = sorted(payload.keys()) if isinstance(payload, dict) else []
        return dataset.name

    def _store_yaml(self, file_group: h5py.Group, file_path: Path) -> str:
        with file_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        text = yaml.safe_dump(payload, sort_keys=False)
        dataset = self._create_text_dataset(file_group, "yaml", text)
        if isinstance(payload, dict):
            dataset.attrs["keys"] = list(payload.keys())
        return dataset.name

    def _store_text(self, file_group: h5py.Group, file_path: Path) -> str:
        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()
        dataset = self._create_text_dataset(file_group, "text", text)
        dataset.attrs["length"] = len(text)
        return dataset.name

    def _store_binary(self, file_group: h5py.Group, file_path: Path) -> str:
        with file_path.open("rb") as f:
            data = f.read()
        dataset = file_group.create_dataset(
            "binary",
            data=np.frombuffer(data, dtype="uint8"),
            compression=self.compression,
            chunks=True,
        )
        dataset.attrs["length"] = len(data)
        return dataset.name

    def _create_dataframe_dataset(
        self,
        group: h5py.Group,
        dataset_name: str,
        df: pd.DataFrame,
    ) -> h5py.Dataset:
        dtype_fields = []
        column_data = []

        for column in df.columns:
            series = df[column]
            if pd.api.types.is_integer_dtype(series):
                dtype = series.to_numpy().dtype
                data = series.to_numpy()
            elif pd.api.types.is_float_dtype(series):
                dtype = "<f8"
                data = series.to_numpy(dtype=np.float64)
            elif pd.api.types.is_bool_dtype(series):
                dtype = "|b1"
                data = series.to_numpy(dtype=np.bool_)
            else:
                dtype = h5py.string_dtype("utf-8")
                data = series.astype(str).to_numpy()
            dtype_fields.append((column, dtype))
            column_data.append(data)

        structured = np.zeros(len(df), dtype=np.dtype(dtype_fields))
        for (column, _), values in zip(dtype_fields, column_data):
            structured[column] = values

        dataset = group.create_dataset(
            dataset_name,
            data=structured,
            compression=self.compression,
        )
        return dataset

    def _create_text_dataset(
        self,
        group: h5py.Group,
        dataset_name: str,
        text: str,
    ) -> h5py.Dataset:
        data = np.array([text], dtype=h5py.string_dtype("utf-8"))
        dataset = group.create_dataset(
            dataset_name,
            data=data,
            compression=self.compression,
        )
        return dataset

    def _sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _infer_file_type(suffix: str) -> str:
        if suffix in TABLE_EXTENSIONS:
            return "table"
        if suffix in JSON_EXTENSIONS:
            return "json"
        if suffix in YAML_EXTENSIONS:
            return "yaml"
        if suffix in TEXT_EXTENSIONS:
            return "text"
        if suffix in FIGURE_EXTENSIONS:
            return "figure"
        return "unknown"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package report artefacts into HDF5")
    parser.add_argument("report_dir", type=Path, help="Report directory to consolidate")
    parser.add_argument(
        "--output",
        type=Path,
        help="Target HDF5 file. Defaults to <report_dir>.h5 in the parent directory.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Exclude binary figure artefacts (PNG, PDF, etc.)",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        help="Compression codec to apply inside the HDF5 file (default: gzip)",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    report_dir: Path = args.report_dir
    if args.output is None:
        output = report_dir.with_suffix(".h5")
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    packager = HDF5ReportPackager(
        report_dir=report_dir,
        include_figures=not args.no_figures,
        compression=args.compression or None,
    )
    records = packager.pack(output)
    print(f"Stored {len(records)} artefacts in {output}")


def dataset_to_dataframe(dataset: h5py.Dataset) -> pd.DataFrame:
    """Convert a structured HDF5 dataset back into a :class:`pandas.DataFrame`."""

    array = dataset[()]
    if array.dtype.names is None:
        raise ValueError("Dataset does not contain named columns")

    data = {}
    for name in array.dtype.names:
        values = array[name]
        dtype = array.dtype.fields[name][0]
        if h5py.check_string_dtype(dtype) is not None:
            decoded = [value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value) for value in values]
            data[name] = decoded
        else:
            data[name] = values
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
