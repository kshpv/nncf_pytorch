# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, cast

import nncf
from nncf.common.utils.os import safe_open
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.tensor import TTensor

METADATA_FILE = "statistics_metadata.json"


def sanitize_filename(filename: str) -> str:
    """
    Replaces any forbidden characters with an underscore.

    :param filename: Original filename.
    :return: Sanitized filename with no forbidden characters.
    """
    return re.sub(r"[^\w]", "_", filename)


def load_metadata(dir_path: Path) -> Dict[str, Any]:
    """
    Loads the metadata, including the mapping and any other metadata information from the metadata file.

    :param dir_path: The directory where the metadata file is stored.
    :return: A dictionary containing the metadata.
    """
    metadata_file = dir_path / METADATA_FILE
    if metadata_file.exists():
        with safe_open(metadata_file, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    raise nncf.InvalidPathError("Metadata file does not exist.")


def save_metadata(metadata: Dict[str, Any], dir_path: Path) -> None:
    """
    Saves metadata to a file in the specified directory.

    :param metadata: Dictionary containing metadata and mapping.
    :param dir_path: Path to the directory where the metadata file will be saved.
    """
    metadata_file = dir_path / METADATA_FILE
    with safe_open(metadata_file, "w") as f:
        json.dump(metadata, cast(TextIO, f), indent=4)


def load_from_dir(dir_path: Path, backend: TensorBackend) -> Tuple[Dict[str, Dict[str, TTensor]], Dict[str, Any]]:
    """
    Loads statistics and metadata from a directory.

    :param dir_path: The path to the directory from which to load the statistics.
    :param backend: Backend type to determine the loading function.
    :return: Tuple containing statistics and metadata.
    """
    statistics: Dict[str, Dict[str, TTensor]] = {}
    if not dir_path.exists():
        raise nncf.ValidationError("The provided directory path does not exist.")

    metadata = load_metadata(dir_path)
    mapping = metadata.get("mapping", {})

    load_file_func = get_safetensors_backend_fn("load_file", backend)
    for file_name, original_name in mapping.items():
        statistics_file = dir_path / file_name
        if not statistics_file.exists():
            raise nncf.InternalError(
                f"No statistics file was found for {original_name}. Probably, metadata is corrupted."
            )
        statistics[original_name] = load_file_func(statistics_file)
    return statistics, {key: value for key, value in metadata.items() if key != "mapping"}


def dump_to_dir(
    statistics: Dict[str, Dict[str, TTensor]],
    dir_path: Path,
    backend: TensorBackend,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves statistics and metadata to a directory.

    :param statistics: A dictionary with statistic names as keys and the statistic data as values.
    :param dir_path: The path to the directory where the statistics will be dumped.
    :param backend: Backend type to determine the saving function.
    :param additional_metadata: A dictionary containing any additional metadata to be saved with the mapping.
    """
    dir_path.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {"mapping": {}}

    unique_map = defaultdict(list)

    save_file_func = get_safetensors_backend_fn("save_file", backend)
    for original_name, statistics_value in statistics.items():
        sanitized_name = sanitize_filename(original_name)

        # Next number of the same sanitized name
        count = len(unique_map[sanitized_name]) + 1
        unique_sanitized_name = f"{sanitized_name}_{count}"
        unique_map[sanitized_name].append(unique_sanitized_name)

        file_path = dir_path / unique_sanitized_name

        # Update the mapping
        metadata["mapping"][unique_sanitized_name] = original_name

        try:
            save_file_func(statistics_value, file_path)
        except Exception as e:
            raise nncf.InternalError(f"Failed to write data to file {file_path}: {e}")

    if additional_metadata:
        metadata |= additional_metadata

    save_metadata(metadata, dir_path)


def get_safetensors_backend_fn(fn_name: str, backend: TensorBackend) -> Callable[..., Any]:
    """
    Returns a function based on the provided function name and backend type.

    :param fn_name: The name of the function.
    :param backend: The backend type for which the function is required.
    :return: The backend-specific function.
    """
    if backend == TensorBackend.numpy:
        import safetensors.numpy as numpy_module

        return getattr(numpy_module, fn_name)
    if backend == TensorBackend.torch:
        import safetensors.torch as torch_module

        return getattr(torch_module, fn_name)
