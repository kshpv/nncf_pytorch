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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Tuple, Type

from nncf.tensor import Tensor


class TensorStatistic:
    """Base class that stores statistic data"""

    TENSOR_STATISTIC_OUTPUT_KEY = "tensor_statistic_output"


@dataclass
class MinMaxTensorStatistic(TensorStatistic):
    MIN_STAT: ClassVar[str] = "min_values"
    MAX_STAT: ClassVar[str] = "max_values"

    min_values: Tensor
    max_values: Tensor

    def get_data(self):
        return self.min_values.data, self.max_values.data

    def load_data(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values


@dataclass
class MeanTensorStatistic(TensorStatistic):
    MEAN_STAT: ClassVar[str] = "mean_values"
    SHAPE_STAT: ClassVar[str] = "shape"

    mean_values: Tensor
    shape: Tuple[int, ...]

    def get_data(self):
        return self.mean_values, self.shape

    def load_data(self, mean_values, shape):
        self.mean_values = mean_values
        self.shape = shape


@dataclass
class MedianMADTensorStatistic(TensorStatistic):
    MEDIAN_VALUES_STAT: ClassVar[str] = "median_values"
    MAD_VALUES_STAT: ClassVar[str] = "mad_values"

    median_values: Tensor
    mad_values: Tensor

    def get_data(self):
        return self.median_values, self.mad_values

    def load_data(self, median_values, mad_values):
        self.median_values = median_values
        self.mad_values = mad_values


@dataclass
class PercentileTensorStatistic(TensorStatistic):
    PERCENTILE_VS_VALUE_DICT: ClassVar[str] = "percentile_vs_values_dict"

    percentile_vs_values_dict: Dict[str, Tensor]

    def get_data(self):
        return self.percentile_vs_values_dict

    def load_data(self, percentile_vs_values_dict):
        self.percentile_vs_values_dict = percentile_vs_values_dict


@dataclass
class RawTensorStatistic(TensorStatistic):
    VALUES_STATS: ClassVar[str] = "values"

    values: Tensor

    def get_data(self):
        return self.values

    def load_data(self, values):
        self.values = values


@dataclass
class WeightQuantizationErrorTensorStatistic(TensorStatistic):
    WEIGHT_QUANTIZATION_ERROR_STATS: ClassVar[str] = "weight_quantization_error"

    weight_quantization_error: Tensor

    def get_data(self):
        return self.weight_quantization_error

    def load_data(self, weight_quantization_error):
        self.weight_quantization_error = weight_quantization_error


@dataclass
class HessianTensorStatistic(TensorStatistic):
    HESSIAN_INPUT_ACTIVATION_STATS: ClassVar[str] = "hessian"

    hessian: Tensor

    def get_data(self):
        return self.values

    def load_data(self, hessian):
        self.hessian = hessian


@dataclass
class MeanVarianceTensorStatistic(TensorStatistic):
    MEAN_VARIANCE_STAT: ClassVar[str] = "mean_variance"

    mean_variance: Tensor

    def get_data(self):
        return self.mean_variance

    def load_data(self, mean_variance):
        self.mean_variance = mean_variance


@dataclass
class MaxVarianceTensorStatistic(TensorStatistic):
    MAX_VARIANCE_STAT: ClassVar[str] = "max_variance"

    max_variance: Tensor

    def get_data(self):
        return self.max_variance

    def load_data(self, max_variance):
        self.max_variance = max_variance


@dataclass
class MeanMagnitudeTensorStatistic(TensorStatistic):
    MEAN_MAGNITUDE_STAT: ClassVar[str] = "mean_magnitude"

    mean_magnitude: Tensor

    def get_data(self):
        return self.mean_magnitude

    def load_data(self, mean_magnitude):
        self.mean_magnitude = mean_magnitude


def build_statistic_container(
    statistic_container_cls: Type[TensorStatistic], kwargs: Dict[Any, Any]
) -> TensorStatistic:
    return statistic_container_cls(**kwargs)
