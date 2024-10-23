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
import gzip
import pickle
from abc import ABC
from abc import abstractmethod
from itertools import islice
from typing import Any, Dict, Optional, TypeVar

import nncf
from nncf.common import factory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.data.dataset import DataItem
from nncf.data.dataset import Dataset
from nncf.data.dataset import ModelInput
from nncf.experimental.common.tensor_statistics.statistics import TensorStatistic

TensorType = TypeVar("TensorType")
TModel = TypeVar("TModel")

EMPTY_DATASET_ERROR = (
    "Calibration dataset must not be empty. Please provide calibration dataset with at least one sample."
)


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    BACKEND = None

    def __init__(self, dataset: Dataset[DataItem, ModelInput]):
        self.dataset = dataset
        self.stat_subset_size = None
        self.statistic_points = StatisticPointsContainer()

    def _get_iterations_number(self) -> Optional[int]:
        """
        Returns number of iterations.

        :return: Number of iterations for statistics collection.
        """
        dataset_length = self.dataset.get_length()
        if dataset_length and self.stat_subset_size:
            return min(dataset_length, self.stat_subset_size)
        return dataset_length or self.stat_subset_size

    def collect_statistics(self, model: TModel, graph: NNCFGraph) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model: Backend-specific model instance.
        :param graph: Model graph.
        """
        if not self.statistic_points:
            return
        model_transformer = factory.ModelTransformerFactory.create(model)
        merged_statistics = self._get_merged_statistic_points(self.statistic_points, model, graph)
        transformation_layout = self._get_transformation_layout_extra_outputs(merged_statistics)
        model_with_outputs: TModel = model_transformer.transform(transformation_layout)
        engine = factory.EngineFactory.create(model_with_outputs)
        iterations_number = self._get_iterations_number()
        processed_samples = 0
        for input_data in track(  # type: ignore
            islice(self.dataset.get_inference_data(), iterations_number),
            total=iterations_number,
            description="Statistics collection",
        ):
            outputs = engine.infer(input_data)
            processed_outputs = self._process_outputs(outputs)
            self._register_statistics(processed_outputs, merged_statistics)
            processed_samples += 1
        if processed_samples == 0:
            raise nncf.ValidationError(EMPTY_DATASET_ERROR)
        if self.stat_subset_size is not None and self.stat_subset_size > processed_samples:
            nncf_logger.warning(
                f"Dataset contains only {processed_samples} samples, "
                f"smaller than the requested subset size {self.stat_subset_size}."
            )

    def load_statistics_from_file(self, file_name: str) -> None:
        """
        Loads statistics from a file and populates the statistic points with the loaded data.

        :param file_name: The name of the file from which to load the statistics.
        """
        loaded_data = StatisticsSerializer.load_from_file(file_name)
        if not StatisticsValidator.check_backend(loaded_data, self.BACKEND):
            raise nncf.ValidationError("Backend key in loaded statistics is not matched to a model backend.")
        self._load_statistics(loaded_data)
        nncf_logger.info(f"Statistics were successfully loaded from a file {file_name}.")

    def _load_statistics(self, data: Dict[str, Any]) -> None:
        """
        Loads statistics into the registered statistic points from the given data.

        :param data: A dictionary containing the statistics loaded from a file.
        """
        for _, statistic_point, tensor_collector in self.statistic_points.get_tensor_collectors():
            statistics = tensor_collector.get_statistics()
            statistics_key = self._get_statistics_key(statistics, statistic_point.target_point)
            if statistics_key not in data:
                raise nncf.ValidationError(f"Not found statistics for {statistics_key}")
            statistics.load_data(data[statistics_key])

    def dump_statistics(self, file_name: str) -> None:
        """
        Dumps the current statistics to a file in a compressed format.

        :param file_name: The name of the file where the statistics will be saved.
        """
        data_to_dump = self._prepare_statistics()
        StatisticsSerializer.dump_to_file(data_to_dump, file_name)
        nncf_logger.info(f"Statistics were successfully saved to a file {file_name}.")

    def _prepare_statistics(self) -> Dict[str, Any]:
        """
        Prepares the statistics data for dumping into a file.

        :return: A dictionary containing the statistics data to be dumped.
        """
        data_to_dump = {"backend": self.BACKEND}
        for _, statistic_point, tensor_collector in self.statistic_points.get_tensor_collectors():
            statistics = tensor_collector.get_statistics()
            statistics_key = self._get_statistics_key(statistics, statistic_point.target_point)
            data = statistics.get_data()
            data_to_dump[statistics_key] = data
        return data_to_dump

    def register_statistic_points(self, statistic_points: StatisticPointsContainer) -> None:
        """
        Register statistic points for statistics collection and recalculates the maximum number samples
        for collecting statistics, based on the maximum value from the all algorithms.

        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """
        for _, _statistic_points in statistic_points.items():
            for _statistic_point in _statistic_points:
                self.statistic_points.add_statistic_point(_statistic_point)

        for _, _statistic_points in self.statistic_points.items():
            for _statistic_point in _statistic_points:
                for _, tensor_collectors in _statistic_point.algorithm_to_tensor_collectors.items():
                    for tensor_collector in tensor_collectors:
                        if self.stat_subset_size is None:
                            self.stat_subset_size = tensor_collector.num_samples
                        elif tensor_collector.num_samples is not None:
                            self.stat_subset_size = max(self.stat_subset_size, tensor_collector.num_samples)

    @abstractmethod
    def _register_statistics(self, outputs: Dict[str, NNCFTensor], statistic_points: StatisticPointsContainer) -> None:
        """
        Process prepared raw model outputs and statistic points for the further usage.

        :param outputs: prepared raw model outputs
        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """

    @abstractmethod
    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        """
        Creates backend-specific transformation layout for the further statistics collection.

        :param statistic_points: StatisticPointsContainer to add outputs
        :return: TransformationLayout with the corresponding transformations
        """

    @staticmethod
    @abstractmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: TModel, graph: NNCFGraph
    ) -> StatisticPointsContainer:
        """
        Creates a new StatisticPointContainer that has no duplicated tensor collectors for one
        unique statistic point. Alters statistic collectors in the given statistic point container so statistics
        collected by merged statistic collectors will be available in all corresponding statistic collectors
        from the given statistic point container.

        :param statistic_points: Registered statistic points with possible tensor collectors duplicates.
        :param model: Backend-specific target model.
        :param graph: Model graph.
        :return: Merged statistic points container bounded with given statistic point container.
        """

    @staticmethod
    @abstractmethod
    def _process_outputs(outputs: Any) -> Dict[str, NNCFTensor]:
        """
        Post-process model outputs for the further statistics collection.

        :param outputs: raw model outputs
        :return: processed model outputs in Dict[str, Tensor] format
        """

    @abstractmethod
    def _get_statistics_key(self, statistics: TensorStatistic, target_point: TargetPoint) -> str:
        """
        Returns key of statistics.

        :param statistics: Statistics value.
        :param target_point: Statistics target point.
        :return: Statistics key.
        """


class StatisticsValidator:
    @staticmethod
    def check_backend(data: Dict[str, Any], backend: Optional[BackendType]) -> bool:
        """
        Checks whether backend in loaded data is equal to a provided backend.

        :param data: Loaded statistics.
        :param backend: Provided backend.
        :return: True, if matched, False - otherwise.
        """
        return bool(data["backend"] == backend)


class StatisticsSerializer:
    @staticmethod
    def load_from_file(file_name: str) -> Any:
        """
        Loads statistics from a gzip-compressed file.
        :param file_name: The name of the file from which to load the statistics.
        :return: The loaded statistics.
        """
        try:
            with gzip.open(file_name, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise nncf.ValidationError(f"File not found: {file_name}")
        except (pickle.UnpicklingError, IOError):
            raise nncf.ValidationError(f"Error loading statistics from {file_name}")

    @staticmethod
    def dump_to_file(statistics: Dict[str, TensorType], file_name: str) -> None:
        """
        Dumps statistics to a gzip-compressed file.
        :param data: The statistics to be dumped.
        :param file_name: The name of the file where the statistics will be dumped.
        """
        try:
            with gzip.open(file_name, "wb") as f:
                pickle.dump(statistics, f)
        except (IOError, pickle.PicklingError) as e:
            nncf_logger.error(f"Failed to write data to file {file_name}: {e}")
            raise
