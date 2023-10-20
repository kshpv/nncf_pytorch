# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from nncf.common.tensor import TensorType
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import NNCFTensor
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.quantization.advanced_parameters import AggregatorType

InplaceInsertionFNType = TypeVar("InplaceInsertionFNType")
AggregationAxes = Tuple[int, ...]


##################################################Reducers##################################################
class TensorReducerBase(ABC):
    """
    Tensor reducer is a callable object that reduces tensors according to
    the specified rule. Could handle tensors inplace or out of place.
    """

    def __init__(self, reduction_axes: Optional[ReductionAxes] = None, inplace: bool = False):
        """
        :param reduction_axes: Reduction axes for reduction calculation. Equal to list(range(len(input.shape)))
            if empty.
        :param inplace: Whether should be calculated inplace or out of place.
        """
        self._reduction_axes = reduction_axes
        self._tensor_processor: NNCFCollectorTensorProcessor = self._get_processor()
        self._inplace = inplace
        self._keepdims = True

    @property
    def inplace(self):
        return self._inplace

    @property
    def output_port_id(self) -> int:
        return 0

    @property
    def name(self):
        return self.__class__.__name__ + str(self.__hash__())

    @staticmethod
    @abstractmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        pass

    @abstractmethod
    def _reduce_out_of_place(self, x: List[TensorType]) -> List[TensorType]:
        """
        Specifies the reduction rule in terms of NNCFCollectorTensorProcessor.

        :param x: Tensor to register.
        """

    @abstractmethod
    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        """
        Returns target output names from target model that is
            modified for statistic collection.

        :param target_node_name: Target node name for reducer.
        :param port_id: Target port id for target node name for reducer.
        :return: Target output names for reducer.
        """

    @abstractmethod
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        """
        Returns correspondent inplace operation builder if inplace operations are available in backend.

        :return: Inplace operation builder if possible else None.
        """

    def __call__(self, x: List[NNCFTensor]):
        if self.inplace:
            return x

        return self._reduce_out_of_place(x)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, self.__class__)
            and self._reduction_axes == __o._reduction_axes
            and self._inplace == __o.inplace
        )

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes))

    def _get_reduction_axes(self, tensor: NNCFTensor) -> ReductionAxes:
        if self._reduction_axes is not None:
            return self._reduction_axes
        return tuple(range(len(tensor.shape)))


class NoopReducer(TensorReducerBase):
    def __init__(self):
        super().__init__(inplace=False)

    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return None

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def _reduce_out_of_place(self, x: List[TensorType]) -> List[TensorType]:
        return x


class MinReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.reduce_min(x, reduction_axes, keepdims=self._keepdims)]


class MaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.reduce_max(x, reduction_axes, keepdims=self._keepdims)]


class AbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = self._tensor_processor.abs(x[0])
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.reduce_max(x, reduction_axes, keepdims=self._keepdims)]


class MeanReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.mean(x, reduction_axes, keepdims=self._keepdims)]


class QuantileReducerBase(TensorReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        quantile: Optional[Union[float, Tuple[float]]] = None,
        inplace: bool = False,
    ):
        super().__init__(reduction_axes=reduction_axes, inplace=False)
        self._quantile = (0.01, 0.99) if quantile is None else quantile

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, tuple(self._quantile)))


class QuantileReducer(QuantileReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return self._tensor_processor.quantile(x, self._quantile, reduction_axes, keepdims=self._keepdims)


class AbsQuantileReducer(QuantileReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        quantile: Optional[Union[float, List[float]]] = None,
        inplace: bool = False,
    ):
        quantile = (0.99,) if quantile is None else quantile
        super().__init__(reduction_axes=reduction_axes, quantile=quantile, inplace=False)

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = self._tensor_processor.abs(x[0])
        reduction_axes = self._get_reduction_axes(x)
        return self._tensor_processor.quantile(x, self._quantile, reduction_axes, keepdims=self._keepdims)


class BatchMeanReducer(TensorReducerBase):
    def __init__(self, inplace: bool = False):
        super().__init__(None, inplace)

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        return [self._tensor_processor.batch_mean(x[0])]


class MeanPerChReducer(TensorReducerBase):
    def __init__(self, channel_axis: int = 1, inplace: bool = False):
        super().__init__(inplace=inplace)
        self._channel_axis = channel_axis

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        return [self._tensor_processor.mean_per_channel(x[0], self._channel_axis)]

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._channel_axis == __o._channel_axis

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, self._channel_axis))


##################################################Aggregators##################################################


class AggregatorBase:
    """
    Base class for aggregators which are using aggregation function fn which
    does not fulfill property fn([x1, x2, x3]) == fn([fn([x1, x2]), x3])
    where x1, x2, x3 are samples to aggregate. Aggregator collects
    all samples in a container and aggregates them in one step.
    """

    def __init__(
        self,
        tensor_processor: NNCFCollectorTensorProcessor,
        aggregation_fn,
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
    ):
        """
        :param tensor_processor: Backend-specific tensor processor.
        :param aggregation_axes: Axes along which to operate.
        :param num_samples: Maximum number of samples to collect. Aggregator
        skips tensor registration if tensor registration was called num_samples times before.
        Aggregator never skips registration if num_samples is None.
        """

        self._tensor_processor = tensor_processor
        self._aggregation_fn = aggregation_fn
        self._tensor_aggregation_axes = tuple(aggregation_axes) if aggregation_axes is not None else aggregation_axes
        self._stacked_tensor_aggregation_axis = 0
        if self._tensor_aggregation_axes is not None:
            self._stacked_tensor_all_aggregation_axes = (
                self._stacked_tensor_aggregation_axis,
                *map(lambda x: x + 1, self._tensor_aggregation_axes),
            )
        else:
            self._stacked_tensor_all_aggregation_axes = self._stacked_tensor_aggregation_axis
        self._keepdims = True
        self._num_samples = num_samples
        self._collected_samples = 0
        self._container = []

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def register_tensor(self, x: NNCFTensor) -> None:
        if self.is_enough_collection():
            return None
        self._container.append(x)
        self._collected_samples = len(self._container)

    def aggregate(self) -> Any:
        """
        Aggregates collected tensors and returns aggregated result.
        In case no tensors were collected returns None.

        :return: Aggregated result.
        """
        if self._collected_samples:
            stacked = self._tensor_processor.stack(self._container, axis=self._stacked_tensor_aggregation_axis)
            aggregated = self._aggregation_fn(
                stacked, axis=self._stacked_tensor_all_aggregation_axes, keepdims=self._keepdims
            )
            return self._postprocess_output(aggregated)
        return None

    def reset(self):
        self._container = []
        self._collected_samples = len(self._container)

    def is_enough_collection(self):
        if self._num_samples is not None and self._collected_samples >= self._num_samples:
            return True

    def _postprocess_output(self, aggregated):
        return self._tensor_processor.squeeze(aggregated, self._stacked_tensor_aggregation_axis).tensor


class NoopAggregator(AggregatorBase):  # TODO: change by NoopAggregator with num_smaples=1
    def __init__(self, tensor_processor, num_samples: Optional[int] = None):
        super().__init__(tensor_processor, lambda x, axis, keepdims: x, num_samples=num_samples)

    def _postprocess_output(self, aggregated):
        return aggregated.tensor


class ShapeAggregator(AggregatorBase):  # TODO: change by NoopAggregator with num_smaples=1
    def __init__(self, tensor_processor):
        super().__init__(tensor_processor, lambda x, axis, keepdims: x, num_samples=1)

    def _postprocess_output(self, aggregated):
        return self._tensor_processor.squeeze(aggregated, self._stacked_tensor_aggregation_axis).shape


class PercentileAggregator(AggregatorBase):
    def __init__(
        self,
        tensor_processor: NNCFCollectorTensorProcessor,
        aggregation_fn,
        percentiles_to_collect: List[float],
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
    ):
        super().__init__(tensor_processor, aggregation_fn, aggregation_axes=aggregation_axes, num_samples=num_samples)
        self._percentiles_to_collect = percentiles_to_collect

    def _postprocess_output(self, percentiles):
        retval = {}
        for idx, percentile in enumerate(self._percentiles_to_collect):
            retval[percentile] = self._tensor_processor.squeeze(
                percentiles[idx], self._stacked_tensor_aggregation_axis
            ).tensor
        return retval


class MedianAbsoluteDeviationAggregator(AggregatorBase):
    def _postprocess_output(self, vals):
        median_per_ch, mad_values = vals
        squeezed_median_per_ch = self._tensor_processor.squeeze(median_per_ch, self._stacked_tensor_aggregation_axis)
        squeezed_mad_values = self._tensor_processor.squeeze(mad_values, self._stacked_tensor_aggregation_axis)
        return {
            MedianMADTensorStatistic.MEDIAN_VALUES_STAT: squeezed_median_per_ch.tensor,
            MedianMADTensorStatistic.MAD_VALUES_STAT: squeezed_mad_values.tensor,
        }


class OnlineAggregatorBase(AggregatorBase):
    """
    Base class for aggregators which are using aggregation function fn with following property:
    fn([x1, x2, x3]) == fn([fn([x1, x2]), x3]) where x1, x2, x3 are samples to aggregate.
    Online aggregation fn([fn([x1, x2]), x3]) allows to keep memory stamp low as only
    one sample is stored during statistic collection.
    """

    def register_tensor(self, x: NNCFTensor) -> None:
        """
        The function aggregates firstly the input tensor.
        :param NNCFTensor x: _description_
        """
        if self.is_enough_collection():
            return None
        if self._tensor_aggregation_axes is not None:  # Should aggregate firstly the tensor
            x = self._aggregation_fn(x, axis=self._tensor_aggregation_axes, keepdims=self._keepdims)
        stacked_tensors = self._tensor_processor.stack(
            [x, *self._container], axis=self._stacked_tensor_aggregation_axis
        )
        aggregated_tensors = self._aggregation_fn(
            stacked_tensors, axis=self._stacked_tensor_aggregation_axis, keepdims=self._keepdims
        )
        self._container = [self._tensor_processor.squeeze(aggregated_tensors, self._stacked_tensor_aggregation_axis)]
        self._collected_samples += 1


class AggregatorFactory:
    # TODO: should be updated with common Tensor
    AGGREGATORS_MAP = {
        AggregatorType.MIN: (OnlineAggregatorBase, NNCFCollectorTensorProcessor.reduce_min),
        AggregatorType.MAX: (OnlineAggregatorBase, NNCFCollectorTensorProcessor.reduce_max),
        AggregatorType.MEAN: (AggregatorBase, NNCFCollectorTensorProcessor.mean),
        AggregatorType.MEAN_NO_OUTLIERS: (AggregatorBase, NNCFCollectorTensorProcessor.masked_mean),
        AggregatorType.MEDIAN: (AggregatorBase, NNCFCollectorTensorProcessor.median),
        AggregatorType.MEDIAN_NO_OUTLIERS: (AggregatorBase, NNCFCollectorTensorProcessor.masked_median),
        AggregatorType.PERCENTILE: (PercentileAggregator, NNCFCollectorTensorProcessor.percentile),
        AggregatorType.MEDIAN_ABSOLUTE_DEVIATION: (
            MedianAbsoluteDeviationAggregator,
            NNCFCollectorTensorProcessor.percentile,
        ),
    }

    @staticmethod
    def create_aggregator(
        aggregator_type: AggregatorType, tensor_processor, num_samples=None, aggregation_axes=None, **func_kwargs
    ):
        aggregator_cls, aggregation_fn = AggregatorFactory.AGGREGATORS_MAP[aggregator_type]
        if "percentiles_to_collect" in func_kwargs:
            return aggregator_cls(
                tensor_processor,
                aggregation_fn=AggregatorFactory._get_func(
                    aggregator_type,
                    tensor_processor,
                    **{"percentile": func_kwargs["percentiles_to_collect"]},
                ),
                aggregation_axes=aggregation_axes,
                num_samples=num_samples,
                **func_kwargs,
            )
        return aggregator_cls(
            tensor_processor,
            aggregation_fn=AggregatorFactory._get_func(aggregator_type, tensor_processor, **func_kwargs),
            aggregation_axes=aggregation_axes,
            num_samples=num_samples,
        )

    # TODO: should be removed after common Tensor  update
    @staticmethod
    def _get_func(aggregator_type, tensor_processor, **kwargs):
        if aggregator_type == AggregatorType.MIN:
            return tensor_processor.reduce_min
        if aggregator_type == AggregatorType.MAX:
            return tensor_processor.reduce_max
        if aggregator_type == AggregatorType.MEAN:
            return tensor_processor.mean
        if aggregator_type == AggregatorType.MEDIAN:
            return tensor_processor.median
        if aggregator_type == AggregatorType.MEAN_NO_OUTLIERS:
            return partial(tensor_processor.mean_outliers_mask, **kwargs)
        if aggregator_type == AggregatorType.MEDIAN_NO_OUTLIERS:
            return partial(tensor_processor.median_outliers_mask, **kwargs)
        if aggregator_type == AggregatorType.PERCENTILE:
            return partial(tensor_processor.percentile, **kwargs)
        if aggregator_type == AggregatorType.MEDIAN_ABSOLUTE_DEVIATION:
            return tensor_processor.median_absolute_deviation


class TensorCollector:
    """
    Calculates statistics at given tensors according to registered statistic branches.
    Statistic branch consists of one reducer and one aggregator instance. TensorCollector
    applies a reducer on a correspondent inputs and then passes the one of the reduced tensors
    chosen by output port id to a correspondent aggregator for each registered statistic branch.
    Receives tensors by `register_input` method. Aggregated values as a TensorStatistic instance or
    a dict could be collected by `get_statistics` call.
    """

    def __init__(self, statistic_container: Optional[TensorStatistic] = None) -> None:
        self._reducers: Set[TensorReducerBase] = set()
        self._aggregators: Dict[Tuple[int, int, int], AggregatorBase] = {}
        self._stat_container_kwargs_map: Dict[str, Tuple[int, int, int]] = {}
        self._stat_container = statistic_container
        self._enabled = True

    @property
    def num_samples(self) -> Optional[int]:
        output = None
        for aggregator in self._aggregators.values():
            if aggregator.num_samples and output:
                output = max(output, aggregator.num_samples)
            else:
                output = aggregator.num_samples
        return output

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reducers(self):
        return self._reducers.copy()

    @property
    def aggregators(self):
        return self._aggregators.copy()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def register_statistic_branch(
        self,
        container_key: str,
        reducer: TensorReducerBase,
        aggregator: AggregatorBase,
        reducer_output_port_id: int = 0,
    ) -> None:
        """
        Registers statistic collection branch for a container key. Correspondent input will be reduced
        by given reducer and reduced value will be registered and aggregated by given aggregator.
        Passed container key should be unique for the TensorCollector instance.
        Passed aggregator instance should never be used twice for one TensorCollector instance.

        :param container_key: Container key to pass aggregated statistic to.
        :param reducer: TensorReducer instance for the statistic collection branch.
        :param aggregator: TensorAggregator instance for the statistic collection branch.
        :reducer_output_port_id: Reducer target output port id.
        """
        if container_key in self._stat_container_kwargs_map:
            raise RuntimeError(
                f"Two different statistic branches for one container key {container_key} are encountered"
            )
        if any(aggr is aggregator for aggr in self._aggregators.values()):
            raise RuntimeError(f"One aggregator instance {aggregator} for different branches is encountered")

        self._reducers.add(reducer)
        key = (hash(reducer), reducer_output_port_id, hash(aggregator))

        if key not in self._aggregators:
            self._aggregators[key] = aggregator
        self._stat_container_kwargs_map[container_key] = key

    def get_output_info(self, target_node_name: str, port_id: int) -> List[Tuple[int, List[str]]]:
        """
        Returns list of pairs of reducers names and correspondent output names.

        :param target_node_name: Target node name to assemble output name.
        :param port_id: Target node specific port id to assemble output name.
        :returns: List of pairs of reducers hashes and correspondent output names.
        """
        retval = []
        for reducer in self._reducers:
            retval.append((hash(reducer), reducer.get_output_names(target_node_name, port_id)))
        return retval

    def register_inputs(self, inputs: Dict[int, List[NNCFTensor]]) -> None:
        """
        Registers given input in TensorCollector.

        :param inputs: Tensor inputs in format of dict where keys
            are reducer names and values are correspondent input tensors
        """
        if not self._enabled:
            return

        reduced_inputs = {}
        for reducer in self._reducers:
            reducer_hash = hash(reducer)
            input_ = inputs[reducer_hash]
            if any(tensor.is_empty() for tensor in input_):
                continue
            reduced_inputs[reducer_hash] = reducer(input_)

        for (
            (reducer_hash, reducer_port_id, _),
            aggregator,
        ) in self._aggregators.items():
            if reducer_hash in reduced_inputs:
                aggregator.register_tensor(reduced_inputs[reducer_hash][reducer_port_id])

    def register_input_for_all_reducers(self, input_: NNCFTensor) -> None:
        """
        Registers given input_ in each avaliable statistic collection branch.

        :param input_: Tensor input to register.
        """
        self.register_inputs({hash(reducer): [input_] for reducer in self._reducers})

    def _aggregate(self) -> None:
        result = {}
        for (
            key,
            aggregator,
        ) in self._aggregators.items():
            val = aggregator.aggregate()
            result[key] = val
        return result

    def get_statistics(self) -> Union[TensorStatistic, Dict[str, Any]]:
        """
        Returns aggregated values in format of a TensorStatistic instance or
        a dict.

        :returns: Aggregated values.
        """

        aggregated_values = self._aggregate()
        kwargs = {}
        for container_key, branch_key in self._stat_container_kwargs_map.items():
            kwargs[container_key] = aggregated_values[branch_key]

        if not self._stat_container:
            return kwargs
        return self._build_statistic_container(self._stat_container, kwargs)

    def get_inplace_fn_info(self) -> List[Tuple[Any, int]]:
        """
        Returns necessary information to insert inplace operation into graph.

        :returns: necessary information to insert inplace operation into graph
            in format of pair of reducer builder and correspondent reducer output port id.
        """
        retval = []
        for reducer in self._reducers:
            if reducer.inplace:
                retval.append((reducer.get_inplace_fn(), reducer.output_port_id))
        return retval

    def any_stat_out_of_place(self) -> bool:
        """
        Returns True if any reducer is calculated out of place.

        :returns: True if any reducer is calculated out of place.
        """
        return any(not reducer.inplace for reducer in self._reducers)

    def replace_aggregator(self, key: Tuple[int, int, int], aggregator: AggregatorBase) -> None:
        """
        Friend method that replaces aggregator instance on equivalent one.
        Key should be valid for for given aggregator and a statistic branch
        with key should be present in TensorCollector.

        :param key: Statistic branch key.
        :param aggregator: Aggregator instance to replace existing instance by given key.
        """
        assert key in self._aggregators
        assert key[2] == hash(aggregator)
        self._aggregators[key] = aggregator

    def reset(self):
        for aggregator in self._aggregators.values():
            aggregator.reset()

    @staticmethod
    def get_tensor_collector_inputs(
        outputs: Dict[str, NNCFTensor], output_info: List[Tuple[int, List[str]]]
    ) -> Dict[int, List[NNCFTensor]]:
        """
        Static method that converts all model outputs and collected output_info
        to a layout required for `register_inputs` method. This method is not a part of
        `register_inputs` to avoid all inputs passing to `TensorCollector.register_inputs` method.

        :param outputs: Target model outputs.
        :param output_info: Output info collected by a `TensorCollector.get_output_info` method.
        :returns: Model outputs in a format required by `TensorCollector.register_inputs` method.
        """
        target_inputs = {}
        for reducer, names in output_info:
            target_inputs[reducer] = [outputs[name] for name in names]
        return target_inputs

    @staticmethod
    def _build_statistic_container(statistic_container_cls: Type[TensorStatistic], kwargs: Dict[Any, Any]):
        if issubclass(statistic_container_cls, MinMaxTensorStatistic):
            return statistic_container_cls(
                min_values=kwargs[MinMaxTensorStatistic.MIN_STAT], max_values=kwargs[MinMaxTensorStatistic.MAX_STAT]
            )
        if issubclass(statistic_container_cls, MeanTensorStatistic):
            return statistic_container_cls(
                mean_values=kwargs[MeanTensorStatistic.MEAN_STAT], shape=kwargs[MeanTensorStatistic.SHAPE_STAT]
            )
        if issubclass(statistic_container_cls, RawTensorStatistic):
            return statistic_container_cls(values=kwargs[RawTensorStatistic.VALUES_STATS])
        if issubclass(statistic_container_cls, MedianMADTensorStatistic):
            return statistic_container_cls(
                median_values=kwargs[MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY][
                    MedianMADTensorStatistic.MEDIAN_VALUES_STAT
                ],
                mad_values=kwargs[MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY][
                    MedianMADTensorStatistic.MAD_VALUES_STAT
                ],
            )
        if issubclass(statistic_container_cls, PercentileTensorStatistic):
            if PercentileTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY in kwargs:
                percentile_vs_values_dict = kwargs[PercentileTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY]
            else:
                percentile_vs_values_dict = {}
                for (_, percentile), value in kwargs.items():
                    percentile_vs_values_dict[percentile] = value
            return statistic_container_cls(percentile_vs_values_dict=percentile_vs_values_dict)
        raise RuntimeError(
            f"Statistic collector class {statistic_container_cls} is not supported by the TensorCollector class."
        )


class MergedTensorCollector(TensorCollector):
    """
    Tensor collector that merge several tensor collectors in one.
    Statistics collected by a merged tensor collector automatically available
    in all tensor collectors that were merged by the merged tensor collector.
    This works because merged tensor collectors share tensor aggregators instances with
    the merged tensor collector.
    """

    def __init__(self, tensor_collectors: List[TensorCollector]) -> None:
        """
        :param tensor_collectors: Tensor collectors to merge.
        """
        super().__init__()
        aggregators: Dict[Tuple[int, int, int], List[Tuple[TensorCollector, AggregatorBase]]] = defaultdict(list)
        for tensor_collector in tensor_collectors:
            if not tensor_collector.enabled:
                continue
            self._reducers.update(tensor_collector.reducers)
            for key, aggregator in tensor_collector.aggregators.items():
                aggregators[key].append((tensor_collector, aggregator))

        for key, aggregators_to_merge in aggregators.items():
            _, unique_aggregator = aggregators_to_merge[0]
            for tensor_collector, _ in aggregators_to_merge[1:]:
                tensor_collector.replace_aggregator(key, unique_aggregator)
            self._aggregators[key] = unique_aggregator
