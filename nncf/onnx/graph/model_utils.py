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
from collections import deque
from typing import Tuple

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import get_attribute_value

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.engine import ONNXEngine
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXQuantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import get_tensor_edge_name
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.quantization.quantizer_parameters import quantize_tensor


def remove_fq_from_inputs(model: onnx.ModelProto, nncf_graph: NNCFGraph) -> onnx.ModelProto:
    """
    This method removes the activation Quantizer nodes from the model.
    It's needed for the further bias shift calculation that relates on quantized weights.

    :param model: onnx.ModelProto instance.
    :param nncf_graph: NNCFGraph instance.
    :return: onnx.ModelProto instance without activation Quantizer nodes.
    """
    transformation_layout = TransformationLayout()
    model_transformer = ModelTransformerFactory.create(model)

    seen_nodes = []
    nodes_queue = deque(nncf_graph.get_input_nodes())
    while nodes_queue:
        current_node = nodes_queue.popleft()
        current_node_name = current_node.node_name

        if current_node_name in seen_nodes:
            continue

        seen_nodes.append(current_node_name)
        if current_node.metatype in [ONNXQuantizeLinearMetatype, ONNXDequantizeLinearMetatype]:
            target_point = ONNXTargetPoint(TargetType.LAYER, current_node_name, 0)
            command = ONNXQDQNodeRemovingCommand(target_point)
            transformation_layout.register(command)
        nodes_queue.extend(nncf_graph.get_next_nodes(current_node))

    return model_transformer.transform(transformation_layout)


def _get_q_linear_params(node: onnx.NodeProto, onnx_graph: ONNXGraph) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Returns axis, scale and zero_point parameters of QuantizeLinear node.

    :param node: QuantizeLinear node.
    :param onnx_graph: ONNXGraph.
    :return: axis, scale, zero_point parameters of the node.
    """
    assert node.op_type in ONNXQuantizeLinearMetatype.get_all_aliases()
    scale = onnx_graph.get_tensor_value(node.input[1])
    zero_point = onnx_graph.get_tensor_value(node.input[2])
    axis = None
    for attr in node.attribute:
        if attr.name == "axis":
            axis = get_attribute_value(attr)
    return axis, scale, zero_point


def _get_input_constant_tensor(node: onnx.NodeProto, port_id: int, onnx_graph: ONNXGraph) -> np.ndarray:
    """
    Returns an input constant tensor of a node.
    There are two cases:
    1) if there is a constant tensor directly used as input to a node - then returns this tensor.
    2) if there is a constant subgraph - extract this subgraph and infer subgraph returns an output tensor.

    :param node: ONNX node.
    :param onnx_graph: ONNXGraph.
    :return: Tensor value.
    """
    if onnx_graph.has_tensor(node.input[port_id]):
        return onnx_graph.get_tensor_value(node.input[port_id])
    extractor = onnx.utils.Extractor(onnx_graph.onnx_model)
    constant_subgraph = extractor.extract_model([], [node.input[port_id]])
    engine = ONNXEngine(constant_subgraph)
    outputs = engine.infer({})
    return outputs[node.input[port_id]]


def remove_node(node: onnx.NodeProto, onnx_graph: ONNXGraph) -> None:
    """
    Remove all parents node while saving

    :param onnx.NodeProto node: _description_
    :param ONNXGraph onnx_graph: _description_
    """
    for i in range(len(node.input)):
        parent = onnx_graph.get_parent(node, i)
        if parent:
            remove_node(parent, onnx_graph)

    node_children = onnx_graph.get_children(node)
    for node_child in node_children:
        for input_id, input_obj in enumerate(node_child.input):
            if input_obj == node.output[0]:
                node_child.input[input_id] = node.input[0]
    onnx_graph.onnx_model.graph.node.remove(node)


def compress_quantize_weights_transformation(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Returns model with compressed weights.
    All QuantizeLinear nodes, which inputs constant tensor, are applied to original precision weight tensor and, then removed from a model.

    :param model: ONNX model.
    :return: ONNX model with conpressed weights.
    """
    onnx_graph = ONNXGraph(model)
    for node in onnx_graph.get_all_nodes():
        if node.op_type in ONNXQuantizeLinearMetatype.get_all_aliases():
            initializer_to_update_name = get_tensor_edge_name(onnx_graph, node=node, port_id=0)
            if initializer_to_update_name:
                original_precision_weight = _get_input_constant_tensor(node, 0, onnx_graph)
                axis, scale, zero_point = _get_q_linear_params(node, onnx_graph)
                quantized_weight = quantize_tensor(original_precision_weight, axis, scale, zero_point)

                initializer = onnx_graph.get_tensor(initializer_to_update_name)
                int8_weight_tensor = numpy_helper.from_array(quantized_weight, name=initializer.name)
                initializer.CopyFrom(int8_weight_tensor)
                remove_node(node, onnx_graph)
                node.input[0] = initializer_to_update_name
    return model
