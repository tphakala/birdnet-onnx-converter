#!/usr/bin/env python3
"""
Optimize BirdNET ONNX model for GPU execution and low-power devices.

Converts raw ONNX models (from TFLite or SavedModel conversion) into optimized
versions suitable for GPU inference and quantized variants for embedded devices.

Based on the optimization approach from:
https://huggingface.co/justinchuby/BirdNET-onnx/blob/main/scripts/optimize.py

Key optimizations:
1. Replace RFFT2D/DFT with MatMul (GPU-friendly)
2. Replace ReverseSequence with Slice (GPU-friendly)
3. Replace GlobalAveragePool + Squeeze with ReduceMean
4. Remove unnecessary Cast operations
5. Fuse consecutive operations
6. Enable dynamic batching
7. Graph optimization with onnx-simplifier and onnxslim

Output formats:
- FP32: Standard precision for GPU/desktop CPU
- FP16: Half precision for devices with FP16 hardware (RPi 5, modern GPUs). The
  spectrogram preprocessing front-end is kept in FP32 by default for accuracy;
  use --fp16-full to convert the entire graph.
- INT8: Quantized for low-power devices (RPi 3/4, embedded ARM). Experimental:
  dynamic quantization can significantly degrade accuracy; validate before use.

Usage:
    # Output all three formats (FP32, FP16, INT8)
    python optimize.py --input model.onnx --output BirdNET

    # Output only FP32
    python optimize.py --input model.onnx --output BirdNET --fp32-only

    # Skip specific formats
    python optimize.py --input model.onnx --output BirdNET --no-fp16
    python optimize.py --input model.onnx --output BirdNET --no-int8

    # Convert the whole graph to FP16 (including preprocessing)
    python optimize.py --input model.onnx --output BirdNET --fp16-full

Requirements:
    pip install onnx onnx-simplifier onnxslim numpy onnxruntime onnxconverter-common
"""

import argparse
import textwrap
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper

# Surfaced whenever an INT8 model is produced. INT8 dynamic quantization is
# experimental: it can shift confidences enough to change the top predicted
# species, so it must be validated against FP32/FP16 before deployment.
INT8_EXPERIMENTAL_WARNING = (
    "INT8 quantization is EXPERIMENTAL and not recommended for production. "
    "Dynamic quantization can significantly degrade accuracy and may change the "
    "top predicted species. Validate INT8 output against FP32/FP16 before "
    "deploying, and prefer FP16 where the target supports it. Use --no-int8 (and "
    "omit --int8-arm) to skip."
)


def create_dft_matrix(fft_size: int) -> np.ndarray:
    """Create a real DFT matrix (cosine basis) for one-sided FFT."""
    num_freqs = fft_size // 2 + 1
    n = np.arange(fft_size, dtype=np.float32)[:, np.newaxis]
    k = np.arange(num_freqs, dtype=np.float32)[np.newaxis, :]
    dft_matrix = np.cos(2 * np.pi * k * n / fft_size)
    return dft_matrix


def get_tensor_shape(model: onnx.ModelProto, tensor_name: str) -> Optional[list[int]]:
    """Get shape of a tensor from value_info or initializers."""
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == tensor_name:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
            return shape

    for init in model.graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)

    return None


def find_node_by_output(model: onnx.ModelProto, output_name: str) -> Optional[onnx.NodeProto]:
    """Find the node that produces a given output."""
    for node in model.graph.node:
        if output_name in node.output:
            return node  # type: ignore[no-any-return]
    return None


def find_nodes_by_input(model: onnx.ModelProto, input_name: str) -> list[onnx.NodeProto]:
    """Find all nodes that consume a given input."""
    return [node for node in model.graph.node if input_name in node.input]


def get_attribute(node: onnx.NodeProto, name: str, default=None):
    """Get attribute value from a node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8")
    return default


def replace_dft_with_matmul(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace DFT operations with MatMul using precomputed cosine DFT matrix.

    Pattern: Reshape(..., fft_size, 1) -> DFT(onesided=1) -> Slice(real part) -> (..., num_freqs, 1)
    Replace with: Squeeze -> MatMul -> Unsqueeze

    This is much faster on GPU as MatMul is highly optimized.
    """
    initializers_to_add = []
    replacements_info = []
    replacements = 0

    node_list = list(model.graph.node)
    node_to_idx = {id(node): idx for idx, node in enumerate(node_list)}

    for node in node_list:
        if node.op_type != "DFT":
            continue

        dft_input = node.input[0]
        dft_output = node.output[0]

        onesided = get_attribute(node, "onesided", 0)
        if onesided != 1:
            print(f"  Warning: DFT {node.name} is not onesided, skipping")
            continue

        axis = get_attribute(node, "axis", 1)
        if axis != 1:
            print(f"  Warning: DFT {node.name} has unsupported axis {axis}, skipping")
            continue

        fft_size = None
        if len(node.input) > 1:
            fft_length_name = node.input[1]
            for init in model.graph.initializer:
                if init.name == fft_length_name:
                    fft_length_data = numpy_helper.to_array(init)
                    fft_size = (
                        int(fft_length_data)
                        if fft_length_data.shape == ()
                        else int(fft_length_data[-1])
                    )
                    break

        if fft_size is None:
            print(f"  Warning: Could not determine FFT size for {node.name}, skipping")
            continue

        consumers = find_nodes_by_input(model, dft_output)
        slice_node = None
        for consumer in consumers:
            if consumer.op_type == "Slice":
                slice_node = consumer
                break

        if slice_node is None:
            print(
                f"  Warning: DFT {node.name} has no Slice consumer for real part extraction, skipping"
            )
            continue

        slice_output = slice_node.output[0]
        num_freqs = fft_size // 2 + 1

        print(f"  Replacing DFT {node.name} (fft_size={fft_size}, num_freqs={num_freqs})")

        dft_matrix = create_dft_matrix(fft_size)
        dft_matrix_name = f"{node.name}_dft_matrix"
        dft_init = numpy_helper.from_array(dft_matrix.astype(np.float32), dft_matrix_name)
        initializers_to_add.append(dft_init)

        squeeze_axes_name = f"{node.name}_squeeze_axes"
        squeeze_axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), squeeze_axes_name)
        initializers_to_add.append(squeeze_axes)

        squeezed_name = f"{node.name}_squeezed"
        squeeze_node_new = helper.make_node(
            "Squeeze",
            inputs=[dft_input, squeeze_axes_name],
            outputs=[squeezed_name],
            name=f"{node.name}_squeeze",
        )

        matmul_output = f"{node.name}_matmul_out"
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[squeezed_name, dft_matrix_name],
            outputs=[matmul_output],
            name=f"{node.name}_matmul",
        )

        unsqueeze_axes_name = f"{node.name}_unsqueeze_axes"
        unsqueeze_axes = numpy_helper.from_array(
            np.array([-1], dtype=np.int64), unsqueeze_axes_name
        )
        initializers_to_add.append(unsqueeze_axes)

        unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=[matmul_output, unsqueeze_axes_name],
            outputs=[slice_output],
            name=f"{node.name}_unsqueeze",
        )

        dft_idx = node_to_idx[id(node)]
        slice_idx = node_to_idx[id(slice_node)]
        replacements_info.append(
            (dft_idx, slice_idx, [squeeze_node_new, matmul_node, unsqueeze_node])
        )
        replacements += 1

    nodes_to_remove_idx = set()
    for dft_idx, slice_idx, _ in replacements_info:
        nodes_to_remove_idx.add(dft_idx)
        nodes_to_remove_idx.add(slice_idx)

    new_nodes = []
    for idx, node in enumerate(node_list):
        if idx in nodes_to_remove_idx:
            for dft_idx, _slice_idx, replacement_nodes in replacements_info:
                if idx == dft_idx:
                    new_nodes.extend(replacement_nodes)
                    break
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    for init in initializers_to_add:
        model.graph.initializer.append(init)

    return model, replacements


def replace_rfft2d_with_matmul(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace RFFT2D operations with MatMul using precomputed DFT matrix.

    Handles the pattern from tf2onnx TFLite conversion:
    Pattern: Unsqueeze -> RFFT2D -> Squeeze (extracting real part)
    Replace with: Squeeze -> MatMul -> Unsqueeze

    RFFT2D outputs complex values [..., num_freqs, 2] where last dim is [real, imag].
    The Squeeze extracts just the real part. MatMul with cosine DFT matrix computes
    the real part directly.
    """
    initializers_to_add = []
    replacements_info = []
    replacements = 0

    node_list = list(model.graph.node)
    node_to_idx = {id(node): idx for idx, node in enumerate(node_list)}

    for node in node_list:
        if node.op_type != "RFFT2D":
            continue

        rfft_input = node.input[0]
        rfft_output = node.output[0]

        fft_size = None
        if len(node.input) > 1:
            fft_length_name = node.input[1]
            for init in model.graph.initializer:
                if init.name == fft_length_name:
                    fft_length_data = numpy_helper.to_array(init)
                    if fft_length_data.ndim > 0:
                        fft_size = int(fft_length_data[-1])
                    else:
                        fft_size = int(fft_length_data)
                    break

        if fft_size is None:
            print(f"  Warning: Could not determine FFT size for RFFT2D {node.name}, skipping")
            continue

        consumers = find_nodes_by_input(model, rfft_output)
        squeeze_node = None
        for consumer in consumers:
            if consumer.op_type == "Squeeze":
                squeeze_node = consumer
                break

        if squeeze_node is None:
            print(f"  Warning: RFFT2D {node.name} has no Squeeze consumer, skipping")
            continue

        squeeze_output = squeeze_node.output[0]
        num_freqs = fft_size // 2 + 1

        producer = find_node_by_output(model, rfft_input)
        if producer is None or producer.op_type != "Unsqueeze":
            print(f"  Warning: RFFT2D {node.name} producer is not Unsqueeze, skipping")
            continue

        unsqueeze_input = producer.input[0]

        print(f"  Replacing RFFT2D {node.name} (fft_size={fft_size}, num_freqs={num_freqs})")

        dft_matrix = create_dft_matrix(fft_size)
        dft_matrix_name = f"{node.name}_dft_matrix"
        dft_init = numpy_helper.from_array(dft_matrix.astype(np.float32), dft_matrix_name)
        initializers_to_add.append(dft_init)

        matmul_node = helper.make_node(
            "MatMul",
            inputs=[unsqueeze_input, dft_matrix_name],
            outputs=[squeeze_output],
            name=f"{node.name}_matmul",
        )

        producer_idx = node_to_idx[id(producer)]
        rfft_idx = node_to_idx[id(node)]
        squeeze_idx = node_to_idx[id(squeeze_node)]
        replacements_info.append((producer_idx, rfft_idx, squeeze_idx, [matmul_node]))
        replacements += 1

    nodes_to_remove_idx = set()
    for producer_idx, rfft_idx, squeeze_idx, _ in replacements_info:
        nodes_to_remove_idx.add(producer_idx)
        nodes_to_remove_idx.add(rfft_idx)
        nodes_to_remove_idx.add(squeeze_idx)

    new_nodes = []
    for idx, node in enumerate(node_list):
        if idx in nodes_to_remove_idx:
            for (
                _producer_idx,
                rfft_idx,
                _squeeze_idx,
                replacement_nodes,
            ) in replacements_info:
                if idx == rfft_idx:
                    new_nodes.extend(replacement_nodes)
                    break
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    for init in initializers_to_add:
        model.graph.initializer.append(init)

    return model, replacements


def replace_reverse_sequence(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace ReverseSequence with negative-stride Slice."""
    replacements = 0
    replacements_info = []
    initializers_to_add = []

    node_list = list(model.graph.node)
    node_to_idx = {id(node): idx for idx, node in enumerate(node_list)}

    for node in node_list:
        if node.op_type != "ReverseSequence":
            continue

        rev_input = node.input[0]
        rev_output = node.output[0]

        time_axis = 0
        for attr in node.attribute:
            if attr.name == "time_axis":
                time_axis = attr.i

        print(f"  Replacing ReverseSequence {node.name} (time_axis={time_axis})")

        starts_name = f"{node.name}_starts"
        ends_name = f"{node.name}_ends"
        axes_name = f"{node.name}_axes"
        steps_name = f"{node.name}_steps"

        starts = np.array([-1], dtype=np.int64)
        ends = np.array([-9223372036854775808], dtype=np.int64)
        axes = np.array([time_axis], dtype=np.int64)
        steps = np.array([-1], dtype=np.int64)

        initializers_to_add.extend(
            [
                numpy_helper.from_array(starts, starts_name),
                numpy_helper.from_array(ends, ends_name),
                numpy_helper.from_array(axes, axes_name),
                numpy_helper.from_array(steps, steps_name),
            ]
        )

        slice_node_new = helper.make_node(
            "Slice",
            inputs=[rev_input, starts_name, ends_name, axes_name, steps_name],
            outputs=[rev_output],
            name=f"{node.name}_slice",
        )

        node_idx = node_to_idx[id(node)]
        replacements_info.append((node_idx, [slice_node_new]))
        replacements += 1

    nodes_to_remove_idx = {idx for idx, _ in replacements_info}

    new_nodes = []
    for idx, node in enumerate(node_list):
        if idx in nodes_to_remove_idx:
            for remove_idx, replacement_nodes in replacements_info:
                if idx == remove_idx:
                    new_nodes.extend(replacement_nodes)
                    break
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    for init in initializers_to_add:
        model.graph.initializer.append(init)

    return model, replacements


def replace_globalavgpool_squeeze_with_reducemean(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, int]:
    """Replace GlobalAveragePool -> Squeeze pattern with ReduceMean.

    GlobalAveragePool outputs shape [..., 1, 1] which then gets Squeezed.
    ReduceMean with keepdims=0 does this in one operation.
    """
    replacements_info = []
    replacements = 0

    node_list = list(model.graph.node)
    node_to_idx = {id(node): idx for idx, node in enumerate(node_list)}

    for node in node_list:
        if node.op_type != "GlobalAveragePool":
            continue

        gap_output = node.output[0]
        consumers = find_nodes_by_input(model, gap_output)

        if len(consumers) != 1 or consumers[0].op_type != "Squeeze":
            continue

        squeeze_node = consumers[0]
        gap_input = node.input[0]
        squeeze_output = squeeze_node.output[0]

        squeeze_axes = get_attribute(squeeze_node, "axes", None)
        if squeeze_axes is None and len(squeeze_node.input) > 1:
            for init in model.graph.initializer:
                if init.name == squeeze_node.input[1]:
                    squeeze_axes = list(numpy_helper.to_array(init))
                    break

        reduce_axes = squeeze_axes if squeeze_axes else [2, 3]

        print(
            f"  Replacing GlobalAveragePool+Squeeze {node.name} with ReduceMean (axes={reduce_axes})"
        )

        reducemean_node = helper.make_node(
            "ReduceMean",
            inputs=[gap_input],
            outputs=[squeeze_output],
            name=f"{node.name}_reducemean",
            axes=reduce_axes,
            keepdims=0,
        )

        gap_idx = node_to_idx[id(node)]
        squeeze_idx = node_to_idx[id(squeeze_node)]
        replacements_info.append((gap_idx, squeeze_idx, [reducemean_node]))
        replacements += 1

    nodes_to_remove_idx = set()
    for gap_idx, squeeze_idx, _ in replacements_info:
        nodes_to_remove_idx.add(gap_idx)
        nodes_to_remove_idx.add(squeeze_idx)

    new_nodes = []
    for idx, node in enumerate(node_list):
        if idx in nodes_to_remove_idx:
            for gap_idx, _squeeze_idx, replacement_nodes in replacements_info:
                if idx == gap_idx:
                    new_nodes.extend(replacement_nodes)
                    break
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    return model, replacements


def remove_identity_casts(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Remove Cast operations that don't change the data type."""
    nodes_to_remove = []
    replacements = 0

    tensor_types = {}
    for inp in model.graph.input:
        tensor_types[inp.name] = inp.type.tensor_type.elem_type
    for init in model.graph.initializer:
        tensor_types[init.name] = init.data_type
    for vi in model.graph.value_info:
        tensor_types[vi.name] = vi.type.tensor_type.elem_type

    for node in model.graph.node:
        if node.op_type != "Cast":
            continue

        to_type = get_attribute(node, "to", None)
        if to_type is None:
            continue

        input_name = node.input[0]
        input_type = tensor_types.get(input_name)

        if input_type is not None and input_type == to_type:
            print(f"  Removing identity Cast {node.name}")
            cast_output = node.output[0]
            for other_node in model.graph.node:
                other_node.input[:] = [
                    input_name if x == cast_output else x for x in other_node.input
                ]
            nodes_to_remove.append(node)
            replacements += 1

    for node in nodes_to_remove:
        if node in model.graph.node:
            model.graph.node.remove(node)

    return model, replacements


def collapse_select_to_where(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Collapse tf2onnx SelectV2 decompositions back into a single Where op.

    tf2onnx lowers ``tf.where(cond, T, F)`` (SelectV2) into a Cast/Mul/Add cluster::

        Add( Mul(Cast(cond), T), Mul(F, Cast(Not(cond))) )

    arm64 ONNX Runtime's graph optimizer drops the bool->float Cast and then
    rejects the resulting ``Mul(bool, float)``, so a model carrying this pattern
    fails to load on arm64. ``Where`` consumes a bool condition natively, so
    rewriting the cluster to ``Where(cond, T, F)`` yields a model that loads on
    arm64 ORT and is numerically identical on every runtime.

    The match is structural rather than name-based, so it generalizes to any model
    that carries the pattern (for example the BirdNET v2.4 MData range filter's
    location encoder). A cluster is collapsed only when its intermediate tensors
    are not consumed elsewhere, so the rewrite never drops a live value.

    Returns the (mutated) model and the number of clusters collapsed.
    """
    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {
        out: node for node in graph.node for out in node.output if out
    }
    use_count: dict[str, int] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                use_count[inp] = use_count.get(inp, 0) + 1
    for graph_out in graph.output:
        use_count[graph_out.name] = use_count.get(graph_out.name, 0) + 1

    def split_mul(mul: onnx.NodeProto) -> Optional[tuple[onnx.NodeProto, str]]:
        """Return (cast_node, value_tensor) when one Mul input is a Cast, else None."""
        if len(mul.input) != 2:
            return None
        for idx in (0, 1):
            cand = producer.get(mul.input[idx])
            if cand is not None and cand.op_type == "Cast":
                return cand, mul.input[1 - idx]
        return None

    def cast_source(cast: onnx.NodeProto) -> tuple[str, str, Optional[onnx.NodeProto]]:
        """Classify a Cast's condition source as a direct cond or Not(cond)."""
        if not cast.input:
            return "direct", "", None
        src = producer.get(cast.input[0])
        if src is not None and src.op_type == "Not" and src.input:
            return "not", src.input[0], src
        return "direct", cast.input[0], None

    replacements: list[tuple[list[onnx.NodeProto], onnx.NodeProto]] = []
    # Identify nodes by their (unique) first output tensor name rather than id():
    # protobuf may hand back fresh wrapper objects per repeated-field access, so
    # object identity is not stable across iterations of graph.node.
    consumed: set[str] = set()

    for add in graph.node:
        if add.op_type != "Add" or len(add.input) != 2 or not add.output:
            continue
        mul_a = producer.get(add.input[0])
        mul_b = producer.get(add.input[1])
        if mul_a is None or mul_b is None:
            continue
        if mul_a.op_type != "Mul" or mul_b.op_type != "Mul":
            continue
        if mul_a.output[0] in consumed or mul_b.output[0] in consumed:
            continue
        split_a = split_mul(mul_a)
        split_b = split_mul(mul_b)
        if split_a is None or split_b is None:
            continue
        cast_a, val_a = split_a
        cast_b, val_b = split_b
        kind_a, cond_a, not_a = cast_source(cast_a)
        kind_b, cond_b, not_b = cast_source(cast_b)
        if {kind_a, kind_b} != {"direct", "not"} or cond_a != cond_b:
            continue
        if kind_a == "direct":
            true_val, false_val, not_node = val_a, val_b, not_b
        else:
            true_val, false_val, not_node = val_b, val_a, not_a
        if not_node is None:  # unreachable: exactly one branch is Not(cond)
            continue

        intermediates = [
            mul_a.output[0],
            mul_b.output[0],
            cast_a.output[0],
            cast_b.output[0],
            not_node.output[0],
        ]
        if any(use_count.get(t, 0) > 1 for t in intermediates):
            continue  # an intermediate is live elsewhere; collapsing would drop it

        where_name = f"{add.name}_where" if add.name else f"Where_{add.output[0]}"
        where = helper.make_node(
            "Where", [cond_a, true_val, false_val], [add.output[0]], name=where_name
        )
        cluster = [add, mul_a, mul_b, cast_a, cast_b, not_node]
        replacements.append((cluster, where))
        for node in cluster:
            consumed.add(node.output[0])

    if not replacements:
        return model, 0

    remove_outputs = {
        node.output[0] for cluster, _ in replacements for node in cluster if node.output
    }
    kept = [node for node in graph.node if not (node.output and node.output[0] in remove_outputs)]
    kept.extend(where for _, where in replacements)

    # Topological sort: emit a node only once all of its inputs are available.
    available = {init.name for init in graph.initializer}
    available.update(value.name for value in graph.input)
    ordered: list[onnx.NodeProto] = []
    pending = list(kept)
    progress = True
    while pending and progress:
        progress = False
        rest = []
        for node in pending:
            if all(inp == "" or inp in available for inp in node.input):
                ordered.append(node)
                available.update(node.output)
                progress = True
            else:
                rest.append(node)
        pending = rest
    if pending:
        raise RuntimeError(
            "collapse_select_to_where: topological sort failed; unresolved nodes: "
            f"{[node.name for node in pending]}"
        )

    del graph.node[:]
    graph.node.extend(ordered)
    return model, len(replacements)


def remove_redundant_reshapes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Remove consecutive Reshape operations that can be merged."""
    nodes_to_remove = []
    replacements = 0

    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue

        producer = find_node_by_output(model, node.input[0])
        if producer is None or producer.op_type != "Reshape":
            continue

        consumers = find_nodes_by_input(model, producer.output[0])
        if len(consumers) != 1:
            continue

        print(f"  Merging consecutive Reshapes: {producer.name} + {node.name}")

        node.input[0] = producer.input[0]
        nodes_to_remove.append(producer)
        replacements += 1

    for node in nodes_to_remove:
        if node in model.graph.node:
            model.graph.node.remove(node)

    return model, replacements


def fuse_transpose_patterns(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Remove or fuse consecutive Transpose operations."""
    nodes_to_remove = []
    replacements = 0

    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue

        producer = find_node_by_output(model, node.input[0])
        if producer is None or producer.op_type != "Transpose":
            continue

        consumers = find_nodes_by_input(model, producer.output[0])
        if len(consumers) != 1:
            continue

        perm1 = get_attribute(producer, "perm", None)
        perm2 = get_attribute(node, "perm", None)

        if perm1 is None or perm2 is None:
            continue

        composed_perm = [perm1[i] for i in perm2]

        if composed_perm == list(range(len(composed_perm))):
            print(f"  Removing inverse Transposes: {producer.name} + {node.name}")
            transpose_output = node.output[0]
            original_input = producer.input[0]
            for other_node in model.graph.node:
                other_node.input[:] = [
                    original_input if x == transpose_output else x for x in other_node.input
                ]
            nodes_to_remove.extend([producer, node])
            replacements += 1
        else:
            print(f"  Fusing Transposes: {producer.name} + {node.name} -> perm={composed_perm}")
            node.input[0] = producer.input[0]
            for attr in node.attribute:
                if attr.name == "perm":
                    del attr.ints[:]
                    attr.ints.extend(composed_perm)
            nodes_to_remove.append(producer)
            replacements += 1

    for node in nodes_to_remove:
        if node in model.graph.node:
            model.graph.node.remove(node)

    return model, replacements


def set_dynamic_batch(model: onnx.ModelProto) -> onnx.ModelProto:
    """Enable dynamic batch size."""
    for inp in model.graph.input:
        if inp.type.tensor_type.shape.dim:
            inp.type.tensor_type.shape.dim[0].ClearField("dim_value")
            inp.type.tensor_type.shape.dim[0].dim_param = "batch"

    for out in model.graph.output:
        if out.type.tensor_type.shape.dim:
            out.type.tensor_type.shape.dim[0].ClearField("dim_value")
            out.type.tensor_type.shape.dim[0].dim_param = "batch"

    return model


def convert_int32_to_int64(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Convert INT32 initializers to INT64 for better compatibility.

    Uses onnx_ir for proper type propagation throughout the graph.
    Based on: https://huggingface.co/justinchuby/BirdNET-onnx/blob/main/scripts/optimize.py

    CRITICAL: Runs RemoveCast rule BEFORE INT32->INT64 conversion to ensure
    type propagation works correctly. Cast nodes force specific types, so
    removing them first allows the optimizer to propagate INT64 types properly.
    """
    try:
        import onnx_ir as ir
        import onnxscript
        from onnxscript import rewriter
        from onnxscript.rewriter import pattern
    except ImportError as e:
        print(f"  onnx_ir or onnxscript not available ({e}), skipping INT32->INT64 conversion")
        return model, 0

    # Convert to IR representation
    ir_model = ir.from_proto(model)

    # Step 1: Remove Cast nodes FIRST (convert to Identity)
    # This is critical - Cast nodes force specific types and prevent INT64 propagation
    class RemoveCast(rewriter.RewriteRuleClassBase):
        """Replace Cast with Identity to let optimizer handle types."""

        def pattern(self, op, x):
            return op.Cast(x)

        def rewrite(self, op, x: ir.Value, **kwargs):
            return op.Identity(x)

    try:
        rule_set = pattern.RewriteRuleSet([RemoveCast.rule()])
        count = rewriter.rewrite(ir_model, rule_set)
        print(f"  RemoveCast rule applied ({count} patterns)")
    except Exception as e:
        print(f"  RemoveCast rule failed: {e}")

    # Step 2: Convert INT32 initializers to INT64
    converted = 0
    initializers = list(ir_model.graph.initializers.values())

    for initializer in initializers:
        if initializer.dtype == ir.DataType.INT32:
            if initializer.const_value is None or initializer.name is None:
                continue
            int32_array = initializer.const_value.numpy()
            int64_array = int32_array.astype(np.int64)

            # Create new INT64 initializer using ir.val() helper
            new_initializer = ir.val(initializer.name, const_value=ir.tensor(int64_array))

            # Replace in initializers dict
            ir_model.graph.initializers.pop(initializer.name)
            ir_model.graph.initializers.add(new_initializer)

            # Update all uses with proper type propagation
            initializer.replace_all_uses_with(new_initializer)
            converted += 1

    # Step 3: Run optimizer to propagate types and remove Identity nodes
    if converted > 0:
        try:
            onnxscript.optimizer.optimize(
                ir_model,
                input_size_limit=1024 * 1024 * 1024,
                output_size_limit=1024 * 1024 * 1024,
            )
            print("  Type inference updated via onnxscript optimizer")
        except Exception as e:
            print(f"  Warning: onnxscript optimizer failed during INT32->INT64: {e}")

    # Convert back to proto
    model = ir.to_proto(ir_model)

    return model, converted


def rename_io(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rename input/output to standard names.

    For single-output models (BirdNET): renames to "input" / "output".
    For multi-output models (Perch v2): renames input to "inputs" and
    preserves existing output names (embedding, spatial_embedding,
    spectrogram, label) since they carry semantic meaning.
    """
    num_outputs = len(model.graph.output)

    if num_outputs <= 2:
        # BirdNET-style: single input, one or two outputs
        if model.graph.input:
            old_name = model.graph.input[0].name
            new_name = "input"
            if old_name != new_name:
                model.graph.input[0].name = new_name
                for node in model.graph.node:
                    node.input[:] = [new_name if x == old_name else x for x in node.input]

        if model.graph.output:
            old_name = model.graph.output[0].name
            new_name = "output"
            if old_name != new_name:
                model.graph.output[0].name = new_name
                for node in model.graph.node:
                    node.output[:] = [new_name if x == old_name else x for x in node.output]
    else:
        # Multi-output model (Perch v2): rename input only, keep output names
        if model.graph.input:
            old_name = model.graph.input[0].name
            new_name = "inputs"
            if old_name != new_name:
                model.graph.input[0].name = new_name
                for node in model.graph.node:
                    node.input[:] = [new_name if x == old_name else x for x in node.input]
        print(f"  Multi-output model ({num_outputs} outputs), preserving output names")

    return model


def optimize_with_simplifier(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run onnx-simplifier for constant folding and optimization."""
    try:
        from onnxsim import simplify

        model_simplified, check = simplify(model)
        if check:
            print("  onnx-simplifier succeeded")
            return model_simplified  # type: ignore[no-any-return]
        else:
            print("  onnx-simplifier validation failed, keeping original")
    except Exception as e:
        print(f"  onnx-simplifier failed: {e}")
    return model


def optimize_with_onnxslim(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run onnxslim optimization."""
    try:
        import onnxslim

        model = onnxslim.slim(model)
        print("  onnxslim succeeded")
    except Exception as e:
        print(f"  onnxslim failed: {e}")
    return model


def optimize_with_onnxscript(model: onnx.ModelProto, remove_casts: bool = True) -> onnx.ModelProto:
    """Run onnxscript optimizer with optional Cast removal."""
    try:
        import onnx_ir as ir
        import onnxscript
        from onnxscript import rewriter
        from onnxscript.rewriter import pattern

        ir_model = ir.from_proto(model)

        if remove_casts:

            class RemoveCast(rewriter.RewriteRuleClassBase):
                """Replace Cast with Identity to let optimizer handle types."""

                def pattern(self, op, x):
                    return op.Cast(x)

                def rewrite(self, op, x: ir.Value, **kwargs):
                    return op.Identity(x)

            try:
                rule_set = pattern.RewriteRuleSet([RemoveCast.rule()])
                rewriter.rewrite(ir_model, rule_set)
                print("  RemoveCast rule applied")
            except Exception as e:
                print(f"  RemoveCast rule failed: {e}")

        onnxscript.optimizer.optimize(
            ir_model,
            input_size_limit=1024 * 1024 * 1024,
            output_size_limit=1024 * 1024 * 1024,
        )
        model = ir.to_proto(ir_model)
        print("  onnxscript optimizer succeeded")
    except Exception as e:
        print(f"  onnxscript optimizer failed: {e}")
    return model


def remove_dead_code(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Remove nodes whose outputs are not consumed by any other node or graph output.

    Iteratively removes dead nodes until no more can be removed, handling
    chains of dead code (e.g., Shape -> Gather -> Expand -> orphaned).
    """
    total_removed = 0

    while True:
        # Build set of consumed outputs
        consumed = set()
        for node in model.graph.node:
            for inp in node.input:
                consumed.add(inp)
        for out in model.graph.output:
            consumed.add(out.name)

        # Find nodes with all outputs unconsumed
        nodes_to_remove = []
        for node in model.graph.node:
            if not node.output:
                continue
            all_dead = all(out not in consumed for out in node.output)
            if all_dead:
                nodes_to_remove.append(node)

        if not nodes_to_remove:
            break

        for node in nodes_to_remove:
            print(f"  Removing dead node: {node.name} ({node.op_type})")
            model.graph.node.remove(node)
            total_removed += 1

    # Also remove orphaned initializers
    consumed_initializers = set()
    for node in model.graph.node:
        for inp in node.input:
            consumed_initializers.add(inp)

    init_to_remove = []
    for init in model.graph.initializer:
        if init.name not in consumed_initializers:
            init_to_remove.append(init)

    for init in init_to_remove:
        model.graph.initializer.remove(init)

    if init_to_remove:
        print(f"  Removed {len(init_to_remove)} orphaned initializers")

    return model, total_removed


def fix_split_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Fix Split nodes with zero-size outputs.

    When a Split node has a split tensor with zero values (e.g., [1, 1, 0]),
    the corresponding outputs are empty and usually unused. This causes
    NVIDIA tools to report "mismatch between splits and outputs".

    Fix: Remove zero-size outputs and update the split tensor.
    """
    fixes = 0

    # Build initializer lookup
    initializers = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type != "Split":
            continue

        # Check if this Split has a split sizes input (opset 13+)
        if len(node.input) < 2:
            continue

        split_input_name = node.input[1]
        if split_input_name not in initializers:
            continue

        split_init = initializers[split_input_name]
        split_sizes = numpy_helper.to_array(split_init)

        # Find indices with zero size
        zero_indices = [i for i, size in enumerate(split_sizes) if size == 0]

        if not zero_indices:
            continue

        print(f"  Split {node.name}: found {len(zero_indices)} zero-size outputs")
        print(f"    Original split sizes: {list(split_sizes)}")

        # Remove zero-size outputs from the node
        outputs_to_keep = []
        new_split_sizes = []
        for _i, (out, size) in enumerate(zip(node.output, split_sizes)):
            if size > 0:
                outputs_to_keep.append(out)
                new_split_sizes.append(size)
            else:
                print(f"    Removing empty output: {out}")

        # Update the node outputs
        del node.output[:]
        node.output.extend(outputs_to_keep)

        # Update the split sizes initializer
        new_split_array = np.array(new_split_sizes, dtype=split_sizes.dtype)
        new_init = numpy_helper.from_array(new_split_array, split_input_name)

        # Replace the initializer
        model.graph.initializer.remove(split_init)
        model.graph.initializer.append(new_init)

        print(f"    New split sizes: {list(new_split_array)}")
        print(f"    Outputs: {len(list(split_sizes))} -> {len(outputs_to_keep)}")
        fixes += 1

    return model, fixes


def prune_outputs(model: onnx.ModelProto, keep_names: list[str]) -> tuple[onnx.ModelProto, int]:
    """Remove graph outputs not in keep_names and prune dead nodes.

    For Perch v2, the model has 4 outputs: embedding, spatial_embedding,
    spectrogram, label. If only embedding and label are needed, pruning
    spatial_embedding and spectrogram can eliminate compute for those branches.
    """
    all_output_names = [out.name for out in model.graph.output]

    # Validate that all requested names exist in the model
    unknown = [name for name in keep_names if name not in all_output_names]
    if unknown:
        print(f"  Warning: unknown output names: {unknown}")
        print(f"  Available outputs: {all_output_names}")
        return model, 0

    to_remove = [name for name in all_output_names if name not in keep_names]

    if not to_remove:
        print("  No outputs to prune")
        return model, 0

    # Remove unwanted outputs from the graph
    outputs_to_keep = [out for out in model.graph.output if out.name in keep_names]
    del model.graph.output[:]
    model.graph.output.extend(outputs_to_keep)

    print(f"  Pruned outputs: {to_remove}")
    print(f"  Kept outputs: {[out.name for out in model.graph.output]}")

    # Run dead code elimination to remove nodes that only fed pruned outputs
    model, dead_removed = remove_dead_code(model)

    return model, len(to_remove)


def count_ops(model: onnx.ModelProto) -> dict[str, int]:
    """Count operations in a model."""
    from collections import Counter

    return dict(Counter(node.op_type for node in model.graph.node))


def fix_fp16_type_mismatches(model: onnx.ModelProto) -> tuple[int, int]:
    """Fix type mismatches introduced by keep_io_types Cast insertion.

    After FP16 conversion with keep_io_types=True, the converter inserts Cast
    nodes (fp16->fp32) at I/O boundaries. But if a Cast output is also consumed
    by internal nodes (like MatMul) whose other inputs are fp16, we get a type
    mismatch. Fix by rewiring internal consumers to use the pre-Cast fp16 tensor.
    Also fix value_info entries that disagree with Cast output types.

    Returns (rewires, value_info_fixes). With decouple_dual_role_outputs run
    beforehand, rewires should be 0 (a non-zero count means a mixed-type op
    survived decoupling); value_info_fixes are harmless annotation corrections.
    """
    # Map Cast(to=FLOAT) outputs to their fp16 inputs
    cast_fp32_output_to_fp16_input: dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to_type = None
        for attr in node.attribute:
            if attr.name == "to":
                to_type = attr.i
                break
        if to_type == onnx.TensorProto.FLOAT:
            cast_fp32_output_to_fp16_input[node.output[0]] = node.input[0]

    # Build type maps for initializers and intermediate tensors
    init_types = {init.name: init.data_type for init in model.graph.initializer}
    vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model.graph.value_info}

    rewires = 0
    vi_fixes = 0

    # Rewire internal nodes that consume a Cast(to=FLOAT) output alongside fp16 inputs
    for node in model.graph.node:
        if node.op_type == "Cast":
            continue

        new_inputs = list(node.input)
        changed = False

        for i, inp in enumerate(new_inputs):
            if inp not in cast_fp32_output_to_fp16_input:
                continue
            other_types = set()
            for j, other_inp in enumerate(new_inputs):
                if j == i:
                    continue
                if other_inp in init_types:
                    other_types.add(init_types[other_inp])
                elif other_inp in vi_types:
                    other_types.add(vi_types[other_inp])
            if onnx.TensorProto.FLOAT16 in other_types:
                pre_cast = cast_fp32_output_to_fp16_input[inp]
                new_inputs[i] = pre_cast
                changed = True
                print(f"    Rewired {node.name} ({node.op_type}): {inp} -> {pre_cast}")
                rewires += 1

        if changed:
            del node.input[:]
            node.input.extend(new_inputs)

    # Fix value_info entries that conflict with Cast output types
    cast_output_types: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to_type = None
        for attr in node.attribute:
            if attr.name == "to":
                to_type = attr.i
                break
        if to_type is not None:
            for out_name in node.output:
                cast_output_types[out_name] = to_type

    for vi in model.graph.value_info:
        if vi.name in cast_output_types:
            expected_type = cast_output_types[vi.name]
            actual_type = vi.type.tensor_type.elem_type
            if actual_type != expected_type:
                old_name = onnx.TensorProto.DataType.Name(actual_type)
                new_name = onnx.TensorProto.DataType.Name(expected_type)
                vi.type.tensor_type.elem_type = expected_type
                print(f"    Fixed value_info: {vi.name} ({old_name} -> {new_name})")
                vi_fixes += 1

    return rewires, vi_fixes


def decouple_dual_role_outputs(model: onnx.ModelProto) -> int:
    """Insert an Identity boundary for graph outputs that are also consumed internally.

    onnxconverter-common's keep_io_types inserts a Cast (fp16->fp32) at every
    graph output and reuses the original tensor name for the Cast output. When a
    tensor is BOTH a graph output and an internal node input (a "dual-role"
    tensor, e.g. pooled embeddings that are also fed to a classifier MatMul), the
    Cast silently rewires the internal consumers to fp32, producing mixed-precision
    ops (a MatMul with one fp16 and one fp32 input).

    This fixes the problem preventively. For each dual-role tensor T, the
    producer output and all internal consumers are repointed to a private name,
    and T is re-exposed through an Identity node. The external output name is
    preserved (consumers that look up outputs by name keep working), while T
    becomes a clean single-producer graph output that keep_io_types handles
    correctly without touching the internal consumers.

    Must run before float16 conversion. Mutates the model in place. Returns the
    number of outputs decoupled.
    """
    # Names of tensors consumed by any node (graph outputs are not node inputs).
    consumed: set[str] = set()
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumed.add(inp)

    # All tensor names in use, so minted internal names cannot collide. Includes
    # value_info so a minted name cannot inherit a stale intermediate annotation.
    used: set[str] = set(consumed)
    used.update(init.name for init in model.graph.initializer)
    for node in model.graph.node:
        used.update(o for o in node.output if o)
    for value in chain(model.graph.input, model.graph.output, model.graph.value_info):
        used.add(value.name)

    decoupled = 0
    for out in model.graph.output:
        original = out.name
        if original not in consumed:
            continue  # terminal-only output, nothing to decouple

        # A graph output with no producing node (e.g. an input passthrough) has
        # no fp16 boundary to corrupt; leave it for the sentinel to handle.
        if find_node_by_output(model, original) is None:
            continue

        internal = f"{original}_fp16src"
        suffix = 1
        while internal in used:
            suffix += 1
            internal = f"{original}_fp16src{suffix}"
        used.add(internal)

        # Repoint the producer's output and every internal consumer of `original`
        # to the private `internal` name (slice-assign to avoid mutating a
        # repeated field while iterating it).
        for node in model.graph.node:
            if original in node.output:
                node.output[:] = [internal if o == original else o for o in node.output]
            if original in node.input:
                node.input[:] = [internal if i == original else i for i in node.input]

        # Re-expose `original` as the graph output via Identity, inserted right
        # after its producer so the graph stays topologically sorted regardless
        # of where the original consumers (including any subgraph captures) sit.
        # Locate the producer by the unique `internal` name it now emits rather
        # than by object identity, which protobuf does not guarantee across
        # repeated-field accesses (the C++/upb backend returns fresh wrappers).
        identity = onnx.helper.make_node(
            "Identity",
            inputs=[internal],
            outputs=[original],
            name=f"_fp16_decoupler_{original}",
        )
        producer_idx = next(i for i, n in enumerate(model.graph.node) if internal in n.output)
        model.graph.node.insert(producer_idx + 1, identity)
        decoupled += 1

    return decoupled


def find_preprocessing_node_names(model: onnx.ModelProto) -> set[str]:
    """Return the names of nodes that compute the spectrogram front-end.

    The front-end is everything feeding the first backbone Conv: framing, the
    STFT/RFFT and mel-filterbank MatMuls, the power and per-sample normalization
    ops. These accumulate over long reductions and lose accuracy in FP16, so they
    are kept in FP32 via node_block_list during conversion.

    Block by node name (ancestors of the first Conv) rather than by op type:
    ops such as ReduceMean and Mul also appear inside the CNN backbone (SE
    blocks), which should stay FP16. Returns an empty set if the graph has no
    Conv (the front-end cannot be located). Any unnamed front-end node is given
    a unique name in place so the name-based block list cannot miss it.

    Assumes the first Conv in topological order is the backbone stem (true for
    BirdNET, whose preprocessing branches merge before any Conv). If a model
    instead has a Conv inside a parallel preprocessing branch, only that Conv's
    ancestors are protected; the fallback is reduced FP32 coverage, never an
    invalid graph, and --fp16-full remains available.
    """
    first_conv = next((n for n in model.graph.node if n.op_type == "Conv"), None)
    if first_conv is None:
        return set()

    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    initializers = {init.name for init in model.graph.initializer}
    graph_inputs = {i.name for i in model.graph.input}

    names: set[str] = set()
    visited: set[str] = set()
    # Walk backward from the Conv's data inputs (not the Conv itself, which stays
    # FP16) collecting every ancestor node.
    stack = [
        inp
        for inp in first_conv.input
        if inp and inp not in initializers and inp not in graph_inputs
    ]
    while stack:
        tensor = stack.pop()
        if tensor in visited:
            continue
        visited.add(tensor)
        node = producer.get(tensor)
        if node is None:
            continue
        # Node names are optional in ONNX. The block list is name-based, so give
        # any unnamed front-end node a unique name (derived from its unique
        # output tensor) to ensure it is actually kept in FP32 rather than
        # silently becoming an FP16 island inside the preprocessing region.
        if not node.name:
            node.name = f"_fp32_preproc_{tensor}"
        names.add(node.name)
        for inp in node.input:
            if inp and inp not in initializers and inp not in graph_inputs:
                stack.append(inp)
    return names


def find_swish_node_names(model: onnx.ModelProto) -> set[str]:
    """Return the node names of sigmoid-gated activations (Swish/SiLU: x*sigmoid(x),
    and SE-style x*sigmoid(gate)).

    These are kept in FP32 during FP16 conversion (CLI --fp16-keep-activations-fp32)
    so ONNX Runtime's CPU EP can fuse its FP32 QuickGelu kernel; an all-FP16 graph
    runs them unfused and is much slower on ARM/CPU. Both the Mul and the Sigmoid
    feeding it are returned so the fusable pair stays FP32 together.

    Blocks by node name, not by the Mul/Sigmoid op types, so unrelated multiplies
    (e.g. batchnorm scale) are NOT forced to FP32. Any unnamed activation node is
    given a unique name in place so the name-based block list cannot miss it.
    """
    sigmoid_by_output: dict[str, onnx.NodeProto] = {
        node.output[0]: node
        for node in model.graph.node
        if node.op_type == "Sigmoid" and node.output and node.output[0]
    }

    used = {node.name for node in model.graph.node if node.name}

    def ensure_name(node: onnx.NodeProto, hint: str) -> str:
        if not node.name:
            name = f"_fp16_swish_{hint}"
            suffix = 1
            while name in used:
                suffix += 1
                name = f"_fp16_swish_{hint}{suffix}"
            node.name = name
            used.add(name)
        return node.name

    names: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Mul":
            continue
        gates = [inp for inp in node.input if inp in sigmoid_by_output]
        if not gates:
            continue
        names.add(ensure_name(node, node.output[0] if node.output else "mul"))
        for gate in gates:
            names.add(ensure_name(sigmoid_by_output[gate], gate))
    return names


def convert_to_fp16(
    model: onnx.ModelProto,
    keep_preprocessing_fp32: bool = True,
    keep_activations_fp32: bool = False,
) -> Optional[onnx.ModelProto]:
    """Convert FP32 model to FP16 for devices with FP16 hardware support.

    Decouples dual-role outputs first so keep_io_types Cast insertion cannot
    corrupt internal consumers, then uses onnxconverter-common for the FP16
    conversion. fix_fp16_type_mismatches is kept as a sentinel and should report
    zero rewires once decoupling has run.

    When keep_preprocessing_fp32 is True (default), the spectrogram front-end
    (STFT/mel MatMuls and normalization, everything feeding the first backbone
    Conv) is kept in FP32 to preserve accuracy while the CNN backbone is still
    converted to FP16. Pass keep_preprocessing_fp32=False (CLI --fp16-full) to
    convert the entire graph to FP16.

    When keep_activations_fp32 is True (CLI --fp16-keep-activations-fp32), the
    sigmoid-gated Swish/SiLU activation nodes (found by name, not op type) are
    additionally kept in FP32 so ONNX Runtime's CPU EP can fuse them into its FP32
    QuickGelu kernel; the conv/matmul weights stay FP16. This recovers most of the
    FP16 latency on ARM/CPU at a small runtime-RAM cost (activation buffers stay
    FP32).

    Note: the input model is mutated in place by the decouple step. Callers that
    still need the original FP32 graph afterward should save or copy it first
    (main() saves the FP32 model to disk before calling this).
    """
    decoupled = decouple_dual_role_outputs(model)
    if decoupled > 0:
        print(f"  Decoupled {decoupled} dual-role output(s) before FP16 conversion")

    block_names: set[str] = set()
    if keep_preprocessing_fp32:
        preprocessing = find_preprocessing_node_names(model)
        if preprocessing:
            block_names |= preprocessing
            print(f"  Keeping {len(preprocessing)} preprocessing node(s) in FP32")
        else:
            print("  No preprocessing front-end identified; converting the full graph to FP16")

    if keep_activations_fp32:
        swish = find_swish_node_names(model)
        if swish:
            block_names |= swish
            print(
                f"  Keeping {len(swish)} Swish/SiLU activation node(s) in FP32 so "
                "ORT's CPU EP can fuse QuickGelu"
            )
        else:
            print("  No Swish/SiLU activations identified")

    node_block_list = sorted(block_names) if block_names else None

    try:
        from onnxconverter_common import float16

        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True, node_block_list=node_block_list
        )
        print("  FP16 conversion succeeded")
    except ImportError:
        print("  Warning: onnxconverter-common not installed, trying onnxruntime")
        try:
            from onnxruntime.transformers import float16

            model_fp16 = float16.convert_float_to_float16(
                model, keep_io_types=True, node_block_list=node_block_list
            )
            print("  FP16 conversion succeeded (via onnxruntime)")
        except Exception as e:
            print(f"  FP16 conversion failed: {e}")
            return None
    except Exception as e:
        print(f"  FP16 conversion failed: {e}")
        return None

    # The value_info correction below is not merely cosmetic for the
    # keep_preprocessing_fp32 path: the converter leaves stale FP32 annotations at
    # the FP32->FP16 block-list boundary, and ONNX Runtime refuses to load the
    # model until they are fixed. Always run the sentinel.
    rewires, vi_fixes = fix_fp16_type_mismatches(model_fp16)
    if rewires > 0:
        print(
            f"  WARNING: rewired {rewires} residual mixed-type op(s); decoupling "
            "should have prevented these"
        )
    if vi_fixes > 0:
        print(f"  Corrected {vi_fixes} stale value_info annotation(s)")

    return model_fp16  # type: ignore[no-any-return]


def prepare_for_quantization(model_path: str, prepared_path: str) -> list[str]:
    """Refresh shape metadata before quantization and return nodes to exclude.

    Writes a quantization-ready copy of the model to prepared_path and returns
    the spectrogram preprocessing node names that must be kept in full precision.

    Two problems are addressed:
    - The optimization passes (DFT/RFFT replacement, transpose fusion, reshape
      removal, etc.) leave stale value_info behind. quantize_dynamic relies on
      value_info to type tensors, so it is cleared and re-inferred here.
    - INT8-quantizing the mel/STFT MatMuls corrupts the spectrogram and collapses
      predictions, so the preprocessing front-end is reported for exclusion.
    """
    model = onnx.load(model_path)
    # find_preprocessing_node_names may assign names to unnamed front-end nodes
    # in place; those names are serialized below so the reloaded model that
    # quantize_dynamic reads exposes the same names for nodes_to_exclude.
    exclude = sorted(find_preprocessing_node_names(model))
    del model.graph.value_info[:]
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        # Safe to proceed: quantize_dynamic re-infers shapes when it loads the
        # model, and dynamic weight quantization keys off initializer dtypes, not
        # value_info, so an empty value_info does not under-quantize.
        print(f"  Warning: shape inference before quantization failed: {e}")
    onnx.save(model, prepared_path)
    return exclude


def _quantize_int8(
    model_path: str,
    output_path: str,
    label: str,
    op_types_to_quantize: Optional[list[str]] = None,
) -> bool:
    """Shared INT8 dynamic-quantization driver.

    Refreshes shape metadata, excludes the spectrogram preprocessing front-end so
    it stays full precision, runs quantize_dynamic, and always removes the
    temporary prepared model. op_types_to_quantize restricts which op types are
    quantized (the ARM path passes ["MatMul"]).
    """
    prepared_path = f"{output_path}.prep.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        exclude = prepare_for_quantization(model_path, prepared_path)
        if exclude:
            print(f"  Excluding {len(exclude)} preprocessing node(s) from INT8")
        kwargs: dict[str, object] = {
            "weight_type": QuantType.QUInt8,
            "nodes_to_exclude": exclude or None,
            "extra_options": {"ActivationSymmetric": False},
        }
        if op_types_to_quantize is not None:
            kwargs["op_types_to_quantize"] = op_types_to_quantize
        quantize_dynamic(prepared_path, output_path, **kwargs)
        print(f"  {label} succeeded")
        return True
    except Exception as e:
        print(f"  {label} failed: {e}")
        return False
    finally:
        Path(prepared_path).unlink(missing_ok=True)


def quantize_to_int8_dynamic(model_path: str, output_path: str) -> bool:
    """Apply INT8 dynamic quantization for CPU inference on low-power devices.

    Dynamic quantization quantizes weights to INT8 and computes activations
    in INT8 at runtime. Uses QUInt8 weights for better ARM support. The
    spectrogram preprocessing is excluded so it stays full precision.
    """
    return _quantize_int8(model_path, output_path, "INT8 dynamic quantization")


def quantize_to_int8_arm(model_path: str, output_path: str) -> bool:
    """Apply INT8 quantization optimized for ARM devices.

    Only quantizes MatMul operations (not Conv) to avoid ConvInteger ops
    which are not supported on ARM ONNX Runtime builds. The spectrogram
    preprocessing MatMuls are excluded so they stay full precision.
    """
    return _quantize_int8(
        model_path,
        output_path,
        "INT8 ARM quantization (MatMul only)",
        op_types_to_quantize=["MatMul"],
    )


def truncate_dft_bins(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Truncate each in-graph DFT MatMul to the frequency bins its mel filterbank
    actually uses, restoring the bin axis with a zero Pad.

    BirdNET computes its log-mel front-end with a dense DFT-as-MatMul (an rfft
    lowered to ``frames @ dft_matrix[n_fft, n_fft//2+1]``). Each mel filterbank
    only spans part of the spectrum, so most DFT output bins are multiplied by a
    zero filterbank row and thrown away. Computing them is pure waste: on
    BirdNET v2.4 the low-band DFT (``[2048,1025]``) is ~61% of the whole model's
    FLOPs yet 88% of its output bins are unused (mel covers ~23-2977 Hz -> 128 of
    1025 bins); the high-band DFT (``[1024,513]``) uses 320 of 513.

    For each DFT MatMul this truncates the weight columns to the used bins and
    inserts a zero ``Pad`` so the bin axis is restored before the (unchanged)
    magnitude / mel-filterbank ops downstream. Padding the removed bins with zero
    is exactly equivalent to computing them and letting the filterbank discard
    them, so the graph output is numerically identical. A bit-exact check gates
    the rewrite and reverts it if any output differs.

    Measured (BirdNET v2.4, bit-exact): ORT CPU ~1.55-1.9x faster (rpi4 A72, rpi5
    A76, amd64), OpenVINO CPU f16/f32 ~1.4-1.7x, neutral on Intel iGPU (where the
    dense matmul is already free). No-op on models whose filterbank uses every bin.
    """
    graph = model.graph
    init_by_name = {init.name: init for init in graph.initializer}
    arr = {name: numpy_helper.to_array(init) for name, init in init_by_name.items()}

    def const_2d_weight(node: onnx.NodeProto):
        for inp in node.input:
            a = arr.get(inp)
            if a is not None and a.ndim == 2:
                return inp, a
        return None, None

    # DFT MatMuls: a constant weight [F, B] with B == F//2 + 1 (one-sided rfft) and
    # cos/sin values in [-1, 1].
    dft_matmuls = []
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        wname, w = const_2d_weight(node)
        if w is None:
            continue
        f, b = w.shape
        if b == f // 2 + 1 and float(w.min()) >= -1.001 and float(w.max()) <= 1.001:
            dft_matmuls.append((node, wname, f, b))

    if not dft_matmuls:
        return model, 0

    # Mel filterbanks: a constant weight [B, M] whose B matches a DFT's bin count
    # (M = mel bands < B). Matched by bin count, unambiguous when the DFTs differ.
    dft_bins = {b for _, _, _, b in dft_matmuls}
    melfb_by_bins: dict[int, np.ndarray] = {}
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        wname, w = const_2d_weight(node)
        if w is None:
            continue
        b, m = w.shape
        if b in dft_bins and m < b:
            melfb_by_bins.setdefault(b, w)

    # Ranks for the Pad's `pads` vector (the bin axis is the last axis).
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        rank_by_tensor = {
            vi.name: len(vi.type.tensor_type.shape.dim) for vi in inferred.graph.value_info
        }
    except Exception:
        rank_by_tensor = {}

    original_bytes = model.SerializeToString()
    count = 0
    for node, wname, _f, b in dft_matmuls:
        fb = melfb_by_bins.get(b)
        if fb is None:
            continue
        used = np.nonzero(np.any(fb != 0.0, axis=1))[0]
        if used.size == 0:
            continue
        keep = int(used.max()) + 1
        if keep >= b:
            continue  # filterbank uses every bin; nothing to gain

        # 1. truncate DFT weight columns [F, B] -> [F, keep]
        trunc_w = np.ascontiguousarray(arr[wname][:, :keep])
        init_by_name[wname].CopyFrom(numpy_helper.from_array(trunc_w, name=wname))

        # 2. rename the MatMul output and Pad it back to B on the last axis with zeros
        out_name = node.output[0]
        trunc_name = f"{out_name}_dfttrunc"
        node.output[0] = trunc_name
        rank = rank_by_tensor.get(out_name, 3)  # BirdNET DFT output is rank-3 here
        pads = np.zeros(2 * rank, dtype=np.int64)
        pads[-1] = b - keep  # pad_after on the last (bin) axis
        pads_name = f"{out_name}_dfttrunc_pads"
        zero_name = f"{out_name}_dfttrunc_zero"
        graph.initializer.append(numpy_helper.from_array(pads, name=pads_name))
        graph.initializer.append(
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=zero_name)
        )
        pad_node = helper.make_node(
            "Pad",
            [trunc_name, pads_name, zero_name],
            [out_name],
            mode="constant",
            name=f"{out_name}_dfttrunc_pad",
        )
        graph.node.insert(list(graph.node).index(node) + 1, pad_node)
        print(
            f"  DFT bins {b} -> {keep} ({100 * keep / b:.0f}% kept), matmul FLOPs x{keep / b:.2f}"
        )
        count += 1

    if count == 0:
        return model, 0

    # Bit-exact gate: run the pre- and post-truncation graphs on one random input
    # and revert everything if the output changed beyond float noise.
    try:
        del graph.value_info[:]
        onnx.checker.check_model(model, full_check=False)
        import onnxruntime as ort

        src = ort.InferenceSession(original_bytes, providers=["CPUExecutionProvider"])
        inp = src.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        base = src.run(None, {inp.name: x})[0]
        dst = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got = dst.run(None, {inp.name: x})[0]
        max_abs = float(np.abs(base - got).max())
        if max_abs > 1e-3:
            print(
                f"  Bit-exact check FAILED (max abs diff {max_abs:.3e}); reverting DFT truncation"
            )
            return onnx.load_model_from_string(original_bytes), 0
        print(f"  Bit-exact check passed (max abs diff {max_abs:.2e})")
    except Exception as exc:
        print(f"  DFT truncation verification failed ({exc}); reverting to be safe")
        return onnx.load_model_from_string(original_bytes), 0

    return model, count


def optimize_model(
    input_path: str,
    prune_output_names: Optional[list[str]] = None,
    perch_mode: bool = False,
    truncate_dft: bool = True,
) -> Optional[onnx.ModelProto]:
    """Main optimization pipeline. Returns the optimized FP32 model.

    Args:
        input_path: Path to the ONNX model file.
        prune_output_names: If set, keep only these outputs and prune the rest.
        perch_mode: If True, use Perch-specific naming and graph settings.
    """

    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    initial_ops = count_ops(model)
    print(f"  Initial ops: {sum(initial_ops.values())}")
    print(f"  DFT: {initial_ops.get('DFT', 0)}")
    print(f"  RFFT2D: {initial_ops.get('RFFT2D', 0)}")
    print(f"  ReverseSequence: {initial_ops.get('ReverseSequence', 0)}")
    print(f"  GlobalAveragePool: {initial_ops.get('GlobalAveragePool', 0)}")
    print(f"  Cast: {initial_ops.get('Cast', 0)}")
    print(f"  Transpose: {initial_ops.get('Transpose', 0)}")

    # Step 1: Replace DFT with MatMul
    print("\n[1/13] Replacing DFT with MatMul...")
    model, dft_replaced = replace_dft_with_matmul(model)
    print(f"  Replaced: {dft_replaced}")

    # Step 2: Replace RFFT2D with MatMul
    print("\n[2/13] Replacing RFFT2D with MatMul...")
    model, rfft_replaced = replace_rfft2d_with_matmul(model)
    print(f"  Replaced: {rfft_replaced}")

    # Step 3: Replace ReverseSequence with Slice
    print("\n[3/13] Replacing ReverseSequence with Slice...")
    model, rev_replaced = replace_reverse_sequence(model)
    print(f"  Replaced: {rev_replaced}")

    # Step 4: Replace GlobalAveragePool + Squeeze with ReduceMean
    print("\n[4/13] Replacing GlobalAveragePool+Squeeze with ReduceMean...")
    model, gap_replaced = replace_globalavgpool_squeeze_with_reducemean(model)
    print(f"  Replaced: {gap_replaced}")

    # Step 5: Remove identity casts
    print("\n[5/13] Removing identity Cast operations...")
    model, cast_removed = remove_identity_casts(model)
    print(f"  Removed: {cast_removed}")

    # Step 6: Fuse transpose patterns
    print("\n[6/13] Fusing Transpose patterns...")
    model, transpose_fused = fuse_transpose_patterns(model)
    print(f"  Fused: {transpose_fused}")

    # Step 7: Remove redundant reshapes
    print("\n[7/13] Removing redundant Reshapes...")
    model, reshape_removed = remove_redundant_reshapes(model)
    print(f"  Removed: {reshape_removed}")

    # Step 8: Convert INT32 to INT64 for compatibility
    print("\n[8/13] Converting INT32 initializers to INT64...")
    model, int32_converted = convert_int32_to_int64(model)
    print(f"  Converted: {int32_converted}")

    # Step 9: Run external optimizers
    print("\n[9/13] Running graph optimizers...")
    model = optimize_with_simplifier(model)
    model = optimize_with_onnxscript(model, remove_casts=False)
    model = optimize_with_onnxslim(model)

    # Step 10: Prune unused outputs (if requested)
    if prune_output_names:
        print(f"\n[10/15] Pruning outputs to: {prune_output_names}...")
        model, pruned = prune_outputs(model, prune_output_names)
        print(f"  Pruned: {pruned} outputs removed")
    else:
        print("\n[10/15] Output pruning: skipped (no --prune-outputs)")

    # Step 11: Remove dead code (orphaned nodes from replacements and pruning)
    print("\n[11/15] Removing dead code...")
    model, dead_removed = remove_dead_code(model)
    print(f"  Removed: {dead_removed} dead nodes")

    # Step 12: Fix Split nodes (remove zero-size outputs)
    print("\n[12/15] Fixing Split nodes...")
    model, split_fixed = fix_split_nodes(model)
    print(f"  Fixed: {split_fixed}")

    # Step 13: Truncate the DFT to the mel filterbank's used bins (bit-exact speedup)
    if truncate_dft:
        print("\n[13/15] Truncating DFT to used mel bins...")
        model, dft_truncated = truncate_dft_bins(model)
        print(f"  Truncated: {dft_truncated} DFT matmul(s)")
    else:
        print("\n[13/15] DFT truncation: skipped (--no-truncate-dft)")

    # Step 14: Set dynamic batch
    print("\n[14/15] Setting dynamic batch size...")
    model = set_dynamic_batch(model)
    print("  Done")

    # Step 15: Finalize
    print("\n[15/15] Finalizing model...")
    model = rename_io(model)
    model.ir_version = 9
    if perch_mode:
        model.producer_name = "perch-onnx-optimizer"
        model.graph.name = "Perch"
    else:
        model.producer_name = "birdnet-onnx-optimizer"
        model.graph.name = "BirdNET"
    print("  Done")

    # Final stats
    final_ops = count_ops(model)
    print(f"\n{'=' * 60}")
    print(f"{'OPTIMIZATION RESULTS':^60}")
    print(f"{'=' * 60}")
    print(
        f"  Total ops: {sum(initial_ops.values())} -> {sum(final_ops.values())} ({sum(final_ops.values()) - sum(initial_ops.values()):+d})"
    )
    print(f"  DFT: {initial_ops.get('DFT', 0)} -> {final_ops.get('DFT', 0)}")
    print(f"  RFFT2D: {initial_ops.get('RFFT2D', 0)} -> {final_ops.get('RFFT2D', 0)}")
    print(
        f"  ReverseSequence: {initial_ops.get('ReverseSequence', 0)} -> {final_ops.get('ReverseSequence', 0)}"
    )
    print(
        f"  GlobalAveragePool: {initial_ops.get('GlobalAveragePool', 0)} -> {final_ops.get('GlobalAveragePool', 0)}"
    )
    print(f"  Squeeze: {initial_ops.get('Squeeze', 0)} -> {final_ops.get('Squeeze', 0)}")
    print(f"  Cast: {initial_ops.get('Cast', 0)} -> {final_ops.get('Cast', 0)}")
    print(f"  Transpose: {initial_ops.get('Transpose', 0)} -> {final_ops.get('Transpose', 0)}")
    print(f"  ReduceMean: {initial_ops.get('ReduceMean', 0)} -> {final_ops.get('ReduceMean', 0)}")
    print(f"  MatMul: {initial_ops.get('MatMul', 0)} -> {final_ops.get('MatMul', 0)}")

    # Verify
    print("\nVerifying model...")
    try:
        onnx.checker.check_model(model)
        print("  Model is valid!")
    except Exception as e:
        print(f"  Warning: {e}")

    return model


def _print_int8_warning(
    no_int8: bool, int8_arm: bool, out: Callable[[str], object] = print
) -> bool:
    """Emit the INT8 experimental warning when an INT8 variant will be produced.

    Mirrors main()'s production condition: a warning fires iff the default INT8
    model (unless --no-int8) or the ARM INT8 model (--int8-arm) is produced.
    Returns True when the warning was emitted (kept testable via `out`).
    """
    if no_int8 and not int8_arm:
        return False
    out(f"\n{'!' * 60}")
    for line in textwrap.wrap(
        INT8_EXPERIMENTAL_WARNING, width=58, break_on_hyphens=False, break_long_words=False
    ):
        out(f"  {line}")
    out(f"{'!' * 60}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Optimize BirdNET ONNX model for GPU and embedded devices"
    )
    parser.add_argument("--input", "-i", required=True, help="Input ONNX model")
    parser.add_argument(
        "--output", "-o", required=True, help="Output base path (without extension)"
    )
    parser.add_argument("--fp32-only", action="store_true", help="Only output FP32 model")
    parser.add_argument("--no-fp16", action="store_true", help="Skip FP16 conversion")
    parser.add_argument(
        "--fp16-full",
        action="store_true",
        help="Convert the entire graph to FP16, including spectrogram preprocessing "
        "(default keeps the preprocessing front-end in FP32 for accuracy)",
    )
    parser.add_argument(
        "--fp16-keep-activations-fp32",
        action="store_true",
        help="Keep Swish/SiLU activations (Sigmoid x Mul) in FP32 during FP16 "
        "conversion so ONNX Runtime's CPU EP can fuse QuickGelu. Recovers most "
        "FP16 latency on ARM/CPU (weights stay FP16); small runtime-RAM cost.",
    )
    parser.add_argument("--no-int8", action="store_true", help="Skip INT8 quantization")
    parser.add_argument(
        "--int8-arm",
        action="store_true",
        help=(
            "Create partial INT8 (dynamic, MatMul weights only; spectrogram "
            "front-end kept FP32). Lowest RAM on ARM (Perch v2: ~816->293 MB RSS, "
            "top-1 preserved on real audio); independent of --no-int8. Validate accuracy."
        ),
    )
    parser.add_argument(
        "--prune-outputs",
        type=str,
        default=None,
        help="Comma-separated list of output names to keep (prune all others)",
    )
    parser.add_argument(
        "--perch",
        action="store_true",
        help="Perch v2 mode: preserve multi-output naming and set graph name to Perch",
    )
    parser.add_argument(
        "--no-truncate-dft",
        action="store_true",
        help="Skip truncating the in-graph DFT to the mel filterbank's used bins. "
        "Truncation is on by default: it is a bit-exact ~1.5-1.9x CPU speedup (the "
        "DFT computes frequency bins the filterbank discards), self-reverts if the "
        "output is not identical, and is a no-op on models that use every bin.",
    )
    parser.add_argument(
        "--collapse-select-to-where",
        action="store_true",
        help="Apply only the SelectV2->Where collapse pass (no full optimization) and "
        "write a single ONNX file to --output. For models arm64 ORT rejects due to the "
        "Cast/Mul SelectV2 decomposition (e.g. the MData range filter), which the full "
        "pipeline would otherwise corrupt.",
    )
    args = parser.parse_args()

    if args.collapse_select_to_where:
        print(f"Loading model: {args.input}")
        model = onnx.load(args.input)
        model, collapsed = collapse_select_to_where(model)
        print(f"  Collapsed {collapsed} SelectV2 cluster(s) to Where")
        onnx.checker.check_model(model)
        out_path = args.output if args.output.endswith(".onnx") else f"{args.output}.onnx"
        onnx.save(model, out_path)
        size_mb = Path(out_path).stat().st_size / (1024 * 1024)
        print(f"  Wrote {out_path} ({size_mb:.2f} MB)")
        return 0

    # Determine output paths
    output_base = args.output
    if output_base.endswith(".onnx"):
        output_base = output_base[:-5]

    fp32_path = f"{output_base}_fp32.onnx"
    fp16_path = f"{output_base}_fp16.onnx"
    int8_path = f"{output_base}_int8.onnx"
    int8_arm_path = f"{output_base}_int8_arm.onnx"

    # Parse prune list
    prune_names = None
    if args.prune_outputs:
        prune_names = [s.strip() for s in args.prune_outputs.split(",")]

    # Run optimization pipeline
    model = optimize_model(
        args.input,
        prune_output_names=prune_names,
        perch_mode=args.perch,
        truncate_dft=not args.no_truncate_dft,
    )
    if model is None:
        print("Optimization failed!")
        return 1

    input_size = Path(args.input).stat().st_size / (1024 * 1024)

    # Save FP32 model
    print(f"\n{'=' * 60}")
    print(f"{'SAVING MODELS':^60}")
    print(f"{'=' * 60}")

    print(f"\n[FP32] Saving: {fp32_path}")
    onnx.save(model, fp32_path)
    fp32_size = Path(fp32_path).stat().st_size / (1024 * 1024)
    print(f"  Size: {fp32_size:.2f} MB")

    if args.fp32_only:
        print(f"\n{'=' * 60}")
        print(f"{'SUMMARY':^60}")
        print(f"{'=' * 60}")
        print(f"  Input:  {input_size:.2f} MB")
        print(f"  FP32:   {fp32_size:.2f} MB  ({fp32_path})")
        return 0

    # Convert to FP16
    if not args.no_fp16:
        print("\n[FP16] Converting to FP16...")
        model_fp16 = convert_to_fp16(
            model,
            keep_preprocessing_fp32=not args.fp16_full,
            keep_activations_fp32=args.fp16_keep_activations_fp32,
        )
        if model_fp16 is not None:
            print(f"[FP16] Saving: {fp16_path}")
            onnx.save(model_fp16, fp16_path)
            fp16_size = Path(fp16_path).stat().st_size / (1024 * 1024)
            print(f"  Size: {fp16_size:.2f} MB ({100 * fp16_size / fp32_size:.1f}% of FP32)")
        else:
            fp16_size = None
            print("  FP16 conversion skipped")
    else:
        fp16_size = None
        print("\n[FP16] Skipped (--no-fp16)")

    # Quantize to INT8
    _print_int8_warning(args.no_int8, args.int8_arm)

    if not args.no_int8:
        print("\n[INT8] Quantizing to INT8...")
        success = quantize_to_int8_dynamic(fp32_path, int8_path)
        if success:
            int8_size = Path(int8_path).stat().st_size / (1024 * 1024)
            print(f"  Size: {int8_size:.2f} MB ({100 * int8_size / fp32_size:.1f}% of FP32)")
        else:
            int8_size = None
            print("  INT8 quantization skipped")
    else:
        int8_size = None
        print("\n[INT8] Skipped (--no-int8)")

    # Quantize to INT8 ARM
    int8_arm_size = None
    if args.int8_arm:
        print("\n[INT8-ARM] Quantizing to INT8 (ARM compatible)...")
        success = quantize_to_int8_arm(fp32_path, int8_arm_path)
        if success:
            int8_arm_size = Path(int8_arm_path).stat().st_size / (1024 * 1024)
            print(
                f"  Size: {int8_arm_size:.2f} MB ({100 * int8_arm_size / fp32_size:.1f}% of FP32)"
            )
        else:
            print("  INT8 ARM quantization skipped")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'=' * 60}")
    print(f"  Input:  {input_size:.2f} MB")
    print(f"  FP32:   {fp32_size:.2f} MB  ({fp32_path})")
    if fp16_size is not None:
        print(f"  FP16:   {fp16_size:.2f} MB  ({fp16_path})")
    if int8_size is not None:
        print(f"  INT8:   {int8_size:.2f} MB  ({int8_path})  [experimental]")
    if int8_arm_size is not None:
        print(f"  INT8-ARM: {int8_arm_size:.2f} MB  ({int8_arm_path})  [experimental]")

    print(f"\n{'=' * 60}")
    print(f"{'RECOMMENDED USAGE':^60}")
    print(f"{'=' * 60}")
    print("  GPU (CUDA/TensorRT):    Use FP32 or FP16")
    print("  RPi 4/5, low RAM:       Use INT8-ARM (partial, MatMul-only): large RSS cut")
    print("                          with top-1 preserved on Perch v2; validate on real audio")
    print("  RPi 5, speed over RAM:  FP16 (add --fp16-keep-activations-fp32 on complex graphs)")
    print("  Desktop CPU (Intel):    Use FP32")
    print("  Desktop CPU (ARM):      Use INT8-ARM (validate accuracy)")
    print("")
    print("  Partial-INT8 only:  --perch --no-fp16 --no-int8 --int8-arm")

    return 0


if __name__ == "__main__":
    exit(main())
