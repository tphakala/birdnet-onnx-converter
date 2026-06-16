"""Tests for the SelectV2 -> Where graph collapse pass.

tf2onnx lowers tf.where(cond, T, F) (SelectV2) into a Cast/Mul/Add cluster that
arm64 ONNX Runtime rejects. ``collapse_select_to_where`` rewrites it back to a
single ``Where`` op. These tests pin the structural rewrite and its numerical
equivalence, plus the conservative guard that leaves shared/unmatched graphs
untouched.
"""

import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto


def _make_selectv2_decomposition() -> onnx.ModelProto:
    """Build the tf2onnx SelectV2 decomposition of where(X > 0, T, F):

        cond  = Greater(X, 0)            # bool
        mul_t = Mul(Cast(cond), T)       # T where cond     (cast first)
        mul_f = Mul(F, Cast(Not(cond)))  # F where not cond (const first)
        OUT   = Add(mul_t, mul_f)

    The two Mul operand orders are intentionally different to exercise Mul
    commutativity in the matcher.
    """
    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    out = h.make_tensor_value_info("OUT", TensorProto.FLOAT, [4])
    zero = h.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    t = h.make_tensor("T", TensorProto.FLOAT, [4], [10.0, 10.0, 10.0, 10.0])
    f = h.make_tensor("F", TensorProto.FLOAT, [4], [-10.0, -10.0, -10.0, -10.0])
    nodes = [
        h.make_node("Greater", ["X", "zero"], ["cond"], name="Greater"),
        h.make_node("Cast", ["cond"], ["cast_t"], name="Cast_t", to=TensorProto.FLOAT),
        h.make_node("Mul", ["cast_t", "T"], ["mul_t"], name="Mul_t"),
        h.make_node("Not", ["cond"], ["ncond"], name="Not"),
        h.make_node("Cast", ["ncond"], ["cast_f"], name="Cast_f", to=TensorProto.FLOAT),
        h.make_node("Mul", ["F", "cast_f"], ["mul_f"], name="Mul_f"),
        h.make_node("Add", ["mul_t", "mul_f"], ["OUT"], name="SelectV2"),
    ]
    graph = h.make_graph(nodes, "selectv2", [x], [out], [zero, t, f])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def test_collapse_rewrites_cluster_to_where():
    """The Cast/Mul/Not/Add cluster collapses to a single Where(cond, T, F)."""
    from optimize import collapse_select_to_where, count_ops

    model, collapsed = collapse_select_to_where(_make_selectv2_decomposition())

    assert collapsed == 1
    ops = count_ops(model)
    assert ops.get("Where", 0) == 1
    for gone in ("Add", "Mul", "Not", "Cast"):
        assert ops.get(gone, 0) == 0, f"{gone} should be gone, ops={ops}"
    # the condition producer is kept and feeds Where natively
    assert ops.get("Greater", 0) == 1
    where = next(n for n in model.graph.node if n.op_type == "Where")
    assert where.output[0] == "OUT", "the original output tensor name must be preserved"
    assert list(where.input) == ["cond", "T", "F"]
    onnx.checker.check_model(model)


def test_collapse_is_numerically_identical():
    """Collapsed model produces byte-identical output to the decomposition."""
    import pytest

    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not available")

    from optimize import collapse_select_to_where

    original = _make_selectv2_decomposition()
    collapsed, n = collapse_select_to_where(_make_selectv2_decomposition())
    assert n == 1

    x = np.array([1.0, -2.0, 3.0, -0.5], dtype=np.float32)

    def run(model: onnx.ModelProto) -> np.ndarray:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run(None, {"X": x})[0]

    np.testing.assert_array_equal(run(original), run(collapsed))


def test_collapse_is_noop_without_pattern():
    """A graph without the Select decomposition is left unchanged."""
    from optimize import collapse_select_to_where

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y = h.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    w = h.make_tensor("w", TensorProto.FLOAT, [4], [1.0, 1.0, 1.0, 1.0])
    graph = h.make_graph([h.make_node("Mul", ["X", "w"], ["Y"])], "g", [x], [y], [w])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    _, n = collapse_select_to_where(model)
    assert n == 0


def test_collapse_skips_shared_intermediate():
    """If an intermediate (Mul output) is consumed outside the cluster, the
    cluster must NOT be collapsed; removing it would drop a live tensor."""
    from optimize import collapse_select_to_where

    model = _make_selectv2_decomposition()
    extra = h.make_tensor_value_info("EXTRA", TensorProto.FLOAT, [4])
    model.graph.output.append(extra)
    model.graph.node.append(h.make_node("Identity", ["mul_t"], ["EXTRA"], name="extra"))

    _, n = collapse_select_to_where(model)
    assert n == 0


def test_collapse_handles_two_clusters():
    """Two independent Select clusters (the real MData shape) both collapse."""
    from optimize import collapse_select_to_where, count_ops

    base = _make_selectv2_decomposition()
    g = base.graph
    # Second cluster on a Less condition feeding a second output.
    out2 = h.make_tensor_value_info("OUT2", TensorProto.FLOAT, [4])
    g.output.append(out2)
    g.initializer.append(h.make_tensor("one", TensorProto.FLOAT, [], [1.0]))
    g.node.extend(
        [
            h.make_node("Less", ["X", "one"], ["cond2"], name="Less"),
            h.make_node("Cast", ["cond2"], ["cast_t2"], name="Cast_t2", to=TensorProto.FLOAT),
            h.make_node("Mul", ["cast_t2", "T"], ["mul_t2"], name="Mul_t2"),
            h.make_node("Not", ["cond2"], ["ncond2"], name="Not2"),
            h.make_node("Cast", ["ncond2"], ["cast_f2"], name="Cast_f2", to=TensorProto.FLOAT),
            h.make_node("Mul", ["F", "cast_f2"], ["mul_f2"], name="Mul_f2"),
            h.make_node("Add", ["mul_t2", "mul_f2"], ["OUT2"], name="SelectV2_1"),
        ]
    )

    model, collapsed = collapse_select_to_where(base)
    assert collapsed == 2
    ops = count_ops(model)
    assert ops.get("Where", 0) == 2
    assert ops.get("Add", 0) == 0 and ops.get("Mul", 0) == 0 and ops.get("Not", 0) == 0
    onnx.checker.check_model(model)
