# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch._inductor.ir import Reduction
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from . import config
from .constants import BATCH_MATMUL_FP8_OP, DEVICE_NAME
from .scheduler import CountedLoopSchedulerNode


def _make_fused(
    nodes: list[SchedulerNode | CountedLoopSchedulerNode],
) -> BaseSchedulerNode | None:
    if len(nodes) > 1:
        return FusedSchedulerNode(nodes[0].scheduler, nodes)
    elif len(nodes) == 1:
        return nodes[0]
    return None


def _is_spyre_node(node: BaseSchedulerNode) -> bool:
    """True if the node computes on the Spyre device."""
    device = node.get_device()
    return device is not None and device.type == DEVICE_NAME


def _is_fp8_matmul(n: BaseSchedulerNode) -> bool:
    """Return True if n is a batchmatmulfp8 reduction node."""
    return (
        isinstance(n, SchedulerNode)
        and n.node is not None
        and isinstance(n.node.data, Reduction)
        and n.node.data.reduction_type == BATCH_MATMUL_FP8_OP
    )


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.

    batchmatmulfp8 must not be fused with upstream pointwise ops whose tensors
    have a different rank. DeepTools assigns dim labels per-op independently;
    when a pointwise of rank R and a BMM of rank R+1 share the same SDSC, the
    global layout merge produces label conflicts (e.g. multiple "mb" or "y"
    candidates) that cause error 2497. Force a bundle boundary immediately
    before every batchmatmulfp8 node so its producer (e.g. qfp8wt) always
    lands in a preceding SDSC.
    """
    if len(nodes) == 0:
        return nodes
    if not config.bundle_symbolic_args:
        # Without symbolic args, tensor addresses are baked-in constants from
        # SEGMENT_OFFSETS, which has a fixed number of slots.  Fusing ops could
        # exceed that slot count, so disable fusion when symbolic args are off.
        return nodes

    fused_nodes: list[BaseSchedulerNode] = []
    cur_nodes: list[SchedulerNode | CountedLoopSchedulerNode] = []

    for n in nodes:
        if isinstance(n, (SchedulerNode, CountedLoopSchedulerNode)) and _is_spyre_node(
            n
        ):
            # batchmatmulfp8 must always start a fresh bundle.
            if _is_fp8_matmul(n) and cur_nodes:
                if fused := _make_fused(cur_nodes):
                    fused_nodes.append(fused)
                cur_nodes = []
            cur_nodes.append(n)
        else:
            # Non-Spyre nodes (Fallback nodes, CPU SchedulerNodes) force a
            # bundle boundary.
            if fused := _make_fused(cur_nodes):
                fused_nodes.append(fused)
            fused_nodes.append(n)
            cur_nodes = []

    if fused := _make_fused(cur_nodes):
        fused_nodes.append(fused)

    return fused_nodes
