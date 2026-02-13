"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass
import inspect
import heapq
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

@dataclass(frozen=True)
class KernelConfig:
    reload_nodes_on_r0: bool = False
    idx_wrap_xor: bool = False
    # Store indices internally as 1-based (j = idx + 1). This turns the update
    #   idx_next = 2*idx + 1 + parity
    # into
    #   j_next = 2*j + parity
    # and we convert back to 0-based only for the final store.
    idx_one_based: bool = True
    # Scheduler heuristic:
    #   * load_first_all prioritizes long-distance (long-latency) loads globally,
    #     which helps keep the load pipeline fed.
    #   * load_dist_cap clamps the priority signal so extremely long-distance loads
    #     don't starve other work.
    scheduler_mode: str = "height_first_flow"
    load_dist_cap: int = 7
    use_vector_parity: bool = False
    hash_fusion: bool = True
    idx_update_scalar: bool = False
    idx_update_add_one_last: bool = False
    idx_update_add_alu: bool = False
    hash_fusion_mask: int = 0b111
    scheduler_tiebreak: str = "reverse_op_id"  # "op_id" or "reverse_op_id"
    scheduler_tie_seed: int = 0
    scheduler_load_bias: bool = False
    tmp3_groups: int = 22  # 0 = per-chunk tmp3, >0 = shared groups
    tmp4_groups: int = 1  # 0 = per-chunk tmp4, >0 = shared groups (can differ from tmp3_groups)
    preload_r_mod3: bool = True
    r_mod2_valu_select: bool = False
    r_mod2_tail_valu_select: bool = False
    use_alu_add_imm: bool = True
    node_xor_alu: bool = True
    r_mod1_valu_select: bool = False
    hash_xor_alu: bool = False
    r_mod1_cond_alu: bool = True
    r_mod2_cond_alu: bool = True
    r_mod2_offset_alu: bool = True
    r_mod2_lane_dedup: bool = False
    idx_wrap_xor_alu: bool = False
    r_mod2_cond1_shift_alu: bool = False
    idx_update_alu: bool = False
    r_mod1_select_mask_alu: bool = False
    # For r_mod==3, derive selector bits directly from idx in ALU instead of
    # loading a precomputed mask vector. This frees load bandwidth and shortens
    # the critical path.
    r_mod3_cond_alu: bool = False
    r_mod3_offset_alu: bool = False
    r_mod3_reuse_conds: bool = True
    r_mod3_pair_valu_select: bool = False
    r_mod3_full_valu_select: bool = False
    r_mod3_sel_low_valu_select: bool = False
    preload_r_mod3_half: bool = False
    r_mod3_mask_select_alu: bool = False
    preload_r_mod4: bool = False
    r_mod4_valu_select: bool = False
    node_addr_valu: bool = False
    idx_reset_on_r0: bool = False
    chunk_block: int = 0
    probe_alu_ops: int = 0
    hash_ping_pong: bool = True
    tmp3_group_barrier: bool = False
    tmp3_group_barrier_rmod3: bool = False
    interleave_load_xor: bool = False
    hash_reuse_idx: bool = False
    split_node_addr_val: bool = False
    scheduler_pressure_tiebreak: bool = False
    val_addr_tmps: int = 2
    val_addr_tmps_load: int = 0
    val_addr_tmps_store: int = 1
    debug_op_tags: bool = False
    debug_dump_valu_subs: bool = False
    debug_dump_valu_subs_top: int = 20


class KernelBuilder:
    def __init__(self, config: KernelConfig | None = None):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.config = config or KernelConfig()

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel with software pipelining and VLIW slot packing.
        """
        # Reset state for rebuild
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        config = self.config

        V = VLEN
        num_chunks = batch_size // V
        tail = batch_size % V
        block_size = (
            config.chunk_block
            if (config.chunk_block and config.chunk_block > 0 and config.chunk_block < num_chunks)
            else num_chunks
        )

        class Op:
            __slots__ = ("engine", "slot", "deps", "users", "height", "reads", "writes")

            def __init__(self, engine, slot, deps, reads, writes):
                self.engine = engine
                self.slot = slot
                # deps: list[(dep_op_id, latency_cycles)]
                self.deps = deps
                # users: list[(user_op_id, latency_cycles)]
                self.users = []
                # A latency-aware critical path estimate in *cycles* (ignores resource limits)
                self.height = 1
                self.reads = reads
                self.writes = writes

        ops: list[Op] = []
        # Track hazards per scratch address.
        #
        # The simulator has read-before-write semantics within a cycle: all reads see
        # the old scratch values, and writes commit at the end of the cycle. This
        # means WAR (write-after-read) hazards can be scheduled in the *same* cycle.
        #
        # We therefore model:
        # - RAW and WAW as 1-cycle latency constraints
        # - WAR as a 0-cycle constraint
        last_write: dict[int, int] = {}
        # Last read since the most recent write to that address
        last_read: dict[int, int] = {}

        def vec_addrs(base: int):
            return list(range(base, base + V))

        tag_ops = config.debug_op_tags or config.debug_dump_valu_subs
        op_tags = [] if tag_ops else None

        def add_op(engine, slot, reads, writes, tag=None):
            # Map dep_op_id -> latency (keep the max latency if duplicated)
            dep_lat: dict[int, int] = {}

            # RAW: reads must see the most recent write, available next cycle
            for addr in reads:
                lw = last_write.get(addr)
                if lw is not None:
                    if dep_lat.get(lw, -1) < 1:
                        dep_lat[lw] = 1

            # WAW: preserve write order (kept as 1-cycle for simplicity)
            # WAR: a write may be in the same cycle as a prior read
            for addr in writes:
                lw = last_write.get(addr)
                if lw is not None:
                    if dep_lat.get(lw, -1) < 1:
                        dep_lat[lw] = 1
                lr = last_read.get(addr)
                if lr is not None:
                    # Only add the WAR edge if it isn't already a 1-cycle dep
                    if dep_lat.get(lr, -1) < 0:
                        dep_lat[lr] = 0

            deps = list(dep_lat.items())
            op = Op(engine, slot, deps, reads, writes)
            op_id = len(ops)
            ops.append(op)
            if op_tags is not None:
                if tag is None:
                    frame = inspect.currentframe()
                    fb = frame.f_back if frame is not None else None
                    if fb is not None:
                        tag = f"{fb.f_code.co_filename}:{fb.f_lineno}"
                    else:
                        tag = "<unknown>"
                op_tags.append(tag)
            for d, lat in deps:
                ops[d].users.append((op_id, lat))

            # Update hazard trackers
            for addr in reads:
                last_read[addr] = op_id
            for addr in writes:
                last_write[addr] = op_id
                # Reads before this write no longer constrain future writes
                if addr in last_read:
                    del last_read[addr]

            return op_id

        def add_imm_op(dest: int, src: int, imm: int):
            if config.use_alu_add_imm:
                imm_const = const(imm)
                add_op(
                    "alu",
                    ("+", dest, src, imm_const),
                    reads=[src, imm_const],
                    writes=[dest],
                )
            else:
                add_op(
                    "flow",
                    ("add_imm", dest, src, imm),
                    reads=[src],
                    writes=[dest],
                )

        def xor_vec_inplace(dst_base: int, src_base: int):
            if config.node_xor_alu:
                for lane in range(V):
                    add_op(
                        "alu",
                        ("^", dst_base + lane, dst_base + lane, src_base + lane),
                        reads=[dst_base + lane, src_base + lane],
                        writes=[dst_base + lane],
                    )
            else:
                add_op(
                    "valu",
                    ("^", dst_base, dst_base, src_base),
                    reads=vec_addrs(dst_base) + vec_addrs(src_base),
                    writes=vec_addrs(dst_base),
                )

        def alloc_vec(name=None):
            return self.alloc_scratch(name, V)

        def const(val, name=None):
            if val in self.const_map:
                return self.const_map[val]
            addr = self.alloc_scratch(name)
            self.const_map[val] = addr
            add_op("load", ("const", addr, val), reads=[], writes=[addr])
            return addr

        vconst_map = {}

        def vconst(val, name=None):
            if val in vconst_map:
                return vconst_map[val]
            src = const(val)
            dest = alloc_vec(name)
            add_op("valu", ("vbroadcast", dest, src), reads=[src], writes=vec_addrs(dest))
            vconst_map[val] = dest
            return dest

        # Constants and base pointers
        forest_values_p = const(7, "forest_values_p")
        forest_values_p_idx = (
            const(6, "forest_values_p_idx") if config.idx_one_based else forest_values_p
        )
        inp_indices_p = const(7 + n_nodes, "inp_indices_p")
        inp_values_p = const(7 + n_nodes + batch_size, "inp_values_p")
        n_nodes_const = const(n_nodes, "n_nodes")
        one_const = const(1)
        zero_const = (
            const(0)
            if (
                config.r_mod1_select_mask_alu
                or config.r_mod3_mask_select_alu
                or (config.idx_reset_on_r0 and (config.idx_update_scalar or config.idx_update_alu))
            )
            else None
        )
        two_const = (
            const(2)
            if (config.r_mod2_cond_alu or config.r_mod3_cond_alu or config.r_mod2_lane_dedup)
            else None
        )
        three_const = (
            const(4)
            if (config.idx_one_based and (config.r_mod2_offset_alu or config.r_mod2_lane_dedup))
            else (const(3) if (config.r_mod2_offset_alu or config.r_mod2_lane_dedup) else None)
        )
        eleven_const = (
            const(12)
            if (config.idx_one_based and config.preload_r_mod3 and config.r_mod3_reuse_conds)
            else (
                const(11) if (config.preload_r_mod3 and config.r_mod3_reuse_conds) else None
            )
        )

        probe_addrs = []
        if config.probe_alu_ops > 0:
            for i in range(config.probe_alu_ops):
                probe_addrs.append(self.alloc_scratch(f"probe_{i}"))

        zero_vec = vconst(0, "zero_vec")
        one_vec = vconst(1, "one_vec")
        two_vec = vconst(2, "two_vec")
        three_vec = vconst(3, "three_vec")
        need_four_vec = (
            config.preload_r_mod4
            or config.preload_r_mod3_half
            or config.r_mod3_full_valu_select
            or config.r_mod3_mask_select_alu
            or (config.preload_r_mod3 and not config.r_mod3_reuse_conds)
        )
        four_vec = vconst(4, "four_vec") if need_four_vec else None

        seven_vec = (
            vconst(8, "seven_vec")
            if (config.idx_one_based and (config.preload_r_mod3 or config.preload_r_mod3_half))
            else (
                vconst(7, "seven_vec")
                if (config.preload_r_mod3 or config.preload_r_mod3_half)
                else None
            )
        )
        if config.preload_r_mod4:
            eight_vec = vconst(8, "eight_vec")
            fifteen_vec = vconst(15, "fifteen_vec")
        forest_values_vec = (
            vconst(6 if config.idx_one_based else 7, "forest_values_vec")
            if config.node_addr_valu
            else None
        )
        r_mod3_offset_const = None
        if config.r_mod3_offset_alu:
            r_mod3_offset_const = const(8) if config.idx_one_based else forest_values_p

        hash_vconsts = {}
        hash_mul_vconsts = {}
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if val1 not in hash_vconsts:
                hash_vconsts[val1] = vconst(val1)
            if val3 not in hash_vconsts:
                hash_vconsts[val3] = vconst(val3)
            if op2 == "+" and op1 == "+" and op3 == "<<":
                mul_val = (1 + (1 << val3)) % (2**32)
                if mul_val not in hash_mul_vconsts:
                    hash_mul_vconsts[mul_val] = vconst(mul_val)

        # Preload small-level node values (indices 0..6) into vectors
        node_vecs = {}
        temp_addr = self.alloc_scratch("node_addr_tmp")
        node_scalar_tmp = self.alloc_scratch("node_scalar_tmp")
        val_addr_tmps_load = []
        val_addr_tmps_store = []
        load_tmps = config.val_addr_tmps_load if config.val_addr_tmps_load > 0 else config.val_addr_tmps
        store_tmps = (
            config.val_addr_tmps_store if config.val_addr_tmps_store > 0 else config.val_addr_tmps
        )
        if load_tmps > 0:
            for i in range(load_tmps):
                val_addr_tmps_load.append(self.alloc_scratch(f"val_addr_tmp_load{i}"))
        if store_tmps > 0:
            for i in range(store_tmps):
                val_addr_tmps_store.append(self.alloc_scratch(f"val_addr_tmp_store{i}"))
        for node_idx in range(7):
            node_vecs[node_idx] = alloc_vec(f"node_{node_idx}_vec")

        def load_node_vecs(start_idx: int, dest_vecs: list[int]):
            add_imm_op(temp_addr, forest_values_p, start_idx)
            for i, dest_vec in enumerate(dest_vecs):
                add_op(
                    "load",
                    ("load", node_scalar_tmp, temp_addr),
                    reads=[temp_addr],
                    writes=[node_scalar_tmp],
                )
                add_op(
                    "valu",
                    ("vbroadcast", dest_vec, node_scalar_tmp),
                    reads=[node_scalar_tmp],
                    writes=vec_addrs(dest_vec),
                )
                if i + 1 < len(dest_vecs):
                    add_imm_op(temp_addr, temp_addr, 1)

        preload0_count = min(7, n_nodes)
        node_vecs_lvl1 = [node_vecs[i] for i in range(preload0_count)]
        if node_vecs_lvl1:
            load_node_vecs(0, node_vecs_lvl1)

        node_vecs_lvl3 = None
        if config.preload_r_mod3 or config.preload_r_mod3_half:
            end_idx = 15 if config.preload_r_mod3 else 11
            node_vecs_lvl3 = [alloc_vec(f"node_{i}_vec") for i in range(7, end_idx)]
            preload3_end = min(end_idx, n_nodes)
            if preload3_end > 7:
                load_node_vecs(7, node_vecs_lvl3[: preload3_end - 7])

        node_vecs_lvl4 = None
        if config.preload_r_mod4 and forest_height >= 4:
            node_vecs_lvl4 = [alloc_vec(f"node_{i}_vec") for i in range(15, 31)]
            preload4_end = min(31, n_nodes)
            if preload4_end > 15:
                load_node_vecs(15, node_vecs_lvl4[: preload4_end - 15])

        # Per-chunk scratch allocation
        idx_bases = []
        val_bases = []
        node_bases = []
        tmp2_bases = []
        tmp3_bases = []
        tmp4_bases = []
        tmp5_bases = []
        tmp6_bases = []
        tmp7_bases = []
        idx_ptrs = []

        tmp3_group_vecs = None
        tmp4_group_vecs = None
        tmp4_groups = 0
        r2_lane_tmp1 = None
        r2_lane_tmp2 = None
        r2_lane_vec = None
        tmp3_group_barriers = None
        if config.tmp3_groups > 0:
            tmp3_group_vecs = [alloc_vec(f"tmp3_g{g}") for g in range(config.tmp3_groups)]
            if config.tmp3_group_barrier or config.tmp3_group_barrier_rmod3:
                tmp3_group_barriers = [
                    self.alloc_scratch(f"tmp3_barrier_g{g}") for g in range(config.tmp3_groups)
                ]

        # tmp4 can be shared independently from tmp3 (to manage scratch without killing ILP)
        if config.preload_r_mod3 or config.preload_r_mod3_half or config.preload_r_mod4:
            tmp4_groups = config.tmp4_groups if config.tmp4_groups > 0 else config.tmp3_groups
            if tmp4_groups > 0:
                tmp4_group_vecs = [alloc_vec(f"tmp4_g{g}") for g in range(tmp4_groups)]

        for c in range(block_size):
            idx_bases.append(alloc_vec(f"idx_{c}"))
            val_bases.append(alloc_vec(f"val_{c}"))
            node_bases.append(alloc_vec())
            tmp2_bases.append(alloc_vec())
            if config.tmp3_groups > 0:
                assert tmp3_group_vecs is not None
                tmp3_bases.append(tmp3_group_vecs[c % config.tmp3_groups])
            else:
                tmp3_bases.append(alloc_vec())

            if config.preload_r_mod3 or config.preload_r_mod3_half or config.preload_r_mod4:
                if tmp4_groups > 0:
                    assert tmp4_group_vecs is not None
                    tmp4_bases.append(tmp4_group_vecs[c % tmp4_groups])
                else:
                    tmp4_bases.append(alloc_vec())
            else:
                tmp4_bases.append(None)
            if config.preload_r_mod4:
                tmp5_bases.append(alloc_vec())
                tmp6_bases.append(alloc_vec())
                if config.r_mod4_valu_select:
                    tmp7_bases.append(alloc_vec())
        if config.r_mod2_lane_dedup:
            r2_lane_tmp1 = self.alloc_scratch("r2_lane_tmp1")
            r2_lane_tmp2 = self.alloc_scratch("r2_lane_tmp2")
            r2_lane_vec = alloc_vec("r2_lane_vec")

        if num_chunks:
            idx_ptrs.append(inp_indices_p)
            for c in range(1, num_chunks):
                idx_ptr = self.alloc_scratch()
                add_imm_op(idx_ptr, idx_ptrs[c - 1], V)
                idx_ptrs.append(idx_ptr)

        period = forest_height + 1

        def emit_block(block_start: int, block_chunks: int):
            # Initial loads for this block
            for c in range(block_chunks):
                ptr = idx_ptrs[block_start + c]
                add_op(
                    "load",
                    ("vload", idx_bases[c], ptr),
                    reads=[ptr],
                    writes=vec_addrs(idx_bases[c]),
                )
                if config.idx_one_based:
                    add_op(
                        "valu",
                        ("+", idx_bases[c], idx_bases[c], one_vec),
                        reads=vec_addrs(idx_bases[c]) + vec_addrs(one_vec),
                        writes=vec_addrs(idx_bases[c]),
                    )
                val_addr_tmp = (
                    val_addr_tmps_load[c % len(val_addr_tmps_load)]
                    if val_addr_tmps_load
                    else temp_addr
                )
                add_imm_op(val_addr_tmp, ptr, batch_size)
                add_op(
                    "load",
                    ("vload", val_bases[c], val_addr_tmp),
                    reads=[val_addr_tmp],
                    writes=vec_addrs(val_bases[c]),
                )

            # Main rounds
            for r in range(rounds):
                r_mod = r % period
                if config.reload_nodes_on_r0 and r_mod == 0 and r > 0 and node_vecs_lvl1:
                    load_node_vecs(0, node_vecs_lvl1)
                for c in range(block_chunks):
                    idx_base = idx_bases[c]
                    val_base = val_bases[c]
                    node_base = node_bases[c]  # reused as tmp1
                    tmp1_base = node_base
                    tmp2_base = tmp2_bases[c]
                    tmp3_base = tmp3_bases[c]
                    tmp4_base = tmp4_bases[c]
                    tmp5_base = tmp5_bases[c] if config.preload_r_mod4 else None
                    tmp6_base = tmp6_bases[c] if config.preload_r_mod4 else None
                    tmp7_base = (
                        tmp7_bases[c]
                        if (config.preload_r_mod4 and config.r_mod4_valu_select)
                        else None
                    )
                    if config.tmp3_group_barrier and config.tmp3_groups > 0 and tmp3_group_barriers is not None:
                        barrier_addr = tmp3_group_barriers[c % config.tmp3_groups]
                        add_op(
                            "alu",
                            ("+", barrier_addr, barrier_addr, one_const),
                            reads=[barrier_addr, one_const],
                            writes=[barrier_addr],
                        )

                    # Select or load node values depending on round
                    if r_mod == 0:
                        xor_vec_inplace(val_base, node_vecs[0])
                    elif r_mod == 1:
                        # cond = val & 1 (pre-hash)
                        if config.r_mod1_cond_alu:
                            for lane in range(V):
                                add_op(
                                    "alu",
                                    ("&", tmp1_base + lane, val_base + lane, one_const),
                                    reads=[val_base + lane, one_const],
                                    writes=[tmp1_base + lane],
                                )
                        else:
                            add_op(
                                "valu",
                                ("&", tmp1_base, val_base, one_vec),
                                reads=vec_addrs(val_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp1_base),
                            )
                        if config.r_mod1_valu_select:
                            # node_val = node1 + cond * (node2 - node1)
                            add_op(
                                "valu",
                                ("-", tmp2_base, node_vecs[2], node_vecs[1]),
                                reads=vec_addrs(node_vecs[2]) + vec_addrs(node_vecs[1]),
                                writes=vec_addrs(tmp2_base),
                            )
                            add_op(
                                "valu",
                                ("multiply_add", tmp2_base, tmp2_base, tmp1_base, node_vecs[1]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base)
                                + vec_addrs(node_vecs[1]),
                                writes=vec_addrs(tmp2_base),
                            )
                            xor_vec_inplace(val_base, tmp2_base)
                        else:
                            if config.r_mod1_select_mask_alu:
                                assert zero_const is not None
                                for lane in range(V):
                                    add_op(
                                        "alu",
                                        ("-", tmp2_base + lane, zero_const, tmp1_base + lane),
                                        reads=[zero_const, tmp1_base + lane],
                                        writes=[tmp2_base + lane],
                                    )
                                    add_op(
                                        "alu",
                                        ("^", tmp3_base + lane, node_vecs[1] + lane, node_vecs[2] + lane),
                                        reads=[node_vecs[1] + lane, node_vecs[2] + lane],
                                        writes=[tmp3_base + lane],
                                    )
                                    add_op(
                                        "alu",
                                        ("&", tmp3_base + lane, tmp3_base + lane, tmp2_base + lane),
                                        reads=[tmp3_base + lane, tmp2_base + lane],
                                        writes=[tmp3_base + lane],
                                    )
                                    add_op(
                                        "alu",
                                        ("^", tmp2_base + lane, node_vecs[1] + lane, tmp3_base + lane),
                                        reads=[node_vecs[1] + lane, tmp3_base + lane],
                                        writes=[tmp2_base + lane],
                                    )
                                xor_vec_inplace(val_base, tmp2_base)
                            else:
                                # node_val = cond ? node2 : node1
                                add_op(
                                    "flow",
                                    ("vselect", tmp2_base, tmp1_base, node_vecs[2], node_vecs[1]),
                                    reads=vec_addrs(tmp1_base)
                                    + vec_addrs(node_vecs[2])
                                    + vec_addrs(node_vecs[1]),
                                    writes=vec_addrs(tmp2_base),
                                )
                                xor_vec_inplace(val_base, tmp2_base)
                    elif r_mod == 2:
                        if config.r_mod2_lane_dedup:
                            # Slow path (default r_mod2 selection)
                            if config.idx_one_based:
                                # cond0 = idx & 1, cond1 = idx & 2
                                if config.r_mod2_cond_alu:
                                    assert two_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("&", tmp2_base + lane, idx_base + lane, one_const),
                                            reads=[idx_base + lane, one_const],
                                            writes=[tmp2_base + lane],
                                        )
                                        add_op(
                                            "alu",
                                            ("&", tmp3_base + lane, idx_base + lane, two_const),
                                            reads=[idx_base + lane, two_const],
                                            writes=[tmp3_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("&", tmp2_base, idx_base, one_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", tmp3_base, idx_base, two_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                            else:
                                if config.r_mod2_offset_alu:
                                    assert three_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("-", tmp1_base + lane, idx_base + lane, three_const),
                                            reads=[idx_base + lane, three_const],
                                            writes=[tmp1_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("-", tmp1_base, idx_base, three_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(three_vec),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # cond0 = offset & 1, cond1 = offset & 2
                                if config.r_mod2_cond_alu:
                                    assert two_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("&", tmp2_base + lane, tmp1_base + lane, one_const),
                                            reads=[tmp1_base + lane, one_const],
                                            writes=[tmp2_base + lane],
                                        )
                                        add_op(
                                            "alu",
                                            ("&", tmp3_base + lane, tmp1_base + lane, two_const),
                                            reads=[tmp1_base + lane, two_const],
                                            writes=[tmp3_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("&", tmp2_base, tmp1_base, one_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", tmp3_base, tmp1_base, two_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                            # sel_low = cond0 ? node4 : node3
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp2_base, node_vecs[4], node_vecs[3]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[4])
                                + vec_addrs(node_vecs[3]),
                                writes=vec_addrs(tmp1_base),
                            )
                            # sel_high = cond0 ? node6 : node5
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp2_base, node_vecs[6], node_vecs[5]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[6])
                                + vec_addrs(node_vecs[5]),
                                writes=vec_addrs(tmp2_base),
                            )
                            # node_val = cond1 ? sel_high : sel_low
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp3_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )

                            # Fast path: compute node value for lane0 and broadcast
                            assert three_const is not None and two_const is not None
                            assert r2_lane_tmp1 is not None and r2_lane_tmp2 is not None
                            assert r2_lane_vec is not None
                            r2_tmp1 = r2_lane_tmp1
                            r2_tmp2 = r2_lane_tmp2
                            if config.idx_one_based:
                                add_op(
                                    "alu",
                                    ("&", r2_tmp1, idx_base + 0, one_const),
                                    reads=[idx_base + 0, one_const],
                                    writes=[r2_tmp1],
                                )
                                add_op(
                                    "alu",
                                    ("&", r2_tmp2, idx_base + 0, two_const),
                                    reads=[idx_base + 0, two_const],
                                    writes=[r2_tmp2],
                                )
                            else:
                                add_op(
                                    "alu",
                                    ("-", r2_tmp1, idx_base + 0, three_const),
                                    reads=[idx_base + 0, three_const],
                                    writes=[r2_tmp1],
                                )
                                add_op(
                                    "alu",
                                    ("&", r2_tmp1, r2_tmp1, one_const),
                                    reads=[r2_tmp1, one_const],
                                    writes=[r2_tmp1],
                                )
                                add_op(
                                    "alu",
                                    ("-", r2_tmp2, idx_base + 0, three_const),
                                    reads=[idx_base + 0, three_const],
                                    writes=[r2_tmp2],
                                )
                                add_op(
                                    "alu",
                                    ("&", r2_tmp2, r2_tmp2, two_const),
                                    reads=[r2_tmp2, two_const],
                                    writes=[r2_tmp2],
                                )
                            # sel_low into tmp2_base
                            add_op(
                                "valu",
                                ("vbroadcast", tmp2_base, r2_tmp1),
                                reads=[r2_tmp1],
                                writes=vec_addrs(tmp2_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp2_base, node_vecs[4], node_vecs[3]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[4])
                                + vec_addrs(node_vecs[3]),
                                writes=vec_addrs(tmp2_base),
                            )
                            # sel_high into tmp3_base
                            add_op(
                                "valu",
                                ("vbroadcast", tmp3_base, r2_tmp1),
                                reads=[r2_tmp1],
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp3_base, tmp3_base, node_vecs[6], node_vecs[5]),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(node_vecs[6])
                                + vec_addrs(node_vecs[5]),
                                writes=vec_addrs(tmp3_base),
                            )
                            # cond1 vec into shared r2_lane_vec
                            add_op(
                                "valu",
                                ("vbroadcast", r2_lane_vec, r2_tmp2),
                                reads=[r2_tmp2],
                                writes=vec_addrs(r2_lane_vec),
                            )
                            # fast_val into tmp2_base
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, r2_lane_vec, tmp3_base, tmp2_base),
                                reads=vec_addrs(r2_lane_vec)
                                + vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base),
                                writes=vec_addrs(tmp2_base),
                            )
                            # eq_mask = (idx == idx0)
                            add_op(
                                "valu",
                                ("vbroadcast", tmp3_base, idx_base + 0),
                                reads=[idx_base + 0],
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "valu",
                                ("==", tmp3_base, idx_base, tmp3_base),
                                reads=vec_addrs(idx_base) + vec_addrs(tmp3_base),
                                writes=vec_addrs(tmp3_base),
                            )
                            # if equal lanes, use fast_val else slow_val
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp3_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            xor_vec_inplace(val_base, tmp1_base)
                        elif config.r_mod2_valu_select:
                            if config.idx_one_based:
                                # b0 = idx & 1
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, idx_base, one_vec),
                                    reads=vec_addrs(idx_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                            else:
                                # offset = idx - 3
                                if config.r_mod2_offset_alu:
                                    assert three_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("-", tmp1_base + lane, idx_base + lane, three_const),
                                            reads=[idx_base + lane, three_const],
                                            writes=[tmp1_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("-", tmp1_base, idx_base, three_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(three_vec),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # b0 = offset & 1
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, tmp1_base, one_vec),
                                    reads=vec_addrs(tmp1_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                            # sel_low = node3 + b0 * (node4 - node3)
                            add_op(
                                "valu",
                                ("-", tmp1_base, node_vecs[4], node_vecs[3]),
                                reads=vec_addrs(node_vecs[4]) + vec_addrs(node_vecs[3]),
                                writes=vec_addrs(tmp1_base),
                            )
                            add_op(
                                "valu",
                                ("multiply_add", tmp1_base, tmp1_base, tmp2_base, node_vecs[3]),
                                reads=vec_addrs(tmp1_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[3]),
                                writes=vec_addrs(tmp1_base),
                            )
                            # sel_high = node5 + b0 * (node6 - node5)
                            add_op(
                                "valu",
                                ("-", tmp3_base, node_vecs[6], node_vecs[5]),
                                reads=vec_addrs(node_vecs[6]) + vec_addrs(node_vecs[5]),
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "valu",
                                ("multiply_add", tmp3_base, tmp3_base, tmp2_base, node_vecs[5]),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[5]),
                                writes=vec_addrs(tmp3_base),
                            )
                            # b1 = ((idx - 3) & 2) >> 1
                            if config.idx_one_based:
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, idx_base, two_vec),
                                    reads=vec_addrs(idx_base) + vec_addrs(two_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                            else:
                                add_op(
                                    "valu",
                                    ("-", tmp2_base, idx_base, three_vec),
                                    reads=vec_addrs(idx_base) + vec_addrs(three_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, tmp2_base, two_vec),
                                    reads=vec_addrs(tmp2_base) + vec_addrs(two_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                            if config.r_mod2_cond1_shift_alu:
                                for lane in range(V):
                                    add_op(
                                        "alu",
                                        (">>", tmp2_base + lane, tmp2_base + lane, one_const),
                                        reads=[tmp2_base + lane, one_const],
                                        writes=[tmp2_base + lane],
                                    )
                            else:
                                add_op(
                                    "valu",
                                    (">>", tmp2_base, tmp2_base, one_vec),
                                    reads=vec_addrs(tmp2_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                            # node_val = sel_low + b1 * (sel_high - sel_low)
                            add_op(
                                "valu",
                                ("-", tmp3_base, tmp3_base, tmp1_base),
                                reads=vec_addrs(tmp3_base) + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "valu",
                                ("multiply_add", tmp1_base, tmp3_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            xor_vec_inplace(val_base, tmp1_base)
                        elif config.r_mod2_tail_valu_select:
                            if config.idx_one_based:
                                # cond0 = idx & 1, cond1 = idx & 2
                                if config.r_mod2_cond_alu:
                                    assert two_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("&", tmp2_base + lane, idx_base + lane, one_const),
                                            reads=[idx_base + lane, one_const],
                                            writes=[tmp2_base + lane],
                                        )
                                        add_op(
                                            "alu",
                                            ("&", tmp3_base + lane, idx_base + lane, two_const),
                                            reads=[idx_base + lane, two_const],
                                            writes=[tmp3_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("&", tmp2_base, idx_base, one_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", tmp3_base, idx_base, two_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                            else:
                                # offset = idx - 3
                                if config.r_mod2_offset_alu:
                                    assert three_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("-", tmp1_base + lane, idx_base + lane, three_const),
                                            reads=[idx_base + lane, three_const],
                                            writes=[tmp1_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("-", tmp1_base, idx_base, three_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(three_vec),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # cond0 = offset & 1, cond1 = offset & 2
                                if config.r_mod2_cond_alu:
                                    assert two_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("&", tmp2_base + lane, tmp1_base + lane, one_const),
                                            reads=[tmp1_base + lane, one_const],
                                            writes=[tmp2_base + lane],
                                        )
                                        add_op(
                                            "alu",
                                            ("&", tmp3_base + lane, tmp1_base + lane, two_const),
                                            reads=[tmp1_base + lane, two_const],
                                            writes=[tmp3_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("&", tmp2_base, tmp1_base, one_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", tmp3_base, tmp1_base, two_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                            # sel_low = cond0 ? node4 : node3
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp2_base, node_vecs[4], node_vecs[3]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[4])
                                + vec_addrs(node_vecs[3]),
                                writes=vec_addrs(tmp1_base),
                            )
                            # sel_high = cond0 ? node6 : node5
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp2_base, node_vecs[6], node_vecs[5]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[6])
                                + vec_addrs(node_vecs[5]),
                                writes=vec_addrs(tmp2_base),
                            )
                            # cond1 = cond1 >> 1 (0/1)
                            if config.r_mod2_cond1_shift_alu:
                                for lane in range(V):
                                    add_op(
                                        "alu",
                                        (">>", tmp3_base + lane, tmp3_base + lane, one_const),
                                        reads=[tmp3_base + lane, one_const],
                                        writes=[tmp3_base + lane],
                                    )
                            else:
                                add_op(
                                    "valu",
                                    (">>", tmp3_base, tmp3_base, one_vec),
                                    reads=vec_addrs(tmp3_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp3_base),
                                )
                            # node_val = sel_low + cond1 * (sel_high - sel_low)
                            add_op(
                                "valu",
                                ("-", tmp2_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp2_base) + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp2_base),
                            )
                            add_op(
                                "valu",
                                ("multiply_add", tmp1_base, tmp2_base, tmp3_base, tmp1_base),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(tmp3_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            xor_vec_inplace(val_base, tmp1_base)
                        else:
                            if config.idx_one_based:
                                # For j in [4..7], offset=j-4 so low bits match j. Compute conds from idx directly.
                                if config.r_mod2_cond_alu:
                                    assert two_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("&", tmp2_base + lane, idx_base + lane, one_const),
                                            reads=[idx_base + lane, one_const],
                                            writes=[tmp2_base + lane],
                                        )
                                        add_op(
                                            "alu",
                                            ("&", tmp3_base + lane, idx_base + lane, two_const),
                                            reads=[idx_base + lane, two_const],
                                            writes=[tmp3_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("&", tmp2_base, idx_base, one_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", tmp3_base, idx_base, two_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                            else:
                                # offset = idx - 3
                                if config.r_mod2_offset_alu:
                                    assert three_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("-", tmp1_base + lane, idx_base + lane, three_const),
                                            reads=[idx_base + lane, three_const],
                                            writes=[tmp1_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("-", tmp1_base, idx_base, three_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(three_vec),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # cond0 = offset & 1, cond1 = offset & 2
                                if config.r_mod2_cond_alu:
                                    assert two_const is not None
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("&", tmp2_base + lane, tmp1_base + lane, one_const),
                                            reads=[tmp1_base + lane, one_const],
                                            writes=[tmp2_base + lane],
                                        )
                                        add_op(
                                            "alu",
                                            ("&", tmp3_base + lane, tmp1_base + lane, two_const),
                                            reads=[tmp1_base + lane, two_const],
                                            writes=[tmp3_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("&", tmp2_base, tmp1_base, one_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", tmp3_base, tmp1_base, two_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                            # sel_low = cond0 ? node4 : node3
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp2_base, node_vecs[4], node_vecs[3]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[4])
                                + vec_addrs(node_vecs[3]),
                                writes=vec_addrs(tmp1_base),
                            )
                            # sel_high = cond0 ? node6 : node5
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp2_base, node_vecs[6], node_vecs[5]),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs[6])
                                + vec_addrs(node_vecs[5]),
                                writes=vec_addrs(tmp2_base),
                            )
                            # node_val = cond1 ? sel_high : sel_low
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp3_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            xor_vec_inplace(val_base, tmp1_base)
                    elif r_mod == 3 and (config.preload_r_mod3 or config.preload_r_mod3_half):
                        if (
                            config.tmp3_group_barrier_rmod3
                            and config.tmp3_groups > 0
                            and tmp3_group_barriers is not None
                        ):
                            barrier_addr = tmp3_group_barriers[c % config.tmp3_groups]
                            add_op(
                                "alu",
                                ("+", barrier_addr, barrier_addr, one_const),
                                reads=[barrier_addr, one_const],
                                writes=[barrier_addr],
                            )
                        assert node_vecs_lvl3 is not None
                        assert tmp4_base is not None
                        half_preload = config.preload_r_mod3_half and not config.preload_r_mod3
                        if half_preload:
                            # Preload only nodes 7..10, load others, then select by cond2
                            # offset = idx - 7
                            add_op(
                                "valu",
                                ("-", tmp2_base, idx_base, seven_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                writes=vec_addrs(tmp2_base),
                            )
                            # cond0 = offset & 1, cond1 = offset & 2
                            add_op(
                                "valu",
                                ("&", tmp3_base, tmp2_base, one_vec),
                                reads=vec_addrs(tmp2_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp4_base, tmp2_base, two_vec),
                                reads=vec_addrs(tmp2_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp4_base),
                            )
                            # pair0 = cond0 ? node8 : node7
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp3_base, node_vecs_lvl3[1], node_vecs_lvl3[0]),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(node_vecs_lvl3[1])
                                + vec_addrs(node_vecs_lvl3[0]),
                                writes=vec_addrs(tmp1_base),
                            )
                            # pair1 = cond0 ? node10 : node9
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp3_base, node_vecs_lvl3[3], node_vecs_lvl3[2]),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(node_vecs_lvl3[3])
                                + vec_addrs(node_vecs_lvl3[2]),
                                writes=vec_addrs(tmp2_base),
                            )
                            # sel_low = cond1 ? pair1 : pair0
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp4_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp4_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            # Load node values for all lanes into tmp2_base
                            if config.node_addr_valu:
                                assert forest_values_vec is not None
                                add_op(
                                    "valu",
                                    ("+", tmp2_base, idx_base, forest_values_vec),
                                    reads=vec_addrs(idx_base) + vec_addrs(forest_values_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                            else:
                                for lane in range(V):
                                    add_op(
                                        "alu",
                                        ("+", tmp2_base + lane, forest_values_p_idx, idx_base + lane),
                                        reads=[forest_values_p_idx, idx_base + lane],
                                        writes=[tmp2_base + lane],
                                    )
                            for lane in range(V):
                                add_op(
                                    "load",
                                    ("load", tmp2_base + lane, tmp2_base + lane),
                                    reads=[tmp2_base + lane],
                                    writes=[tmp2_base + lane],
                                )
                            # cond2 = (idx - 7) & 4
                            add_op(
                                "valu",
                                ("-", tmp3_base, idx_base, seven_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp3_base, tmp3_base, four_vec),
                                reads=vec_addrs(tmp3_base) + vec_addrs(four_vec),
                                writes=vec_addrs(tmp3_base),
                            )
                            # node_val = cond2 ? loaded : sel_low
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp3_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            xor_vec_inplace(val_base, tmp1_base)
                        elif config.r_mod3_reuse_conds:
                            if config.r_mod3_full_valu_select:
                                # Full arithmetic selection (no vselect)
                                def select_ma(dest, cond01, a_true, b_false):
                                    add_op(
                                        "valu",
                                        ("-", dest, a_true, b_false),
                                        reads=vec_addrs(a_true) + vec_addrs(b_false),
                                        writes=vec_addrs(dest),
                                    )
                                    add_op(
                                        "valu",
                                        ("multiply_add", dest, dest, cond01, b_false),
                                        reads=vec_addrs(dest) + vec_addrs(cond01) + vec_addrs(b_false),
                                        writes=vec_addrs(dest),
                                    )
                                def select_ma_bfalse(dest_bfalse, scratch, cond01, a_true, b_false):
                                    add_op(
                                        "valu",
                                        ("-", scratch, a_true, b_false),
                                        reads=vec_addrs(a_true) + vec_addrs(b_false),
                                        writes=vec_addrs(scratch),
                                    )
                                    add_op(
                                        "valu",
                                        ("multiply_add", dest_bfalse, scratch, cond01, b_false),
                                        reads=vec_addrs(scratch) + vec_addrs(cond01) + vec_addrs(b_false),
                                        writes=vec_addrs(dest_bfalse),
                                    )

                                # offset = idx - 7
                                if config.r_mod3_offset_alu:
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("-", tmp1_base + lane, idx_base + lane, r_mod3_offset_const),
                                            reads=[idx_base + lane, r_mod3_offset_const],
                                            writes=[tmp1_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("-", tmp1_base, idx_base, seven_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # b0 = offset & 1
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, tmp1_base, one_vec),
                                    reads=vec_addrs(tmp1_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                # b1 = (offset & 2) >> 1
                                add_op(
                                    "valu",
                                    ("&", tmp3_base, tmp1_base, two_vec),
                                    reads=vec_addrs(tmp1_base) + vec_addrs(two_vec),
                                    writes=vec_addrs(tmp3_base),
                                )
                                add_op(
                                    "valu",
                                    (">>", tmp3_base, tmp3_base, one_vec),
                                    reads=vec_addrs(tmp3_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp3_base),
                                )

                                # pair0, pair1 -> sel_low (tmp1_base)
                                select_ma(tmp1_base, tmp2_base, node_vecs_lvl3[1], node_vecs_lvl3[0])
                                select_ma(tmp4_base, tmp2_base, node_vecs_lvl3[3], node_vecs_lvl3[2])
                                select_ma_bfalse(tmp1_base, tmp4_base, tmp3_base, tmp4_base, tmp1_base)

                                # pair2, pair3 -> sel_high (tmp3_base)
                                select_ma(tmp4_base, tmp2_base, node_vecs_lvl3[5], node_vecs_lvl3[4])
                                select_ma(tmp3_base, tmp2_base, node_vecs_lvl3[7], node_vecs_lvl3[6])

                                # recompute b1 into tmp2 (reuse slot after b0 no longer needed)
                                add_op(
                                    "valu",
                                    ("-", tmp2_base, idx_base, seven_vec),
                                    reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, tmp2_base, two_vec),
                                    reads=vec_addrs(tmp2_base) + vec_addrs(two_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                add_op(
                                    "valu",
                                    (">>", tmp2_base, tmp2_base, one_vec),
                                    reads=vec_addrs(tmp2_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                select_ma(tmp3_base, tmp2_base, tmp3_base, tmp4_base)

                                # b2 = (offset & 4) >> 2
                                add_op(
                                    "valu",
                                    ("-", tmp2_base, idx_base, seven_vec),
                                    reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, tmp2_base, four_vec),
                                    reads=vec_addrs(tmp2_base) + vec_addrs(four_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                add_op(
                                    "valu",
                                    (">>", tmp2_base, tmp2_base, two_vec),
                                    reads=vec_addrs(tmp2_base) + vec_addrs(two_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                # node_val = b2 ? sel_high : sel_low
                                select_ma_bfalse(tmp1_base, tmp4_base, tmp2_base, tmp3_base, tmp1_base)
                                xor_vec_inplace(val_base, tmp1_base)
                            elif config.r_mod3_mask_select_alu:
                                assert zero_const is not None
                                def mask_select_lane_safe(dest_lane, mask_lane, a_lane, b_lane, scratch_lane):
                                    add_op(
                                        "alu",
                                        ("^", scratch_lane, a_lane, b_lane),
                                        reads=[a_lane, b_lane],
                                        writes=[scratch_lane],
                                    )
                                    add_op(
                                        "alu",
                                        ("&", scratch_lane, scratch_lane, mask_lane),
                                        reads=[scratch_lane, mask_lane],
                                        writes=[scratch_lane],
                                    )
                                    add_op(
                                        "alu",
                                        ("^", dest_lane, a_lane, scratch_lane),
                                        reads=[a_lane, scratch_lane],
                                        writes=[dest_lane],
                                    )

                                def maskify_vec(bits_base):
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("-", bits_base + lane, zero_const, bits_base + lane),
                                            reads=[zero_const, bits_base + lane],
                                            writes=[bits_base + lane],
                                        )

                                def compute_mask_b0(dst_base):
                                    add_op(
                                        "valu",
                                        ("-", dst_base, idx_base, seven_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", dst_base, dst_base, one_vec),
                                        reads=vec_addrs(dst_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    maskify_vec(dst_base)

                                def compute_mask_b1(dst_base):
                                    add_op(
                                        "valu",
                                        ("-", dst_base, idx_base, seven_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", dst_base, dst_base, two_vec),
                                        reads=vec_addrs(dst_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    add_op(
                                        "valu",
                                        (">>", dst_base, dst_base, one_vec),
                                        reads=vec_addrs(dst_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    maskify_vec(dst_base)

                                def compute_mask_b2(dst_base):
                                    add_op(
                                        "valu",
                                        ("-", dst_base, idx_base, seven_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", dst_base, dst_base, four_vec),
                                        reads=vec_addrs(dst_base) + vec_addrs(four_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    add_op(
                                        "valu",
                                        (">>", dst_base, dst_base, two_vec),
                                        reads=vec_addrs(dst_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(dst_base),
                                    )
                                    maskify_vec(dst_base)

                                # mask_b0 in tmp2, mask_b1 in tmp3
                                compute_mask_b0(tmp2_base)
                                compute_mask_b1(tmp3_base)

                                # pair0 (node8 vs node7) -> tmp1_base, scratch tmp4
                                for lane in range(V):
                                    mask_select_lane_safe(
                                        tmp1_base + lane,
                                        tmp2_base + lane,
                                        node_vecs_lvl3[0] + lane,
                                        node_vecs_lvl3[1] + lane,
                                        tmp4_base + lane,
                                    )
                                # pair1 (node10 vs node9) -> tmp4_base (scratch=dest)
                                for lane in range(V):
                                    mask_select_lane_safe(
                                        tmp4_base + lane,
                                        tmp2_base + lane,
                                        node_vecs_lvl3[2] + lane,
                                        node_vecs_lvl3[3] + lane,
                                        tmp4_base + lane,
                                    )
                                # sel_low = b1 ? pair1 : pair0 -> tmp1_base (scratch tmp2, overwrite mask_b0)
                                for lane in range(V):
                                    mask_select_lane_safe(
                                        tmp1_base + lane,
                                        tmp3_base + lane,
                                        tmp1_base + lane,
                                        tmp4_base + lane,
                                        tmp2_base + lane,
                                    )

                                # recompute mask_b0 into tmp2
                                compute_mask_b0(tmp2_base)
                                # pair2 (node12 vs node11) -> tmp4_base (scratch=dest)
                                for lane in range(V):
                                    mask_select_lane_safe(
                                        tmp4_base + lane,
                                        tmp2_base + lane,
                                        node_vecs_lvl3[4] + lane,
                                        node_vecs_lvl3[5] + lane,
                                        tmp4_base + lane,
                                    )
                                # pair3 (node14 vs node13) -> tmp3_base (scratch=dest, overwrites mask_b1)
                                for lane in range(V):
                                    mask_select_lane_safe(
                                        tmp3_base + lane,
                                        tmp2_base + lane,
                                        node_vecs_lvl3[6] + lane,
                                        node_vecs_lvl3[7] + lane,
                                        tmp3_base + lane,
                                    )

                                # recompute mask_b1 into tmp2 (overwrite mask_b0)
                                compute_mask_b1(tmp2_base)
                                # sel_high = b1 ? pair3 : pair2 -> tmp4_base (scratch tmp3)
                                for lane in range(V):
                                    mask_select_lane_safe(
                                        tmp4_base + lane,
                                        tmp2_base + lane,
                                        tmp4_base + lane,
                                        tmp3_base + lane,
                                        tmp3_base + lane,
                                    )

                                # mask_b2 into tmp2
                                compute_mask_b2(tmp2_base)
                                # final select = b2 ? sel_high : sel_low -> tmp1_base (scratch tmp3)
                                for lane in range(V):
                                    mask_select_lane_safe(
                                        tmp1_base + lane,
                                        tmp2_base + lane,
                                        tmp1_base + lane,
                                        tmp4_base + lane,
                                        tmp3_base + lane,
                                    )
                                xor_vec_inplace(val_base, tmp1_base)
                            else:
                                # offset = idx - 7
                                if config.r_mod3_offset_alu:
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("-", tmp1_base + lane, idx_base + lane, r_mod3_offset_const),
                                            reads=[idx_base + lane, r_mod3_offset_const],
                                            writes=[tmp1_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("-", tmp1_base, idx_base, seven_vec),
                                        reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # cond0 = offset & 1, cond1 = offset & 2
                                if config.r_mod3_cond_alu:
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            ("&", tmp2_base + lane, tmp1_base + lane, one_const),
                                            reads=[tmp1_base + lane, one_const],
                                            writes=[tmp2_base + lane],
                                        )
                                        add_op(
                                            "alu",
                                            ("&", tmp3_base + lane, tmp1_base + lane, two_const),
                                            reads=[tmp1_base + lane, two_const],
                                            writes=[tmp3_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("&", tmp2_base, tmp1_base, one_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("&", tmp3_base, tmp1_base, two_vec),
                                        reads=vec_addrs(tmp1_base) + vec_addrs(two_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                                use_pair_valu = (
                                    config.r_mod3_pair_valu_select
                                    or config.r_mod3_sel_low_valu_select
                                )
                                use_sel_low_valu = config.r_mod3_sel_low_valu_select
                                if use_sel_low_valu:
                                    add_op(
                                        "valu",
                                        (">>", tmp3_base, tmp3_base, one_vec),
                                        reads=vec_addrs(tmp3_base) + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp3_base),
                                    )
                                # pair0 = cond0 ? node8 : node7
                                if use_pair_valu:
                                    add_op(
                                        "valu",
                                        ("-", tmp1_base, node_vecs_lvl3[1], node_vecs_lvl3[0]),
                                        reads=vec_addrs(node_vecs_lvl3[1])
                                        + vec_addrs(node_vecs_lvl3[0]),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                    add_op(
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp1_base,
                                            tmp1_base,
                                            tmp2_base,
                                            node_vecs_lvl3[0],
                                        ),
                                        reads=vec_addrs(tmp1_base)
                                        + vec_addrs(tmp2_base)
                                        + vec_addrs(node_vecs_lvl3[0]),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                else:
                                    add_op(
                                        "flow",
                                        (
                                            "vselect",
                                            tmp1_base,
                                            tmp2_base,
                                            node_vecs_lvl3[1],
                                            node_vecs_lvl3[0],
                                        ),
                                        reads=vec_addrs(tmp2_base)
                                        + vec_addrs(node_vecs_lvl3[1])
                                        + vec_addrs(node_vecs_lvl3[0]),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # pair1 = cond0 ? node10 : node9
                                if use_pair_valu:
                                    add_op(
                                        "valu",
                                        ("-", tmp4_base, node_vecs_lvl3[3], node_vecs_lvl3[2]),
                                        reads=vec_addrs(node_vecs_lvl3[3])
                                        + vec_addrs(node_vecs_lvl3[2]),
                                        writes=vec_addrs(tmp4_base),
                                    )
                                    add_op(
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp4_base,
                                            tmp4_base,
                                            tmp2_base,
                                            node_vecs_lvl3[2],
                                        ),
                                        reads=vec_addrs(tmp4_base)
                                        + vec_addrs(tmp2_base)
                                        + vec_addrs(node_vecs_lvl3[2]),
                                        writes=vec_addrs(tmp4_base),
                                    )
                                else:
                                    add_op(
                                        "flow",
                                        (
                                            "vselect",
                                            tmp4_base,
                                            tmp2_base,
                                            node_vecs_lvl3[3],
                                            node_vecs_lvl3[2],
                                        ),
                                        reads=vec_addrs(tmp2_base)
                                        + vec_addrs(node_vecs_lvl3[3])
                                        + vec_addrs(node_vecs_lvl3[2]),
                                        writes=vec_addrs(tmp4_base),
                                    )
                                # sel_low = cond1 ? pair1 : pair0
                                if use_sel_low_valu:
                                    add_op(
                                        "valu",
                                        ("-", tmp4_base, tmp4_base, tmp1_base),
                                        reads=vec_addrs(tmp4_base) + vec_addrs(tmp1_base),
                                        writes=vec_addrs(tmp4_base),
                                    )
                                    add_op(
                                        "valu",
                                        ("multiply_add", tmp1_base, tmp4_base, tmp3_base, tmp1_base),
                                        reads=vec_addrs(tmp4_base)
                                        + vec_addrs(tmp3_base)
                                        + vec_addrs(tmp1_base),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                else:
                                    add_op(
                                        "flow",
                                        ("vselect", tmp1_base, tmp3_base, tmp4_base, tmp1_base),
                                        reads=vec_addrs(tmp3_base)
                                        + vec_addrs(tmp4_base)
                                        + vec_addrs(tmp1_base),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                # pair2 = cond0 ? node12 : node11
                                if config.r_mod3_pair_valu_select:
                                    add_op(
                                        "valu",
                                        ("-", tmp4_base, node_vecs_lvl3[5], node_vecs_lvl3[4]),
                                        reads=vec_addrs(node_vecs_lvl3[5])
                                        + vec_addrs(node_vecs_lvl3[4]),
                                        writes=vec_addrs(tmp4_base),
                                    )
                                    add_op(
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp4_base,
                                            tmp4_base,
                                            tmp2_base,
                                            node_vecs_lvl3[4],
                                        ),
                                        reads=vec_addrs(tmp4_base)
                                        + vec_addrs(tmp2_base)
                                        + vec_addrs(node_vecs_lvl3[4]),
                                        writes=vec_addrs(tmp4_base),
                                    )
                                else:
                                    add_op(
                                        "flow",
                                        (
                                            "vselect",
                                            tmp4_base,
                                            tmp2_base,
                                            node_vecs_lvl3[5],
                                            node_vecs_lvl3[4],
                                        ),
                                        reads=vec_addrs(tmp2_base)
                                        + vec_addrs(node_vecs_lvl3[5])
                                        + vec_addrs(node_vecs_lvl3[4]),
                                        writes=vec_addrs(tmp4_base),
                                    )
                                # pair3 = cond0 ? node14 : node13 (dest overwrites cond0)
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp2_base,
                                        tmp2_base,
                                        node_vecs_lvl3[7],
                                        node_vecs_lvl3[6],
                                    ),
                                    reads=vec_addrs(tmp2_base)
                                    + vec_addrs(node_vecs_lvl3[7])
                                    + vec_addrs(node_vecs_lvl3[6]),
                                    writes=vec_addrs(tmp2_base),
                                )
                                # sel_high = cond1 ? pair3 : pair2
                                add_op(
                                    "flow",
                                    ("vselect", tmp4_base, tmp3_base, tmp2_base, tmp4_base),
                                    reads=vec_addrs(tmp3_base)
                                    + vec_addrs(tmp2_base)
                                    + vec_addrs(tmp4_base),
                                    writes=vec_addrs(tmp4_base),
                                )
                                # cond2: idx < 11 selects the *low* half (nodes 7-10)
                                assert eleven_const is not None
                                for lane in range(V):
                                    add_op(
                                        "alu",
                                        (
                                            "<",
                                            tmp3_base + lane,
                                            idx_base + lane,
                                            eleven_const,
                                        ),
                                        reads=[idx_base + lane, eleven_const],
                                        writes=[tmp3_base + lane],
                                    )

                                # node_val = (idx < 11) ? sel_low : sel_high
                                add_op(
                                    "flow",
                                    ("vselect", tmp1_base, tmp3_base, tmp1_base, tmp4_base),
                                    reads=vec_addrs(tmp3_base)
                                    + vec_addrs(tmp1_base)
                                    + vec_addrs(tmp4_base),
                                    writes=vec_addrs(tmp1_base),
                                )
                                xor_vec_inplace(val_base, tmp1_base)
                        else:
                            # offset = idx - 7
                            add_op(
                                "valu",
                                ("-", tmp1_base, idx_base, seven_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                writes=vec_addrs(tmp1_base),
                            )
                            # cond0 = offset & 1, cond1 = offset & 2
                            if config.r_mod3_cond_alu:
                                for lane in range(V):
                                    add_op(
                                        "alu",
                                        ("&", tmp2_base + lane, tmp1_base + lane, one_const),
                                        reads=[tmp1_base + lane, one_const],
                                        writes=[tmp2_base + lane],
                                    )
                                    add_op(
                                        "alu",
                                        ("&", tmp3_base + lane, tmp1_base + lane, two_const),
                                        reads=[tmp1_base + lane, two_const],
                                        writes=[tmp3_base + lane],
                                    )
                            else:
                                add_op(
                                    "valu",
                                    ("&", tmp2_base, tmp1_base, one_vec),
                                    reads=vec_addrs(tmp1_base) + vec_addrs(one_vec),
                                    writes=vec_addrs(tmp2_base),
                                )
                                add_op(
                                    "valu",
                                    ("&", tmp3_base, tmp1_base, two_vec),
                                    reads=vec_addrs(tmp1_base) + vec_addrs(two_vec),
                                    writes=vec_addrs(tmp3_base),
                                )
                            # pair0 = cond0 ? node8 : node7
                            if config.r_mod3_pair_valu_select:
                                add_op(
                                    "valu",
                                    ("-", tmp1_base, node_vecs_lvl3[1], node_vecs_lvl3[0]),
                                    reads=vec_addrs(node_vecs_lvl3[1])
                                    + vec_addrs(node_vecs_lvl3[0]),
                                    writes=vec_addrs(tmp1_base),
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp1_base,
                                        tmp1_base,
                                        tmp2_base,
                                        node_vecs_lvl3[0],
                                    ),
                                    reads=vec_addrs(tmp1_base)
                                    + vec_addrs(tmp2_base)
                                    + vec_addrs(node_vecs_lvl3[0]),
                                    writes=vec_addrs(tmp1_base),
                                )
                            else:
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp1_base,
                                        tmp2_base,
                                        node_vecs_lvl3[1],
                                        node_vecs_lvl3[0],
                                    ),
                                    reads=vec_addrs(tmp2_base)
                                    + vec_addrs(node_vecs_lvl3[1])
                                    + vec_addrs(node_vecs_lvl3[0]),
                                    writes=vec_addrs(tmp1_base),
                                )
                            # pair1 = cond0 ? node10 : node9
                            add_op(
                                "flow",
                                (
                                    "vselect",
                                    tmp2_base,
                                    tmp2_base,
                                    node_vecs_lvl3[3],
                                    node_vecs_lvl3[2],
                                ),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs_lvl3[3])
                                + vec_addrs(node_vecs_lvl3[2]),
                                writes=vec_addrs(tmp2_base),
                            )
                            # sel_low = cond1 ? pair1 : pair0
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp3_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            # pair2 = cond0 ? node12 : node11
                            add_op(
                                "valu",
                                ("-", tmp2_base, idx_base, seven_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                writes=vec_addrs(tmp2_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp2_base, tmp2_base, one_vec),
                                reads=vec_addrs(tmp2_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp2_base),
                            )
                            add_op(
                                "flow",
                                (
                                    "vselect",
                                    tmp2_base,
                                    tmp2_base,
                                    node_vecs_lvl3[5],
                                    node_vecs_lvl3[4],
                                ),
                                reads=vec_addrs(tmp2_base)
                                + vec_addrs(node_vecs_lvl3[5])
                                + vec_addrs(node_vecs_lvl3[4]),
                                writes=vec_addrs(tmp2_base),
                            )
                            # pair3 = cond0 ? node14 : node13
                            add_op(
                                "valu",
                                ("-", tmp3_base, idx_base, seven_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp3_base, tmp3_base, one_vec),
                                reads=vec_addrs(tmp3_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "flow",
                                (
                                    "vselect",
                                    tmp3_base,
                                    tmp3_base,
                                    node_vecs_lvl3[7],
                                    node_vecs_lvl3[6],
                                ),
                                reads=vec_addrs(tmp3_base)
                                + vec_addrs(node_vecs_lvl3[7])
                                + vec_addrs(node_vecs_lvl3[6]),
                                writes=vec_addrs(tmp3_base),
                            )
                            # sel_high = cond1 ? pair3 : pair2
                            add_op(
                                "valu",
                                ("-", tmp4_base, idx_base, seven_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                writes=vec_addrs(tmp4_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp4_base, tmp4_base, two_vec),
                                reads=vec_addrs(tmp4_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp4_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp4_base, tmp3_base, tmp2_base),
                                reads=vec_addrs(tmp4_base)
                                + vec_addrs(tmp3_base)
                                + vec_addrs(tmp2_base),
                                writes=vec_addrs(tmp2_base),
                            )
                            # node_val = cond2 ? sel_high : sel_low
                            add_op(
                                "valu",
                                ("-", tmp4_base, idx_base, seven_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(seven_vec),
                                writes=vec_addrs(tmp4_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp4_base, tmp4_base, four_vec),
                                reads=vec_addrs(tmp4_base) + vec_addrs(four_vec),
                                writes=vec_addrs(tmp4_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp4_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp4_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            xor_vec_inplace(val_base, tmp1_base)
                    elif r_mod == 4 and config.preload_r_mod4:
                        assert node_vecs_lvl4 is not None
                        assert tmp4_base is not None
                        assert tmp5_base is not None and tmp6_base is not None
                        assert four_vec is not None
                        assert eight_vec is not None
                        assert fifteen_vec is not None

                        if config.r_mod4_valu_select:
                            assert tmp7_base is not None

                            def select_ma(dest, cond, a_true, b_false):
                                add_op(
                                    "valu",
                                    ("-", tmp7_base, a_true, b_false),
                                    reads=vec_addrs(a_true) + vec_addrs(b_false),
                                    writes=vec_addrs(tmp7_base),
                                )
                                add_op(
                                    "valu",
                                    ("multiply_add", dest, tmp7_base, cond, b_false),
                                    reads=vec_addrs(tmp7_base)
                                    + vec_addrs(cond)
                                    + vec_addrs(b_false),
                                    writes=vec_addrs(dest),
                                )

                            # offset = idx - 15
                            add_op(
                                "valu",
                                ("-", tmp7_base, idx_base, fifteen_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(fifteen_vec),
                                writes=vec_addrs(tmp7_base),
                            )
                            # b0 = offset & 1
                            add_op(
                                "valu",
                                ("&", tmp5_base, tmp7_base, one_vec),
                                reads=vec_addrs(tmp7_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            # b1 = (offset & 2) >> 1
                            add_op(
                                "valu",
                                ("&", tmp6_base, tmp7_base, two_vec),
                                reads=vec_addrs(tmp7_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp6_base),
                            )
                            add_op(
                                "valu",
                                (">>", tmp6_base, tmp6_base, one_vec),
                                reads=vec_addrs(tmp6_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp6_base),
                            )

                            # group0 (nodes 15..18) -> tmp1_base
                            select_ma(tmp1_base, tmp5_base, node_vecs_lvl4[1], node_vecs_lvl4[0])
                            select_ma(tmp2_base, tmp5_base, node_vecs_lvl4[3], node_vecs_lvl4[2])
                            select_ma(tmp1_base, tmp6_base, tmp2_base, tmp1_base)

                            # group1 (nodes 19..22) -> tmp2_base
                            select_ma(tmp2_base, tmp5_base, node_vecs_lvl4[5], node_vecs_lvl4[4])
                            select_ma(tmp3_base, tmp5_base, node_vecs_lvl4[7], node_vecs_lvl4[6])
                            select_ma(tmp2_base, tmp6_base, tmp3_base, tmp2_base)

                            # group2 (nodes 23..26) -> tmp3_base
                            select_ma(tmp3_base, tmp5_base, node_vecs_lvl4[9], node_vecs_lvl4[8])
                            select_ma(tmp4_base, tmp5_base, node_vecs_lvl4[11], node_vecs_lvl4[10])
                            select_ma(tmp3_base, tmp6_base, tmp4_base, tmp3_base)

                            # group3 (nodes 27..30) -> tmp4_base (reuse tmp5_base for pair1)
                            select_ma(tmp4_base, tmp5_base, node_vecs_lvl4[13], node_vecs_lvl4[12])
                            select_ma(tmp5_base, tmp5_base, node_vecs_lvl4[15], node_vecs_lvl4[14])
                            select_ma(tmp4_base, tmp6_base, tmp5_base, tmp4_base)

                            # b2 = (offset & 4) >> 2, b3 = (offset & 8) >> 3
                            add_op(
                                "valu",
                                ("-", tmp7_base, idx_base, fifteen_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(fifteen_vec),
                                writes=vec_addrs(tmp7_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp5_base, tmp7_base, four_vec),
                                reads=vec_addrs(tmp7_base) + vec_addrs(four_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "valu",
                                (">>", tmp5_base, tmp5_base, two_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp6_base, tmp7_base, eight_vec),
                                reads=vec_addrs(tmp7_base) + vec_addrs(eight_vec),
                                writes=vec_addrs(tmp6_base),
                            )
                            add_op(
                                "valu",
                                (">>", tmp6_base, tmp6_base, three_vec),
                                reads=vec_addrs(tmp6_base) + vec_addrs(three_vec),
                                writes=vec_addrs(tmp6_base),
                            )

                            # final select among group results
                            select_ma(tmp1_base, tmp5_base, tmp2_base, tmp1_base)
                            select_ma(tmp3_base, tmp5_base, tmp4_base, tmp3_base)
                            select_ma(tmp1_base, tmp6_base, tmp3_base, tmp1_base)
                            xor_vec_inplace(val_base, tmp1_base)
                        else:
                            # Original flow-based selection
                            # group0: nodes 15..18 -> tmp1_base
                            add_op(
                                "valu",
                                ("-", tmp5_base, idx_base, fifteen_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(fifteen_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp6_base, tmp5_base, two_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp6_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp5_base, tmp5_base, one_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp5_base, node_vecs_lvl4[1], node_vecs_lvl4[0]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[1])
                                + vec_addrs(node_vecs_lvl4[0]),
                                writes=vec_addrs(tmp1_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp5_base, tmp5_base, node_vecs_lvl4[3], node_vecs_lvl4[2]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[3])
                                + vec_addrs(node_vecs_lvl4[2]),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp6_base, tmp5_base, tmp1_base),
                                reads=vec_addrs(tmp6_base)
                                + vec_addrs(tmp5_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )

                            # group1: nodes 19..22 -> tmp2_base
                            add_op(
                                "valu",
                                ("-", tmp5_base, idx_base, fifteen_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(fifteen_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp6_base, tmp5_base, two_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp6_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp5_base, tmp5_base, one_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp5_base, node_vecs_lvl4[5], node_vecs_lvl4[4]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[5])
                                + vec_addrs(node_vecs_lvl4[4]),
                                writes=vec_addrs(tmp2_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp5_base, tmp5_base, node_vecs_lvl4[7], node_vecs_lvl4[6]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[7])
                                + vec_addrs(node_vecs_lvl4[6]),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp6_base, tmp5_base, tmp2_base),
                                reads=vec_addrs(tmp6_base)
                                + vec_addrs(tmp5_base)
                                + vec_addrs(tmp2_base),
                                writes=vec_addrs(tmp2_base),
                            )

                            # group2: nodes 23..26 -> tmp3_base
                            add_op(
                                "valu",
                                ("-", tmp5_base, idx_base, fifteen_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(fifteen_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp6_base, tmp5_base, two_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp6_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp5_base, tmp5_base, one_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp3_base, tmp5_base, node_vecs_lvl4[9], node_vecs_lvl4[8]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[9])
                                + vec_addrs(node_vecs_lvl4[8]),
                                writes=vec_addrs(tmp3_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp5_base, tmp5_base, node_vecs_lvl4[11], node_vecs_lvl4[10]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[11])
                                + vec_addrs(node_vecs_lvl4[10]),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp3_base, tmp6_base, tmp5_base, tmp3_base),
                                reads=vec_addrs(tmp6_base)
                                + vec_addrs(tmp5_base)
                                + vec_addrs(tmp3_base),
                                writes=vec_addrs(tmp3_base),
                            )

                            # group3: nodes 27..30 -> tmp4_base
                            add_op(
                                "valu",
                                ("-", tmp5_base, idx_base, fifteen_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(fifteen_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp6_base, tmp5_base, two_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(two_vec),
                                writes=vec_addrs(tmp6_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp5_base, tmp5_base, one_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp4_base, tmp5_base, node_vecs_lvl4[13], node_vecs_lvl4[12]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[13])
                                + vec_addrs(node_vecs_lvl4[12]),
                                writes=vec_addrs(tmp4_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp5_base, tmp5_base, node_vecs_lvl4[15], node_vecs_lvl4[14]),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(node_vecs_lvl4[15])
                                + vec_addrs(node_vecs_lvl4[14]),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp4_base, tmp6_base, tmp5_base, tmp4_base),
                                reads=vec_addrs(tmp6_base)
                                + vec_addrs(tmp5_base)
                                + vec_addrs(tmp4_base),
                                writes=vec_addrs(tmp4_base),
                            )

                            # cond2/cond3 to select among groups
                            add_op(
                                "valu",
                                ("-", tmp5_base, idx_base, fifteen_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(fifteen_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp6_base, tmp5_base, eight_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(eight_vec),
                                writes=vec_addrs(tmp6_base),
                            )
                            add_op(
                                "valu",
                                ("&", tmp5_base, tmp5_base, four_vec),
                                reads=vec_addrs(tmp5_base) + vec_addrs(four_vec),
                                writes=vec_addrs(tmp5_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp5_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp2_base, tmp5_base, tmp4_base, tmp3_base),
                                reads=vec_addrs(tmp5_base)
                                + vec_addrs(tmp4_base)
                                + vec_addrs(tmp3_base),
                                writes=vec_addrs(tmp2_base),
                            )
                            add_op(
                                "flow",
                                ("vselect", tmp1_base, tmp6_base, tmp2_base, tmp1_base),
                                reads=vec_addrs(tmp6_base)
                                + vec_addrs(tmp2_base)
                                + vec_addrs(tmp1_base),
                                writes=vec_addrs(tmp1_base),
                            )
                            xor_vec_inplace(val_base, tmp1_base)
                    else:
                        node_addr_base = node_base
                        node_val_base = tmp2_base if config.split_node_addr_val else node_base
                        # Compute addresses for node loads (store into node_base)
                        if config.node_addr_valu:
                            assert forest_values_vec is not None
                            add_op(
                                "valu",
                                ("+", node_addr_base, idx_base, forest_values_vec),
                                reads=vec_addrs(idx_base) + vec_addrs(forest_values_vec),
                                writes=vec_addrs(node_addr_base),
                            )
                        else:
                            for lane in range(V):
                                add_op(
                                    "alu",
                                    ("+", node_addr_base + lane, forest_values_p_idx, idx_base + lane),
                                    reads=[forest_values_p_idx, idx_base + lane],
                                    writes=[node_addr_base + lane],
                                )
                        # Load node values (overwrite addresses)
                        if config.interleave_load_xor and config.node_xor_alu:
                            for lane in range(V):
                                if config.split_node_addr_val:
                                    add_op(
                                        "load",
                                        ("load_offset", node_val_base, node_addr_base, lane),
                                        reads=[node_addr_base + lane],
                                        writes=[node_val_base + lane],
                                    )
                                else:
                                    add_op(
                                        "load",
                                        ("load_offset", node_addr_base, node_addr_base, lane),
                                        reads=[node_addr_base + lane],
                                        writes=[node_addr_base + lane],
                                    )
                                add_op(
                                    "alu",
                                    ("^", val_base + lane, val_base + lane, node_val_base + lane),
                                    reads=[val_base + lane, node_val_base + lane],
                                    writes=[val_base + lane],
                                )
                        else:
                            for lane in range(V):
                                if config.split_node_addr_val:
                                    add_op(
                                        "load",
                                        ("load_offset", node_val_base, node_addr_base, lane),
                                        reads=[node_addr_base + lane],
                                        writes=[node_val_base + lane],
                                    )
                                else:
                                    add_op(
                                        "load",
                                        ("load", node_addr_base + lane, node_addr_base + lane),
                                        reads=[node_addr_base + lane],
                                        writes=[node_addr_base + lane],
                                    )
                            # val = val ^ node_val
                            xor_vec_inplace(val_base, node_val_base)
    
                    # Hash stages (optimized)
                    fuse_stage_map = {0: 0, 2: 1, 4: 2}
                    hash_curr = val_base
                    ping_pong_active = config.hash_ping_pong and not config.hash_reuse_idx
                    hash_alt = tmp1_base if ping_pong_active else val_base
                    hash_tmp1 = tmp2_base
                    hash_tmp2 = tmp3_base
                    if config.hash_reuse_idx:
                        add_op(
                            "valu",
                            ("+", tmp1_base, idx_base, zero_vec),
                            reads=vec_addrs(idx_base) + vec_addrs(zero_vec),
                            writes=vec_addrs(tmp1_base),
                        )
                        hash_tmp1 = idx_base
                        hash_alt = val_base
                    for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        const1 = hash_vconsts[val1]
                        const3 = hash_vconsts[val3]
                        fuse_idx = fuse_stage_map.get(stage_idx, None)
                        can_fuse = (
                            config.hash_fusion
                            and op2 == "+"
                            and op1 == "+"
                            and op3 == "<<"
                            and fuse_idx is not None
                            and ((config.hash_fusion_mask >> fuse_idx) & 1)
                        )
                        if can_fuse:
                            mul_val = (1 + (1 << val3)) % (2**32)
                            mul_const = hash_mul_vconsts[mul_val]
                            add_op(
                                "valu",
                                ("multiply_add", hash_alt, hash_curr, mul_const, const1),
                                reads=vec_addrs(hash_curr)
                                + vec_addrs(mul_const)
                                + vec_addrs(const1),
                                writes=vec_addrs(hash_alt),
                            )
                        else:
                            add_op(
                                "valu",
                                (op1, hash_tmp1, hash_curr, const1),
                                reads=vec_addrs(hash_curr) + vec_addrs(const1),
                                writes=vec_addrs(hash_tmp1),
                            )
                            hash_shift_dest = hash_alt if ping_pong_active else hash_tmp2
                            add_op(
                                "valu",
                                (op3, hash_shift_dest, hash_curr, const3),
                                reads=vec_addrs(hash_curr) + vec_addrs(const3),
                                writes=vec_addrs(hash_shift_dest),
                            )
                            if config.hash_xor_alu and op2 == "^":
                                for lane in range(V):
                                    add_op(
                                        "alu",
                                        (
                                            "^",
                                            hash_alt + lane,
                                            hash_tmp1 + lane,
                                            hash_shift_dest + lane,
                                        ),
                                        reads=[hash_tmp1 + lane, hash_shift_dest + lane],
                                        writes=[hash_alt + lane],
                                    )
                            else:
                                add_op(
                                    "valu",
                                    (op2, hash_alt, hash_tmp1, hash_shift_dest),
                                    reads=vec_addrs(hash_tmp1) + vec_addrs(hash_shift_dest),
                                    writes=vec_addrs(hash_alt),
                                )
                        if ping_pong_active:
                            hash_curr, hash_alt = hash_alt, hash_curr
                    if config.hash_reuse_idx:
                        add_op(
                            "valu",
                            ("+", idx_base, tmp1_base, zero_vec),
                            reads=vec_addrs(tmp1_base) + vec_addrs(zero_vec),
                            writes=vec_addrs(idx_base),
                        )

                    if probe_addrs:
                        for addr in probe_addrs:
                            add_op(
                                "alu",
                                ("+", addr, addr, one_const),
                                reads=[addr, one_const],
                                writes=[addr],
                            )
    
                    if r_mod == forest_height:
                        # On the bottom level, the next idx always wraps to 0 (or j=1 if one-based)
                        if not config.idx_reset_on_r0:
                            if config.idx_one_based:
                                add_op(
                                    "flow",
                                    ("vselect", idx_base, zero_vec, one_vec, one_vec),
                                    reads=vec_addrs(zero_vec) + vec_addrs(one_vec),
                                    writes=vec_addrs(idx_base),
                                )
                            elif config.idx_wrap_xor:
                                if config.idx_wrap_xor_alu:
                                    for lane in range(V):
                                        add_op(
                                            "alu",
                                            (
                                                "^",
                                                idx_base + lane,
                                                idx_base + lane,
                                                idx_base + lane,
                                            ),
                                            reads=[idx_base + lane],
                                            writes=[idx_base + lane],
                                        )
                                else:
                                    add_op(
                                        "valu",
                                        ("^", idx_base, idx_base, idx_base),
                                        reads=vec_addrs(idx_base),
                                        writes=vec_addrs(idx_base),
                                    )
                            else:
                                add_op(
                                    "flow",
                                    ("vselect", idx_base, zero_vec, zero_vec, zero_vec),
                                    reads=vec_addrs(zero_vec),
                                    writes=vec_addrs(idx_base),
                                )
                    else:
                        # idx = idx * 2 + 1 + (val & 1)  (compute parity per-lane with ALU)
                        if config.use_vector_parity:
                            add_op(
                                "valu",
                                ("&", tmp1_base, val_base, one_vec),
                                reads=vec_addrs(val_base) + vec_addrs(one_vec),
                                writes=vec_addrs(tmp1_base),
                            )
                        else:
                            for lane in range(V):
                                add_op(
                                    "alu",
                                    ("&", tmp1_base + lane, val_base + lane, one_const),
                                    reads=[val_base + lane, one_const],
                                    writes=[tmp1_base + lane],
                                )
                        if config.idx_update_scalar or config.idx_update_alu:
                            idx_src_scalar = None
                            if config.idx_reset_on_r0 and r_mod == 0:
                                idx_src_scalar = one_const if config.idx_one_based else zero_const
                            for lane in range(V):
                                idx_src = (
                                    idx_base + lane
                                    if idx_src_scalar is None
                                    else idx_src_scalar
                                )
                                add_op(
                                    "alu",
                                    ("<<", tmp2_base + lane, idx_src, one_const),
                                    reads=[idx_src, one_const],
                                    writes=[tmp2_base + lane],
                                )
                                if not config.idx_one_based:
                                    add_op(
                                        "alu",
                                        ("+", tmp2_base + lane, tmp2_base + lane, one_const),
                                        reads=[tmp2_base + lane, one_const],
                                        writes=[tmp2_base + lane],
                                    )
                                add_op(
                                    "alu",
                                    ("+", idx_base + lane, tmp2_base + lane, tmp1_base + lane),
                                    reads=[tmp2_base + lane, tmp1_base + lane],
                                    writes=[idx_base + lane],
                                )
                        else:
                            if config.idx_one_based:
                                # j_next = 2*j + parity
                                idx_src_vec = (
                                    one_vec
                                    if (config.idx_reset_on_r0 and r_mod == 0)
                                    else idx_base
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        idx_base,
                                        idx_src_vec,
                                        two_vec,
                                        tmp1_base,
                                    ),
                                    reads=vec_addrs(idx_src_vec)
                                    + vec_addrs(two_vec)
                                    + vec_addrs(tmp1_base),
                                    writes=vec_addrs(idx_base),
                                )
                            else:
                                idx_src_vec = (
                                    zero_vec
                                    if (config.idx_reset_on_r0 and r_mod == 0)
                                    else idx_base
                                )
                                if config.idx_update_add_one_last:
                                    # Use flow vselect to form the +1/+2 increment vector.
                                    # This removes one VALU '+' per update at the cost of a FLOW op.
                                    # inc = (parity ? 2 : 1)
                                    add_op(
                                        "flow",
                                        ("vselect", tmp1_base, tmp1_base, two_vec, one_vec),
                                        reads=vec_addrs(tmp1_base)
                                        + vec_addrs(two_vec)
                                        + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp1_base),
                                    )
                                    # idx = 2*idx + inc
                                    add_op(
                                        "valu",
                                        (
                                            "multiply_add",
                                            idx_base,
                                            idx_src_vec,
                                            two_vec,
                                            tmp1_base,
                                        ),
                                        reads=vec_addrs(idx_src_vec)
                                        + vec_addrs(two_vec)
                                        + vec_addrs(tmp1_base),
                                        writes=vec_addrs(idx_base),
                                    )
                                else:
                                    add_op(
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp2_base,
                                            idx_src_vec,
                                            two_vec,
                                            one_vec,
                                        ),
                                        reads=vec_addrs(idx_src_vec)
                                        + vec_addrs(two_vec)
                                        + vec_addrs(one_vec),
                                        writes=vec_addrs(tmp2_base),
                                    )
                                    if config.idx_update_add_alu:
                                        for lane in range(V):
                                            add_op(
                                                "alu",
                                                (
                                                    "+",
                                                    idx_base + lane,
                                                    tmp2_base + lane,
                                                    tmp1_base + lane,
                                                ),
                                                reads=[tmp2_base + lane, tmp1_base + lane],
                                                writes=[idx_base + lane],
                                            )
                                    else:
                                        add_op(
                                            "valu",
                                            ("+", idx_base, tmp2_base, tmp1_base),
                                            reads=vec_addrs(tmp2_base)
                                            + vec_addrs(tmp1_base),
                                            writes=vec_addrs(idx_base),
                                        )
            # Final stores
            for c in range(block_chunks):
                ptr = idx_ptrs[block_start + c]
                if config.idx_one_based:
                    add_op(
                        "valu",
                        ("-", tmp2_bases[c], idx_bases[c], one_vec),
                        reads=vec_addrs(idx_bases[c]) + vec_addrs(one_vec),
                        writes=vec_addrs(tmp2_bases[c]),
                    )
                    add_op(
                        "store",
                        ("vstore", ptr, tmp2_bases[c]),
                        reads=[ptr] + vec_addrs(tmp2_bases[c]),
                        writes=[],
                    )
                else:
                    add_op(
                        "store",
                        ("vstore", ptr, idx_bases[c]),
                        reads=[ptr] + vec_addrs(idx_bases[c]),
                        writes=[],
                    )
                val_addr_tmp = (
                    val_addr_tmps_store[c % len(val_addr_tmps_store)]
                    if val_addr_tmps_store
                    else temp_addr
                )
                add_imm_op(val_addr_tmp, ptr, batch_size)
                add_op(
                    "store",
                    ("vstore", val_addr_tmp, val_bases[c]),
                    reads=[val_addr_tmp] + vec_addrs(val_bases[c]),
                    writes=[],
                )

        if num_chunks:
            for block_start in range(0, num_chunks, block_size):
                block_chunks = min(block_size, num_chunks - block_start)
                emit_block(block_start, block_chunks)

        # Tail elements (scalar) if needed
        if tail:
            tmp_idx = self.alloc_scratch("tail_idx")
            tmp_val = self.alloc_scratch("tail_val")
            tmp_node_val = self.alloc_scratch("tail_node_val")
            tmp_addr = self.alloc_scratch("tail_addr")
            tmp1 = self.alloc_scratch("tail_tmp1")
            tmp2 = self.alloc_scratch("tail_tmp2")
            zero_const = const(0)
            one_const = const(1)
            two_const = const(2)
            n_nodes_plus1_const = const(n_nodes + 1) if config.idx_one_based else None

            for r in range(rounds):
                for i in range(num_chunks * V, batch_size):
                    i_const = const(i)
                    add_op(
                        "alu",
                        ("+", tmp_addr, inp_indices_p, i_const),
                        reads=[inp_indices_p, i_const],
                        writes=[tmp_addr],
                    )
                    add_op(
                        "load",
                        ("load", tmp_idx, tmp_addr),
                        reads=[tmp_addr],
                        writes=[tmp_idx],
                    )
                    if config.idx_one_based:
                        add_op(
                            "alu",
                            ("+", tmp_idx, tmp_idx, one_const),
                            reads=[tmp_idx, one_const],
                            writes=[tmp_idx],
                        )
                    add_op(
                        "alu",
                        ("+", tmp_addr, inp_values_p, i_const),
                        reads=[inp_values_p, i_const],
                        writes=[tmp_addr],
                    )
                    add_op(
                        "load",
                        ("load", tmp_val, tmp_addr),
                        reads=[tmp_addr],
                        writes=[tmp_val],
                    )
                    add_op(
                        "alu",
                        ("+", tmp_addr, forest_values_p_idx, tmp_idx),
                        reads=[forest_values_p_idx, tmp_idx],
                        writes=[tmp_addr],
                    )
                    add_op(
                        "load",
                        ("load", tmp_node_val, tmp_addr),
                        reads=[tmp_addr],
                        writes=[tmp_node_val],
                    )
                    add_op(
                        "alu",
                        ("^", tmp_val, tmp_val, tmp_node_val),
                        reads=[tmp_val, tmp_node_val],
                        writes=[tmp_val],
                    )
                    for op1, val1, op2, op3, val3 in HASH_STAGES:
                        c1 = const(val1)
                        c3 = const(val3)
                        add_op(
                            "alu",
                            (op1, tmp1, tmp_val, c1),
                            reads=[tmp_val, c1],
                            writes=[tmp1],
                        )
                        add_op(
                            "alu",
                            (op3, tmp2, tmp_val, c3),
                            reads=[tmp_val, c3],
                            writes=[tmp2],
                        )
                        add_op(
                            "alu",
                            (op2, tmp_val, tmp1, tmp2),
                            reads=[tmp1, tmp2],
                            writes=[tmp_val],
                        )
                    add_op(
                        "alu",
                        ("&", tmp1, tmp_val, one_const),
                        reads=[tmp_val, one_const],
                        writes=[tmp1],
                    )
                    add_op(
                        "alu",
                        ("*", tmp2, tmp_idx, two_const),
                        reads=[tmp_idx, two_const],
                        writes=[tmp2],
                    )
                    if not config.idx_one_based:
                        add_op(
                            "alu",
                            ("+", tmp2, tmp2, one_const),
                            reads=[tmp2, one_const],
                            writes=[tmp2],
                        )
                    add_op(
                        "alu",
                        ("+", tmp_idx, tmp2, tmp1),
                        reads=[tmp2, tmp1],
                        writes=[tmp_idx],
                    )
                    if config.idx_one_based:
                        assert n_nodes_plus1_const is not None
                        add_op(
                            "alu",
                            ("<", tmp1, tmp_idx, n_nodes_plus1_const),
                            reads=[tmp_idx, n_nodes_plus1_const],
                            writes=[tmp1],
                        )
                        add_op(
                            "alu",
                            ("*", tmp2, tmp_idx, tmp1),
                            reads=[tmp_idx, tmp1],
                            writes=[tmp2],
                        )
                        add_op(
                            "alu",
                            ("-", tmp1, one_const, tmp1),
                            reads=[one_const, tmp1],
                            writes=[tmp1],
                        )
                        add_op(
                            "alu",
                            ("+", tmp_idx, tmp2, tmp1),
                            reads=[tmp2, tmp1],
                            writes=[tmp_idx],
                        )
                    else:
                        add_op(
                            "alu",
                            ("<", tmp1, tmp_idx, n_nodes_const),
                            reads=[tmp_idx, n_nodes_const],
                            writes=[tmp1],
                        )
                        add_op(
                            "alu",
                            ("*", tmp_idx, tmp_idx, tmp1),
                            reads=[tmp_idx, tmp1],
                            writes=[tmp_idx],
                        )
                    add_op(
                        "alu",
                        ("+", tmp_addr, inp_indices_p, i_const),
                        reads=[inp_indices_p, i_const],
                        writes=[tmp_addr],
                    )
                    if config.idx_one_based:
                        add_op(
                            "alu",
                            ("-", tmp2, tmp_idx, one_const),
                            reads=[tmp_idx, one_const],
                            writes=[tmp2],
                        )
                        add_op(
                            "store",
                            ("store", tmp_addr, tmp2),
                            reads=[tmp_addr, tmp2],
                            writes=[],
                        )
                    else:
                        add_op(
                            "store",
                            ("store", tmp_addr, tmp_idx),
                            reads=[tmp_addr, tmp_idx],
                            writes=[],
                        )
                    add_op(
                        "alu",
                        ("+", tmp_addr, inp_values_p, i_const),
                        reads=[inp_values_p, i_const],
                        writes=[tmp_addr],
                    )
                    add_op(
                        "store",
                        ("store", tmp_addr, tmp_val),
                        reads=[tmp_addr, tmp_val],
                        writes=[],
                    )

        dummy_dst = self.alloc_scratch("dummy_dst")

        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"

        if config.debug_dump_valu_subs and op_tags is not None:
            counts = {}
            for op, tag in zip(ops, op_tags):
                if op.engine == "valu" and op.slot[0] == "-":
                    counts[tag] = counts.get(tag, 0) + 1
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            print("VALU '-' counts by tag (top):")
            for tag, count in items[: config.debug_dump_valu_subs_top]:
                print(f"{count}\t{tag}")

        # Compute critical path lengths (ignoring 0-latency edges for priority heuristics)
        for op_id in range(len(ops) - 1, -1, -1):
            if ops[op_id].users:
                ops[op_id].height = 1 + max(ops[u].height for (u, _lat) in ops[op_id].users)

        weights = {
            "alu": 1 / SLOT_LIMITS["alu"],
            "valu": 1 / SLOT_LIMITS["valu"],
            "load": 1 / SLOT_LIMITS["load"],
            "store": 1 / SLOT_LIMITS["store"],
            "flow": 1 / SLOT_LIMITS["flow"],
        }
        res_height = [weights[op.engine] for op in ops]
        for op_id in range(len(ops) - 1, -1, -1):
            if ops[op_id].users:
                res_height[op_id] = weights[ops[op_id].engine] + max(
                    res_height[u] for (u, _lat) in ops[op_id].users
                )

        INF = 10**9
        load_dist = [INF for _ in ops]
        for op_id in range(len(ops) - 1, -1, -1):
            op = ops[op_id]
            if op.engine == "load":
                load_dist[op_id] = 0
            for u, _lat in op.users:
                cand = load_dist[u] + 1
                if cand < load_dist[op_id]:
                    load_dist[op_id] = cand

        LOAD_DIST_CAP = config.load_dist_cap
        capped_ld = [
            (ld if ld < INF else INF) if ld <= LOAD_DIST_CAP else LOAD_DIST_CAP
            for ld in load_dist
        ]
        next_use = None
        if config.scheduler_pressure_tiebreak:
            next_use = [INF for _ in ops]
            next_access: dict[int, int] = {}
            for op_id in range(len(ops) - 1, -1, -1):
                op = ops[op_id]
                min_next = INF
                for addr in op.writes:
                    nxt = next_access.get(addr)
                    if nxt is not None:
                        dist = nxt - op_id
                        if dist < min_next:
                            min_next = dist
                next_use[op_id] = min_next
                for addr in op.reads:
                    next_access[addr] = op_id
                for addr in op.writes:
                    next_access[addr] = op_id

        # List scheduling with resource limits and mixed latencies.
        # RAW/WAW use 1-cycle latency (writes are visible next cycle), while WAR is 0-cycle
        # due to the machine's read-before-write semantics.
        remaining = [len(op.deps) for op in ops]
        earliest = [0 for _ in ops]
        scheduled = [False for _ in ops]

        engines = ["alu", "valu", "load", "store", "flow"]
        ready_heaps = {e: [] for e in engines}
        future_heaps = {e: [] for e in engines}

        tie_keys = None
        if config.scheduler_tiebreak == "random":
            rng = random.Random(config.scheduler_tie_seed)
            tie_keys = [rng.getrandbits(32) for _ in ops]

        def priority(op_id: int):
            mode = config.scheduler_mode
            eng = ops[op_id].engine
            height_first = False
            if mode == "height_first_all":
                height_first = True
            elif mode == "height_first_nonload":
                height_first = eng != "load"
            elif mode == "height_first_flow":
                height_first = eng == "flow"
            elif mode == "height_first_valu":
                height_first = eng == "valu"
            elif mode == "height_first_alu":
                height_first = eng == "alu"
            elif mode == "height_first_load":
                height_first = eng == "load"
            if mode == "res_weighted":
                base = (-res_height[op_id], capped_ld[op_id])
            elif height_first:
                base = (-ops[op_id].height, capped_ld[op_id])
            else:
                base = (capped_ld[op_id], -ops[op_id].height)
            load_bias = 0 if eng == "load" else 1
            if config.scheduler_tiebreak == "op_id":
                tie = op_id
            elif config.scheduler_tiebreak == "reverse_op_id":
                tie = -op_id
            elif config.scheduler_tiebreak == "random":
                assert tie_keys is not None
                tie = tie_keys[op_id]
            else:
                tie = ((op_id * 2654435761) ^ config.scheduler_tie_seed) & 0xFFFFFFFF
            extra = (next_use[op_id],) if config.scheduler_pressure_tiebreak else ()
            if config.scheduler_load_bias:
                return base + extra + (load_bias, tie, op_id)
            return base + extra + (tie, op_id)

        for op_id, op in enumerate(ops):
            if remaining[op_id] == 0:
                heapq.heappush(ready_heaps[op.engine], priority(op_id))

        cycle = 0
        scheduled_count = 0
        instrs = []

        while scheduled_count < len(ops):
            # Move ops that have become ready for this cycle
            for eng in engines:
                fut = future_heaps[eng]
                while fut and fut[0][0] <= cycle:
                    ec, *prio = heapq.heappop(fut)
                    op_id = prio[-1]
                    if scheduled[op_id] or remaining[op_id] != 0:
                        continue
                    if earliest[op_id] > ec:
                        heapq.heappush(fut, (earliest[op_id], *prio))
                        continue
                    heapq.heappush(ready_heaps[eng], tuple(prio))

            bundle = {}
            avail = {e: SLOT_LIMITS[e] for e in engines}

            # Allow 0-latency edges to unlock additional same-cycle scheduling.
            made_progress = True
            while made_progress:
                made_progress = False
                for eng in engines:
                    if avail[eng] <= 0:
                        continue
                    heap = ready_heaps[eng]
                    while heap and avail[eng] > 0:
                        *_, op_id = heapq.heappop(heap)
                        if scheduled[op_id] or remaining[op_id] != 0:
                            continue
                        if earliest[op_id] > cycle:
                            heapq.heappush(
                                future_heaps[eng],
                                (earliest[op_id],) + priority(op_id),
                            )
                            continue

                        scheduled[op_id] = True
                        scheduled_count += 1
                        bundle.setdefault(eng, []).append(ops[op_id].slot)
                        avail[eng] -= 1
                        made_progress = True

                        for u, lat in ops[op_id].users:
                            if remaining[u] <= 0:
                                continue
                            remaining[u] -= 1
                            ne = cycle + lat
                            if earliest[u] < ne:
                                earliest[u] = ne
                            if remaining[u] == 0:
                                ueng = ops[u].engine
                                if earliest[u] <= cycle:
                                    heapq.heappush(ready_heaps[ueng], priority(u))
                                else:
                                    heapq.heappush(
                                        future_heaps[ueng],
                                        (earliest[u],) + priority(u),
                                    )

            if not bundle:
                # Should be rare; insert a harmless alu op to advance time
                bundle = {"alu": [("+", dummy_dst, dummy_dst, dummy_dst)]}
                instrs.append(bundle)
                cycle += 1
                continue

            instrs.append(bundle)
            cycle += 1
        self.instrs = instrs

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
