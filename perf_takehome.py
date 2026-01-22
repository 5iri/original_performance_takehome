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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self._const_pending = []

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
            # Record constant to be emitted later in grouped loads
            self.const_map[val] = addr
            self._const_pending.append((addr, val))
        return self.const_map[val]

    def emit_const_inits(self):
        """Emit pending constant loads in grouped load slots (2 per cycle).

        Call this before any instructions that depend on the constant
        scratch addresses (e.g., `vbroadcast`). Grouping reduces number of
        load instructions produced by `scratch_const` calls.
        """
        if not self._const_pending:
            return
        load_slots = []
        for addr, val in self._const_pending:
            load_slots.append(("const", addr, val))
            if len(load_slots) == 2:
                self.emit({"load": load_slots})
                load_slots = []
        if load_slots:
            self.emit({"load": load_slots})
        self._const_pending = []

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def emit(self, instr):
        """Add a packed instruction bundle"""
        self.instrs.append(instr)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized + packed VLIW + unrolled with bitwise ops.
        Overlap scalar ALU (12 slots) with valu (6 slots) operations.
        """
        UNROLL = 16
        
        tmp1 = self.alloc_scratch("tmp1")
        
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        addr_tmps = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(len(init_vars))]
        for i, v in enumerate(init_vars):
            self.emit({"load": [("const", addr_tmps[i], i)]})
        for i in range(0, len(init_vars), 2):
            loads = [("load", self.scratch[init_vars[i]], addr_tmps[i])]
            if i + 1 < len(init_vars):
                loads.append(("load", self.scratch[init_vars[i+1]], addr_tmps[i+1]))
            self.emit({"load": loads})

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Emit pending constant loads before using them in vbroadcast
        self.emit_const_inits()

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        
        self.emit({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
        ]})

        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            vc1 = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
            vc3 = self.alloc_scratch(f"v_hc3_{hi}", VLEN)
            v_hash_consts.append((vc1, vc3, c1, c3))
        # Emit hash-related constants before broadcasting them
        self.emit_const_inits()
        for i in range(0, len(v_hash_consts), 3):
            valu_slots = []
            for j in range(3):
                if i + j < len(v_hash_consts):
                    vc1, vc3, c1, c3 = v_hash_consts[i + j]
                    valu_slots.append(("vbroadcast", vc1, c1))
                    valu_slots.append(("vbroadcast", vc3, c3))
            self.emit({"valu": valu_slots})

        self.emit({"flow": [("pause",)]})

        v_idx = [self.alloc_scratch(f"v_idx_{u}", VLEN) for u in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val_{u}", VLEN) for u in range(UNROLL)]
        v_node = [self.alloc_scratch(f"v_node_{u}", VLEN) for u in range(UNROLL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", VLEN) for u in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", VLEN) for u in range(UNROLL)]
        addr_scratch = [self.alloc_scratch(f"addr_{u}", VLEN) for u in range(UNROLL)]
        idx_base = [self.alloc_scratch(f"idx_base_{u}") for u in range(UNROLL)]
        val_base = [self.alloc_scratch(f"val_base_{u}") for u in range(UNROLL)]

        loop_counter = self.alloc_scratch("loop_counter")
        batch_offset = self.alloc_scratch("batch_offset")
        batch_size_const = self.scratch_const(batch_size)
        total_iters = self.scratch_const(rounds * (batch_size // VLEN) // UNROLL)
        stride_const = self.scratch_const(VLEN * UNROLL)

        # Emit loop-related constants so subsequent ALU/flow ops read valid values
        self.emit_const_inits()

        # Create vector offsets for base calculation
        v_offset_consts = self.alloc_scratch("v_offset_consts", 16)
        v_offsets_0 = v_offset_consts
        v_offsets_8 = v_offset_consts + 8
        
        # Initialize offset constants
        load_slots = []
        for i in range(16):
            load_slots.append(("const", v_offset_consts + i, i * VLEN))
            if len(load_slots) == 2:
                self.emit({"load": load_slots})
                load_slots = []
        if load_slots:
            self.emit({"load": load_slots})

        # Initialize running pointers and loop counter
        self.emit({"alu": [
            ("+", idx_base[0], self.scratch["inp_indices_p"], zero_const),
            ("+", val_base[0], self.scratch["inp_values_p"], zero_const),
        ]})
        self.emit({"load": [("const", loop_counter, 0), ("const", batch_offset, 0)]})

        loop_start = len(self.instrs)

        # Calculate base addresses: idx_base[0..15] = idx_base[0] + offsets (2 cycles using VALU)
        # Cycle 1: Broadcast base[0]
        v_tmp_base_idx = self.alloc_scratch("v_tmp_base_idx", VLEN)
        v_tmp_base_val = self.alloc_scratch("v_tmp_base_val", VLEN)
        self.emit({"valu": [
            ("vbroadcast", v_tmp_base_idx, idx_base[0]),
            ("vbroadcast", v_tmp_base_val, val_base[0]),
        ]})
        
        # Cycle 2: Add offsets
        self.emit({"valu": [
            ("+", idx_base[0], v_tmp_base_idx, v_offsets_0),
            ("+", val_base[0], v_tmp_base_val, v_offsets_0),
            ("+", idx_base[8], v_tmp_base_idx, v_offsets_8),
            ("+", val_base[8], v_tmp_base_val, v_offsets_8),
        ]})

        # Load indices/values
        for u in range(UNROLL):
            self.emit({"load": [("vload", v_idx[u], idx_base[u]), ("vload", v_val[u], val_base[u])]})

        # Compute scatter addresses
        for u in range(0, UNROLL, 6):
            valu_slots = [("+", addr_scratch[u+i], v_forest_p, v_idx[u+i]) for i in range(min(6, UNROLL - u))]
            self.emit({"valu": valu_slots})

        # Scatter loads
        for i in range(VLEN):
            for u in range(0, UNROLL, 2):
                loads = [("load_offset", v_node[u], addr_scratch[u], i)]
                if u + 1 < UNROLL:
                    loads.append(("load_offset", v_node[u+1], addr_scratch[u+1], i))
                self.emit({"load": loads})

        # XOR
        for u in range(0, UNROLL, 6):
            valu_slots = [("^", v_val[u+i], v_val[u+i], v_node[u+i]) for i in range(min(6, UNROLL - u))]
            self.emit({"valu": valu_slots})

        # Hash
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1, vc3, _, _ = v_hash_consts[hi]
            for u in range(0, UNROLL, 3):
                valu_slots = []
                for i in range(min(3, UNROLL - u)):
                    valu_slots.append((op1, v_tmp1[u+i], v_val[u+i], vc1))
                    valu_slots.append((op3, v_tmp2[u+i], v_val[u+i], vc3))
                self.emit({"valu": valu_slots})
            for u in range(0, UNROLL, 6):
                valu_slots = [(op2, v_val[u+i], v_tmp1[u+i], v_tmp2[u+i]) for i in range(min(6, UNROLL - u))]
                self.emit({"valu": valu_slots})

        # idx = 2*idx + (1 + (val & 1))
        for u in range(0, UNROLL, 3):
            valu_slots = []
            for i in range(min(3, UNROLL - u)):
                valu_slots.append(("&", v_tmp1[u+i], v_val[u+i], v_one))
                valu_slots.append(("*", v_idx[u+i], v_idx[u+i], v_two))
            self.emit({"valu": valu_slots})
        
        for u in range(0, UNROLL, 6):
            valu_slots = [("+", v_tmp1[u+i], v_tmp1[u+i], v_one) for i in range(min(6, UNROLL - u))]
            self.emit({"valu": valu_slots})
        
        for u in range(0, UNROLL, 6):
            valu_slots = [("+", v_idx[u+i], v_idx[u+i], v_tmp1[u+i]) for i in range(min(6, UNROLL - u))]
            self.emit({"valu": valu_slots})

        # idx = idx & mask where mask = -(idx < n_nodes)
        for u in range(0, UNROLL, 6):
            valu_slots = [("<", v_tmp1[u+i], v_idx[u+i], v_n_nodes) for i in range(min(6, UNROLL - u))]
            self.emit({"valu": valu_slots})
        
        for u in range(0, UNROLL, 6):
            valu_slots = [("-", v_tmp1[u+i], v_zero, v_tmp1[u+i]) for i in range(min(6, UNROLL - u))]
            self.emit({"valu": valu_slots})
        
        for u in range(0, UNROLL, 6):
            valu_slots = [("&", v_idx[u+i], v_idx[u+i], v_tmp1[u+i]) for i in range(min(6, UNROLL - u))]
            self.emit({"valu": valu_slots})

        # Store with loop control overlapped
        for u in range(UNROLL - 1):
            self.emit({"store": [("vstore", idx_base[u], v_idx[u]), ("vstore", val_base[u], v_val[u])]})
        
        # Last store with loop control
        self.emit({
            "store": [("vstore", idx_base[UNROLL-1], v_idx[UNROLL-1]), ("vstore", val_base[UNROLL-1], v_val[UNROLL-1])],
            "alu": [("+", batch_offset, batch_offset, stride_const), ("+", loop_counter, loop_counter, one_const)],
        })
        
        self.emit({"alu": [("%", batch_offset, batch_offset, batch_size_const), ("<", tmp1, loop_counter, total_iters)]})
        self.emit({
            "flow": [("cond_jump", tmp1, loop_start)],
            "alu": [
                ("+", idx_base[0], self.scratch["inp_indices_p"], batch_offset),
                ("+", val_base[0], self.scratch["inp_values_p"], batch_offset)
            ]
        })
        self.emit({"flow": [("pause",)]})

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
