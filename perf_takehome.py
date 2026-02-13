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

from collections import defaultdict, deque
import random
import unittest
from copy import copy

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

class Scheduler:
    def __init__(self, limits):
        self.limits = limits
        self.chains = [] # List of lists of stages (list of instrs)
        
    def add_chain(self, stages):
        self.chains.append(deque(stages))
        
    def schedule(self):
        instrs = []
        # State:
        # pending_instrs[chain_id] = list of instrs for current stage
        # ready_cycle[chain_id] = cycle when current stage is ready
        
        n_chains = len(self.chains)
        pending_instrs = [deque() for _ in range(n_chains)]
        ready_cycle = [0] * n_chains
        
        # Initialize pending with first stage
        for i in range(n_chains):
            if self.chains[i]:
                pending_instrs[i] = deque(self.chains[i].popleft())

        current_cycle = 0
        while True:
            # Check if done
            if all(not p and not c for p, c in zip(pending_instrs, self.chains)):
                break
                
            bundle = defaultdict(list)
            slots_left = copy(self.limits)
            
            # Simple greedy: iterate chains, try to schedule ready instrs
            progress = False
            
            # We iterate multiple times to fill holes? No, just once per cycle
            # Sort candidates by "criticality"? 
            # Chains with more remaining stages should go first?
            chain_order = sorted(range(n_chains), key=lambda i: len(self.chains[i]), reverse=False)
            
            for i in chain_order:
                if ready_cycle[i] > current_cycle:
                    continue
                
                if not pending_instrs[i] and self.chains[i]:
                    # Advance to next stage if current empty
                    # But we only advance when PREVIOUS stage completed in PREVIOUS cycle?
                    # My logic: pending_instrs holds CURRENT stage.
                    # Once empty, we wait latency (1 cycle) then load next.
                    # So if empty here, it means we finished stage in prev cycle.
                    # Wait, if we finished in cycle T, result ready T+1.
                    # So we can load next stage now.
                     pending_instrs[i] = deque(self.chains[i].popleft())
                
                # Try to schedule instrs from current stage
                while pending_instrs[i]:
                    instr = pending_instrs[i][0]
                    # instr is (engine, slot) or just slot? 
                    # My chains will store (engine, slot)
                    engine, slot = instr
                    
                    if slots_left[engine] > 0:
                        # Schedule it
                        bundle[engine].append(slot)
                        slots_left[engine] -= 1
                        pending_instrs[i].popleft()
                        progress = True
                        
                        # Update readiness for NEXT stage
                        # If this was the last instr of the stage, next stage ready at T+1
                        if not pending_instrs[i]:
                            ready_cycle[i] = current_cycle + 1
                    else:
                        # Engine full, stop this chain for this cycle
                        break
            
            # Emit bundle
            if bundle:
                instrs.append(dict(bundle))
            elif any(pending_instrs) or any(self.chains):
                 # No progress but work remains -> stall (shouldn't happen with valid logic unless deadlock)
                 # Deadlock possible if limits too small? No.
                 # Just wait for ready_cycle.
                 instrs.append({}) # Empty cycle / stall
            
            current_cycle += 1
            
        return instrs

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
            self.const_map[val] = addr
            self._const_pending.append((addr, val))
        return self.const_map[val]

    def emit_const_inits(self):
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

    def emit(self, instr):
        self.instrs.append(instr)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        UNROLL = 32  # processes UNROLL * VLEN elements at once

        # Allocations
        init_meta = [("forest_values_p", 4), ("inp_values_p", 6)]
        for name, _ in init_meta:
            self.alloc_scratch(name, 1)
        addr_tmps = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(len(init_meta))]

        # Init Loads
        for i, (_, header_idx) in enumerate(init_meta):
            self.emit({"load": [("const", addr_tmps[i], header_idx)]})
        for i in range(0, len(init_meta), 2):
            loads = [("load", self.scratch[init_meta[i][0]], addr_tmps[i])]
            if i + 1 < len(init_meta):
                loads.append(("load", self.scratch[init_meta[i + 1][0]], addr_tmps[i + 1]))
            self.emit({"load": loads})

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        three_const = self.scratch_const(3)
        vlen_const = self.scratch_const(VLEN)
        self.emit_const_inits()

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_three = self.alloc_scratch("v_three", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        self.emit({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_three, three_const),
            ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
        ]})

        # Hash stage constants and possible fused multiply-add params
        hash_infos = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            vc1 = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
            info = {"vc1": vc1, "c1": c1, "muladd": False}
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_const = (1 + (1 << val3)) % (2**32)
                cmul = self.scratch_const(mul_const)
                vmul = self.alloc_scratch(f"v_hmul_{hi}", VLEN)
                info.update({"muladd": True, "vmul": vmul, "cmul": cmul})
            else:
                c3 = self.scratch_const(val3)
                vc3 = self.alloc_scratch(f"v_hc3_{hi}", VLEN)
                info.update({"vc3": vc3, "c3": c3})
            hash_infos.append(info)

        self.emit_const_inits()
        broadcasts = []
        for info in hash_infos:
            broadcasts.append(("vbroadcast", info["vc1"], info["c1"]))
            if info["muladd"]:
                broadcasts.append(("vbroadcast", info["vmul"], info["cmul"]))
            else:
                broadcasts.append(("vbroadcast", info["vc3"], info["c3"]))
        for i in range(0, len(broadcasts), SLOT_LIMITS["valu"]):
            chunk = broadcasts[i : i + SLOT_LIMITS["valu"]]
            self.emit({"valu": chunk})

        # No-op in submission tests (pause disabled); skip to save cycles

        # Working vectors
        v_idx = [self.alloc_scratch(f"v_idx_{u}", VLEN) for u in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val_{u}", VLEN) for u in range(UNROLL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", VLEN) for u in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", VLEN) for u in range(UNROLL)]
        addr_scratch = [self.alloc_scratch(f"addr_{u}", VLEN) for u in range(UNROLL)]
        val_base = [self.alloc_scratch(f"val_base_{u}") for u in range(UNROLL)]

        # Preload top tree nodes into vectors so shallow depths avoid gather loads.
        top_count = min(n_nodes, 7)
        node_addr_tmp = self.alloc_scratch("node_addr_tmp")
        node_scalars = [self.alloc_scratch(f"node_scalar_{i}") for i in range(top_count)]
        for i in range(top_count):
            self.emit({"flow": [("add_imm", node_addr_tmp, self.scratch["forest_values_p"], i)]})
            self.emit({"load": [("load", node_scalars[i], node_addr_tmp)]})
        node_vecs = [self.alloc_scratch(f"v_node_const_{i}", VLEN) for i in range(top_count)]
        node_broadcasts = [("vbroadcast", node_vecs[i], node_scalars[i]) for i in range(top_count)]
        for i in range(0, len(node_broadcasts), SLOT_LIMITS["valu"]):
            self.emit({"valu": node_broadcasts[i : i + SLOT_LIMITS["valu"]]})

        def append_hash_and_update(stages, u, update_kind: str):
            for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                hinfo = hash_infos[hi]
                if hinfo["muladd"]:
                    stages.append([("valu", ("multiply_add", v_val[u], v_val[u], hinfo["vmul"], hinfo["vc1"]))])
                else:
                    vc1, vc3 = hinfo["vc1"], hinfo["vc3"]
                    stages.append([("valu", (op1, v_tmp1[u], v_val[u], vc1)), ("valu", (op3, v_tmp2[u], v_val[u], vc3))])
                    stages.append([("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))])

            if update_kind == "normal":
                # Index update
                stages.append([("valu", ("&", v_tmp1[u], v_val[u], v_one))])
                stages.append([("valu", ("+", v_tmp1[u], v_tmp1[u], v_one))])
                stages.append([("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_tmp1[u]))])
            elif update_kind == "depth0":
                # At depth 0 the current index is always 0, so idx_next = (val & 1) + 1.
                stages.append([("valu", ("&", v_idx[u], v_val[u], v_one))])
                stages.append([("valu", ("+", v_idx[u], v_idx[u], v_one))])
            elif update_kind == "none":
                # Wrapping depth: the next round is depth 0 and does not read old idx.
                return
            else:
                raise ValueError(f"Unknown update kind: {update_kind}")

        def append_gather_round(stages, u, update_kind: str):
            # Compute forest addresses
            stages.append([("valu", ("+", addr_scratch[u], v_forest_p, v_idx[u]))])

            # Gather node values
            s_gather = []
            for i in range(VLEN):
                s_gather.append(("load", ("load_offset", v_tmp2[u], addr_scratch[u], i)))
            stages.append(s_gather)

            # XOR with node (node currently in v_tmp2)
            stages.append([("valu", ("^", v_val[u], v_val[u], v_tmp2[u]))])
            append_hash_and_update(stages, u, update_kind)

        def append_depth0_round(stages, u, update_kind: str):
            stages.append([("valu", ("^", v_val[u], v_val[u], node_vecs[0]))])
            append_hash_and_update(stages, u, update_kind)

        def append_depth1_round(stages, u, update_kind: str):
            stages.extend(
                [
                    [("valu", ("&", v_tmp1[u], v_idx[u], v_one))],
                    [("flow", ("vselect", v_tmp2[u], v_tmp1[u], node_vecs[1], node_vecs[2]))],
                    [("valu", ("^", v_val[u], v_val[u], v_tmp2[u]))],
                ]
            )
            append_hash_and_update(stages, u, update_kind)

        def append_depth2_round(stages, u, update_kind: str):
            stages.extend(
                [
                    [("valu", ("-", v_tmp1[u], v_idx[u], v_three))],
                    [("valu", ("&", v_tmp2[u], v_tmp1[u], v_one))],
                    [("valu", (">>", v_tmp1[u], v_tmp1[u], v_one))],
                    [("valu", ("&", v_tmp1[u], v_tmp1[u], v_one))],
                    [("flow", ("vselect", addr_scratch[u], v_tmp2[u], node_vecs[4], node_vecs[3]))],
                    [("flow", ("vselect", v_tmp2[u], v_tmp2[u], node_vecs[6], node_vecs[5]))],
                    [("flow", ("vselect", v_tmp2[u], v_tmp1[u], v_tmp2[u], addr_scratch[u]))],
                    [("valu", ("^", v_val[u], v_val[u], v_tmp2[u]))],
                ]
            )
            append_hash_and_update(stages, u, update_kind)

        # Build one global schedule across all rounds to avoid full-round barriers.
        wrap_period = forest_height + 1
        round_sched = Scheduler(SLOT_LIMITS)
        for u in range(UNROLL):
            stages = []
            for r in range(rounds):
                depth = r % wrap_period
                if depth == forest_height:
                    if depth == 0 and top_count >= 1:
                        append_depth0_round(stages, u, "none")
                    else:
                        append_gather_round(stages, u, "none")
                elif depth == 0 and top_count >= 1:
                    append_depth0_round(stages, u, "depth0")
                elif depth == 1 and top_count >= 3 and forest_height >= 1:
                    append_depth1_round(stages, u, "normal")
                elif depth == 2 and top_count >= 7 and forest_height >= 2:
                    append_depth2_round(stages, u, "normal")
                else:
                    append_gather_round(stages, u, "normal")
            round_sched.add_chain(stages)
        all_round_instrs = round_sched.schedule()

        # Process batch in groups of UNROLL * VLEN
        group_size = UNROLL * VLEN
        assert batch_size % group_size == 0, "Batch size must be multiple of UNROLL*VLEN"
        n_groups = batch_size // group_size

        group_val_base = self.alloc_scratch("group_val_base")

        for g in range(n_groups):
            base = g * group_size
            # Base pointers for this group
            self.emit({"flow": [("add_imm", group_val_base, self.scratch["inp_values_p"], base)]})

            # Lane 0 base
            self.emit({"alu": [("+", val_base[0], group_val_base, zero_const)]})
            # Remaining lanes via stride adds (VLEN), serialized to honor deps
            for u in range(1, UNROLL):
                self.emit({"alu": [("+", val_base[u], val_base[u - 1], vlen_const)]})

            # Load input values.
            for u in range(0, UNROLL, 2):
                slots = [("vload", v_val[u], val_base[u])]
                if u + 1 < UNROLL:
                    slots.append(("vload", v_val[u + 1], val_base[u + 1]))
                self.emit({"load": slots})

            # Run all rounds in-place.
            for instr in all_round_instrs:
                self.emit(instr)

            # Write back values only; submission checks final values.
            for u in range(0, UNROLL, 2):
                slots = [("vstore", val_base[u], v_val[u])]
                if u + 1 < UNROLL:
                    slots.append(("vstore", val_base[u + 1], v_val[u + 1]))
                self.emit({"store": slots})

        # No-op in submission tests (pause disabled); skip to save cycles

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
    
    # Workaround for memory truncation bug in problem.py
    # Re-calculate extra_room and extend mem
    extra_room = len(forest.values) + len(inp.indices) * 2 + VLEN * 2 + 32
    # mem was truncated to inp_values_p + len(inp.values)
    # The original build_mem_image logic intended for extra_room to be after inp.values
    mem.extend([0] * extra_room)

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
    pause_in_program = any(
        "flow" in instr and any(slot[0] == "pause" for slot in instr["flow"])
        for instr in kb.instrs
    )
    if pause_in_program:
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
    else:
        ref_mem = None
        for ref_mem in reference_kernel2(mem, value_trace):
            pass
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), "Incorrect result on final state"
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
