"""vLLM monkeypatch v4: Pass device_group (NCCL) to AiterCommunicator.

v3 bug: passed only cpu_group (gloo) → NCCL fallback used gloo → CPU ops → no graph replay.
v4 fix: pass device_group for NCCL fallback, cpu_group for iris init only.

Usage: python3 vllm_monkeypatch_v4.py
  Patches vllm/distributed/.../cuda_communicator.py in-place.
  Safe to run multiple times (idempotent).
"""
import glob
import sys

cc_files = glob.glob('/usr/local/lib/python*/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py')
if not cc_files:
    cc_files = glob.glob('/usr/local/lib/python*/dist-packages/vllm/distributed/cuda_communicator.py')
if not cc_files:
    print("ERROR: cuda_communicator.py not found")
    sys.exit(1)

cc_path = cc_files[0]
print(f"Patching {cc_path}")

with open(cc_path, 'r') as f:
    cc_src = f.read()

if 'aiter_comm' in cc_src:
    print("Already patched, skipping")
    sys.exit(0)

# --- Patch 1: Add aiter_comm initialization after qr_comm ---
qr_init_anchor = 'self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)'
if qr_init_anchor not in cc_src:
    print("ERROR: QuickAllReduce init not found")
    sys.exit(1)

AITER_INIT = """
                # Initialize iris gluon all-reduce (aiter communicator)
                try:
                    from aiter.ops.triton.comms.communicator import AiterCommunicator
                    self.aiter_comm = AiterCommunicator(
                        group=self.cpu_group,
                        device=self.device,
                        device_group=self.device_group,
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "AiterCommunicator init failed: %s", e
                    )
                    self.aiter_comm = None"""

cc_src = cc_src.replace(
    qr_init_anchor,
    qr_init_anchor + AITER_INIT
)

# --- Patch 2: Add dispatch before QuickReduce ---
AITER_DISPATCH = """\
        # try aiter (iris gluon) first if enabled
        aiter_comm = getattr(self, "aiter_comm", None)
        if (
            aiter_comm is not None
            and not aiter_comm.disabled
            and aiter_comm.should_allreduce(input_)
        ):
            out = aiter_comm.all_reduce(input_)
            return out
"""

qr_comment = '# always try quick reduce first'
fallback_anchor = 'qr_comm = self.qr_comm'
anchor = qr_comment if qr_comment in cc_src else (fallback_anchor if fallback_anchor in cc_src else None)
if anchor is None:
    print("ERROR: No dispatch anchor found")
    sys.exit(1)

lines = cc_src.split('\n')
anchor_idx = None
anchor_indent = 8
for i, line in enumerate(lines):
    if anchor in line:
        anchor_idx = i
        anchor_indent = len(line) - len(line.lstrip())
        break

if anchor_idx is None:
    print("ERROR: Anchor line not found")
    sys.exit(1)

dispatch_lines = []
for dl in AITER_DISPATCH.split('\n'):
    if dl.strip() == '':
        dispatch_lines.append('')
    else:
        stripped = dl.lstrip()
        original_indent = len(dl) - len(stripped)
        relative_indent = original_indent - 8
        dispatch_lines.append(' ' * (anchor_indent + relative_indent) + stripped)

new_lines = lines[:anchor_idx] + dispatch_lines + lines[anchor_idx:]
cc_src = '\n'.join(new_lines)

print(f"Inserted aiter init (with device_group) after QuickAllReduce")
print(f"Inserted aiter dispatch before '{anchor}'")

with open(cc_path, 'w') as f:
    f.write(cc_src)

import py_compile
py_compile.compile(cc_path, doraise=True)
print("Monkeypatch v4 applied successfully")
