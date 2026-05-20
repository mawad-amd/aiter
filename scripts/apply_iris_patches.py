"""Apply iris library patches required for v10h communicator.

Two patches:
1. Remove refresh_peer_access() from symmetric_heap.py allocate() —
   crashes CUDA graph capture.
2. Set _NCCL_FALLBACK_BYTES=0 in iris/ccl/all_reduce.py —
   route all small tensors through iris CCL instead of NCCL.

Usage: python3 apply_iris_patches.py
  Patches the installed iris package in-place.
"""
import os
import subprocess
import sys


def patch_symmetric_heap():
    result = subprocess.run(
        ['python3', '-c',
         'import iris; import os; print(os.path.join('
         'os.path.dirname(iris.__file__), '
         '"host", "memory", "symmetric_heap.py"))'],
        capture_output=True, text=True, timeout=10
    )
    heap_path = result.stdout.strip()
    if not heap_path or not os.path.exists(heap_path):
        print("WARNING: Could not find iris symmetric_heap.py")
        return
    with open(heap_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    new_lines = []
    in_allocate = False
    for i, line in enumerate(lines):
        if 'def allocate(' in line:
            in_allocate = True
        elif in_allocate and line.strip().startswith('def '):
            in_allocate = False
        if in_allocate and 'self.refresh_peer_access()' in line:
            print(f"  Removed refresh_peer_access() from allocate() at line {i+1}")
            continue
        new_lines.append(line)
    with open(heap_path, 'w') as f:
        f.write('\n'.join(new_lines))


def patch_ccl_threshold():
    result = subprocess.run(
        ['python3', '-c',
         'import iris.ccl.all_reduce; import os; '
         'print(os.path.abspath(iris.ccl.all_reduce.__file__))'],
        capture_output=True, text=True, timeout=10
    )
    ar_path = result.stdout.strip()
    if not ar_path or not os.path.exists(ar_path):
        print("WARNING: Could not find iris/ccl/all_reduce.py")
        return
    with open(ar_path, 'r') as f:
        content = f.read()
    content = content.replace(
        '_NCCL_FALLBACK_BYTES = 4 * 1024 * 1024',
        '_NCCL_FALLBACK_BYTES = 0  # patched: route all sizes through iris'
    )
    with open(ar_path, 'w') as f:
        f.write(content)
    print(f"  Patched _NCCL_FALLBACK_BYTES=0 in {ar_path}")


if __name__ == '__main__':
    print("Patching iris symmetric_heap...")
    patch_symmetric_heap()
    print("Patching iris CCL threshold...")
    patch_ccl_threshold()
    print("Iris patches applied successfully")
