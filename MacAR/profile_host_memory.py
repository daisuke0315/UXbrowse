"""Read macOS VM and per-process counters using the installed SDK ABI."""
import ctypes
import re
import subprocess


class RUsageV2(ctypes.Structure):
    _fields_ = [('ri_uuid', ctypes.c_uint8*16)] + [(name, ctypes.c_uint64) for name in (
        'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
        'wired_size', 'resident_size', 'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime',
        'child_user_time', 'child_system_time', 'child_pkg_idle_wkups', 'child_interrupt_wkups',
        'child_pageins', 'child_elapsed_abstime', 'diskio_bytesread', 'diskio_byteswritten')]


def process_memory(pid):
    info = RUsageV2()
    lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    if lib.proc_pid_rusage(pid, 2, ctypes.byref(info)):
        raise OSError(ctypes.get_errno(), 'proc_pid_rusage failed')
    return {name: getattr(info, name) for name in ('user_time', 'system_time', 'pageins',
        'wired_size', 'resident_size', 'phys_footprint', 'diskio_bytesread', 'diskio_byteswritten')}


def host_memory(pid):
    vm = subprocess.check_output(['vm_stat'], text=True)
    counters = {k: int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', vm, re.M)}
    return dict(vm=counters, page_size=int(re.search(r'page size of (\d+)', vm)[1]),
        swap=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip(),
        process=process_memory(pid))
