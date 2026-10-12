"""
Self-contained mock-driven tests for the SystemInfo bug-fix in
torch/_dynamo/package.py.

These tests exercise the LOGIC of the three changed code paths in isolation,
without importing the full package.py (which has many PyTorch-internal
dependencies). Each test:
  - Describes what the old (buggy) code would have done.
  - Asserts the new (fixed) behaviour.

Run with:
    python3 test_system_info_fix.py      # plain unittest
    pytest  test_system_info_fix.py -v   # if pytest is installed
"""

import dataclasses
import platform
import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Inline the fixed logic under test
# (mirrors the exact code in the fixed package.py, minus PyTorch internals)
# ---------------------------------------------------------------------------

def _make_torch_stub(*, accel_type: str | None = None, accel_gpu_name: str | None = None,
                     accel_toolkit: str | None = None) -> types.ModuleType:
    """Build a minimal torch stub for mock-driven tests."""
    torch = types.ModuleType("torch")
    torch.__version__ = "2.13.0a0"

    # torch.accelerator sub-module
    accel_mod = types.ModuleType("torch.accelerator")
    if accel_type is not None:
        mock_accel = MagicMock()
        mock_accel.type = accel_type
        accel_mod.current_accelerator = MagicMock(return_value=mock_accel)
    else:
        accel_mod.current_accelerator = MagicMock(
            side_effect=RuntimeError("no accelerator available")
        )
    torch.accelerator = accel_mod

    # torch.version sub-module
    version_ns = types.SimpleNamespace(cuda=None, xpu=None)
    if accel_type is not None:
        setattr(version_ns, accel_type, accel_toolkit)
    torch.version = version_ns

    # Device sub-module
    if accel_type is not None:
        dev_mod = MagicMock()
        dev_mod.is_available.return_value = True
        dev_mod.get_device_name.return_value = accel_gpu_name
        setattr(torch, accel_type, dev_mod)

    # Always provide cuda/xpu stubs (unavailable)
    for dt in ("cuda", "xpu"):
        if not hasattr(torch, dt):
            m = MagicMock()
            m.is_available.return_value = False
            setattr(torch, dt, m)

    return torch


@dataclasses.dataclass(frozen=True)
class _SystemInfo:
    """Inline copy of the fixed SystemInfo dataclass fields."""
    python_version: str
    torch_version: str
    toolkit_version: str | None
    triton_version: tuple[int, int] | None
    gpu_name: str | None
    CHECK_GPUS = ("cuda", "xpu")

    @classmethod
    def current(cls, *, _torch=None) -> "_SystemInfo":
        """Fixed current() — uses torch.accelerator.current_accelerator."""
        import platform as _platform
        torch = _torch  # injected in tests; real code uses the global

        gpu_name: str | None = None
        toolkit_version: str | None = None

        try:
            accel = torch.accelerator.current_accelerator(check_available=True)
            device_type: str = accel.type

            device_module = getattr(torch, device_type, None)
            if device_module is not None:
                try:
                    gpu_name = device_module.get_device_name()
                except Exception:
                    pass
                toolkit_version = getattr(torch.version, device_type, None)
        except RuntimeError:
            pass  # CPU-only host

        return cls(
            python_version=_platform.python_version(),
            torch_version=torch.__version__,
            toolkit_version=toolkit_version,
            triton_version=(0, 0),
            gpu_name=gpu_name,
        )

    def check_compatibility(
        self, other: "_SystemInfo", device_type: str = "cpu", *, _torch=None
    ) -> None:
        """Fixed check_compatibility() — gates on any torch.<type>.is_available()."""
        torch = _torch

        if self.python_version != other.python_version:
            raise RuntimeError(
                f"Compile package was created with a different Python version: {self.python_version}"
            )
        if self.torch_version != other.torch_version:
            raise RuntimeError(
                f"Compile package was created with a different PyTorch version: {self.torch_version}"
            )

        device_module = getattr(torch, device_type, None)
        _is_accelerator = (
            callable(getattr(device_module, "is_available", None))
            and device_module.is_available()
        )

        if _is_accelerator:
            if not device_module.is_available():
                raise RuntimeError(f"{device_type} is not available")
            if self.toolkit_version != other.toolkit_version:
                raise RuntimeError(
                    f"Compile package was created with a different toolkit version: {self.toolkit_version}"
                )
            if (
                other.triton_version != (0, 0)
                and self.triton_version != other.triton_version
            ):
                raise RuntimeError(
                    f"Compile package was created with a different Triton version: {self.triton_version}"
                )
            if other.gpu_name is not None and self.gpu_name != other.gpu_name:
                raise RuntimeError(
                    f"Compile package was created with different GPU: "
                    f"cached={self.gpu_name}, current={other.gpu_name}"
                )


def _collapse_device_types(device_types: frozenset[str], *, _torch) -> str:
    """Fixed _collapse_device_types() — prefers the running accelerator."""
    try:
        accel_type = _torch.accelerator.current_accelerator(check_available=True).type
        if accel_type in device_types:
            return accel_type
    except RuntimeError:
        pass

    for device_type in _SystemInfo.CHECK_GPUS:
        if device_type in device_types:
            return device_type

    return next((d for d in sorted(device_types) if d != "cpu"), "cpu")


def _base_info(**kw) -> _SystemInfo:
    defaults = dict(
        python_version=platform.python_version(),
        torch_version="2.13.0a0",
        toolkit_version=None,
        triton_version=(0, 0),
        gpu_name=None,
    )
    defaults.update(kw)
    return _SystemInfo(**defaults)


# ===========================================================================
# Test Suite 1: SystemInfo.current()
# ===========================================================================

class TestSystemInfoCurrent(unittest.TestCase):

    def test_cpu_only_returns_none_identity(self):
        """BUG: old code iterated CHECK_GPUS and left gpu_name=None on non-cuda hosts.
        FIX: current() catches RuntimeError from current_accelerator, returns None/None."""
        torch = _make_torch_stub(accel_type=None)  # no accelerator
        info = _SystemInfo.current(_torch=torch)
        self.assertIsNone(info.gpu_name,
                          "cpu-only host must return gpu_name=None")
        self.assertIsNone(info.toolkit_version,
                          "cpu-only host must return toolkit_version=None")

    def test_cuda_identity_recorded(self):
        """CUDA — byte-identical to old behaviour."""
        torch = _make_torch_stub(accel_type="cuda",
                                 accel_gpu_name="NVIDIA A100-SXM4-80GB",
                                 accel_toolkit="12.4")
        info = _SystemInfo.current(_torch=torch)
        self.assertEqual(info.gpu_name, "NVIDIA A100-SXM4-80GB")
        self.assertEqual(info.toolkit_version, "12.4")

    def test_mps_identity_recorded(self):
        """BUG: mps not in CHECK_GPUS → gpu_name=None before fix.
        FIX: accelerator chain covers mps → gpu_name populated."""
        torch = _make_torch_stub(accel_type="mps",
                                 accel_gpu_name="Apple M2 Pro",
                                 accel_toolkit=None)
        info = _SystemInfo.current(_torch=torch)
        self.assertEqual(info.gpu_name, "Apple M2 Pro",
                         "MPS identity must be recorded")

    def test_hpu_identity_recorded(self):
        """BUG: hpu not in CHECK_GPUS → no hardware identity recorded.
        FIX: gpu_name='Gaudi2' is populated."""
        torch = _make_torch_stub(accel_type="hpu",
                                 accel_gpu_name="Gaudi2",
                                 accel_toolkit="1.15.0")
        info = _SystemInfo.current(_torch=torch)
        self.assertEqual(info.gpu_name, "Gaudi2")
        self.assertEqual(info.toolkit_version, "1.15.0")

    def test_private_use1_renamed_backend(self):
        """Out-of-tree PrivateUse1-renamed backend: identity must be captured."""
        torch = _make_torch_stub(accel_type="mybackend",
                                 accel_gpu_name="MyDevice-v3",
                                 accel_toolkit="9.9.9")
        info = _SystemInfo.current(_torch=torch)
        self.assertEqual(info.gpu_name, "MyDevice-v3")

    def test_get_device_name_failure_is_swallowed(self):
        """If get_device_name() raises, current() must still return (best-effort)."""
        torch = _make_torch_stub(accel_type="cuda",
                                 accel_gpu_name=None,
                                 accel_toolkit="12.4")
        torch.cuda.get_device_name.side_effect = RuntimeError("driver error")
        info = _SystemInfo.current(_torch=torch)
        self.assertIsNone(info.gpu_name)  # best-effort: None, not a crash

    def test_accelerator_called_not_check_gpus_loop(self):
        """BUG: old code called torch.cuda.is_available() first.
        FIX: torch.accelerator.current_accelerator() is called, NOT cuda.is_available."""
        torch = _make_torch_stub(accel_type="mps",
                                 accel_gpu_name="Apple M1",
                                 accel_toolkit=None)
        _SystemInfo.current(_torch=torch)
        torch.accelerator.current_accelerator.assert_called_once()
        # cuda.is_available must NOT have been called by the new code path
        torch.cuda.is_available.assert_not_called()


# ===========================================================================
# Test Suite 2: check_compatibility()
# ===========================================================================

class TestCheckCompatibility(unittest.TestCase):

    def test_cpu_skips_toolkit_check(self):
        """cpu device has no torch.cpu.is_available → accel block is skipped."""
        torch = _make_torch_stub(accel_type=None)
        saved = _base_info(toolkit_version="old", gpu_name="A100")
        current = _base_info(toolkit_version="new", gpu_name="H100")
        # Must NOT raise — toolkit/gpu checks are skipped for cpu.
        saved.check_compatibility(current, device_type="cpu", _torch=torch)

    def test_meta_skips_all_accel_checks(self):
        """meta device has no is_available() → all accel checks skipped."""
        torch = _make_torch_stub(accel_type=None)
        saved = _base_info(gpu_name="A100")
        current = _base_info(gpu_name="H100")
        saved.check_compatibility(current, device_type="meta", _torch=torch)

    def test_cuda_toolkit_mismatch_raises(self):
        """CUDA: toolkit mismatch → RuntimeError (byte-identical to old behaviour)."""
        torch = _make_torch_stub(accel_type="cuda", accel_toolkit="12.4")
        saved = _base_info(gpu_name="A100", toolkit_version="11.8")
        current = _base_info(gpu_name="A100", toolkit_version="12.4")
        with self.assertRaises(RuntimeError):
            saved.check_compatibility(current, device_type="cuda", _torch=torch)

    def test_cuda_gpu_name_mismatch_raises(self):
        torch = _make_torch_stub(accel_type="cuda", accel_toolkit="12.4")
        saved = _base_info(gpu_name="A100", toolkit_version="12.4")
        current = _base_info(gpu_name="H100", toolkit_version="12.4")
        with self.assertRaises(RuntimeError):
            saved.check_compatibility(current, device_type="cuda", _torch=torch)

    def test_cuda_compatible_no_raise(self):
        torch = _make_torch_stub(accel_type="cuda", accel_toolkit="12.4")
        saved = _base_info(gpu_name="A100", toolkit_version="12.4")
        current = _base_info(gpu_name="A100", toolkit_version="12.4")
        saved.check_compatibility(current, device_type="cuda", _torch=torch)

    def test_mps_gpu_name_mismatch_raises(self):
        """BUG: mps not in CHECK_GPUS → check silently skipped → wrong artifact loads.
        FIX: gpu_name mismatch on mps must raise RuntimeError."""
        torch = _make_torch_stub(accel_type="mps")
        saved = _base_info(gpu_name="Apple M2")
        current = _base_info(gpu_name="Apple M3")
        with self.assertRaises(RuntimeError,
                               msg="MPS gpu_name mismatch should raise after fix"):
            saved.check_compatibility(current, device_type="mps", _torch=torch)

    def test_mps_compatible_no_raise(self):
        torch = _make_torch_stub(accel_type="mps")
        saved = _base_info(gpu_name="Apple M2")
        current = _base_info(gpu_name="Apple M2")
        saved.check_compatibility(current, device_type="mps", _torch=torch)

    def test_hpu_toolkit_mismatch_raises(self):
        """BUG: hpu silently skipped in old code.
        FIX: toolkit mismatch on hpu must raise."""
        torch = _make_torch_stub(accel_type="hpu")
        saved = _base_info(gpu_name="Gaudi2", toolkit_version="1.14.0")
        current = _base_info(gpu_name="Gaudi2", toolkit_version="1.15.0")
        with self.assertRaises(RuntimeError,
                               msg="HPU toolkit mismatch should raise after fix"):
            saved.check_compatibility(current, device_type="hpu", _torch=torch)

    def test_unavailable_device_skips_checks(self):
        """A device whose is_available() returns False acts like cpu — checks skipped."""
        torch = _make_torch_stub(accel_type=None)
        offline = MagicMock()
        offline.is_available.return_value = False
        setattr(torch, "hpu", offline)

        saved = _base_info(gpu_name="Gaudi2", toolkit_version="1.14.0")
        current = _base_info(gpu_name="Gaudi3", toolkit_version="1.15.0")
        # Device reports offline → _is_accelerator=False → no checks → no raise.
        saved.check_compatibility(current, device_type="hpu", _torch=torch)


# ===========================================================================
# Test Suite 3: _collapse_device_types()
# ===========================================================================

class TestCollapseDeviceTypes(unittest.TestCase):

    def test_running_accelerator_wins_over_cpu(self):
        torch = _make_torch_stub(accel_type="mps")
        result = _collapse_device_types(frozenset({"mps", "cpu"}), _torch=torch)
        self.assertEqual(result, "mps")

    def test_running_accelerator_wins_over_check_gpus_member(self):
        """BUG: old code prioritised CHECK_GPUS; mps lost to cuda in the frozenset.
        FIX: the running accelerator wins over any CHECK_GPUS member."""
        torch = _make_torch_stub(accel_type="mps")
        result = _collapse_device_types(frozenset({"mps", "cuda", "cpu"}), _torch=torch)
        self.assertEqual(result, "mps",
                         "Running accelerator must beat CHECK_GPUS in priority")

    def test_running_accel_not_in_set_falls_back_to_check_gpus(self):
        """Accel is hpu but frozenset only contains cuda; legacy fallback fires."""
        torch = _make_torch_stub(accel_type="hpu")
        result = _collapse_device_types(frozenset({"cuda", "cpu"}), _torch=torch)
        self.assertEqual(result, "cuda")

    def test_cpu_only_frozenset_returns_cpu(self):
        torch = _make_torch_stub(accel_type=None)
        result = _collapse_device_types(frozenset({"cpu"}), _torch=torch)
        self.assertEqual(result, "cpu")

    def test_empty_frozenset_returns_cpu(self):
        torch = _make_torch_stub(accel_type=None)
        result = _collapse_device_types(frozenset(), _torch=torch)
        self.assertEqual(result, "cpu")

    def test_alphabetical_among_non_cpu_without_accel(self):
        """Without a running accelerator and no CHECK_GPUS member, alphabetical."""
        torch = _make_torch_stub(accel_type=None)
        result = _collapse_device_types(frozenset({"xla", "hpu", "cpu"}), _torch=torch)
        self.assertEqual(result, "hpu",  # 'h' < 'x'
                         "Alphabetical fallback among non-cpu types")

    def test_xpu_preserved_via_check_gpus_legacy(self):
        torch = _make_torch_stub(accel_type=None)
        result = _collapse_device_types(frozenset({"xpu", "cpu"}), _torch=torch)
        self.assertEqual(result, "xpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
