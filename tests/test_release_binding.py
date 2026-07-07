from __future__ import annotations

import sys
from pathlib import Path

from scripts.check_release_binding import check


def test_release_binding_check_passes_without_require_present(tmp_path: Path) -> None:
    package_dir = tmp_path / "tinyagent"
    package_dir.mkdir()

    errors = check(package_dir=package_dir)

    assert errors == []


def test_release_binding_check_requires_staged_binary_when_requested(tmp_path: Path) -> None:
    package_dir = tmp_path / "tinyagent"
    package_dir.mkdir()

    errors = check(package_dir=package_dir, require_present=True)

    assert errors == [
        "No staged `_alchemy` binary found in src/tinyagent/. "
        "Build the binding from the external tinyagent-alchemy repo and copy "
        "the resulting artifact into src/tinyagent/ before building release wheels."
    ]


def test_release_binding_check_rejects_incompatible_staged_binary(tmp_path: Path) -> None:
    package_dir = tmp_path / "tinyagent"
    package_dir.mkdir()

    staged_binary = package_dir / "_alchemy.abi3.so"
    if sys.platform == "darwin":
        staged_binary.write_bytes(b"\x7fELF" + b"\x00" * 60)
        expected_actual = "ELF"
        expected_target = "Mach-O"
    elif sys.platform.startswith("linux"):
        staged_binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 60)
        expected_actual = "Mach-O"
        expected_target = "ELF"
    elif sys.platform.startswith("win"):
        staged_binary.write_bytes(b"\x7fELF" + b"\x00" * 60)
        expected_actual = "ELF"
        expected_target = "PE"
    else:
        staged_binary.write_bytes(b"\x00" * 64)
        expected_actual = None
        expected_target = None

    errors = check(package_dir=package_dir, require_present=True)

    if expected_target is None:
        assert errors == []
    else:
        assert errors == [
            "Staged `_alchemy` binary is incompatible with this host platform: "
            f"tinyagent/_alchemy.abi3.so is {expected_actual}, expected {expected_target}. "
            "Build/copy the binding artifact for this platform before packaging release wheels."
        ]
