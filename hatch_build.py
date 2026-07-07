from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface  # type: ignore[misc]


class CustomBuildHook(BuildHookInterface):  # type: ignore[misc]
    """Detect pre-built _alchemy native binary and tag the wheel as platform-specific."""

    def initialize(self, version: str, build_data: dict) -> None:  # type: ignore[type-arg]
        if self.target_name != "wheel":
            return

        pkg_dir = Path(self.root) / "src" / "tinyagent"
        binary_paths = sorted(pkg_dir.glob("_alchemy.*"))  # .so / .pyd / .dylib
        if not binary_paths:
            return

        build_data["infer_tag"] = True
        build_data["pure_python"] = False

        artifacts: list[str] = build_data.setdefault("artifacts", [])
        for path in binary_paths:
            artifacts.append(str(path.relative_to(Path(self.root))))
