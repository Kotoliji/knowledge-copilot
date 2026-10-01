"""Versioned prompt registry.

Prompts live in YAML files next to this module: `<name>.v<N>.yaml`.
Changing a prompt = adding a new version file, so every answer can be traced
back to the exact prompt text that produced it (`prompt_version` in the API response).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_FILE = re.compile(r"^(?P<name>[a-z0-9_]+)\.v(?P<version>\d+)\.yaml$")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    template: str
    description: str = ""

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class PromptRegistry:
    def __init__(self, directory: str | Path, pins: dict[str, int] | None = None):
        self.directory = Path(directory)
        self.pins = pins or {}
        self._cache: dict[tuple[str, int], Prompt] = {}

    def _load(self, name: str, version: int) -> Prompt:
        key = (name, version)
        if key not in self._cache:
            path = self.directory / f"{name}.v{version}.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            self._cache[key] = Prompt(name=name, version=version, template=data["template"],
                                      description=data.get("description", ""))
        return self._cache[key]

    def versions(self, name: str) -> list[int]:
        versions = []
        for path in self.directory.glob(f"{name}.v*.yaml"):
            m = _FILE.match(path.name)
            if m and m.group("name") == name:
                versions.append(int(m.group("version")))
        if not versions:
            raise KeyError(f"No prompt named {name!r} in {self.directory}")
        return sorted(versions)

    def get(self, name: str, version: int | None = None) -> Prompt:
        version = version or self.pins.get(name) or self.versions(name)[-1]
        return self._load(name, version)

    def list_all(self) -> dict[str, list[int]]:
        names = {m.group("name") for p in self.directory.glob("*.v*.yaml") if (m := _FILE.match(p.name))}
        return {n: self.versions(n) for n in sorted(names)}
