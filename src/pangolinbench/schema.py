"""Schemas and lightweight config loading for PangolinBench."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when PangolinBench input data does not match the schema."""


def _require_mapping(data: Any, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaValidationError(f"{label} must be an object")
    return data


def _require_non_empty_str(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{label}.{key} is required and must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaValidationError(f"{key} must be a string when provided")
    return value


def _optional_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{key} must be a number when provided")
    return float(value)


def _optional_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaValidationError(f"{key} must be a list of strings when provided")
    return list(value)


@dataclass(frozen=True)
class BenchmarkExample:
    """A single PangolinBench text example."""

    id: str
    subset: str
    text: str
    weight: float = 1.0
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    license: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, allow_empty_text: bool = False) -> "BenchmarkExample":
        data = _require_mapping(data, "BenchmarkExample")
        example_id = _require_non_empty_str(data, "id", "BenchmarkExample")
        subset = _require_non_empty_str(data, "subset", "BenchmarkExample")
        if "text" not in data or not isinstance(data["text"], str):
            raise SchemaValidationError("BenchmarkExample.text is required and must be a string")
        if not allow_empty_text and data["text"] == "":
            raise SchemaValidationError("BenchmarkExample.text must not be empty")
        return cls(
            id=example_id,
            subset=subset,
            text=data["text"],
            weight=_optional_float(data, "weight", 1.0),
            tags=_optional_str_list(data, "tags"),
            source=_optional_str(data, "source"),
            license=_optional_str(data, "license"),
            notes=_optional_str(data, "notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "subset": self.subset,
            "text": self.text,
            "weight": self.weight,
            "tags": list(self.tags),
        }
        if self.source is not None:
            result["source"] = self.source
        if self.license is not None:
            result["license"] = self.license
        if self.notes is not None:
            result["notes"] = self.notes
        return result


@dataclass(frozen=True)
class LexiconItem:
    """A PangolinBench lexicon item used for Taiwan term over-segmentation checks."""

    term: str
    category: str | None = None
    weight: float = 1.0
    tags: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LexiconItem":
        data = _require_mapping(data, "LexiconItem")
        return cls(
            term=_require_non_empty_str(data, "term", "LexiconItem"),
            category=_optional_str(data, "category"),
            weight=_optional_float(data, "weight", 1.0),
            tags=_optional_str_list(data, "tags"),
            variants=_optional_str_list(data, "variants"),
            notes=_optional_str(data, "notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "term": self.term,
            "weight": self.weight,
            "tags": list(self.tags),
            "variants": list(self.variants),
        }
        if self.category is not None:
            result["category"] = self.category
        if self.notes is not None:
            result["notes"] = self.notes
        return result


@dataclass(frozen=True)
class ManifestExampleSpec:
    path: str
    subset: str
    format: str = "jsonl"
    text_field: str = "text"
    license: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestExampleSpec":
        data = _require_mapping(data, "ManifestExampleSpec")
        return cls(
            path=_require_non_empty_str(data, "path", "ManifestExampleSpec"),
            subset=_require_non_empty_str(data, "subset", "ManifestExampleSpec"),
            format=str(data.get("format", "jsonl")),
            text_field=str(data.get("text_field", "text")),
            license=_optional_str(data, "license"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "path": self.path,
            "subset": self.subset,
            "format": self.format,
            "text_field": self.text_field,
        }
        if self.license is not None:
            result["license"] = self.license
        return result


@dataclass(frozen=True)
class ManifestLexiconSpec:
    path: str
    format: str = "tsv"
    term_field: str = "term"
    category_field: str | None = "category"
    weight_field: str | None = "weight"
    license: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestLexiconSpec":
        data = _require_mapping(data, "ManifestLexiconSpec")
        return cls(
            path=_require_non_empty_str(data, "path", "ManifestLexiconSpec"),
            format=str(data.get("format", "tsv")),
            term_field=str(data.get("term_field", "term")),
            category_field=_optional_str(data, "category_field") if "category_field" in data else "category",
            weight_field=_optional_str(data, "weight_field") if "weight_field" in data else "weight",
            license=_optional_str(data, "license"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "format": self.format,
            "term_field": self.term_field,
        }
        if self.category_field is not None:
            result["category_field"] = self.category_field
        if self.weight_field is not None:
            result["weight_field"] = self.weight_field
        if self.license is not None:
            result["license"] = self.license
        return result


@dataclass(frozen=True)
class CorpusManifest:
    """Manifest for private or licensed PangolinBench heldout data."""

    name: str
    version: str
    description: str | None = None
    examples: list[ManifestExampleSpec] = field(default_factory=list)
    lexicons: list[ManifestLexiconSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusManifest":
        data = _require_mapping(data, "CorpusManifest")
        examples = data.get("examples", [])
        lexicons = data.get("lexicons", [])
        if not isinstance(examples, list):
            raise SchemaValidationError("CorpusManifest.examples must be a list")
        if not isinstance(lexicons, list):
            raise SchemaValidationError("CorpusManifest.lexicons must be a list")
        return cls(
            name=_require_non_empty_str(data, "name", "CorpusManifest"),
            version=_require_non_empty_str(data, "version", "CorpusManifest"),
            description=_optional_str(data, "description"),
            examples=[ManifestExampleSpec.from_dict(item) for item in examples],
            lexicons=[ManifestLexiconSpec.from_dict(item) for item in lexicons],
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "examples": [item.to_dict() for item in self.examples],
            "lexicons": [item.to_dict() for item in self.lexicons],
        }
        if self.description is not None:
            result["description"] = self.description
        return result


def load_manifest(path: str | Path) -> CorpusManifest:
    data = load_mapping_file(path)
    return CorpusManifest.from_dict(data)


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return _require_mapping(json.loads(text), str(path))
    return _load_yaml_like_mapping(text, path)


def _load_yaml_like_mapping(text: str, path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    return _require_mapping(loaded, str(path))


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple mapping-only YAML used by the default config.

    This fallback intentionally supports only nested mappings and scalar values.
    Projects with richer manifests should install PyYAML or provide JSON.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise SchemaValidationError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(mark in value for mark in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value
