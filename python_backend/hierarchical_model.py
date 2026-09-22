"""Universal Hierarchical Document Model.

Represents all document types (Magazines, Quarterly Reports, Tables, Forms,
TOC/Index, and Content) as a dynamic, recursive Key -> Sub-key -> Value tree.
"""
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple


class HierarchicalNode:
    """A recursive tree node representing Key -> Children or Key -> Value.

    Follows the universal schema:
    {
      "key": "Parent Key",
      "children": [
        {"key": "Sub-key", "value": "Value"},
        {"key": "Another Sub-key", "children": [...]}
      ]
    }
    """

    def __init__(
        self,
        key: str,
        value: Optional[str] = None,
        children: Optional[List[HierarchicalNode]] = None,
        source_page: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.key: str = str(key or "").strip()
        self.value: Optional[str] = str(value).strip() if value is not None else None
        self.children: List[HierarchicalNode] = children if children is not None else []
        self.source_page: Optional[int] = source_page
        self.metadata: Dict[str, Any] = metadata or {}

    def add_child(self, child: HierarchicalNode) -> HierarchicalNode:
        if child is not None:
            self.children.append(child)
        return child

    def to_dict(self, include_source_page: bool = False) -> Dict[str, Any]:
        """Convert to pure JSON-compatible dict."""
        res: Dict[str, Any] = {"key": self.key}
        if self.value is not None:
            res["value"] = self.value
        if self.children:
            res["children"] = [c.to_dict(include_source_page=include_source_page) for c in self.children]
        if include_source_page and self.source_page is not None:
            res["source_page"] = self.source_page
        return res

    def find(self, search_key: str) -> Optional[HierarchicalNode]:
        """Case-insensitive, whitespace-normalized recursive search for first matching node."""
        s = search_key.lower().strip()
        if self.key.lower().strip() == s:
            return self
        for child in self.children:
            match = child.find(search_key)
            if match is not None:
                return match
        return None

    def find_all(self, search_key: str) -> List[HierarchicalNode]:
        """Find all nodes matching search_key recursively."""
        results: List[HierarchicalNode] = []
        s = search_key.lower().strip()
        if self.key.lower().strip() == s:
            results.append(self)
        for child in self.children:
            results.extend(child.find_all(search_key))
        return results

    def flatten_paths(self, prefix: str = "") -> List[Tuple[str, str, Optional[int]]]:
        """Returns a list of (Hierarchical_Path, Value, source_page) tuples.

        e.g. ("Category > Sub-category > Metric Key", "0", 1)
        """
        current_path = f"{prefix} > {self.key}" if prefix else self.key
        paths: List[Tuple[str, str, Optional[int]]] = []
        if self.value is not None:
            paths.append((current_path, self.value, self.source_page))
        for child in self.children:
            paths.extend(child.flatten_paths(current_path))
        return paths

    def to_indexable_points(
        self,
        doc_metadata: Dict[str, Any],
        parent_path: Optional[List[str]] = None,
        include_branches: bool = True,
    ) -> List[Dict[str, Any]]:
        """Recursively converts this node and its descendants into indexable Qdrant point dictionaries.

        Constructs complete hierarchical paths:
          - Leaf nodes: 'Parent > Child > Sub-key = Value' (node_type='leaf')
          - Branch nodes: 'Parent > Child' (node_type='branch', if include_branches=True)
        """
        doc_id = str(doc_metadata.get("document_id", "default_doc"))
        file_name = str(doc_metadata.get("file_name", doc_metadata.get("filename", "document")))
        doc_type = str(doc_metadata.get("document_type", "magazine")).lower()
        year_val = doc_metadata.get("year")
        try:
            year_int = int(year_val) if year_val is not None else None
        except (ValueError, TypeError):
            year_int = None
        quarter_val = doc_metadata.get("quarter")
        report_period = doc_metadata.get("report_period")

        points: List[Dict[str, Any]] = []

        current_path = list(parent_path) if parent_path else []
        if self.key:
            current_path.append(self.key)

        # 1. Leaf node (holds concrete value)
        if self.value is not None and str(self.value).strip() != "":
            val_str = str(self.value).strip()
            path_text = " > ".join(current_path) + f" = {val_str}" if current_path else f"= {val_str}"
            leaf_id = f"{doc_id}::hier::leaf::{path_text}"
            points.append({
                "id": leaf_id,
                "document_id": doc_id,
                "file_name": file_name,
                "filename": file_name,
                "source_filename": file_name,
                "document_type": doc_type,
                "year": year_int,
                "quarter": quarter_val,
                "report_period": report_period,
                "source_page": self.source_page,
                "page": self.source_page,
                "page_number": self.source_page,
                "node_key": self.key,
                "node_value": val_str,
                "hierarchy_path": list(current_path),
                "hierarchy_path_text": path_text,
                "text": path_text,
                "node_type": "leaf",
            })

        # 2. Branch node (contains children)
        elif self.children and include_branches and current_path:
            path_text = " > ".join(current_path)
            branch_id = f"{doc_id}::hier::branch::{path_text}"
            branch_page = self.source_page
            if branch_page is None:
                for ch in self.children:
                    if ch.source_page is not None:
                        branch_page = ch.source_page
                        break
            points.append({
                "id": branch_id,
                "document_id": doc_id,
                "file_name": file_name,
                "filename": file_name,
                "source_filename": file_name,
                "document_type": doc_type,
                "year": year_int,
                "quarter": quarter_val,
                "report_period": report_period,
                "source_page": branch_page,
                "page": branch_page,
                "page_number": branch_page,
                "node_key": self.key,
                "node_value": None,
                "hierarchy_path": list(current_path),
                "hierarchy_path_text": path_text,
                "text": path_text,
                "node_type": "branch",
            })

        # Recurse into children
        for child in self.children:
            points.extend(
                child.to_indexable_points(
                    doc_metadata=doc_metadata,
                    parent_path=current_path,
                    include_branches=include_branches,
                )
            )

        return points

    def count_nodes(self) -> int:
        """Total number of nodes in this subtree."""
        return 1 + sum(child.count_nodes() for child in self.children)

    def __repr__(self) -> str:
        if self.value is not None:
            return f"HierarchicalNode(key={self.key!r}, value={self.value!r})"
        return f"HierarchicalNode(key={self.key!r}, children={len(self.children)})"


class DocumentHierarchy:
    """Wrapper holding document-level metadata separate from the recursive knowledge tree."""

    def __init__(
        self,
        document_id: str,
        file_name: str,
        document_type: str,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        report_period: Optional[str] = None,
        total_pages: int = 1,
        root: Optional[HierarchicalNode] = None,
    ):
        self.metadata: Dict[str, Any] = {
            "document_id": document_id,
            "file_name": file_name,
            "document_type": document_type,
            "year": year,
            "quarter": quarter,
            "report_period": report_period,
            "total_pages": total_pages,
        }
        self.root: HierarchicalNode = root or HierarchicalNode(key=file_name)

    def to_dict(self, include_source_page: bool = False) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "hierarchy": self.root.to_dict(include_source_page=include_source_page),
        }

    def flatten_all_paths(self) -> List[Tuple[str, str, Optional[int]]]:
        return self.root.flatten_paths()

    def to_indexable_points(self, include_branches: bool = True) -> List[Dict[str, Any]]:
        """Converts the full DocumentHierarchy into a list of indexable Qdrant point dictionaries.

        To ensure clean paths like 'Section > Table > Sub-key = Value', if root
        only acts as the document container and has children, traversal begins
        at root's children with parent_path=[].
        """
        if not self.root:
            return []

        if self.root.children and self.root.value is None:
            points: List[Dict[str, Any]] = []
            for child in self.root.children:
                points.extend(
                    child.to_indexable_points(
                        doc_metadata=self.metadata,
                        parent_path=[],
                        include_branches=include_branches,
                    )
                )
            return points
        else:
            return self.root.to_indexable_points(
                doc_metadata=self.metadata,
                parent_path=[],
                include_branches=include_branches,
            )


class HierarchicalStore:
    """Document-isolated persistence for universal hierarchical JSON representations."""

    def __init__(self, storage_dir: Optional[str] = None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.storage_dir = storage_dir or os.path.join(base, "doc_metadata")
        os.makedirs(self.storage_dir, exist_ok=True)

    def _path(self, document_id: str) -> str:
        safe_id = re.sub(r"[^\w.\-]", "_", document_id)
        return os.path.join(self.storage_dir, f"{safe_id}_hierarchy.json")

    def save_hierarchy(self, document_id: str, filename: str, hierarchy: DocumentHierarchy) -> str:
        fpath = self._path(document_id)
        payload = hierarchy.to_dict(include_source_page=True)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return fpath

    def load_hierarchy(self, document_id: str) -> Optional[Dict[str, Any]]:
        fpath = self._path(document_id)
        if not os.path.exists(fpath):
            return None
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def delete_document(self, document_id: str) -> None:
        fpath = self._path(document_id)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
