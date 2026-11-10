"""Render an HTML gallery of extracted visual regions for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shafi.ingestion.page_regionizer import RenderedPage, render_region_gallery_html
from shafi.models.legal_objects import VisualRegion


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-json", type=Path, required=True)
    parser.add_argument("--regions-json", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    return parser


def main() -> int:
    """Run the region rendering CLI.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    pages = [RenderedPage.model_validate(item) for item in json.loads(args.pages_json.read_text(encoding="utf-8"))]
    raw_regions = json.loads(args.regions_json.read_text(encoding="utf-8"))
    regions_by_page = {
        page_id: [VisualRegion.model_validate(region) for region in regions]
        for page_id, regions in raw_regions.items()
    }
    html_output = render_region_gallery_html(pages, regions_by_page)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html_output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
