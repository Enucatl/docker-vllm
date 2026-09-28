#!/usr/bin/env python3
"""Generate fixed PDF probes and verify Paddle's complete-PDF API."""

import argparse
import base64
import subprocess
import threading
import time
from pathlib import Path

import requests
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def make_samples(directory: Path) -> None:
    """Create fixed PDFs covering order, blank pages, tables, and Unicode."""
    directory.mkdir(parents=True, exist_ok=True)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", font))

    pdf = canvas.Canvas(str(directory / "multipage.pdf"))
    for page, marker in enumerate(("FIRST ALPHA", "MIDDLE BRAVO", "LAST CHARLIE"), 1):
        pdf.setFont("DejaVu", 17)
        pdf.drawString(72, 740, f"PAGE {page} {marker}")
        pdf.drawString(72, 680, "Café München κόσμος Привет")
        pdf.drawString(72, 50, f"Footer for page {page}")
        pdf.showPage()
    pdf.save()

    pdf = canvas.Canvas(str(directory / "table.pdf"))
    pdf.setFont("DejaVu", 16)
    pdf.drawString(72, 750, "Table 1: Quarterly values")
    for y in (700, 660, 620):
        pdf.line(72, y, 420, y)
    for x in (72, 230, 420):
        pdf.line(x, 620, x, 700)
    pdf.drawString(85, 670, "Quarter")
    pdf.drawString(245, 670, "Value")
    pdf.drawString(85, 635, "Q1")
    pdf.drawString(245, 635, "42")
    pdf.drawString(72, 580, "Equation: E = mc²")
    pdf.save()

    pdf = canvas.Canvas(str(directory / "scanned.pdf"))
    pdf.drawImage("examples/test-ocr.png", 72, 500, width=450, height=140)
    pdf.save()

    pdf = canvas.Canvas(str(directory / "blank-middle.pdf"))
    for marker in ("BEFORE BLANK", None, "AFTER BLANK"):
        if marker:
            pdf.setFont("DejaVu", 16)
            pdf.drawString(72, 740, marker)
        pdf.showPage()
    pdf.save()

    pdf = canvas.Canvas(str(directory / "long-41.pdf"))
    for page in range(1, 42):
        pdf.setFont("DejaVu", 17)
        pdf.drawString(72, 740, f"LONG DOCUMENT PAGE {page:02d}")
        pdf.showPage()
    pdf.save()


def gpu_memory() -> int:
    """Read total GPU memory in MiB used by all processes."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
    )
    return int(result.stdout.splitlines()[0].strip())


def check_pdf(endpoint: str, path: Path, timeout: int) -> None:
    """Send a complete PDF and check every source page has a clean result."""
    page_count = len(PdfReader(path).pages)
    payload = {
        "file": base64.b64encode(path.read_bytes()).decode("ascii"),
        "fileType": 0,
        "useLayoutDetection": True,
        "markdownIgnoreLabels": [],
        "returnMarkdownImages": False,
        "visualize": False,
        "restructurePages": False,
    }
    peak = [gpu_memory()]
    finished = threading.Event()

    def sample_gpu() -> None:
        """Record peak GPU memory while the request runs."""
        while not finished.wait(1):
            peak[0] = max(peak[0], gpu_memory())

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    start = time.monotonic()
    try:
        response = requests.post(
            endpoint + "/layout-parsing", json=payload, timeout=timeout
        )
    finally:
        finished.set()
        sampler.join()
    elapsed = time.monotonic() - start
    response.raise_for_status()
    body = response.json()
    assert body["errorCode"] == 0, body.get("errorMsg")
    result = body["result"]
    assert result["dataInfo"]["numPages"] == page_count
    assert len(result["dataInfo"]["pages"]) == page_count
    pages = result["layoutParsingResults"]
    assert len(pages) == page_count
    for page in pages:
        assert isinstance(page["prunedResult"], dict)
        assert isinstance(page["markdown"]["text"], str)
        assert page["markdown"].get("images") in (None, {})
        assert page.get("inputImage") in (None, "")
        assert page.get("outputImages") in (None, {})
        assert page.get("exports") in (None, {})
    texts = [page["markdown"]["text"] for page in pages]
    if path.name == "multipage.pdf":
        for text, marker in zip(texts, ("FIRST", "MIDDLE", "LAST")):
            assert marker in text.upper(), marker
        assert "CAFÉ" in " ".join(texts).upper()
        assert "FOOTER" in " ".join(texts).upper()
    elif path.name == "table.pdf":
        text = texts[0].upper()
        for marker in ("TABLE", "QUARTER", "VALUE", "Q1", "42", "E = MC"):
            assert marker in text, marker
    elif path.name == "blank-middle.pdf":
        assert "BEFORE" in texts[0].upper()
        assert not texts[1].strip()
        assert "AFTER" in texts[2].upper()
    elif path.name == "scanned.pdf":
        assert "THIS TEXT IS EASY TO EXTRACT" in texts[0].upper().replace("\n", " ")
    elif path.name == "long-41.pdf":
        for index in (0, 20, 40):
            assert f"PAGE {index + 1:02d}" in texts[index].upper()
    print(
        f"{path.name}: {page_count} pages, {elapsed:.1f}s, peak GPU {peak[0]} MiB, OK"
    )


def main() -> None:
    """Generate samples or check the full parsing API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--endpoint", default="http://localhost:8108")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--samples", type=Path, default=Path("examples/paddle"))
    parser.add_argument("pdfs", type=Path, nargs="*")
    args = parser.parse_args()
    if args.generate:
        make_samples(args.samples)
        return
    health = requests.get(args.endpoint + "/health", timeout=10)
    health.raise_for_status()
    paths = args.pdfs or [
        args.samples / name
        for name in (
            "multipage.pdf",
            "table.pdf",
            "scanned.pdf",
            "blank-middle.pdf",
            "long-41.pdf",
        )
    ]
    for path in paths:
        check_pdf(args.endpoint, path, args.timeout)
    invalid = requests.post(
        args.endpoint + "/layout-parsing",
        json={"file": base64.b64encode(b"not a PDF").decode("ascii"), "fileType": 0},
        timeout=30,
    )
    assert invalid.status_code >= 400 or invalid.json().get("errorCode") != 0
    print(f"invalid PDF: HTTP {invalid.status_code}, rejected")


if __name__ == "__main__":
    main()
