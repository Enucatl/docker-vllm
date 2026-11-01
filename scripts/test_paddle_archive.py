"""Verify opt-in export uses corrected pixels and preserves default responses."""

import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from paddle_archive import patch_serving


@pytest.fixture
def patched_api(tmp_path):
    """Exercise the extension against the pinned handler's source structure."""
    schema = tmp_path / "inference/serving/schemas/paddleocr_vl.py"
    app = tmp_path / "inference/serving/basic_serving/_pipeline_apps/paddleocr_vl.py"
    schema.parent.mkdir(parents=True)
    app.parent.mkdir(parents=True)
    schema.write_text(
        "from typing import Optional\n"
        "from pydantic import BaseModel, Field, field_validator, model_validator\n"
        "class OutputFormatsMixin(BaseModel): pass\n"
        "class ocr:\n    BaseInferRequest = BaseModel\n"
        "class InferRequest(OutputFormatsMixin, ocr.BaseInferRequest):\n"
        "    pass\n"
        "class LayoutParsingResult(BaseModel):\n"
        "    prunedResult: dict\n"
    )
    app.write_text(
        "from typing import Any, Dict, List\n"
        "def handle(request, item):\n"
        "    layout_parsing_results = []\n"
        "    if True:\n"
        "        if True:\n"
        "            entry = {'prunedResult': {}}\n"
        "            layout_parsing_results.append(entry)\n"
        "    return layout_parsing_results[0]\n"
    )
    patch_serving(tmp_path)
    first = (schema.read_text(), app.read_text())
    patch_serving(tmp_path)
    assert first == (schema.read_text(), app.read_text())
    models, handler = {}, {}
    exec(compile(schema.read_text(), str(schema), "exec"), models)  # noqa: S102
    exec(compile(app.read_text(), str(app), "exec"), handler)  # noqa: S102
    return models, handler["handle"]


def test_default_response_unchanged(patched_api):
    """Default requests neither inspect image results nor emit new fields."""
    models, handle = patched_api
    request = models["InferRequest"]()
    assert request.includeArchiveImages is False
    entry = handle(request, {})
    assert models["LayoutParsingResult"](**entry).model_dump() == {"prunedResult": {}}


def test_export_preserves_clean_corrected_pixels(patched_api):
    """PNG colors and dimensions match the actual output, not visualizations."""
    models, handle = patched_api
    corrected = np.zeros((13, 7, 3), dtype=np.uint8)
    corrected[:6] = [12, 34, 250]
    result = {"doc_preprocessor_res": {"output_img": corrected}}
    entry = handle(SimpleNamespace(includeArchiveImages=True), result)
    serialized = models["LayoutParsingResult"](**entry).model_dump()
    png = base64.b64decode(serialized["archiveImage"], validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, corrected)


def test_missing_corrected_page_fails(patched_api):
    """Never substitute an original page or a comparison visualization."""
    _, handle = patched_api
    with pytest.raises(KeyError, match="doc_preprocessor_res"):
        handle(SimpleNamespace(includeArchiveImages=True), {})


def test_incompatible_upstream_fails(tmp_path):
    """Reject an unsupported image before changing either installed module."""
    schema = tmp_path / "inference/serving/schemas/paddleocr_vl.py"
    schema.parent.mkdir(parents=True)
    schema.write_text("incompatible upstream")
    with pytest.raises(RuntimeError, match="Pinned PaddleX source changed"):
        patch_serving(tmp_path)
    assert schema.read_text() == "incompatible upstream"
