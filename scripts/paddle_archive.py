"""Add opt-in corrected-page PNG export to the pinned PaddleX serving API."""

import importlib.util
import inspect
from pathlib import Path


def encode_archive_image(result) -> str:
    """Encode the preprocessor's actual output, preserving its BGR pixels."""
    import base64

    import cv2

    image = result["doc_preprocessor_res"]["output_img"]
    success, png = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode corrected page as PNG")
    return base64.b64encode(png).decode("ascii")


def replace_once(source: str, old: str, new: str) -> str:
    """Reject incompatible upstream changes rather than silently losing export."""
    if new in source:
        return source
    if source.count(old) != 1:
        raise RuntimeError(f"Pinned PaddleX source changed: {old!r}")
    return source.replace(old, new, 1)


def patch_serving(package: Path) -> None:
    """Extend the request and response only for explicitly requested images."""
    schema_path = package / "inference/serving/schemas/paddleocr_vl.py"
    app_path = (
        package / "inference/serving/basic_serving/_pipeline_apps/paddleocr_vl.py"
    )
    schema = schema_path.read_text()
    schema = replace_once(
        schema,
        "from pydantic import BaseModel, Field, field_validator, model_validator",
        "from pydantic import BaseModel, Field, field_validator, model_validator, model_serializer",
    )
    schema = replace_once(
        schema,
        "class InferRequest(OutputFormatsMixin, ocr.BaseInferRequest):\n",
        "class InferRequest(OutputFormatsMixin, ocr.BaseInferRequest):\n"
        "    includeArchiveImages: bool = False\n",
    )
    schema = replace_once(
        schema,
        "class LayoutParsingResult(BaseModel):\n",
        "class LayoutParsingResult(BaseModel):\n"
        "    archiveImage: Optional[str] = None\n\n"
        '    @model_serializer(mode="wrap")\n'
        "    def serialize_archive_image(self, handler):\n"
        "        data = handler(self)\n"
        "        if self.archiveImage is None:\n"
        '            data.pop("archiveImage", None)\n'
        "        return data\n\n",
    )
    app = replace_once(
        app_path.read_text(),
        "from typing import Any, Dict, List\n",
        "from typing import Any, Dict, List\n\n\n"
        + inspect.getsource(encode_archive_image)
        + "\n",
    )
    app = replace_once(
        app,
        "            layout_parsing_results.append(entry)\n",
        "            if request.includeArchiveImages:\n"
        '                entry["archiveImage"] = encode_archive_image(item)\n'
        "            layout_parsing_results.append(entry)\n",
    )
    schema_path.write_text(schema)
    app_path.write_text(app)


if __name__ == "__main__":
    patch_serving(Path(importlib.util.find_spec("paddlex").origin).parent)
