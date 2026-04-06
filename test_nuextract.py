#!/usr/bin/env python3
"""Test NuExtract-2.0-8B-GPTQ structured extraction via local vLLM endpoint."""

import json
import time
import sys
from openai import OpenAI

# Configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8103/v1"
MODEL = "numind/NuExtract-2.0-8B-GPTQ"

# Define extraction schema
EXTRACTION_SCHEMA = {
    "title_summarizing_subject_clear_concise_descriptive": "string",
    "document_date": "date-time",
    "issuing_organization_or_sender": "string"
}

# Sample document
SAMPLE_DOCUMENT = """RJR INTER-OFFICE MEMORANDUM

Subject: Weekly Highlights
Date: August 24, 1983
To: Mr. J. D. Phillips
From: J. P. Wheeler
Materials Technology

ITEM FOR WEEKLY BRIEF

1. Cigarette Inks
A new cigarette monogram ink has been implemented on CAMEL Filter cigarettes being run on the Protos Makers in Manufacturing.

2. Coresta Porosity Measurements
A study has been completed on a method to standardize the Phobos meters used by RJR to measure porosity of paper materials.

GENERAL ITEMS

1. Mobil Films' Plant Tour
A visit was made by personnel in Materials Technology Division to Mobil Films in Macedon, N. Y.

2. Intermediate Plug Wraps
Two intermediate plug wraps will be tested on CAMEL Filter as a vehicle for 'tar' reduction.

3. Salem Ultra Light 100
Salem Ultra Light 100 has recently experienced high 'tar' and high ventilation variability.

xc: Applied R&D Managers

J.P. Wheeler
50492 3009"""


def wait_for_service(max_retries=30, delay=2):
    """Wait for the service to be ready."""
    import requests

    url = f"{OPENAI_API_BASE.replace('/v1', '')}/health"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ Service is ready at {OPENAI_API_BASE}")
                return True
        except requests.RequestException:
            pass

        if attempt < max_retries - 1:
            print(f"  Waiting for service... ({attempt + 1}/{max_retries})")
            time.sleep(delay)

    print(f"✗ Service did not become ready after {max_retries * delay}s")
    return False


def extract_with_nuextract(document: str, schema: dict) -> dict:
    """Send extraction request to NuExtract using OpenAI client."""

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )

    # Format the schema as a JSON string for the chat template
    schema_str = json.dumps(schema, indent=4)

    print("\n📤 Sending request to NuExtract...")
    print(f"   Schema fields: {list(schema.keys())}")
    print(f"   Document length: {len(document)} chars")

    chat_response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": document}],
            },
        ],
        extra_body={
            "chat_template_kwargs": {
                "template": schema_str
            },
        }
    )

    # Extract response
    if chat_response.choices and len(chat_response.choices) > 0:
        content = chat_response.choices[0].message.content
        try:
            # Try to parse as JSON
            extracted = json.loads(content)
            return extracted
        except json.JSONDecodeError:
            return {"raw_response": content}

    return {"error": "No response from model"}


def main():
    print("=" * 60)
    print("NuExtract-2.0-8B-GPTQ Structured Extraction Test")
    print("=" * 60)

    # Wait for service
    if not wait_for_service():
        return 1

    try:
        # Run extraction
        result = extract_with_nuextract(SAMPLE_DOCUMENT, EXTRACTION_SCHEMA)

        print("\n📥 Extraction Result:")
        print(json.dumps(result, indent=2))

        # Verify expected fields
        if "raw_response" not in result and "error" not in result:
            print("\n✓ Extracted Fields:")
            for field, value in result.items():
                print(f"  {field}: {value}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
