from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers import make_chunk, rag


class DatasetImportTest(unittest.TestCase):
    def load_importer_module(self):
        script_path = PROJECT_ROOT / "scripts" / "import_customer_support_dataset.py"
        spec = importlib.util.spec_from_file_location("customer_support_importer", script_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_customer_support_dataset_rows_render_to_long_markdown_documents(self) -> None:
        module = self.load_importer_module()

        rows = [
            {
                "issue_area": "Cancellations and returns",
                "issue_category": "Cash on Delivery (CoD) Refunds",
                "issue_sub_category": "Refund timelines for Cash on Delivery returns",
                "product_category": "Appliances",
                "product_sub_category": "Water Purifier",
                "issue_complexity": "medium",
                "conversation": "Agent: Hello. Customer: When will my refund arrive? Agent: Refunds are processed after pickup and quality check.",
                "qa": json.dumps(
                    {
                        "knowledge": [
                            {
                                "customer_summary_question": "When will my refund arrive?",
                                "agent_summary_solution": "Refunds are processed after pickup and quality check.",
                            }
                        ]
                    }
                ),
            }
        ]

        docs = module.rows_to_markdown_documents(rows, rows_per_doc=1)

        self.assertEqual(len(docs), 1)
        self.assertIn("dataset_source: rjac/e-commerce-customer-support-qa", docs[0].content)
        self.assertIn("## Case 1: Refund timelines for Cash on Delivery returns", docs[0].content)
        self.assertIn("Refunds are processed after pickup and quality check.", docs[0].content)
        self.assertGreater(len(docs[0].content), 700)

    def test_huggingface_download_error_mentions_optional_import_and_proxy(self) -> None:
        module = self.load_importer_module()

        message = module.format_huggingface_download_error(
            RuntimeError("LocalEntryNotFoundError"),
            RuntimeError("datasets-server unavailable"),
        )

        self.assertIn("optional Hugging Face dataset", message)
        self.assertIn("does not block the main production RAG experiment", message)
        self.assertIn("Dataset Viewer fallback", message)
        self.assertIn("HTTPS_PROXY", message)
        self.assertIn("HF_ENDPOINT", message)

    def test_dataset_viewer_rows_api_can_load_fallback_rows(self) -> None:
        module = self.load_importer_module()
        captured_urls: list[str] = []

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "rows": [
                            {
                                "row": {
                                    "issue_area": "Order",
                                    "issue_category": "Delivery",
                                    "issue_sub_category": "Late delivery",
                                    "conversation": "Agent: hello",
                                    "qa": "{\"knowledge\": []}",
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(url: str, timeout: int) -> FakeResponse:
            captured_urls.append(url)
            return FakeResponse()

        with patch("urllib.request.urlopen", fake_urlopen):
            rows = module.load_dataset_viewer_rows(limit=1)

        self.assertEqual(rows[0]["issue_area"], "Order")
        self.assertIn("datasets-server.huggingface.co/rows", captured_urls[0])
        self.assertIn("dataset=rjac%2Fe-commerce-customer-support-qa", captured_urls[0])


if __name__ == "__main__":
    unittest.main()
