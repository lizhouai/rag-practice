# Dataset Sources

This practice project now includes a larger seed knowledge base under `data/raw/` and a script for importing open customer-support rows into Markdown casebooks. The checked-in seed documents are intentionally Chinese-first, with only three English support casebooks kept for cross-lingual retrieval tests.

## English Source Kept For Cross-Lingual Tests

- Dataset: `rjac/e-commerce-customer-support-qa`
- Host: Hugging Face Datasets
- License: MIT
- Size: 1,000 rows
- Format: Parquet
- Why keep it: it is close to the practice domain, includes realistic customer-support conversations, has issue taxonomy fields, and uses a permissive license suitable for a production-style practice project.

Import script:

```bash
pip install -r requirements-dataset.txt
python scripts/import_customer_support_dataset.py --limit 120 --rows-per-doc 12
```

This import step is optional and requires network access to Hugging Face from Git Bash/Python. The script first tries `datasets.load_dataset()` and then falls back to the Hugging Face Dataset Viewer rows API. If your browser can open Hugging Face but the script still fails, Git Bash/Python may be unable to access parquet/xet file downloads or `datasets-server.huggingface.co`; configure `HTTPS_PROXY` / `HTTP_PROXY` or the `HF_ENDPOINT` used by your environment before rerunning it.

Only three checked-in English BrownBox casebooks are retained: account login, returns/replacement, and COD refund. To use real downloaded rows, run the importer above; it will generate Markdown documents with `dataset_source` and `dataset_license` metadata.

## Chinese Seed Resources

Most checked-in long documents are Chinese support resources maintained directly in this repository. They cover logistics exceptions, invoice and tax handling, marketplace seller boundaries, product recall and safety, promotions and loyalty, warranty repair, public service rules, partner support collaboration, finance reconciliation, and account-risk operations.

The seed set intentionally includes multiple `permission_scope` values:

- `public`: consumer-facing baseline rules.
- `internal`: normal internal support knowledge.
- `partner_support`: partner and merchant collaboration material.
- `finance_restricted`: refund reconciliation and finance operations.
- `security_restricted`: account-risk and security operations.

These Chinese seed documents are not copied rows from a third-party dataset. They are written as production-like support knowledge and are easier to use in a Chinese公众号/RAG teaching repository without importing non-commercial-license data by default.

## Other Sources Considered

- `PaDaS-Lab/webfaq`: very large multilingual FAQ collection, CC BY 4.0. Useful for open-domain retrieval, but downstream users must check original website terms and Common Crawl constraints.
- `OpenStellarTeam/Chinese-EcomQA`: Chinese e-commerce QA benchmark with 1,800 rows. Useful for Chinese e-commerce evaluation, but the license is CC BY-NC-SA 4.0, so it is less convenient for production or commercial practice defaults.
- `JDDC 2.0`: large-scale Chinese e-commerce customer-service dialogue research corpus. Useful as a reference direction for Chinese customer-service RAG, but it is better treated as an external research resource unless the downstream project has verified its exact distribution and usage constraints.
