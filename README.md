# QuackQuery

A university document assistant with cited answers and isolated knowledge bases.
Built with Streamlit, ChromaDB, Sentence Transformers, and Groq.

## Current capabilities

- Recursively extracts PDF and DOCX files, including DOCX table rows.
- Keeps the 20-document synthetic demo separate from the original, unverified documents.
- Shares one semantic embedding configuration between ingestion and retrieval; no hash fallback.
- Builds a fresh collection before publishing its name through an atomic manifest replacement.
  Removed files and obsolete chunks disappear from the active snapshot after successful ingestion.
- Records source paths, content hashes, publication dates when explicitly labeled, and source type.
- Shows document locations without inventing DOCX page numbers.
- Escapes source and generated content before rendering HTML.

## Run locally

Use Python 3.10 or newer with dependencies from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m src.ingest --dry-run
python -m src.ingest --corpus synthetic
python -m src.ingest --corpus unverified
streamlit run app.py
```

Set `GROQ_API_KEY` in a local `.env` file. The first ingestion may download
`all-MiniLM-L6-v2`. Select a knowledge base in the sidebar.
The default is the fictional synthetic demo, not official university guidance.
The original documents have not been verified against official sources.

`python src/ingest.py` also works. Paths are relative to the repository, not the shell directory.
PDF takes precedence over a same-name DOCX in the same folder.
Folders containing `synthetic` in their relative path belong to the synthetic corpus;
keep all synthetic test documents under the supplied synthetic folder.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover source isolation, duplicate representations, DOCX provenance,
short passages, snapshot replacement, and preserving the active index on extraction failure.
Snapshot tests use deterministic test vectors; they do not measure semantic retrieval quality.

## Storage and limitations

`chroma_db/quackquery.json` contains the active collection names and source inventory.
Existing legacy collections are not modified. Prior snapshots are retained for recovery
and currently require manual cleanup. Run only one ingestion process at a time.
An empty or unreadable corpus does not replace an existing index.
Scanned PDFs require a future OCR stage. DOCX extraction handles body paragraphs and
tables, not headers, footers, embedded images, or pagination.
Conflict detection remains prompt-based and is not guaranteed.

## Next milestones

1. Review a labeled evaluation set and record retrieval quality and latency.
2. Compare keyword/semantic hybrid retrieval and reranking against the baseline.
3. Add structured answers, citation validation, and a source viewer.
4. Add conversational follow-ups and explicit academic-year/program metadata.
5. Add deployment automation and operational monitoring.

## License

See [LICENSE](LICENSE).
