import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src import ingest


def docx(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>' + text + '</w:t></w:r></w:p></w:body></w:document>')


class IngestionTests(unittest.TestCase):
    def test_isolation_and_duplicate_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docx(root / "synthetic" / "test.docx", "Test policy.")
            docx(root / "original.docx", "Original policy.")
            (root / "original.pdf").touch()
            self.assertEqual(len(ingest.discover_documents("synthetic", root)), 1)
            self.assertEqual([p.name for p in ingest.discover_documents("unverified", root)], ["original.pdf"])

    def test_docx_provenance_and_short_passages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docx(root / "synthetic" / "test.docx", "Publication date: September 1, 2026")
            ids, docs, metas, inventory = ingest.prepare_corpus("synthetic", root)
            self.assertTrue(docs)
            self.assertEqual(metas[0]["locator"], "Document text")
            self.assertEqual(metas[0]["source_type"], "synthetic_test_data")
            self.assertEqual(metas[0]["publication_date"], "September 1, 2026")
            self.assertEqual(len(ingest.split_text("Short requirement.")), 1)

    def test_snapshot_replacement_and_failure_preserve_active_index(self):
        import chromadb
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "index.json"
            def read():
                return json.loads(manifest.read_text()) if manifest.exists() else {}
            records = (["old", "kept"], ["old text", "new text"], [{"filename": "a"}] * 2, [])
            with patch.object(ingest, "CHROMA_DIR", root / "db"), patch.object(ingest, "MANIFEST", manifest), patch.object(ingest, "read_manifest", read), patch.object(ingest, "embed_batch", side_effect=lambda texts: [[1.0, 0.0] for _ in texts]), patch.object(ingest, "prepare_corpus", return_value=records) as prepare:
                ingest.ingest_documents()
                first = read()["synthetic"]["collection"]
                prepare.return_value = (["kept"], ["new text"], [{"filename": "a"}], [])
                ingest.ingest_documents()
                second = read()["synthetic"]["collection"]
                self.assertNotEqual(first, second)
                client = chromadb.PersistentClient(path=str(root / "db"))
                self.assertEqual(client.get_collection(second).get()["ids"], ["kept"])
                prepare.side_effect = ValueError("Unreadable document")
                with self.assertRaises(ValueError):
                    ingest.ingest_documents()
                self.assertEqual(read()["synthetic"]["collection"], second)
                client._system.stop()


if __name__ == "__main__":
    unittest.main()
