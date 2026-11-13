import tempfile
import unittest
from pathlib import Path

from syncytia_tools.health import csv_inventory


class CsvInventoryTests(unittest.TestCase):
    def test_counts_csv_files_and_nonempty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "curve.csv").write_text("time,value\n0,1\n1,2\n", encoding="utf-8")
            self.assertEqual(csv_inventory(root), {"csvFiles": 1, "dataRows": 2})

    def test_rejects_empty_fixture_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no CSV fixtures"):
                csv_inventory(Path(directory))


if __name__ == "__main__":
    unittest.main()
