import json
from pathlib import Path
import tempfile
import unittest

from AADL.experiments.aggregate import (
    bootstrap_interval,
    load_records,
    summarize,
    write_csv,
    write_latex,
    write_markdown,
)


class AggregationTests(unittest.TestCase):
    def _record(self, method, seed, loss):
        return {
            "schema_version": 1,
            "status": "completed",
            "total_seconds": 2.0 + seed,
            "summary": {"loss": loss},
            "config": {
                "family": "controlled", "workload": "quadratic",
                "seed": seed, "method": {"name": method},
            },
        }

    def test_bootstrap_is_reproducible_and_contains_mean(self):
        first = bootstrap_interval([1, 2, 3, 4], samples=500, seed=4)
        second = bootstrap_interval([1, 2, 3, 4], samples=500, seed=4)
        self.assertEqual(first, second)
        self.assertLess(first[0], 2.5)
        self.assertGreater(first[1], 2.5)

    def test_load_summarize_and_write_all_table_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [self._record("plain", 0, 2.0), self._record("plain", 1, 1.0)]
            for index, record in enumerate(records):
                (root / f"run-{index}.json").write_text(json.dumps(record))
            (root / "invalid.json").write_text("not json")
            loaded = load_records([root])
            rows = summarize(
                loaded, ["config.method.name"], {"loss": "summary.loss"},
                samples=200,
            )
            self.assertEqual(len(loaded), 2)
            self.assertEqual(rows[0]["runs"], 2)
            self.assertEqual(rows[0]["loss_mean"], 1.5)
            for writer, suffix in (
                (write_csv, "csv"), (write_markdown, "md"), (write_latex, "tex")
            ):
                target = root / f"summary.{suffix}"
                writer(rows, target)
                self.assertTrue(target.read_text().strip())


if __name__ == "__main__":
    unittest.main()
