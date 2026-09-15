import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from AADL.experiments import ExperimentConfig, create_workload, list_workloads
from AADL.experiments.__main__ import _override
from AADL.experiments.benchmarks import benchmark_anderson_kernel
from AADL.experiments.runner import run_experiment
from AADL.experiments.workloads._common import batches, partition_indices


class TestExperimentConfig(unittest.TestCase):
    def test_nested_configuration_round_trip(self):
        source = {
            "family": "controlled",
            "workload": "controlled.quadratic",
            "epochs": 2,
            "method": {"optimizer": "adam", "learning_rate": 0.001},
            "execution": {"device": "cpu"},
        }
        config = ExperimentConfig.from_dict(source)
        self.assertEqual(config.method.optimizer, "adam")
        self.assertEqual(config.to_dict()["execution"]["device"], "cpu")

    def test_dotted_cli_override_parses_json_values(self):
        data = {"method": {"name": "old"}}
        _override(data, "method.name=new")
        _override(data, "execution.distributed=true")
        _override(data, "method.acceleration={}")
        self.assertEqual(data["method"]["name"], "new")
        self.assertTrue(data["execution"]["distributed"])
        self.assertEqual(data["method"]["acceleration"], {})

    def test_invalid_optimizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported optimizer"):
            ExperimentConfig.from_dict({
                "family": "controlled", "workload": "controlled.quadratic",
                "method": {"optimizer": "made-up"},
            })

    def test_all_checked_in_paper_configs_validate(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "examples" / "paper" / "configs").glob("*.json"))
        self.assertGreaterEqual(len(paths), 9)
        for path in paths:
            with self.subTest(path=path.name):
                ExperimentConfig.from_dict(json.loads(path.read_text()))


class TestWorkloadRegistry(unittest.TestCase):
    def test_all_reference_families_are_registered(self):
        self.assertEqual(
            set(list_workloads()),
            {
                "controlled.quadratic", "graph.synthetic",
                "graph.ogbg-molhiv", "transformer.glue-sst2",
                "transformer.synthetic", "transformer.wikitext103",
                "vision.cifar10", "vision.cifar100", "vision.imagenet",
                "vision.synthetic",
            },
        )

    def test_family_mismatch_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig.from_dict({
                "family": "vision", "workload": "controlled.quadratic",
                "output": {"directory": directory},
            })
            with self.assertRaisesRegex(ValueError, "belongs to 'controlled'"):
                run_experiment(config)

    def test_unknown_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown workload options"):
            create_workload("vision.synthetic", {"typo": 1})

    def test_rank_partition_is_disjoint_and_equal_length(self):
        x = torch.arange(24).view(12, 2)
        y = torch.arange(12)
        rank0 = batches(x, y, 2, seed=3, epoch=0, rank=0, world_size=2)
        rank1 = batches(x, y, 2, seed=3, epoch=0, rank=1, world_size=2)
        labels0 = torch.cat([batch[1] for batch in rank0])
        labels1 = torch.cat([batch[1] for batch in rank1])
        self.assertEqual(len(rank0), len(rank1))
        self.assertTrue(set(labels0.tolist()).isdisjoint(labels1.tolist()))

    def test_dirichlet_partition_is_reproducible_disjoint_and_balanced(self):
        labels = torch.arange(120).remainder(3)
        shards = [
            partition_indices(labels, rank, 3, 11, 0, "dirichlet", 0.3)
            for rank in range(3)
        ]
        self.assertEqual(len({len(shard) for shard in shards}), 1)
        self.assertEqual(shards[0].tolist(), partition_indices(
            labels, 0, 3, 11, 0, "dirichlet", 0.3,
        ).tolist())
        self.assertEqual(len(set(shards[0].tolist()) & set(shards[1].tolist())), 0)


class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.random.get_rng_state()

    def tearDown(self):
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.random.set_rng_state(self.torch_rng)

    def test_plain_and_accelerated_runs_write_structured_records(self):
        for acceleration in ({}, {
            "acceleration_type": "anderson",
            "wait_iterations": 0,
            "history_depth": 3,
            "frequency": 1,
            "reg_acc": 1e-7,
            "sketch_fraction": 0.5,
            "sketch_policy": "backward_error",
        }):
            with self.subTest(accelerated=bool(acceleration)):
                with tempfile.TemporaryDirectory() as directory:
                    config = ExperimentConfig.from_dict({
                        "family": "controlled",
                        "workload": "controlled.quadratic",
                        "epochs": 1,
                        "workload_options": {
                            "dimension": 8, "samples": 16, "batch_size": 4,
                        },
                    "method": {
                        "learning_rate": 0.01,
                        "scheduler": {"name": "step", "step_size": 1, "gamma": 0.5},
                        "acceleration": acceleration,
                        },
                        "execution": {"device": "cpu"},
                        "output": {"directory": directory},
                    })
                    record, path = run_experiment(config)
                    self.assertEqual(record["status"], "completed")
                    self.assertEqual(record["epochs"][0]["steps"], 4)
                    self.assertEqual(record["epochs"][0]["learning_rates"], [0.01])
                    self.assertTrue(path.exists())
                    persisted = json.loads(path.read_text())
                    self.assertEqual(persisted["schema_version"], 1)
                    self.assertEqual(persisted["config"]["seed"], 0)
                    if acceleration:
                        self.assertTrue(persisted["controller"])

    def test_each_neural_family_executes(self):
        for family in ("vision", "graph", "transformer"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig.from_dict({
                    "family": family,
                    "workload": f"{family}.synthetic",
                    "workload_options": {"samples": 8, "batch_size": 4},
                    "output": {"directory": directory, "save_trace": False},
                })
                record, _ = run_experiment(config)
                self.assertEqual(record["status"], "completed")
                self.assertIn("loss", record["summary"])

    def test_kernel_benchmark_returns_normalized_timing(self):
        result = benchmark_anderson_kernel(
            parameter_count=32, history_depth=4, repeats=2,
            sketch_fraction=0.5,
        )
        self.assertEqual(result["repeats"], 2)
        self.assertGreater(result["seconds_per_call"], 0)

    def test_no_gradient_loss_path_is_compatible_with_safeguards(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig.from_dict({
                "family": "controlled", "workload": "controlled.quadratic",
                "workload_options": {
                    "dimension": 4, "samples": 12, "batch_size": 4,
                },
                "method": {"acceleration": {
                    "acceleration_type": "anderson", "history_depth": 3,
                    "wait_iterations": 0, "frequency": 1,
                }},
                "output": {"directory": directory},
            })
            record, _ = run_experiment(config)
            self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()
