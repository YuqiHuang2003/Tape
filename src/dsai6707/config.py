"""Configuration loading, resolution and strict experiment checks."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .io import write_json

LOCKED_TASK_DATASETS: dict[str, dict[str, str]] = {
    "arc_easy": {
        "hub_repository": "allenai/ai2_arc",
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
    },
    "openbookqa": {
        "hub_repository": "allenai/openbookqa",
        # This is the dataset_path in lm-eval 0.4.8's official task YAML.
        "path": "openbookqa",
        "name": "main",
        "split": "test",
        "revision": "388097ea7776314e93a529163e0fea805b8a6454",
    },
}

FORMAL_EVALUATION_PROTOCOL_ID = "formal_main_arc_obqa_v1"
SUPPLEMENTAL_EVALUATION_CONFIG_SCHEMA = "supplemental-evaluation-config-v1"
SUPPLEMENTAL_EVALUATION_PROTOCOL_ID = "post_hoc_binary_validation_v1"
LOCKED_SUPPLEMENTAL_TASK_DATASETS: dict[str, dict[str, Any]] = {
    "winogrande": {
        "hub_repository": "allenai/winogrande",
        # These path/name values are from lm-eval 0.4.8's official task YAML.
        "path": "winogrande",
        "name": "winogrande_xl",
        "split": "validation",
        "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "output_type": "multiple_choice",
    },
    "piqa": {
        "hub_repository": "ybisk/piqa",
        "path": "piqa",
        "name": None,
        "split": "validation",
        "revision": "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        "output_type": "multiple_choice",
    },
}
LOCKED_SUPPLEMENTAL_EXPECTED_SAMPLES = {"winogrande": 1267, "piqa": 1838}
LOCKED_SUPPLEMENTAL_CHANCE_ACCURACY = {"winogrande": 0.5, "piqa": 0.5}
LOCKED_SUPPLEMENTAL_CONTENT_EVIDENCE = {
    "winogrande": {
        "huggingface_fingerprint": "d86221ec28ba3c4f",
        "content_sha256": "55e62bfddc453259e604a1fadd4556c608eaf0b1ff7b3897b2054b862a50f088",
        "first_row_sha256": "2a63f8d827c33439b8820402049e51b90ba58cc135211c9f52d0f520094c087e",
        "last_row_sha256": "5fab81ad6dfe400c4207806fa9ebdd10d154dc7e2d72ad71866ec25f050da142",
        "feature_schema_sha256": "dd44b45d375690cdc95573e0ba3641b043edf10f927d08ebecf7315fa65ab25f",
    },
    "piqa": {
        "huggingface_fingerprint": "c305b46a7e84e936",
        "content_sha256": "d17e130340a3db225953e915a900d3ea080daf86a83d5a94350d84d3011f08e5",
        "first_row_sha256": "cd8ff9842262a1c4127a3fa427d3e4b6c2e823fe65e9438ec681ed3562cf8383",
        "last_row_sha256": "ee6fe00e1f3705670320283f32d65d2c3c612380bfa74e405c16b90cf9b8c119",
        "feature_schema_sha256": "9d2b1f970a62ebb75366040272bcfdc629b68f5020b003945a2cb47bf3cdfbbd",
    },
}

FORMAL_MAIN_STUDY_TYPE = "formal_main"
COMPRESSION_ABLATION_STUDY_TYPE = "compression_ablation"
CAPACITY_TWO_ABLATION_STUDY_TYPE = "capacity_two_ablation"
SUPPORTED_STUDY_TYPES = frozenset(
    {
        FORMAL_MAIN_STUDY_TYPE,
        COMPRESSION_ABLATION_STUDY_TYPE,
        CAPACITY_TWO_ABLATION_STUDY_TYPE,
    }
)
PINNED_GROUPING_SOURCE_REFERENCE = (
    "https://github.com/wazenmai/HC-SMoE/blob/"
    "bd77eba1b55ca58606a47e2d0807a24218da73c3/scripts/README.md"
)
PINNED_CAPACITY_TWO_SOURCE_REFERENCE = (
    "https://github.com/networkx/networkx/blob/networkx-3.4.2/" "networkx/algorithms/matching.py"
)
CAPACITY_TWO_STRATEGY = "capacity_two_minimum_weight_matching"
CAPACITY_TWO_ALGORITHM = "networkx_blossom_exact_float64_v1"
CAPACITY_TWO_NETWORKX_VERSION = "3.4.2"
CAPACITY_TWO_DISTANCE_ENCODING = "ieee754_exact_common_denominator"
CAPACITY_TWO_TIE_BREAK = "unique_binary_edge_rank"
CAPACITY_TWO_EA_PROTOCOL_PATH = "docs/CAPACITY2_EA_EXTENSION_PROTOCOL.md"
CAPACITY_TWO_EA_PROTOCOL_SHA256 = "0288ffc9079c1a0e1cd6ead3d74b52b1e8e6961fd57f337d98be7ca3fd0736af"
CAPACITY_TWO_EA_PILOT_SCOPE = "one_seed_pilot"
CAPACITY_TWO_EA_PILOT_PROTOCOL_PATH = "docs/G16_EA_ONE_SEED_PILOT_PROTOCOL.md"
CAPACITY_TWO_EA_PILOT_PROTOCOL_SHA256 = (
    "4c3fd60b41b52cf0ea1cf1caa74303630e7a8ff7ec7175984af807cc02da7f89"
)
CAPACITY_TWO_EA_PILOT_PROJECT_NAME = "granite_capacity2_g16_ea_pilot_seed6707_v2"
CAPACITY_TWO_EA_PILOT_RUN_DIR = "results/runs/granite_capacity2_g16_ea_pilot_seed6707_v2"
CAPACITY_TWO_EA_PILOT_SELECTION = {
    "metric_order": ["validation_nll", "best_fitness_nll", "strategy_order"],
    "strategy_order": ["coordinate", "layer_block"],
    "comparator": "exact_lexicographic_ascending",
    "evaluate_only_winner": True,
    "forbid_downstream_metric_selection": True,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(payload)
    return payload


def load_supplemental_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load the independent, post-hoc downstream-evaluation protocol."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("supplemental evaluation configuration root must be a mapping")
    validate_supplemental_evaluation_config(payload)
    return payload


def _require(mapping: dict[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {location}.{key}")
    return mapping[key]


def validate_config(config: dict[str, Any]) -> None:
    for section in (
        "project",
        "model",
        "data",
        "grouping",
        "fusion",
        "search",
        "evaluation",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing or invalid configuration section: {section}")

    project = config["project"]
    study_type = project.get("study_type", FORMAL_MAIN_STUDY_TYPE)
    if not isinstance(study_type, str) or study_type not in SUPPORTED_STUDY_TYPES:
        raise ValueError(f"unknown project.study_type: {study_type!r}")

    is_synthetic = str(config["data"].get("dataset", "")) == "synthetic"

    model = config["model"]
    for key in ("id", "revision", "tokenizer_revision", "transformers_version"):
        value = str(_require(model, key, "model")).strip()
        if not value or "REQUIRED" in value.upper():
            raise ValueError(f"model.{key} must be explicitly pinned")
    if str(model["transformers_version"]) != "4.46.3":
        raise ValueError("the Granite adapter is validated only for transformers 4.46.3")
    if not is_synthetic:
        if model["id"] != "ibm-granite/granite-3.1-1b-a400m-base":
            raise ValueError("the formal protocol is locked to Granite-3.1-1B-A400M-Base")
        locked_model_revision = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
        for key in ("revision", "tokenizer_revision"):
            if str(model[key]) != locked_model_revision:
                raise ValueError(f"formal protocol requires model.{key}={locked_model_revision}")
        if model.get("dtype") != "bfloat16":
            raise ValueError("the formal protocol is locked to bfloat16")
        if model.get("device") != "cuda":
            raise ValueError("the formal protocol is locked to CUDA execution")

    data = config["data"]
    for key in ("dataset", "revision", "config"):
        value = str(_require(data, key, "data")).strip()
        if not value or "REQUIRED" in value.upper():
            raise ValueError(f"data.{key} must be explicitly pinned")
    if int(data.get("sequence_length", 0)) < 2:
        raise ValueError("data.sequence_length must be at least 2")
    for key in ("statistics_segments", "fitness_tokens", "validation_tokens"):
        if int(data.get(key, 0)) <= 0:
            raise ValueError(f"data.{key} must be positive")
    if not is_synthetic:
        if data["dataset"] != "Salesforce/wikitext" or data["config"] != "wikitext-2-raw-v1":
            raise ValueError("the formal protocol is locked to WikiText-2 raw")
        if str(data["revision"]) != "b08601e04326c79dfdd32d625aee71d232d685c3":
            raise ValueError("the formal protocol requires the full pinned WikiText commit")
        locked_data = {
            "sequence_length": 512,
            "statistics_segments": 32,
            "signature_tokens_per_layer": 16384,
            "fitness_tokens": 4096,
            "validation_tokens": 4096,
            "seed": 6707,
        }
        for key, expected in locked_data.items():
            if int(data.get(key, -1)) != expected:
                raise ValueError(f"formal protocol requires data.{key}={expected}")

    grouping = config["grouping"]
    targets = grouping.get("target_groups_per_layer")
    if targets is None:
        raise ValueError(
            "grouping.target_groups_per_layer must be locked from a cited public setting"
        )
    if not str(grouping.get("source_reference", "")).strip():
        raise ValueError("grouping.source_reference must record the configuration provenance")
    if study_type == COMPRESSION_ABLATION_STUDY_TYPE:
        if type(targets) is not int or not 17 <= targets <= 31:
            raise ValueError(
                "compression ablation requires an integer target_groups_per_layer in 17..31"
            )
        try:
            retention_ratio = float(grouping.get("retention_ratio"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "compression ablation retention_ratio must equal target_groups_per_layer/32"
            ) from exc
        expected_ratio = targets / 32
        if retention_ratio != expected_ratio:
            raise ValueError(
                "compression ablation retention_ratio must equal " "target_groups_per_layer/32"
            )
        if project.get("post_hoc") is not True:
            raise ValueError("compression ablation requires project.post_hoc=true")
        parent_run = project.get("parent_run")
        if not isinstance(parent_run, str) or not parent_run.strip():
            raise ValueError("compression ablation requires a non-empty project.parent_run")
        decision_provenance = grouping.get("decision_provenance")
        if not isinstance(decision_provenance, str) or not decision_provenance.strip():
            raise ValueError("compression ablation requires non-empty grouping.decision_provenance")
    if study_type == CAPACITY_TWO_ABLATION_STUDY_TYPE:
        if type(targets) is not int or targets not in {16, 24}:
            raise ValueError(
                "capacity-two ablation requires target_groups_per_layer to be 16 or 24"
            )
        try:
            retention_ratio = float(grouping.get("retention_ratio"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "capacity-two retention_ratio must equal target_groups_per_layer/32"
            ) from exc
        if retention_ratio != targets / 32:
            raise ValueError("capacity-two retention_ratio must equal target_groups_per_layer/32")
        if project.get("post_hoc") is not True:
            raise ValueError("capacity-two ablation requires project.post_hoc=true")
        method_scope = project.get("method_scope")
        if method_scope not in {
            "frequency_only",
            "all_methods",
            CAPACITY_TWO_EA_PILOT_SCOPE,
        }:
            raise ValueError(
                "capacity-two ablation requires project.method_scope to be "
                "frequency_only or all_methods, or one_seed_pilot"
            )
        if method_scope == "frequency_only":
            forbidden_extension_fields = {
                "frequency_reference_run",
                "ea_extension_protocol",
                "ea_extension_protocol_sha256",
                "ea_pilot_protocol",
                "ea_pilot_protocol_sha256",
            }
            if forbidden_extension_fields.intersection(project):
                raise ValueError(
                    "frequency-only capacity-two runs must not declare EA extension fields"
                )
            if "pilot_selection" in config:
                raise ValueError(
                    "frequency-only capacity-two runs must not declare pilot_selection"
                )
        elif method_scope == "all_methods":
            expected_project_name = f"granite_capacity2_g{targets}_ea_ablation"
            expected_run_dir = f"results/runs/{expected_project_name}"
            if (
                project.get("name") != expected_project_name
                or project.get("run_dir") != expected_run_dir
            ):
                raise ValueError(
                    "capacity-two all-methods experiments must use the isolated EA "
                    f"identity {expected_project_name} at {expected_run_dir}"
                )
            expected_frequency_run = f"results/runs/granite_capacity2_g{targets}_ablation"
            if project.get("frequency_reference_run") != expected_frequency_run:
                raise ValueError(
                    "capacity-two all-methods run must name the matching immutable "
                    f"frequency reference: {expected_frequency_run}"
                )
            if project.get("ea_extension_protocol") != CAPACITY_TWO_EA_PROTOCOL_PATH:
                raise ValueError(
                    "capacity-two all-methods run must name the locked EA extension protocol"
                )
            if project.get("ea_extension_protocol_sha256") != CAPACITY_TWO_EA_PROTOCOL_SHA256:
                raise ValueError(
                    "capacity-two all-methods run must pin the EA extension protocol SHA-256"
                )
            if {"ea_pilot_protocol", "ea_pilot_protocol_sha256"}.intersection(project):
                raise ValueError(
                    "capacity-two all-methods run must not declare one-seed pilot fields"
                )
            if "pilot_selection" in config:
                raise ValueError("capacity-two all-methods run must not declare pilot_selection")
        else:
            if targets != 16:
                raise ValueError("the one-seed EA pilot is locked to G16")
            if (
                project.get("name") != CAPACITY_TWO_EA_PILOT_PROJECT_NAME
                or project.get("run_dir") != CAPACITY_TWO_EA_PILOT_RUN_DIR
            ):
                raise ValueError(
                    "the one-seed EA pilot must use the isolated identity "
                    f"{CAPACITY_TWO_EA_PILOT_PROJECT_NAME} at "
                    f"{CAPACITY_TWO_EA_PILOT_RUN_DIR}"
                )
            expected_frequency_run = "results/runs/granite_capacity2_g16_ablation"
            if project.get("frequency_reference_run") != expected_frequency_run:
                raise ValueError(
                    "the one-seed EA pilot must name the immutable G16 Frequency "
                    f"reference: {expected_frequency_run}"
                )
            if project.get("ea_pilot_protocol") != CAPACITY_TWO_EA_PILOT_PROTOCOL_PATH:
                raise ValueError("the one-seed EA pilot must name its locked pilot protocol")
            if project.get("ea_pilot_protocol_sha256") != CAPACITY_TWO_EA_PILOT_PROTOCOL_SHA256:
                raise ValueError("the one-seed EA pilot must pin the pilot protocol SHA-256")
            if {"ea_extension_protocol", "ea_extension_protocol_sha256"}.intersection(project):
                raise ValueError(
                    "the one-seed EA pilot must not reuse full-extension protocol fields"
                )
            if config.get("pilot_selection") != CAPACITY_TWO_EA_PILOT_SELECTION:
                raise ValueError("the one-seed EA pilot selection rule differs from the lock")
        parent_run = project.get("parent_run")
        if not isinstance(parent_run, str) or not parent_run.strip():
            raise ValueError("capacity-two ablation requires a non-empty project.parent_run")
        decision_provenance = grouping.get("decision_provenance")
        if not isinstance(decision_provenance, str) or not decision_provenance.strip():
            raise ValueError(
                "capacity-two ablation requires non-empty grouping.decision_provenance"
            )
        locked_capacity_two = {
            "similarity": "expert_output_cosine",
            "strategy": CAPACITY_TWO_STRATEGY,
            "pairing_algorithm": CAPACITY_TWO_ALGORITHM,
            "networkx_version": CAPACITY_TWO_NETWORKX_VERSION,
            "max_group_size": 2,
            "distance_encoding": CAPACITY_TWO_DISTANCE_ENCODING,
            "tie_break": CAPACITY_TWO_TIE_BREAK,
            "rounding_rule": "exact_integer",
            "source_reference": PINNED_CAPACITY_TWO_SOURCE_REFERENCE,
        }
        for key, expected in locked_capacity_two.items():
            if grouping.get(key) != expected:
                raise ValueError(f"capacity-two grouping requires grouping.{key}={expected}")
        if "linkage" in grouping:
            raise ValueError("capacity-two grouping must not declare a linkage method")
    if not is_synthetic and study_type == FORMAL_MAIN_STUDY_TYPE:
        if (
            type(targets) is not int
            or targets != 16
            or float(grouping.get("retention_ratio", -1)) != 0.5
        ):
            raise ValueError("formal grouping must retain 16/32 experts per layer")
    if study_type == COMPRESSION_ABLATION_STUDY_TYPE or (
        not is_synthetic and study_type == FORMAL_MAIN_STUDY_TYPE
    ):
        if (
            grouping.get("similarity") != "expert_output_cosine"
            or grouping.get("linkage") != "average"
        ):
            raise ValueError("formal grouping requires output cosine and average linkage")
        if grouping.get("rounding_rule") != "exact_integer":
            raise ValueError("formal grouping requires rounding_rule=exact_integer")
        if grouping.get("source_reference") != PINNED_GROUPING_SOURCE_REFERENCE:
            raise ValueError("formal grouping provenance must use the pinned HC-SMoE commit")

    fusion = config["fusion"]
    if not is_synthetic:
        if fusion.get("baseline") != "routing_frequency":
            raise ValueError("formal baseline must use routing frequency")
        if float(fusion.get("smoothing", 0)) != 1.0e-6:
            raise ValueError("formal protocol requires fusion.smoothing=1e-6")
        if fusion.get("accumulation_dtype") != "float32":
            raise ValueError("fusion accumulation must use float32")
        if fusion.get("source_weight_device") != "model":
            raise ValueError("formal protocol requires fusion.source_weight_device=model")

    search = config["search"]
    population = int(search.get("population_size", 0))
    budget = int(search.get("evaluation_budget", 0))
    if population < 2 or budget < population:
        raise ValueError("search budget must contain a complete population of at least two")
    warmup_tokens = int(search.get("warmup_tokens", 0))
    if warmup_tokens <= 0:
        raise ValueError("search.warmup_tokens must be positive")
    if warmup_tokens > int(data["fitness_tokens"]):
        raise ValueError("search.warmup_tokens cannot exceed data.fitness_tokens")
    seeds = search.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 1 or len(set(map(int, seeds))) != len(seeds):
        raise ValueError("search.seeds must contain unique integer seeds")
    if not is_synthetic:
        locked_search = {
            "warmup_tokens": 4096,
            "population_size": 16,
            "offspring_size": 16,
            "evaluation_budget": 300,
            "tournament_size": 2,
        }
        for key, expected in locked_search.items():
            if int(search.get(key, -1)) != expected:
                raise ValueError(f"formal protocol requires search.{key}={expected}")
        locked_seeds = (
            [6707]
            if study_type == CAPACITY_TWO_ABLATION_STUDY_TYPE
            and project.get("method_scope") == CAPACITY_TWO_EA_PILOT_SCOPE
            else [6707, 6708, 6709]
        )
        if [int(value) for value in seeds] != locked_seeds:
            raise ValueError(f"formal protocol requires seeds {locked_seeds}")
        if search.get("strategies") != ["coordinate", "layer_block"]:
            raise ValueError("formal protocol requires both pre-registered strategies in order")
        locked_sigma = {"init_sigma": 0.02, "mutation_sigma": 0.15}
        for key, expected in locked_sigma.items():
            if float(search.get(key, -1)) != expected:
                raise ValueError(f"formal protocol requires search.{key}={expected}")
        if int(search.get("micro_batch_size", -1)) != 1:
            raise ValueError("formal protocol requires search.micro_batch_size=1")

    evaluation = config["evaluation"]
    if evaluation.get("tasks") != ["arc_easy", "openbookqa"]:
        raise ValueError("evaluation.tasks must preserve the pre-registered task order")
    if int(evaluation.get("num_fewshot", -1)) != 0:
        raise ValueError("this protocol pre-registers zero-shot evaluation")
    if evaluation.get("primary_metric") != "acc":
        raise ValueError("this protocol pre-registers acc as the primary metric")
    task_datasets = evaluation.get("task_datasets")
    if not isinstance(task_datasets, dict) or list(task_datasets) != evaluation["tasks"]:
        raise ValueError("evaluation.task_datasets must follow the registered task order")
    required_dataset_keys = ("hub_repository", "path", "name", "split", "revision")
    for task in evaluation["tasks"]:
        dataset_config = task_datasets.get(task)
        if not isinstance(dataset_config, dict):
            raise ValueError(f"evaluation.task_datasets.{task} must be a mapping")
        for key in required_dataset_keys:
            value = str(_require(dataset_config, key, f"evaluation.task_datasets.{task}"))
            if not value.strip():
                raise ValueError(f"evaluation.task_datasets.{task}.{key} cannot be empty")
        if re.fullmatch(r"[0-9a-f]{40}", str(dataset_config["revision"])) is None:
            raise ValueError(f"evaluation.task_datasets.{task}.revision must be a full HF commit")
        if dataset_config != LOCKED_TASK_DATASETS[task]:
            raise ValueError(
                f"evaluation.task_datasets.{task} differs from the pinned official dataset"
            )
    if not is_synthetic:
        locked_evaluation = {
            "report_metrics": ["acc", "acc_norm"],
            "batch_size": 1,
            "random_seed": 0,
            "numpy_random_seed": 1234,
            "torch_random_seed": 1234,
            "fewshot_random_seed": 1234,
            "log_samples": False,
            "expected_samples": {"arc_easy": 2376, "openbookqa": 500},
        }
        for key, expected in locked_evaluation.items():
            if evaluation.get(key) != expected:
                raise ValueError(f"formal protocol requires evaluation.{key}={expected}")


def validate_supplemental_evaluation_config(config: dict[str, Any]) -> None:
    """Validate the isolated post-hoc WinoGrande/PIQA protocol.

    This intentionally does not call or weaken :func:`validate_config`: the
    pre-registered ARC-Easy/OpenBookQA configuration remains unchanged.
    """

    expected_root_keys = {
        "schema_version",
        "protocol_id",
        "study_type",
        "post_hoc",
        "decision_provenance",
        "lm_eval_version",
        "evaluation",
    }
    if set(config) != expected_root_keys:
        raise ValueError("supplemental evaluation config must contain exactly the locked root keys")
    if config.get("schema_version") != SUPPLEMENTAL_EVALUATION_CONFIG_SCHEMA:
        raise ValueError("unsupported supplemental evaluation config schema")
    if config.get("protocol_id") != SUPPLEMENTAL_EVALUATION_PROTOCOL_ID:
        raise ValueError("unknown supplemental evaluation protocol")
    if config.get("study_type") != "post_hoc_supplemental" or config.get("post_hoc") is not True:
        raise ValueError("supplemental evaluation must be explicitly marked post-hoc")
    if not str(config.get("decision_provenance", "")).strip():
        raise ValueError("supplemental evaluation requires decision provenance")
    if config.get("lm_eval_version") != "0.4.8":
        raise ValueError("supplemental evaluation is locked to lm-eval==0.4.8")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("supplemental evaluation.evaluation must be a mapping")
    expected_evaluation_keys = {
        "protocol_id",
        "tasks",
        "task_datasets",
        "num_fewshot",
        "primary_metric",
        "report_metrics",
        "batch_size",
        "random_seed",
        "numpy_random_seed",
        "torch_random_seed",
        "fewshot_random_seed",
        "log_samples",
        "expected_samples",
        "expected_content_evidence",
        "chance_accuracy",
        "full_validation_only",
    }
    if set(evaluation) != expected_evaluation_keys:
        raise ValueError(
            "supplemental evaluation block must contain exactly the locked protocol keys"
        )
    locked_values = {
        "protocol_id": SUPPLEMENTAL_EVALUATION_PROTOCOL_ID,
        "tasks": ["winogrande", "piqa"],
        "task_datasets": LOCKED_SUPPLEMENTAL_TASK_DATASETS,
        "num_fewshot": 0,
        "primary_metric": "acc",
        # WinoGrande's official lm-eval 0.4.8 task reports acc but not acc_norm.
        "report_metrics": ["acc"],
        "batch_size": 1,
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
        "log_samples": False,
        "expected_samples": LOCKED_SUPPLEMENTAL_EXPECTED_SAMPLES,
        "expected_content_evidence": LOCKED_SUPPLEMENTAL_CONTENT_EVIDENCE,
        "chance_accuracy": LOCKED_SUPPLEMENTAL_CHANCE_ACCURACY,
        "full_validation_only": True,
    }
    for key, expected in locked_values.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"supplemental protocol requires evaluation.{key}={expected}")


def config_digest(config: dict[str, Any]) -> str:
    canonical = yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def lock_run_config(config: dict[str, Any], run_dir: str | Path) -> str:
    """Create or verify an immutable resolved configuration inside a run directory."""

    validate_config(config)
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_path = destination / "config.resolved.yaml"
    expected_digest = config_digest(config)
    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if not isinstance(existing, dict) or config_digest(existing) != expected_digest:
            raise ValueError(
                f"run directory {destination} is locked to a different resolved configuration"
            )
    else:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            prefix=".config.resolved.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
            temporary = Path(handle.name)
        os.replace(temporary, resolved_path)
    write_json(
        destination / "config.lock.json",
        {
            "schema_version": "config-lock-v1",
            "resolved_config": "config.resolved.yaml",
            "sha256": expected_digest,
        },
    )
    return expected_digest


def lock_supplemental_evaluation_config(config: dict[str, Any], run_dir: str | Path) -> str:
    """Lock a supplemental protocol without changing the experiment config lock."""

    validate_supplemental_evaluation_config(config)
    destination = Path(run_dir) / "supplemental_evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    resolved_path = destination / "protocol.resolved.yaml"
    expected_digest = config_digest(config)
    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if not isinstance(existing, dict) or config_digest(existing) != expected_digest:
            raise ValueError(
                f"run directory {run_dir} is locked to a different supplemental protocol"
            )
    else:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            prefix=".protocol.resolved.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
            temporary = Path(handle.name)
        os.replace(temporary, resolved_path)
    write_json(
        destination / "protocol.lock.json",
        {
            "schema_version": "supplemental-evaluation-config-lock-v1",
            "resolved_config": "protocol.resolved.yaml",
            "sha256": expected_digest,
        },
    )
    return expected_digest
