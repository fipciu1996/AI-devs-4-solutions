from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fire_in_the_hole import (
    TaskFlags,
    TaskResult,
    apply_negotiations_defaults,
    build_task_registry,
    collect_success_flags,
    empty_token_totals,
    extract_flags_from_log_text,
    parse_args,
    render_flags_table,
    render_task_run_table,
    task_cost_path,
    write_run_cost_summary,
    write_task_cost_report,
)


class FireInTheHoleFlagTests(unittest.TestCase):
    def test_build_task_registry_includes_recent_single_script_tasks(self) -> None:
        registry = build_task_registry()

        self.assertIn("phonecall", registry)
        self.assertEqual(registry["phonecall"].description, "Run the phonecall solver.")
        self.assertIn("goingthere", registry)
        self.assertEqual(registry["goingthere"].description, "Run the goingthere solver.")
        self.assertIn("timetravel", registry)
        self.assertEqual(registry["timetravel"].description, "Run the timetravel solver.")

    def test_people_pipeline_uses_separate_findhim_directory(self) -> None:
        registry = build_task_registry()

        steps = registry["people"].build_steps(SimpleNamespace(verify=True))

        self.assertEqual(Path(steps[0].command[-2]).as_posix(), "people/filter_people.py")
        self.assertEqual(steps[0].command[-1], "--verify")
        self.assertEqual(Path(steps[1].command[-2]).as_posix(), "findhim/solve_findhim.py")
        self.assertEqual(steps[1].command[-1], "--verify")

    def test_extract_flags_from_log_text_reads_main_and_secret_flags(self) -> None:
        flags = extract_flags_from_log_text(
            "categorize",
            'INFO message: "ACCEPTED - {FLG:SMUGGLER} / sekret: {FLG:JUMPJUMP}"\n',
        )

        self.assertEqual(
            flags,
            TaskFlags(
                task_name="categorize",
                main_flag="{FLG:SMUGGLER}",
                secret_flag="{FLG:JUMPJUMP}",
            ),
        )

    def test_extract_flags_from_log_text_reads_single_main_flag(self) -> None:
        flags = extract_flags_from_log_text(
            "reactor",
            "SUCCESS | reactor | Flag: {FLG:INSTALLED}\n",
        )

        self.assertEqual(
            flags,
            TaskFlags(
                task_name="reactor",
                main_flag="{FLG:INSTALLED}",
                secret_flag=None,
            ),
        )

    def test_extract_flags_from_log_text_uses_second_unique_flag_as_secret(self) -> None:
        flags = extract_flags_from_log_text(
            "people",
            "INFO | people-filter | {FLG:FIRST}\nINFO | findhim | {FLG:SECOND}\n",
        )

        self.assertEqual(
            flags,
            TaskFlags(
                task_name="people",
                main_flag="{FLG:FIRST}",
                secret_flag="{FLG:SECOND}",
            ),
        )

    def test_render_flags_table_includes_headers_and_placeholders(self) -> None:
        table = render_flags_table(
            [
                TaskFlags("reactor", "{FLG:INSTALLED}", None),
                TaskFlags("categorize", "{FLG:SMUGGLER}", "{FLG:JUMPJUMP}"),
            ]
        )

        self.assertIn("Task name", table)
        self.assertIn("Main flag", table)
        self.assertIn("Secret flag", table)
        self.assertIn("reactor", table)
        self.assertIn("{FLG:INSTALLED}", table)
        self.assertIn("categorize", table)
        self.assertIn("{FLG:JUMPJUMP}", table)
        self.assertIn(" -", table.replace("|", " "))

    def test_render_task_run_table_includes_duration_models_and_flag_status(self) -> None:
        tasks = [
            type("Task", (), {"name": "domatowo"})(),
            type("Task", (), {"name": "reactor"})(),
        ]
        domatowo_cost_path = Path(__file__).with_name("_domatowo_run_table_cost.json")
        reactor_log_path = Path(__file__).with_name("_reactor_run_table.log")
        for path in (domatowo_cost_path, reactor_log_path):
            path.unlink(missing_ok=True)

        domatowo_cost_path.write_text(
            json.dumps({"models": ["model-b", "model-a"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        reactor_log_path.write_text(
            "SUCCESS | reactor | Flag: {FLG:INSTALLED}\n",
            encoding="utf-8",
        )
        try:
            results_by_name = {
                "domatowo": TaskResult(
                    name="domatowo",
                    status="success",
                    duration_seconds=12.34,
                    step_count=1,
                    cost_path=domatowo_cost_path,
                ),
                "reactor": TaskResult(
                    name="reactor",
                    status="success",
                    duration_seconds=2.0,
                    step_count=1,
                    log_path=reactor_log_path,
                ),
            }

            table = render_task_run_table(tasks, results_by_name)
        finally:
            for path in (domatowo_cost_path, reactor_log_path):
                path.unlink(missing_ok=True)

        self.assertIn("Task name", table)
        self.assertIn("Time from starting task", table)
        self.assertIn("Models used", table)
        self.assertIn("If flag fetched", table)
        self.assertIn("domatowo", table)
        self.assertIn("12.3s", table)
        self.assertIn("model-a, model-b", table)
        self.assertIn("reactor", table)
        self.assertIn("2.0s", table)
        self.assertIn("yes", table)
        self.assertIn("no", table)

    def test_collect_success_flags_keeps_placeholder_rows_for_success_without_flag(self) -> None:
        tasks = [
            type("Task", (), {"name": "domatowo"})(),
            type("Task", (), {"name": "reactor"})(),
        ]
        results_by_name = {
            "domatowo": TaskResult(
                name="domatowo",
                status="success",
                duration_seconds=1.0,
                step_count=1,
                log_path=None,
            ),
            "reactor": TaskResult(
                name="reactor",
                status="success",
                duration_seconds=1.0,
                step_count=1,
                log_path=Path(__file__),
            ),
        }

        with patch(
            "fire_in_the_hole.extract_task_flags",
            side_effect=[None, TaskFlags("reactor", "{FLG:INSTALLED}", None)],
        ):
            flags = collect_success_flags(tasks, results_by_name)

        self.assertEqual(
            flags,
            [
                TaskFlags("domatowo", None, None),
                TaskFlags("reactor", "{FLG:INSTALLED}", None),
            ],
        )

    def test_collect_success_flags_includes_failed_pipeline_when_flag_was_logged(self) -> None:
        tasks = [
            type("Task", (), {"name": "people"})(),
            type("Task", (), {"name": "reactor"})(),
        ]
        results_by_name = {
            "people": TaskResult(
                name="people",
                status="failed",
                duration_seconds=1.0,
                step_count=2,
                log_path=Path(__file__),
            ),
            "reactor": TaskResult(
                name="reactor",
                status="success",
                duration_seconds=1.0,
                step_count=1,
                log_path=None,
            ),
        }

        with patch(
            "fire_in_the_hole.extract_task_flags",
            side_effect=[TaskFlags("people", "{FLG:SURVIVORS}", None), None],
        ):
            flags = collect_success_flags(tasks, results_by_name)

        self.assertEqual(
            flags,
            [
                TaskFlags("people", "{FLG:SURVIVORS}", None),
                TaskFlags("reactor", None, None),
            ],
        )

    def test_collect_success_flags_includes_findhim_as_separate_summary_row(self) -> None:
        tasks = [type("Task", (), {"name": "people"})()]
        log_path = Path(__file__).with_name("_people_pipeline.log")
        log_path.unlink(missing_ok=True)
        try:
            log_path.write_text(
                "\n".join(
                    [
                        "Task: people",
                        "Description: People pipeline",
                        "Started: 2026-04-10 12:00:00",
                        "Step count: 2",
                        "",
                        "=== Step 1/2: people-filter ===",
                        "CWD: C:\\repo",
                        "CMD: python people/filter_people.py --verify",
                        "",
                        "INFO | people-filter | {FLG:SURVIVORS}",
                        "",
                        "=== Step 2/2: findhim ===",
                        "CWD: C:\\repo",
                        "CMD: python findhim/solve_findhim.py --verify",
                        "",
                        "INFO | findhim | {FLG:BUSTED}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            results_by_name = {
                "people": TaskResult(
                    name="people",
                    status="success",
                    duration_seconds=1.0,
                    step_count=2,
                    log_path=log_path,
                ),
            }

            flags = collect_success_flags(tasks, results_by_name)
        finally:
            log_path.unlink(missing_ok=True)

        self.assertEqual(
            flags,
            [
                TaskFlags("people", "{FLG:SURVIVORS}", "{FLG:BUSTED}"),
                TaskFlags("findhim", "{FLG:BUSTED}", None),
            ],
        )

    def test_parse_args_enables_verify_by_default_and_supports_no_verify(self) -> None:
        with patch.object(sys, "argv", ["fire_in_the_hole.py"]):
            default_args = parse_args()
        with patch.object(sys, "argv", ["fire_in_the_hole.py", "--no-verify"]):
            disabled_args = parse_args()

        self.assertTrue(default_args.verify)
        self.assertFalse(disabled_args.verify)

    def test_apply_negotiations_defaults_enables_ngrok_from_repo_env(self) -> None:
        with patch.object(sys, "argv", ["fire_in_the_hole.py"]):
            args = parse_args()
        with patch.dict(
            "os.environ",
            {
                "NGROK_AUTH_TOKEN": "token-from-env",
                "NGROK_DOMAIN": "negotiations.example.ngrok.app",
            },
            clear=True,
        ):
            apply_negotiations_defaults(args)

        self.assertTrue(args.negotiations_use_ngrok)
        self.assertEqual(args.negotiations_ngrok_domain, "negotiations.example.ngrok.app")

    def test_write_task_cost_report_creates_zero_totals_when_usage_file_is_missing(self) -> None:
        cost_path = Path(__file__).with_name("_fire_in_the_hole_domatowo_cost.json")
        cost_path.unlink(missing_ok=True)
        try:
            result = TaskResult(
                name="domatowo",
                status="success",
                duration_seconds=1.25,
                step_count=1,
                cost_path=cost_path,
            )

            payload = write_task_cost_report(result)

            self.assertIsNotNone(payload)
            self.assertEqual(payload["totals"], empty_token_totals())
            self.assertEqual(json.loads(cost_path.read_text(encoding="utf-8"))["status"], "success")
        finally:
            cost_path.unlink(missing_ok=True)

    def test_write_run_cost_summary_aggregates_all_task_totals(self) -> None:
        cost_dir = Path(__file__).resolve().parent
        domatowo_path = cost_dir / "_domatowo_cost.json"
        reactor_path = cost_dir / "_reactor_cost.json"
        summary_path = cost_dir / "fire_in_the_hole_total.json"
        for path in (domatowo_path, reactor_path, summary_path):
            path.unlink(missing_ok=True)

        tasks = [
            type("Task", (), {"name": "domatowo"})(),
            type("Task", (), {"name": "reactor"})(),
        ]
        results_by_name = {
            "domatowo": TaskResult(
                name="domatowo",
                status="success",
                duration_seconds=2.0,
                step_count=1,
                cost_path=domatowo_path,
            ),
            "reactor": TaskResult(
                name="reactor",
                status="failed",
                duration_seconds=3.0,
                step_count=1,
                cost_path=reactor_path,
                detail="boom",
            ),
        }

        domatowo_path.write_text(
            json.dumps(
                {
                    "task": "domatowo",
                    "request_count": 1,
                    "models": ["model-a"],
                    "totals": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                        "cache_write_tokens": 0,
                        "total_tokens": 15,
                    },
                    "usage_by_model": {},
                    "calls": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            generated_summary_path = write_run_cost_summary(
                tasks,
                results_by_name,
                cost_dir=cost_dir,
            )

            self.assertEqual(generated_summary_path, summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["totals"]["total_tokens"], 15)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["tasks"]["domatowo"]["totals"]["total_tokens"], 15)
            self.assertEqual(summary["tasks"]["reactor"]["totals"]["total_tokens"], 0)
        finally:
            for path in (domatowo_path, reactor_path, summary_path):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
