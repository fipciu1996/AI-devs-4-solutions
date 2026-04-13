import unittest

from sensors.solve_sensors import (
    DEFAULT_SITE_NAME,
    StaticFinding,
    NoteReview,
    SensorRecord,
    analyze_measurements,
    build_file_decisions,
    note_cache_key,
    select_note_review_candidates,
    split_note_parts,
)


def build_record(**overrides):
    payload = {
        "file_id": "0001",
        "sensor_type": "temperature",
        "timestamp": 1774000000,
        "temperature_K": 600.0,
        "pressure_bar": 0.0,
        "water_level_meters": 0.0,
        "voltage_supply_v": 0.0,
        "humidity_percent": 0.0,
        "operator_notes": (
            "Everything checks out, the observed pattern remains trustworthy, "
            "and I closed this check without action for this logged checkpoint."
        ),
    }
    payload.update(overrides)
    return SensorRecord(**payload)


class SolveSensorsTests(unittest.TestCase):
    def test_default_site_name_uses_solver_name_not_course_task_name(self) -> None:
        self.assertEqual(DEFAULT_SITE_NAME, "AI Devs 4 - sensors")

    def test_analyze_measurements_detects_out_of_range_and_inactive_noise(self) -> None:
        record = build_record(
            sensor_type="temperature/voltage",
            temperature_K=1101.0,
            voltage_supply_v=225.0,
            water_level_meters=9.3,
        )

        reasons = analyze_measurements(record)

        self.assertIn("temperature_K:out_of_range", reasons)
        self.assertIn("voltage_supply_v:out_of_range", reasons)
        self.assertIn("water_level_meters:inactive_nonzero", reasons)

    def test_build_file_decisions_marks_problem_note_as_anomaly_when_measurements_are_normal(self) -> None:
        record = build_record()
        findings = {
            record.file_id: StaticFinding(
                file_id=record.file_id,
                measurement_reasons=(),
                note_parts=split_note_parts(record.operator_notes),
            )
        }
        reviews = {
            note_cache_key(record.operator_notes): NoteReview(
                key=note_cache_key(record.operator_notes),
                operator_notes=record.operator_notes,
                note_claim="problem",
                reason="The note explicitly reports a fault.",
                source="model",
            )
        }

        decisions = build_file_decisions([record], findings, reviews)

        self.assertTrue(decisions[0].is_anomaly)
        self.assertEqual(decisions[0].note_claim, "problem")

    def test_split_note_parts_keeps_unusual_single_sentence_note(self) -> None:
        note = "The report looks completely normal. I will go to check status of all other devices."

        self.assertEqual(split_note_parts(note), (note,))

    def test_note_cache_key_is_stable_for_same_note(self) -> None:
        note = (
            "This state looks unstable, because the output suggests a potential fault, "
            "and I submitted it for root-cause analysis before this can be accepted."
        )

        self.assertEqual(note_cache_key(note), note_cache_key(note))

    def test_select_note_review_candidates_keeps_only_escalated_records(self) -> None:
        normal = build_record(file_id="0001")
        problem_note = build_record(
            file_id="0002",
            operator_notes=(
                "Everything checks out, a warning suggests a possible fault in this area, "
                "and I escalated the checkpoint for a deeper review."
            ),
        )
        unusual_note = build_record(
            file_id="0003",
            operator_notes="Single sentence warning about a possible fault.",
        )
        measurement_anomaly = build_record(
            file_id="0004",
            temperature_K=1001.0,
        )
        records = [normal, problem_note, unusual_note, measurement_anomaly]
        findings = {
            record.file_id: StaticFinding(
                file_id=record.file_id,
                measurement_reasons=analyze_measurements(record),
                note_parts=split_note_parts(record.operator_notes),
            )
            for record in records
        }

        candidates = select_note_review_candidates(records, findings)

        self.assertEqual(
            {record.file_id for record in candidates},
            {"0002", "0003", "0004"},
        )


if __name__ == "__main__":
    unittest.main()
