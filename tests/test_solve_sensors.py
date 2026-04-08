import unittest

from sensors.solve_sensors import (
    SensorRecord,
    analyze_measurements,
    classify_note_polarity,
    note_cache_key,
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

    def test_classify_note_polarity_handles_positive_and_negative_templates(self) -> None:
        positive = (
            "Current status remains healthy, system response remains predictable, "
            "so this entry is approved as-is for the latest service snapshot."
        )
        negative = (
            "This state looks unstable, because the output suggests a potential fault, "
            "and I submitted it for root-cause analysis before this can be accepted."
        )

        self.assertEqual(classify_note_polarity(positive), "positive")
        self.assertEqual(classify_note_polarity(negative), "negative")

    def test_split_note_parts_keeps_unusual_single_sentence_note(self) -> None:
        note = "The report looks completely normal. I will go to check status of all other devices."

        self.assertEqual(split_note_parts(note), (note,))

    def test_note_cache_key_is_stable_for_same_note(self) -> None:
        note = (
            "This state looks unstable, because the output suggests a potential fault, "
            "and I submitted it for root-cause analysis before this can be accepted."
        )

        self.assertEqual(note_cache_key(note), note_cache_key(note))


if __name__ == "__main__":
    unittest.main()
