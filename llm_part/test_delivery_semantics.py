#!/usr/bin/env python3
"""Regression tests for the generated delivery protocol contract."""

from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from llm_input import delivery_semantic_errors, events_for_mission


ROOT = Path(__file__).resolve().parent.parent


def _baseline_payload() -> dict[str, object]:
    data = ET.parse(
        ROOT
        / "automata"
        / "baseline_automata"
        / "delivery"
        / "delivery_task_specification_corrected.xml"
    ).getroot().find("data")
    assert data is not None
    states = {state.attrib["id"]: state.attrib["name"] for state in data.findall("state")}
    events = {event.attrib["id"]: event.attrib["name"] for event in data.findall("event")}
    transitions = [
        f'(\"{states[item.attrib["source"]]}\", '
        f'\"{events[item.attrib["event"]]}\", '
        f'\"{states[item.attrib["target"]]}\")'
        for item in data.findall("transition")
    ]
    return {"automata": [{"name": "hidden_reference", "transitions": transitions}]}


class DeliverySemanticValidationTest(unittest.TestCase):
    def test_llm_delivery_alphabet_excludes_low_level_motion(self):
        events = events_for_mission("delivery")
        self.assertTrue(
            {
                "search_object",
                "approach_object",
                "search_zone",
                "approach_zone",
            }
            <= set(events)
        )
        self.assertTrue(
            {
                "move_forward",
                "rotate_clockwise",
                "rotate_counterclockwise",
                "full_rotate",
            }.isdisjoint(events)
        )

    def test_hidden_reference_satisfies_contract(self):
        self.assertEqual(delivery_semantic_errors(_baseline_payload()), [])

    def test_split_color_protocols_are_rejected(self):
        baseline = _baseline_payload()["automata"][0]
        payload = {
            "automata": [
                {**baseline, "name": "red_delivery_specification"},
                {**baseline, "name": "green_delivery_specification"},
                {**baseline, "name": "blue_delivery_specification"},
            ]
        }
        errors = delivery_semantic_errors(payload)
        self.assertTrue(
            any("exactly one unified delivery protocol" in error for error in errors)
        )

    def test_known_blind_approach_candidate_is_rejected(self):
        path = (
            ROOT
            / "automata"
            / "resulting_automata"
            / "YAML"
            / "S_20260731_161911.llm_output.json"
        )
        if not path.is_file():
            self.skipTest("Known failed run artifact is not available")
        errors = delivery_semantic_errors(json.loads(path.read_text(encoding="utf-8")))
        self.assertTrue(any("approach_object" in error and "self-loop" in error for error in errors))
        self.assertTrue(any("not preceded by red_object_visible" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
