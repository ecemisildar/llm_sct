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
        / "delivery_task_specification.xml"
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
    def test_llm_delivery_alphabet_includes_low_level_motion(self):
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
                "move_backward",
                "rotate_clockwise",
                "rotate_counterclockwise",
                "u_turn",
            }
            <= set(events)
        )

    def test_baseline_task_stops_after_color_assignment(self):
        path = (
            ROOT
            / "automata"
            / "baseline_automata"
            / "delivery"
            / "delivery_task_specification.xml"
        )
        data = ET.parse(path).getroot().find("data")
        assert data is not None
        events = {event.attrib["name"] for event in data.findall("event")}
        self.assertEqual(
            events,
            {
                "claim_red",
                "claim_green",
                "claim_blue",
                "received_claim_red",
                "received_claim_green",
                "received_claim_blue",
            },
        )
        marked_states = {
            state.attrib["name"]
            for state in data.findall("state")
            if state.attrib["marked"] == "True"
        }
        self.assertEqual(
            marked_states,
            {"red_task_assigned", "green_task_assigned", "blue_task_assigned"},
        )

    def test_claiming_receiving_plant_uses_claim_events_only(self):
        directory = ROOT / "automata" / "baseline_automata" / "delivery"
        data = ET.parse(directory / "claiming_receiving_plant.xml").getroot().find("data")
        assert data is not None
        events = {event.attrib["name"] for event in data.findall("event")}
        self.assertEqual(
            events,
            {
                *(f"claim_{color}" for color in ("red", "green", "blue")),
                *(f"received_claim_{color}" for color in ("red", "green", "blue")),
            },
        )

    def test_pickup_drop_plant_has_no_result_events(self):
        path = (
            ROOT
            / "automata"
            / "baseline_automata"
            / "delivery"
            / "pickup_and_drop_plant.xml"
        )
        data = ET.parse(path).getroot().find("data")
        assert data is not None
        events = {event.attrib["name"] for event in data.findall("event")}
        self.assertEqual(events, {"pick_up_object", "drop_object"})

    def test_robot_delivery_task_is_claim_gated_and_ordered(self):
        task_actions = {
            "search_object", "approach_object", "pick_up_object",
            "search_zone", "approach_zone", "drop_object",
        }
        for color in ("red", "green", "blue"):
            with self.subTest(color=color):
                path = (
                    ROOT / "automata" / "baseline_automata" / "delivery"
                    / f"{color}_robot_delivery_task.xml"
                )
                data = ET.parse(path).getroot().find("data")
                assert data is not None
                states = {
                    state.attrib["id"]: state.attrib["name"]
                    for state in data.findall("state")
                }
                marked_states = {
                    state.attrib["name"]
                    for state in data.findall("state")
                    if state.attrib["marked"] == "True"
                }
                events = {
                    event.attrib["id"]: event.attrib["name"]
                    for event in data.findall("event")
                }
                transitions = {
                    (states[item.attrib["source"]], events[item.attrib["event"]]):
                    states[item.attrib["target"]]
                    for item in data.findall("transition")
                }
                self.assertTrue(
                    task_actions.isdisjoint(
                        event for state, event in transitions if state == "waiting"
                    )
                )
                self.assertEqual(
                    marked_states,
                    {f"{color}_done", *(f"passive_{other}" for other in ("red", "green", "blue") if other != color)},
                )
                self.assertEqual(
                    transitions[("waiting", f"claim_{color}")],
                    f"{color}_search_object",
                )
                self.assertEqual(
                    transitions[(f"{color}_pickup", "pick_up_object")],
                    f"{color}_search_zone",
                )
                self.assertEqual(
                    transitions[(f"{color}_drop", "drop_object")],
                    f"{color}_done",
                )

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
        self.assertTrue(any("search_object" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
