#!/usr/bin/env python3
"""Regression tests for pre-conversion generated-event validation."""

from __future__ import annotations

import unittest
from pathlib import Path

from llm_input import (
    authoritative_event_errors, events_for_mission, events_available_in_automata,
    events_from_automata, add_event_meanings,
)


class AuthoritativeEventValidationTest(unittest.TestCase):
    def test_mission_event_interfaces_match_baselines(self):
        exploration = events_for_mission("exploration")
        patrolling = events_for_mission("patrolling")
        delivery = events_for_mission("delivery")

        for event in (
            "move_forward",
            "move_backward",
            "rotate_clockwise",
            "rotate_counterclockwise",
            "u_turn",
        ):
            self.assertIn(event, exploration)
        for event in (
            "task_move_forward",
            "task_rotate_clockwise",
            "task_rotate_counterclockwise",
        ):
            self.assertNotIn(event, exploration)
            self.assertNotIn(event, patrolling)
            self.assertNotIn(event, delivery)

        for color in ("red", "green", "blue"):
            self.assertNotIn(f"search_{color}", patrolling)
            self.assertNotIn(f"approach_{color}", patrolling)

        self.assertTrue(patrolling["search_color"])
        self.assertTrue(patrolling["approach_color"])
        for event in (
            "search_object",
            "approach_object",
            "search_zone",
            "approach_zone",
        ):
            self.assertTrue(delivery[event])
        expected_delivery_events = {
            "search_object", "approach_object", "search_zone", "approach_zone",
            "pick_up_object", "drop_object",
            "move_forward", "move_backward", "rotate_clockwise",
            "rotate_counterclockwise", "u_turn",
            *(f"claim_{color}" for color in ("red", "green", "blue")),
            "obstacle_front", "obstacle_left", "obstacle_right", "path_clear",
            "timeout",
            *(
                f"{color}_{target}_{observation}"
                for color in ("red", "green", "blue")
                for target in ("object", "zone")
                for observation in ("visible", "not_visible", "reached")
            ),
            *(f"received_claim_{color}" for color in ("red", "green", "blue")),
        }
        baseline = Path(__file__).resolve().parent.parent / "automata/baseline_automata/delivery"
        plants = [baseline / name for name in (
            "obstacle_sensor.xml", "motion_plant.xml", "pickup_and_drop_plant.xml",
            "color_sensor.xml", "claiming_receiving_plant.xml",
        )]
        self.assertEqual(set(events_available_in_automata(delivery, plants)), expected_delivery_events)

    def test_delivery_stack_alphabet_has_meanings_and_plant_controllability(self):
        baseline = Path(__file__).resolve().parent.parent / "automata/baseline_automata/delivery_stack"
        events = events_from_automata(sorted(baseline.glob("G*.xml")))
        self.assertEqual(len(events), 31)
        self.assertTrue(events["drop_zone_red"])
        self.assertFalse(events["recieve_drop_zone_red"])
        self.assertFalse(events["object_not_reached"])
        self.assertNotIn("claim_red", events)
        prompt = add_event_meanings("", events)
        for event in events:
            self.assertIn(f"`{event}`:", prompt)

    def test_unused_declared_controllable_is_repaired_before_conversion(self):
        payload = {
            "automata": [
                {
                    "name": "patrol_specification",
                    "states": ["searching"],
                    "events": ["search_color", "approach_color"],
                    "transitions": [
                        '("searching", "search_color", "searching")'
                    ],
                }
            ]
        }
        errors = authoritative_event_errors(
            payload,
            {"search_color": True, "approach_color": True},
        )
        self.assertTrue(
            any("approach_color" in error and "never uses" in error for error in errors)
        )

    def test_used_controllables_are_accepted(self):
        payload = {
            "automata": [
                {
                    "name": "patrol_specification",
                    "states": ["searching", "approaching"],
                    "events": ["search_color", "approach_color"],
                    "transitions": [
                        '("searching", "search_color", "searching")',
                        '("approaching", "approach_color", "approaching")',
                    ],
                }
            ]
        }
        self.assertEqual(
            authoritative_event_errors(
                payload,
                {"search_color": True, "approach_color": True},
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
