#!/usr/bin/env python3
"""Regression tests for pre-conversion generated-event validation."""

from __future__ import annotations

import unittest

from llm_input import authoritative_event_errors, events_for_mission


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
            "full_rotate",
        ):
            self.assertIn(event, exploration)
        for event in (
            "task_move_forward",
            "task_rotate_clockwise",
            "task_rotate_counterclockwise",
        ):
            self.assertIn(event, exploration)
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
            *(f"claim_{color}" for color in ("red", "green", "blue")),
            "obstacle_front", "obstacle_left", "obstacle_right", "path_clear",
            *(
                f"{color}_{target}_{observation}"
                for color in ("red", "green", "blue")
                for target in ("object", "zone")
                for observation in ("visible", "not_visible", "reached")
            ),
            *(f"received_claim_{color}" for color in ("red", "green", "blue")),
        }
        self.assertEqual(set(delivery), expected_delivery_events)

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
