#!/usr/bin/env python3
"""Regression tests for fixed versus LLM collision-control conversion."""

import unittest

from llm_json_to_xml import build_xml


class CollisionControlTest(unittest.TestCase):
    def setUp(self):
        self.events = {
            "move_forward": True,
            "rotate_clockwise": True,
            "obstacle_front": False,
        }
        self.payload = {
            "name": "motion_specification",
            "initial_state": "clear",
            "marked_states": ["clear", "blocked"],
            "transitions": [
                ["clear", "move_forward", "clear"],
                ["clear", "obstacle_front", "blocked"],
                ["blocked", "obstacle_front", "blocked"],
            ],
        }

    def test_fixed_mode_keeps_missing_motion_enabled(self):
        _, xml = build_xml(
            self.payload,
            "fallback",
            self.events,
            allowed_generated_events=set(self.events),
            complete_event_alphabet=self.events,
            protected_self_loop_events={"move_forward", "rotate_clockwise"},
        )
        text = xml.decode()
        self.assertIn('name="rotate_clockwise"', text)
        self.assertIn('source="1" target="1" event="0"', text)

    def test_llm_mode_preserves_missing_motion_restriction(self):
        _, xml = build_xml(
            self.payload,
            "fallback",
            self.events,
            allowed_generated_events=set(self.events),
            complete_event_alphabet=self.events,
        )
        text = xml.decode()
        self.assertIn('name="rotate_clockwise"', text)
        self.assertNotIn('source="1" target="1" event="2"', text)

if __name__ == "__main__":
    unittest.main()
