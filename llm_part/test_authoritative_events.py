#!/usr/bin/env python3
"""Regression tests for pre-conversion generated-event validation."""

from __future__ import annotations

import unittest

from llm_input import authoritative_event_errors


class AuthoritativeEventValidationTest(unittest.TestCase):
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
