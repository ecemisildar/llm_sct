import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_input import (
    add_allowed_event_list,
    add_event_meanings,
    add_existing_automata_context,
    combine_prompt,
    normalize_llm_event_aliases,
    save_llm_prompt,
    validate_compact_response,
)


class PromptCompositionTests(unittest.TestCase):
    def test_compact_policy_response_is_accepted(self) -> None:
        validate_compact_response({
            "automata": [{
                "name": "exploration_specification",
                "initial_state": "moving",
                "marked_states": ["moving"],
                "transitions": [["moving", "move_forward", "moving"]],
            }]
        })

    def test_only_user_input_and_fixed_automata_vary(self) -> None:
        base = "INVARIANT INSTRUCTIONS"
        context = {"fixed_automata": [{"name": "task_plant"}]}

        prompt = add_existing_automata_context(
            combine_prompt(base, "Visit red then blue"), context
        )

        self.assertTrue(prompt.startswith(base))
        self.assertIn("USER INPUT\nVisit red then blue", prompt)
        self.assertIn('"name": "task_plant"', prompt)
        self.assertNotIn("DELIVERY RUNTIME REQUIREMENTS", prompt)
        self.assertNotIn("AUTHORITATIVE EVENT LIST", prompt)
        self.assertNotIn("PREVIOUS RUN FEEDBACK", prompt)

    def test_empty_user_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "control task is empty"):
            combine_prompt("INVARIANT INSTRUCTIONS", "  ")

    def test_complete_prompt_is_saved_beside_output(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "generation.json"
            saved = save_llm_prompt("complete prompt", output)

            self.assertEqual(saved.name, "generation.llm_prompt.txt")
            self.assertEqual(saved.read_text(encoding="utf-8"), "complete prompt\n")

    def test_authoritative_events_are_split_by_controllability(self) -> None:
        prompt = add_allowed_event_list(
            "COMPLETE PROMPT",
            {"search_color": True, "red_visible": False},
        )

        self.assertIn("AUTHORITATIVE EVENT LIST", prompt)
        self.assertIn('Controllable events: ["search_color"]', prompt)
        self.assertIn(
            'Uncontrollable observation events: ["red_visible"]', prompt
        )

    def test_only_allowed_event_meanings_are_added(self) -> None:
        prompt = add_event_meanings(
            "BASE", {"move_forward": True, "red_visible": False}
        )

        self.assertIn("`move_forward`", prompt)
        self.assertIn("`red_visible`", prompt)
        self.assertNotIn("`move_backward`", prompt)
        self.assertNotIn("`blue_visible`", prompt)

    def test_u_turn_is_exposed_without_aliasing(self) -> None:
        prompt = add_allowed_event_list("PROMPT", {"u_turn": True})
        self.assertIn('Controllable events: ["u_turn"]', prompt)

        payload = {
            "automata": [{
                "events": ["u_turn"],
                "transitions": ['("ready", "u_turn", "ready")'],
            }],
            "explanation": {"event_selection": "Use u_turn."},
        }
        normalized = normalize_llm_event_aliases(payload)
        self.assertEqual(normalized["automata"][0]["events"], ["u_turn"])
        self.assertEqual(
            normalized["automata"][0]["transitions"],
            ['("ready", "u_turn", "ready")'],
        )
        self.assertEqual(
            normalized["explanation"]["event_selection"], "Use u_turn."
        )


if __name__ == "__main__":
    unittest.main()
