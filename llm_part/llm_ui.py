#!/usr/bin/env python3
"""Small desktop interface for generating SCT automata JSON with OpenAI."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from llm_input import DEFAULT_MODEL
from run_pipeline import run_pipeline


class LLMTaskApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SCT Automata Generator")
        self.root.geometry("820x650")
        self.root.minsize(650, 500)
        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=2)
        frame.rowconfigure(7, weight=1)

        ttk.Label(frame, text="Control task").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="Describe the behavior and requirements; the LLM will select relevant events.",
        ).grid(row=1, column=0, sticky="w", pady=(2, 6))

        self.task_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=12
        )
        self.task_text.grid(row=2, column=0, sticky="nsew")
        self.task_text.focus_set()

        options = ttk.Frame(frame)
        options.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(options, textvariable=self.model_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )

        self.generate_button = ttk.Button(
            frame, text="Generate JSON", command=self.start_generation
        )
        self.generate_button.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        self.status_var = tk.StringVar(value="Enter a control task")
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=5, column=0, sticky="w", pady=(8, 8)
        )

        ttk.Label(frame, text="Generated JSON").grid(row=6, column=0, sticky="w")
        self.output_text = scrolledtext.ScrolledText(
            frame, wrap=tk.NONE, height=12, state=tk.DISABLED
        )
        self.output_text.grid(row=7, column=0, sticky="nsew", pady=(6, 0))

    def start_generation(self) -> None:
        task = self.task_text.get("1.0", tk.END).strip()
        model = self.model_var.get().strip()
        if not task:
            messagebox.showwarning("Missing task", "Enter a control task first.")
            self.task_text.focus_set()
            return
        if not model:
            messagebox.showwarning("Missing model", "Enter an OpenAI model ID.")
            return

        self.generate_button.config(state=tk.DISABLED)
        self.status_var.set("Starting pipeline…")
        self._set_output("")
        threading.Thread(
            target=self._generate_worker,
            args=(task, model),
            daemon=True,
        ).start()
        self.root.after(100, self._poll_worker)

    def _generate_worker(self, task: str, model: str) -> None:
        try:
            result = run_pipeline(
                task=task,
                model=model,
                status=lambda message: self.ui_queue.put(("status", message)),
            )
        except Exception as error:
            self.ui_queue.put(("error", str(error)))
            return
        preview = json.dumps(result.payload, indent=2, ensure_ascii=False)
        self.ui_queue.put(("success", (preview, result.yaml_path)))

    def _poll_worker(self) -> None:
        finished = False
        while True:
            try:
                message_type, value = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if message_type == "status":
                self.status_var.set(str(value))
            elif message_type == "success":
                preview, yaml_path = value
                self._generation_finished(preview, yaml_path)
                finished = True
            elif message_type == "error":
                self._generation_failed(str(value))
                finished = True
        if not finished:
            self.root.after(100, self._poll_worker)

    def _generation_finished(self, preview: str, yaml_path: Path) -> None:
        self._set_output(preview)
        self.status_var.set(f"Pipeline complete — YAML: {yaml_path}")
        self.generate_button.config(state=tk.NORMAL)

    def _generation_failed(self, message: str) -> None:
        self.status_var.set("Generation failed")
        self.generate_button.config(state=tk.NORMAL)
        messagebox.showerror("Generation failed", message)

    def _set_output(self, value: str) -> None:
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", value)
        self.output_text.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    LLMTaskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
