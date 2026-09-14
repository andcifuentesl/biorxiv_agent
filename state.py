import json
from pathlib import Path
from typing import Optional

from config import STATE_FILE


class StateManager:
    def __init__(self, state_file: Optional[str] = None):
        self.state_file = Path(state_file or STATE_FILE)
        self._state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "last_check_date": "",
            "processed_dois": [],
            "selected_dois": [],
            "downloaded_dois": [],
        }

    def save(self) -> None:
        try:
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save state: {e}")

    def is_processed(self, doi: str) -> bool:
        return doi in self._state.get("processed_dois", [])

    def mark_processed(self, doi: str) -> None:
        if "processed_dois" not in self._state:
            self._state["processed_dois"] = []
        if doi not in self._state["processed_dois"]:
            self._state["processed_dois"].append(doi)
            if len(self._state["processed_dois"]) > 10000:
                self._state["processed_dois"] = self._state["processed_dois"][-5000:]
            self.save()

    def is_selected(self, doi: str) -> bool:
        return doi in self._state.get("selected_dois", [])

    def mark_selected(self, doi: str) -> None:
        if "selected_dois" not in self._state:
            self._state["selected_dois"] = []
        if doi not in self._state["selected_dois"]:
            self._state["selected_dois"].append(doi)
            self.save()

    def is_downloaded(self, doi: str) -> bool:
        return doi in self._state.get("downloaded_dois", [])

    def mark_downloaded(self, doi: str) -> None:
        if "downloaded_dois" not in self._state:
            self._state["downloaded_dois"] = []
        if doi not in self._state["downloaded_dois"]:
            self._state["downloaded_dois"].append(doi)
            self.save()