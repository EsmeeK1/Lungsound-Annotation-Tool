from __future__ import annotations

from typing import Any

from ..app_settings import METADATA_FIELDS, save_prefs


class MetadataMixin:
    """
    Handles recording metadata.

    Current supported metadata fields:
    - environment
    - notes

    Environment is remembered during the current session and added to the
    environment dropdown as a recent value.
    """

    def _on_meta_inline_changed(self: Any, values: dict) -> None:
        """
        Update file metadata when the inline editor changes.

        Only supported metadata fields are stored.
        Empty fields are removed.

        Environment is also saved as a session default, so the next loaded file
        automatically gets the last used environment unless it already has one.
        """
        if not self.state:
            return

        vals = dict(values or {})

        # Keep only supported, non-empty metadata fields.
        vals = {
            key: value
            for key, value in vals.items()
            if key in METADATA_FIELDS and str(value).strip() != ""
        }

        self.state.meta = dict(self.state.meta or {})

        # Remove managed metadata fields first, so clearing a field removes it.
        for key in METADATA_FIELDS:
            self.state.meta.pop(key, None)

        # Remove old metadata keys from previous versions.
        for old_key in (
            "subject_id",
            "microphone_type",
            "sample_rate",
            "location",
            "recording_id",
            "source_type",
            "device_type",
            "recording_location",
        ):
            self.state.meta.pop(old_key, None)

        self.state.meta.update(vals)

        # Session default: remember last environment for following files.
        environment = vals.get("environment")
        if environment:
            self.session_meta["environment"] = environment
            self.bump_recents(loc=str(environment))

        self.save_json()

    def bump_recents(
        self: Any,
        mic: str | None = None,
        loc: str | None = None,
    ) -> None:
        """
        Add recent metadata values and persist preferences.

        The 'mic' argument is kept only for backward compatibility with older
        code paths. The current metadata workflow only uses 'loc' as recent
        environment.
        """
        changed = False

        # Backward compatibility only.
        if mic and hasattr(self.prefs, "recents_mic_types"):
            if mic in self.prefs.recents_mic_types:
                self.prefs.recents_mic_types.remove(mic)
            self.prefs.recents_mic_types.append(mic)
            self.prefs.recents_mic_types = self.prefs.recents_mic_types[-8:]
            changed = True

        # Current behavior: recents_locations stores recent environments.
        if loc:
            if loc in self.prefs.recent_environments:
                self.prefs.recent_environments.remove(loc)

            self.prefs.recent_environments.append(loc)
            self.prefs.recent_environments = self.prefs.recent_environments[-8:]
            changed = True

        if not changed:
            return

        save_prefs(self.prefs)

        self._refresh_location_choices()

    def _refresh_location_choices(self: Any) -> None:
        """
        Rebuild the environment dropdown from defaults + recent environments.

        The method name is kept as _refresh_location_choices for compatibility
        with existing calls. Later, after the refactor is complete, it can be
        renamed to refresh_environment_choices().
        """
        defaults = [
            "",
            "indoor",
            "outdoor",
            "quiet",
            "noisy",
            "traffic",
            "home",
            "lab",
            "clinical",
            "other",
        ]

        recent_environments = getattr(self.prefs, "recent_environments", [])
        merged = list(dict.fromkeys(defaults + list(recent_environments)))

        self.meta_inline.set_recent_locations(merged)