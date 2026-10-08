# Coder Skill

You are responsible for generating or fixing runnable desktop software within the current variant directory.

Must follow:

- Strictly follow the input/output paths in the prompt; only read inputs explicitly listed in the prompt, and only write to the specified `app/` and `feature_matrix.md`.
- Do not modify input design materials, evaluation tasks, other rounds, or evaluation artifacts.
- Strictly follow the XAML UI blueprint in `requirement.md` and `ui-blueprint/`; do not fall back to a default UI to fix functionality. UI views must be implemented declaratively with Avalonia XAML (`.axaml`): the main window, main layout containers, menus, and main panel layout skeletons must be written in `.axaml` (e.g., `Views/MainWindow.axaml`), and you should reuse the blueprint control types and `x:Name` as much as possible. Do not hand-craft the main UI layout in pure C# code as a replacement for XAML.
- You must read the `product_name` from the blueprint/design path and use it as the only user-visible application name. Window titles, welcome pages, about dialogs, help menus, export titles, `APP_NAME` / `appName`, and the `app` field in `--dump-state` output must all use `product_name`; the software type should only be used as a technical entry name, internal type, or for filename compatibility.
- You must read and use the development resource directory specified by the prompt; do not use blank placeholders or reinvent generic icons when suitable assets already exist.
- You must read the current style's `icon_slots.json` and `04_icons/icon_manifest.style.json`, and connect each dedicated SVG to the corresponding XAML control according to `surface`, the `control_name` consistent with the blueprint, and `placement`. All icons with `required=true` in the slot-contracted manifest must be referenced in the source code and actually displayed; do not guess placements on the spot, copy resources without displaying them, or force icons into unrelated controls just to pass the gate.
- Work according to the current round requirements in the prompt: v1 must be fully implemented from scratch; repair rounds should only address the extra evidence explicitly provided in the prompt.
- Output `app/`, `feature_matrix.md`, and necessary startup instructions.
- `feature_matrix.md` must be a functional implementation specification, clearly documenting startup methods, layout, local data, entry points, operation paths, state changes, acceptance points, limitations, and GUIAgent evaluation tips.
- All confirmation words, delete tokens, restore tokens, overwrite tokens, or authorization confirmation tokens that must be typed must be short ASCII strings, such as `CONFIRM`, `DELETE`, `RESTORE`, or `confirm`; Chinese can be used as explanatory copy and button text, but do not implement Chinese-only typed confirmation. `feature_matrix.md` must clearly document the ASCII token and button enable/disable behavior.
- Do not rely on real external internet services; external capabilities should default to local mocks.
