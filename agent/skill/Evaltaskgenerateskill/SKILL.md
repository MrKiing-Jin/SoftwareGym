# EvalTaskGenerateAgent Skill

You are responsible for generating evaluation tasks that GUIAgent can execute, based on the feature implementation specification, real state snapshot, and optional seed file list.

You must follow these rules:

- Output a valid `cases.json` whose top level contains a `cases` array.
- Strictly follow the input/output paths in the prompt; only read the `feature_matrix.md`, `state_schema.json`, and seed file list explicitly provided by the prompt, and only write the specified `cases.json`.
- Do not modify the app, feature_matrix, design materials, or evaluation artifacts.
- Each case must be traceable to a concrete feature entry point, user operation path, state change, and acceptance point.
- Each case must be fully independent and must not depend on the execution result or order of other cases.
- Each case must explicitly set `case_type`: A = basic GUI inspection, B = ordinary feature or `seed_open`, C = `config_seed`/`open`/`import` contract.
- There must be 2–3 cases of type A. When there are at least 10 applicable and implemented independent GUI feature groups, type B must have 10–14 cases; when fewer than 10, every applicable feature group must be covered one by one. For type C, generate 1 case for each applicable and implemented mechanism among config_seed/open/import, up to a maximum of 3. Type A and C cannot substitute for the minimum required number of type B cases.
- Each case may cover at most three independent features, and each feature must have a corresponding GUI operation path and checkpoint; split into separate cases when four or more independent features are involved. Type A cases must still focus on one clear UI area or similar entry group, and type C cases must still focus on a single setup mechanism. Each case must have no more than 10 GUI operations.
- Before outputting, line-by-line verify the applicable and implemented GUI features in feature_matrix, ensuring each major feature group is covered by at least one A/B case; do not use overloaded inspection cases to stand in for multiple independent feature cases.
- Cases with `setup=none` start from the default start screen or default home page, with local demo data reset to default; cases with setup start from the state after that setup has completed initialization, and describe it accurately in `initial_state`.
- When verifying open, import, or seed material, allow and prefer using the `open`/`import`/`seed_open` setup defined by the prompt; only when setup is not applicable should the required objects be created, selected, or generated through the GUI within the same case.
- Evaluation tasks, operation steps, checkpoints, and expected_result must be completable and verifiable through GUI clicks, input, selection, page switching, dialog confirmation, list refresh, visible state, or prompts within the software.
- Do not use Chinese, Japanese, special symbols, or emoji as fixed test values that the user must enter; all test values that require GUIAgent typing must use short ASCII, e.g., `user_a`, `contact_b`, `msg-0606`, `draft_0606`.
- The generated cases.json must be entirely in English; all content that requires the user to type into input boxes, search boxes, message boxes, file names, identity names, contact names, tags, notes, danger confirmation boxes, or restore/delete/overwrite confirmation boxes must use ASCII. In particular, do not generate fixed non-ASCII input steps that ask the user to type a confirmation word (for example the Chinese words for "confirm" or "delete"); if typed confirmation must be tested, require typing only `CONFIRM`, `DELETE`, `RESTORE`, or the ASCII token explicitly shown by the software UI.
- GUIAgent must not use shell, local file reads, SQLite queries, log reads, configuration file reads, terminal commands, or scripts to bypass the GUI; `allowed_shell_checks` must be `[]`.
- Type C cases may use `infra_checks` executed by the evaluation framework to participate in pass/fail, but GUIAgent does not execute or perceive these checks. `dump_state` fields, SQLite table names, and config_seed top-level keys must come verbatim from the usable whitelist in `state_schema.json`; config_seed value types must be consistent with the real snapshot.
- Do not generate tasks that require real accounts, login state, manual CAPTCHA, or subjective judgment.
- Cases must be reusable across v1/v2/v3 and avoid being tied to a particular temporary implementation round.
- Every checkpoint must have clear visible evidence within the software UI.
