---
name: mimic-commit
description: Create git commits in the mimic-video repository with a short, title-only commit message and the Claude Co-Authored-By trailer. Use whenever asked to commit, make a commit, commit staged changes or "commit this".
---

# Committing changes in mimic-video

## Commit message

Keep the message short and to the point: **a single title line**, not a paragraph.

- Imperative mood, capitalized, no trailing period, at most about 72 characters. Examples: `Add SO-101 policy server and robot client`, `Make the base model rollout optional in the video eval`.
- No body. Do not list files or restate the diff. Add one extra line only if the title cannot carry something essential, such as a breaking change.
- End with a blank line and the Co-Authored-By trailer, naming the model that is running now (from the attribution system reminder, e.g. `Claude Opus 5.5` or `Claude Sonnet 5.5`):

```bash
git commit -q -F - <<'EOF'
Make the base model rollout optional in the video eval

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

## What to commit

1. Run `git status --short` and `git diff --cached --stat` to see what is staged and what is not.
2. If the user says "what I have staged", commit only the staged changes and leave the rest alone. Otherwise, stage only the files that belong to the change being committed, by path. Do not use `git add -A` or `git add .`.
3. Never commit `secrets.env`, checkpoints, `uv.lock` or other gitignored or generated files.
4. Commit directly on `main`. Do not create a branch unless the user asks for one.
5. Do not push unless the user asks.
6. After committing, show `git log -1 --stat --format='%h %s'` and report the hash to the user.
