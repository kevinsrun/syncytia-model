# Repository Instructions

This repository is managed with Engineering Fleet. These instructions extend
the global Fleet doctrine; they do not replace it.

## Architecture

- Codex is the architectural decision gate and final technical reviewer.
- Workers operate only within explicitly assigned scope.
- Record important decisions and their evidence in version control.

## Verification

- Run `fleet verify` before proposing or promoting changes.
- A worker claim is not proof. Treat only recorded deterministic verification
  or identified production evidence as proof.

## Repository-specific constraints

- Add project boundaries, generated-file rules, and risk-sensitive paths here.
