# AGENTS.md

This repository implements CT + IQL for causal intervention planning.

When modifying CT+IQL:
1. Preserve backward compatibility. New methods must be controlled by Hydra config flags.
2. Do not modify CT encoder internals unless explicitly requested.
3. Do not redesign reward/cost when implementing DW-IQL.
4. DW-IQL should be implemented as density-ratio loss weighting for IQL.
5. Keep reward clipping, reward scaling, gradient clipping, and exp_adv clipping.
6. Add diagnostics for all new weighting behavior.
7. Do not remove existing validation/checkpoint selection logic.
8. Prefer small modular files over large rewrites.