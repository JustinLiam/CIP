"""Disabled legacy two-stage IQL trainer.

The paper configuration now uses the one-stage CT+IQL EM trainer in
``runnables/train_ct_iql_em.py``. Keeping the old two-stage entrypoint runnable
would make it too easy to produce checkpoints with incompatible representation
learning semantics, so this module fails fast with an explicit message.
"""


def main() -> None:
    raise SystemExit(
        "The legacy two-stage IQL trainer is disabled. "
        "Use `python runnables/train_ct_iql_em.py` for the current one-stage method."
    )


if __name__ == "__main__":
    main()
