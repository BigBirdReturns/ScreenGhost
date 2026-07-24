# AndroidWorld adapter

`AndroidWorldBackend` is an optional bridge over the current AndroidWorld environment interface. It converts the environment's synchronized pixels and processed accessibility elements into the same runtime projection and temporal-alignment nodes used by PhoneWorld.

The adapter supports ordinary click, text, and Back actions for future task experiments, but the shipped `androidworld-smoke` command is read-only and records zero actions. AndroidWorld remains an external optional dependency under its own Apache-2.0 license.

The adapter deliberately imports AndroidWorld only when invoked. Core ScreenGhost tests and the deterministic campaign do not depend on it.
