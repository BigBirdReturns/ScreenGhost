# Vendor source ledger

This ledger records the public vendor behavior that each provider or baseline
assumption depends on. The implementation does not scrape these pages at runtime.

## MEmu

- MEMUC command reference:
  `https://www.memuplay.com/blog/memucommand-reference-manual.html`
  - instance inventory, create, clone, import, export, start, stop, remove;
  - configuration through `getconfigex` and `setconfigex`;
  - application control and wrapped ADB commands.
- Operation Recorder:
  `https://www.memuplay.com/blog/en/how-to-use-operation-record.html`
  - record, replay settings, import/export, composition, cycles, intervals;
  - recommends restoring the original state before replay;
  - exposes optional random click jitter and stop-on-return-to-desktop behavior.
- Synchronizer:
  `https://www.memuplay.com/blog/how-to-use-synchronizer.html`
  - broadcasts operations across instances and recommends matching geometry.

Implementation consequence: MEmu lifecycle and ADB are automated only through the
documented MEMUC interface. `.mir` action bytes remain opaque because no stable
documented action schema or headless playback command is claimed.

## LDPlayer

- Command-line interface:
  `https://www.ldplayer.net/blog/introduction-to-ldplayer-command-line-interface.html`
  - `list2`, launch, quit, add, copy, remove, modify, application control, ADB;
  - instance selection by name or index.
- Keyboard macro language:
  `https://www.ldplayer.net/blog/introduction-to-keyboard-macro.html`
  - `size`, `touch`, `wait`, `press`, `release`, `key`, `text`, and control syntax.
- Synchronizer:
  `https://www.ldplayer.net/blog/introduction-to-synchronizer.html`
  - mirrors clicks, drags, and typing across multiple instances;
  - recommends consistent resolution and DPI.

Implementation consequence: LDPlayer text macros are parsed conservatively.
Unsupported control programs remain explicit and prevent distillation.

## BlueStacks

- Multi-instance Manager:
  `https://support.bluestacks.com/hc/en-us/articles/360052834092-How-to-create-and-manage-instances-using-the-Multi-instance-Manager-on-BlueStacks-5`
- Operation synchronization:
  `https://support.bluestacks.com/hc/en-us/articles/4403071374861-How-to-sync-operations-across-BlueStacks-5-instances`
- Macro Manager:
  `https://support.bluestacks.com/hc/en-us/articles/360056012412-How-to-record-and-manage-macros-using-the-Macro-manager-on-BlueStacks-5`
- Eco Mode:
  `https://support.bluestacks.com/hc/en-us/articles/360052834772-How-to-run-multiple-instances-of-BlueStacks-5-more-efficiently-using-Eco-mode`

Implementation consequence: exported macro JSON may be imported as demonstration
evidence. Lifecycle and synchronization remain operator-controlled through
supported product surfaces because this package does not invoke undocumented
executables.
