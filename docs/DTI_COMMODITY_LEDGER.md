# DTI community, OSS, and commodity ledger

## Classification

This ledger records mechanisms that can be consumed without transferring planning or action authority to an external project. Project claims are attributed to their own documentation. Community game data remains provisional until observed in the live client. No exploit, account-farming, anti-cheat-bypass, or modified-client implementation is admitted.

## Existing ScreenGhost floor

PR #13 already supplies the governing mechanisms: teacher/runtime separation, model-free warm visual indexing, confidence and margin gates, app-transition graphs, run-scoped model timeouts, single-flight pending-to-settle-to-verify execution, idempotent receipts, and deterministic emulator fixtures. DTI is a cartridge over those seams.

Adopted:

- current-pixel target resolution rather than durable coordinates;
- visual variants and stable-region masks;
- graph reuse before model escalation;
- explicit unknown-screen state;
- one action in flight;
- visible postcondition receipts; and
- privileged teaching that cannot authorize runtime actions.

## Roblox platform boundary

Sources:

- https://devforum.roblox.com/t/an-update-on-using-third-party-emulators/3867040
- https://en.help.roblox.com/hc/en-us/articles/24275616578708-Anti-cheat-Messages
- https://en.help.roblox.com/hc/en-us/articles/203312450-Cheating-and-Exploiting
- https://en.help.roblox.com/hc/en-us/articles/203313410-Roblox-Community-Standards

Adopted:

- official native Windows client only;
- no modified client or emulator;
- no exploit or bypass path;
- no disruptive automation; and
- no unattended public play, automated voting, or farming.

The emulator restriction makes MEmu a test fixture, not the live DTI deployment target.

## Dress to Impress Freeplay mechanics

Community-maintained sources:

- https://dti-dress-to-impress.fandom.com/wiki/Freeplay_Mode
- https://dti-dress-to-impress.fandom.com/wiki/Quick_Teleport
- https://dti-dress-to-impress.fandom.com/wiki/Changelogs

Adopted:

- Freeplay as the first teaching venue because it has no ordinary theme timer;
- the on-demand Walk the Runway transaction as the first complete loop; and
- Quick Teleport labels as semantic route edges between the Dressing Room and Freeplay Runway.

These mechanics require live verification because the DTI map and UI continue to change.

## DXcam

Source:

- https://github.com/ra1nty/DXcam

License: MIT. Current project metadata identifies version `0.3.0` and Python 3.10 through 3.14 Windows wheels.

Adopted:

- Desktop Duplication / Windows Graphics Capture commodity;
- region capture for the verified Roblox client rectangle;
- timestamped ring-buffer path for later burst observation; and
- ordinary NumPy/Pillow output into ScreenGhost perception.

ScreenGhost retains window custody, profile geometry, frame hashing, settlement, and action authority.

## pyrobloxbot

Source:

- https://github.com/Mews/pyrobloxbot

License: MIT.

Adopted as mechanism:

- keyboard-first Roblox locomotion;
- short bounded holds instead of long monolithic macros; and
- a global failsafe concept.

Not adopted:

- multi-account operation;
- automatic session joining or rejoining;
- chat automation; and
- a package-level bot runtime above ScreenGhost.

The cartridge implements its own smaller SendInput adapter so the target-window, process, geometry, policy, and receipt guards remain one custody chain.

## HackMIT 2024 DTI AI stylist

Source:

- https://github.com/amyjun26/roblox_ai_assistant_DTI

License: MIT.

The project demonstrates a DTI-specific observation decomposition: fixed screen regions for the theme, equipped-item grid, runway header, player count, outfit image, and star delta; perceptual hashing for repeated icons; OCR for changing text; and separate theme-accuracy, color-coordination, creativity, and technical-fashion judges.

Adopted:

- DTI-specific observation regions as an enrollment concept;
- perceptual hashing for repeated stable icon checks;
- decomposed outfit criticism rather than one opaque score; and
- round receipts containing theme, outfit, player count, and result.

Not adopted:

- 2024 absolute pixel offsets;
- a hardcoded username;
- cloud-only OCR or judging;
- fixed UI assumptions without profile/version receipts; and
- a judge score as motor authority.

## OmniParser and Windows desktop-agent practice

Sources:

- https://github.com/microsoft/OmniParser
- https://github.com/ahmetdenizyilmaz/desktop-control-mcp

Adopted as mechanism:

- per-monitor DPI awareness;
- explicit active-window reporting;
- time-spread screenshot bursts for animation and settlement;
- separate OCR and interactable-region detection; and
- annotated set-of-marks only as an escalation or teaching surface.

OmniParser is not a mandatory DTI dependency. Known screens should terminate at the local visual index. A GUI grounder is an escalation for an unknown or moved target.

## Direct-input libraries

Sources:

- https://github.com/ReggX/pydirectinput_rgx
- https://github.com/Moon-Playground/fisch-angler

Adopted as contingency:

- scan-code-oriented keyboard input is often more reliable in games than high-level GUI typing;
- native Windows OCR and fuzzy matching can be useful for small, changing text regions; and
- low-latency region capture should be decoupled from the planner.

The v0 driver uses Win32 SendInput directly. `pydirectinput-rgx` remains a replaceable fallback only if a physical Roblox campaign demonstrates an input compatibility gap. No kernel driver or anti-cheat bypass is admissible.

## DTI community databases and guides

Candidate sources:

- https://dress-to-impress.wiki/
- https://dtiguide.com/
- other community theme, item, code, and outfit indexes

Adopted as data schema:

- theme canonical name, aliases, interpretation, palette, silhouette, motifs, mistakes, and last-verified time;
- item ID, slot, source, location, access tier, toggles, pattern support, update history, and related outfits; and
- complete outfit recipe containing base, layer, shoes, hair, makeup, colors, pose, and optional details.

Community records are imported into `ThemeCard` and `WardrobeItem` only after provenance labeling. They cannot supply action coordinates. A current local demonstration binds a semantic record to the live visual surface.

## Rejected classes

The following are excluded from the floor even when source is available:

- Lua executors, remote loaders, memory readers, and injected Roblox scripts;
- emulator-based general-player deployment;
- kernel input drivers marketed for anti-cheat bypass;
- anti-idle, auto-rejoin, currency, rank, code-redemption, or reward farming;
- multi-account orchestration;
- captcha solving;
- automatic public voting or chat; and
- opaque binaries without inspectable source and a compatible license.

## Commodity boundary

The community may own the current theme list, item list, map labels, fashion examples, and practical shortcuts. DXcam owns fast Windows capture. Windows owns ordinary input. GUI-agent projects supply useful grounding and failsafe patterns. ScreenGhost owns the durable transaction: current pixels, semantic intent, bounded action, visible settlement, receipt, refusal, and rollback to human control.
