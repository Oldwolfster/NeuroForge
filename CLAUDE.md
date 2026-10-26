# IPUI — Project CLAUDE.md

## Vision

IPUI is a structural engine for pygame UIs. The goal is to make Swift and Flutter look like COBOL and Fortran got drunk and made a baby. Pit-of-success design — the API shoves callers toward correct usage where defaults and structure make correct usage easier than incorrect usage.


## Architecture

- **Language/Runtime:** Python, pygame-ce
- **Project path:** `C:\SynologyDrive\Development\PyCharm\IdiotProofUIV3\`
- **Package structure:** `src/ipui/` for framework, `pyproject.toml`, `pip install -e .` for local dev
- **Source:** `src/` directory, `main.py` at root
- **Docs:** `docs/` at repo root with a `.bat` that copies `.md` files into `src/ipui/docs/` for runtime access
- **Automation:** `.bat` files for running outside PyCharm and for regression testing
- **Version control:** GitHub under username Oldwolfster

## Working Rules for Claude Code

- **You are already in the project root.** Never prefix shell commands with `cd`. Run grep/search commands directly.
- **Never prefix names with an underscore** — methods, attributes, or variables. Backing storage uses the `private_` prefix.
- **All code in classes. One class per file, named identically.**
- **Small methods.** 5 lines is ideal; 20 is a rare exception.
- **Vertically align** `=` and identical code sections where neighbors are aligned.
- **Prefer single-line forms.** Don't wrap short calls across lines for PEP 8 line length. When converting a multi-line call, keep its parentheses balanced.
- **Before deleting any method or file,** confirm zero callers across the repo, including `getattr`/`hasattr` and string references. If anything still references it, don't delete; report it.
- **Discuss before** changing an existing public API or adding extras beyond the request.
- **Pure refactors stay pure.** Don't mix behavior changes into a refactor.
- **Finish multi-step tasks with a report:** what changed, what was skipped, and why.

## Merging Code Packets from Claude.ai

Design happens in Claude.ai. Code arrives as a packet pasted into your
prompt. Your job is to integrate it exactly — not to redesign it.

**Packet format**
- Files are separated by banners: `class deleteme:  # ═══ File.py ═══`.
  The `class deleteme` wrapper exists only for indentation. Never paste it.
- Each method has a header: `#🔶 File.py  method: name  NEW|Update: why`.
- Line markers at column zero:
  `#🔶 NEW`, `#🔶 WAS: <old line>`, `#🔶 DELETING <line>`,
  `#🔶 MOVED FROM <method> — delete it there`, `#🔶 KEPT`.
- `#🔶 DELETE ENTIRELY:` blocks list things to remove.
- Non-method snippets (class attributes, imports) carry a placement note.

**Rules**
1. **Verify the base first.** Every `WAS:` and `DELETING` line must exist
   in the current method. If the current method differs from what the
   packet assumes in any other way, STOP and show me the difference.
   Never overwrite newer work.
2. `Update` = replace the existing method of that name in that file.
3. `NEW` = place it where the header says; otherwise beside related methods.
4. `MOVED FROM` = also delete it at the source.
   `DELETE ENTIRELY` = zero-caller check first, then delete.
5. Strip every 🔶 line and every `deleteme` wrapper.
   Done means `grep -rn 🔶 src/` returns nothing.
6. Don't improve packet code. If something looks wrong, ask.
7. If a section of the packet appears twice (a paste glitch), apply it
   once and mention it in the report.
8. Report: methods replaced, added, and deleted, plus anything that
   didn't match the packet.
9. Edits are scoped to the method named in the header. Never replace a
   matching line outside that method. If a WAS: line also appears in
   other methods, leave those alone and list them in the report.


## Core Design Principles

- **IoC (Inversion of Control)** is a core design principle throughout.
- **Construction IS Attachment.** No floating widgets or `add()` calls. If you build it inside a container, it's attached. No voids, no leaks.
- **Resolution Independent.** UI scales automatically to physical screen height.
- **Width is top-down, height is bottom-up.** Flex decides width; content decides height. Horizontal scrolling is explicitly out of scope for v0.1.
- **Decouple layout passes.** Mixing word wrap into combinatorial layout causes exponential complexity. Layout first, fix wraps second.
- **Fix framework bugs at the framework level.** Don't paper over systemic issues with per-pane workarounds.
- **Self-documenting.** Documentation stays accurate by reading the framework source code directly.

## Naming Conventions

- **`ip_` prefix:** Reserved for per-frame pane lifecycle hooks.
- **`on_` prefix:** Reserved for event properties (things you assign callbacks to).
- **`handle_` prefix:** For handler methods.
- **`ip_setup` / `ip_setup_pipeline`:** Setup lifecycle hooks.
- **`private_` prefix:** For backing storage attributes (e.g., `private_enabled`). Never Guido's underscore.
- **`public_` prefix:** Planned for public attrs. These prefixes enable future `public_attrs()`/`private_attrs()` helpers.
- **SCREAMING_CASE:** For structural class-level declaration dicts (`TAB_LAYOUT`, `BINDINGS`, `PIPELINE_DEFAULTS`).
- **Lowercase:** For configuration attributes.

## Key Systems

### Tab System
- Declarative `TAB_LAYOUT` dict as single source of truth.
- Lazy file discovery for tab/pane resolution.
- `MissingTabUI` generates helpful scaffolding when tab files don't exist yet.

### Pipeline System
- Centralized key-value store on `_BaseForm`.
- `PIPELINE_DEFAULTS` dict processed automatically before any tab builds.
- `initialize_pipeline()` override hook for computed defaults.
- `ip.cache` is local scratch (no reactive side effects); pipeline is the reactive system — never conflate them.

### State Machines
- State machines as delegates, not string containers.
- Each state maps to a delegate function that fires automatically on tick — no big if/elif blocks.

### Widget Registry
- Automatic. Access any widget via the global registry. Stop passing references around.

### Developer Tools (Magic Tab)
- Live Widget Catalog (refreshes from source code every run)
- Widget Tree view
- Magic debug sub-tabs: Pipeline (State Store), DAG (Reactivity Graph), Registries (Auto-Wiring), Event Log

## Runtime Details

- **`IP()` timing:** Created before any form or pane in `GameLoop.__init__`, stored as `IPUI.ip` at class level, eliminating per-frame reassignment.
- **dt clamping:** `min(self.clock.tick(60) / 1000.0, 0.05)` prevents background OS processes from causing large physics time jumps.

## Issue Tracking

Numbered issue list maintained in project conversations. Current range: #35–#250+.



## 🛠️ The IPUI Advantage

- 🗂️ **First-Class Tab System:** Define your app's tabs and pane layout from a single dictionary. IPUI scaffolds the structure and keeps each tab cleanly modular.
- 📱 **Resolution Independent:** UI scales automatically to physical screen height, so it stays usable on an old laptop or a 4K monitor.
- 📐 **Declarative Layout:** Simple, flexible syntax that handles the math so you can focus on the logic.
- 🧩 **Built to Extend:** Custom widgets get layout, events, and styling automatically. Standard widgets take 5–10 LOC; even tools like a network diagram widget come in under 150 LOC.
- 📜 **One-Touch Scrolling:** Make any Card scroll_v with a single parameter—no complex viewport setup required. Scrollbars are draggable and styled automatically.
- 🔗 **Construction IS Attachment:** No floating widgets or `add()` calls. If you build it inside a container, it's attached automatically.
- 🔄 **Multiple Update Styles:** Use DAG-based reactivity, pipeline-driven synchronization, or direct widget access—whichever fits the job best.
- ⛓️ **Data Pipeline:** Bind widgets to a Pipeline Key and let IPUI propagate updates automatically. Derives stay in sync with zero manual update code.
- 🎮 **Pygame Lifecycle Hooks:** `ip_think`, `ip_draw`, and `ip_draw_hud` give you full access to the game loop without fighting the framework.
- 💡 **Multi-Tier Tooltips:** Choose between standard hover tips or "Super Tooltips"—pinnable, scroll_v windows capable of displaying deep technical data.
- 🗃️ **Automatic Widget Registry:** When DAG or pipeline isn't the right fit, named widgets stay easy to reach across tabs and panes—no globals, no reference plumbing required.
- 🐞 **Pro Debug Mode:** Includes a live Widget Tree and layout overlays to make positioning issues easy to diagnose.
- 💻 **Beautiful Code Boxes:** Display source code by passing a string or a file path; IPUI handles the formatting.
- 🗺️ **Tab Map:** A bird's-eye view of your entire application for quick review and navigation.
- 📊 **Grid:** The most capable grid in the pygame ecosystem. Automatic sorting across pages, pagination, and SQL-ready. Feed it lists, dicts, or databases—it adapts to your workflow.
- 📚 **Self-Documenting:** Documentation stays in sync with the framework by reading the source code directly.
- 📈 **Live Matplotlib Charts:** Embed real-time updating research visuals directly in your pygame UI—useful for training curves, experiment monitoring, and diagnostics.

---

## Quick Start

IPUI is built to grow across files, but the fastest way to start is with **one file**.

Get it running first. Then let IPUI help you split things out as your app grows.

---

### Step 1: First Taste — Run in 30 Seconds

Save this as `SmokeTest.py` (or any name you like):
```python
# SmokeTest.py  New: one-class smoke test using form-as-pane
from ipui import *

class SmokeTest(_BaseForm):
    TAB_LAYOUT = {
        "Smoke Test"    :["go"],            # ← This one works immediately
        "Widgets"       :["demo"],          # ← Will trigger template picker
        "Relax"         :["sit", "chill"],  # ← Will trigger template picker
    }

    def go(self, parent):                    # ← matches "go" in TAB_LAYOUT
        Banner  (parent, "IPUI"              , text_align=CENTER, glow=True)
        Title   (parent, "Easy to get right!", text_align=CENTER)
        Body    (parent, "Hard to get wrong.", text_align=CENTER)
        Button  (parent, "Click Me :)"       , on_click=self.show_hello,
                 color_bg=Style.COLOR_PAL_GREEN_DARK)

    def show_hello(self): self.show_modal("Hello World!\nWelcome to IPUI")

if __name__ == "__main__": show(SmokeTest)
```

```bash
python SmokeTest.py
```

Three tabs appear immediately:
  - **Smoke Test**  — fully working with banner, text, and button
  - **Widgets**     — show IPUI’s friendly Houston helper card with template options
  - **Relax**       — show IPUI’s friendly Houston helper card with template options

---

### Step 2: Open Widgets — Let IPUI Forge the File

Change to the 'Widgets' tab.


> Smoke Test already has real content.
> Widgets and Relax do not have matching .py files yet.

Problem? Not even a little.

Instead of throwing an error or even showing an empty tab, IPUI steps in with a helper card:

<!-- SCREENSHOT: ipui/assets/images/houston_card.png — the Houston helper card offering to scaffold a missing tab -->

Pick Full Showcase on the Widgets tab. IPUI will create Widgets.py and hot-swap in a complete, interactive widget playground with real working controls (buttons, textboxes, cards, grids, etc.).

It’s not a dead stub — it’s live code you can immediately click, rearrange, and copy-paste from.

---

### Step 3: Customize and Scale

After IPUI generates the file (e.g. Widgets.py), it's packed with working examples. Trim it down to just what you need:

```python
from ipui import *

class Widgets(_BaseTab):
    def demo(self, parent):
        Title(parent, "Widget Playground")
        Button(parent, "Test Me", on_click=lambda: self.form.show_modal("Nice"))
```

Save the file — changes appear instantly.

This is the normal workflow:

1) Add (or modify) entries in TAB_LAYOUT
2) Let IPUI discover or generate the file(s)
3) Edit the builder methods on your _BaseTab class
4) Save and keep going

You can define pane methods directly inside _BaseForm (as in the smoke test) or in separate files — both work seamlessly.

---

### How Tab Discovery Works

The `TAB_LAYOUT` dictionary is the blueprint for your application. 
* **The Keys** define the names of your tabs.
* **The Values** are lists that divide that tab into one or more **Panes**.
* You can size panes by including a flex number (below, `chill` gets 2/3 and `None` gets 1/3)

```python
TAB_LAYOUT = {
    "Smoke Test"    :["go"],                        # Tab 'Smoke Test'  with one pane 'go'            
    "Widgets"       :["demo"],                      # Tab 'Widgets'     with one pane 'demo'
    "Relax"         :[("chill", 2)  , (None, 1)],   # Tab 'Relax'       with two panes 'chill' and a blank Pygame area
}
```
(Note: A pane value of None creates a blank region for you to draw directly to with Pygame!)

### What Panes Do

Each pane name in your TAB_LAYOUT maps to a builder method with the exact same name. 
**IPUI is highly flexible** and will look for that builder method in two places:

1. **The Main Form File** (Fastest)
Just like in SmokeTest.py, you can define the builder method directly inside your _BaseForm class. Perfect for quick prototypes.

2. **A Dedicated Tab File** (Most Scalable)
When a tab grows, you can move it to its own file. If you have a tab named "Hey There", IPUI will scan your project folder (and subfolders) for a file named Hey_There.py and HeyThere.py. 
Inside that file, IPUI looks for any class inheriting from _BaseTab. The actual class name doesn't matter!

```python
# Widgets.py
from ipui import *

# The class name can be anything, as long as it inherits from _BaseTab
class TotallyWhateverNameYouWant(_BaseTab):
    
    # This matches the 'demo' pane in TAB_LAYOUT
    def demo(self, parent):
        Title(parent, "Hello from Widgets.py")
```

### The Golden Rule: _BaseTab Wins

What happens if IPUI finds a demo() pane builder in both your main _BaseForm and an external Widgets.py file?

**The external** _BaseTab **file always wins**. This is deliberate. The main form is great for a fast start, but once a tab earns its own file, that file becomes the boss. If you extract a method into a new file and leave the old one behind, IPUI gracefully switches over to the new dedicated file.

### Why the `__name__` Guard Is Necessary

Your main file should always end with:
```python
if __name__ == "__main__": show(SmokeTest)    
```

Don't skip this! In a one-file setup, this standard Python guard prevents accidental re-entry during import.

---

### The Philosophy

IPUI makes the right path the easy path.

- Simple things should be trivial
- Missing structure should be fixable, not fatal
- Scaling out should feel natural
- Boilerplate should be forged, not copied around by hand
- Learning should happen by playing with real, running examples

> That's why the **Full Showcase** template gives you a fully functional widget gallery — click, rearrange, copy-paste, and keep building. Start stealing code before you've written your first line.

No event loop setup. No manual sizing. No coordinate math. IPUI handles the Pygame lifecycle, layout, rendering, and event dispatch automatically.



The ip Service Portal
Every lifecycle hook receives ip — one object that provides everything a pane needs. Type ip. and autocomplete shows the full API.
Families:

Identity: ip.form, ip.pane, ip.pane_name, ip.is_active_tab
Timing: ip.dt, ip.fps, ip.frame, ip.elapsed
Geometry: ip.rect_pane, ip.rect_tab_area, ip.rect_screen
Coordinate helpers: ip.to_screen(nx, ny), ip.to_local(sx, sy), ip.scale_x(n), ip.scale_y(n)
Mouse: ip.mouse_pos, ip.mouse_down/pressed/released("left"), ip.mouse_inside(widget), ip.mouse_inside_pane()
Keyboard: ip.key_down/pressed/released("space"), ip.mod_shift, ip.mod_ctrl, ip.mod_alt
Rendering: ip.surface, ip.events, ip.unhandled
Cache: ip.cache_get/set/has/del — scratch pad, NOT reactive. For reactive state use pipeline_set/read.
State machine: ip.state.set("NAME"), ip.state.current, ip.state.next(), ip.state.is_("NAME") — auto-transitions and flash messages via STATES dict
Discovery: ip.find("widget_name"), ip.help()