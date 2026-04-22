# Demo Assets

`demo.gif` is referenced from the top of the project README. Until it's
recorded, the README image link will render broken — that's intentional, so
the placeholder doesn't ship as a real screenshot.

## How to record `demo.gif`

15-20 seconds is the target. Show the value-add: type a question, watch the
right-hand panel light up.

1. Boot the demo locally:
   ```bash
   DIALOGUE_MODEL_MODE=mock python src/evaluation/human_eval_app.py
   ```
2. Open `http://localhost:7860`, pick the **blacksmith** character.
3. Start a recorder pointing at the browser viewport:
   - macOS: [Kap](https://getkap.co) (free) or `Cmd+Shift+5` → "Record selected portion"
   - Cross-platform: [LICEcap](https://www.cockos.com/licecap/) (free)
4. Record this sequence (~15s):
   - Type: `Got any swords for sale?`
   - Hit Send. Wait for response.
   - Open the **🔬 Trace Inspector** accordion.
   - Click **Refresh traces**.
   - Click the row to load detail.
5. Save as `docs/assets/demo.gif`. Aim for ≤5 MB — drop the framerate to 10 fps
   if needed. Tools like [gifski](https://gif.ski/) and [ezgif.com](https://ezgif.com)
   can compress an oversized GIF without obvious quality loss.
6. Commit:
   ```bash
   git add docs/assets/demo.gif
   git commit -m "docs: add glass-box demo GIF"
   ```

## Why a GIF instead of a video?

GitHub READMEs render GIFs inline. Embedded video requires hosting elsewhere
and a click-through. For a 15-second loop, a well-compressed GIF wins.
