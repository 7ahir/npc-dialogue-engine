# Deploying the Demo to Hugging Face Spaces

The files in this directory are everything a HF Space needs to host the
glass-box Gradio demo on free CPU hardware (mock-model mode).

## One-time setup

1. Create a new Space at https://huggingface.co/new-space
   - **Owner:** `7ahir`
   - **Space name:** `npc-dialogue-engine`
   - **License:** MIT
   - **SDK:** Gradio
   - **Hardware:** CPU basic (free)

2. Clone the empty Space locally (separate from this repo):
   ```bash
   git clone https://huggingface.co/spaces/7ahir/npc-dialogue-engine /tmp/npc-space
   cd /tmp/npc-space
   ```

3. Copy the Space files **plus the source the app imports** from this repo:
   ```bash
   REPO=/Users/tahiro/projects/npc-dialogue-engine
   cp $REPO/deploy/huggingface-space/{README.md,app.py,requirements.txt} .
   cp -r $REPO/src .
   cp -r $REPO/configs .
   cp -r $REPO/data .
   ```

4. Commit and push — HF will build and deploy automatically:
   ```bash
   git add .
   git commit -m "Initial Space deployment"
   git push
   ```

## Updates

The Space is a copy, not a submodule, so re-run the `cp` step in step 3 and
push again whenever the upstream repo changes the UI or pipeline.

If the upstream `human_eval_app.py` signature or pipeline imports change,
update `app.py` in this directory to match and copy it back into the Space.

## Why a separate Space repo?

HF Spaces require their config (`README.md` frontmatter, `app.py` at the root,
`requirements.txt`) to live at the *repo* root, which would conflict with
this project's own README and packaging. Keeping the Space as a sibling repo
avoids that conflict and keeps the main repo's structure clean.
