# Presentation Workspace

## Structure
- `deck_outline.md` — agreed slide structure and asset checklist.
- `speaker_notes.md` — 10-minute speaking script.
- `assets/figures` — images for slides.
- `assets/tables` — exported tables/CSV snippets.
- `prompts/gemini_full_deck_prompt.md` — one-shot prompt for full deck generation.
- `prompts/gemini_key_slides_prompts.md` — targeted prompts for specific slides.

## Recommended Workflow
1. Finalize figures/tables in `assets/`.
2. Generate first deck draft in Gemini Slides using full-deck prompt.
3. Regenerate weak slides using key-slide prompts.
4. Copy final slide text back into notes if needed.
5. Run one rehearsal using `speaker_notes.md`.

## Immediate Next Inputs Needed
- Malta runtime metrics per tile/scene.
- Malta qualitative figures (building masks + multiclass + back-projection).
- Final acknowledgements details.
