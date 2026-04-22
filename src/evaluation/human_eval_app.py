#!/usr/bin/env python3
"""Gradio-based human evaluation app for NPC dialogue.

"Glass box" demo showing not just the chatbot output but the full
pipeline internals: intent classification, sentiment, retrieved lore
chunks, latency breakdown, and optional Tree of Thoughts candidates.

Usage:
    python src/evaluation/human_eval_app.py
    DIALOGUE_MODEL_MODE=mock python src/evaluation/human_eval_app.py
"""

import time

import gradio as gr

from src.models.dialogue_model import create_dialogue_model
from src.models.intent_classifier import IntentClassifier
from src.pipeline.context_manager import ContextManager
from src.pipeline.dialogue_pipeline import DialoguePipeline, DialogueResponse
from src.pipeline.prompt_templates import CharacterLoader, PromptBuilder
from src.rag.embeddings import EmbeddingService
from src.rag.lore_indexer import LoreIndexer
from src.rag.retriever import LoreRetriever
from src.utils.config import get_config
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _build_pipeline() -> tuple[DialoguePipeline, ContextManager, CharacterLoader]:
    """Initialize the full dialogue pipeline."""
    config = get_config()
    model = create_dialogue_model(config.model)
    context_manager = ContextManager()
    prompt_builder = PromptBuilder()
    character_loader = CharacterLoader()

    # Try to initialize retriever; graceful fallback if ChromaDB not available.
    # Auto-index lore on first boot so fresh deployments (e.g. HF Spaces) get
    # RAG working without requiring a manual `npc-index-lore` step.
    try:
        retriever = LoreRetriever()
        try:
            indexer = LoreIndexer(embedding_service=EmbeddingService())
            # index_directory is idempotent — it deletes + re-adds if the
            # collection already has docs, so calling on every boot is safe
            # but wasteful. Skip if collection already populated.
            existing = retriever._client.get_or_create_collection(  # noqa: SLF001
                name=config.rag.collection_name
            ).count()
            if existing == 0:
                logger.info("auto_indexing_lore_on_boot")
                indexer.index_directory()
        except Exception as exc:  # pragma: no cover — best-effort
            logger.warning("auto_index_failed", error=str(exc))
    except Exception:
        logger.warning("retriever_init_failed", msg="Running without RAG")
        retriever = None

    # Try to initialize classifier; graceful fallback
    try:
        classifier = IntentClassifier()
    except Exception:
        logger.warning("classifier_init_failed", msg="Running without intent classification")
        classifier = None

    pipeline = DialoguePipeline(
        model=model,
        retriever=retriever,
        classifier=classifier,
        prompt_builder=prompt_builder,
        context_manager=context_manager,
        config=config,
    )

    return pipeline, context_manager, character_loader


# ─── Global State ─────────────────────────────────────────────────

pipeline, context_manager, character_loader = _build_pipeline()


def get_character_choices() -> list[str]:
    """Get available character IDs."""
    config = get_config()
    return [p.stem for p in config.characters_dir.glob("*.yaml")]


def get_character_info(character_id: str) -> str:
    """Get character description for display."""
    try:
        data = character_loader.load(character_id)
        traits = ", ".join(data.get("personality_traits", []))
        phrases = "\n".join(f'  • "{p}"' for p in data.get("example_phrases", [])[:3])
        return (
            f"**{data['name']}** — {data['role']}\n\n"
            f"{data.get('description', '')}\n\n"
            f"**Traits:** {traits}\n\n"
            f"**Example speech:**\n{phrases}"
        )
    except Exception:
        return f"Character: {character_id}"


def chat(
    message: str,
    history: list[dict],
    character_id: str,
    session_id: str,
    use_tot: bool,
) -> tuple[list[dict], str, str]:
    """Process a chat message and return updated history + debug info.

    Returns:
        Tuple of (chat_history, debug_panel_text, latency_text)
    """
    if not message.strip():
        return history, "", ""

    # Process through pipeline
    start = time.perf_counter()
    response: DialogueResponse = pipeline.process(
        player_message=message,
        character_id=character_id,
        session_id=session_id,
        use_tot=use_tot,
    )
    total_ms = (time.perf_counter() - start) * 1000

    # Update chat history
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response.npc_response},
    ]

    # Build debug panel
    debug_lines = [
        "## 🔍 Pipeline Internals\n",
        f"**Intent:** `{response.intent}` (confidence: {response.confidence:.2%})",
        f"**Sentiment:** {response.sentiment:+.2f}",
        f"**Model:** `{response.model_version}`",
        f"**Tree of Thoughts:** {'enabled' if use_tot else 'disabled'}",
        "",
        "### Retrieved Lore",
    ]

    if response.lore_refs:
        for ref in response.lore_refs:
            debug_lines.append(f"  • `{ref}`")
    else:
        debug_lines.append("  _(no lore retrieved)_")

    debug_lines.extend(
        [
            "",
            "### Latency Breakdown",
            f"  • Total: **{total_ms:.0f}ms**",
            f"  • Pipeline internal: **{response.latency_ms:.0f}ms**",
        ]
    )

    # Session info
    session = context_manager.get_or_create_session(session_id, character_id)
    debug_lines.extend(
        [
            "",
            "### Session",
            f"  • ID: `{session_id}`",
            f"  • Turns: {len(session.history) // 2}",
        ]
    )

    debug_text = "\n".join(debug_lines)
    latency_text = f"⏱ {total_ms:.0f}ms"

    return history, debug_text, latency_text


def reset_session(session_id: str) -> tuple[list, str, str]:
    """Clear chat history and session."""
    context_manager.reset_session(session_id)
    return [], "", ""


# ─── Trace Inspector ──────────────────────────────────────────────


def get_trace_summary_md() -> str:
    """Render the aggregate p50/p95 summary as a Markdown block."""
    summary = pipeline.trace_store.summary()
    if summary["count"] == 0:
        return "_No traces recorded yet — send a message in the **Chat** tab._"

    total = summary["total_ms"]
    lines = [
        f"### Aggregate latency — last {summary['count']} request(s)",
        "",
        f"**Total**: p50 `{total['p50']}ms` · p95 `{total['p95']}ms` · max `{total['max']}ms`",
        "",
        "**Per-stage breakdown:**",
        "",
        "| Stage | count | p50 (ms) | p95 (ms) | max (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in summary["spans"].items():
        lines.append(
            f"| `{name}` | {stats['count']} | {stats['p50']} | {stats['p95']} | {stats['max']} |"
        )
    return "\n".join(lines)


def get_recent_traces_table() -> list[list]:
    """Return rows for the recent-traces dataframe (most recent first)."""
    traces = pipeline.trace_store.list(limit=20)
    return [
        [
            t.trace_id,
            t.started_at.split("T")[1][:8] if "T" in t.started_at else t.started_at,
            t.metadata.get("character_id", "—"),
            t.metadata.get("intent", "—"),
            round(t.total_ms, 1),
            len(t.spans),
        ]
        for t in traces
    ]


def render_trace_detail(trace_id: str) -> str:
    """Render a single trace's per-span breakdown as Markdown."""
    if not trace_id or not trace_id.strip():
        return "_Enter a trace ID above (or click a row in the table) to see its breakdown._"

    trace = pipeline.trace_store.get(trace_id.strip())
    if trace is None:
        return f"⚠️ Trace `{trace_id}` not found — it may have been evicted from the buffer."

    lines = [
        f"### Trace `{trace.trace_id}`",
        "",
        f"**Started:** `{trace.started_at}`  ",
        f"**Total:** `{round(trace.total_ms, 1)}ms`  ",
        f"**Spans:** {len(trace.spans)}",
        "",
        "**Request metadata:**",
        "",
        "```json",
        _format_metadata(trace.metadata),
        "```",
        "",
        "**Spans:**",
        "",
        "| # | Stage | Start (ms) | Duration (ms) | Metadata |",
        "|---:|---|---:|---:|---|",
    ]
    for i, span in enumerate(trace.spans, 1):
        meta_str = ", ".join(f"`{k}={v}`" for k, v in span.metadata.items()) or "—"
        lines.append(
            f"| {i} | `{span.name}` | {round(span.start_ms, 1)} | "
            f"{round(span.duration_ms, 1)} | {meta_str} |"
        )
    return "\n".join(lines)


def _format_metadata(meta: dict) -> str:
    """Compact JSON-ish formatting for the metadata block."""
    if not meta:
        return "{}"
    pairs = [f'  "{k}": {_json_value(v)}' for k, v in meta.items()]
    return "{\n" + ",\n".join(pairs) + "\n}"


def _json_value(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def refresh_traces() -> tuple[str, list[list]]:
    """Refresh both the summary panel and the traces table."""
    return get_trace_summary_md(), get_recent_traces_table()


def select_trace_row(evt) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    """When a row is clicked in the traces table, populate the detail view."""
    # gradio passes a SelectData event with .index = [row, col] and .value
    try:
        # Look up trace_id from the selected row index
        row_idx = evt.index[0] if isinstance(evt.index, list) else evt.index
        rows = get_recent_traces_table()
        if 0 <= row_idx < len(rows):
            trace_id = rows[row_idx][0]
            return trace_id, render_trace_detail(trace_id)
    except Exception:  # noqa: BLE001 — selection event is best-effort UX
        pass
    return "", "_Click a row to see its details._"


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    characters = get_character_choices()
    default_char = characters[0] if characters else "blacksmith"

    with gr.Blocks(
        title="NPC Dialogue Engine — Glass Box Demo",
        theme=gr.themes.Soft(),
        css="""
        .debug-panel { font-size: 0.85em; }
        .character-info {
            background: #f0f4f8;
            padding: 12px;
            border-radius: 8px;
            color: #1a1a1a;
        }
        /* Force readable text inside the card across light + dark themes */
        .character-info p,
        .character-info li,
        .character-info strong,
        .character-info em { color: #1a1a1a !important; }
        """,
    ) as demo:
        gr.Markdown(
            "# 🎮 NPC Dialogue Engine\n"
            "### AI-powered character dialogue with full pipeline transparency\n"
            "Talk to NPCs and see the ML pipeline internals in real-time."
        )

        with gr.Row():
            # Left: Chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Dialogue",
                    height=450,
                    type="messages",
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/twitter/376/mage_1f9d9.png",
                    ),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Speak to the NPC...",
                        scale=4,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    latency_display = gr.Textbox(
                        label="Latency",
                        interactive=False,
                        scale=1,
                        max_lines=1,
                    )
                    clear_btn = gr.Button("🗑 Reset Session", scale=1)

            # Right: Glass box panel
            with gr.Column(scale=2):
                with gr.Accordion("⚙️ Configuration", open=True):
                    character_dropdown = gr.Dropdown(
                        choices=characters,
                        value=default_char,
                        label="Character",
                        interactive=True,
                    )
                    session_input = gr.Textbox(
                        value="demo_session",
                        label="Session ID",
                        max_lines=1,
                    )
                    tot_toggle = gr.Checkbox(
                        label="🌳 Tree of Thoughts (3 candidates → best)",
                        value=False,
                    )

                character_info = gr.Markdown(
                    value=get_character_info(default_char),
                    label="Character Profile",
                    elem_classes=["character-info"],
                )

                debug_panel = gr.Markdown(
                    value="*Send a message to see pipeline internals*",
                    label="Pipeline Debug",
                    elem_classes=["debug-panel"],
                )

        # ─── Event Handlers ───────────────────────────────────────

        # Send message
        send_inputs = [msg_input, chatbot, character_dropdown, session_input, tot_toggle]
        send_outputs = [chatbot, debug_panel, latency_display]

        send_btn.click(
            fn=chat,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(fn=lambda: "", outputs=msg_input)

        msg_input.submit(
            fn=chat,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(fn=lambda: "", outputs=msg_input)

        # Reset session
        clear_btn.click(
            fn=reset_session,
            inputs=[session_input],
            outputs=[chatbot, debug_panel, latency_display],
        )

        # Character selection
        character_dropdown.change(
            fn=get_character_info,
            inputs=[character_dropdown],
            outputs=[character_info],
        )

        # Example prompts
        gr.Examples(
            examples=[
                ["Hello there!", "blacksmith", False],
                ["What do you have for sale?", "blacksmith", False],
                ["Tell me about the old legends.", "mysterious_sage", False],
                ["Heard any rumors lately?", "tavern_keeper", False],
                ["I need a quest. Got anything?", "blacksmith", True],
                ["Give me what I want or else.", "tavern_keeper", False],
            ],
            inputs=[msg_input, character_dropdown, tot_toggle],
            label="Try these prompts:",
        )

        # ─── Trace Inspector ────────────────────────────────────
        # Per-stage timing for every request. Click a row to drill in.
        with gr.Accordion("🔬 Trace Inspector — pipeline timings per request", open=False):
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", scale=1)
                gr.Markdown(
                    "_Each request is a trace. Pick one to see per-stage durations and metadata._",
                    elem_classes=["debug-panel"],
                )

            trace_summary_md = gr.Markdown(
                value=get_trace_summary_md(),
                elem_classes=["debug-panel"],
            )

            traces_table = gr.Dataframe(
                value=get_recent_traces_table(),
                headers=["trace_id", "time", "character", "intent", "total_ms", "spans"],
                datatype=["str", "str", "str", "str", "number", "number"],
                interactive=False,
                wrap=True,
                label="Recent traces (most recent first)",
            )

            with gr.Row():
                trace_id_input = gr.Textbox(
                    label="Trace ID",
                    placeholder="Click a row above, or paste a trace_id",
                    scale=4,
                )
                show_btn = gr.Button("Show details", scale=1)

            trace_detail_md = gr.Markdown(
                value="_Send a message in the chat above, then refresh._",
                elem_classes=["debug-panel"],
            )

            # Refresh both summary + table
            refresh_btn.click(
                fn=refresh_traces,
                outputs=[trace_summary_md, traces_table],
            )

            # Row click → populate trace ID + render detail
            traces_table.select(
                fn=select_trace_row,
                outputs=[trace_id_input, trace_detail_md],
            )

            # Manual lookup
            show_btn.click(
                fn=render_trace_detail,
                inputs=[trace_id_input],
                outputs=[trace_detail_md],
            )

            # Auto-refresh after every chat message so the table stays current
            send_btn.click(
                fn=refresh_traces,
                outputs=[trace_summary_md, traces_table],
            )
            msg_input.submit(
                fn=refresh_traces,
                outputs=[trace_summary_md, traces_table],
            )

    return demo


# ─── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
