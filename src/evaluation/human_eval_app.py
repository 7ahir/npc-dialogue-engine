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

    # Try to initialize retriever; graceful fallback if ChromaDB not available
    try:
        retriever = LoreRetriever()
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
            f"  • Turns: {len(session.messages) // 2}",
        ]
    )

    debug_text = "\n".join(debug_lines)
    latency_text = f"⏱ {total_ms:.0f}ms"

    return history, debug_text, latency_text


def reset_session(session_id: str) -> tuple[list, str, str]:
    """Clear chat history and session."""
    context_manager.reset_session(session_id)
    return [], "", ""


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    characters = get_character_choices()
    default_char = characters[0] if characters else "blacksmith"

    with gr.Blocks(
        title="NPC Dialogue Engine — Glass Box Demo",
        theme=gr.themes.Soft(),
        css="""
        .debug-panel { font-size: 0.85em; }
        .character-info { background: #f0f4f8; padding: 12px; border-radius: 8px; }
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

    return demo


# ─── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
