import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
os.environ["HF_HUB_CACHE"] = str(APP_DIR / "checkpoints" / "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.chdir(APP_DIR)

print("[1/3] Loading libraries...", flush=True)

import argparse
import tempfile
import traceback
import gradio as gr
import soundfile as sf
import torch
import app_vc as engine


def initialize_models():
    if torch.cuda.is_available():
        engine.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        engine.device = torch.device("mps")
    else:
        engine.device = torch.device("cpu")

    args = argparse.Namespace(checkpoint=None, config=None, fp16=True)
    print("[2/3] Loading models...", flush=True)
    print("Cache:", os.environ["HF_HUB_CACHE"], flush=True)
    (
        engine.model,
        engine.semantic_fn,
        engine.vocoder_fn,
        engine.campplus_model,
        engine.to_mel,
        engine.mel_fn_args,
    ) = engine.load_models(args)
    engine.max_context_window = engine.sr // engine.hop_length * 30
    engine.overlap_wave_len = engine.overlap_frame_len * engine.hop_length


def convert_audio(source, reference, consent, steps):
    if not source:
        raise gr.Error("Record or upload source audio first.")
    if not reference:
        raise gr.Error("Upload reference audio first.")
    if not consent:
        raise gr.Error("Confirm the reference speaker has given permission.")

    try:
        completed_audio = None
        for _, full_audio in engine.voice_conversion(
            source, reference, int(steps), 1.0, 0.7
        ):
            if full_audio is not None:
                completed_audio = full_audio
        if completed_audio is None:
            raise RuntimeError("No completed audio was returned.")
        sample_rate, waveform = completed_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = handle.name
        sf.write(output_path, waveform, sample_rate, subtype="PCM_16")
        return output_path, output_path
    except Exception:
        traceback.print_exc()
        raise gr.Error("Conversion failed. See the terminal for details.")


def build_interface():
    with gr.Blocks(title="Local Voice Conversion") as demo:
        gr.Markdown("# Local Voice Conversion")
        gr.Markdown("Source audio supplies the words; reference audio supplies voice characteristics.")
        with gr.Row():
            source = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="1. Record or upload source speech",
            )
            reference = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="2. Upload reference audio",
            )
        consent = gr.Checkbox(
            value=False,
            label="I am the reference speaker or have their permission.",
        )
        with gr.Accordion("Advanced settings", open=False):
            steps = gr.Slider(1, 100, value=10, step=1, label="Diffusion steps")
        button = gr.Button("Convert voice", variant="primary")
        output = gr.Audio(label="Converted speech", interactive=False)
        download = gr.File(label="Download WAV", interactive=False)
        gr.Markdown("Powered by pretrained Seed-VC models. Custom interface and integration.")
        button.click(
            fn=convert_audio,
            inputs=[source, reference, consent, steps],
            outputs=[output, download],
            concurrency_limit=1,
        )
    return demo


def main():
    initialize_models()
    print("[3/3] Starting interface...", flush=True)
    demo = build_interface()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
