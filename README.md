# Local Voice Conversion

A local voice-conversion application with a custom Gradio interface around pretrained Seed-VC models.

Record or upload source speech, provide a consenting speaker's reference recording, and convert the speech toward that voice while aiming to preserve the spoken content.

## Features

- Microphone recording or source-audio upload.
- Reference-speaker audio upload and permission confirmation.
- Local model inference, completed-audio playback, and WAV download.
- Adjustable diffusion steps.
- Offline operation after dependencies and required model assets are downloaded.

## My contribution

I integrated the existing Seed-VC engine into a custom application, adding the interface, input validation, permission confirmation, conversion-generator handling, WAV export, launch script, and offline configuration and testing. I did not develop or train the underlying pretrained models.

## Tested environment

- MacBook Air running macOS
- Python 3.10 in a Conda environment named `voice-conversion`
- PyTorch using Apple MPS
- Local Gradio interface

## Setup

```bash
git clone https://github.com/preeti2898/local-voice-conversion.git
cd local-voice-conversion
conda create -n voice-conversion python=3.10 -y
conda activate voice-conversion
pip install -r requirements-mac.txt
```

Internet access is required for dependency installation and the initial model downloads. Model weights are not included in this repository.

With Internet access enabled, start the original application:

```bash
GRADIO_ANALYTICS_ENABLED=False python app_vc.py --share False
```

Allow the required model downloads to finish and test a conversion. Stop the application with Ctrl+C before starting the custom interface. For upstream setup information, see [README_SEED_VC.md](README_SEED_VC.md).

## Run

Activate the environment and run from the repository folder:

```bash
conda activate voice-conversion
bash run_local.sh
```

Alternatively, run `python local_app.py`. Open the local address printed in the terminal, normally `http://127.0.0.1:7860`.

The custom application enables Hugging Face offline mode and configures its Hub cache under `checkpoints/hf_cache`. Missing assets in the configured cache can prevent startup; complete setup before testing offline operation.

## Usage

1. Record or upload source speech.
2. Upload a consenting speaker's reference recording.
3. Confirm permission.
4. Click Convert voice.
5. Listen to the result and download the WAV.

## Offline validation

On the development MacBook Air, the custom application was restarted with Wi-Fi disconnected and successfully converted speech. Playback and WAV download were also tested.

To repeat the test, stop the app, disconnect all Internet access, restart it, and test microphone recording, conversion, playback, and download. A new machine still requires initial dependency installation and model downloads.

## Limitations

- Speaker similarity and content preservation vary; listen to verify results.
- Processing time depends on hardware, clip length, and diffusion steps.
- The interface returns completed audio, not live microphone conversion.
- Output uses temporary storage; save desired WAV files separately.
- The custom application has not been validated on other operating systems.

## Responsible use

Use only your voice or recordings from speakers who have given permission. Do not use converted audio for deception or unauthorized impersonation. The permission checkbox is a user confirmation, not independent consent verification.

Ignore rules exclude new private recordings and generated audio from normal Git additions; they do not remove any files already tracked by Git.

## Attribution and licensing

This application integrates [Seed-VC](https://github.com/Plachtaa/seed-vc) and retains its upstream source and history. Original documentation is preserved in [README_SEED_VC.md](README_SEED_VC.md).

Retain and follow the existing repository license and applicable dependency and pretrained-model terms.
