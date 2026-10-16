---
title: Swedish English Translator
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# Swedish → English | Character-level Transformer

Language translator. This Space runs inference using
the model trained in `17_02_translation_transformer_shared_embs.ipynb`.

The model uses shared character embeddings, sinusoidal positional embeddings,
PyTorch's encoder–decoder Transformer, and autoregressive greedy decoding.
It **does not train the model when started**.

## Add your two trained files

The Repo includes the trained weights or vocabulary,
which are saved on your computer.(If Required) Before running or uploading the Space,
put both files at the root of this repository, alongside `app.py`:

- `pico_model.pth` — weights from `torch.save(translator.state_dict(), MODEL_NAME)`
- `pico_translator.vocab` — JSON vocabulary from `json.dump(token2idx, f)`

Both files **must be from the same training run**. Avoid accidentally
replacing a best-validation checkpoint by rerunning cell 104 at the end.

## Run locally

Python 3.11–3.14 can be used if compatible builds of PyTorch and Gradio
are available. Create a virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install gradio==6.5.1
python app.py
```

Open the local URL printed by Gradio (usually `http://127.0.0.1:7860`).

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space and select **Gradio**.
2. Upload all the files in this folder **plus** your `.pth` and `.vocab` files
   using the Space's **Files → Add file → Upload files** interface.
3. Commit the uploads and wait for the Space build to finish.
4. Open the **App** tab and test `Viens manger`.

This directory is already a Space repository: the YAML front matter at the
top of this README tells Hugging Face to launch `app.py`. A separate GitHub
repo is optional, not necessary for Spaces.

If you use the Git command line for a large `.pth` file instead of browser
upload, install Git LFS before committing/pushing (`git lfs install`).

## Troubleshooting

- **Missing `.pth`/`.vocab`:** Copy both files into the same directory as `app.py`.
- **`size mismatch`:** The vocabulary or weights came from different runs.
- **Very slow translations:** Greedy decoding performs a Transformer decoder
  forward pass for each output character. Server CPU inference can be slow.
- **Bad translations:** Verify training convergence, saved checkpoint quality,
  and that the model was actually trained French → English.
- **Different language pair:** Adjust labels/examples in `app.py` and use the
  appropriate weights and vocabulary saved for that pair.
- **Different number of heads:** This repository assumes `nhead=8` as in the
  supplied notebook; adjust `model.py` if you trained with another value.

## Files

- `app.py` — Gradio user interface
- `model.py` — notebook-compatible architecture, loading, inference
- `requirements.txt` — PyTorch runtime dependency; Gradio version is set in README metadata
- `README.md` — Space configuration and usage
- `.gitattributes` — Git LFS rule for the checkpoint when using Git
- `.gitignore` — excludes local caches but **not** the model weights

## Notebook
- Just for Training purposes you can safely ignore it while building zerogpu Hugging Face Spaces

## Authors
- The Original Author of the Notebook is Pierre Nugues 
