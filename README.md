# DeBERTa-v3 Person-Name Tagger: Convert and Train

`2026_oct2_convertandtrain.ipynb` fine-tunes Microsoft's `deberta-v3-base` model to recognise **personal names** (`PER` entities) in text. It takes annotations exported from Label Studio, converts them into token-level training data, trains a token-classification model, and saves the result along with a per-epoch metrics report.

The model is intended to extract names from Reginald R. Sharpe's *Calendar of letter-books preserved among the achives of the Corporation of the City of London at the Guildhall* (1899-1912), where the names are irregular in spelling and form.

## Purpose

The notebook has two jobs, reflected in its name:

1. **Convert** — turn Label Studio JSON exports (paragraphs with character-offset name annotations) into the BIO-tagged token sequences a transformer needs (`O`, `B-PER`, `I-PER`).
2. **Train** — fine-tune DeBERTa-v3 on that data with the Hugging Face `Trainer`, select the best checkpoint by validation F1, and save it for downstream use.

## Organization

The notebook runs top to bottom in the following stages.

### 1. Setup and configuration
Sets the working directory, imports libraries, and reports the PyTorch/Transformers versions and whether an Intel GPU (XPU) is available. A single configuration cell holds everything you would normally change between runs: the base model, a run ID (used to name the output folder), the list of training files, the target label set, window length and stride, the random seed, and training hyperparameters (epochs, learning rate, batch size, gradient accumulation, early-stopping patience).

### 2. Label scheme and tokenizer
Defines the label order explicitly (`O` first, then `B-PER`, `I-PER`) and loads one fast tokenizer that is used for all tokenization and saved alongside the model. The notebook asserts that the tokenizer is a "fast" one, because label alignment depends on character offsets that only fast tokenizers provide.

### 3. Loading and cleaning annotations
- `loaddata()` reads and concatenates the Label Studio export files.
- `extract_annotations()` pulls the paragraph text and `PER` spans out of each task, keeping them as character offsets. It trims stray whitespace and trailing punctuation that annotators often include, removes duplicate spans, and skips tasks with no completed annotation.

A spot-check cell prints a sample document with its extracted spans.

### 4. Train/validation split
Builds a Hugging Face `Dataset` from the cleaned documents and splits it 90/10 into training and validation sets. The split is done on whole documents *before* any windowing, so overlapping text from one document can never appear on both sides.

### 5. Tokenization and label alignment
`tokenize_and_align_labels()` tokenizes each document into overlapping windows (so long documents are not truncated) and projects the character-level name spans onto tokens as BIO tags. Key behaviours:
- Special tokens get the label `-100` so they are ignored by the loss.
- It compensates for DeBERTa's SentencePiece tokens that absorb a leading space.
- A name cut off by a window boundary is masked rather than partially tagged; the overlap guarantees it appears whole in a neighbouring window.

### 6. Data quality checks
Two diagnostic cells verify the conversion before any training happens:
- **Coverage check** — confirms every annotated name falls entirely within at least one window, and lists any that do not.
- **Alignment check** — prints the actual tokens and labels for a few training examples, showing exactly what the model will learn from.

### 7. Model, metrics and training
- `compute_metrics()` scores predictions with `seqeval` (precision, recall, F1, accuracy), ignoring masked positions.
- The pretrained model is loaded with a token-classification head sized to the three labels.
- `TrainingArguments` configures epoch-level evaluation and checkpointing, keeps the best model by F1, and enables early stopping. Several settings are chosen specifically for Intel GPUs: non-fused AdamW (these GPUs lack fp64 support for the fused kernel), full float32 precision (reduced precision has caused overflow with DeBERTa on XPU), and gradient checkpointing to save memory.
- The `Trainer` runs training with the early-stopping callback.

### 8. Saving and reporting
Saves the best model and the tokenizer to the run's output folder, then writes `training_metrics_report.txt` with training loss and validation metrics for each epoch plus the best F1.

### 9. End-to-end test
Reloads the saved model from disk through a Transformers `pipeline`, exactly as downstream code would, and runs it on a couple of sample sentences to confirm it tags names correctly.

## Inputs and outputs

**Inputs:** Label Studio JSON exports listed in `TRAIN_FILES`, each task containing a `ParagraphText` field and `PER` span annotations.

**Outputs** (in `./<RUN_ID>_deberta-3_base/`):
- the fine-tuned model weights and config
- the tokenizer files
- `training_metrics_report.txt`
- a `checkpoints/` folder holding the most recent per-epoch checkpoints

## Requirements

Python with `torch` (with Intel XPU support if training on an Intel GPU), `transformers`, `datasets`, `evaluate`, `seqeval`, and `numpy`. The working-directory path in the first cell is machine-specific and should be updated before running elsewhere.

## This README.md was created with assistance from Claude