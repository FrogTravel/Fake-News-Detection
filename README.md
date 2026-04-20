# Fake News Ensemble

A stacked-ensemble binary classifier for fake-news detection on short news headlines. The project trains 35 diverse base learners (classical ML, deep learning, and a fine-tuned transformer) across 5 stratified folds, stores their out-of-fold and test predictions on disk, and then trains a second-level meta-learner on top of them to produce the final submission.

The design is deliberately modular: base learners are trained once and never re-run, the meta-layer is cheap to iterate on, and new models can be added without touching anything that already exists.

## Dataset

- `dataset/training_data.csv` — 34,152 labeled headlines, tab-separated, columns `labels` (0 = real, 1 = fake) and `text`.
- `dataset/testing_data.csv` — 9,984 headlines in the same format. The label column is a placeholder (`2`) until it is overwritten with final predictions.

Both files are loaded with `pd.read_csv(..., header=None, delimiter="\t", names=["labels", "text"])`.

## Architecture

The solution follows the classic two-level stacking pattern, with a strict separation between Level 0 (diverse base learners trained via K-fold) and Level 1 (a small meta-learner trained on the base learners' out-of-fold probabilities).

```
             ┌───────────────────────────────────────────────────────────────┐
             │                    training_data.csv                          │
             │                    testing_data.csv                           │
             └───────────────────────────────┬───────────────────────────────┘
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       │ Feature represent.  │                     │
                       │  • raw text         │                     │
                       │  • BOW / TF-IDF     │                     │
                       │  • char n-grams     │                     │
                       │  • MiniLM embeds    │                     │
                       │  • MPNet embeds     │                     │
                       │  • token id seqs    │                     │
                       └─────────────────────┬─────────────────────┘
                                             │
             ┌───────────────────────────────▼───────────────────────────────┐
             │                LEVEL 0  —  35 base learners                   │
             │                                                               │
             │   Linear / NB / Tree     Neural (PyTorch)     Transformer     │
             │   LogReg, LinearSVC,     MLP_small/medium,    DistilBERT_1    │
             │   SGD, Ridge, NB,        TextCNN, BiLSTM      DistilBERT_2    │
             │   RF, GBM, Stacking                           DistilBERT_3ep  │
             │                                                               │
             │   Each model is trained with 5-fold StratifiedKFold           │
             │   → produces one OOF prob vector + one test prob vector       │
             └───────────────────────────────┬───────────────────────────────┘
                                             │
                                             │   oofs/<MODEL>.csv       (test preds)
                                             │   oofs/<MODEL>_oof.csv   (OOF preds)
                                             ▼
             ┌────────────────────────────────────────────────────────────────┐
             │              LEVEL 1  —  meta-learner (ensemble.ipynb)         │
             │                                                                │
             │   X_meta_train = [oof_probs_model_1, ..., oof_probs_model_35]  │
             │   X_meta_test  = [test_probs_model_1, ..., test_probs_model_35]│
             │                                                                │
             │   StackingClassifier(                                          │
             │       estimators=[LogReg, MultinomialNB, RandomForest],        │
             │       final_estimator=LogReg                                   │
             │   )                                                            │
             └───────────────────────────────┬────────────────────────────────┘
                                             │
                                             ▼
                                  final_submission.csv
```

### Level 0: diverse base learners

Diversity is the whole point of Level 0, and it is engineered on three independent axes so that the base models make different kinds of mistakes:

1. **Feature representation.** The same raw headline is encoded six different ways: raw text (for BERT), Bag-of-Words counts, TF-IDF (unigrams, bigrams, and character n-grams), dense sentence embeddings from `all-MiniLM-L6-v2`, richer embeddings from `all-mpnet-base-v2`, and token-id sequences over a 30k-word custom vocabulary for the sequence models.
2. **Model family.** Linear models (LogReg, LinearSVC, SGD, Ridge), Naive Bayes, tree ensembles (RandomForest, GradientBoosting), a two-tier sklearn `StackingClassifier`, feed-forward MLPs, 1D convolutional text classifiers, bidirectional LSTMs, and a fine-tuned DistilBERT. Each family has structurally different inductive biases.
3. **Hyperparameters within a family.** Several base learners are the same model trained with different settings — LogReg at `C ∈ {0.01, 0.1, 10}`, small vs. medium MLPs, small vs. large CNNs, short vs. deep BiLSTMs, DistilBERT at `lr ∈ {2e-5, 1e-5}` and at different epoch counts. These variants are cheap extra signal for the meta-learner.

All 35 base models share the same training protocol, implemented once in `fit_predict_save` (and its neural counterparts `fit_predict_save_mlp`, `fit_predict_save_seq`, `fit_predict_save_bert`):

```
for each of 5 stratified folds:
    fit model on the 4 training folds
    predict probabilities on the held-out fold   → write into oof[val_idx]
    predict probabilities on the full test set   → accumulate into pred
pred /= 5
save oof  to oofs/<MODEL>_oof.csv
save pred to oofs/<MODEL>.csv
```

This gives two important guarantees. First, every training row has a prediction produced by a model that never saw it during training, so the OOF matrix used by Level 1 is leakage-free. Second, test predictions are the average of five independently trained models, which is already a mild bagging effect on top of everything else.

A few base learners need small adaptations to fit this contract: `MultinomialNB` requires non-negative features, so for dense embeddings the data is shifted by its minimum; `LinearSVC` and `RidgeClassifier` don't expose `predict_proba`, so they are wrapped in `CalibratedClassifierCV`; sequence models (LSTM) need real lengths passed alongside padded inputs for `pack_padded_sequence`; DistilBERT is retrained from the pretrained checkpoint on every fold and the MPS/CUDA cache is cleared between folds to keep memory bounded.

### Level 1: the meta-learner

The Level 1 notebook (`ensemble.ipynb`) is almost stateless. It walks `oofs/`, pairs each `<MODEL>.csv` with its `<MODEL>_oof.csv`, and stacks them column-wise:

- `X_meta_train` has shape `(34152, 35)` — one column per base model, values are the OOF probabilities on training data.
- `X_meta_test`  has shape `(9984, 35)` — one column per base model, values are the averaged 5-fold probabilities on test data.

On top of that matrix, an sklearn `StackingClassifier` (LogReg + MultinomialNB + RandomForest as sub-estimators, LogReg as final estimator) is trained with an 80/20 split. The meta-learner's job is small and well-defined: given 35 probability estimates per sample, decide how much to trust each one and how to combine them. A shallow, heavily regularizable meta-model is preferred here because the signal is already dense and the danger is overfitting the specific failure modes of the 35 base learners.

### Why this architecture

Keeping Level 0 and Level 1 on disk (`oofs/*.csv`) rather than in a single end-to-end pipeline has concrete benefits:

- Base learners are expensive (DistilBERT × 5 folds, neural nets × 5 folds, heavy TF-IDF fits) but only need to be trained once. The meta-learner is cheap and can be iterated on in seconds.
- Adding a new base learner is a drop-in change: run its cell, two new CSVs appear in `oofs/`, and `ensemble.ipynb` picks them up automatically on the next run.
- Comparing a single model against the ensemble is a one-liner (see Cell 2 of `ensemble.ipynb`), because every model's OOF predictions live in the same normalized format.
- If a base learner misbehaves, its CSV pair can be moved into `archive/` to remove it from the ensemble without deleting or re-running anything.

## Repository layout

```
FakeNewsEnsemble/
├── dataset/
│   ├── training_data.csv              # 34,152 labeled headlines
│   └── testing_data.csv               # 9,984 unlabeled headlines
├── oofs/                              # One pair of CSVs per base learner (35 models × 2 files)
│   ├── <MODEL>.csv                    # averaged 5-fold test-set probabilities
│   └── <MODEL>_oof.csv                # out-of-fold train-set probabilities
├── archive/                           # Retired base learners, same CSV-pair convention
├── news_classification_oof_to_csv.ipynb   # Level 0 — trains all base models, writes oofs/
├── ensemble.ipynb                     # Level 1 — trains meta-learner, writes final_submission.csv
└── README.md
```

## The OOF contract

Every base learner, classical or neural, conforms to the same on-disk contract. This is what makes the meta-layer trivial.

- `oofs/<MODEL>.csv` is a single-column CSV whose header is the model name and whose 9,984 rows are the averaged 5-fold probabilities for the positive class on the test set.
- `oofs/<MODEL>_oof.csv` is a single-column CSV whose header is `<MODEL>_oof` and whose 34,152 rows are the leakage-free OOF probabilities for the positive class on the training set.

Any model — sklearn, PyTorch, Hugging Face, or anything else — can join the ensemble simply by emitting these two files.

## Running the pipeline

1. Train the base learners once: run `news_classification_oof_to_csv.ipynb` top to bottom. It populates `oofs/` with 35 CSV pairs. The DistilBERT sections benefit from a GPU or Apple MPS; the classical sections run on CPU.
2. Train the meta-learner: run `ensemble.ipynb`. It auto-discovers every CSV pair in `oofs/`, stacks the probability columns, fits the meta-learner, prints the ensemble's OOF accuracy and F1, and writes `final_submission.csv` in the same TSV format as `testing_data.csv` with predicted labels in place of the placeholder.
3. Optional — Cell 2 of `ensemble.ipynb` prints every individual base learner's OOF accuracy and F1 and renders `model_comparison.png`, a bar chart with the ensemble's score drawn as a reference line so regressions are easy to spot.

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `torch`, `transformers`, `matplotlib`. A GPU (CUDA) or Apple Silicon GPU (MPS) is strongly recommended for the DistilBERT and sequence-model sections; the code auto-detects and falls back to CPU.
