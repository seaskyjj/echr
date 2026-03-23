import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset import ECHRDataset


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_hf_metrics(pred):
    labels = pred.label_ids
    logits = extract_logits(pred.predictions)
    preds = logits.argmax(axis=-1)
    return compute_metrics(labels, preds)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split_paths(data_dir: str):
    if os.path.exists(os.path.join(data_dir, "train.csv")):
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "val.csv")
        test_path = os.path.join(data_dir, "test.csv")
    else:
        train_path = os.path.join(data_dir, "processed", "train.csv")
        val_path = os.path.join(data_dir, "processed", "val.csv")
        test_path = os.path.join(data_dir, "processed", "test.csv")
    return train_path, val_path, test_path


def resolve_data_dir(args) -> str:
    if args.split_dir:
        return args.split_dir
    if args.data_dir:
        return args.data_dir
    return str(Path(args.dataset_dir) / "data" / "processed")


def resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.working_dir) / "results" / "legal_bert"


def extract_logits(predictions):
    if isinstance(predictions, tuple):
        return predictions[0]
    return predictions


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_class_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts().sort_index()
    num_classes = len(counts)
    total = int(counts.sum())
    weights = [total / (num_classes * counts[idx]) for idx in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def compute_sample_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts().sort_index()
    per_class = {cls: 1.0 / count for cls, count in counts.items()}
    return labels.map(per_class).astype(float).to_numpy()


def threshold_objective_value(metrics: dict, metric_name: str) -> float:
    if metric_name not in {"f1", "macro_f1", "balanced_accuracy"}:
        raise ValueError(f"Unsupported threshold metric: {metric_name}")
    return float(metrics[metric_name])


def tune_binary_threshold(y_true, positive_probs, grid_size: int, metric_name: str):
    candidate_thresholds = np.linspace(0.01, 0.99, grid_size)
    best = None
    for threshold in candidate_thresholds:
        preds = (positive_probs >= threshold).astype(int)
        metrics = compute_metrics(y_true, preds)
        score = (
            threshold_objective_value(metrics, metric_name),
            metrics["accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or score > best["score"]:
            best = {
                "threshold": float(threshold),
                "metrics": metrics,
                "score": score,
                "threshold_metric": metric_name,
            }
    return best


class ImbalanceAwareTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights=None,
        use_weighted_loss=False,
        sample_weights=None,
        use_weighted_sampler=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_weighted_loss = use_weighted_loss
        self.use_weighted_sampler = use_weighted_sampler
        self.class_weights = class_weights
        self.sample_weights = sample_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not self.use_weighted_loss:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        base_model = model.module if hasattr(model, "module") else model
        num_labels = getattr(getattr(base_model, "config", None), "num_labels", logits.shape[-1])
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if not self.use_weighted_sampler:
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.sample_weights, dtype=torch.double),
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def build_training_arguments(args, output_dir: Path, device: str):
    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": "none",
        "save_total_limit": 2,
        "fp16": (device == "cuda"),
    }

    init_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_key = "eval_strategy" if "eval_strategy" in init_params else "evaluation_strategy"
    save_key = "save_strategy"
    training_kwargs[eval_key] = "epoch"
    training_kwargs[save_key] = "epoch"

    if args.warmup_steps is not None:
        training_kwargs["warmup_steps"] = args.warmup_steps
    else:
        training_kwargs["warmup_ratio"] = args.warmup_ratio

    if args.eval_accumulation_steps is not None:
        training_kwargs["eval_accumulation_steps"] = args.eval_accumulation_steps
    if args.gradient_checkpointing:
        training_kwargs["gradient_checkpointing"] = True

    return TrainingArguments(**training_kwargs)


def build_trainer(
    model,
    tokenizer,
    training_args,
    train_dataset,
    val_dataset,
    class_weights,
    sample_weights,
    callbacks,
    args,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_hf_metrics,
        "class_weights": class_weights,
        "use_weighted_loss": args.use_weighted_loss,
        "sample_weights": sample_weights,
        "use_weighted_sampler": args.use_weighted_sampler,
    }

    trainer_init_params = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    return ImbalanceAwareTrainer(**trainer_kwargs)


def prediction_frame(df: pd.DataFrame, logits: np.ndarray, preds: np.ndarray, threshold_used=None, mode="default"):
    probs = softmax_probs(logits)
    pred_df = df.copy()
    pred_df["prediction"] = preds.astype(int)
    pred_df["logit_0"] = logits[:, 0]
    pred_df["logit_1"] = logits[:, 1]
    pred_df["prob_1"] = probs[:, 1]
    pred_df["score"] = pred_df["prob_1"]
    pred_df["correct"] = (pred_df["label"] == pred_df["prediction"]).astype(int)
    pred_df["prediction_mode"] = mode
    pred_df["threshold_used"] = np.nan if threshold_used is None else float(threshold_used)
    return pred_df


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_predictions(y_true, preds):
    return compute_metrics(y_true, preds)


def save_test_outputs(
    trainer,
    val_dataset,
    test_dataset,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    args,
):
    val_output = trainer.predict(val_dataset)
    test_output = trainer.predict(test_dataset)

    val_logits = extract_logits(val_output.predictions)
    test_logits = extract_logits(test_output.predictions)
    val_probs = softmax_probs(val_logits)[:, 1]
    test_probs = softmax_probs(test_logits)[:, 1]

    default_test_preds = test_logits.argmax(axis=-1)
    default_metrics = evaluate_predictions(test_df["label"], default_test_preds)
    default_pred_df = prediction_frame(
        df=test_df,
        logits=test_logits,
        preds=default_test_preds,
        threshold_used=0.5,
        mode="argmax",
    )

    default_pred_path = output_dir / "test_predictions_default.csv"
    default_metrics_path = output_dir / "test_metrics_default.json"
    default_pred_df.to_csv(default_pred_path, index=False)
    save_json(default_metrics_path, default_metrics)

    active_pred_df = default_pred_df
    active_metrics = default_metrics
    active_pred_path = output_dir / "test_predictions.csv"
    active_metrics_path = output_dir / "test_metrics.json"

    tuning_summary = None
    if args.use_threshold_tuning:
        tuning = tune_binary_threshold(
            y_true=val_df["label"].to_numpy(),
            positive_probs=val_probs,
            grid_size=args.threshold_grid_size,
            metric_name=args.threshold_metric,
        )
        tuned_threshold = tuning["threshold"]
        tuned_test_preds = (test_probs >= tuned_threshold).astype(int)
        tuned_metrics = evaluate_predictions(test_df["label"], tuned_test_preds)
        tuned_pred_df = prediction_frame(
            df=test_df,
            logits=test_logits,
            preds=tuned_test_preds,
            threshold_used=tuned_threshold,
            mode="threshold_tuned",
        )

        tuning_summary = {
            "best_threshold": tuned_threshold,
            "grid_size": args.threshold_grid_size,
            "threshold_metric": args.threshold_metric,
            "val_metrics": tuning["metrics"],
            "test_metrics": tuned_metrics,
        }

        tuned_pred_path = output_dir / "test_predictions_threshold_tuned.csv"
        tuned_metrics_path = output_dir / "test_metrics_threshold_tuned.json"
        tuning_path = output_dir / "threshold_tuning.json"

        tuned_pred_df.to_csv(tuned_pred_path, index=False)
        save_json(tuned_metrics_path, tuned_metrics)
        save_json(tuning_path, tuning_summary)

        active_pred_df = tuned_pred_df
        active_metrics = tuned_metrics

    active_pred_df.to_csv(active_pred_path, index=False)
    summary_metrics = dict(active_metrics)
    summary_metrics["prediction_mode"] = active_pred_df["prediction_mode"].iloc[0]
    if tuning_summary is not None:
        summary_metrics["threshold_tuning"] = tuning_summary
    save_json(active_metrics_path, summary_metrics)

    return {
        "pred_path": active_pred_path,
        "metrics_path": active_metrics_path,
        "metrics": summary_metrics,
        "default_pred_path": default_pred_path,
        "default_metrics_path": default_metrics_path,
    }


def main(args):
    set_seed(args.seed)
    device = detect_device()
    print(f"Using device: {device}")

    data_dir = resolve_data_dir(args)
    train_path, val_path, test_path = load_split_paths(data_dir)
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if device in {"cuda", "mps"}:
        model.to(device)

    train_dataset = ECHRDataset(train_path, tokenizer, args.max_len)
    val_dataset = ECHRDataset(val_path, tokenizer, args.max_len)
    test_dataset = ECHRDataset(test_path, tokenizer, args.max_len)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(
        "Strategy flags:",
        {
            "weighted_loss": args.use_weighted_loss,
            "weighted_sampler": args.use_weighted_sampler,
            "threshold_tuning": args.use_threshold_tuning,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    )

    class_weights = compute_class_weights(train_df["label"])
    sample_weights = compute_sample_weights(train_df["label"])
    print(f"Class weights: {[round(float(x), 4) for x in class_weights.tolist()]}")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_arguments(args, output_dir, device)
    if args.force_single_gpu and device == "cuda":
        training_args._n_gpu = 1
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    callbacks = []
    if args.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights,
        sample_weights=sample_weights,
        callbacks=callbacks,
        args=args,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    eval_results = trainer.evaluate(test_dataset)
    print("Test results:", eval_results)

    model_save_path = output_dir / "final_model"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))

    save_json(
        output_dir / "run_config.json",
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": args.warmup_steps,
            "grad_accum_steps": args.grad_accum_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "use_weighted_loss": args.use_weighted_loss,
            "use_weighted_sampler": args.use_weighted_sampler,
            "use_threshold_tuning": args.use_threshold_tuning,
            "threshold_grid_size": args.threshold_grid_size,
            "threshold_metric": args.threshold_metric,
            "gradient_checkpointing": args.gradient_checkpointing,
            "eval_accumulation_steps": args.eval_accumulation_steps,
            "force_single_gpu": args.force_single_gpu,
        },
    )

    output_info = save_test_outputs(
        trainer=trainer,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        val_df=val_df,
        test_df=test_df,
        output_dir=output_dir,
        args=args,
    )

    print(f"Model saved to {model_save_path}")
    print(f"Saved active test predictions to {output_info['pred_path']}")
    print(f"Saved active test metrics to {output_info['metrics_path']}")
    print(f"Saved default test predictions to {output_info['default_pred_path']}")
    print(f"Saved default test metrics to {output_info['default_metrics_path']}")
    print(f"Final active test metrics summary: {output_info['metrics']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=".",
        help="Dataset root containing data/raw and data/processed.",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default=".",
        help="Writable root for outputs. Useful on Kaggle, e.g. /kaggle/working.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Backward-compatible data path. Can point to a root with processed/ or directly to the split directory.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Override directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nlpaueb/legal-bert-base-uncased",
        help="Hugging Face model name or a local model directory.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Optional warmup steps. If omitted, warmup_ratio is used.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Disable with a negative value.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum metric improvement for early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_weighted_loss",
        action="store_true",
        help="Use inverse-frequency class weights in the loss.",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Use WeightedRandomSampler for the training loader.",
    )
    parser.add_argument(
        "--use_threshold_tuning",
        action="store_true",
        help="Tune a probability threshold on the validation set before scoring test.",
    )
    parser.add_argument(
        "--threshold_grid_size",
        type=int,
        default=181,
        help="Number of thresholds to search between 0.01 and 0.99.",
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="balanced_accuracy",
        choices=["f1", "macro_f1", "balanced_accuracy"],
        help="Validation metric used to select the threshold when threshold tuning is enabled.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for long-context models to reduce memory use.",
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=None,
        help="Accumulate eval predictions on CPU to reduce memory pressure during validation and test.",
    )
    parser.add_argument(
        "--force_single_gpu",
        action="store_true",
        help="Force Trainer to stay on one GPU and avoid DataParallel wrapping in notebook environments.",
    )

    args = parser.parse_args()
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        args.early_stopping_patience = None
    main(args)
