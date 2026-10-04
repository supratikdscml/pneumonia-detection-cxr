"""Patient-grouped development folds for the chest X-ray dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_REQUIRED_COLUMNS = {"path", "label", "split", "patient_id", "subtype"}
_DEVELOPMENT_SPLITS = {"train", "val"}


def assert_no_group_leakage(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    fold: int | None = None,
) -> None:
    """Raise when any filename-derived patient group occurs on both sides."""
    shared_groups = set(training["patient_id"]) & set(validation["patient_id"])
    if shared_groups:
        examples = sorted(shared_groups)[:10]
        fold_description = f" in fold {fold}" if fold is not None else ""
        raise AssertionError(
            f"Patient groups overlap{fold_description}: {examples}"
        )


def build_grouped_splits(
    metadata: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create long-form train/validation assignments from official train+val.

    Each source image appears once in validation and once in training for each
    fold. Official test rows are deliberately excluded.
    """
    missing_columns = _REQUIRED_COLUMNS - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    development = metadata.loc[
        metadata["split"].isin(_DEVELOPMENT_SPLITS),
        ["path", "label", "split", "patient_id", "subtype"],
    ].copy()
    if development.empty:
        raise ValueError("No official train or validation rows were found")
    if development[["path", "label", "patient_id"]].isna().any().any():
        raise ValueError("Development metadata contains a missing path, label, or patient_id")
    if development["path"].duplicated().any():
        examples = development.loc[development["path"].duplicated(), "path"].head(5).tolist()
        raise ValueError(f"Development paths must be unique; duplicates: {examples}")

    labels_per_group = development.groupby("patient_id")["label"].nunique()
    mixed_label_groups = labels_per_group[labels_per_group > 1].index.tolist()
    if mixed_label_groups:
        raise ValueError(
            "Each patient group must have one class label; mixed groups: "
            f"{sorted(mixed_label_groups)[:10]}"
        )

    groups_per_class = development.drop_duplicates("patient_id").groupby("label").size()
    insufficient_classes = groups_per_class[groups_per_class < n_splits]
    if not insufficient_classes.empty:
        raise ValueError(
            f"Each class needs at least {n_splits} patient groups; got "
            f"{insufficient_classes.to_dict()}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_assignments = []
    for fold, (training_indices, validation_indices) in enumerate(
        splitter.split(
            X=development["path"],
            y=development["label"],
            groups=development["patient_id"],
        )
    ):
        training = development.iloc[training_indices]
        validation = development.iloc[validation_indices]
        assert_no_group_leakage(training, validation, fold)

        fold_assignments.extend(
            [
                training.assign(fold=fold, partition="train"),
                validation.assign(fold=fold, partition="validation"),
            ]
        )

    splits = pd.concat(fold_assignments, ignore_index=True)
    splits = splits.rename(columns={"split": "official_split"})
    return splits[
        ["fold", "partition", "path", "label", "patient_id", "subtype", "official_split"]
    ].sort_values(["fold", "partition", "path"], ignore_index=True)


def save_grouped_splits(
    metadata: pd.DataFrame,
    output_path: str | Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build grouped folds and save them as a CSV file."""
    splits = build_grouped_splits(metadata, n_splits, random_state)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(destination, index=False)
    return splits