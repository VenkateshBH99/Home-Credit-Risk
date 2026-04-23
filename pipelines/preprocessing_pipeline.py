import gc
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from . import project_config


# =============================================================================
# PCA & VARIMAX ROTATION
# =============================================================================


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """Varimax rotation for PCA components."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                np.asarray(Lambda) ** 3
                - (gamma / p)
                * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda)))),
            )
        )
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)


# =============================================================================
# POST-SPLIT TRANSFORMS (moved from combiner_pipeline.py)
# =============================================================================


def _is_binary_col(series):
    """Returns True only if a non-empty series contains exclusively 0s and 1s."""
    vals = set(series.dropna().unique())
    return len(vals) > 0 and vals.issubset({0, 1})


def fit_imputer(train_df, strategy="median"):
    """Fits imputer on train features only. Returns (imputer, impute_cols)."""
    features = [c for c in train_df.columns if c != project_config.TARGET_COL]
    num_cols = train_df[features].select_dtypes(include=[np.number]).columns.tolist()
    binary_cols = {c for c in num_cols if _is_binary_col(train_df[c])}
    impute_cols = [
        c for c in num_cols if c not in binary_cols and train_df[c].isnull().any()
    ]

    imputer = SimpleImputer(strategy=strategy)

    if impute_cols:
        imputer.fit(train_df[impute_cols])
        print(
            f"   Imputer ({strategy}) fitted on train: {len(impute_cols)} column(s) targeted."
        )
    else:
        imputer = None
        print("   No numerical columns required imputation.")

    return imputer, impute_cols


def fit_categorical_imputer(train_df):
    """Fits imputer on train categorical features only. Returns (imputer, impute_cols)."""
    features = [c for c in train_df.columns if c != project_config.TARGET_COL]
    num_cols = train_df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]
    impute_cols = [c for c in cat_cols if train_df[c].isnull().any()]

    imputer = SimpleImputer(strategy="most_frequent")

    if impute_cols:
        imputer.fit(train_df[impute_cols])
        print(
            f"   Categorical imputer (most_frequent) fitted on train: {len(impute_cols)} column(s) targeted."
        )
    else:
        imputer = None
        print("   No categorical columns required imputation.")

    return imputer, impute_cols


def apply_categorical_imputer(df, imputer, impute_cols, split_name):
    """Applies a pre-fitted categorical imputer to one split."""
    out = df.copy()

    if imputer is not None and impute_cols:
        out[impute_cols] = imputer.transform(out[impute_cols])
        print(f"   [{split_name}] Categorical imputation applied to {len(impute_cols)} column(s).")
    else:
        print(f"   [{split_name}] No categorical imputation needed.")
    
    return out


def apply_imputer(df, imputer, impute_cols, split_name):
    """Applies a pre-fitted imputer to one split, then catches any residual NaNs with fillna(0)."""
    out = df.copy()

    if imputer is not None and impute_cols:
        out[impute_cols] = imputer.transform(out[impute_cols])

    features = [c for c in out.columns if c != project_config.TARGET_COL]
    remaining = out[features].isnull().sum().sum()
    if remaining > 0:
        out[features] = out[features].fillna(0)
        print(
            f"   [{split_name}] Safety fillna(0): {remaining:,} residual NaN(s) cleared."
        )
    else:
        print(f"   [{split_name}] No residual NaNs.")
    return out


def select_low_variance_cols(train_df, threshold=0.01):
    """Identifies near-zero variance columns from train only. Returns list of cols to drop."""
    features = [c for c in train_df.columns if c != project_config.TARGET_COL]
    num_cols = train_df[features].select_dtypes(include=[np.number]).columns
    variances = train_df[num_cols].var()
    drop_cols = variances[variances < threshold].index.tolist()

    if drop_cols:
        print(
            f"   Dropping {len(drop_cols)} low-variance column(s) (threshold={threshold}):"
        )
        for c in drop_cols:
            print(f"     - {c} (var={variances[c]:.6f})")
    else:
        print(f"   No low-variance columns found (threshold={threshold}).")
    return drop_cols


def select_correlated_cols(train_df, threshold=0.8):
    """Identifies redundant columns from each highly correlated X-X pair using train only. Returns list of cols to drop."""
    features = [c for c in train_df.columns if c != project_config.TARGET_COL]
    num_cols = train_df[features].select_dtypes(include=[np.number]).columns
    corr_matrix = train_df[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    pairs, drop_cols = [], set()
    for col in upper_tri.columns:
        for corr_col in upper_tri.index[upper_tri[col] >= threshold].tolist():
            pairs.append((corr_col, col, corr_matrix.loc[corr_col, col]))
            drop_cols.add(col)

    print(f"\n   Highly correlated pairs (|r| >= {threshold}):")
    if pairs:
        for f1, f2, val in sorted(pairs, key=lambda x: x[2], reverse=True):
            print(f"     {f1} <-> {f2}: {val:.4f}")
    else:
        print("     None found.")
    print(
        f"\n   Total pairs: {len(pairs)} | Dropping {len(drop_cols)} redundant column(s)."
    )
    return list(drop_cols)


def fit_label_encoders(train_df):
    """Fits LabelEncoders on train categorical features only. Returns (encoders_dict, cat_cols)."""
    features = [c for c in train_df.columns if c != project_config.TARGET_COL]
    num_cols = train_df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]
    
    encoders = {}
    
    if cat_cols:
        for col in cat_cols:
            encoder = LabelEncoder()
            # Fit on non-null values in train
            encoder.fit(train_df[col].dropna().astype(str))
            encoders[col] = encoder
        print(f"   LabelEncoders fitted on train: {len(cat_cols)} categorical column(s).")
    else:
        print("   No categorical columns to encode.")
    
    return encoders, cat_cols


def apply_label_encoders(df, encoders, cat_cols, split_name):
    """Applies pre-fitted LabelEncoders to one split, handling unseen values gracefully."""
    out = df.copy()
    
    if not encoders or not cat_cols:
        print(f"   [{split_name}] No categorical encoding needed.")
        return out
    
    for col in cat_cols:
        if col in encoders:
            encoder = encoders[col]
            # Handle NaNs: fill temporarily, encode, then restore NaNs as -1 or 0
            mask = out[col].isnull()
            col_str = out[col].astype(str)
            
            try:
                out[col] = encoder.transform(col_str)
            except ValueError as e:
                # Handle unseen values by mapping them to a default class
                # Get the most frequent class index as fallback
                out[col] = col_str.map(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ 
                    else np.where(encoder.classes_ == encoder.classes_[0])[0][0]
                )
            
            # Set NaN positions to -1 to indicate missing
            out.loc[mask, col] = -1
    
    print(f"   [{split_name}] Categorical encoding applied to {len(cat_cols)} column(s).")
    return out


def sanitize_feature_names(df):
    """Renames columns to be XGBoost/LightGBM-safe. Returns (df, rename_map)."""
    raw = df.columns.tolist()
    cleaned = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in raw]

    seen, deduped = {}, []
    for c in cleaned:
        if c in seen:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            deduped.append(c)

    rename_map = {old: new for old, new in zip(raw, deduped) if old != new}
    if rename_map:
        print(f"   Sanitized {len(rename_map)} column name(s).")
    else:
        print("   All column names already XGBoost/LightGBM-compatible.")

    out = df.copy()
    out.columns = deduped
    return out, rename_map


# =============================================================================
# LOADING & EXPORTING
# =============================================================================


def load_combined_splits(combined_dir, filenames):
    """Loads train, validation, test CSVs from combined directory."""
    dfs = {}
    for key in ["train", "validation", "test"]:
        path = os.path.join(combined_dir, filenames[key])
        dfs[key] = pd.read_csv(path)
        print(
            f"   {filenames[key]:30s} -> {dfs[key].shape[0]:>10,} rows x {dfs[key].shape[1]:>3} cols"
        )
    return dfs["train"], dfs["validation"], dfs["test"]


def load_traditional_combined_splits(filenames):
    """Loads train, validation, test CSVs from both traditional and combined directories."""
    splits = {}
    
    for pipeline_type, dir_path in [
        ("traditional", project_config.AGGREGATED_TRADITIONAL_DIR),
        ("combined", project_config.AGGREGATED_COMBINED_DIR),
    ]:
        print(f"\n   Loading {pipeline_type} splits:")
        dfs = {}
        for key in ["train", "validation", "test"]:
            path = os.path.join(dir_path, filenames[key])
            dfs[key] = pd.read_csv(path)
            print(
                f"     {filenames[key]:30s} -> {dfs[key].shape[0]:>10,} rows x {dfs[key].shape[1]:>3} cols"
            )
        splits[pipeline_type] = (dfs["train"], dfs["validation"], dfs["test"])
    
    return splits


def print_dataset_statistics(splits):
    """Prints statistics for train/val/test splits."""
    print("\n" + "=" * 65)
    print("FINAL PREPROCESSED DATASET STATISTICS")
    print("=" * 65)
    for name, df in splits:
        print(f"\n  [{name}]")
        print(f"    Rows    : {df.shape[0]:>10,}")
        print(f"    Columns : {df.shape[1]:>10,}")
        if project_config.TARGET_COL in df.columns:
            counts = df[project_config.TARGET_COL].value_counts().sort_index()
            pcts = (
                df[project_config.TARGET_COL].value_counts(normalize=True).sort_index()
                * 100
            )
            print(f"    Class Balance ({project_config.TARGET_COL}):")
            for cls in counts.index:
                print(f"      Class {int(cls)}: {counts[cls]:>8,}  ({pcts[cls]:.2f}%)")
        total_nulls = df.isnull().sum().sum()
        cols_with_nulls = df.isnull().any().sum()
        print(
            f"    Remaining NaNs : {total_nulls:,} across {cols_with_nulls} column(s)"
        )
    print("\n" + "=" * 65)


def export_preprocessed_splits(train, val, test, output_dir, filenames):
    """Exports preprocessed train/val/test to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    for df, key in [(train, "train"), (val, "validation"), (test, "test")]:
        path = os.path.join(output_dir, filenames[key])
        df.to_csv(path, index=False)
        print(f"   Saved -> {path}")
    print_dataset_statistics(
        [
            (f"Train      ({filenames['train']})", train),
            (f"Validation ({filenames['validation']})", val),
            (f"Test       ({filenames['test']})", test),
        ]
    )


def check_preprocessed_files_exist(output_dir, filenames):
    """Checks if all preprocessed files already exist."""
    folder = Path(output_dir)
    if not folder.is_dir():
        return False
    for name in filenames.values():
        file_path = folder / name
        if not file_path.exists():
            return False
    return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_pipeline_single(
    pipeline_type,
    combined_dir,
    combined_filenames=project_config.AGGREGATED_FILENAMES,
    output_dir=project_config.PREPROCESSED_DIR,
    filenames=project_config.PREPROCESSED_FILENAMES,
    impute_strategy="median",
    low_var_threshold=0.01,
    corr_threshold=0.8,
    pca_variance_explained=0.90,
    use_rotation=True,
    use_pca=True,
    check_existing=True,
):
    """
    Post-split preprocessing pipeline for a single pipeline type (traditional/combined).
    use_pca: If False, skips PCA and uses numerical features directly.
    """
    pca_tag = "pca" if use_pca else "no_pca"
    print(f"\n{'='*65}")
    print(f"Running {pipeline_type.upper()} preprocessing pipeline ({pca_tag})...")
    print(f"{'='*65}\n")

    if check_existing and check_preprocessed_files_exist(output_dir, filenames):
        print(f"Preprocessed {pipeline_type} files already exist. Skipping pipeline.")
        return

    # Step 1: Load splits
    print(f"Step 1/7: Loading {pipeline_type} splits...")
    train, val, test = load_combined_splits(combined_dir, combined_filenames)

    # Step 2: Imputation
    print(f"\nStep 2/7: Imputation (fit on train)...")
    imputer, impute_cols = fit_imputer(train, strategy=impute_strategy)
    train = apply_imputer(train, imputer, impute_cols, "train")
    val = apply_imputer(val, imputer, impute_cols, "val")
    test = apply_imputer(test, imputer, impute_cols, "test")
    
    # Step 2b: Categorical imputation
    print(f"\nStep 2b/7: Categorical imputation (fit on train)...")
    cat_imputer, cat_impute_cols = fit_categorical_imputer(train)
    train = apply_categorical_imputer(train, cat_imputer, cat_impute_cols, "train")
    val = apply_categorical_imputer(val, cat_imputer, cat_impute_cols, "val")
    test = apply_categorical_imputer(test, cat_imputer, cat_impute_cols, "test")

    # Step 3: Low-variance filtering
    print(f"\nStep 3/7: Low-variance column removal (fit on train)...")
    low_var_cols = select_low_variance_cols(train, threshold=low_var_threshold)
    train = train.drop(columns=low_var_cols)
    val = val.drop(columns=low_var_cols)
    test = test.drop(columns=low_var_cols)

    # Step 4: Correlation filtering
    print(f"\nStep 4/7: Correlation-based column removal (fit on train)...")
    corr_cols = select_correlated_cols(train, threshold=corr_threshold)
    train = train.drop(columns=corr_cols)
    val = val.drop(columns=corr_cols)
    test = test.drop(columns=corr_cols)

    # Identify categorical columns BEFORE encoding (so we don't scale them after)
    features = [c for c in train.columns if c != project_config.TARGET_COL]
    num_cols = train[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_to_encode = [c for c in features if c not in num_cols]

    # Step 5: Label encoding for categorical features
    print(f"\nStep 5/7: Label encoding categorical features (fit on train)...")
    encoders, _ = fit_label_encoders(train)
    train = apply_label_encoders(train, encoders, cat_cols_to_encode, "train")
    val = apply_label_encoders(val, encoders, cat_cols_to_encode, "val")
    test = apply_label_encoders(test, encoders, cat_cols_to_encode, "test")

    # Step 6: Feature name sanitization
    print(f"\nStep 6/7: Feature name sanitization...")
    train, rename_map = sanitize_feature_names(train)
    val = val.rename(columns=rename_map)
    test = test.rename(columns=rename_map)
    # Update cat_cols names if any were sanitized
    cat_cols = [rename_map.get(c, c) for c in cat_cols_to_encode]

    # Step 7: PCA + Rotation
    print(f"\nStep 7/7: PCA transformation (use_pca={use_pca})...")

    if use_pca:
        # Fit PCA on train numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train[num_cols])

        pca = PCA(n_components=pca_variance_explained, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)

        # Apply varimax rotation
        if use_rotation:
            X_train_rot = varimax(X_train_pca)
            print(f"   Varimax rotation applied: {X_train_pca.shape[1]} PCA components")
        else:
            X_train_rot = X_train_pca
            print(f"   No rotation: {X_train_pca.shape[1]} PCA components")

        n_components = X_train_rot.shape[1]
        pca_cols = [f"PC{i+1}" for i in range(n_components)]

        # Transform train
        train_out = pd.concat(
            [
                pd.DataFrame(X_train_rot, columns=pca_cols, index=train.index),
                train[cat_cols].reset_index(drop=True),
                train[[project_config.TARGET_COL]].reset_index(drop=True),
            ],
            axis=1,
        )

        # Transform val and test
        outputs = {"train": train_out}

        for split_name, df in [("validation", val), ("test", test)]:
            X_scaled = scaler.transform(df[num_cols])
            X_pca = pca.transform(X_scaled)
            
            if use_rotation:
                X_rot = varimax(X_pca)
            else:
                X_rot = X_pca

            out_df = pd.concat(
                [
                    pd.DataFrame(X_rot, columns=pca_cols, index=df.index),
                    df[cat_cols].reset_index(drop=True),
                    df[[project_config.TARGET_COL]].reset_index(drop=True),
                ],
                axis=1,
            )
            outputs[split_name] = out_df

        train = outputs["train"]
        val = outputs["validation"]
        test = outputs["test"]
    else:
        # No PCA: scale and keep numerical features as-is
        scaler = StandardScaler()
        
        # Fit on train
        X_train_scaled = scaler.fit_transform(train[num_cols])
        train_num = pd.DataFrame(X_train_scaled, columns=num_cols, index=train.index)
        
        # Transform val
        X_val_scaled = scaler.transform(val[num_cols])
        val_num = pd.DataFrame(X_val_scaled, columns=num_cols, index=val.index)
        
        # Transform test
        X_test_scaled = scaler.transform(test[num_cols])
        test_num = pd.DataFrame(X_test_scaled, columns=num_cols, index=test.index)
        
        # Concatenate with categorical and target columns
        train = pd.concat(
            [
                train_num,
                train[cat_cols].reset_index(drop=True),
                train[[project_config.TARGET_COL]].reset_index(drop=True),
            ],
            axis=1,
        )
        
        val = pd.concat(
            [
                val_num,
                val[cat_cols].reset_index(drop=True),
                val[[project_config.TARGET_COL]].reset_index(drop=True),
            ],
            axis=1,
        )
        
        test = pd.concat(
            [
                test_num,
                test[cat_cols].reset_index(drop=True),
                test[[project_config.TARGET_COL]].reset_index(drop=True),
            ],
            axis=1,
        )
        
        print(f"   Standard scaling applied to {len(num_cols)} numerical features (no PCA)")

    # Step 8: Export
    print(f"\nStep 8/8: Exporting {pipeline_type} preprocessed splits...")
    export_preprocessed_splits(train, val, test, output_dir, filenames)

    print(f"\n{pipeline_type.capitalize()} preprocessing pipeline completed successfully!")


def run_pipeline(
    combined_filenames=project_config.AGGREGATED_FILENAMES,
    preprocessed_filenames=project_config.PREPROCESSED_FILENAMES,
    impute_strategy="median",
    low_var_threshold=0.01,
    corr_threshold=0.8,
    pca_variance_explained=0.90,
    use_rotation=True,
    check_existing=True,
    pipeline_type="both",
    pca_variant="both",
):
    """
    Main preprocessing pipeline runner.
    pipeline_type: 'traditional', 'combined', or 'both' (default)
    pca_variant: 'pca', 'no_pca', or 'both' (default) - which variants to run
    """
    # Define output dir mapping
    dir_mapping = {
        ("traditional", True): project_config.PREPROCESSED_TRADITIONAL_PCA_DIR,
        ("traditional", False): project_config.PREPROCESSED_TRADITIONAL_NO_PCA_DIR,
        ("combined", True): project_config.PREPROCESSED_COMBINED_PCA_DIR,
        ("combined", False): project_config.PREPROCESSED_COMBINED_NO_PCA_DIR,
    }

    # Define aggregated dir mapping
    agg_dir_mapping = {
        "traditional": project_config.AGGREGATED_TRADITIONAL_DIR,
        "combined": project_config.AGGREGATED_COMBINED_DIR,
    }

    pipeline_types_to_run = (
        ["traditional", "combined"]
        if pipeline_type == "both"
        else [pipeline_type]
    )
    pca_variants_to_run = (
        [True, False] if pca_variant == "both" else [pca_variant == "pca"]
    )

    for p_type in pipeline_types_to_run:
        for use_pca in pca_variants_to_run:
            run_pipeline_single(
                pipeline_type=p_type,
                combined_dir=agg_dir_mapping[p_type],
                combined_filenames=combined_filenames,
                output_dir=dir_mapping[(p_type, use_pca)],
                filenames=preprocessed_filenames,
                impute_strategy=impute_strategy,
                low_var_threshold=low_var_threshold,
                corr_threshold=corr_threshold,
                pca_variance_explained=pca_variance_explained,
                use_rotation=use_rotation,
                use_pca=use_pca,
                check_existing=check_existing,
            )


if __name__ == "__main__":
    run_pipeline()

