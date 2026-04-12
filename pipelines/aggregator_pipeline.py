import gc
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from pipelines import project_config

# =============================================================================
# STEP 1: LOAD
# =============================================================================


def load_datasets(data_dir):
    paths = {
        "app_train": "application_train.csv",
        "bureau": "bureau.csv",
        "bb": "bureau_balance.csv",
        "prev": "previous_application.csv",
        "pos": "POS_CASH_balance.csv",
        "ins": "installments_payments.csv",
        "cc": "credit_card_balance.csv",
    }
    dfs = {}
    for key, fname in paths.items():
        dfs[key] = pd.read_csv(os.path.join(data_dir, fname))
        print(
            f"   {fname:40s} -> {dfs[key].shape[0]:>10,} rows x {dfs[key].shape[1]:>3} cols"
        )
    return dfs


def load_application_only(data_dir):
    """Loads only application_train.csv for traditional pipeline."""
    paths = {
        "app_train": "application_train.csv",
    }
    dfs = {}
    for key, fname in paths.items():
        dfs[key] = pd.read_csv(os.path.join(data_dir, fname))
        print(
            f"   {fname:40s} -> {dfs[key].shape[0]:>10,} rows x {dfs[key].shape[1]:>3} cols"
        )
    return dfs


# =============================================================================
# STEP 2: AGGREGATE EACH SUPPLEMENTARY TABLE TO SK_ID_CURR LEVEL
# =============================================================================


def aggregate_bureau(bureau):
    bur_agg = (
        bureau.groupby("SK_ID_CURR")
        .agg(
            bur_n_credits=("SK_ID_BUREAU", "nunique"),
            bur_n_active=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
            bur_n_closed=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
            bur_n_bad_debt=("CREDIT_ACTIVE", lambda x: (x == "Bad debt").sum()),
            bur_days_credit_mean=("DAYS_CREDIT", "mean"),
            bur_days_credit_min=("DAYS_CREDIT", "min"),
            bur_days_credit_max=("DAYS_CREDIT", "max"),
            bur_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
            bur_overdue_sum=("CREDIT_DAY_OVERDUE", "sum"),
            bur_amt_credit_mean=("AMT_CREDIT_SUM", "mean"),
            bur_amt_credit_sum=("AMT_CREDIT_SUM", "sum"),
            bur_amt_credit_max=("AMT_CREDIT_SUM", "max"),
            bur_amt_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            bur_amt_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
            bur_amt_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            bur_amt_max_overdue=("AMT_CREDIT_MAX_OVERDUE", "max"),
            bur_annuity_sum=("AMT_ANNUITY", "sum"),
            bur_prolong_sum=("CNT_CREDIT_PROLONG", "sum"),
            bur_update_mean=("DAYS_CREDIT_UPDATE", "mean"),
            bur_enddate_mean=("DAYS_CREDIT_ENDDATE", "mean"),
        )
        .reset_index()
    )

    bur_agg["bur_active_ratio"] = bur_agg["bur_n_active"] / bur_agg["bur_n_credits"]
    bur_agg["bur_debt_credit_ratio"] = bur_agg["bur_amt_debt_sum"] / (
        bur_agg["bur_amt_credit_sum"] + 1
    )
    bur_agg["bur_avg_credit_duration"] = (
        bur_agg["bur_days_credit_max"] - bur_agg["bur_days_credit_min"]
    )

    credit_type_counts = (
        bureau.groupby(["SK_ID_CURR", "CREDIT_TYPE"]).size().unstack(fill_value=0)
    )
    credit_type_counts.columns = [
        "bur_type_" + re.sub(r"\s+", "_", str(c))[:25]
        for c in credit_type_counts.columns
    ]
    top_types = credit_type_counts.sum().nlargest(5).index
    bur_agg = bur_agg.merge(
        credit_type_counts[top_types].reset_index(), on="SK_ID_CURR", how="left"
    )

    print(
        f"   bureau:              {bureau.shape[0]:>10,} rows -> {bur_agg.shape[0]:>8,} applicants x {bur_agg.shape[1]} features"
    )
    return bur_agg


def aggregate_bureau_balance(bb, bureau):
    bb = bb.copy()
    bb["status_num"] = (
        bb["STATUS"]
        .map({"C": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "X": 0})
        .fillna(0)
    )

    bb_credit = (
        bb.groupby("SK_ID_BUREAU")
        .agg(
            bb_months_count=("MONTHS_BALANCE", "count"),
            bb_months_span=("MONTHS_BALANCE", lambda x: x.max() - x.min()),
            bb_dpd_max=("status_num", "max"),
            bb_dpd_mean=("status_num", "mean"),
            bb_dpd_sum=("status_num", "sum"),
            bb_n_dpd_months=("status_num", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    bb_credit = bb_credit.merge(
        bureau[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU", how="left"
    )

    bb_agg = (
        bb_credit.groupby("SK_ID_CURR")
        .agg(
            bb_months_total=("bb_months_count", "sum"),
            bb_dpd_worst=("bb_dpd_max", "max"),
            bb_dpd_mean_avg=("bb_dpd_mean", "mean"),
            bb_dpd_months_total=("bb_n_dpd_months", "sum"),
            bb_credits_tracked=("SK_ID_BUREAU", "nunique"),
        )
        .reset_index()
    )

    bb_agg["bb_dpd_month_ratio"] = bb_agg["bb_dpd_months_total"] / (
        bb_agg["bb_months_total"] + 1
    )

    print(
        f"   bureau_balance:      {bb.shape[0]:>10,} rows -> {bb_agg.shape[0]:>8,} applicants x {bb_agg.shape[1]} features"
    )
    del bb_credit
    gc.collect()
    return bb_agg


def aggregate_previous_application(prev):
    prev_agg = (
        prev.groupby("SK_ID_CURR")
        .agg(
            prev_n_apps=("SK_ID_PREV", "nunique"),
            prev_n_approved=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
            prev_n_refused=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
            prev_n_canceled=("NAME_CONTRACT_STATUS", lambda x: (x == "Canceled").sum()),
            prev_amt_app_mean=("AMT_APPLICATION", "mean"),
            prev_amt_app_max=("AMT_APPLICATION", "max"),
            prev_amt_credit_mean=("AMT_CREDIT", "mean"),
            prev_amt_credit_sum=("AMT_CREDIT", "sum"),
            prev_amt_down_mean=("AMT_DOWN_PAYMENT", "mean"),
            prev_amt_goods_mean=("AMT_GOODS_PRICE", "mean"),
            prev_annuity_mean=("AMT_ANNUITY", "mean"),
            prev_annuity_max=("AMT_ANNUITY", "max"),
            prev_days_decision_mean=("DAYS_DECISION", "mean"),
            prev_days_decision_min=("DAYS_DECISION", "min"),
            prev_hour_apply_mode=(
                "HOUR_APPR_PROCESS_START",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan,
            ),
            prev_n_installments_mean=("CNT_PAYMENT", "mean"),
            prev_rate_down_mean=("RATE_DOWN_PAYMENT", "mean"),
        )
        .reset_index()
    )

    prev_agg["prev_approval_rate"] = prev_agg["prev_n_approved"] / (
        prev_agg["prev_n_apps"] + 1
    )
    prev_agg["prev_refused_rate"] = prev_agg["prev_n_refused"] / (
        prev_agg["prev_n_apps"] + 1
    )
    prev_agg["prev_credit_vs_app"] = prev_agg["prev_amt_credit_mean"] / (
        prev_agg["prev_amt_app_mean"] + 1
    )

    print(
        f"   previous_application:{prev.shape[0]:>10,} rows -> {prev_agg.shape[0]:>8,} applicants x {prev_agg.shape[1]} features"
    )
    return prev_agg


def aggregate_pos_cash(pos):
    pos_agg = (
        pos.groupby("SK_ID_CURR")
        .agg(
            pos_n_contracts=("SK_ID_PREV", "nunique"),
            pos_months_total=("MONTHS_BALANCE", "count"),
            pos_months_span=("MONTHS_BALANCE", lambda x: x.max() - x.min()),
            pos_dpd_max=("SK_DPD", "max"),
            pos_dpd_mean=("SK_DPD", "mean"),
            pos_dpd_sum=("SK_DPD", "sum"),
            pos_dpd_def_max=("SK_DPD_DEF", "max"),
            pos_dpd_def_sum=("SK_DPD_DEF", "sum"),
            pos_n_completed=(
                "NAME_CONTRACT_STATUS",
                lambda x: (x == "Completed").sum(),
            ),
            pos_n_active=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum()),
            pos_installments_left_mean=("CNT_INSTALMENT_FUTURE", "mean"),
            pos_installments_left_max=("CNT_INSTALMENT_FUTURE", "max"),
        )
        .reset_index()
    )

    pos_agg["pos_dpd_ratio"] = pos_agg["pos_dpd_sum"] / (
        pos_agg["pos_months_total"] + 1
    )
    pos_agg["pos_completion_rate"] = pos_agg["pos_n_completed"] / (
        pos_agg["pos_n_completed"] + pos_agg["pos_n_active"] + 1
    )

    print(
        f"   POS_CASH_balance:    {pos.shape[0]:>10,} rows -> {pos_agg.shape[0]:>8,} applicants x {pos_agg.shape[1]} features"
    )
    return pos_agg


def aggregate_installments(ins):
    ins = ins.copy()
    ins["ins_days_diff"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["ins_payment_diff"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    ins_agg = (
        ins.groupby("SK_ID_CURR")
        .agg(
            ins_n_payments=("SK_ID_PREV", "count"),
            ins_n_contracts=("SK_ID_PREV", "nunique"),
            ins_days_early_mean=("ins_days_diff", "mean"),
            ins_days_early_min=("ins_days_diff", "min"),
            ins_days_early_max=("ins_days_diff", "max"),
            ins_n_late=("ins_days_diff", lambda x: (x < 0).sum()),
            ins_n_ontime=("ins_days_diff", lambda x: (x >= 0).sum()),
            ins_payment_diff_mean=("ins_payment_diff", "mean"),
            ins_payment_diff_max=("ins_payment_diff", "max"),
            ins_n_underpaid=("ins_payment_diff", lambda x: (x > 0).sum()),
            ins_amt_payment_mean=("AMT_PAYMENT", "mean"),
            ins_amt_payment_sum=("AMT_PAYMENT", "sum"),
            ins_amt_instalment_mean=("AMT_INSTALMENT", "mean"),
            ins_version_max=("NUM_INSTALMENT_VERSION", "max"),
        )
        .reset_index()
    )

    ins_agg["ins_late_ratio"] = ins_agg["ins_n_late"] / (ins_agg["ins_n_payments"] + 1)
    ins_agg["ins_underpay_ratio"] = ins_agg["ins_n_underpaid"] / (
        ins_agg["ins_n_payments"] + 1
    )
    ins_agg["ins_payment_ratio"] = ins_agg["ins_amt_payment_sum"] / (
        ins_agg["ins_amt_instalment_mean"] * ins_agg["ins_n_payments"] + 1
    )

    print(
        f"   installments:        {ins.shape[0]:>10,} rows -> {ins_agg.shape[0]:>8,} applicants x {ins_agg.shape[1]} features"
    )
    return ins_agg


def aggregate_credit_card(cc):
    cc_agg = (
        cc.groupby("SK_ID_CURR")
        .agg(
            cc_n_cards=("SK_ID_PREV", "nunique"),
            cc_months_total=("MONTHS_BALANCE", "count"),
            cc_balance_mean=("AMT_BALANCE", "mean"),
            cc_balance_max=("AMT_BALANCE", "max"),
            cc_credit_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            cc_drawings_atm_mean=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
            cc_drawings_total_mean=("AMT_DRAWINGS_CURRENT", "mean"),
            cc_payment_total_mean=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
            cc_min_installment_mean=("AMT_INST_MIN_REGULARITY", "mean"),
            cc_dpd_max=("SK_DPD", "max"),
            cc_dpd_sum=("SK_DPD", "sum"),
            cc_dpd_def_max=("SK_DPD_DEF", "max"),
            cc_receivable_mean=("AMT_RECEIVABLE_PRINCIPAL", "mean"),
            cc_n_drawings_mean=("CNT_DRAWINGS_CURRENT", "mean"),
            cc_n_installments_mean=("CNT_INSTALMENT_MATURE_CUM", "mean"),
        )
        .reset_index()
    )

    cc_agg["cc_utilization_rate"] = cc_agg["cc_balance_mean"] / (
        cc_agg["cc_credit_limit_mean"] + 1
    )
    cc_agg["cc_dpd_ratio"] = cc_agg["cc_dpd_sum"] / (cc_agg["cc_months_total"] + 1)
    cc_agg["cc_payment_vs_min"] = cc_agg["cc_payment_total_mean"] / (
        cc_agg["cc_min_installment_mean"] + 1
    )

    print(
        f"   credit_card_balance: {cc.shape[0]:>10,} rows -> {cc_agg.shape[0]:>8,} applicants x {cc_agg.shape[1]} features"
    )
    return cc_agg


# =============================================================================
# STEP 3: MERGE INTO MASTER
# =============================================================================


def build_master(app_train, bur_agg, bb_agg, prev_agg, pos_agg, ins_agg, cc_agg):
    master = app_train.copy()
    print(f"\n   Base application_train: {master.shape}")

    master["has_bureau"] = master["SK_ID_CURR"].isin(bur_agg["SK_ID_CURR"]).astype(int)
    master["has_bb"] = master["SK_ID_CURR"].isin(bb_agg["SK_ID_CURR"]).astype(int)
    master["has_prev"] = master["SK_ID_CURR"].isin(prev_agg["SK_ID_CURR"]).astype(int)
    master["has_pos"] = master["SK_ID_CURR"].isin(pos_agg["SK_ID_CURR"]).astype(int)
    master["has_ins"] = master["SK_ID_CURR"].isin(ins_agg["SK_ID_CURR"]).astype(int)
    master["has_cc"] = master["SK_ID_CURR"].isin(cc_agg["SK_ID_CURR"]).astype(int)

    for label, agg_df in [
        ("bureau", bur_agg),
        ("bureau_balance", bb_agg),
        ("prev_app", prev_agg),
        ("pos_cash", pos_agg),
        ("installments", ins_agg),
        ("credit_card", cc_agg),
    ]:
        master = master.merge(agg_df, on="SK_ID_CURR", how="left")
        print(f"   + {label:20s} -> {master.shape}")

    print(f"\n   Master shape: {master.shape[0]:,} rows x {master.shape[1]} cols")
    print("\n   Coverage per supplementary table:")
    for col in ["has_bureau", "has_bb", "has_prev", "has_pos", "has_ins", "has_cc"]:
        print(f"     {col:15s}: {master[col].mean()*100:5.1f}% of applicants")

    return master


def build_master_traditional(app_train):
    """Builds master from only application data (no supplementary tables)."""
    master = app_train.copy()
    print(f"\n   Base application_train (traditional): {master.shape}")
    print(f"\n   Master shape: {master.shape[0]:,} rows x {master.shape[1]} cols")
    return master


# =============================================================================
# STEPS 4-5: PRE-SPLIT TRANSFORMS  (deterministic, no fitting)
# =============================================================================


def fix_outliers(data):
    df = data.copy()
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            df["DAYS_EMPLOYED"].dropna().max(), np.nan
        )
    
    cols_to_replace_xna = [
        "CODE_GENDER",
        "ORGANIZATION_TYPE",
    ]
    for col in cols_to_replace_xna:
        if col in df.columns:
            df[col] = df[col].replace("XNA", np.nan)

    cols_to_drop_domain = [
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
        "SK_ID_CURR",
    ]
    df = df.drop(
        columns=[c for c in cols_to_drop_domain if c in df.columns], errors="ignore"
    )
    return df


def drop_high_null_columns(data, threshold=0.6):
    """Drops columns where more than `threshold` fraction of values are null."""
    df = data.copy()
    features = [c for c in df.columns if c != project_config.TARGET_COL]
    null_pct = df[features].isnull().mean()
    drop = null_pct[null_pct > threshold].index.tolist()

    if drop:
        print(f"   Dropped {len(drop)} column(s) with >{threshold*100:.0f}% nulls:")
        for c in drop:
            print(f"     - {c} ({null_pct[c]*100:.1f}% null)")
    else:
        print(f"   No columns exceed the {threshold*100:.0f}% null threshold.")
    return df.drop(columns=drop)


def feature_extraction_application_data(data):
    """Performs feature engineering and one-hot encoding."""
    df = data.copy()

    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_PERCENT"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(
        0, np.nan
    )
    df["FAMILY_CNT_INCOME_PERCENT"] = df["AMT_INCOME_TOTAL"] / df[
        "CNT_FAM_MEMBERS"
    ].replace(0, np.nan)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["BIRTH_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(
        0, np.nan
    )
    df["CHILDREN_CNT_INCOME_PERCENT"] = df["AMT_INCOME_TOTAL"] / df[
        "CNT_CHILDREN"
    ].replace(0, np.nan)
    df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["EMPLOYED_REGISTRATION_PERCENT"] = df["DAYS_EMPLOYED"] / df[
        "DAYS_REGISTRATION"
    ].replace(0, np.nan)
    df["BIRTH_REGISTRATION_PERCENT"] = df["DAYS_BIRTH"] / df[
        "DAYS_REGISTRATION"
    ].replace(0, np.nan)
    df["ID_REGISTRATION_DIFF"] = df["DAYS_ID_PUBLISH"] - df["DAYS_REGISTRATION"]
    df["ANNUITY_LENGTH_EMPLOYED_PERCENT"] = df["CREDIT_TERM"] / df[
        "DAYS_EMPLOYED"
    ].replace(0, np.nan)
    df["AGE_LOAN_FINISH"] = df["DAYS_BIRTH"] * (-1.0 / 365) + (
        df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    ) * (1.0 / 12)

    if "OWN_CAR_AGE" in df.columns:
        df["CAR_AGE_EMP_PERCENT"] = df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"].replace(
            0, np.nan
        )
        df["CAR_AGE_BIRTH_PERCENT"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"].replace(
            0, np.nan
        )

    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df["PHONE_CHANGE_EMP_PERCENT"] = df["DAYS_LAST_PHONE_CHANGE"] / df[
            "DAYS_EMPLOYED"
        ].replace(0, np.nan)
        df["PHONE_CHANGE_BIRTH_PERCENT"] = df["DAYS_LAST_PHONE_CHANGE"] / df[
            "DAYS_BIRTH"
        ].replace(0, np.nan)

    for col, new_col in {
        "NAME_CONTRACT_TYPE": "MEDIAN_INCOME_CONTRACT_TYPE",
        "NAME_TYPE_SUITE": "MEDIAN_INCOME_SUITE_TYPE",
        "NAME_HOUSING_TYPE": "MEDIAN_INCOME_HOUSING_TYPE",
        "ORGANIZATION_TYPE": "MEDIAN_INCOME_ORG_TYPE",
        "OCCUPATION_TYPE": "MEDIAN_INCOME_OCCU_TYPE",
        "NAME_EDUCATION_TYPE": "MEDIAN_INCOME_EDU_TYPE",
    }.items():
        if col in df.columns:
            df[new_col] = df[col].map(df.groupby(col)["AMT_INCOME_TOTAL"].median())

    df["ORG_TYPE_INCOME_PERCENT"] = df["MEDIAN_INCOME_ORG_TYPE"] / df[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)
    df["OCCU_TYPE_INCOME_PERCENT"] = df["MEDIAN_INCOME_OCCU_TYPE"] / df[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)
    df["EDU_TYPE_INCOME_PERCENT"] = df["MEDIAN_INCOME_EDU_TYPE"] / df[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    return df


# =============================================================================
# STEP 6: SPLIT  (must happen before any fitting)
# =============================================================================


def split_data(master):
    """Splits master into train (70%), validation (15%), test (15%)."""
    train, temp = train_test_split(master, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)
    print(f"   train={train.shape[0]:,}  val={val.shape[0]:,}  test={test.shape[0]:,}") # type: ignore
    return train, val, test


# =============================================================================
# STEP 6: EXPORT + STATISTICS
# =============================================================================


def print_dataset_statistics(splits):
    print("\n" + "=" * 65)
    print("FINAL COMBINED DATASET STATISTICS")
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


def export_splits(train, val, test, output_dir, filenames):
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


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def check_combined_files_exist(output_dir, filenames):
    folder = Path(output_dir)
    if not folder.is_dir():
        return False
    for name in filenames.values():
        file_path = folder / name
        if not file_path.exists():
            return False
    return True


def run_pipeline_traditional(
    data_dir=project_config.ORIGINAL_DIR,
    output_dir=project_config.AGGREGATED_TRADITIONAL_DIR,
    filenames=project_config.AGGREGATED_FILENAMES,
    high_null_threshold=0.7,
    check_existing=True,
):
    """Runs traditional pipeline using only application_train.csv data."""
    print("Running TRADITIONAL aggregator pipeline (application data only)...")
    print()

    if check_existing and check_combined_files_exist(output_dir, filenames):
        print("Traditional aggregated files already exist. Skipping pipeline.")
        return

    print("1/5: Loading application dataset...")
    dfs = load_application_only(data_dir)

    print("\n2/5: Building master dataset (no supplementary tables)...")
    master = build_master_traditional(dfs["app_train"])
    del dfs
    gc.collect()

    print("\n3/5: Domain rules, null fixes, dropping low-value columns...")
    master = fix_outliers(master)
    master = drop_high_null_columns(master, threshold=high_null_threshold)

    print("\n4/5: Feature engineering and one-hot encoding...")
    master = feature_extraction_application_data(master)

    print("\n5/5: Splitting -> train / val / test  (before any fitting)...")
    train, val, test = split_data(master)
    del master
    gc.collect()

    print("\n   Exporting traditional splits (unsanitized, for preprocessing pipeline)...")
    export_splits(train, val, test, output_dir, filenames)


def run_pipeline_combined(
    data_dir=project_config.ORIGINAL_DIR,
    output_dir=project_config.AGGREGATED_COMBINED_DIR,
    filenames=project_config.AGGREGATED_FILENAMES,
    high_null_threshold=0.7,
    check_existing=True,
):
    """Runs combined pipeline aggregating from all supplementary tables."""
    print("Running COMBINED aggregator pipeline (all tables aggregated)...")
    print()

    if check_existing and check_combined_files_exist(output_dir, filenames):
        print("Combined aggregated files already exist. Skipping pipeline.")
        return

    print("1/7: Loading raw datasets...")
    dfs = load_datasets(data_dir)

    print("\n2/7: Aggregating supplementary tables to applicant level...")
    bur_agg = aggregate_bureau(dfs["bureau"])
    bb_agg = aggregate_bureau_balance(dfs["bb"], dfs["bureau"])
    prev_agg = aggregate_previous_application(dfs["prev"])
    pos_agg = aggregate_pos_cash(dfs["pos"])
    ins_agg = aggregate_installments(dfs["ins"])
    cc_agg = aggregate_credit_card(dfs["cc"])
    del dfs["bureau"], dfs["bb"], dfs["prev"], dfs["pos"], dfs["ins"], dfs["cc"]
    gc.collect()

    print("\n3/7: Merging aggregations into master dataset...")
    master = build_master(
        dfs["app_train"], bur_agg, bb_agg, prev_agg, pos_agg, ins_agg, cc_agg
    )
    del bur_agg, bb_agg, prev_agg, pos_agg, ins_agg, cc_agg
    gc.collect()

    print("\n4/7: Domain rules, null fixes, dropping low-value columns...")
    master = fix_outliers(master)
    master = drop_high_null_columns(master, threshold=high_null_threshold)

    print("\n5/7: Feature engineering and one-hot encoding...")
    master = feature_extraction_application_data(master)

    print("\n6/7: Splitting -> train / val / test  (before any fitting)...")
    train, val, test = split_data(master)
    del master
    gc.collect()

    print("\n7/7: Exporting combined splits (unsanitized, for preprocessing pipeline)...")
    export_splits(train, val, test, output_dir, filenames)


def run_pipeline(
    data_dir=project_config.ORIGINAL_DIR,
    output_dir=project_config.AGGREGATED_DIR,
    filenames=project_config.AGGREGATED_FILENAMES,
    high_null_threshold=0.7,
    check_existing=True,
    pipeline_type="both",
):
    """
    Main pipeline runner. 
    pipeline_type: 'traditional', 'combined', or 'both' (default)
    """
    if pipeline_type in ["traditional", "both"]:
        run_pipeline_traditional(
            data_dir=data_dir,
            output_dir=project_config.AGGREGATED_TRADITIONAL_DIR,
            filenames=filenames,
            high_null_threshold=high_null_threshold,
            check_existing=check_existing,
        )
    
    if pipeline_type in ["combined", "both"]:
        run_pipeline_combined(
            data_dir=data_dir,
            output_dir=project_config.AGGREGATED_COMBINED_DIR,
            filenames=filenames,
            high_null_threshold=high_null_threshold,
            check_existing=check_existing,
        )


if __name__ == "__main__":
    run_pipeline()
