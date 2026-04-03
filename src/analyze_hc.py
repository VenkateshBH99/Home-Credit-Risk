#!/usr/bin/env python3
"""Analyze the Home Credit Default Risk dataset."""
import pandas as pd
import os

DIR = "/Users/venkateshbh/Documents/NUS/IFRM/Project/home-credit-default-risk"
os.chdir(DIR)

print("=" * 90)
print("  HOME CREDIT DEFAULT RISK — Detailed Analysis")
print("=" * 90)

# 1. Application train
train = pd.read_csv("application_train.csv", encoding="latin-1")
print(f"\n  application_train.csv  ({len(train):,} rows × {len(train.columns)} cols)")
print(f"    Unique applicants (SK_ID_CURR): {train['SK_ID_CURR'].nunique():,}")
print(f"    TARGET distribution:")
print(f"      0 (no default): {(train['TARGET']==0).sum():,} ({(train['TARGET']==0).mean()*100:.1f}%)")
print(f"      1 (default):    {(train['TARGET']==1).sum():,} ({(train['TARGET']==1).mean()*100:.1f}%)")
print(f"    Contract types:   {train['NAME_CONTRACT_TYPE'].value_counts().to_dict()}")
print(f"    Gender:           {train['CODE_GENDER'].value_counts().to_dict()}")
print(f"    Top income types: {train['NAME_INCOME_TYPE'].value_counts().head(5).to_dict()}")
print(f"    Education:        {train['NAME_EDUCATION_TYPE'].value_counts().to_dict()}")
print(f"    Family status:    {train['NAME_FAMILY_STATUS'].value_counts().to_dict()}")
print(f"    Housing:          {train['NAME_HOUSING_TYPE'].value_counts().to_dict()}")
print(f"    AMT_CREDIT:       min={train['AMT_CREDIT'].min():,.0f}  max={train['AMT_CREDIT'].max():,.0f}  mean={train['AMT_CREDIT'].mean():,.0f}")
print(f"    AMT_INCOME_TOTAL: min={train['AMT_INCOME_TOTAL'].min():,.0f}  max={train['AMT_INCOME_TOTAL'].max():,.0f}  mean={train['AMT_INCOME_TOTAL'].mean():,.0f}")
print(f"    AMT_ANNUITY:      min={train['AMT_ANNUITY'].min():,.0f}  max={train['AMT_ANNUITY'].max():,.0f}  mean={train['AMT_ANNUITY'].mean():,.0f}")
print(f"    EXT_SOURCE_1:     non-null={train['EXT_SOURCE_1'].notna().sum():,} ({train['EXT_SOURCE_1'].notna().mean()*100:.1f}%)  mean={train['EXT_SOURCE_1'].mean():.4f}")
print(f"    EXT_SOURCE_2:     non-null={train['EXT_SOURCE_2'].notna().sum():,} ({train['EXT_SOURCE_2'].notna().mean()*100:.1f}%)  mean={train['EXT_SOURCE_2'].mean():.4f}")
print(f"    EXT_SOURCE_3:     non-null={train['EXT_SOURCE_3'].notna().sum():,} ({train['EXT_SOURCE_3'].notna().mean()*100:.1f}%)  mean={train['EXT_SOURCE_3'].mean():.4f}")

# Missing values
null_pct = train.isnull().mean().sort_values(ascending=False)
high_null = null_pct[null_pct > 0.3]
print(f"\n    Columns with >30% missing ({len(high_null)}/{len(train.columns)}):")
for col, pct in high_null.head(20).items():
    print(f"      {col:45s} {pct*100:5.1f}%")

low_null = null_pct[(null_pct > 0) & (null_pct <= 0.3)]
print(f"\n    Columns with 0-30% missing ({len(low_null)}):")
for col, pct in low_null.items():
    print(f"      {col:45s} {pct*100:5.1f}%")

no_null = null_pct[null_pct == 0]
print(f"\n    Columns with 0% missing: {len(no_null)}")

# 2. Application test
test = pd.read_csv("application_test.csv", encoding="latin-1")
print(f"\n  application_test.csv  ({len(test):,} rows × {len(test.columns)} cols)")
train_ids = set(train["SK_ID_CURR"])
test_ids = set(test["SK_ID_CURR"])
print(f"    Overlap with train: {len(train_ids & test_ids):,}")

# 3. Bureau
bureau = pd.read_csv("bureau.csv")
print(f"\n  bureau.csv  ({len(bureau):,} rows × {len(bureau.columns)} cols)")
print(f"    Unique SK_ID_CURR:    {bureau['SK_ID_CURR'].nunique():,}")
print(f"    Unique SK_ID_BUREAU:  {bureau['SK_ID_BUREAU'].nunique():,}")
print(f"    Credits per customer: mean={bureau.groupby('SK_ID_CURR').size().mean():.1f}  max={bureau.groupby('SK_ID_CURR').size().max()}")
print(f"    CREDIT_ACTIVE:  {bureau['CREDIT_ACTIVE'].value_counts().to_dict()}")
print(f"    Top CREDIT_TYPE:")
for ct, cnt in bureau["CREDIT_TYPE"].value_counts().head(8).items():
    print(f"      {ct:35s} {cnt:>10,}")

# 4. Bureau balance
bb = pd.read_csv("bureau_balance.csv")
print(f"\n  bureau_balance.csv  ({len(bb):,} rows × {len(bb.columns)} cols)")
print(f"    Unique SK_ID_BUREAU: {bb['SK_ID_BUREAU'].nunique():,}")
print(f"    STATUS:  {bb['STATUS'].value_counts().to_dict()}")
print(f"    MONTHS_BALANCE range: {bb['MONTHS_BALANCE'].min()} to {bb['MONTHS_BALANCE'].max()}")

# 5. Previous application
prev = pd.read_csv("previous_application.csv", encoding="latin-1")
print(f"\n  previous_application.csv  ({len(prev):,} rows × {len(prev.columns)} cols)")
print(f"    Unique SK_ID_CURR: {prev['SK_ID_CURR'].nunique():,}")
print(f"    Unique SK_ID_PREV: {prev['SK_ID_PREV'].nunique():,}")
print(f"    Apps per customer: mean={prev.groupby('SK_ID_CURR').size().mean():.1f}  max={prev.groupby('SK_ID_CURR').size().max()}")
print(f"    Contract status:   {prev['NAME_CONTRACT_STATUS'].value_counts().to_dict()}")
print(f"    Contract type:     {prev['NAME_CONTRACT_TYPE'].value_counts().to_dict()}")

# 6. POS CASH balance
pos = pd.read_csv("POS_CASH_balance.csv")
print(f"\n  POS_CASH_balance.csv  ({len(pos):,} rows × {len(pos.columns)} cols)")
print(f"    Unique SK_ID_CURR: {pos['SK_ID_CURR'].nunique():,}")
print(f"    Contract status:   {pos['NAME_CONTRACT_STATUS'].value_counts().head(5).to_dict()}")

# 7. Credit card balance
cc = pd.read_csv("credit_card_balance.csv")
print(f"\n  credit_card_balance.csv  ({len(cc):,} rows × {len(cc.columns)} cols)")
print(f"    Unique SK_ID_CURR: {cc['SK_ID_CURR'].nunique():,}")
print(f"    Contract status:   {cc['NAME_CONTRACT_STATUS'].value_counts().to_dict()}")

# 8. Installments payments
inst = pd.read_csv("installments_payments.csv")
print(f"\n  installments_payments.csv  ({len(inst):,} rows × {len(inst.columns)} cols)")
print(f"    Unique SK_ID_CURR: {inst['SK_ID_CURR'].nunique():,}")

# Summary
print(f"\n{'=' * 90}")
print(f"  DATASET SUMMARY")
print(f"{'=' * 90}")
total_rows = len(train) + len(test) + len(bureau) + len(bb) + len(prev) + len(pos) + len(cc) + len(inst)
total_size = sum(os.path.getsize(f) for f in os.listdir(".") if f.endswith(".csv"))
print(f"    Total rows:        {total_rows:,}")
print(f"    Total size:        {total_size/1024/1024/1024:.2f} GB")
print(f"    Train applicants:  {len(train_ids):,}")
print(f"    Test applicants:   {len(test_ids):,}")
print(f"    Default rate:      {train['TARGET'].mean()*100:.2f}% (imbalanced)")

print(f"\n  TABLE RELATIONSHIP MAP:")
print(f"    application_train/test  ──(SK_ID_CURR)──►  bureau  ──(SK_ID_BUREAU)──►  bureau_balance")
print(f"    application_train/test  ──(SK_ID_CURR)──►  previous_application  ──(SK_ID_PREV)──►  POS_CASH_balance")
print(f"                                                                      ──(SK_ID_PREV)──►  credit_card_balance")
print(f"                                                                      ──(SK_ID_PREV)──►  installments_payments")
print(f"{'=' * 90}")
