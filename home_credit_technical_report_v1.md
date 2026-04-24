# Home Credit Risk Technical Report

## Codebase-Grounded Documentation of the Current Repository

### Scope and intent

This document is a comprehensive technical walkthrough of the current `Home-Credit-Risk` repository as it exists in the workspace today. It is written by inspecting the actual code, notebooks, generated data artifacts, and supporting utilities in the repo rather than by relying only on presentation slides or older narrative reports.

The goal of this report is to make the codebase dissertation-ready by answering four questions in one place:

1. What exactly is in this repository?
2. How does data move from raw files to model-ready datasets?
3. What does each notebook or script do, in practical and methodological terms?
4. Which design decisions, assumptions, and caveats matter when this work is later summarized in the dissertation?

### What this report treats as source of truth

The repository contains both reusable pipeline code and exploratory notebook code. They are related, but they are not identical.

- The cleanest reusable workflow is the modular pipeline in `pipelines/` together with `runner.ipynb`.
- The richest experimental and methodological content lives in the notebooks under `src/`, plus `modeling_undersample_fold.ipynb` and `pca_factor_analysis.ipynb`.
- The `docs/` folder, slide exports, and teammate markdown reports are documentation artifacts and evidence of work, but they are not the executable source of truth for the implemented pipeline.

This distinction matters because the repo is not a single linear program. It is a research codebase with multiple parallel analysis paths:

- A modular local data pipeline path.
- A monolithic "all-in-one" combined modeling notebook path.
- An application-only traditional modeling path.
- A preprocessed-artifact benchmarking path.
- A PCA/factor-analysis interpretability path.

## 1. Repository At A Glance

### 1.1 Top-level structure

| Path | Role in project | Practical meaning |
| --- | --- | --- |
| `runner.ipynb` | Top-level orchestration notebook | Calls dataset fetch, aggregation, and preprocessing pipelines in sequence |
| `pipelines/project_config.py` | Central configuration | Defines directory constants, filenames, and target column |
| `pipelines/fetch_dataset_pipeline.py` | Raw data acquisition | Downloads the dataset folder from Google Drive via `gdown` |
| `pipelines/aggregator_pipeline.py` | Feature engineering and split creation | Creates applicant-level traditional and combined datasets |
| `pipelines/preprocessing_pipeline.py` | Post-split preprocessing | Imputation, variance filtering, correlation filtering, encoding, scaling, PCA, export |
| `src/home_credit_eda.ipynb` | Full exploratory data analysis | Detailed EDA over application plus all supplementary tables |
| `src/home_credit_modeling_combined.ipynb` | Large end-to-end research notebook | Inline aggregation, modeling, scorecard, fairness, SHAP, LIME, monitoring, reject inference |
| `src/final_traditional.ipynb` | Application-only traditional ML notebook | Direct baseline modeling on `application_train.csv` |
| `src/final_traditional_pca.ipynb` | Application-only PCA variant notebook | Same application-only baseline with PCA on continuous features |
| `modeling_undersample_fold.ipynb` | Structured model benchmarking notebook | Uses the preprocessed CSV variants and trains ensemble-under-sampled models |
| `pca_factor_analysis.ipynb` | Statistical interpretability notebook | PCA, varimax rotation, factor analysis, KMO/Bartlett, cross-dataset comparison |
| `src/pca.ipynb` | Older PCA scratch notebook | Smaller standalone PCA experiment |
| `src/scratch_plot_ext_sources.py` | Small utility script | Saves EXT_SOURCE histograms to disk |
| `LLMutils.py` | Documentation/asset utility module | Extracts images, rewrites markdown, manages dissertation assets |
| `dataset/original/` | Raw downloaded CSVs | Kaggle/Home Credit raw data copied from Drive |
| `dataset/aggregated/` | Intermediate engineered datasets | Traditional and combined train/validation/test CSVs |
| `dataset/preprocessed/` | Final modeling datasets | Four downstream-ready variants: traditional/combined x PCA/no-PCA |
| `docs/` | Output artifacts | Notebook images, presentation assets, PDF reports, and exported visuals |

### 1.2 Technology stack

The current `requirements.txt` is short but representative of the repo's main stack:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `gdown`
- `lightgbm`
- `xgboost`
- `shap`
- `fairlearn`

The notebooks also use additional packages inline, especially `lime` and optionally `factor_analyzer`.

### 1.3 Important framing about the current state of the repo

- The repo is notebook-heavy. A large share of the methodology, plots, and model-comparison logic exists in notebooks rather than in Python packages.
- The pipeline modules are more reusable and deterministic than the notebooks.
- Several notebooks still assume Google Colab execution or older directory conventions.
- The current `README.md` is not a reliable technical specification of the Home Credit workflow because it mixes relevant Home Credit setup with clearly unrelated legacy content.

## 2. End-To-End Workflow Architecture

### 2.1 The modular local workflow

The cleanest end-to-end workflow currently implemented is:

```text
runner.ipynb
  -> fetch_dataset_pipeline.run_pipeline()
  -> aggregator_pipeline.run_pipeline(check_existing=True)
  -> preprocessing_pipeline.run_pipeline(check_existing=True)
```

That flow produces:

```text
dataset/original/
  -> raw CSVs

dataset/aggregated/traditional/
  -> train.csv / validation.csv / test.csv

dataset/aggregated/combined/
  -> train.csv / validation.csv / test.csv

dataset/preprocessed/traditional_no_pca/
dataset/preprocessed/traditional_pca/
dataset/preprocessed/combined_no_pca/
dataset/preprocessed/combined_pca/
  -> train.csv / validation.csv / test.csv in each variant
```

### 2.2 The parallel notebook workflows

Besides the modular path above, the repo contains notebook workflows that bypass or duplicate parts of the pipeline:

- `src/home_credit_modeling_combined.ipynb` rebuilds aggregation and preprocessing inline from raw data, then performs the most extensive research modeling.
- `src/final_traditional.ipynb` works directly on `application_train.csv` only.
- `src/final_traditional_pca.ipynb` also works directly on `application_train.csv`, but adds PCA.
- `modeling_undersample_fold.ipynb` is different again: it uses the CSV outputs produced by the modular pipeline and benchmarks models across all four preprocessed variants.

This means the repo does not have a single modeling implementation. Instead, it has a reusable ETL/preprocessing path and multiple modeling branches built for different analytical questions.

## 3. Data Assets And Relational Structure

### 3.1 Raw files downloaded into `dataset/original`

The fetched dataset folder contains the following major files:

| File | Rows | Role |
| --- | ---: | --- |
| `application_train.csv` | 307,511 | Main labeled applicant table with `TARGET` |
| `application_test.csv` | 48,744 | Unlabeled holdout/applicant table |
| `bureau.csv` | 1,716,428 | External bureau credit records at bureau-loan level |
| `bureau_balance.csv` | 27,299,925 | Monthly bureau credit statuses at bureau-loan-month level |
| `previous_application.csv` | 1,670,214 | Previous Home Credit applications |
| `POS_CASH_balance.csv` | 10,001,358 | Monthly POS/cash snapshots |
| `installments_payments.csv` | 13,605,401 | Installment payment history |
| `credit_card_balance.csv` | 3,840,312 | Monthly credit card snapshots |
| `HomeCredit_columns_description.csv` | small metadata file | Column descriptions |
| `sample_submission.csv` | small metadata file | Kaggle submission template |

### 3.2 Relational linkage used across the codebase

The relationship logic used repeatedly across the repo is:

```text
application_train / application_test
  -- SK_ID_CURR -->
    bureau
      -- SK_ID_BUREAU --> bureau_balance

application_train / application_test
  -- SK_ID_CURR -->
    previous_application
      -- SK_ID_PREV --> POS_CASH_balance
      -- SK_ID_PREV --> credit_card_balance
      -- SK_ID_PREV --> installments_payments
```

### 3.3 Why this structure matters

The key technical challenge of the project is that the raw data is not already at one row per borrower. The model, however, ultimately needs one row per borrower or application. Almost all meaningful engineering work in the repo is about converting multi-row historical behavior into applicant-level summary features.

That is why aggregation is the core technical bridge between raw relational data and downstream credit-risk modeling.

## 4. Configuration And Orchestration

### 4.1 `pipelines/project_config.py`

This file is the repository's configuration spine for the modular workflow. It defines:

- `DATASET_ROOT = "dataset"`
- `ORIGINAL_DIR = "dataset/original"`
- `AGGREGATED_DIR = "dataset/aggregated"`
- `AGGREGATED_TRADITIONAL_DIR`
- `AGGREGATED_COMBINED_DIR`
- `PREPROCESSED_DIR = "dataset/preprocessed"`
- The four output directories for traditional/combined x PCA/no-PCA
- `TARGET_COL = "TARGET"`
- Standard train/validation/test filenames
- `DATASET_GOOGLE_DRIVE_FOLDER_ID`

This is important because path management is centralized. The pipeline code does not hardcode paths everywhere; instead, it reads them from this configuration module.

### 4.2 `runner.ipynb`

`runner.ipynb` is intentionally thin. It contains six code cells:

1. A placeholder `print('helo')`
2. A shell `!ls`
3. Import of `aggregator_pipeline`, `fetch_dataset_pipeline`, and `preprocessing_pipeline`
4. `fetch_dataset_pipeline.run_pipeline()`
5. `aggregator_pipeline.run_pipeline(check_existing=True)`
6. `preprocessing_pipeline.run_pipeline(check_existing=True)`

Technically, `runner.ipynb` is not a polished production runner. It is a lightweight orchestration notebook that triggers the modular pipeline.

Its real importance is conceptual:

- it defines the intended execution order;
- it treats `pipelines/` as the reusable implementation layer;
- and it relies on `check_existing=True` so that already-built artifacts are not recomputed unnecessarily.

## 5. Raw Data Acquisition Pipeline

### 5.1 `pipelines/fetch_dataset_pipeline.py`

This file implements the dataset download stage.

### 5.2 Main functions

| Function | Purpose | Notes |
| --- | --- | --- |
| `ensure_gdown()` | Imports `gdown`, and if missing installs it via `pip` at runtime | This is a convenience feature, but it assumes network access and package installation permissions |
| `run_pipeline(check_existing=True)` | Creates `dataset/original`, builds the Google Drive folder URL, and downloads the folder contents | Uses `gdown.download_folder` with `skip_download=check_existing` |

### 5.3 Behavior worth documenting

- The fetch stage downloads a whole Google Drive folder rather than single files one by one.
- It is meant to be idempotent when `check_existing=True`.
- It uses Drive rather than a Kaggle API workflow.
- It downloads both modeling-relevant files and support files like `sample_submission.csv` and the column-description CSV.
- The file imports `project_config` twice, which is harmless but redundant.

## 6. Aggregation And Feature Engineering Pipeline

`pipelines/aggregator_pipeline.py` is the most important reusable feature-engineering file in the repo. It contains the logic that transforms the relational dataset into applicant-level CSVs.

### 6.1 Two pipeline modes

The module supports two distinct branches:

- `traditional`: application data only
- `combined`: application data plus aggregated supplementary tables

This distinction maps directly onto one of the project's central design questions: how much predictive value is added by alternative/behavioral/relational data beyond the main application table?

### 6.2 Loader functions

| Function | What it loads |
| --- | --- |
| `load_datasets(data_dir)` | `application_train.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv` |
| `load_application_only(data_dir)` | `application_train.csv` only |

Note that the modular aggregator does not use `application_test.csv`. That file is only used later in the combined modeling notebook for reject-inference experimentation.

### 6.3 Supplementary-table aggregation functions

#### `aggregate_bureau(bureau)`

This function aggregates external bureau records to one row per applicant using `SK_ID_CURR`.

It creates:

- counts of bureau credits;
- counts of active, closed, and bad-debt credits;
- age-of-credit statistics from `DAYS_CREDIT`;
- overdue statistics from `CREDIT_DAY_OVERDUE`;
- credit amount, debt, overdue amount, annuity, prolongation, and update-date summaries;
- `bur_active_ratio`;
- `bur_debt_credit_ratio`;
- `bur_avg_credit_duration`;
- counts for the top five bureau `CREDIT_TYPE` categories.

This is the codebase's main external-credit-exposure feature block.

#### `aggregate_bureau_balance(bb, bureau)`

This is a two-stage aggregation:

1. Convert `STATUS` to a numeric delinquency severity (`C`, `0`, `X` -> 0; `1`..`5` -> 1..5).
2. Aggregate monthly records to bureau-credit level via `SK_ID_BUREAU`.
3. Merge back to `bureau` to recover `SK_ID_CURR`.
4. Aggregate again to applicant level.

The resulting features capture:

- months tracked;
- delinquency worst case;
- mean delinquency severity;
- total delinquent months;
- number of bureau credits with tracked history;
- `bb_dpd_month_ratio`.

This is how monthly bureau history is turned into applicant-level delinquency behavior.

#### `aggregate_previous_application(prev)`

This builds internal application-history features such as:

- number of previous applications;
- counts of approved, refused, and canceled applications;
- amount summaries for application, credit, down payment, goods price, and annuity;
- timing summaries from `DAYS_DECISION`;
- modal application hour;
- mean installment count and down-payment rate;
- derived approval/refusal rates;
- `prev_credit_vs_app`.

This block captures Home Credit's own prior relationship and internal demand behavior.

#### `aggregate_pos_cash(pos)`

This produces installment-behavior features from POS/cash monthly data:

- number of contracts;
- number and span of tracked months;
- DPD and DPD_DEF summaries;
- counts of completed vs active contracts;
- remaining-installment summaries;
- `pos_dpd_ratio`;
- `pos_completion_rate`.

#### `aggregate_installments(ins)`

This is one of the most important behavioral blocks. It first derives:

- `ins_days_diff = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT`
- `ins_payment_diff = AMT_INSTALMENT - AMT_PAYMENT`

It then aggregates:

- counts of payments and contracts;
- early/late timing summaries;
- counts of late and on-time payments;
- payment shortfall summaries;
- payment totals and installment totals;
- maximum installment version;
- `ins_late_ratio`;
- `ins_underpay_ratio`;
- `ins_payment_ratio`.

This turns raw repayment events into consistent measures of lateness and underpayment.

#### `aggregate_credit_card(cc)`

This produces revolving-credit features:

- number of cards;
- number of tracked months;
- balance, limit, drawing, payment, receivable, and installment summaries;
- DPD and DPD_DEF summaries;
- `cc_utilization_rate`;
- `cc_dpd_ratio`;
- `cc_payment_vs_min`.

### 6.4 Merge logic

#### `build_master(...)`

The combined pipeline starts from `application_train` and:

- creates coverage flags:
  - `has_bureau`
  - `has_bb`
  - `has_prev`
  - `has_pos`
  - `has_ins`
  - `has_cc`
- left-joins each aggregated block on `SK_ID_CURR`

The coverage flags are a strong design choice. They preserve the information that "no history exists in this table" rather than letting that absence disappear into generic nulls.

#### `build_master_traditional(app_train)`

The traditional branch simply copies `application_train` and treats it as the master dataset.

### 6.5 Domain cleaning and pre-split feature engineering

#### `fix_outliers(data)`

This function applies domain-specific cleaning:

- replaces the maximum `DAYS_EMPLOYED` value with `NaN` to handle the well-known 365243 anomaly;
- replaces `XNA` with `NaN` in `CODE_GENDER` and `ORGANIZATION_TYPE`;
- drops:
  - `FLAG_DOCUMENT_2`
  - `FLAG_DOCUMENT_4` through `FLAG_DOCUMENT_21`
  - `SK_ID_CURR`

This is where the modular pipeline explicitly removes low-value or suspicious document flags and the applicant identifier.

#### `drop_high_null_columns(data, threshold=0.6)`

This drops columns whose null fraction exceeds the threshold. In actual pipeline execution, the threshold passed from `run_pipeline_*` is `0.7`.

#### `feature_extraction_application_data(data)`

This is the main application-level derived-feature function. It creates:

- credit/income, annuity/income, credit/annuity ratios;
- family and children income ratios;
- `CREDIT_TERM`;
- `BIRTH_EMPLOYED_PERCENT`;
- credit vs goods-price difference;
- registration/employment/ID timing ratios;
- estimated age at loan finish;
- car-age and phone-change ratios where available;
- group-median-income contextual features by:
  - contract type
  - suite type
  - housing type
  - organization type
  - occupation type
  - education type
- organization, occupation, and education income percentile-style ratios;
- final one-hot encoding of all object columns via `pd.get_dummies`.

This function does two conceptually different things at once:

- ratio/interaction engineering for applicant features;
- categorical expansion into dummy columns.

### 6.6 Splitting and export

#### `split_data(master)`

This creates:

- 70% train
- 15% validation
- 15% test

using two `train_test_split` calls with `random_state=42`.

The split is consistent and reproducible, but it is not stratified on `TARGET`.

#### `export_splits(train, val, test, output_dir, filenames)`

This writes CSVs and prints dataset statistics, including class balance and remaining null counts.

### 6.7 Main entry points

| Function | Role |
| --- | --- |
| `run_pipeline_traditional(...)` | Builds application-only aggregated splits |
| `run_pipeline_combined(...)` | Builds all-table aggregated splits |
| `run_pipeline(..., pipeline_type="both")` | Runs one or both branches |

### 6.8 What the aggregated outputs materially represent

By the time this module finishes:

- the raw relational schema has already been flattened to one row per applicant;
- a large amount of domain feature engineering has already happened;
- application categoricals have already been one-hot encoded;
- the datasets are ready for downstream statistical preprocessing.

## 7. Post-Split Preprocessing Pipeline

`pipelines/preprocessing_pipeline.py` takes the aggregated train/validation/test CSVs and creates four model-ready variants.

### 7.1 Variants created

| Data regime | No PCA | PCA |
| --- | --- | --- |
| Traditional | `dataset/preprocessed/traditional_no_pca/` | `dataset/preprocessed/traditional_pca/` |
| Combined | `dataset/preprocessed/combined_no_pca/` | `dataset/preprocessed/combined_pca/` |

### 7.2 Helper functions and their purpose

| Function | Purpose |
| --- | --- |
| `varimax(Phi, ...)` | Orthogonal rotation helper used after PCA |
| `_is_binary_col(series)` | Detects purely binary numeric columns |
| `fit_imputer(train_df, strategy="median")` | Fits numeric imputer on training split only |
| `fit_categorical_imputer(train_df)` | Fits most-frequent imputer on training categorical columns |
| `apply_imputer(...)` | Applies numeric imputer and then safety `fillna(0)` |
| `apply_categorical_imputer(...)` | Applies categorical imputer |
| `select_low_variance_cols(train_df, threshold=0.01)` | Identifies near-zero-variance numeric columns |
| `select_correlated_cols(train_df, threshold=0.8)` | Drops one feature from each high-correlation pair |
| `fit_label_encoders(train_df)` | Fits sklearn `LabelEncoder`s on categorical features |
| `apply_label_encoders(...)` | Applies encoders and handles unseen values |
| `sanitize_feature_names(df)` | Makes column names safe for XGBoost/LightGBM |
| `load_combined_splits(...)` | Loads train/val/test CSVs for one aggregated regime |
| `export_preprocessed_splits(...)` | Writes final preprocessed CSVs |

### 7.3 Actual preprocessing order

For each regime and PCA choice, `run_pipeline_single(...)` does:

1. Load aggregated train/validation/test.
2. Fit numeric imputer on train only and apply to all splits.
3. Fit categorical imputer on train only and apply to all splits.
4. Identify and drop low-variance columns from train, then mirror the drops in val/test.
5. Identify and drop highly correlated columns from train, then mirror the drops in val/test.
6. Fit label encoders for any remaining categorical columns.
7. Sanitize feature names.
8. Either:
   - standard-scale numeric features and keep them directly (`use_pca=False`), or
   - standard-scale numeric features, run PCA to a variance threshold, optionally rotate, then concatenate retained categorical columns and `TARGET` (`use_pca=True`).
9. Export the final CSVs.

### 7.4 Default parameterization

The default settings encoded in `run_pipeline(...)` are:

- `impute_strategy="median"`
- `low_var_threshold=0.01`
- `corr_threshold=0.8`
- `pca_variance_explained=0.90`
- `use_rotation=True`
- `pipeline_type="both"`
- `pca_variant="both"`

So a full run builds:

- traditional no-PCA
- traditional PCA
- combined no-PCA
- combined PCA

### 7.5 How categorical handling works here

The preprocessing module includes robust handling for remaining categorical features, including:

- mode imputation;
- label encoding;
- unseen-value fallback;
- explicit `-1` assignment for missing categories after encoding.

In practice, the aggregated datasets are already mostly numeric because `feature_extraction_application_data()` has already one-hot encoded application categoricals. The categorical machinery in this file is therefore more of a defensive general-purpose layer than the primary transformation path.

### 7.6 Important implementation detail about the PCA path

The pipeline's PCA branch:

- fits `StandardScaler` on training numeric columns;
- fits `PCA(n_components=0.90)` on training numeric features;
- applies `varimax(...)` to the PCA-transformed score matrix.

It then repeats a fresh `varimax(...)` call on validation and test PCA score matrices separately. This is a meaningful implementation detail and should be remembered when writing the dissertation: the current code rotates split-specific component scores rather than learning one rotation matrix on training loadings and reusing it unchanged across splits.

### 7.7 What gets persisted and what does not

The pipeline persists final CSVs only. It does not serialize:

- imputers
- encoders
- scalers
- PCA objects
- rotation matrices
- rename maps

Operationally, this means the repo's reproducibility is currently artifact-based rather than object-serialization-based.

## 8. Materialized Dataset Variants In The Current Workspace

The current workspace already contains the generated artifacts. Based on the files present:

| Stage | Variant | Rows per split | Columns | Notes |
| --- | --- | --- | ---: | --- |
| Aggregated | Traditional | train 215,257 / val 46,127 / test 46,127 | 252 | Includes `TARGET` |
| Aggregated | Combined | train 215,257 / val 46,127 / test 46,127 | 337 | Includes `TARGET`; extra features come from supplementary tables and `has_*` flags |
| Preprocessed | Traditional No PCA | same row counts | 187 | Scaled numeric features retained directly |
| Preprocessed | Traditional PCA | same row counts | 174 | PCA-reduced traditional dataset |
| Preprocessed | Combined No PCA | same row counts | 242 | Scaled combined features retained directly |
| Preprocessed | Combined PCA | same row counts | 208 | PCA-reduced combined dataset |

### 8.1 Feature-space evolution

This is one of the most important quantitative summaries of the repo:

- Traditional aggregated: 252 columns
- Combined aggregated: 337 columns
- Traditional no-PCA preprocessed: 187 columns
- Traditional PCA preprocessed: 174 columns
- Combined no-PCA preprocessed: 242 columns
- Combined PCA preprocessed: 208 columns

So the codebase is explicitly supporting four downstream modeling regimes:

- application-only without PCA
- application-only with PCA
- combined relational data without PCA
- combined relational data with PCA

## 9. `home_credit_eda.ipynb`: Full Exploratory Data Analysis

This notebook is the main descriptive analytics document for the raw Home Credit data. It contains 60 cells and reads like a structured EDA report rather than a reusable software module.

### 9.1 Setup and execution style

The notebook:

- mounts Google Drive in Colab;
- unzips a Home Credit archive;
- sets plotting styles and display options;
- loads datasets manually from a `DATA_DIR`.

This means it is oriented toward interactive notebook analysis, not pipeline reuse.

### 9.2 What it analyzes

The notebook covers the following major sections:

| Section | What is done |
| --- | --- |
| Dataset overview | Loads `application_train` and `application_test`, prints shapes, data types, and file sizes |
| Target analysis | Shows class imbalance and default rate |
| Missing values | Quantifies missingness and plots top missing columns |
| Categorical univariate analysis | Uses a helper `plot_categorical()` to show category counts and default rates |
| Numerical univariate analysis | KDE plots for income/credit/annuity/goods price by target |
| Age analysis | Converts `DAYS_BIRTH` to years and examines default rate by age bucket |
| Employment anomaly analysis | Identifies the 365243 `DAYS_EMPLOYED` sentinel |
| External source analysis | Plots `EXT_SOURCE_1/2/3` distributions and their missingness/correlation |
| Outlier analysis | Boxplots of financial variables by target |
| Correlation analysis | Top 20 correlations with `TARGET`, heatmap of selected features |
| Bivariate analysis | Scatter plots of key financial relationships |
| Bureau analysis | Credit activity status, credit types, credits-per-applicant, overdue analysis |
| Previous application analysis | Contract status, goods categories, approval mix, prior-application counts |
| POS cash analysis | Contract status, DPD incidence, remaining-installment trends |
| Credit card analysis | Contract status, utilization, monthly balance, DPD incidence |
| Installments analysis | Payment delay, underpayment, installment-version behavior |
| Skewness and kurtosis | Flags heavily skewed numerical variables |
| Bureau balance analysis | Monthly status distribution, delinquent vs non-delinquent share, months per credit |
| FLAG_DOCUMENT analysis | Submission rates and default-rate differences |
| Temporal feature analysis | `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE` |
| Credit bureau inquiry analysis | Inquiry count distributions and target correlations |
| Binary flag analysis | Car ownership, realty, phone/email/contact flags |
| Financial binning analysis | Default rate by deciles of income, credit, annuity, goods price |

### 9.3 What the notebook contributes to the project

This notebook is where the project establishes its empirical intuition:

- the target is highly imbalanced;
- `EXT_SOURCE_*` features are unusually strong predictors;
- the `DAYS_EMPLOYED` sentinel needs handling;
- many application features are weak in isolation;
- supplementary tables carry meaningful history and behavior signals;
- severe missingness is structurally part of the dataset and sometimes informative.

In dissertation terms, this notebook is the evidence base for the project's later feature engineering and model-choice decisions.

## 10. `home_credit_modeling_combined.ipynb`: The Large Integrated Research Notebook

This is the single richest notebook in the repository. It contains the most complete narrative of the team's combined-data credit-risk research and it goes far beyond plain model training.

### 10.1 What makes it different from the modular pipeline

Unlike `runner.ipynb` plus `pipelines/`, this notebook:

- rebuilds aggregation inline rather than importing `aggregator_pipeline.py`;
- performs its own preprocessing inline;
- trains models directly;
- adds governance, interpretability, fairness, monitoring, scorecarding, expected-loss, business-threshold, and reject-inference analyses.

In other words, it is both a modeling notebook and a methodological compendium.

### 10.2 Major implemented sections

#### 10.2.1 Inline raw-to-master aggregation

The notebook:

- loads all raw tables;
- reproduces bureau, bureau balance, previous application, POS cash, installments, and credit card aggregations;
- creates the same `has_*` coverage flags;
- merges everything into a combined master dataframe.

This mirrors the modular aggregator very closely, which confirms that the pipeline code and notebook research work were developed around the same conceptual feature blocks.

#### 10.2.2 Inline preprocessing

The notebook then:

- replaces infinite values with nulls;
- label-encodes all object columns;
- separates `TARGET` and features;
- drops columns with more than 70% nulls;
- median-imputes remaining values.

This preprocessing path is simpler and more notebook-centric than the reusable `preprocessing_pipeline.py`.

#### 10.2.3 Correlation and data-source attribution

The notebook analyzes:

- top positive and negative feature correlations with default;
- heatmaps and clustermaps;
- highly correlated feature pairs;
- correlation contribution by source group:
  - application
  - bureau
  - bureau balance
  - previous application
  - POS cash
  - installments
  - credit card
  - coverage flags

This source-attribution logic is important because the project is explicitly comparing traditional application features against supplementary relational signals.

#### 10.2.4 PCA analysis

The notebook performs extensive PCA work:

- scaling before PCA;
- full PCA fit;
- variance thresholds at 50/80/90/95%;
- scree plots and cumulative variance plots;
- loading inspection for the first components;
- 2D PCA scatter plots;
- PC correlations with `TARGET`.

This is not just dimensionality reduction for model training. It is also an interpretability exercise about latent structure in the engineered feature space.

#### 10.2.5 Logistic-regression scorecard path

The scorecard branch includes:

- a custom `calc_woe_iv()` routine;
- IV computation for all eligible numeric features;
- selection of features with `IV > 0.02`;
- `LogisticRegression` with:
  - `C=0.05`
  - `penalty='l1'`
  - `solver='saga'`
  - `class_weight='balanced'`
- performance evaluation via AUC and Brier score;
- inspection of non-zero coefficients;
- scorecard conversion from log-odds to a normalized 300-850 score scale;
- score-band default-rate analysis.

This is the notebook's most explicitly credit-risk-industry-aligned path because it uses WoE/IV logic and a scorecard framing rather than only generic ML.

#### 10.2.6 Gradient boosting and ensemble path

The notebook then trains:

- LightGBM with early stopping;
- XGBoost with early stopping and `scale_pos_weight`;
- simple average blending;
- weighted blending optimized on validation AUC.

The LightGBM and XGBoost branches both use reasonably production-flavored parameterizations and are treated as challenger/champion-style models relative to the scorecard baseline.

#### 10.2.7 A-score / B-score / C-score decomposition

One of the more conceptually ambitious parts of the notebook is the score-family decomposition:

- `A-Score`: application-time features
- `B-Score`: behavioral internal payment features
- `C-Score`: delinquency/collections features
- `Bureau Score`: external-credit-only features

These score families are trained with LightGBM on feature subsets defined by name prefixes or delinquency-related keywords.

This is important for dissertation writing because it shows the project is not just searching for the best AUC. It is thinking in terms of operational score roles across the customer lifecycle.

#### 10.2.8 Model comparison and classical diagnostics

The notebook includes:

- comparison tables for AUC, Brier, Gini, and feature count;
- ROC and precision-recall curves across models;
- KS statistic computation;
- KS plot;
- calibration curve;
- lift chart.

This is where the notebook becomes very close to a model-validation pack.

#### 10.2.9 Cross-validation, feature selection, and calibration improvements

The "improvements" section adds:

- 5-fold stratified CV for scorecard, LightGBM, and XGBoost;
- correlation-based feature removal at `|r| > 0.85`;
- LightGBM retraining on the reduced feature set;
- probability calibration using:
  - Platt scaling
  - isotonic regression

This section demonstrates that the repo is not only comparing models but also iterating on stability and probability quality.

#### 10.2.10 SHAP and LIME explainability

The notebook installs and uses:

- `shap`
- `lime`

It then produces:

- SHAP beeswarm plots for LightGBM, XGBoost, and logistic regression;
- dependence plots;
- cross-model SHAP ranking comparison;
- LIME explanations for one default and one non-default example for each major model family.

This is a key dissertation asset because it connects black-box predictive power back to human-interpretable feature behavior.

#### 10.2.11 Fairlearn fairness analysis

The fairness section is unusually extensive.

It evaluates fairness:

- by gender;
- by age group;
- across multiple models;
- using:
  - selection rate
  - true positive rate
  - false positive rate
  - demographic parity difference
  - demographic parity ratio
  - equalized odds difference

It also goes beyond measurement and applies:

- `ThresholdOptimizer` post-processing bias mitigation under demographic parity constraints.

That is not a trivial add-on. It is a genuine attempt to connect model performance to responsible-AI governance.

#### 10.2.12 Stability monitoring

The notebook computes:

- PSI: score-distribution drift from train to validation;
- CSI: feature-level drift across the same development/validation boundary.

These are operational monitoring tools, not just academic metrics, and they make the repo much more dissertation-friendly from a risk-governance perspective.

#### 10.2.13 Expected loss and threshold optimization

The notebook turns PD predictions into business metrics:

- expected loss `EL = PD x LGD x EAD`
- portfolio EL summaries
- EL by PD risk band
- EL concentration curves
- cost-sensitive threshold optimization
- profit curves
- approval-rate vs threshold curves
- default-rate-among-approved vs threshold curves

This is where the codebase explicitly connects risk modeling to lending economics.

#### 10.2.14 Reject inference

The notebook uses `application_test.csv` as a proxy population of rejected or unlabeled applicants and explores two reject-inference strategies:

- hard cutoff / parceling
- fuzzy augmentation with probabilistic weights

This is analytically interesting and dissertation-relevant, but it is also clearly an experimental approximation rather than a production-grade reject-inference implementation.

#### 10.2.15 Governance narrative

The notebook also contains markdown-only sections that read like a model-governance template:

- model identification;
- purpose and scope;
- methodology;
- performance;
- limitations and risks.

These markdown sections are not pipeline code, but they are valuable drafting material for the dissertation because they already translate technical implementation into model-risk language.

## 11. `final_traditional.ipynb`: Application-Only Traditional Modeling

This notebook is a clean baseline branch focused only on `application_train.csv`.

### 11.1 Core workflow

It:

- loads `application_train.csv`;
- removes `SK_ID_CURR`;
- separates `TARGET`;
- median-imputes numeric columns;
- mode-imputes categorical columns;
- one-hot encodes categoricals with `drop_first=True`;
- sanitizes column names for LightGBM/XGBoost;
- scales all features with `StandardScaler`;
- creates an 80/20 stratified train/test split.

### 11.2 Models trained

The notebook trains:

- logistic regression:
  - `class_weight='balanced'`
  - `max_iter=1000`
- random forest:
  - `n_estimators=100`
  - `max_depth=5`
  - balanced class weights
- LightGBM:
  - `n_estimators=100`
  - `max_depth=8`
  - `num_leaves=20`
  - `learning_rate=0.1`
- XGBoost:
  - `n_estimators=90`
  - `max_depth=5`
  - `learning_rate=0.1`
  - `scale_pos_weight` based on class ratio

### 11.3 Diagnostics and outputs

For each model, the notebook computes:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- confusion matrix
- classification report
- train vs test comparison
- feature importance or coefficients
- ROC curve
- SHAP explanations

### 11.4 Fairness layer

At the end, it performs fairness analysis by `CODE_GENDER` using `MetricFrame`, demographic parity, and equalized odds metrics across all four models.

### 11.5 What this notebook represents conceptually

This notebook is the project's pure application-only benchmark. It deliberately excludes the supplementary relational tables so that the uplift from richer data can be judged later against a simpler baseline.

## 12. `final_traditional_pca.ipynb`: Application-Only PCA Variant

This notebook shares the same application-only philosophy as `final_traditional.ipynb`, but it adds explicit PCA on continuous features.

### 12.1 Preprocessing differences from the non-PCA notebook

It:

- separates numeric and categorical features explicitly;
- median-imputes numeric values;
- mode-imputes categorical values;
- standard-scales numeric columns before PCA;
- fits PCA to 95% explained variance;
- one-hot encodes categoricals;
- computes correlation among encoded categorical columns;
- removes highly correlated dummy columns using a threshold of `0.8`;
- concatenates PCA components with filtered categorical columns;
- rescales the combined matrix.

### 12.2 Modeling and evaluation

It then repeats the same four-model suite:

- logistic regression
- random forest
- LightGBM
- XGBoost

with the same style of:

- metric reporting
- confusion matrices
- ROC curves
- SHAP analyses
- fairness analysis

### 12.3 Why this notebook matters

This notebook tests whether application-only data can become more compact, less redundant, or more stable via PCA without needing the full combined-data engineering path.

It is also methodologically distinct from the modular preprocessing pipeline because:

- it uses 95% PCA variance rather than the pipeline's 90%;
- it does not use the pipeline's varimax rotation path;
- it does not consume the preprocessed CSV variants created by `pipelines/preprocessing_pipeline.py`.

## 13. `modeling_undersample_fold.ipynb`: Benchmarking On Pipeline Outputs

This notebook is especially important because it is the main place where the modular pipeline outputs are actually used as designed.

### 13.1 Inputs

It loads the four preprocessed variants via `pipelines.project_config`:

- `traditional_pca`
- `traditional_no_pca`
- `combined_pca`
- `combined_no_pca`

### 13.2 Core design idea

Instead of relying only on class weights, it uses an ensemble-under-sampling strategy:

- split the majority class into `K` folds;
- keep all minority samples in every fold;
- train one model per fold on all minority plus one fold of majority;
- average predicted probabilities across the fold-specific models.

This is implemented by:

- `make_ensemble_splits`
- `train_ensemble`
- `predict_proba_ensemble`
- `evaluate_ensemble`

### 13.3 Models benchmarked

For each dataset variant, it builds ensemble versions of:

- logistic regression
- LightGBM
- XGBoost
- random forest

### 13.4 Evaluation workflow

The notebook produces:

- baseline validation metrics;
- AUC comparison bar charts;
- model-vs-dataset AUC heatmaps;
- ROC curves;
- precision-recall curves;
- confusion matrices on the best dataset;
- feature importance plots;
- lift chart and KS plot;
- optional hyperparameter tuning via `RandomizedSearchCV`:
  - currently disabled by `DO_HYPERTUNING = False`;
- final test-set evaluation using either baseline or tuned models;
- SHAP analyses across all models on the best dataset.

### 13.5 Why this notebook is strategically important

This is the repo's cleanest multi-variant benchmark notebook because it actually consumes the modularly generated artifacts rather than rebuilding data ad hoc inside the notebook.

For dissertation writing, this notebook is the best bridge between:

- the reusable data pipeline;
- and comparative model evaluation across the four prepared dataset regimes.

## 14. `pca_factor_analysis.ipynb`: Statistical Interpretability Notebook

This notebook is not a predictive-model notebook in the narrow sense. It is a deep statistical analysis of the engineered feature spaces.

### 14.1 Data used

It loads:

- `dataset/aggregated/traditional/train.csv`
- `dataset/aggregated/combined/train.csv`

and their validation/test companions.

So unlike `modeling_undersample_fold.ipynb`, it works from the aggregated stage rather than the preprocessed stage.

### 14.2 What it studies

It contains:

- dataset overview and missing-value analysis;
- target distribution;
- feature preparation and scaling;
- variance analysis;
- target-correlation plots;
- WoE/IV computation;
- correlation heatmaps and clustermaps;
- counts of high-correlation pairs at varying thresholds;
- feature-source correlation summaries for combined data;
- full PCA decomposition for traditional and combined datasets;
- scree plots;
- Kaiser criterion analysis;
- loadings analysis by principal component;
- PC-to-target correlation analysis;
- PCA scatter plots;
- PCA threshold comparison at 80/90/95%;
- varimax rotation;
- factor analysis via maximum likelihood;
- communalities and uniqueness analysis;
- KMO and Bartlett tests when `factor_analyzer` is available;
- comparison of PCA vs factor analysis variance structure;
- categorical correlation filtering;
- a cross-dataset comparison summary table.

### 14.3 Why it exists in the repo

This notebook is best understood as a dissertation support notebook. It provides the kind of statistical evidence needed to justify dimensionality reduction, factor interpretability, and latent-structure claims rather than simply optimizing predictive performance.

## 15. `src/pca.ipynb`: Earlier PCA Scratch Work

This is a smaller standalone PCA experiment. It:

- reads a dataset from `../dataset/cleaned/train.csv`;
- drops `TARGET`;
- keeps numeric columns only;
- standard-scales features;
- fits PCA;
- inspects eigenvalues and eigenvectors;
- computes varimax rotation on the top five components;
- plots scree, cumulative variance, and a PC1-vs-PC2 loading map.

This notebook appears to be an older or more isolated PCA exploration and uses a dataset path that does not match the main current modular pipeline structure.

## 16. Small Utility Scripts And Non-Model Code

### 16.1 `src/scratch_plot_ext_sources.py`

This script is a one-purpose plotting helper. It:

- reads only `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` from `application_train.csv`;
- plots simple histograms with KDE overlays;
- saves the result to `ext_sources_dist.png`.

It uses absolute project paths, so it is clearly a local convenience script rather than a portable pipeline component.

### 16.2 `LLMutils.py`

This file is not part of the predictive pipeline. It is a documentation and asset-management helper for report/dissertation preparation.

Its functions include:

- extracting base64 images from markdown exports;
- extracting notebook PNG outputs from `.ipynb` files;
- converting all-caps lines into headers;
- replacing embedded base64 links with local file links;
- inserting slide images into markdown based on matching heuristics;
- deduplicating image references.

This is a strong signal that the repo has already entered the dissertation-assembly phase, not just the experimentation phase.

## 17. Design Decisions Embedded In The Repository

This section summarizes the major methodological choices that the repo repeatedly encodes.

### 17.1 Traditional vs combined data strategy

The codebase clearly treats "application-only" and "application + supplementary history" as two separate regimes. This is not accidental. It reflects the project's central research question: whether alternative and behavioral data materially improve default prediction beyond static application information.

### 17.2 One row per applicant as the canonical modeling grain

Every serious modeling path eventually collapses the relational tables to applicant-level records. This is the foundational architectural decision of the project.

### 17.3 Coverage flags as signal, not just bookkeeping

`has_bureau`, `has_bb`, `has_prev`, `has_pos`, `has_ins`, and `has_cc` are modeled as features because the absence of historical data is itself informative.

### 17.4 Split-before-fit for the dedicated preprocessing stage

The modular pipeline explicitly creates train/validation/test splits before fitting imputers, scalers, and PCA objects in the preprocessing stage. This is a strong anti-leakage design principle, even though some earlier deterministic transformations happen before splitting in the aggregation stage.

### 17.5 Interpretable and high-performance model families are both preserved

The repo never commits to only one modeling philosophy. It keeps:

- scorecard/logistic-regression style models for interpretability and credit-risk familiarity;
- tree ensembles and boosting models for nonlinear predictive performance.

### 17.6 Dimensionality reduction is treated as both engineering and interpretation

PCA appears in multiple places because the team is using it for two different reasons:

- reducing redundancy and stabilizing model inputs;
- understanding latent structure in the engineered feature space.

### 17.7 Governance is treated as part of the modeling problem

The presence of SHAP, LIME, Fairlearn, PSI, CSI, expected loss, threshold optimization, and governance markdown shows that the project is not framed as a Kaggle-only exercise. It is framed as a lending-risk solution that must also be explainable, monitorable, and defensible.

### 17.8 Artifact checkpointing is part of the workflow

The modular pipeline persists intermediate and final CSVs at multiple stages. This matters because it allows:

- repeatable downstream experiments;
- reuse of expensive feature engineering outputs;
- comparison across modeling branches without re-running the whole ETL chain.

## 18. Important Caveats And Dissertation Notes

These are not criticisms for their own sake. They are the implementation realities that should be remembered when converting repo work into dissertation prose.

| Topic | Current repo behavior | Why it matters in the dissertation |
| --- | --- | --- |
| Orchestration | `runner.ipynb` is a notebook, not a hardened CLI or package entrypoint | Present it as a research orchestration layer, not a production scheduler |
| Raw data fetch | Dataset download depends on Google Drive and runtime `gdown` installation | The pipeline is reproducible, but not self-contained offline |
| README quality | Current README mixes relevant and unrelated material | This report is a better technical source of truth than the README |
| Notebook execution context | Several notebooks assume Colab or specific relative paths | Reproducibility of notebooks is less uniform than reproducibility of the modular pipeline |
| Modeling duplication | `home_credit_modeling_combined.ipynb` duplicates aggregation/preprocessing logic inline | Results in that notebook should be described as experimental notebook research, not the sole canonical ETL implementation |
| Application-only notebooks | `final_traditional*` bypass pipeline artifacts and work directly from `application_train.csv` | These are useful baselines, but they are parallel experiments rather than the endpoint of the modular ETL path |
| Split strategy | Modular split uses fixed random seed but not stratification | Mention class imbalance handling elsewhere in the workflow when describing evaluation choices |
| Preprocessing persistence | Fitted scalers, imputers, PCA objects, and encoders are not serialized separately | The project persists transformed datasets rather than deployable preprocessing objects |
| PCA rotation implementation | Pipeline PCA path recomputes varimax on validation and test score matrices | This should be described carefully if claiming a canonical rotated-component methodology |
| Temporal/business framing | Combined-data notebooks sometimes use post-origination behavioral tables alongside origination scoring ideas | The dissertation should clearly distinguish application-time scoring from post-origination monitoring or behavioral scoring |
| Reject inference | `application_test.csv` is used as a proxy rejected population | This is an experimental approximation, not observed reject-outcome data |

## 19. Documentation And Presentation Assets Already In The Repo

The repo already contains a large amount of supporting material that can be reused when assembling the dissertation:

| Path | Contents |
| --- | --- |
| `docs/notebook_images/` | Extracted figures from notebooks |
| `docs/pptximages/` | Slide-deck images used in presentations |
| `docs/discussionimages/` | Curated discussion figures for later writing |
| `docs/*.pdf` and `docs/*.docx` | Exported reports, technical reports, and slide artifacts |
| `FINAL PPT converted.md` | Presentation text converted into markdown |
| `venky work - EDA-feature engineering report.md` | Narrative report on EDA/feature engineering work |
| `sandesh work - Predictive analytics and fairness audit technical report.md` | Narrative report on predictive modeling and fairness work |
| `LLMutils.py` | Utility code that supports image extraction and markdown cleanup for documentation |

This matters because the dissertation is not starting from zero. The repository already contains both code and many of the figures, summaries, and narrative fragments needed for a polished final document.

## 20. What This Repository Currently Delivers

Taken as a whole, the current repo already implements a substantial amount of work:

- raw data retrieval into a reproducible local directory structure;
- applicant-level feature engineering from seven supplementary relational tables;
- application-only and combined-data dataset construction;
- post-split preprocessing with both PCA and non-PCA variants;
- comprehensive raw-data EDA;
- application-only baseline modeling;
- combined-data research modeling with scorecard, boosting, ensembling, and score-family decomposition;
- fairness analysis, post-hoc explainability, and model monitoring experiments;
- expected-loss and profit-oriented threshold analysis;
- reject-inference experimentation;
- statistical PCA/factor-analysis support for dissertation interpretation;
- asset-management tooling for report assembly.

## 21. Closing Synthesis

The best way to understand this codebase is not as a single script, but as an ecosystem of connected workstreams:

- the `pipelines/` package is the reusable data-engineering backbone;
- `runner.ipynb` is the lightweight orchestrator for that backbone;
- `home_credit_eda.ipynb` documents the empirical understanding of the raw data;
- `home_credit_modeling_combined.ipynb` is the richest research notebook and contains the broadest set of modeling, governance, and business-facing analyses;
- `final_traditional.ipynb` and `final_traditional_pca.ipynb` provide application-only baselines;
- `modeling_undersample_fold.ipynb` is the cleanest benchmarking notebook over the four prepared dataset variants;
- `pca_factor_analysis.ipynb` and `src/pca.ipynb` support the dimensionality-reduction and interpretability story;
- `docs/`, slide exports, and `LLMutils.py` support the dissertation packaging phase.

If this project is later summarized in dissertation form, the most faithful high-level story is:

1. The team built a modular borrower-level data pipeline from a complex relational credit dataset.
2. They evaluated multiple modeling regimes: application-only, combined relational data, and PCA-reduced variants.
3. They retained both interpretable and high-performance model families.
4. They extended the analysis beyond prediction into explainability, fairness, monitoring, expected loss, and decision-threshold economics.
5. The repository already contains both the code and the documentation assets needed to turn that work into a rigorous dissertation narrative.
