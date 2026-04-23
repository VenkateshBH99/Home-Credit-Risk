# Home Credit Default Risk - Dissertation Project

## Project Overview

**Team:** Sovereign Prism (5 members: Kaushik, Venkatesh, Samarth, Norbert, Sandesh)
**Course:** EBA5008 - Intelligent Financial Risk Management, NUS ISS
**Title:** "The Invisible Applicant: Illuminating the Journey from Data Desert to Credit Worthiness"
**Supervisor:** Prof Dr Chirag Desai

The project builds a fairness-aware credit risk model for thin-file/credit-invisible borrowers using the Home Credit Kaggle dataset (307K applicants, 7 relational tables, 55M+ supplementary records). It evaluates LR, RF, XGBoost, LightGBM across 4 dataset variants (traditional/combined × PCA/no-PCA), with fairness analysis, SHAP/LIME explainability, and an A/B/C scoring framework.

## Dissertation Writing - Current State and Phases

### Phase 1: Template Restructure ✅ COMPLETE
- Restructured LaTeX from last semester's pairs-trading project to 8-chapter credit risk dissertation
- Created all chapter .tex files with first-draft content and TODO markers
- Copied 197 images into `images/` (organized: pptx/, notebook/, discussion/, venky/, cover/)
- Updated frontmatter (abstract, abbreviations, acknowledgements)
- Updated references.bib with 17 project-specific references
- Cleaned up old files (old .bib files, old chapter content, old proposal PDF)

### Phase 2: Content Writing 🔄 IN PROGRESS
Fill in all TODO markers across chapters 1-8. Each chapter file has `% TODO:` comments showing exactly what needs to be written.

Priority order:
1. Chapter 1 (Introduction) - mostly done, needs figure uncommenting
2. Chapter 3 (Data Collection) - mostly done, needs figure integration + additional EDA prose
3. Chapter 4 (Feature Engineering) - mostly done, needs figures
4. Chapter 5 (Predictive Modeling) - needs ensemble/CV/model comparison sections expanded
5. Chapter 6 (Credit Scoring) - needs expected loss and reject inference expansion
6. Chapter 7 (Fairness) - needs SHAP per-model comparison expanded, LIME section
7. Chapter 8 (Conclusion) - mostly done, light polish
8. Chapter 2 (Literature Review) - HEAVIEST LIFT, mostly section headers with TODOs

### Phase 3: Image Integration ⬜ NOT STARTED
- Uncomment all `\includegraphics` calls across chapters
- Verify image paths resolve (images are in `images/pptx/`, `images/notebook/`, etc.)
- Write proper captions for each figure
- Decide which notebook images go into main body vs Appendix D
- Populate Appendix D with supplementary figures

### Phase 4: Polish ⬜ NOT STARTED
- Fix cross-references (compile twice for \ref resolution)
- Run biber for bibliography
- Verify List of Figures, List of Tables render
- Verify abbreviations render (acronym package)
- Final proofreading pass
- Ensure all professor feedback points are addressed

## LaTeX Compilation

**Engine:** LuaLaTeX required (nus.cls uses fontspec + Lato/Consolas fonts)
**Compile sequence:** `lualatex -> biber -> lualatex -> lualatex`
**Known issue:** Lato and Consolas fonts must be installed on the system. Works on Overleaf with the template.
**Known issue:** `??` in cross-references resolves after second lualatex pass.

## Directory Structure

```
SovereignPrism_IFRM_dissertation/
├── dissertation_main.tex          # Main file - \input all chapters
├── nus.cls                        # NUS class file (DO NOT MODIFY)
├── references.bib                 # All bibliography entries
├── frontmatter/
│   ├── abstract.tex               # Updated for Home Credit
│   ├── abbreviations.tex          # 38 acronyms (AUC, SHAP, PSI, etc.)
│   ├── acknowledgements.tex       # Updated for this semester
│   ├── copyright.tex
│   └── dedication.tex
├── chap1/introduction_main.tex    # Ch1: Introduction (background, biz/tech objectives, scope, stakeholders)
├── chap2/literature_review.tex    # Ch2: Literature Review (MOST TODOs - needs heavy writing)
├── chap3/data_collection.tex      # Ch3: Data Collection & EDA (good first draft)
├── chap4/feature_engineering.tex  # Ch4: Feature Engineering & Selection (good first draft)
├── chap5/predictive_modeling.tex  # Ch5: Predictive Modeling & Results (needs expansion)
├── chap6/credit_scoring.tex       # Ch6: Credit Scoring & Business Application
├── chap7/fairness_explainability.tex # Ch7: Fairness, Explainability, Governance
├── chap8/conclusion.tex           # Ch8: Conclusion & Future Work
├── appA/appendix_a_main.tex       # App A: Project Proposal
├── appB/appendix_b_main.tex       # App B: Mapped System Functionalities
├── appC/appendix_c_main.tex       # App C: Code Architecture
├── appD/appendix_d_main.tex       # App D: Supplementary Figures (needs populating)
└── images/
    ├── pptx/                      # 44 images from presentation slides
    ├── notebook/                  # 112 images from Jupyter notebooks
    ├── discussion/                # 22 images from design discussions
    ├── venky/                     # 19 images from Venky's EDA report
    ├── cover/                     # Cover image
    └── Table_Relations_home_credit.png  # Physical data model
```

## Source Material Locations

All source material for writing is in the repo root:
- `FINAL PPT converted.md` - Full presentation content (38 slides) with image references
- `FINAL PPT script.md` - Presentation scripts showing reasoning and talking points
- `proposal.md` - Original project proposal with detailed methodology
- `sandesh work - Predictive analytics and fairness audit technical report.md` - Modeling & fairness deep dive
- `venky work - EDA-feature engineering report.md` - EDA & feature engineering deep dive
- `home_credit_technical_report_v1.md` - Codex's comprehensive technical documentation of the repo
- `docs/*.pdf` - Claude-generated EDA reports per supplementary table
- `docs/reference.pdf` - Another team's analysis of the same dataset

## Professor's Feedback (MUST be addressed)

1. **Business vs Technical objectives** - DONE in Ch1 (separate sections 1.3 and 1.4)
2. **How alternative data helped or not** - DONE in Ch5 Section 5.5, needs expansion
3. **Credit scoring dataset disjointness** - DONE in Ch6 Section 6.2 (ABC framework)
4. **Why no Decision Tree baseline** - DONE in Ch5 Section 5.1.3
5. **Business goal slide = business objectives** - DONE in Ch1

## Key Quantitative Results to Reference

| Metric | Value | Context |
|---|---|---|
| Best AUC (traditional) | 0.76 | LightGBM, no PCA |
| Best AUC (combined) | 0.78 | LightGBM, undersampled |
| AUC uplift from alt data | +3.1% | Combined vs application-only |
| Best Recall (traditional) | 0.69 | LightGBM |
| KS Statistic | 0.428 | "Good" band (0.4-0.6) |
| Lift (top decile) | 3.74× | Top 10% are 3.74× more likely to default |
| Score differential | 20× | Lowest vs highest score band default rate |
| A-Score AUC | 0.716 | 170 features, application-time |
| B-Score AUC | 0.616 | 34 features, post-origination |
| C-Score AUC | 0.579 | 13 features, collections |
| Default rate | 8.07% | 1:11.4 class imbalance |
| IV > 0.02 features | ~70 | Out of 163 total |
| PCA for 90% variance | 51 components | No strong latent structure |
| EXT_SOURCE_1 corr | -0.155 | 56.4% missing |
| EXT_SOURCE_2 corr | -0.161 | 0.2% missing |
| EXT_SOURCE_3 corr | -0.179 | 19.8% missing, 814 unique values |

## Image Naming Convention

Images in `images/pptx/` have descriptive names matching PPT slides, e.g.:
- `SHAP explainability - LightGBM - combined.png`
- `customer risk scoring validatoin results - default rate by score band.png`
- `fairness evaluation by gender.png`

Images in `images/notebook/` follow pattern: `{notebook}_{section}_{description}_{number}.png`
Images in `images/discussion/` have descriptive names of design decisions
Images in `images/venky/` have descriptive names of EDA visualisations

## Writing Style Notes

- British English spelling (colour, behaviour, modelling, etc.) used throughout
- Academic/formal tone suitable for NUS ISS dissertation
- Every claim backed by figure reference or table
- Figures commented out with `% TODO:` - uncomment when integrating images
- All `\label` and `\ref` use consistent naming: `chap:`, `sec:`, `fig:`, `tab:`


# chat history
This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:
   The user (Samarth, NUS ISS Masters student) asked for help writing a LaTeX dissertation for their Home Credit Default Risk project ("The Invisible Applicant"). The project builds a fairness-aware credit risk model for thin-file/credit-invisible borrowers using the Kaggle Home Credit dataset (307K applicants, 7 relational tables, 55M+ supplementary records). The team is Sovereign Prism (5 members). The user wanted: (a) thorough understanding of all project materials, (b) a dissertation structure plan adapted from last semester's template, (c) execution of that plan chapter by chapter. The user explicitly said NOT to read the actual LaTeX content from last semester (just use it as a structural template to overwrite). The user also requested a CLAUDE.md for handover continuity to Codex if session limits are reached. There are 4 phases total: Phase 1 (Template Restructure - DONE), Phase 2 (Content Writing - IN PROGRESS), Phase 3 (Image Integration), Phase 4 (Polish).

2. Key Technical Concepts:
   - Credit risk modeling with LR, Random Forest, XGBoost, LightGBM across 4 dataset variants (traditional/combined × PCA/no-PCA)
   - A/B/C credit scoring framework (origination, behavioral, collection scores)
   - WoE/IV feature selection for scorecards
   - Feature engineering: aggregating 7 relational tables to applicant-level (55M+ rows → 307K applicants)
   - Class imbalance handling (8.07% default rate, 1:11.4 ratio): class_weight, scale_pos_weight, ensemble undersampling
   - EXT_SOURCE_1/2/3 interpretation (private bureau / telco / government scores)
   - Fairlearn fairness evaluation (demographic parity, equalized odds, ThresholdOptimizer)
   - SHAP TreeExplainer and LIME for explainability
   - PSI/CSI model monitoring
   - Expected loss (PD × LGD × EAD) and cost-sensitive threshold optimization
   - PCA/factor analysis with varimax rotation
   - Logistic regression scorecard (300-850 scale, PDO=20)
   - Professor's 5 feedback points that must be addressed

3. Files and Code Sections:

   - `/Users/SamarthSoni/Projects/Home-Credit-Risk/CLAUDE.md` (CREATED)
     - Comprehensive handover document for continuity across sessions
     - Contains: project overview, current state of all 4 phases, directory structure, source material locations, professor's feedback, key quantitative results, image naming conventions, writing style notes
     - Critical for Codex or future Claude sessions to pick up where this session left off

   - `SovereignPrism_IFRM_dissertation/dissertation_main.tex` (REWRITTEN)
     - Main LaTeX file restructured from 6 chapters to 8 chapters + 4 appendices
     - Updated `\graphicspath` to include all image subdirectories: `{./images/}{./images/pptx/}{./images/notebook/}{./images/discussion/}{./images/venky/}{./images/cover/}` plus chapter-specific dirs
     - Changed chapter inputs from old names to new: `chap1/introduction_main`, `chap2/literature_review`, `chap3/data_collection`, `chap4/feature_engineering`, `chap5/predictive_modeling`, `chap6/credit_scoring`, `chap7/fairness_explainability`, `chap8/conclusion`
     - Added appendices C and D: `appC/appendix_c_main`, `appD/appendix_d_main`
     - Removed old bib resource lines (chap1, chap2, chap3 bibs), kept only `references.bib`

   - `SovereignPrism_IFRM_dissertation/chap1/introduction_main.tex` (REWRITTEN + EXPANDED)
     - Fully written with: Background/Motivation, Business Problem, Business Objectives (sec 1.3), Technical Objectives (sec 1.4), Project Scope (data/methodology/out-of-scope), Stakeholder Analysis, Report Organisation
     - Contains 3 uncommented stakeholder figures, model architecture figure, pipeline stages table, dataset overview table
     - Labels: `\label{chap:introduction}`, `\label{sec:business_objectives}`, `\label{sec:technical_objectives}`, etc.
     - Cross-references chapters 2-8 using `\ref{chap:litreview}`, `\ref{chap:data}`, etc.

   - `SovereignPrism_IFRM_dissertation/chap2/literature_review.tex` (FULLY WRITTEN - was the heaviest lift)
     - ~300 lines of complete prose covering: Traditional Credit Scoring (bureau systems, A/B/C taxonomy, observation-scoring-performance window), Alternative Data (types, prior Kaggle work, regulatory landscape), ML for Credit Risk (evolution LR→GBM, class imbalance, why trees suit credit, why no Decision Tree baseline), Evaluation Metrics (accuracy failure, recall/precision trade-off, AUC-ROC vs PR curves, KS/Gini/calibration/lift, profit/loss threshold), Algorithmic Fairness (definitions, sources of bias, mitigation strategies), Model Interpretability (SHAP, LIME, PSI/CSI monitoring, regulatory requirements)
     - Includes: alt data proxy mapping table, PR vs ROC figure, profit formula equations, threshold equation
     - References: cite keys cfpb2022, siddiqi2012, berg2020, kaggle2018, krakow2018, narayanan2020, breiman2001, chen2016, ke2017, chawla2002, fawcett2006, saito2015, hardt2016, bird2020, lundberg2017, ribeiro2016, yurdakul2020, occ2011

   - `SovereignPrism_IFRM_dissertation/chap3/data_collection.tex` (FIRST DRAFT)
     - Covers: Dataset Overview (physical data model figure, relational linkage), Data Quality (missing values, DAYS_EMPLOYED anomaly, outliers), EDA sections (target variable, categorical, numerical, EXT_SOURCE with interpretation, correlation, bureau, previous apps, behavioral data, additional EDA), Preprocessing Pipeline
     - Most figures still commented out with `% TODO:` markers
     - EXT_SOURCE interpretation included: EXT_SOURCE_1 = private bureau (56.4% missing), EXT_SOURCE_2 = telco (0.2% missing), EXT_SOURCE_3 = government score band (814 unique values)

   - `SovereignPrism_IFRM_dissertation/chap4/feature_engineering.tex` (FIRST DRAFT)
     - Covers: Aggregation Architecture (coverage flags design), Feature Engineering by Source (application, bureau, previous apps, POS/CC, installments), Feature-Space Evolution table (337→242→208), Traditional vs Alternative Data Strategy (with alt-data proxy table), Feature Selection (WoE/IV with distribution table, variance filtering, correlation removal), PCA Analysis
     - Key table: Feature-Space Evolution showing column counts at each pipeline stage

   - `SovereignPrism_IFRM_dissertation/chap5/predictive_modeling.tex` (FIRST DRAFT)
     - Contains: Experimental Design (4 variants, 4 models, DT justification, imbalance handling), Model Selection Rationale (per model), Traditional Results table, Combined Results table, Alt Data Impact section, placeholders for ensemble/CV/comparison
     - Results tables include Accuracy, Recall, AUC, Misclassified Defaults for all models

   - `SovereignPrism_IFRM_dissertation/chap6/credit_scoring.tex` (FIRST DRAFT)
     - Contains: Scorecard (formula, score bands table, validation), ABC Framework (with feature table showing AUC per score), Validation Diagnostics (KS=0.428, calibration, lift=3.74×), Expected Loss (EL=PD×LGD×EAD, profit formula, cost-ratio threshold), Reject Inference (hard cutoff, fuzzy augmentation)

   - `SovereignPrism_IFRM_dissertation/chap7/fairness_explainability.tex` (FIRST DRAFT)
     - Contains: SHAP per-model comparison (LR, RF, XGBoost, LightGBM), LIME section, Fairness Evaluation (gender analysis, XNA group, accuracy-fairness trade-off, mitigation strategies), Model Monitoring (PSI, CSI, AUC drift, fairness stability), Governance (consent framework, risk mitigation matrix table)

   - `SovereignPrism_IFRM_dissertation/chap8/conclusion.tex` (FIRST DRAFT)
     - Contains: Summary of Findings (6 key points), Addressing Professor's Feedback (explicit section), Limitations (data/model/fairness), Recommendations table, Future Work (7 items), Data Governance in Production

   - `SovereignPrism_IFRM_dissertation/appA-D/*.tex` (STUBS)
     - appA: Project Proposal (placeholder for PDF include)
     - appB: Mapped System Functionalities (updated for Home Credit/IFRM)
     - appC: Code Architecture (repo structure table, tech stack, pipeline flow, dataset variants)
     - appD: Supplementary Figures (section headers for organizing notebook images)

   - `SovereignPrism_IFRM_dissertation/frontmatter/abstract.tex` (REWRITTEN)
     - Updated from FinSight to Home Credit project, includes key findings, keywords

   - `SovereignPrism_IFRM_dissertation/frontmatter/abbreviations.tex` (REWRITTEN)
     - 38 acronyms relevant to credit risk (AUC, SHAP, PSI, WoE, PD, LGD, EAD, etc.)

   - `SovereignPrism_IFRM_dissertation/frontmatter/acknowledgements.tex` (UPDATED)
     - Changed professor name to Prof Dr Chirag Desai, removed FinSight-specific references

   - `SovereignPrism_IFRM_dissertation/references.bib` (REWRITTEN)
     - 17 entries replacing old pairs-trading references: siddiqi2012, cfpb2022, bird2020, hardt2016, lundberg2017, ribeiro2016, chen2016, ke2017, breiman2001, chawla2002, fawcett2006, saito2015, kaggle2018, jolliffe2002, yurdakul2020, berg2020, krakow2018, narayanan2020, occ2011

   - `SovereignPrism_IFRM_dissertation/images/` (POPULATED)
     - `images/pptx/` - 44 presentation images
     - `images/notebook/` - 112 notebook output images
     - `images/discussion/` - 22 design discussion images
     - `images/venky/` - 19 EDA report images
     - `images/cover/cover.png` - cover image
     - `images/Table_Relations_home_credit.png` - physical data model

   - Old files DELETED: `chap1/introduction_biblio.bib`, `chap2/background_and_lit_overview_biblio.bib`, `chap3/materials_and_methods_biblio.bib`, `chap2/Market Research.tex`, `chap3/Data Collection and Preparation.tex`, `chap4/System Design.tex`, `chap4/analyst_subsys.tex`, `chap5/Result and Progress.tex`, `chap6/conclusion_main.tex`, `appA/Group_14_proposal_final.pdf`

4. Errors and Fixes:
   - **chap1 Write failed**: "File has not been read yet" error because `introduction_main.tex` already existed. Fixed by reading the file first, then writing.
   - **LaTeX compilation with pdflatex**: Failed with "Fatal Package fontspec Error: requires XeTeX or LuaTeX". The nus.cls requires LuaLaTeX.
   - **LaTeX compilation with xelatex**: Failed with "Class nus Error: This class requires LuaLaTeX" and missing Lato/Consolas fonts.
   - **LaTeX compilation with lualatex**: Font errors (Lato, Consolas not installed on system). User said to ignore compilation issues and focus on content writing.
   - **User feedback on font checking**: User rejected the font-checking tool call saying "focus on writing the report, as long as your syntax is proper these library issues can be fixed later" - this is important guidance to NOT spend time on compilation/font issues.
   - **?? references in Report Organisation**: These resolve with multiple lualatex compilation passes (needs 2+ passes for cross-references). Not a content issue.
   - **List of figures/tables not rendering**: Because most figures are still commented out with `% TODO:` markers. Will resolve as figures are uncommented in Phase 2/3.

5. Problem Solving:
   - Adapted last semester's multi-module system structure (5 modules: news, stocks, trading, AI analyst) to a single ML pipeline research structure (8 chapters following the data→engineering→modeling→scoring→fairness→conclusion arc)
   - Addressed all 5 professor feedback points explicitly in the chapter structure
   - Created comprehensive CLAUDE.md for handover continuity
   - Identified that Chapter 2 (Literature Review) was the heaviest writing lift and completed it fully
   - Organized 197 images across 5 subdirectories with descriptive naming for easy reference

6. All User Messages:
   - **Message 1**: Long initial request explaining the project, pointing to all source materials, explaining the dissertation.md template from last semester, asking to understand the project in detail and create a plan. Key quote: "your major task is to go through the various docs... think of how a dissertation for this project might look like... come up with the detailed plan of the work that we must do now." Also: "SPECIFICALLY NOT GO THROUGH THE ACTUAL CONTENTS IN THE CURRENT LATEX, WHICH IS JUST A TEMPLATE FOR US TO OVERWRITE OVER"
   - **Message 2**: "yes" - approving the plan and asking to proceed with Phase 1
   - **Message 3**: "yes. How many phases are there? i want to ensure continuity can be achieved say using codex if your session limit runs out. so can you as immediate next step create a detailed claude.md for handover if need be and then continue. Just also to let you know, currently the list of figures/tables/abbreviations isn't rendered on the pdf even though i see some tables. also some chapter numbers show as ?? right now under report organisation. now start"
   - **Message 4** (tool rejection): "focus on writing the report, as long as your syntax is proper these library issues can be fixed later"

7. Pending Tasks:
   - Phase 2c: Fill Chapter 3 (Data Collection) - expand EDA sections, uncomment figures
   - Phase 2d: Fill Chapter 4 (Feature Engineering) - uncomment figures, expand prose
   - Phase 2e: Fill Chapter 5 (Predictive Modeling) - expand ensemble/CV/model comparison sections
   - Phase 2f: Fill Chapter 6 (Credit Scoring) - expand expected loss and reject inference
   - Phase 2g: Fill Chapter 7 (Fairness) - expand SHAP per-model comparison, LIME section
   - Phase 2h: Fill Chapter 8 (Conclusion) - final polish
   - Phase 3: Image integration - uncomment all `\includegraphics` calls, verify paths, write captions
   - Phase 4: Polish - cross-references, bibliography (biber), appendix D population, final compilation

8. Current Work:
   I had just finished writing the complete Chapter 2 (Literature Review) - `SovereignPrism_IFRM_dissertation/chap2/literature_review.tex`. This was the heaviest content lift, going from section headers with TODO comments to ~300 lines of full academic prose covering traditional credit scoring, alternative data, ML for credit risk, evaluation metrics, algorithmic fairness, and model interpretability. Before that, I expanded Chapter 1 by uncommenting 3 stakeholder figures, adding a pipeline stages table, and adding the model architecture figure. The todo list shows Chapter 2 as "in_progress" (just completed the write), with Chapters 3-8 still pending.

9. Optional Next Step:
   Continue with Phase 2c: Fill Chapter 3 (Data Collection and Preparation). This chapter already has a good first draft but needs: (a) uncommenting of figures (target distribution donut, data quality tables, financial feature distributions, EXT_SOURCE distributions, correlation matrix, bureau/previous app/behavioral data figures), (b) expanding the Additional EDA section (flag documents, temporal features, credit bureau inquiry, binary flags, skewness), and (c) richer prose in the categorical and numerical feature analysis subsections. The user explicitly said "now start" and to focus on content writing. The most recent work was completing Chapter 2, so the natural continuation is Chapter 3 as listed in the todo progression.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /Users/SamarthSoni/.claude/projects/-Users-SamarthSoni-Projects-Home-Credit-Risk/8c0a1093-242d-448e-94e5-c99472ff1791.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.