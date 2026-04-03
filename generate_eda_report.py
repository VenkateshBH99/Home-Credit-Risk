"""
Generate a comprehensive PDF report from the Home Credit EDA notebook.
Extracts all images and combines them with detailed analysis commentary.
"""

import json
import base64
import os
import tempfile
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
from io import BytesIO

# ==============================================================================
# Configuration
# ==============================================================================
NOTEBOOK_PATH = "home_credit_eda.ipynb"
OUTPUT_PDF = "Home_Credit_EDA_Report.pdf"
PAGE_W, PAGE_H = A4
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# ==============================================================================
# Styles
# ==============================================================================
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'CustomTitle', parent=styles['Title'],
    fontSize=28, leading=34, spaceAfter=6,
    textColor=HexColor('#1a237e'), alignment=TA_CENTER
)
subtitle_style = ParagraphStyle(
    'CustomSubtitle', parent=styles['Normal'],
    fontSize=14, leading=18, spaceAfter=20,
    textColor=HexColor('#455a64'), alignment=TA_CENTER
)
heading1_style = ParagraphStyle(
    'Heading1', parent=styles['Heading1'],
    fontSize=18, leading=22, spaceBefore=16, spaceAfter=8,
    textColor=HexColor('#1565c0'), borderWidth=1,
    borderColor=HexColor('#1565c0'), borderPadding=4
)
heading2_style = ParagraphStyle(
    'Heading2', parent=styles['Heading2'],
    fontSize=14, leading=17, spaceBefore=12, spaceAfter=6,
    textColor=HexColor('#1976d2')
)
body_style = ParagraphStyle(
    'BodyText', parent=styles['Normal'],
    fontSize=10.5, leading=14, spaceAfter=8,
    alignment=TA_JUSTIFY
)
bullet_style = ParagraphStyle(
    'BulletText', parent=body_style,
    leftIndent=18, bulletIndent=6, spaceAfter=4
)
key_finding_style = ParagraphStyle(
    'KeyFinding', parent=body_style,
    leftIndent=12, borderWidth=1, borderColor=HexColor('#4caf50'),
    borderPadding=6, backColor=HexColor('#e8f5e9'), spaceAfter=10
)
table_note_style = ParagraphStyle(
    'TableNote', parent=body_style,
    fontSize=9, leading=12, textColor=HexColor('#616161'),
    spaceAfter=4, alignment=TA_LEFT
)

# ==============================================================================
# Helper functions
# ==============================================================================

def extract_images_from_notebook(nb_path):
    """Extract all base64-encoded PNG images from notebook cell outputs."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    images = []  # list of (cell_index, temp_file_path)
    temp_dir = tempfile.mkdtemp(prefix='eda_report_')

    for ci, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        for output in cell.get('outputs', []):
            img_data = None
            if output.get('output_type') == 'display_data' or output.get('output_type') == 'execute_result':
                img_data = output.get('data', {}).get('image/png')
            if img_data:
                if isinstance(img_data, list):
                    img_data = ''.join(img_data)
                raw = base64.b64decode(img_data)
                # Get image dimensions
                pil_img = PILImage.open(BytesIO(raw))
                w, h = pil_img.size
                fname = os.path.join(temp_dir, f'cell_{ci}_img.png')
                with open(fname, 'wb') as out:
                    out.write(raw)
                images.append((ci, fname, w, h))
    return images, temp_dir


def add_image(story, img_path, orig_w, orig_h, max_w=None, max_h=None):
    """Add an image to story, scaled to fit within max_w x max_h."""
    if max_w is None:
        max_w = CONTENT_W
    if max_h is None:
        max_h = 4.5 * inch

    aspect = orig_w / orig_h
    w = max_w
    h = w / aspect
    if h > max_h:
        h = max_h
        w = h * aspect
    story.append(Image(img_path, width=w, height=h))
    story.append(Spacer(1, 6))


def hr():
    return HRFlowable(width="100%", thickness=1, color=HexColor('#bdbdbd'),
                      spaceBefore=6, spaceAfter=6)


# ==============================================================================
# Report content – detailed analysis for each section
# ==============================================================================

SECTIONS = [
    {
        "title": "1. Dataset Overview",
        "image_keywords": [],  # no image for this section
        "body": [
            ("text", "The Home Credit Default Risk dataset comprises <b>8 interrelated CSV tables</b> capturing comprehensive information about loan applicants. The primary table, <b>application_train.csv</b>, contains 307,511 loan applications with 122 features and a binary TARGET variable (1 = default, 0 = no default)."),
            ("text", "Supplementary tables include:"),
            ("bullet", "<b>bureau.csv</b> — Credit history from other financial institutions (1,716,428 records)"),
            ("bullet", "<b>bureau_balance.csv</b> — Monthly balance snapshots for bureau credits (~27.3M records)"),
            ("bullet", "<b>previous_application.csv</b> — Previous Home Credit applications (1,670,214 records)"),
            ("bullet", "<b>POS_CASH_balance.csv</b> — Monthly POS/cash loan snapshots (10,001,358 records)"),
            ("bullet", "<b>credit_card_balance.csv</b> — Monthly credit card balance snapshots (3,840,312 records)"),
            ("bullet", "<b>installments_payments.csv</b> — Payment history for previous credits (13,605,401 records)"),
            ("bullet", "<b>sample_submission.csv / application_test.csv</b> — Test set (48,744 applications)"),
            ("text", "Together, the dataset spans approximately <b>2.5 GB</b> and provides a rich, multi-dimensional view of each applicant's financial behavior."),
        ]
    },
    {
        "title": "2. Target Variable Distribution",
        "image_count": 1,
        "body": [
            ("text", "The target variable exhibits <b>severe class imbalance</b>: 91.93% of applications resulted in no default (TARGET=0), while only 8.07% defaulted (TARGET=1). This 11.4:1 ratio poses significant challenges for model training."),
            ("heading2", "Key Observations"),
            ("bullet", "Out of 307,511 applications, only <b>24,825</b> resulted in payment difficulties (default)"),
            ("bullet", "The remaining <b>282,686</b> applications were repaid successfully"),
            ("bullet", "This imbalance necessitates techniques like SMOTE, class weighting, or stratified sampling during modeling"),
            ("heading2", "Implications for Modeling"),
            ("text", "A naive classifier predicting 'no default' for all applications would achieve 91.93% accuracy, making accuracy an unreliable metric. The evaluation should focus on <b>AUC-ROC</b>, which measures ranking ability regardless of threshold. Precision-Recall curves and the F1 score are also more informative for this skewed distribution."),
        ]
    },
    {
        "title": "3. Missing Values Analysis",
        "image_count": 1,
        "body": [
            ("text", "Missing data is pervasive across the application dataset. The analysis reveals that several feature groups have <b>40–70%</b> missing values, while other critical features are complete."),
            ("heading2", "Key Findings"),
            ("bullet", "<b>COMMONAREA_MEDI/AVG/MODE</b> — Highest missingness at ~69.9%. These normalized housing features are unavailable for the majority of applicants."),
            ("bullet", "<b>NONLIVINGAPARTMENTS, FONDKAPREMONT_MODE, LIVINGAPARTMENTS</b> — All above 60% missing."),
            ("bullet", "<b>Housing-related features</b> (APARTMENTS, BASEMENTAREA, YEARS_BUILD, etc.) cluster around 47–65% missing."),
            ("bullet", "<b>OWN_CAR_AGE</b> — ~66% missing, corresponding closely to the ~66% of applicants who don't own a car."),
            ("bullet", "<b>Financial features</b> (AMT_INCOME, AMT_CREDIT, AMT_ANNUITY) have near-zero missingness."),
            ("heading2", "Treatment Strategy"),
            ("text", "Missing values in housing features likely indicate that the applicant didn't provide this information, which itself may be predictive. Creating <b>binary missingness indicators</b> for high-missing features can capture this signal. For numerical features, imputation with median values is standard, while categorical features can use a dedicated 'Missing' category."),
        ]
    },
    {
        "title": "4. Categorical Features Analysis",
        "image_count": 1,
        "body": [
            ("text", "Eight key categorical features were analyzed for their distribution and relationship with default rate:"),
            ("heading2", "Contract Type (NAME_CONTRACT_TYPE)"),
            ("bullet", "<b>Cash loans</b> dominate (~90%) with a higher default rate (~8.5%) vs <b>Revolving loans</b> (~10%) with a lower default rate (~5.5%)."),
            ("heading2", "Gender (CODE_GENDER)"),
            ("bullet", "<b>Males</b> have a notably higher default rate (~10%) compared to <b>females</b> (~7%). Females constitute about 65% of applicants."),
            ("heading2", "Income Type"),
            ("bullet", "<b>Working</b> is the most common category; <b>Maternity leave</b> and <b>Unemployed</b> (rare) show significantly higher default rates (~40% and ~36%, respectively). <b>Pensioners</b> have among the lowest default rates (~5%)."),
            ("heading2", "Education"),
            ("bullet", "Applicants with <b>lower secondary</b> education show the highest default rate (~11%), while <b>academic degree</b> holders default least (~2%)."),
            ("heading2", "Family Status"),
            ("bullet", "<b>Single / not married</b> and <b>civil marriage</b> applicants default more (~10%) than <b>married</b> or <b>widowed</b> applicants (~7–8%)."),
            ("heading2", "Housing Type"),
            ("bullet", "Applicants living in <b>rented apartments</b> or <b>with parents</b> have higher default rates (~12–10%) than <b>house/apartment owners</b> (~7.5%)."),
        ]
    },
    {
        "title": "5. Numerical Feature Distributions (KDE)",
        "image_count": 1,
        "body": [
            ("text", "Kernel Density Estimation (KDE) plots reveal the distribution shape and separation between defaulters and non-defaulters for four core financial features:"),
            ("heading2", "AMT_INCOME_TOTAL"),
            ("bullet", "Highly right-skewed with most applicants earning below 300,000. The KDE curves for defaulters and non-defaulters overlap heavily, indicating limited standalone discriminative power."),
            ("heading2", "AMT_CREDIT"),
            ("bullet", "Multi-modal distribution with peaks near 250K, 500K, and 675K. Defaulters show a slight shift toward mid-range credits."),
            ("heading2", "AMT_ANNUITY"),
            ("bullet", "Roughly normal, centered around 25,000. The two groups are nearly indistinguishable from the density plot alone."),
            ("heading2", "AMT_GOODS_PRICE"),
            ("bullet", "Spiky distribution reflecting discrete product price tiers (e.g., 225K, 450K, 675K). Similar overlap between the two classes."),
            ("text", "Overall, these raw financial features show <b>limited univariate separation</b> between default and non-default groups, suggesting that <b>feature engineering</b> (ratios, interactions) and multi-variate models are necessary to capture the default signal."),
        ]
    },
    {
        "title": "6. Age Analysis (DAYS_BIRTH)",
        "image_count": 1,
        "body": [
            ("text", "The age distribution and its relationship with default risk show a clear and monotonic pattern:"),
            ("heading2", "Key Findings"),
            ("bullet", "<b>Younger applicants default more</b>: The 20–25 age group has a ~12.5% default rate, significantly above the overall 8.07% average."),
            ("bullet", "<b>Default rate declines steadily with age</b>: Each successive 5-year age bin shows a lower rate — from ~12.5% (20–25) to ~4% (65–70)."),
            ("bullet", "The <b>age distribution</b> peaks at 30–40 years, indicating the core applicant base."),
            ("bullet", "The age–default relationship is nearly linear in the 25–60 range, making age (DAYS_BIRTH / -365) a strong, reliable predictor."),
            ("heading2", "Interpretation"),
            ("text", "Younger applicants tend to have shorter credit histories, less stable employment, and lower savings — all contributing to higher default risk. This feature is one of the most interpretable predictors and is consistently ranked among the top features in importance analyses."),
        ]
    },
    {
        "title": "7. DAYS_EMPLOYED Anomaly",
        "image_count": 1,
        "body": [
            ("text", "The DAYS_EMPLOYED feature contains a major data anomaly: approximately <b>55,374 applicants</b> (~18%) have a sentinel value of <b>365,243 days</b> (equivalent to ~1,000 years of employment), which is clearly not a real measurement."),
            ("heading2", "Key Observations"),
            ("bullet", "After removing the anomaly, the realistic employment duration ranges from 0 to ~49 years, with a peak at 0–5 years."),
            ("bullet", "Applicants with the anomalous value actually have a <b>lower default rate</b> (~5.4%) compared to the general population (~8.07%)."),
            ("bullet", "This suggests the sentinel value represents a specific applicant group (possibly pensioners, or applicants for whom employment data is not applicable)."),
            ("heading2", "Recommended Treatment"),
            ("text", "Replace the sentinel value 365,243 with NaN, and create a binary indicator feature <b>DAYS_EMPLOYED_ANOM</b> (1 if anomalous, 0 otherwise). This preserves the information that these applicants are different while allowing proper analysis of the realistic employment range."),
        ]
    },
    {
        "title": "8. External Source Scores (EXT_SOURCE_1/2/3)",
        "image_count": 1,
        "body": [
            ("text", "The three external source scores are the <b>single most powerful predictors</b> in the dataset, each showing strong separation between defaulters and non-defaulters:"),
            ("heading2", "EXT_SOURCE_1 (correlation: -0.155)"),
            ("bullet", "Ranges from 0 to ~0.9. Defaulters are concentrated at lower values (0–0.3), while non-defaulters spread more evenly, peaking around 0.5–0.7."),
            ("heading2", "EXT_SOURCE_2 (correlation: -0.160)"),
            ("bullet", "The most complete of the three (only ~0.2% missing). Shows the clearest bimodal separation — defaulters peak at 0.0–0.2, non-defaulters peak at 0.5–0.6."),
            ("heading2", "EXT_SOURCE_3 (correlation: -0.179)"),
            ("bullet", "Strongest individual correlation with TARGET. Defaulters cluster below 0.3, non-defaulters peak around 0.4–0.5. Has ~19.8% missing values."),
            ("heading2", "Modeling Implications"),
            ("text", "These externally-sourced scores likely represent normalized credit scores from external bureaus. They should be prioritized in feature selection. <b>Interaction features</b> (e.g., EXT_SOURCE_2 × EXT_SOURCE_3, mean of all three) typically rank among the top 5 features in competitive models and can boost AUC significantly."),
        ]
    },
    {
        "title": "9. Box Plots — Financial Features by Target",
        "image_count": 1,
        "body": [
            ("text", "Box plots comparing the distribution of key financial features between defaulters (TARGET=1) and non-defaulters (TARGET=0) reveal:"),
            ("bullet", "<b>AMT_INCOME_TOTAL</b>: Nearly identical distributions for both groups, with extensive outliers on the high end. Income alone has minimal discriminative power."),
            ("bullet", "<b>AMT_CREDIT</b>: Defaulters tend to have slightly higher median credit amounts. The interquartile range for defaulters shifts upward compared to non-defaulters."),
            ("bullet", "<b>AMT_ANNUITY</b>: Minimal difference between groups at the median level, but defaulters show slightly higher annuity amounts on average."),
            ("text", "The presence of <b>extreme outliers</b> in AMT_INCOME_TOTAL (values exceeding 10M) may require capping or log transformation for certain algorithms. The relatively small differences between groups reinforce the need for multivariate models rather than simple threshold rules."),
        ]
    },
    {
        "title": "10. Correlation Analysis",
        "image_count": 2,
        "body": [
            ("text", "The correlation analysis examines both the top features most correlated with TARGET and the inter-feature correlations:"),
            ("heading2", "Top Correlations with TARGET"),
            ("bullet", "<b>EXT_SOURCE_2</b> (-0.160), <b>EXT_SOURCE_3</b> (-0.179), <b>EXT_SOURCE_1</b> (-0.155) — strongest negative correlations (higher scores → less likely to default)."),
            ("bullet", "<b>DAYS_BIRTH</b> (+0.078) — older applicants default less (positive because DAYS_BIRTH is negative)."),
            ("bullet", "<b>DAYS_EMPLOYED</b> (-0.045) — longer employment slightly reduces default risk."),
            ("bullet", "<b>REGION_RATING_CLIENT_W_CITY</b> (+0.060) — higher regional risk rating correlates with more defaults."),
            ("heading2", "Correlation Heatmap Insights"),
            ("bullet", "The three EXT_SOURCE features are moderately correlated with each other (0.1–0.3), suggesting they capture partly overlapping but complementary information."),
            ("bullet", "AMT_CREDIT and AMT_GOODS_PRICE are very highly correlated (~0.99), indicating potential multicollinearity — one could be dropped."),
            ("bullet", "AMT_ANNUITY is moderately correlated with both AMT_CREDIT (~0.77) and AMT_GOODS_PRICE (~0.77)."),
            ("text", "Overall, <b>most features have weak individual correlations</b> with TARGET (|r| < 0.1), which explains why ensemble methods like LightGBM and XGBoost (which capture non-linear interactions) outperform linear models on this dataset."),
        ]
    },
    {
        "title": "11. Bivariate Analysis — Scatter & Violin Plots",
        "image_count": 2,
        "body": [
            ("text", "Bivariate scatter plots (EXT_SOURCE_2 vs EXT_SOURCE_3, AMT_CREDIT vs AMT_ANNUITY) colored by TARGET provide insight into the joint feature space:"),
            ("heading2", "EXT_SOURCE_2 vs EXT_SOURCE_3"),
            ("bullet", "Defaulters (orange) cluster in the <b>lower-left quadrant</b> (low scores on both), while non-defaulters spread across higher values."),
            ("bullet", "The combined use of both scores provides better separation than either alone, motivating interaction/product features."),
            ("heading2", "AMT_CREDIT vs AMT_ANNUITY"),
            ("bullet", "A strong positive linear relationship exists (annuity ~ fixed percentage of credit). Defaulters are interspersed throughout, with no clear 2D boundary."),
            ("heading2", "Violin Plots"),
            ("text", "Violin plots for AMT_CREDIT and AMT_ANNUITY split by TARGET show nearly identical distribution shapes, confirming that these financial amounts alone are insufficient for classification and must be combined with behavioral and external features."),
        ]
    },
    {
        "title": "12. Bureau Data Analysis",
        "image_count": 1,
        "body": [
            ("text", "The bureau.csv table contains credit history information from other financial institutions. The analysis reveals:"),
            ("heading2", "Number of Bureau Credits per Applicant"),
            ("bullet", "Distribution is right-skewed — most applicants have 0–10 bureau credits, but some have 30+."),
            ("bullet", "Higher number of credits can indicate either a richer credit history or potential over-leveraging."),
            ("heading2", "Credit Types"),
            ("bullet", "The most common bureau credit types are <b>Consumer credit</b> and <b>Credit card</b>, followed by Car loan, Mortgage, and Microloan."),
            ("heading2", "Credit Status"),
            ("bullet", "<b>Closed</b> credits dominate, followed by <b>Active</b> credits. A small fraction are in <b>Bad debt</b> or <b>Sold</b> status — these are strong default indicators."),
            ("text", "Aggregating bureau features per applicant (e.g., count of active credits, count of bad debts, mean overdue days, total debt) creates powerful predictive features for the model."),
        ]
    },
    {
        "title": "13. Previous Application Analysis",
        "image_count": 1,
        "body": [
            ("text", "The previous_application.csv file contains historical application data at Home Credit:"),
            ("heading2", "Application Status Distribution"),
            ("bullet", "<b>Approved</b> applications are most common, followed by <b>Cancelled</b>, <b>Refused</b>, and <b>Unused offer</b>."),
            ("bullet", "A high proportion of cancelled or refused previous applications may signal riskier applicants."),
            ("heading2", "Previous Contract Type"),
            ("bullet", "<b>Cash loans</b> and <b>Consumer loans</b> dominate previous applications. <b>Revolving loans</b> are less common."),
            ("heading2", "Modeling Value"),
            ("text", "Previous application history captures the applicant's relationship with Home Credit over time. Features such as <b>number of previous refusals</b>, <b>ratio of approved to total applications</b>, and <b>change in credit amount over time</b> can significantly improve predictive power."),
        ]
    },
    {
        "title": "14. POS Cash & Credit Card Balance Analysis",
        "image_count": 2,
        "body": [
            ("text", "Monthly snapshots from POS/Cash loans and credit card accounts reveal payment behavior patterns:"),
            ("heading2", "POS Cash Balance"),
            ("bullet", "Payment status distribution shows the vast majority of months are <b>on-time (status 0)</b>, with a decreasing tail of delinquent statuses."),
            ("bullet", "A small but significant number of records show <b>DPD (days past due)</b> values, which when aggregated per applicant, become strong default risk indicators."),
            ("heading2", "Credit Card Balance"),
            ("bullet", "Credit utilization patterns (balance / limit ratio) vary widely. High utilization is associated with higher default risk."),
            ("bullet", "Payment patterns over time — whether the balance is increasing or decreasing — provide trend-based features."),
            ("text", "Both tables provide <b>temporal behavioral features</b> that are among the most valuable for credit risk modeling. Common aggregations include: max/mean DPD, number of late payments, credit utilization trend, and balance velocity."),
        ]
    },
    {
        "title": "15. Installments Payments Analysis",
        "image_count": 1,
        "body": [
            ("text", "The installments_payments.csv tracks actual payment behavior against scheduled payments for previous credits:"),
            ("heading2", "Key Metrics"),
            ("bullet", "Comparison of <b>AMT_PAYMENT</b> (actual) vs <b>AMT_INSTALMENT</b> (scheduled) reveals underpayment patterns."),
            ("bullet", "<b>DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT</b> difference shows early vs. late payments."),
            ("heading2", "Behavioral Insights"),
            ("bullet", "Applicants who consistently underpay or pay late on previous installments are significantly more likely to default on new loans."),
            ("bullet", "The <b>payment-to-installment ratio</b> and <b>days-late statistics</b> (mean, max, count) are among the most powerful engineered features."),
            ("text", "This table captures <b>actual repayment discipline</b>, making it arguably the most predictive supplementary data source. Features computed here regularly appear in the top 20 feature importances of competitive models."),
        ]
    },
    {
        "title": "16. Feature Skewness & Kurtosis",
        "image_count": 1,
        "body": [
            ("text", "The skewness and kurtosis analysis of all numerical features provides guidance on which transformations may be needed:"),
            ("heading2", "Skewness"),
            ("bullet", "A large proportion of features are <b>positively skewed</b> (long right tail), particularly financial amounts and count features."),
            ("bullet", "Features with |skewness| > 2 (approximately 40–50% of numerical columns) are candidates for <b>log or Box-Cox transformation</b>."),
            ("bullet", "The skewness histogram shows a concentration near zero but with a long right tail of highly skewed features."),
            ("heading2", "Kurtosis"),
            ("bullet", "Many features exhibit <b>leptokurtic</b> distributions (kurtosis >> 0), indicating heavy tails and outliers."),
            ("bullet", "Extreme kurtosis values (> 100) indicate features dominated by a few extreme values — these need capping or robust scaling."),
            ("heading2", "Practical Implications"),
            ("text", "Tree-based models (LightGBM, XGBoost) are largely invariant to skewness and don't require transformation. However, for linear models, logistic regression, or neural networks, applying <b>np.log1p()</b> to highly skewed features can improve performance. Features with extreme kurtosis should be capped at the 99th percentile to prevent outlier domination."),
        ]
    },
    {
        "title": "17. Bureau Balance Analysis",
        "image_count": 2,
        "body": [
            ("text", "The bureau_balance.csv file contains ~27.3 million monthly status records for credits reported by other financial institutions. Each record represents one month of one bureau credit."),
            ("heading2", "STATUS Distribution"),
            ("bullet", "<b>'C' (Closed)</b> is the most frequent status at ~13.6 million records, indicating many bureau credits are historical/completed."),
            ("bullet", "<b>'0' (On-time)</b> follows with ~7.5 million records, representing active credits with timely payments."),
            ("bullet", "<b>'X' (Unknown)</b> accounts for ~5.8 million records."),
            ("bullet", "<b>Delinquent statuses (1–5)</b> are extremely rare — collectively only <b>1.3%</b> of all records."),
            ("heading2", "Temporal Pattern (Stacked Area Chart)"),
            ("bullet", "Moving from older months (MONTHS_BALANCE = -100) to recent (0), the <b>'C' (Closed) share grows steadily</b> from ~20% to ~55%, indicating accumulation of closed credits."),
            ("bullet", "<b>'X' (Unknown)</b> shrinks from ~35% to ~15% over time, suggesting better status tracking for recent records."),
            ("bullet", "Delinquent statuses remain a <b>thin sliver at the top</b> throughout all periods."),
            ("heading2", "Feature Engineering Value"),
            ("text", "Key aggregated features from bureau_balance include: <b>proportion of delinquent months per credit</b>, number of status transitions (e.g., 0→1), <b>maximum delinquency depth reached</b>, and recency of last delinquent status. These behavioral features capture credit management quality over time."),
        ]
    },
    {
        "title": "18. FLAG_DOCUMENT Analysis",
        "image_count": 1,
        "body": [
            ("text", "The application data contains 20+ binary FLAG_DOCUMENT_* columns indicating which supporting documents each applicant provided. The analysis examines both <b>submission rates</b> and their <b>association with default risk</b>."),
            ("heading2", "Submission Rates"),
            ("bullet", "<b>FLAG_DOCUMENT_3</b> is overwhelmingly dominant at <b>71.0%</b> submission rate — this is likely an essential/standard document."),
            ("bullet", "<b>FLAG_DOCUMENT_6</b> (8.8%) and <b>FLAG_DOCUMENT_8</b> (8.1%) are the next most common. All other documents are submitted by <b>&lt;2%</b> of applicants."),
            ("bullet", "Several documents (FLAG_DOCUMENT_2, 4, 7, 10, 12, 17, 21) have near-zero submission rates (&lt;0.1%)."),
            ("heading2", "Default Rate Differences"),
            ("bullet", "<b>FLAG_DOCUMENT_2</b> shows a dramatic <b>+22 percentage point</b> increase in default rate when provided — but is based on extremely few applicants, making it unreliable as a standalone feature."),
            ("bullet", "<b>FLAG_DOCUMENT_21</b> and <b>FLAG_DOCUMENT_3</b> also show notable positive differences (+5–8 pp)."),
            ("bullet", "Documents like <b>FLAG_DOCUMENT_18, 6, 9</b> show <b>negative differences</b> (submitting them correlates with <b>lower</b> default rate)."),
            ("text", "Most FLAG_DOCUMENT features have such low prevalence that their default rate differences are <b>statistically unreliable</b>. In modeling, they are typically grouped or used as a count feature (total documents provided)."),
        ]
    },
    {
        "title": "19. Other DAYS_* Features Analysis",
        "image_count": 1,
        "body": [
            ("text", "Beyond DAYS_BIRTH and DAYS_EMPLOYED, three additional temporal features reveal important patterns:"),
            ("heading2", "Years Since Registration"),
            ("bullet", "KDE shows defaulters are slightly overrepresented among <b>recently registered</b> applicants (0–5 years)."),
            ("bullet", "Binned default rate generally decreases from ~9% (recent) to ~4% (35+ years), though the oldest bin spikes (~12.5%) due to small sample size."),
            ("heading2", "Years Since ID Published"),
            ("bullet", "Clear monotonic trend: default rate drops from <b>~10%</b> (ID published 0–2 years ago) to <b>~5.5%</b> (17+ years ago)."),
            ("bullet", "Recently issued IDs strongly correlate with higher risk — possibly reflecting newer/less established financial profiles."),
            ("heading2", "Years Since Phone Change"),
            ("bullet", "<b>Strongest signal</b> among the three: recent phone changes (0–1 year) show default rates of <b>~9–10%</b>, dropping to <b>~4.5%</b> for those unchanged 8+ years."),
            ("bullet", "Contact instability (frequent phone changes) is a well-known risk indicator in consumer lending."),
            ("text", "All three features confirm that <b>stability and tenure</b> — longer registration, older ID, stable contact — correlate with lower default risk. These should be included as-is or binned in the feature set."),
        ]
    },
    {
        "title": "20. Credit Bureau Inquiry Analysis",
        "image_count": 1,
        "body": [
            ("text", "Six AMT_REQ_CREDIT_BUREAU_* features capture the number of credit inquiries made within different time windows (hour, day, week, month, quarter, year):"),
            ("heading2", "Short-term Inquiries (Hour / Day / Week)"),
            ("bullet", "More inquiries in the recent past correlate with <b>higher default rates</b>, peaking at 10–12% for 2–4 inquiries."),
            ("bullet", "Active credit-seeking over short periods signals urgency or financial stress."),
            ("heading2", "Monthly Inquiries"),
            ("bullet", "An <b>inverse pattern</b>: default rate drops from ~7.8% (0 inquiries) to ~3–4% for high-inquiry applicants."),
            ("bullet", "Frequent monthly engagement may indicate financial awareness rather than distress."),
            ("heading2", "Quarterly Inquiries"),
            ("bullet", "Notable spike: applicants with <b>10+ quarterly inquiries</b> show ~50% default rate, though this group is very small."),
            ("heading2", "Annual Inquiries"),
            ("bullet", "Clear upward trend: default rate rises from <b>~1–2%</b> (0 inquiries) to <b>~8–10%</b> (5+ inquiries)."),
            ("text", "The inquiry pattern reveals a nuanced signal: <b>short-term inquiry bursts signal risk</b>, while moderate long-term activity may indicate responsible credit management. Feature engineering should capture both the raw counts and the short-to-long-term inquiry ratio."),
        ]
    },
    {
        "title": "21. Binary Flag Features (Ownership & Contact)",
        "image_count": 1,
        "body": [
            ("text", "Seven binary flag features representing ownership status and contact information were analyzed for their default rate differentials:"),
            ("heading2", "Ownership Flags"),
            ("bullet", "<b>FLAG_OWN_CAR</b>: Car owners (Y, n=104,587) show lower default rate (~7.3%) vs non-owners (N, n=202,924, ~8.5%). A <b>modest but useful signal</b>."),
            ("bullet", "<b>FLAG_OWN_REALTY</b>: Minimal difference — both groups near 8%, making realty ownership weakly discriminative."),
            ("heading2", "Contact Flags"),
            ("bullet", "<b>FLAG_MOBIL</b>: Near-universal (307,510 out of 307,511 have a mobile) — <b>essentially useless</b> as a predictor."),
            ("bullet", "<b>FLAG_WORK_PHONE</b>: Counterintuitively, having a work phone correlates with <b>higher</b> default (~9.5% vs ~7.7%). This likely reflects occupation type — blue-collar workers may have work phones but also higher default risk."),
            ("bullet", "<b>FLAG_CONT_MOBILE</b>: Nearly everyone has it (n=306,937 vs 574) — <b>no useful signal</b>."),
            ("bullet", "<b>FLAG_PHONE</b>: Having a landline phone correlates with <b>lower</b> default (~7% vs ~8.3%). Landline ownership may proxy for residential stability."),
            ("bullet", "<b>FLAG_EMAIL</b>: Having email shows slightly lower default (~7.8% vs ~8.1%) — a minor effect."),
            ("text", "Among these flags, <b>FLAG_OWN_CAR</b> and <b>FLAG_WORK_PHONE</b> offer the most discriminative power. The near-universal flags (MOBIL, CONT_MOBILE) should be excluded from modeling as they carry no information."),
        ]
    },
    {
        "title": "22. Default Rate by Binned Financial Features",
        "image_count": 2,
        "body": [
            ("text", "Continuous financial features were binned into deciles to reveal <b>non-linear relationships</b> with default risk:"),
            ("heading2", "AMT_INCOME_TOTAL"),
            ("bullet", "An <b>inverted-U pattern</b>: mid-range incomes (100K–162K) show peak default rates (~8.8–9%), while both lowest and highest income deciles have lower rates (~6%)."),
            ("bullet", "The highest earners (>270K) have the lowest default rate, confirming that wealth provides a buffer."),
            ("heading2", "AMT_CREDIT"),
            ("bullet", "Default rate rises with credit amount, peaking at <b>~10–10.5%</b> in upper-mid deciles (513K–755K), then drops for the highest amounts (~4.5%)."),
            ("bullet", "Very large loans may undergo stricter screening, explaining the lower default rate at the top."),
            ("heading2", "AMT_ANNUITY"),
            ("bullet", "<b>Monotonically increasing</b>: from ~7% (lowest annuity bin) to ~10% (highest). Higher monthly obligations directly increase repayment strain."),
            ("heading2", "AMT_GOODS_PRICE"),
            ("bullet", "Similar to AMT_CREDIT — peaks at mid-upper range (~13% for 450K–522K) then drops."),
            ("heading2", "Financial Ratios"),
            ("bullet", "<b>CREDIT_INCOME_RATIO</b>: Peaks at ratios of 3.2–4.6× income (~9–9.3%), decreasing for both lower and higher ratios. This is a strong candidate for feature engineering."),
            ("bullet", "<b>ANNUITY_INCOME_RATIO</b>: Steadily increases from ~7% (lowest) to ~9% (highest), confirming that payment burden drives default."),
            ("text", "These non-linear patterns underscore the importance of using <b>tree-based models</b> that naturally capture bin-like splits, or explicitly engineering <b>binned and ratio features</b> for linear models."),
        ]
    },
    {
        "title": "23. Summary & Key Findings",
        "image_count": 0,
        "body": [
            ("heading2", "Dataset Characteristics"),
            ("bullet", "<b>307,511 applications</b> with 122 features and severe class imbalance (8.07% default rate)."),
            ("bullet", "8 interconnected tables providing ~50 million supplementary records."),
            ("bullet", "Extensive missing data in housing-related features (40–70%)."),
            ("heading2", "Top Predictive Features"),
            ("bullet", "<b>EXT_SOURCE_1/2/3</b>: External credit scores with the strongest correlations (-0.155 to -0.179)."),
            ("bullet", "<b>DAYS_BIRTH</b>: Strong monotonic relationship — younger applicants default more."),
            ("bullet", "<b>DAYS_EMPLOYED</b>: Contains anomalous sentinel value requiring special handling."),
            ("bullet", "<b>DAYS_LAST_PHONE_CHANGE</b>: Recent phone changes signal higher risk."),
            ("heading2", "Key Insights for Feature Engineering"),
            ("bullet", "Financial ratios (credit/income, annuity/income) reveal non-linear default patterns."),
            ("bullet", "EXT_SOURCE interaction features (products, means) boost model performance."),
            ("bullet", "Bureau and installments data provide the most powerful supplementary behavioral features."),
            ("bullet", "Binary flag features have limited individual power but contribute in ensemble."),
            ("heading2", "Modeling Recommendations"),
            ("bullet", "1. Handle class imbalance via SMOTE, class weights, or stratified sampling."),
            ("bullet", "2. Engineer features from EXT_SOURCE scores (interactions, polynomial)."),
            ("bullet", "3. Replace DAYS_EMPLOYED sentinel (365,243) with NaN + binary indicator."),
            ("bullet", "4. Aggregate supplementary tables (bureau, installments, POS/CC balance) per applicant."),
            ("bullet", "5. Use AUC-ROC as primary metric; tree-based ensembles (LightGBM) as primary models."),
            ("bullet", "6. Apply log transformation to highly skewed features for linear models."),
            ("bullet", "7. Bin or interact financial ratios to capture non-linear relationships."),
        ]
    },
]


# ==============================================================================
# Build PDF
# ==============================================================================

def build_pdf():
    images, temp_dir = extract_images_from_notebook(NOTEBOOK_PATH)
    print(f"Extracted {len(images)} images from notebook")

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Home Credit Default Risk — Exploratory Data Analysis Report",
        author="EDA Analysis"
    )

    story = []

    # ---- Title Page ----
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Home Credit Default Risk", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Exploratory Data Analysis Report", subtitle_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(HRFlowable(width="60%", thickness=2, color=HexColor('#1565c0'),
                            spaceBefore=10, spaceAfter=10))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Comprehensive analysis of 307,511 loan applications across 8 datasets",
                           ParagraphStyle('CenterBody', parent=body_style, alignment=TA_CENTER,
                                          fontSize=12, textColor=HexColor('#616161'))))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("NUS — Institute for Financial Risk Management (IFRM)",
                           ParagraphStyle('CenterBody2', parent=body_style, alignment=TA_CENTER,
                                          fontSize=11, textColor=HexColor('#455a64'))))
    story.append(PageBreak())

    # ---- Table of Contents ----
    story.append(Paragraph("Table of Contents", heading1_style))
    story.append(Spacer(1, 12))
    toc_style = ParagraphStyle('TOC', parent=body_style, fontSize=11, leading=18, leftIndent=10)
    for sec in SECTIONS:
        story.append(Paragraph(sec["title"], toc_style))
    story.append(PageBreak())

    # ---- Sections ----
    img_idx = 0  # track which image we're on

    for sec in SECTIONS:
        story.append(Paragraph(sec["title"], heading1_style))
        story.append(hr())

        img_count = sec.get("image_count", 0)

        # Render body content
        for item_type, content in sec["body"]:
            if item_type == "text":
                story.append(Paragraph(content, body_style))
            elif item_type == "bullet":
                story.append(Paragraph(content, bullet_style, bulletText='•'))
            elif item_type == "heading2":
                story.append(Paragraph(content, heading2_style))
            elif item_type == "key_finding":
                story.append(Paragraph(content, key_finding_style))

        # Add images for this section
        if img_count > 0 and img_idx < len(images):
            story.append(Spacer(1, 8))
            for _ in range(img_count):
                if img_idx < len(images):
                    ci, fpath, w, h = images[img_idx]
                    add_image(story, fpath, w, h)
                    img_idx += 1

        story.append(PageBreak())

    # Build
    doc.build(story)
    print(f"\nPDF generated: {OUTPUT_PDF}")
    print(f"Total images included: {img_idx}")

    # Cleanup temp images
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    build_pdf()
