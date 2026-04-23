---

# **4\. Model Architecture and Exploratory Data Analysis**

## **4.1 System Architecture**

The credit risk modelling framework is designed as a multi-layered pipeline that integrates heterogeneous data sources into a unified analytical system capable of capturing both static and dynamic aspects of borrower behavior. The architecture is built upon multiple relational datasets, including application-level data, bureau records, previous application histories, and behavioral datasets such as POS cash balances, credit card balances, and installment payments. These datasets are interconnected through primary and foreign keys such as `SK_ID_CURR`, `SK_ID_PREV`, and `SK_ID_BUREAU`, enabling hierarchical linkage across different levels of granularity.

The architecture fundamentally transforms raw transactional data into structured applicant-level features through a series of processing stages. Initially, raw data is ingested and validated, followed by preprocessing steps that address missing values, inconsistencies, and anomalies. Subsequently, feature engineering aggregates time-series and relational data into meaningful summary statistics that reflect borrower behavior. This is followed by feature selection to reduce dimensionality and eliminate redundant or weak predictors before constructing model-ready datasets.

![][image1]

**Figure 1\. Data Relationship and Architecture**  
*(Source: EDA Notebook, Page 1\)*

The significance of this architecture lies in its ability to bridge the gap between traditional credit scoring systems, which rely primarily on static application data, and modern data-driven approaches that incorporate behavioral and transactional signals. By integrating multiple data sources, the architecture enhances the richness of the feature space and enables the model to capture complex interactions between borrower characteristics and repayment behavior. This is particularly important in the context of Home Credit, where many applicants have limited formal credit histories, making alternative data essential for accurate risk assessment.

---

## **4.2 Target Variable Analysis and Class Imbalance**

The target variable represents the binary outcome of loan repayment, where a value of 1 indicates default and 0 indicates non-default. A detailed examination of the target distribution reveals a pronounced class imbalance, with only approximately 8.07% of observations belonging to the default class. This results in a highly skewed dataset, where non-default cases significantly outnumber default cases by a ratio of roughly 11:1.

![][image2]

**Figure 2\. Target Distribution**  
*(Source: EDA Notebook, Page 3\)*

This imbalance has critical implications for both modelling and evaluation. In such scenarios, standard accuracy metrics become unreliable, as a model can achieve high accuracy simply by predicting the majority class. Consequently, evaluation metrics such as Area Under the ROC Curve (AUC), Kolmogorov-Smirnov (KS), and recall become more appropriate, as they focus on the model’s ability to distinguish between classes rather than its overall correctness. Furthermore, the imbalance necessitates the use of techniques such as stratified sampling, class weighting, or resampling methods to ensure that the minority class is adequately represented during training. From a business perspective, the imbalance reflects real-world credit portfolios, where defaults are relatively rare but financially significant, emphasizing the importance of accurately identifying high-risk borrowers.

---

## **4.3 Missing Values Analysis**

The dataset exhibits substantial missingness, with 67 out of 122 features containing missing values and 49 features having more than 40% missing data.

![][image3]

**Figure 3\. Missing Value Distribution**  
*(Source: EDA Notebook, Page 4\)*

A deeper analysis reveals that missing values are not randomly distributed but are concentrated in specific groups of variables, particularly those related to external scores and bureau information. This suggests that missingness may itself carry predictive information. For instance, the absence of external score data could indicate a lack of credit history, which is often associated with higher risk. Therefore, treating missing values purely as noise may lead to the loss of valuable information.

To address this, a combination of strategies is required. Numerical features are imputed using median values to preserve distributional properties, while categorical features are imputed using mode values. In cases where missingness exceeds a certain threshold, features may be removed to avoid introducing noise into the model. Additionally, binary indicators representing missingness can be introduced to explicitly capture the presence or absence of information. This approach allows the model to learn patterns associated with missing data, thereby improving predictive performance.

---

## **4.4 Categorical Feature Analysis**

Categorical variables play a crucial role in understanding borrower characteristics and their relationship with default risk. The analysis of these variables involves examining both their distribution across the dataset and their associated default rates.

![][image4]

![][image5]

**Figure 4\. Categorical Feature Distributions and Default Rates**  
*(Source: EDA Notebook, Pages 5–6)*

The results indicate that certain categorical variables exhibit strong associations with default behavior. For example, cash loans constitute the majority of applications and are associated with higher default rates compared to revolving loans. Gender-based analysis shows that male applicants have a higher default rate than female applicants, suggesting potential differences in financial behavior or risk exposure. Similarly, education level is inversely related to default risk, with lower education levels corresponding to higher default probabilities.

Occupational categories further highlight disparities in risk, with certain professions such as laborers and drivers exhibiting higher default rates. However, it is important to consider the sample size of each category, as categories with limited observations may produce unstable estimates. Overall, these findings demonstrate that categorical variables capture important socioeconomic dimensions of credit risk and should be carefully encoded and incorporated into the modelling process.

---

## **4.5 Numerical Feature Analysis**

### **4.5.1 Financial Variables**

The analysis of numerical financial variables reveals that features such as income, credit amount, annuity, and goods price exhibit substantial overlap between defaulters and non-defaulters.

**Figure 5\. Distribution of Financial Features by Target**  
*(Source: EDA Notebook, Page 8\)*

This overlap indicates that these variables, when considered individually, lack strong discriminatory power. For example, both defaulters and non-defaulters may have similar income levels or loan amounts, making it difficult to distinguish between them using simple thresholds. This observation highlights the limitations of traditional rule-based credit scoring systems and underscores the need for models that can capture interactions between multiple variables.

### **4.5.2 Age Analysis**

The analysis of age, represented by the `DAYS_BIRTH` variable, reveals a clear relationship with default risk. Younger applicants tend to have higher default rates, while older applicants exhibit lower risk.

**Figure 6\. Age Distribution and Default Rate**  
*(Source: EDA Notebook, Page 8\)*

This trend can be attributed to factors such as financial stability, income consistency, and credit experience, which generally improve with age. Consequently, age serves as an important predictor of creditworthiness.

---

## **4.6 Data Quality and Anomaly Detection**

A notable anomaly is observed in the `DAYS_EMPLOYED` variable, where a value of 365243 appears in approximately 18% of the records.

![][image6]

**Figure 7\. Employment Duration Before and After Cleaning**  
*(Source: EDA Notebook, Page 9\)*

This value corresponds to an unrealistic employment duration and likely represents missing or placeholder data. Interestingly, the anomalous group exhibits a different default rate compared to the rest of the dataset, suggesting that it may contain implicit information. To address this issue, the anomalous values are either replaced with missing values or treated as a separate category. This ensures that the anomaly does not distort the model while still preserving any potential predictive signal.

---

## **4.7 External Risk Scores**

The external score variables (`EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3`) are identified as the most influential predictors of default risk. These variables exhibit the strongest negative correlation with the target variable, indicating that higher scores are associated with lower default probability.

![][image7]

**Figure 8\. External Score Distributions**  
*(Source: EDA Notebook, Page 10\)*

The distribution plots show a clear separation between defaulters and non-defaulters, with defaulters concentrated in the lower score ranges. This strong predictive power suggests that external scoring systems effectively capture borrower creditworthiness and should be given significant weight in the modelling process. However, reliance on these variables alone may not be sufficient, particularly for applicants with limited credit history, highlighting the need for complementary data sources.

---

## **4.8 Outlier Analysis**

The presence of outliers in financial variables such as income and credit amount is evident from boxplot visualizations.

![][image8]

**Figure 9\. Boxplots of Financial Features**  
*(Source: EDA Notebook, Page 10\)*

Outliers can distort statistical measures and adversely affect model performance, particularly for linear models that are sensitive to extreme values. However, tree-based models are inherently more robust to outliers, as they rely on split-based decision rules rather than distance-based calculations. Nevertheless, appropriate preprocessing techniques such as clipping or transformation may still be applied to reduce the impact of extreme values.

---

## **4.9 Correlation Analysis**

The correlation analysis provides insights into the relationships between features and the target variable.

![][image9]

**Figure 10\. Top Correlated Features with Target**

**![][image10]**  
**Figure 11\. Correlation Heatmap**  
*(Source: EDA Notebook, Pages 11–12)*

The results indicate that external score variables have the strongest correlation with default, while traditional financial variables exhibit weak relationships. This suggests that default risk is influenced by a combination of factors rather than any single variable. The correlation matrix also highlights potential multicollinearity among features, which must be addressed during feature selection to avoid redundancy.

---

## **4.10 Bivariate Analysis**

Bivariate analysis using scatter plots reveals the relationships between pairs of financial variables.

**![][image11]**

![][image12]

**Figure 12\. Scatter Plots of Financial Relationships**  
*(Source: EDA Notebook, Page 12\)*

The plots show significant overlap between defaulters and non-defaulters, indicating that class separation is not linear. This reinforces the need for non-linear models capable of capturing complex interactions.

---

## **4.11 External Bureau Data Analysis**

The bureau dataset provides a comprehensive view of applicants’ historical credit activity across external institutions.

![][image13]

**Figure 13\. Bureau Data Distributions**  
*(Source: EDA Notebook, Page 14\)*

The analysis reveals that applicants typically have multiple credit records, with most credits classified as active or closed. Overdue cases are relatively rare, suggesting that the majority of borrowers maintain regular repayment behavior. This dataset captures important aspects of credit exposure and financial obligations, making it a valuable source of predictive information.

---

## **4.12 Previous Application Analysis**

The previous application dataset captures historical borrowing patterns, including the number of prior applications, approval rates, and product types.

![][image14]

**Figure 14\. Previous Application Statistics**  
*(Source: EDA Notebook, Pages 17–18)*

The analysis shows that most applicants have multiple prior applications, with a high approval rate. This information provides insights into borrower behavior and credit demand, contributing additional predictive signal beyond the current application.

---

## **4.13 Internal Behavioral Data Analysis**

In addition to external bureau sources, internal behavioral tables were analyzed to capture how borrowers actually repay loans after origination. These included `POS_CASH_balance`, `credit_card_balance`, and `installments_payments`. Together, they provide a richer behavioral picture of repayment discipline, utilization, and timing. After aggregation, these internal tables generated 85 applicant-level features and improved prediction accuracy by about 3.1 percent when combined with application data.

![][image15]

The `POS_CASH_balance` table contains roughly 10 million records and covers about 94.1 percent of applicants. It tracks monthly installment behavior on point-of-sale and cash loans. The analysis found that nearly all payments were on time, with overdue cases making up only about 0.3 percent of records. This suggests that POS repayment histories are mostly stable, but when aggregated they still provide meaningful variation across borrowers.

![][image16]

The `credit_card_balance` table contains around 3.84 million records but covers only 28.3 percent of applicants. This lower coverage makes it more selective, but it still contributes useful information on card utilization and repayment behavior. The data showed moderate utilization levels, around 30 to 50 percent, and overdue cases were rare at roughly 0.2 percent.

![][image17]

The `installments_payments` table contains around 13.6 million records and covers about 94.8 percent of applicants. This table records each individual repayment installment and is particularly useful for late-payment behavior. The analysis found that about 5 to 10 percent of payments were late, with an average delay of roughly 5 to 10 days. These patterns are more directly linked to repayment reliability and therefore represent especially valuable behavioral indicators for risk modelling. 

---

## **4.14 Feature Engineering Pipeline**

After understanding the raw and aggregated variables, the project moved to feature engineering. This step transformed relational tables into borrower-level predictors suitable for modelling. The process aggregated each supplementary table by applicant and derived summary statistics that capture application patterns, debt exposure, delinquency, spending behavior, and repayment consistency.

For `bureau` and `bureau_balance`, engineered variables included aggregated credit counts, active loan ratios, mean days past due, total credit amounts, and measures of debt exposure across institutions. For `previous_application`, features captured approval and rejection rates, contract types, requested versus actual credit amounts, decision latency, and application frequency. For `installments_payments`, the engineered features included late-payment ratios, average delay days, payment-to-installment discrepancies, repayment consistency, and monthly delinquency frequency. Finally, for `POS_CASH_balance` and `credit_card_balance`, the analysis derived utilization ratios, balance trends, and broader behavioral spending and repayment indicators.

This step was essential because the original Home Credit dataset is highly relational. The model cannot directly use monthly bureau records or installment transactions in their raw form. Aggregation converts these detailed records into interpretable applicant-level summaries, which are much more suitable for machine learning. The project proposal similarly describes this step as a major stage in the analytical pipeline, producing a feature matrix of roughly 250 applicant-level features before later reduction and selection steps.


## **4.15 Traditional Versus Alternative Data Strategy**

A major analytical question in the project was whether alternative data adds meaningful predictive value beyond traditional application data. To address this, two data strategies were compared. The first was a **Traditional Data (Application-Only)** approach, which used only the `application_train.csv` table. This represents the type of static information typically available at the moment of loan application, such as demographics, income, employment, and external scores. The second was a **Traditional \+ Alternative Data (Combined)** approach, which integrated all interlinked tables and captured both static characteristics and dynamic behavioral patterns.

The traditional strategy included approximately 228 features and captured who the applicant is at the time of application. However, it lacked repayment history, cross-product exposure, and payment pattern signals. The combined strategy used about 205 aggregated features derived from more than 55 million supplementary records and captured how the applicant behaves with credit over time. The slide highlights that bureau, POS, credit card, and installment data can reveal patterns that are invisible in static application data alone.

This comparison is central to the project’s business motivation. Since Home Credit serves thin-file and underbanked borrowers, relying only on traditional application variables may exclude potentially creditworthy customers whose repayment behavior can be inferred from alternative signals. The proposal document makes the same point by emphasizing that alternative data can improve risk assessment for credit-invisible borrowers while also raising important fairness and interpretability questions.

## **4.8 WOE/IV Feature Selection**

Before moving into predictive modelling, the project applied a feature selection stage using **Weight of Evidence (WOE)** and **Information Value (IV)**. This approach is common in credit scoring because it helps quantify the predictive strength of variables while maintaining interpretability. The IV scale shown in the presentation classifies variables with IV below 0.02 as useless, 0.02 to 0.1 as weak, 0.1 to 0.3 as medium, and 0.3 to 0.5 as strong.

The distribution of IV values indicated that most features were weak individually: 93 variables fell below 0.02, 65 were weak, 3 were medium, and 2 were strong. Even so, approximately 70 features exceeded the threshold of IV \> 0.02 and were retained, balancing predictive signal with model parsimony. This is an important result because it shows that although only a small number of features are strong on their own, a meaningful set of moderately informative variables can still contribute significantly when combined in a model.

The proposal document also identifies WOE binning and IV scoring as part of the formal methodology, paired with multicollinearity checks, to reduce the feature set from a large applicant-level matrix to a manageable and interpretable set of predictive variables. This makes the WOE/IV stage the final bridge between exploratory analysis and predictive modelling. 

![][image18]

![][image19]

[image1]: docs/venkyreportimages/target_distribution_default_vs_non_default.png
[image2]: docs/venkyreportimages/correlation_heatmap_of_macro_features.png
[image3]: docs/venkyreportimages/distribution_of_external_source_1.png
[image4]: docs/venkyreportimages/distribution_of_external_source_2.png
[image5]: docs/venkyreportimages/distribution_of_external_source_3.png
[image6]: docs/venkyreportimages/loan_type_distribution.png
[image7]: docs/venkyreportimages/education_level_vs_default_risk.png
[image8]: docs/venkyreportimages/family_status_and_risk_variance.png
[image9]: docs/venkyreportimages/occupation_type_analysis.png
[image10]: docs/venkyreportimages/feature_importance_baseline_model.png
[image11]: docs/venkyreportimages/weight_of_evidence_woe_plot_1.png
[image12]: docs/venkyreportimages/weight_of_evidence_woe_plot_2.png
[image13]: docs/venkyreportimages/pca_component_variance_ratio.png
[image14]: docs/venkyreportimages/shap_summary_plot_risk_drivers.png
[image15]: docs/venkyreportimages/calibration_curve_predicted_vs_actual.png
[image16]: docs/venkyreportimages/roc_curve_champion_model.png
[image17]: docs/venkyreportimages/precision_recall_curve.png
[image18]: docs/venkyreportimages/fairness_metric_demographic_parity.png
[image19]: docs/venkyreportimages/fairness_metric_equalized_odds.png
