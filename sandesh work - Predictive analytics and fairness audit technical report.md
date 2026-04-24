Comprehensive Technical Report: Home Credit Default Risk Prediction

Advanced Predictive Analytics & Fairness Audit

# 1\. Project Motivation and Objectives

In modern finance, the ability to accurately distinguish between high-risk and low-risk borrowers is essential. This project utilizes the Home Credit dataset to build a sophisticated classification pipeline. The primary objective is to maximize the identification of potential defaults (Recall) while maintaining high overall precision, ensuring that the lending process remains both profitable and inclusive.

# 2\. Data Engineering & Preprocessing Pipeline

A machine learning model is only as good as the data it consumes. Our pipeline involved several critical stages:

## 2.1 Handling Missingness

Real-world financial data often contains gaps. We implemented median imputation for numerical features to mitigate the impact of outliers. For categorical features, mode imputation was used to maintain the most frequent classes.

## 2.2 Feature Scaling and Transformation

We applied StandardScaler to all numerical inputs. This transformation is vital for the Logistic Regression baseline, as it prevents features with large magnitudes (like Credit Amount) from dominating the weight updates over smaller but potentially more significant features (like Age).

# 3\. Advanced Modeling Architectures

We explored a spectrum of models, from linear baselines to complex gradient boosting machines.

## 3.1 Logistic Regression with L2 Regularization : The Interpretability Baseline

Logistic Regression was utilized to establish a benchmark. We incorporated L2 (Ridge) regularization to prevent overfitting by penalizing excessively large coefficients. To address the class imbalance, we set the 'class_weight' parameter to 'balanced', effectively increasing the cost of misclassifying a default.

- **Design Decision:** Implementation of **L2 (Ridge) Regularization** and **Feature Scaling**.
- **Rationale:** Logistic Regression is highly sensitive to the magnitude of features. Without StandardScaler, variables with larger ranges (like total loan amount) would overshadow more predictive features with smaller ranges (like age).
- **Imbalance Handling:** The class_weight='balanced' parameter was critical. It automatically adjusts the weights inversely proportional to class frequencies, ensuring the model doesn't simply predict "0" for every applicant to achieve high accuracy.

## 3.2 Random Forest Ensemble : Capturing Non-Linearity

Random Forest was chosen for its ability to handle non-linear decision boundaries through the aggregation of multiple decision trees. By bootstrapping the data and selecting random feature subsets for each split, it significantly reduces model variance.

- **Design Decision:** **Bagging (Bootstrap Aggregating)** and **Feature Subsampling**.
- **Rationale:** The dataset contains hundreds of features, many of which are correlated. Random Forest restricts each split in a tree to a random subset of features. This "decorrelates" the trees, ensuring the ensemble is more robust and less prone to overfitting than a single complex decision tree.
- **Tuning Focus:** Our Grid Search focused on max_depth to prevent individual trees from growing too complex and memorizing noise in the training data.

## 3.3 Extreme Gradient Boosting (XGBoost): The Predictive Powerhouse

Boosting algorithms represent the state-of-the-art for tabular data. XGBoost builds trees sequentially, with each tree minimizing the gradient of the loss function relative to the previous iteration's errors. We utilized the 'scale_pos_weight' hyperparameter to specifically tune the model for the minority 'Default' class.

- **Design Decision:** **Gradient Boosting with Sparse-Aware Splitting** and **Scale_Pos_Weight**.
- **Rationale:** XGBoost was designed to handle missing values automatically (sparse-aware), which is common in credit applications. The decision to use scale_pos_weight (calculated as the ratio of negative to positive samples) was the most significant factor in its high **Recall**.
- **Regularization:** Unlike many other boosting implementations, XGBoost includes built-in L1 and L2 regularization in its objective function, which was essential to stabilize the model given the high number of encoded categorical variables.

## 3.4 LightGBM : Efficiency in High Dimensions

Chosen for its "Leaf-wise" growth strategy, which is highly effective at identifying complex patterns in high-dimensional tabular data.

- **Design Decision:** **Leaf-wise (Best-first) Tree Growth**.
- **Rationale:** Traditional boosting models grow level-wise. LightGBM's leaf-wise strategy chooses the leaf that results in the greatest loss reduction. This typically leads to lower loss and higher accuracy than level-wise growth on large tabular datasets like Home Credit.
- **Efficiency:** It was specifically chosen for its memory efficiency and speed, which was necessary when iterating through the massive feature space during hyperparameter tuning

# 4\. Ethical Fairness and Bias Mitigation

The analysis utilized two primary metrics to quantify potential bias:

- **Selection Rate (Demographic Parity):** This measures the percentage of each group that is predicted to "Default." If one gender has a significantly higher selection rate than another, the model may be using proxy variables that unfairly penalize that group.
- **Recall Disparity (Equal Opportunity):** This checks if the model is equally good at catching actual defaults in both groups. A lower recall for one group means the model is "missing" defaults for them, which could lead to riskier lending in that specific segment.

**Findings from the Analysis**

The fairness audit revealed several critical insights into how the models handled sensitive attributes:

- **Baseline Disparity:** The unweighted Logistic Regression model showed the highest disparity. Because it focuses on overall accuracy, it tended to favor the majority group (which has more data), leading to unequal treatment of the minority group.
- **The Impact of scale_pos_weight:** In the XGBoost model, the use of scale_pos_weight served a dual purpose. By forcing the model to pay more attention to the "Default" class, it reduced the model's tendency to rely on easy-to-identify demographic trends, forcing it to look deeper into individual financial indicators.
- **Selection Rate Comparison:** The visualizations in the report show the selection rates across genders. While slight differences exist, the gradient boosting models achieved a more balanced distribution of risk assignment compared to the baseline.

**Mitigation Strategies**

To achieve the levels of fairness observed in the final report, several design decisions were made:

- **Feature Masking:** Sensitive attributes (like Gender) were monitored but not used as primary drivers for the prediction. The model was trained to find patterns in credit history and debt-to-income ratios that are independent of gender.
- **Threshold Optimization:** Rather than using a standard 0.5 probability cutoff for all groups, we evaluated how shifting the decision threshold could equalize the "Selection Rate" without significantly compromising the AUC-ROC.
- **Group-Specific Evaluation:** We did not just look at the global accuracy. We looked at the **MetricFrame** results to ensure that performance was consistent. If a model has 80% accuracy globally but only 60% for a specific subgroup, that model is considered unfair.

**Why This Matters for the Report**

Including this fairness context in your report demonstrates **Responsible AI** practices. It shows that the model is not a "black box" that might be discriminating against applicants, but a transparent tool that has been audited for equity. This is a standard requirement for models used in regulated industries like banking and insurance.