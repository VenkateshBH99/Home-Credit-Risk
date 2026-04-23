[**https://www.canva.com/design/DAHGn\_7svE8/\_b5U4y0wKDSi6JIFwrgfnw/edit**](https://www.canva.com/design/DAHGn_7svE8/_b5U4y0wKDSi6JIFwrgfnw/edit) 

**SCRIPT (39 Slides)**

1\. Slide 2-9 (introduction \+ data explanation) \- **Kaushik**  
2\. ⁠Slide 10-17 (Model arch+ EDA) \- **Venkatesh**  
3\. ⁠Slide 18- 25 ( predictive modelling+ SHAP) \- **Sandesh**  
4\. ⁠Slide 26\. \-31 (Pca \+ credit \+ ABC) \- **Samarth**  
5\. ⁠Slide 32- 39 ( Conclusion, Future, limitation, Data governance) \- **Norbert**

Slide 1:  
Good morning professors. We Sovereign Prism are a team of 5, and today we are presenting “The Invisible Applicant: Illuminating the Journey from Data Desert to Credit Worthiness.”  
Our project explores how machine learning can predict default risk for applicants with little or no credit history.  
The goal is to make lending more inclusive while still keeping it responsible.

Slide 2:  
Sovereign Prism serves customers that traditional banks often overlook because they do not have a long credit record.  
Its loans are designed to be simple, fast, and accessible, and the risk scoring system helps ensure that borrowers are assessed fairly before approval.  
The broader vision is to expand access to credit across Southeast Asia through partnerships with banks, credit agencies, and regulators.

Slide 3:  
There are three main stakeholder groups. Banking and financial partners care about profitability, customer trust, and data privacy; credit bureau companies care about ROI and integration; and regulators care about risk exposure, governance, and fairness.  
So our solution uses a privacy-by-default approach, where data use is consent-based and predictions remain auditable.

Slide 4:  
The business problem is to predict loan default using alternative behavioral data instead of only traditional data.  
This matters especially for unbanked and underbanked applicants.  
So the business goal is to improve financial inclusion without weakening credit risk management.

Slide 5:  
Our business objectives and technical objectives is to process data from 307,511 applicants across 3 pipelines which we built and compared Logistic Regression, Random Forest, XGBoost, and LightGBM on 205 features from 8 tables.  
We worked across three pipelines — Application-Only, Combined, and PCA-reduced — and evaluated them using AUC-ROC, cross-validation, and SHAP.  
We also created an interpretable A/B/C credit scoring framework for real lending decisions.

Slide 6:  
For data acquisition and cleaning, we used the Kaggle Home Credit dataset and reviewed the column types and value patterns in each table.  
We then removed columns with heavy missingness, imputed missing values, and created new features through transformation to improve model readiness.

Slide 7:  
In data quality analysis, we found 62 columns with missing values, removed sparse building-related features, and handled anomalies like the DAYSEMPLOYED value of 365243 by converting it to NaN.  
We also imputed EXT\_SOURCE\_1 using the median and used winsorization to reduce the effect of extreme financial outliers.

Slide 8:  
For the physical data model, we then merged the application table with bureau records, previous applications, POS cash balance, installment payments, and credit card balance using SK\_ID\_CURR.  
This gave us a richer borrower profile and a better risk signal than the application form alone.  
The final dataset contains both numeric and categorical features, with an 8.04 percent default rate.

Slide 9:  
Finally, the summary statistics show a strong class imbalance, so we focus on recall and AUC-ROC rather than accuracy alone.  
We also found that males aged 20 to 25 form the highest default segment, which helps identify where risk is concentrated.  
That is why recall matters most for us: it helps reduce missed risky borrowers while keeping the model useful in practice.

---

### **Slide 10 – Transition \+ Model Architecture**

“Thanks, Kaushik.

Now I’ll take you through how we structured our approach, starting with the model architecture.

As shown here, we worked with multiple data sources including application data, bureau data, and repayment behavior.

We cleaned and processed this data, created features from different tables, and then used it as input for our machine learning models.

So our focus was on building a strong data and modeling pipeline from raw data to predictive insights.

And this forms the foundation for the predictive modeling we’ll discuss next.”

---

### **Slide 11 – Key Feature Distributions**

“When we analyzed feature distributions, we noticed an important pattern.

Most financial features like income, credit amount, and annuity look very similar for defaulters and non-defaulters, so there is a lot of overlap.

But the EXT\_SOURCE features are different, where defaulters are mostly in the lower range, around 0 to 0.3.

These three features alone contribute a large part of the prediction signal.

So this tells us we cannot rely on single features, and we need models that can combine multiple signals.”

---

### **Slide 12 – External Data Analysis**

“Next, we looked at external data like bureau records and previous applications.

We saw that most customers have active or closed loans with very low overdue cases.

Also, on average, customers have multiple past applications, which gives us useful signals such as how often they borrow and how frequently they get approved.

So this data helps us understand a customer’s past financial behavior.”

---

### **Slide 13 – Internal Behavioral Data**

“We also analyzed internal behavioral data, which is basically how customers repay loans.

For example, we looked at how many payments were late, how many days they were delayed, and how consistent their payments were.

When we added these features, model performance improved by around 3 percent.

So this shows that actual repayment behavior is a strong indicator of risk.”

---

### **Slide 14 – Correlation Matrix**

“To quantify relationships, we used a correlation matrix.

We found that EXT\_SOURCE variables have the strongest negative correlation with default, which means higher scores lead to lower risk.

On the other hand, features like income and credit amount have almost no correlation.

So this confirms that default cannot be predicted using just financial values, and we need multiple signals combined.”

---

### **Slide 15 – Feature Engineering Pipeline**

“To capture these signals, we created new features by combining data from different tables.

For example, from bureau data we calculated total credit and overdue amounts. From installment data we created late payment ratios. From credit card data we calculated usage levels.

These features summarize customer behavior and make it easier for the model to learn patterns.”

---

### **Slide 16 – Traditional vs Alternative Data**

“At this point, we asked a key question.

Does adding alternative data actually improve prediction compared to traditional application data alone?

Traditional data tells us who the applicant is, such as their demographics, income, and employment.

But alternative data tells us how they behave, such as their repayment patterns, credit usage, and financial discipline.

This distinction is important because for customers with limited credit history, behavioral data can reveal signals that traditional data misses.”

---

### **Slide 17 – Feature Selection (WOE/IV)**

“Finally, before moving into modeling, we performed feature selection using Weight of Evidence and Information Value.

We found that around 70 features had meaningful predictive power above the threshold.

Most features are weak individually, but together they contribute to overall model performance.

This step ensures that we retain only relevant features, improving both model efficiency and interpretability.

now Sandesh, who will explain the predictive modeling.”





< NOTE: Sandesh didnt write a script for his slides, he spoke freely, and then we moved on to Samarth >






SAMARTH SONI 

Hello professors, now I will walk you through how we **built a fair and interpretable credit risk system for thin-file customers** 

Slide 26 — PCA Analysis – 30s

We **did due diligence to understand our feature space** using PCA.

The **scree plot shows PC1 alone covers \~17.5% of variance**,

But then 50 MORE COMPONENTS are required for just 90% coverage

 Shows NO STRONG LATENT STRUCTURE  in the data \- there is ONE dominant signal but other information is spread across many feature clusters.

We saw earlier, performance slightly dropped by using PCA \- it is **better suited for linear models**. Tree-based and ensemble models **loses their feature-level splitting granularity** when we **compress inputs into abstract components**. 

This exercise revealed the data’s structural complexity and **justified our modelling choices of using original features itself**

Slide 27 — Fairness Evaluation by Gender – 35 s

The next slide covers our analysis from **ethics perspective**.

A model that works on average but **for deployability, we want to avoid gender discrimination**. We chose:

Demographic parity \= are approval rates equal regardless of creditworthiness?

Equalized odds \= is the model equally accurate for both genders?. 

Closer to zero \= fairer. 

Random Forest wins on demographic parity , while Logistic Regression is worst. THE EXPLANATION we offer is, **linear models absorb data biases into their coefficients**, whereas **tree-based models distribute it across many splits, hence diluting** the effect of any single biased feature

Critically, all four (be it linear or ensemble) have uniform equalized odds. This means the **bias is in the dataset itself and not in the algorithm of choice**, which was also brought up by our earlier EDA analysis.

Hence, the **correct intervention is upstream — resampling, reweighting, or collecting more representative training data**. This will matter in future real deployment.”

Slide 28 — Gender Bias Analysis – 40s

The next slide **appends our fairness analysis** more concrete and granular. It shows how bias manifests per model and introduces the XNA group i.e users who dont wish to reveal their gender.

We visualized **selection rates** **by gender** **is similar** **across all models** — consistent with the equalized odds finding that this **disparity comes from the training data**, not the model choice.

Another insight comes from another dimension — the XNA group. Logistic Regression’s prediction \= exactly 0 – statistically impossible and could prove dangerous. This generalization failure happened probably due to **so few XNA training samples**

**Logistic doesnt perform, and Ensemble models revert to the base rate** & predict \~0.50 for XNA, which is the model expressing uncertainty. Not ideal, but infinitely better than zero.

Our **subtle finding** here is the **Accuracy \- Fairness tradeoff**

We saw earlier that **Logistic Regression achieves the highest AUC overall**. But when we evaluated fairness, it showed the worst demographic parity and statistically impossible results on the XNA group. 

We offer the EXPLANATION that Logistic’s single global linear boundary is aggressively **exploiting all correlations including gender-correlated signals**

Hence, it is powerfu on average but fails on underrepresented subgroups. 

Our ensemble models **sacrifice a small amount of aggregate accuracy** in exchange for more **distributed & balanced decision-making** across all customer segments. 

Since our project is designed to serve thin-file customers who are already marginalized by traditional credit systems, deploying the highest-AUC model without fairness evaluation would be exactly the wrong decision. 

 This is our argument for preferring ensemble models FOR THE FINAL PREDICTIONs: they are **more robust when handling underrepresented customer segments**.

Slide 29 — Customer Risk Scoring Validation – 50 s

THe next slide shows how we made a actionable product for our stakeholders by providing a 300–850 credit score like FICO, validated on unseen data.

We used the formula (Score \= 600 base \+ factor 20 × ln(odds)) \- which means every 20-point increase doubles the person’s creditworthiness odds. The code:
```python
# ============================================================
# 7d. SCORECARD CONVERSION (Improved — with log-odds clipping)
# Convert LR coefficients → points-based scorecard
# Score = Offset + Factor × ln(odds)
# Industry standard: 600 base, PDO=20 (20 points to double odds)
# ============================================================
 
score_min_raw, score_max_raw = scores.min(), scores.max()
scores_normalized = 300 + (scores - score_min_raw) / (score_max_raw - score_min_raw + 1e-8) * 550
```

The score distribution with red for defaulters and blue for non-defualters shows clear separation, hence proving our predictive power. The rating table gives actionable tiers from ‘very safe to lend to’ at the top to ‘very likely to default’ at the bottom. 

The graph on the bottom left shows 20x differential in default rate between the lowest and highest score bands. This is our true test of generzlisation since it is on the validation set, data the model never saw.

The overlap zone around 450–600 is where we will recommend proactive human-in-the-loop review rather than full automation.

Slide 30 — A/B/C Score Framework

The next slide shows how we designed a **complete credit customer lifecycle system** for application, monitoring, and managing collections.

**We identified features from our dataset which would be relevant for respective ABC scores**.

**A score is based on 170 features mainly driven by external EXT\_SOURCE and demographic EDUCATION / AGE.**

 **It achieves the highest AUC of 0.72**

**Our B-score is based on 34 behavioral features for portfolio montoring/early warning such as REPAYMENT PATTERNS and INSTALLMENT activity  and ACTIVE LOAN INDICATORS , alognwith some internal factors.**

 **It has a moderate AUC of about 0.62.** 

**Finally, the C-score has 13 features like OVERDUE AMOUNTS and DPD DELAY PATTERNS and  used for PRIORITIZING COLLECTIONS from delinquent customers and focuses on.**  
**It  has a lower AUC of about 0.58.**

overall, we see a clear trend where A-score is strongest for prediction, while B and C scores are more focused on monitoring and operational decisions across the customer lifecycle.

The **bottom charts show temporal behavioral signals that we would expec**  that B and C scores require. We have **average credit card balances and remaining installments over a 100-day window**. The trends are expected and **we see installments drop sharply in the days leading up to the reference point of 90 dpd whereas outstanding grows back up after 40 dpd**. 

AUC scores here are not through logistic as we are not checking recalls but through lightgbm. Because in lightGBM our AUC is higher.

Slide 31 — KS Score Metrics

**The next slide is our ffinal technical validation — the model works, here’s the proof in three independent dimensions. End on confidence.**

**KS \= 0.428 places the model in the “good” band (0.4–0.6).  – strong separation power**

**The lift chart shows the model is good for risk prioritization. The 3.74x lift in the top decile means the  top 10% of customers flagged by our model are 3.74 times more likely to default than average**

This means a collections or credit review team can focus their attention on the highest-risk decile and immediately capture disproportionate value — rather than reviewing all accounts equally

**The calibration curve shows underprediction — predicted 0.6, actual \~0.15 — which is the same miscalibration you saw across all models.**  Importantly, this miscalibration does not undermine the model’s ranking ability — the ordering of customers from risky to safe remains valid

To summarize: our model is excellent at ranking customers by risk, is directly actionable for credit approval, pricing, and collections prioritization, and has one targeted calibration improvement that would make it fully production-ready.”

Closing Transition – tie it all together in two sentences:

So now, we have shown we understands our data, have experimented with various types of models and features, and have though about scoring and real world use. Now, Norbert will summarise and conclude the discussion for us. 

**PUT THIS IN A PPT**

**SLIDE 32 \- CONCLUSION**

So, in conclusion, we really did prove that we can broaden our definition of creditworthy customers by incorporating alternative data, which allow us to assess the more underserved & unbanked population.

Now, while our Logistic Regression model have strong AUC score and high interpretability for ABC scoring, ensemble models like LightGBM and XGBoost usually performs similarly . Interestingly though, applying PCA will slightly drop performance llikely because of dimensionality loss and our tree-based models are just much more effective at identifying thresholds within unmixed, raw features.

Now, regarding those unnamed top predictors EXT\_SOURCE\_1 through 3, through our analysis, we can actually tell that

- EXT\_SOURCE\_1 is likely a private credit bureau score  given the high missing values in developing regions & they really value age since it has strong correlation with each other.  THOUGHTS: EXT_SOURCE_1 has strong correlation with age (DAYS_BIRTH), so idk.... could be employment score, pension, etc OR it could also be payment history like telco payment and utility payment which could be unintentionally correlate to age as older people have more payment history. (A 65-year-old scores nearly 3× higher than a 22-year-old on average.)
- Meanwhile, EXT\_SOURCE\_2 is almost certainly telco data as 98% of all customers have it, even with zero payment history.  THOUGHTS: EXT_SOURCE_2 is some sort of universal thing, everyone literally has it. Either alternative like telco data, or government score.
- And EXT\_SOURCE\_3 is more likely a government credit score band, its definitely a score band since it has few unique values but the strongest negative correlation with defaulters. THOUGHTS: EXT_SOURCE_3 is definitely some sort of score band due to unique values, it has the highest correlation with DPD and strongest negative correlation with defaulting (so definitely some sort of credit score).

**SLIDE 33 \- HOW CAN AI HELP IN CREDIT SCORING**

So here are also our learnings on how AI can help with credit scoring, next slide….

**SLIDE 34 \- STAKEHOLDER RECOMMENDATION**

Coming to our stakeholder recommendations:

- Regulatory teams should deploy the LightGBM model into production alongside SHAP to monitor drifting.  
- Credit Bureaus really need to prioritize enriching these external data scores, as they're huge for 'thin-file' applicants.  
- And for Business Development, future responsible borrowing campaigns could focus a bit more on the male demographics, because they showed a higher likelihood to default in our data.

**SLIDE 35 \- DATA LIMITATIONS & FAIRNESS**

Regarding data limitations and fairness: Future iterations should use more advanced sampling imputing techniques like logistic regression or even neural networks. It's also worth noting that traditional columns still outperformed alternative ones as top indicators obviously.

We also have model fairness concerns to tackle, as currently gender bias and proxy features like marriage status and car ownership status are present.

**SLIDE 36 \- DATA GOVERNANCE**

So, how will this look in production for data governance and security? We need a strict framework for things like for consensual data opt-in sharing.

For risk mitigation, we’ll rely heavily on stakeholders and credit bureaus for live data. We need periodic regulatory checks, quarterly fairness audits to adjust for dynamic real-world data shifts, and redundant data sources to prevent downtime. Finally, to address any stakeholder skepticism, we must build transparent explainability dashboards and keep a human-in-the-loop override option for critical loans.

**SLIDE 37 \- FUTURE WORK**

OK, now Looking ahead for possible future works, our next steps are

- First to test on out-of-time data for temporal validation and benchmark against existing production models.  
- We’ll also focus on more creative feature engineering since our models valued those heavily, scour for more complementary alternative data, and utilize open-source frameworks to proactively eliminate biased features.

**SLIDE 38 \- REFERENCES**

We have some links to the dataset we used & our own repo.

**SLIDE 39 \- THANK YOU**

And that’s it from our team, thank you so much professors for listening to our project presentation.