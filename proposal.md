# EBA5008: Graduate Certificate in Intelligent Financial Risk Management

## The Invisible Applicant: Illuminating the Journey from Data Desert to Credit Worthiness

**A Fairness-Aware Risk Engine and Gateway for Thin-File and Credit-Invisible Borrowers**

Date of Proposal

22 March 2026

Team Name

Sovereign Prism

Team Members

Kaushik Radhakrishnan Musuwathi – A0327229B

Venkatesh Basavaraj Honnaksaturi – A0329909M

Samarth Soni – A0329960U 

Norbert Oliver – A0328685M

Sandesh Sreepathy Upadhyaya – A0327834X  

## 1. **Introduction**

No File? No Problem\! At Sovereign Prism, we aim to be the vanguard for Invisible Applicants:  broaden inclusive lending for underserved borrowers without a safety net, by leveraging alternative data and baking in fairness and accuracy into our compliance-first algorithms. In this way, we aim to Crack the Credit Gap for The Invisible Applicant.

Traditional credit scoring relies heavily on bureau data from agencies such as TransUnion, Equifax, and Experian. While effective for consumers with established credit histories, this approach systematically excludes approximately 26 million Americans who are "credit invisible" and 19 million more with "thin files" \- groups that disproportionately include minority communities, younger consumers, and recent immigrants.

Non-traditional data sources \- including digital footprints, phone usage patterns, social network risk signals, identity verification depth, and external scores; offer the potential to score previously unscorable consumers. However, their adoption raises critical questions about algorithmic fairness, and model interpretability.

This project investigates whether alternative data features can improve credit default prediction accuracy; particularly for thin-file applicants; while maintaining algorithmic fairness across gender and socioeconomic groups. The aim is to design and develop a Risk Modelling tool for the home credit sector.

## 2. **Business Problem & Objectives**

### Business Problem

Credit scorecards are mathematical formulae that assign points to applicant characteristics to predict future default behaviour. Traditional scorecards favour closeness (5–15 interpretable variables); however, millions of creditworthy consumers are excluded due to insufficient bureau data. Additionally, as models incorporate richer feature sets, the risk of proxy discrimination increases \- features that appear neutral (e	.g., phone stability) may correlate with protected attributes such as age, gender, or socioeconomic status.

### Business Objectives

This project builds a credit risk model that responsibly expands access to formal credit for applicants who are invisible to traditional bureau-based scoring. Leveraging all seven Home Credit relational tables \- linking application data with bureau records, payment histories, and behavioural signals \- we aim to achieve three outcomes: a 3–5 percentage point AUC-ROC improvement over a bureau-only baseline, reliable risk ranking for 60–80% of applicants who would otherwise go unassessed, and fairness metrics within defensible bounds (disparate impact ratios ≥ 0.80, equalized odds differences \< 0.05) so that lending decisions remain explainable and compliant.

### Technical Objectives

The project constructs an end-to-end pipeline from raw ingestion of all seven Home Credit tables through feature engineering (\~250 applicant-level features), model training, fairness assessment and monitoring. LightGBM serves as the primary model with XGBoost as comparator, evaluated across four controlled experiments: traditional-only baseline, traditional-plus-alternative integration, Fairlearn-constrained optimisation targeting gender/age/income fairness, and thin-file subgroup analysis. Model monitoring covering PSI, AUC/Gini drift, and fairness stability.

### Business Impact & Value

The modelling framework enhances risk differentiation for thin-file and credit-invisible applicants, expanding credit access. Ongoing monitoring maintains model governance and performance stability over time. The approach delivers measurable improvements in portfolio risk management, capital efficiency, and operational compliance.  
Ultimately, Sovereign Prism uses fairness-aware AI to reveal the true creditworthiness hidden within 'invisible' financial profiles. We don't just box applicants, we illuminate their potential, just like a prism that reveals the hidden colors within a seemingly singular beam of light.

## 3. **Project Scope and Design**

### Dataset

The primary dataset is the Home Credit Default Risk competition dataset from Kaggle, comprising real-world consumer lending data with 7 relational tables linked by applicant and previous application identifiers.

### Dataset Structure and Data Fields

| Table | Rows | Columns | Content |
| :---- | :---- | :---- | :---- |
| application\_train.csv | 307,511 | 122 | Applicant demographics, loan terms, TARGET label |
| application\_test.csv | 48,744 | 121 | Holdout evaluation set |
| bureau.csv | 1,716,428 | 17 | External credit bureau records |
| bureau\_balance.csv | 27,299,925 | 3 | Monthly bureau status history |
| previous\_application.csv | 1,670,214 | 37 | Prior Home Credit applications |
| POS\_CASH\_balance.csv | 10,001,358 | 8 | POS/cash loan balances |
| credit\_card\_balance.csv | 3,840,312 | 23 | Credit card account snapshots |
| installments\_payments.csv | 13,605,401 | 8 | Instalment payment records |

### description from Kaggle:
- application_{train|test}.csv
  - This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
  - Static data for all applications. One row represents one loan in our data sample.
- bureau.csv
  - All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
  - For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
- bureau_balance.csv
  - Monthly balances of previous credits in Credit Bureau.
  - This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
- POS_CASH_balance.csv
  - Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
  - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
- credit_card_balance.csv
  - Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
  - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credit card) rows.
- previous_application.csv
  - All previous applications for Home Credit loans of clients who have loans in our sample.
  - There is one row for each previous application related to loans in our data sample.
- installments_payments.csv
  - Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
  - There is a) one row for every payment that was made plus b) one row each for missed payment.
  - One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
- HomeCredit_columns_description.csv
  - This file contains descriptions for the columns in the various data files.
- Credit scores we already have in the dataset:
  - A-Score (origination) — application + bureau features for approve/decline
  - B-Score (behavioral) — POS, installment, CC payment history for portfolio monitoring
  - C-Score (collection) — DPD/overdue patterns for delinquent account prioritisation
  - Bureau Score — isolated external credit bureau signal

![][image1]

### **How Alternative Data Flows Through the Pipeline**
Here's the concrete picture — how each Home Credit feature group acts as alternative data and what it actually does in the model:

| Alt-Data Proxy | Real-World Equivalent | What It Captures | How It Helps Thin-File Applicants |
|-----------|-------------------|-------|--------------------------------------|
| EXT_SOURCE_1/2/3 | LexisNexis RiskView, Equifax IDA scores | External model scores from non-bureau sources | Provides a risk signal even with zero bureau records — EXT_SOURCE_2 alone has ~0.16 correlation with TARGET |
| Phone flags (FLAG_MOBIL, FLAG_CONT_MOBILE, FLAG_EMP_PHONE, DAYS_LAST_PHONE_CHANGE) | Telco/digital footprint data | Device stability — someone who keeps the same phone number for years signals stability | Credit-invisible applicants still have phones; this gives a scorable signal where bureau data is empty |
| Social circle (OBS/DEF_30/60_CNT_SOCIAL_CIRCLE) | Social network risk scoring | How many of your known contacts defaulted in the last 30/60 days | "Birds of a feather" — even without personal credit history, peer default patterns are predictive |
| Document flags (FLAG_DOCUMENT_2–21) | KYC/identity verification depth | How many identity documents the applicant submitted | Willingness to verify identity correlates with repayment intent; more docs = lower risk |
| Housing/property (40+ APARTMENTS, FLOORS, LIVINGAREA, YEARS_BUILD columns) | LexisNexis property records, utility data | Asset proxies — property quality, age, size | Thin-file borrowers with stable housing are lower risk than the bureau-only model can detect |
| Regional risk (REGION_RATING_CLIENT, REGION_POPULATION_RELATIVE) | Area-level socioeconomic data | Geographic risk concentration | Adds context for applicants in low-risk regions who lack personal credit history |
| Credit card behavior (credit_card_balance table) | Open banking / transaction data | Utilization patterns, minimum payment behavior, cash advances | Behavioral signals that complement traditional DPD-based bureau scores |


**The Key Insight**

Traditional features only work when the applicant already has bureau records. For the ~1,700 credit-invisible and ~50,000 thin-file applicants in the dataset:
Exp 1 (Traditional only): Model has almost nothing to score them on → high uncertainty, generic PD assignment
Exp 2 (+ Alt Data): EXT_SOURCE scores, phone stability, social circle defaults, document depth, and housing quality provide independent risk signals that don't require prior credit history
Exp 4 (Thin-file subgroup): This is where you prove the value — the AUC lift from alt data should be 5–8 pp for thin-file vs only 1–2 pp for thick-file, demonstrating that alternative data specifically helps the underserved segment
The fairness angle (Exp 3) then asks: does this inclusion come at the cost of discrimination? For example, phone stability correlates with age (younger people change phones more), and housing quality correlates with income — so Fairlearn checks whether the alt-data model inadvertently widens demographic gaps.
 

1. Observation Period (Historical Data)
•- Duration: Typically 3–12 months (commonly 6 months)
•- Purpose: Collect customer behavioral data to create model features.
•- Examples: credit utilization, payment history, delinquency, spending patterns.
 
2. Scoring Month (Snapshot Date)
•- The month when the behavior score is calculated.
•- All variables are computed using data up to this month.
 
3. Performance Period (Future Outcome Window)
•- Duration: Typically 6–12 months.
•- Used to determine whether the customer becomes Good or Bad.
•- Example BAD definition: 90+ days past due, default, or charge-off.
 
Example Timeline:
•Observation (Jan–Jun) → Score Month (Jul) → Performance (Aug–Jan)
 
Multiple monthly scoring snapshots are usually used to increase sample size and reduce seasonal effects in model development.

### Our research into Recall vs precision for this problem
 
Metric 	Focus	Importance in Credit
Recall	Identifying all actual defaults (Minimizing False Negatives)	Highest (Reduces financial losses)
Precision	Ensuring predicted defaults are actually defaults (Minimizing False Positives)	Secondary (Reduces unnecessary rejection of good customers)

#### Intuition on precision vs recall
In this scenario, your model’s job is to catch the "bad" cases (Default = 1). If you flip the labels so that "Not Defaulted" (0) becomes the positive class (1), your metrics shift from measuring risk detection to measuring reliability detection.
Here is how the "effect" changes when you flip the labels:
1. Precision: "How sure are we?"
Original (Default = 1): Precision tells you, "Of all the people we predicted would default, how many actually did?"High precision here means you aren't falsely accusing good borrowers of being risky.
Flipped (No Default = 1): Precision now tells you, "Of all the people we predicted would pay us back, how many actually did?" This is the "Safe Bet" metric. High precision here means your "green-lit" loans are almost guaranteed to be safe.
2. Recall: "How many did we catch?"
Original (Default = 1): Recall tells you, "Of everyone who actually defaulted, how many did we catch beforehand?" This is your Loss Prevention metric. High recall means you missed very few "bad" loans.
Flipped (No Default = 1): Recall now tells you, "Of all the people who are actually good borrowers, how many did we successfully identify?" This is your Opportunity metric. High recall here means you aren't accidentally turning away good customers.
Why they don't just "swap"
Imagine a bank that is extremely conservative. They reject almost everyone.
For Defaults (1): Their Precision might be 100% (the few they labeled as defaulters definitely defaulted), but their Recall is low (they missed many other defaulters by just rejecting everyone).
If you flip it (Safe = 1): Now, their Recall for "Safe" borrowers is terrible (they rejected most good people), and their Precision for "Safe" borrowers is 100% (the few they let through were definitely safe).
The Key Difference:
In banking, Precision for Default (1) protects your reputation and customer relationships, while Recall for Default (1)protects your bottom line (capital). When you flip the labels, you are no longer measuring "risk"—you are measuring "success."

To map these intuitively, you just need to identify your "Point of Interest" (the event you are looking for). In your case, that is the Default (1).
Once you fix the "Point of Interest," you can map every metric to a simple "Real World" philosophy using this mental grid:
1. The "Mental Map" of Errors
Stop thinking of $P$ and $N$ as abstract letters. Think of them as The Model's Verdict:
True Positive (TP): You caught a Defaulter. (Success)
False Positive (FP): You accused a Good Payer. (The "False Alarm")
False Negative (FN): A Defaulter snuck through. (The "Silent Killer")
True Negative (TN): You correctly identified a Good Payer. (Routine)
2. The Three Pillars (The Cheat Sheet)
Here is how to instantly translate the math into the business result:
A. Loss Prevention = Recall ($TP / [TP + FN]$)
The Philosophy: "Don't let anyone slip through."
The Math: You are looking at all Actual Defaulters ($TP + FN$). How many did you successfully catch ($TP$)?
The Risk: If this is low, your FN (Silent Killers) are high. You are losing money because people are defaulting and you didn't see it coming.
B. Safe Bet on defaulting = Precision ($TP / [TP + FP]$)
The Philosophy: "When I say 'Default,' I mean it."
The Math: You are looking at your Predictions ($TP + FP$). How many of your "calls" were right ($TP$)?
The Risk: If this is low, your FP (False Alarms) are high. You are annoying good customers by flagging them as risks when they aren't.
C. Opportunity Cost = Recall of the Flipped Class ($TN / [TN + FP]$)
The Philosophy: "How many good customers did I throw away?"
The Math: Look at all Actual Good Payers ($TN + FP$). How many did you correctly keep ($TN$)?
The Risk: If this is low, your FP is high relative to your good customers. You are leaving money on the table by rejecting people who would have paid you back.
D. Safe Bet on credit worthiness = Precision of flipped classes ($TN / [TN + FN]$)
The Philosophy: "When I say 'Creditworthy,' I mean it."
The Math: You are looking at your negative Predictions ($TN + FN$). How many of your "calls" were right ($TN$)?
The Risk: If this is low, your FN (False Green signal) are high. You are risking your banks wallet by flagging people as safe bets when they aren't.
3. How to Interchange Them Instantly
To keep the formulas interchangeable in your head, use the "Denominator Rule":
If the denominator is "Actuals" ($TP + FN$): You are measuring Coverage/Recall. (Did I find all the bad guys?)
If the denominator is "Predictions" ($TP + FP$): You are measuring Reliability/Precision. (Can I trust my alarm?)
Quick Summary for your Brain:
Low Recall (Default): You are bleeding cash (Loss Prevention failure).
Low Precision (Default): You are offending customers (Safe Bet failure).
Low Recall (Non-Default): You are shrinking your market (Opportunity Cost failure).
Low Precision (Non Default): You are giving loans to defaulters (Safe bet failure in a different sense)

#### Flipping the target variable DOES NOT mean interchanging precision and Recall
However, the following "Horizontal vs. Vertical"distinction can help us visualize why the math prevents a simple swap

This is the final "aha!" moment of binary metrics. They are paired, but they aren't swaps; they are reflections.
Here is why they are "two sides of the same coin" and why the "swap" theory is a trap.
1. The Pairings: The "Reality" vs. The "Verdict"
Each pair looks at the exact same error, but from a different perspective.
Pair 1: The "Silent Killer" (False Negatives / FN)
These are the people you thought were Safe (0) but who actually Defaulted (1).
A. Recall (Loss Prevention): Measures this from the perspective of Reality. "Of all people who actually defaulted, how many did I miss?" ($TP / [TP + FN]$)
D. NPV (Safe Bet - Creditworthy): Measures this from the perspective of Your Verdict. "Of all the people I labeled safe, how many were actually bad?" ($TN / [TN + FN]$)
The Coin: The "FN" coin. One side is your coverage of the bad guys (A); the other is the reliability of your "Safe" label (D).
Pair 2: The "False Alarm" (False Positives / FP)
These are the people you thought would Default (1) but who were actually Safe (0).
B. Precision (Safe Bet - Risk): Measures this from the perspective of Your Verdict. "When I say 'Risk,' can I be trusted?" ($TP / [TP + FP]$)
C. Specificity / Recall of 0s (Opportunity): Measures this from the perspective of Reality. "Of all the good customers out there, how many did I accidentally insult?" ($TN / [TN + FP]$)
The Coin: The "FP" coin. One side is the reliability of your "Risk" label (B); the other is your coverage of the good guys (C).
2. Why it’s WRONG to say they simply "swap"
If you flip your 0s and 1s, Precision (B) does not become Recall (A).
The Math Reason:
Precision is a "Vertical" look at the Confusion Matrix (columns). It’s about your Predictions.
Recall is a "Horizontal" look at the Confusion Matrix (rows). It’s about Actual Reality.
Flipping the labels moves you from the top row to the bottom row, but it does not rotate your head from horizontal to vertical.
The Philosophy Reason:
Recall is always about exhaustion (Did I find all of them?).
Precision is always about purity (Is my bucket clean?).
When you flip the labels, you are still measuring exhaustion (New Recall) and purity (New Precision), but you are just looking at a different bucket (the 0s instead of the 1s).
Summary Table: 
![ReflectionSummaryTable](/Users/SamarthSoni/Projects/Home-Credit-Risk/docs/ReflectionSummaryTable.png)

#### Type of mistake intuition
To keep the formulas interchangeable, think of the "Type of Mistake" that ruins the metric:
A and D are ruined by the same mistake (FN), while B and C are ruined by the same mistake (FP).
Here is the breakdown of how these "mistakes" infect your metrics:
The "Silent Killer" Pair: Ruined by False Negatives (FN)
The mistake: You predicted "Safe (0)" but they actually "Defaulted (1)."
A. Recall (Loss Prevention):
How it's ruined: Every time a defaulter slips through as a FN, your "catch rate" drops. You didn't find all the bad guys.
The Math: $TP / (TP + \mathbf{FN})$
D. "Safe Bet" Creditworthiness (NPV):
How it's ruined: Every time a defaulter slips through as a FN, your "Green Light" becomes less trustworthy. Your "Safe" bucket is now contaminated.
The Math: $TN / (TN + \mathbf{FN})$
The shared pain: Both of these metrics drop when your model is too "lenient" and lets bad borrowers pass.
The "False Alarm" Pair: Ruined by False Positives (FP)
The mistake: You predicted "Default (1)" but they were actually "Safe (0)."
B. "Safe Bet" Risk (Precision):
How it's ruined: Every time you cry wolf on a good borrower (FP), your "Risk" label loses credibility. You are accusing innocent people.
The Math: $TP / (TP + \mathbf{FP})$
C. Opportunity (Specificity):
How it's ruined: Every time you flag a good borrower as a risk (FP), you lose a customer. You are failing to capture the "Total Opportunity" of the market.
The Math: $TN / (TN + \mathbf{FP})$
The shared pain: Both of these metrics drop when your model is too "strict" and flags good borrowers as risks.
The "Interchangeable" Logic Summary
![InterchangeableLogicSummary](/Users/SamarthSoni/Projects/Home-Credit-Risk/docs/InterchangeableLogicSummaryTable.jpeg)
The Error
Metrics Ruined
Business Consequence
False Negative (FN)
A (Recall) & D (NPV)
Direct Loss: You lose the principal of the loan.
False Positive (FP)
B (Precision) & C (Specificity)

Opportunity Loss: You lose interest income and hurt your reputation.

#### How to set a threshold (e.g., 0.3 instead of 0.5) to favor one pair over the other…
To favor one pair over the other, you move the Classification Threshold. By default, most models use 0.5 (if the probability of default is > 50%, label it 1).
When you change this number, you are essentially adjusting the "sensitivity" of your alarm.
1. If you want to protect A & D (The "Lender's Shield")
Goal: Minimize False Negatives (FN). You want to catch every single potential defaulter.
Action: Lower the threshold (e.g., to 0.2 or 0.1).
The Logic: "If there is even a 20% chance this person defaults, I’m labeling them a 1 (Risk)."
Result:
A (Recall) goes UP (You caught almost everyone).
D (NPV) goes UP (Your 'Safe' pile is incredibly pure).
The Cost: Your B and C will crash because you’ll have tons of False Alarms (FP). You’ll be rejecting perfectly good people just to be safe.
2. If you want to protect B & C (The "Market Grower")
Goal: Minimize False Positives (FP). You want to stop insulting good customers.
Action: Raise the threshold (e.g., to 0.8 or 0.9).
The Logic: "I will only label someone a 1 (Risk) if I am 90% certain they will fail."
Result:
B (Precision) goes UP (When you call someone a risk, you’re almost always right).
C (Specificity/Opportunity) goes UP (You are green-lighting almost every good customer).
The Cost: Your A and D will crash because your Silent Killers (FN) will skyrocket. Defaulters will stroll right through your 90% filter.
The "Trade-off" Summary
![TradeoffSummary](/Users/SamarthSoni/Projects/Home-Credit-Risk/docs/TradeoffSummary.jpeg)
Strategy
Threshold
Focus
Priority Metrics
Trade-off (The Pain)
Conservative
Low (0.1)
Safety First
A & D (Protect Capital)
Losing good customers (B & Cdrop)
Aggressive
High (0.9)
Growth First
B & C (Capture Market)
Higher default rates (A & D drop)
How to choose?
In banking, this is usually a Dollar Value calculation.
If a default (FN) costs you $10,000 in lost principal...
But a missed customer (FP) only costs you $500 in lost interest...
...then a Low Threshold is mathematically mandatory to protect A and D.
Now, let’s see how to visualize this trade-off using a Precision-Recall Curve or an ROC Curve
![PRvROC](/Users/SamarthSoni/Projects/Home-Credit-Risk/docs/PRvROC.jpeg)
Visualising these curves helps you see exactly where to "set the dial" for your bank’s risk tolerance.
1. The Precision-Recall (PR) Curve
This curve is the best tool for imbalanced data (like loans, where most people pay back). [1, 2]
The Trade-off: As you push for higher Recall (A) to catch more defaulters, your Precision (B) will inevitably drop because you start flagging good people as risks.
Goal: You want your model to stay as close to the top-right corner as possible (high A and high B).
Action: If you are a Conservative Bank, you move to the right along this curve (accepting lower precision to ensure high recall). [3, 4, 5]
2. The ROC Curve
This curve shows the trade-off between Recall (A) and the False Positive Rate (1 - C). [6, 7]
The "Diagonal" of Doom: The dashed diagonal line represents a model that is no better than flipping a coin.
The Curve: A good model "bows" toward the top-left corner. This corner represents the perfect bank: catching 100% of defaulters (A = 1) while insulting 0% of good customers (C = 1). [6, 8, 9, 10]
Choosing Your Spot
To pick the best threshold, you look at the Area Under the Curve (AUC). [8, 11]
A High PR AUC means your bank is great at identifying the rare "bad" borrowers without ruining the experience for the many "good" ones.
A High ROC AUC means your model is generally excellent at separating the two groups across any possible strategy. [12, 13, 14]
Now let’s see how to calculate the Profit/Loss optimal threshold by plugging in actual dollar values for $FP$ and $FN$
To find the most profitable threshold, you don't look at the model in isolation; you look at the Expected Value of your decisions. In banking, the goal is to maximize the net revenue by balancing the interest earned against the capital lost to defaults. [1, 2, 3]
The Profit Formula
You calculate the total profit for every possible threshold (from 0 to 1) and pick the one that results in the highest number. The simplified profit formula for a loan portfolio is: [4, 5]
$$Total\ Profit = (TN \times Gain_{interest}) - (FN \times Cost_{default}) - (FP \times Cost_{admin})$$
TN (True Negatives): Good borrowers you approved. You earn the interest.
FN (False Negatives): Defaulters you approved. You lose the loan principal.
FP (False Positives): Good borrowers you rejected. You lose the marketing/admin cost spent to acquire them. [6, 7, 8]
Step-by-Step Calculation
Assign Real Dollar Values:
$Gain_{interest}$: e.g., $+\$1,000$ (Profit from a successful loan).
$Cost_{default}$: e.g., $-\$10,000$ (Loss from a single default).
$Cost_{admin}$: e.g., $-\$500$ (Acquisition cost for a rejected customer).
Generate Predictions at Different Thresholds:
Run your model's probability scores through a range of thresholds (e.g., $0.1, 0.2, \dots, 0.9$). For each threshold, count your $TN$, $FN$, and $FP$.
Find the Peak:
Plug those counts into the profit formula. You will see a "hump" in the data where profit is maximized. [1, 4, 6, 9, 10]
The Shortcut: The "Cost-Ratio" Rule
If your model is well-calibrated, you can estimate the optimal threshold ($\tau$) using the ratio of your costs: [11, 12]
$$\tau = \frac{Cost_{FP}}{Cost_{FP} + Cost_{FN}}$$
In our example, where a default ($FN$) is 20 times more expensive than a false alarm ($FP$):
$$\tau = \frac{500}{500 + 10,000} \approx 0.047$$
This suggests you should be extremely conservative, flagging anyone with even a ~5% chance of default as a risk to protect your capital. [11]
Would you like to try plugging in your own bank's specific numbers to see where your "sweet spot" threshold would land?
```python
import numpy as np
def calculate_profit(threshold, y_true, y_probs, gain_tp, cost_fp, cost_fn, gain_tn):
   y_pred = (y_probs >= threshold).astype(int)
   tp = np.sum((y_true == 1) & (y_pred == 1))
   fp = np.sum((y_true == 0) & (y_pred == 1))
   fn = np.sum((y_true == 1) & (y_pred == 0))
   tn = np.sum((y_true == 0) & (y_pred == 0))
# Profit = (Benefits of being right) - (Costs of being wrong)
# In banking, TP is 'prevented loss', but usually we focus on the net:
# Profit = (Interest from TN) - (Principal lost from FN) - (Admin cost of FP)
   total_profit = (tn * gain_tn) - (fn * cost_fn) - (fp * cost_fp)
   return total_profit
# Simulation
np.random.seed(42)
y_true = np.random.binomial(1, 0.1, 1000) # 10% default rate
y_probs = np.where(y_true == 1, np.random.beta(5, 2, 1000), np.random.beta(2, 5, 1000))
# Business Values
gain_tn = 1000   # Interest earned from a good loan
cost_fn = 10000  # Principal lost from a default
cost_fp = 500    # Admin/Marketing cost of a rejected good customer
thresholds = np.linspace(0, 1, 100)
profits = [calculate_profit(t, y_true, y_probs, 0, cost_fp, cost_fn, gain_tn) for t in thresholds]
best_threshold = thresholds[np.argmax(profits)]
max_profit = max(profits)
print(f"{best_threshold=}")
print(f"{max_profit=}")
```


# **4\. Methodology & Architecture**

The project follows a structured end-to-end pipeline from raw data ingestion through to model deployment, with fairness evaluation embedded at every decision point.

| Stage | Component | Output |
| :---- | :---- | :---- |
|  Data Ingestion & Exclusions | Load 7 relational CSVs; apply scoring/performance exclusions; G/B/I waterfall classification | Clean training set |
| Feature Engineering | Aggregate bureau/payment/balance tables to applicant level; classify Traditional vs Alternative features | Feature matrix (\~250   features) |
| Dimension Reduction | PCA / Factor Analysis (Varimax rotation) to identify latent financial dimensions; select 1–2 raw variables per factor | Shortlisted variables   (\~40–60) |
| Feature Selection | WoE binning \+ Information Value scoring; VIF multicollinearity check | Final feature set (15–30   variables) |
| Model Training | LightGBM primary; XGBoost & Logistic Regression as comparisons; stratified 5-fold CV and other models | 4 trained A-Score variants   \+ B/C-Scores |
| Fairness Evaluation | Fairlearn MetricFrame on gender, age, income groups; compute DI ratios, equalized odds | Fairness metrics per group |
| Fairness Mitigation | Threshold Optimizer (post-processing); Exponentiated Gradient (in-processing) | Fairness-constrained model |
| Explainability | SHAP TreeExplainer; LIME local explanations (all models)  | Feature importance, LIME explanations, reason   codes |
| Model Monitoring | PSI, AUC stability, Gini   drift across time-based validation splits | Monitoring dashboard   metrics   |

## 4. **Key Deliverables**

​​EDA & G/B/I Analysis: The project will begin with a thorough exploratory analysis of all seven relational tables, producing visualisations of class imbalance (8.1% default rate), missing value distributions (50+ columns with \>30% missing), and demographic breakdowns. A waterfall classification will be applied to derive Good, Bad, and Indeterminate labels from bureau\_balance STATUS codes, with the indeterminate band excluded from model training to sharpen Good/Bad separation.

Feature Engineering Pipeline: Modular Python implementation generating applicant-level feature matrix (307,511×250) with explicit traditional/alternative partitioning (\~120 traditional, \~130 alternative features), thin-file segmentation logic (1,700 credit-invisible \+ 50,000 thin-file applicants), and comprehensive documentation of transformations, imputation strategies (50+ columns \>30% missing), and outlier handling.

Model Artifacts and Experimental Framework: 8 serialized LightGBM/XGBoost models across four experiments, stratified 5-fold CV performance tables (target: baseline AUC 0.74-0.76 → full AUC 0.78-0.80, Gini uplift 6-10pp), ROC/PR curve visualizations, feature importance rankings.

Credit Score & Behavioral Score: Two scoring models built for different stages of lending. The Credit Score (A-Score) evaluates applicants at the point of application using their personal details, loan information, and external credit history to decide whether to approve or decline. The Behavioral Score (B-Score) tracks how borrowers actually pay after receiving a loan \- using their repayment patterns across POS, installment, and credit card accounts \- to flag early signs of financial difficulty before missed payments occur. Together, these scores help lenders make better decisions both when granting a loan and while managing it.

Fairness Assessment Suite: Fairlearn MetricFrame reports across gender×age×income intersections (3×5×5 combinations), mitigation analysis (pre-/in-/post-processing) comparing baseline vs. constrained models (DI≥0.80, equalized odds \<0.05, max AUC degradation \<1pp), accuracy-fairness Pareto frontiers.

Explainability Toolkit: SHAP TreeExplainer implementation generating portfolio-level feature importance (top-20 features) and applicant-specific adverse action reason codes for 10,000-sample validation cohort. LIME (Local Interpretable Model-agnostic Explanations) applied to all models (LightGBM, XGBoost, Logistic Regression) providing individual-level prediction explanations with local feature contributions visualised in a comparative 3×2 grid.

Model Monitoring Framework: Module computing PSI (\<0.10), AUC/Gini/KS drift, fairness stability (∆DI\<0.02) across 4 time-based validation splits (quarterly windows via DAYS\_REGISTRATION), configurable green/amber/red alerting thresholds for 15 protected group combinations.

## 5. **Effort Estimates and Timeline**

(skipping)

## **Gantt Chart (Estimation Timeline)**

(skipping)

## 6. **References** 

   • 	CFPB (2022). Data Point: Credit Invisibles. Consumer Financial Protection Bureau.

   • 	Bird, S., et al. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft.

   • 	Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. NeurIPS.

   • 	Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.

   • 	Siddiqi, N. (2012). Credit Risk Scorecards. Wiley.

   • 	Kaggle (2018). Home Credit Default Risk Competition. https://www.kaggle.com/c/home-credit-default-risk

[image1]: /Users/SamarthSoni/Projects/Home-Credit-Risk/Table_Relations_home_credit.png
