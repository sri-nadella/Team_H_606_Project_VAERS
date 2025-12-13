 VaxShield: Predicting Serious Adverse Events in VAERS 
 
 ABSTRACT

VaxShield is a data science project that analyzes Vaccine Adverse Event Reporting System (VAERS) data to assess and predict the risk of adverse events (AEs) associated with vaccines. The project focuses on four key vaccines: COVID-19, Varicella-Zoster (VARZOS), Influenza (FLU), and Pneumococcal vaccine polyvalent (PPV). Using machine learning techniques, VaxShield aims to develop a predictive model for serious AEs based on patient demographics, symptoms, and health history. The project's goal is to create a tool that helps users and healthcare professionals assess vaccine safety and make informed decisions.

I. INTRODUCTION

I.1 The description of the problem and the data

Vaccine safety monitoring is a critical component of public health, especially during large-scale immunization efforts. The Vaccine Adverse Event Reporting System (VAERS), co-managed by the CDC and FDA, collects post-vaccination adverse event (AE) reports from healthcare providers, manufacturers, and the public. While most reported AEs are mild, a small portion are classified as serious (e.g., death, life-threatening events, hospitalization or prolonged hospitalization, disability, congenital anomalies). Because VAERS is large and includes both structured fields and extensive free-text narratives, manually reviewing and triaging reports at scale is difficult. This motivates machine learning methods that can support rapid identification of reports more likely to be serious.

The goal of this project is to predict whether a VAERS report is Serious or Non-Serious and to deploy the final models in an interactive system called VaxShield. The dataset used spans VAERS reports from 2015–2025 and is composed of three primary files released annually: VAERSDATA (demographics, outcomes, narrative text), VAERSVAX (vaccine information such as type and manufacturer), and VAERSSYMPTOMS (up to five symptoms coded in MedDRA terminology). After merging these files on VAERS_ID and applying preprocessing, the combined dataset contained approximately 2.76 million records and 55 attributes, including numeric variables (e.g., age), categorical variables (e.g., sex, manufacturer, state), outcome indicators, and multiple free-text fields (e.g., symptom narrative and medical history).

A key modelling challenge is class imbalance: serious outcomes are relatively rare compared to non-serious reports. In addition, reporting patterns differ across vaccine types. Based on reporting volume and analysis goals, we focused on four major vaccine groups: COVID-19, Influenza (FLU), Varicella-Zoster (VARZOS), and Pneumococcal (PPV). This project follows a full data science lifecycle: Phase 1 performed EDA to understand the data and identify challenges; Phase 2 developed baseline ML models; and Phase 3 implemented per-vaccine CatBoost models and deployed them through VaxShield.

I.2 Background & Literature Review/Survey

VAERS has been widely used for post-marketing vaccine safety surveillance and has also been explored in data science research for trend analysis and predictive modeling. Prior work commonly applies traditional machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting) with feature engineering on structured fields and vectorization of text narratives (e.g., TF-IDF). A consistent theme in the literature is the difficulty of modeling rare serious outcomes due to class imbalance, missingness, and high variability in free-text symptom descriptions. Additionally, approaches that train a single global model across all vaccines can be dominated by high-volume vaccine categories, potentially reducing performance for vaccines with smaller sample sizes but distinct symptom profiles.

Our Phase 2 experiments mirrored this baseline landscape by implementing Logistic Regression, Random Forest, and XGBoost models using TF-IDF-based text representations combined with structured features. These baselines provided a performance reference and highlighted practical limitations in handling heterogeneous VAERS inputs and imbalanced outcomes. To address these issues in Phase 3, we adopted CatBoost, which is well-suited for mixed feature types and can incorporate text features more directly. We also trained separate models per vaccine type (COVID, FLU, VARZOS, PPV) to better capture vaccine-specific patterns and reduce domination by the largest category. Finally, we operationalized the models via VaxShield, enabling interactive validation through a dashboard and a chatbot that accepts natural-language descriptions and returns seriousness predictions.
     
      II. METHODS

II.1 Description of Data Processing and Feature Engineering Employed

Data sources and integration. We used the public VAERS data released in yearly files and constructed an analysis-ready dataset by integrating the three core components: VAERSDATA (demographics, outcomes, narrative fields), VAERSVAX (vaccine details), and VAERSSYMPTOMS (up to five MedDRA-coded symptoms). Records were merged using the shared identifier (VAERS_ID). We then restricted modeling to four high-volume vaccine groups COVID, FLU, VARZOS, and PPV to ensure adequate sample sizes per group and to support vaccine-specific modeling.

Cleaning and standardization. We removed obvious duplicates after merging, standardized column types (e.g., parsing dates where applicable), and handled missingness across structured and text fields. For text fields, missing values were replaced with empty strings. For structured features (e.g., age, sex, state, manufacturer), missing values were handled consistently either via explicit “Unknown” categories (categoricals) or imputation strategies suitable for the model (numeric).

Target label definition (Serious vs Non-Serious). The prediction task is binary classification. A report is labeled Serious when any serious outcome indicator is present (e.g., death, life-threatening, hospitalization/prolonged hospitalization, disability, congenital anomaly). All remaining reports are labeled Non-Serious. This creates a naturally imbalanced target, since serious outcomes are a minority of VAERS submissions.

Feature set construction (structured + text). VAERS contains mixed data types, so we engineered features in two groups:
•	Structured features: demographics and report metadata (e.g., age, sex, state) and vaccine/report descriptors (e.g., vaccine type/manufacturer when used).
•	Text features: multiple free-text fields capturing clinical context (e.g., symptom narrative and related medical history fields).

For baseline models, we represented text using TF-IDF vectorization. Because multiple text fields were used, we implemented a ColumnTransformer pipeline where each text column was independently transformed by a TfidfVectorizer, then concatenated with structured features (passed through as-is or scaled as needed). This design preserved signal from different narrative sources rather than collapsing everything into a single text column.

II.2 Description of Analyses Employed (Models / Parameters / Splits / Evaluation)

Train/test split. For each vaccine group, we created a hold-out evaluation split using 70% training and 30% testing, with stratification to preserve the serious/non-serious ratio in both sets. This ensured that performance reflects generalization to unseen reports while maintaining the minority class in the test set.
Baseline modeling (Phase 2). We trained traditional ML baselines using the TF-IDF + structured-feature pipeline to establish reference performance and identify limitations of standard approaches on heterogeneous VAERS data. The baseline family included:
•	Logistic Regression (linear classifier over the TF-IDF + structured feature space),
•	Random Forest (nonlinear ensemble baseline), and
•	XGBoost (gradient boosting baseline).
These baselines were used primarily to understand how text representation, mixed feature types, and imbalance affect minority-class detection (Serious).

Final modeling (Phase 3: CatBoost). We selected CatBoost for the final approach because it is well-suited for real-world tabular data with mixed feature types and can handle categorical variables and missingness effectively. Instead of training a single global model, we trained four separate CatBoost classifiers one for each vaccine type (COVID, FLU, VARZOS, PPV). This per-vaccine strategy reduces distribution mixing across vaccines (different demographics and symptom profiles) and supports clearer interpretation of results by vaccine context.

Imbalance-aware evaluation. Because serious events are rarer but clinically important, we evaluated models using a set of metrics appropriate for imbalanced classification, including Precision, Recall, F1-score, ROC-AUC, PR-AUC, and MCC, in addition to overall Accuracy. Metrics were computed on the held-out test set for each vaccine-specific model. We emphasized PR-AUC and Recall as key indicators of performance on the minority class while monitoring Precision to avoid excessive false positives.

II.3 System Implementation: VaxShield Dashboard and Model-Validation Chatbot

Application overview. To operationalize and demonstrate the models, we implemented VaxShield, a Streamlit-based application that provides: (1) a model performance dashboard summarizing evaluation metrics per vaccine model, and (2) a chatbot interface that accepts free-text descriptions of post-vaccination experiences and returns a seriousness prediction.
Entity extraction and routing. The chatbot performs lightweight information extraction from user input to identify key fields required for prediction most importantly vaccine type, age, sex, and symptoms and then routes the request to the appropriate vaccine-specific model. To improve user trust and reduce confusing outputs, extracted entities are displayed back to the user in a transparency panel (e.g., detected vaccine, detected/estimated age, detected sex, and detected symptoms).

Prediction presentation and trust safeguards. The system outputs a probability score for seriousness and maps it to user-friendly severity messaging (e.g., “low seriousness” vs “high seriousness”), along with clear guidance that the output is decision-support and not medical diagnosis. We also designed the UI to avoid misleading specificity when the input is vague (e.g., showing “age > 50” when the text only indicates “elderly” rather than outputting an exact age). This helps maintain consistency between what the user wrote and what the system displays, improving interpretability and trust.

This section summarizes the outcomes of our data processing, exploratory analysis, baseline modeling (Phase 2), final modeling (Phase 3), and deployment validation through the VaxShield application. All reported results are computed on held-out test data using the same target definition (Serious vs Non-Serious) established in the Methods section, and metrics are reported per vaccine group to reflect vaccine-specific data distributions and clinical reporting patterns.

Data preparation outcomes. After merging VAERSDATA, VAERSVAX, and VAERSSYMPTOMS across 2015–2025 and filtering to our four focus vaccine groups (COVID, FLU, VARZOS, PPV), we obtained a large, heterogeneous dataset containing both structured variables (e.g., demographics, outcomes, vaccine metadata) and unstructured clinical narratives (e.g., symptom text and history). The resulting feature space includes numeric, categorical, and multiple free-text fields, requiring mixed-feature modeling.

Phase 1 (EDA) key findings. Exploratory analysis showed that report volume is highly concentrated among a small number of vaccines, motivating our decision to focus on COVID, FLU, VARZOS, and PPV. We also observed strong class imbalance: serious outcomes represent a minority of all reports, meaning accuracy alone can be misleading and minority-class metrics (Recall, F1, PR-AUC) are critical. Additionally, demographic and symptom patterns differed substantially by vaccine type (e.g., age distributions and symptom frequencies), providing a strong justification for vaccine-specific modeling rather than a single global classifier.


Phase 2 baseline model performance. In Phase 2, we trained baseline classifiers (Logistic Regression, Random Forest, and XGBoost) using a pipeline that combined TF-IDF representations of multiple text fields with structured features. These baselines established reference performance and highlighted the limitations of standard workflows on VAERS: while overall accuracy was often reasonable, minority-class detection (Serious) was more challenging, reflecting both class imbalance and high variability in free-text reporting. We summarize baseline results per vaccine using metrics appropriate for imbalanced classification (Precision, Recall, F1, ROC-AUC, PR-AUC, and MCC).

Phase 3 CatBoost results (final models). Based on Phase 2 limitations, we adopted CatBoost and trained four separate models—one per vaccine group—to better capture vaccine-specific distributions and reduce cross-vaccine dilution (especially from high-volume COVID reports). Across all four vaccines, CatBoost provided stronger imbalance-aware performance, particularly improving metrics that reflect Serious-case detection (e.g., Recall, F1, PR-AUC) while maintaining high overall discrimination (ROC-AUC).

Cross-vaccine comparison and combined predictions. To support unified analysis and visualization, we consolidated outputs from the four vaccine-specific models into a combined predictions dataset (e.g., combined_vaccine_predictions.csv). This enabled consistent comparison of performance behavior and error patterns across vaccine groups, and it served as an input source for dashboard reporting in the deployed system.

Deployment validation via VaxShield. We integrated the final per-vaccine CatBoost models into VaxShield, a Streamlit-based dashboard and chatbot designed for real-time model validation. The dashboard presents per-vaccine performance summaries and enables quick comparison across models. The chatbot accepts free-text adverse event descriptions, extracts key entities (e.g., vaccine type, age category, sex, symptoms), routes the query to the correct vaccine-specific model, and returns a seriousness probability with user-friendly interpretation. To improve trust and consistency, the interface avoids misleading specificity when the input is ambiguous (e.g., displaying “age > 50” rather than an exact age when the user text only implies an older adult).

Discussion of Key Findings

This project demonstrates that seriousness prediction from VAERS reports is feasible when models can handle heterogeneous inputs (structured demographics/outcomes and multiple free-text fields) and when evaluation emphasizes imbalance-aware performance. Across the four vaccine groups (COVID, FLU, VARZOS, PPV), the final CatBoost models achieved strong discriminative performance, with ROC-AUC values near 0.97–1.00 and consistently high PR-AUC values (Table 3.3). These results indicate that the models can separate serious from non-serious reports substantially better than chance even under class imbalance, and that performance is not solely driven by accuracy. In particular, the COVID and FLU models achieved a favorable trade off between precision and recall for the Serious class, supporting the goal of identifying higher-risk reports while limiting excessive false positives.

A major design choice was training separate models per vaccine instead of one global model. This approach is justified by Phase-1 findings showing that demographics, symptom patterns, and serious-outcome prevalence differ by vaccine category. Vaccine-specific training reduces distribution mixing and prevents high-volume vaccines (e.g., COVID) from dominating the learned decision boundary. The per-vaccine strategy also improves interpretability: model performance and failure modes can be discussed in the context of a specific vaccine population rather than averaged across heterogeneous groups.

Interpretation of Differences Across Vaccines
Model behavior varied across vaccine groups in a way that is consistent with dataset composition. COVID and FLU had high performance with strong PR-AUC and solid recall, reflecting both larger sample sizes and richer signal in reported narratives. VARZOS exhibited comparatively lower recall than COVID/FLU, which may be explained by smaller effective sample size, different age distributions, and more ambiguous symptom text that overlaps with non-serious patterns. PPV achieved near-perfect metrics on the held out test set, which is unusually high for real world clinical text classification. While this may reflect strong separability in PPV report patterns or label composition, it also raises the possibility of optimistic evaluation due to sampling effects, potential information leakage through highly predictive fields, or temporal/reporting artifacts. Therefore, PPV performance should be interpreted cautiously and validated through additional stress testing (e.g., time-based splits and feature ablation).

Limitations

This work has several limitations. First, VAERS is a passive surveillance system and does not establish causality; predictions reflect patterns in reports, not confirmed vaccine-caused outcomes. Second, the dataset contains missing values, inconsistent free text reporting, and potential duplicate or correlated reports, which can introduce noise. Third, seriousness labels depend on reported outcome fields and the IME mapping strategy, which may introduce labelling bias if outcomes are under-reported or inconsistently marked. Fourth, our evaluation used a hold-out split; while appropriate for initial validation, additional validation strategies (e.g., time-based splitting) would better test real world generalization across reporting periods.

Conclusion

In conclusion, we developed an end-to-end VAERS seriousness prediction system and deployed it as VaxShield, an interactive Streamlit dashboard and chatbot for real-time model validation. The final CatBoost models achieved strong imbalance-aware performance across four vaccine groups, demonstrating that mixed structured + narrative VAERS data can support reliable seriousness classification. The project contributes a practical, vaccine-specific modelling framework and a deployment that improves transparency by showing extracted entities and model outputs in a user friendly format. Overall, VaxShield provides a proof-of-concept decision-support tool that can help summarize and triage large-scale adverse event reports, while acknowledging the limitations of passive surveillance data.


FUTURE WORK

Several extensions can strengthen both the scientific validity and real-world utility of VaxShield. First, we will evaluate time-based validation (training on earlier years and testing on later years) to better reflect real deployment conditions and to reduce the risk of optimistic performance due to temporal overlap. Second, we will perform feature ablation and leakage checks by systematically removing highly predictive outcome-related fields and re-evaluating performance, ensuring the models rely on meaningful clinical signals rather than proxies that may not be available at inference. Third, because this is an imbalanced classification problem, we will apply probability calibration (e.g., Platt scaling or isotonic regression) and optimize decision thresholds per vaccine to meet targeted operating points (e.g., higher recall for serious-event screening) while monitoring false positive burden.

From a modeling perspective, we will explore cost-sensitive learning and more explicit imbalance handling (class weights, focal loss variants, or re-sampling strategies) and compare them with the current CatBoost approach. We will also investigate stronger NLP methods for narrative fields, such as contextual embeddings (e.g., clinical BERT-style encoders) and symptom normalization to controlled vocabularies, which may improve robustness to misspellings, abbreviations, and ambiguous symptom phrasing. Finally, on the system side, we plan to add enhanced explainability (e.g., SHAP-based feature contributions for structured and text features), richer error analysis dashboards (false negative review panels), and a continuous update workflow for incorporating new VAERS releases, enabling VaxShield to remain current and more reliable as data evolves.



 


 





