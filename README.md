Credit Scoring Business Understanding
Basel II Accord and Model Interpretability

The Basel II Capital Accord emphasizes accurate measurement, monitoring, and management of credit risk to ensure that financial institutions hold sufficient capital against potential losses. In this context, our credit risk model must be transparent, interpretable, and well-documented, as model outputs may influence lending decisions, capital allocation, and regulatory reporting. An interpretable model allows stakeholders—such as risk managers, auditors, and regulators—to understand how input variables affect credit decisions, assess model stability, and validate that the model aligns with regulatory expectations. Poorly documented or opaque models increase model risk and can lead to regulatory non-compliance or mispricing of credit risk.

Proxy Default Variable and Associated Business Risks

The dataset does not contain an explicit loan default label, which makes it impossible to directly train a traditional supervised credit risk model. To address this, we create a proxy target variable based on customer behavioral patterns, specifically Recency, Frequency, and Monetary (RFM) metrics. Customers who are disengaged—characterized by low transaction frequency, low monetary value, and long inactivity—are labeled as high-risk.

While this proxy enables model development, it introduces business risks. The proxy may not perfectly reflect true default behavior, leading to misclassification risk. Some customers labeled as high-risk may actually repay loans, while others labeled as low-risk may default. This can result in lost revenue opportunities, higher default rates, or unfair credit decisions. Therefore, the proxy definition must be carefully justified, monitored, and refined as more outcome data becomes available.

Trade-offs Between Interpretable and Complex Models

In a regulated financial environment, there is a fundamental trade-off between model interpretability and predictive performance. Simple models, such as Logistic Regression combined with Weight of Evidence (WoE), offer high interpretability, stability, and ease of explanation. These models are easier to validate, audit, and communicate to regulators, making them well-suited for compliance-driven use cases.

On the other hand, complex models, such as Gradient Boosting or Random Forests, often deliver superior predictive performance by capturing non-linear relationships and feature interactions. However, they are less transparent, harder to explain, and more challenging to govern in a regulatory setting. In this project, we explore both approaches to balance performance and interpretability, ensuring that the final model achieves strong predictive power while remaining suitable for deployment in a regulated financial institution.