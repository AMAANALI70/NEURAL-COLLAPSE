# Master Experiment Table — Neural Collapse under Medical Imbalance

| Experiment ID | Head | Imbalance Ratio | NC Reg | Best Acc | Macro F1 | ROC-AUC | Melanoma Recall | DF Recall | Vascular Recall | NC1 | NC2 | NC3 | NC4 | Key Observation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-01 | ETF | Natural | 0.01 | 60.6 | 0.271 | 0.766 | 0.165 | 0.059 | 0.083 | 12.89 | 1.187 | 0.945 | 0.389 | ETF improves minority sensitivity |
| EXP-02 | Linear | Natural | 0.01 | 71.9 | 0.216 | 0.761 | 0.033 | 0.000 | 0.000 | 12.98 | 1.128 | 0.901 | 0.561 | Majority-class dominance |
| EXP-03 | ETF | 10 | 0.01 | 59.4 | 0.252 | 0.744 | 0.522 | 0.176 | 0.167 | 5.58 | 1.100 | 0.904 | 0.281 | Moderate imbalance improved minority geometry |
| EXP-04 | ETF | 50 | 0.01 | 53.1 | 0.203 | 0.744 | 0.154 | 0.059 | 0.500 | 16.04 | 1.244 | 0.966 | 0.580 | Severe imbalance fragmented geometry |

---

# Emerging Trends

## 1. Linear classifiers optimize dominant classes
- Highest overall accuracy
- Worst minority sensitivity
- Complete collapse on DF and Vascular classes

---

## 2. ETF geometry improves clinically important minority recall
- Strong improvements in melanoma sensitivity
- Better macro-F1 despite lower overall accuracy

---

## 3. Imbalance effects appear non-monotonic
- Moderate imbalance (ratio=10) improved:
  - NC1
  - NC4
  - melanoma recall
- Severe imbalance (ratio=50) sharply degraded geometry

---

## 4. NC metrics correlate with minority behavior
Lower NC1 / NC4 consistently aligned with:
- cleaner class geometry,
- improved minority recall,
- stronger representation compactness.

---

# Current Hypothesis

Moderate imbalance may reduce dominant-class representation interference and improve minority geometric separability, while severe imbalance eventually destabilizes feature collapse and destroys minority structure.
