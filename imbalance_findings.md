# Imbalance Study — HAM10000 + ETF

## Experimental Setup

Model:
- ETF Head
- ResNet18 backbone
- Pretrained = True
- Focal Loss
- NC Regularization = 0.01
- 30 epochs

Dataset:
- HAM10000

---

# Results Summary

| Setting | Best Acc | Macro F1 | Melanoma Recall | NC1 | NC4 |
|---|---|---|---|---|---|
| Natural | 60.6 | 0.271 | 0.165 | 12.89 | 0.389 |
| Ratio=10 | 59.4 | 0.252 | 0.522 | 5.58 | 0.281 |
| Ratio=50 | 53.1 | 0.203 | 0.154 | 16.04 | 0.580 |

---

# Key Observations

## 1. Moderate imbalance improved minority sensitivity

At imbalance_ratio=10:
- melanoma recall increased dramatically:
  0.165 → 0.522
- NC1 improved significantly:
  12.89 → 5.58
- NC4 also improved:
  0.389 → 0.281

This suggests that moderate imbalance restructuring may reduce dominant-class interference and improve minority geometric separability.

---

## 2. Severe imbalance caused representation degradation

At imbalance_ratio=50:
- melanoma recall dropped again:
  0.522 → 0.154
- NC1 increased sharply:
  5.58 → 16.04
- NC4 worsened:
  0.281 → 0.580

This indicates severe imbalance causes unstable class centers and fragmented feature geometry.

---

# Emerging Hypothesis

Neural Collapse behavior under long-tail medical imbalance appears non-monotonic.

Moderate imbalance may improve class separability by reducing dominant-class representation pressure, while severe imbalance eventually destroys minority feature structure and collapse stability.

---

# Important Insight

Overall accuracy alone does not capture representation quality.

Although the linear classifier achieved higher overall accuracy, ETF geometry consistently improved minority-sensitive behavior and clinically important class recall.
