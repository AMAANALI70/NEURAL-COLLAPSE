# ETF vs Linear — HAM10000

| Metric | ETF | Linear |
|---|---|---|
| Best Accuracy | 65.6 | 71.9 |
| Macro F1 | 0.244 | 0.216 |
| ROC-AUC | 0.791 | 0.761 |
| NC2 | 1.096 | 1.128 |
| NC3 | 0.937 | 0.901 |
| NC4 | 0.587 | 0.561 |

## Key Minority Results

| Class | ETF Recall | Linear Recall |
|---|---|---|
| Melanoma | 0.165 | 0.033 |
| DF | 0.059 | 0.0 |
| Vascular | 0.083 | 0.0 |

## Key Observation

Linear classifiers optimize dominant-class accuracy but collapse minority classes under imbalance.


ETF geometry improves minority-sensitive behavior and preserves clinically important class structure, despite lower overall accuracy.
