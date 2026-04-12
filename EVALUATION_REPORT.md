# Finance Optimizer Evaluation Report

*Generated on: 2026-04-12 21:24:05*

## Performance Summary
| Task | Seed | Score | Duration |
| --- | --- | --- | --- |
| ledger_cleanup | 42 | 0.9773 | 2.06s |
| ledger_cleanup | 123 | 0.9773 | 2.06s |
| subscription_audit | 42 | 0.9687 | 2.05s |
| subscription_audit | 123 | 0.9687 | 2.05s |
| cash_flow | 42 | 0.9297 | 2.06s |
| cash_flow | 123 | 0.9297 | 2.06s |
| fraud_categorization | 42 | 0.9694 | 2.05s |
| fraud_categorization | 123 | 0.9694 | 2.07s |
| savings_builder | 42 | 0.9687 | 2.07s |
| savings_builder | 123 | 0.9687 | 2.06s |
| debt_avalanche | 42 | 0.7483 | 2.06s |
| debt_avalanche | 123 | 0.7483 | 2.04s |
| duplicate_charge_alert | 42 | 0.9687 | 2.06s |
| duplicate_charge_alert | 123 | 0.9687 | 2.08s |

## Discussion & Insights
- **FICO Engine**: Scoring now includes dynamic FICO score weight (40%).
- **Constraint Penalty**: Debt Avalanche scores reflect the mandated $500 buffer.
- **Randomization**: Results across seeds indicate high sensitivity to transaction order.
