import numpy as np


def ab_test(y_true, preds_a, preds_b, threshold=0.5):
    """
    Compare two models using a simple A/B simulation.

    Model A: baseline (Logistic Regression)
    Model B: improved (XGBoost)
    """

    # Convert to numpy (safe handling)
    y_true = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)

    # Approval decisions (lower risk = approve)
    approve_a = preds_a < threshold
    approve_b = preds_b < threshold

    # Default rates among approved customers
    default_rate_a = y_true[approve_a].mean()
    default_rate_b = y_true[approve_b].mean()

    # Approval rates
    approval_rate_a = approve_a.mean()
    approval_rate_b = approve_b.mean()

    print("\n--- A/B TEST RESULTS ---")
    print(f"Approval Rate (Model A - LR):  {approval_rate_a:.4f}")
    print(f"Approval Rate (Model B - XGB): {approval_rate_b:.4f}")

    print(f"Default Rate (Model A - LR):  {default_rate_a:.4f}")
    print(f"Default Rate (Model B - XGB): {default_rate_b:.4f}")

    improvement = default_rate_a - default_rate_b
    print(f"\nRisk Reduction (B vs A): {improvement:.4f}")

    return {
        "approval_rate_a": approval_rate_a,
        "approval_rate_b": approval_rate_b,
        "default_rate_a": default_rate_a,
        "default_rate_b": default_rate_b,
        "improvement": improvement,
    }

def ab_business_insight(y_true, lr_probs, xgb_probs, threshold=0.5):
    import numpy as np

    y_true = np.array(y_true)

    approve_lr = lr_probs < threshold
    approve_xgb = xgb_probs < threshold

    approval_lr = approve_lr.mean()
    approval_xgb = approve_xgb.mean()

    default_lr = y_true[approve_lr].mean()
    default_xgb = y_true[approve_xgb].mean()

    improvement = default_lr - default_xgb
    approval_gain = approval_xgb - approval_lr

    return {
        "approval_lr": approval_lr,
        "approval_xgb": approval_xgb,
        "default_lr": default_lr,
        "default_xgb": default_xgb,
        "risk_reduction": improvement,
        "approval_gain": approval_gain
    }