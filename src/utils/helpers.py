import pandas as pd


def append_mean_std_for_method(method_df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """
    Takes all rows for one method, appends Mean & Std rows,
    and then rounds numeric columns to five decimals.
    """
    mean_vals = method_df.mean(numeric_only=True).to_dict()
    mean_vals["Run"] = "Mean"
    mean_vals["Method"] = method_name
    std_vals = method_df.std(numeric_only=True).to_dict()
    std_vals["Run"] = "Std"
    std_vals["Method"] = method_name
    extra_rows = pd.DataFrame([mean_vals, std_vals])
    appended = pd.concat([method_df, extra_rows], ignore_index=True)
    return appended