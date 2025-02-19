import pandas as pd


def append_mean_std_for_method(method_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes all rows for one method, appends Mean & Std rows,
    and then rounds numeric columns to five decimals.
    """
    # Compute mean over numeric columns only
    mean_vals = method_df.mean(numeric_only=True).to_dict()
    mean_vals["Run"] = "Mean"
    mean_vals["Method"] = method_df["Method"].iloc[0]  # same method for whole group

    # Compute std over numeric columns
    std_vals = method_df.std(numeric_only=True).to_dict()
    std_vals["Run"] = "Std"
    std_vals["Method"] = method_df["Method"].iloc[0]

    # Convert them to DataFrame
    extra_rows = pd.DataFrame([mean_vals, std_vals])

    # Concatenate runs + mean/std rows
    appended = pd.concat([method_df, extra_rows], ignore_index=True)

    # Round numeric columns to 5 decimals
    numeric_cols = appended.select_dtypes(include='number').columns
    appended[numeric_cols] = appended[numeric_cols].round(5)

    return appended