from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from metric import precision_at_recall, recall_at_fpr


RANDOM_STATE = 42
MODEL_SEEDS = (42, 17, 2025)
META_COLUMNS = [
    "cookie_id",
    "cookie_created_at",
    "window_start_ts",
    "window_end_ts",
]
EVENT_COLUMNS = [
    "cookie_id",
    "event_ts",
    "eid",
    "event_name",
    "platform",
    "user_agent",
    "item_id",
    "item_category",
    "item_location",
    "seller_type",
    "search_query",
    "search_page",
    "pointer_x",
    "pointer_y",
]
CATEGORICAL_FEATURES = [
    "ua_mode",
    "browser_mode",
    "os_mode",
    "platform_mode",
    "event_mode",
    "category_mode",
    "location_mode",
    "seller_mode",
]
EXCLUDED_MODEL_FEATURES = {"duplicate_events"}
BOT_UA_PATTERN = re.compile(r"bot|crawler|spider|headless|selenium|playwright|curl|python-requests", re.I)


def _mode(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False)
    return str(counts.index[0]) if len(counts) else "__missing__"


def _entropy(values: pd.Series) -> float:
    frequencies = values.value_counts(normalize=True, dropna=False).to_numpy()
    return float(-(frequencies * np.log(frequencies)).sum()) if len(frequencies) else 0.0


def _group_entropy(events: pd.DataFrame, column: str, feature_name: str) -> pd.Series:
    counts = events.groupby(["cookie_id", column], observed=True, dropna=False).size()
    proportions = counts.div(counts.groupby(level=0, sort=False).transform("sum"))
    return (-(proportions * np.log(proportions)).groupby(level=0, sort=False).sum()).rename(
        feature_name
    )


def _group_mode(events: pd.DataFrame, column: str, feature_name: str) -> pd.Series:
    counts = events.groupby(["cookie_id", column], observed=True, dropna=False).size()
    ranked = counts.rename("count").reset_index()
    ranked = ranked.sort_values(
        ["cookie_id", "count", column],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ranked.drop_duplicates("cookie_id").set_index("cookie_id")[column].rename(
        feature_name
    )


def _parse_user_agent(events: pd.DataFrame) -> pd.DataFrame:
    ua = events["user_agent"].fillna("").astype(str)
    ua_lower = ua.str.lower()
    browser_rules = [
        (r"edg/|edge/", "edge"),
        (r"yabrowser/", "yandex"),
        (r"opr/|opera", "opera"),
        (r"chrome/|crios/", "chrome"),
        (r"firefox/|fxios/", "firefox"),
        (r"safari/", "safari"),
    ]
    browser = pd.Series("other", index=events.index, dtype="object")
    for pattern, name in browser_rules:
        browser.loc[ua_lower.str.contains(pattern, regex=True)] = name

    os_rules = [
        (r"windows", "windows"),
        (r"android", "android"),
        (r"iphone|ipad|ipod|ios", "ios"),
        (r"macintosh|mac os", "macos"),
        (r"linux|x11", "linux"),
    ]
    operating_system = pd.Series("other", index=events.index, dtype="object")
    for pattern, name in os_rules:
        operating_system.loc[ua_lower.str.contains(pattern, regex=True)] = name

    browser_version = ua.str.extract(
        r"(?:Edg|Edge|YaBrowser|Chrome|CriOS|Firefox|FxiOS|Version)/([\d.]+)",
        expand=False,
    ).str.split(".").str[0]

    return pd.DataFrame(
        {
            "ua_mode": ua.replace("", "__missing__"),
            "browser": browser,
            "operating_system": operating_system,
            "browser_major": browser_version.fillna("__missing__"),
            "ua_automation_hint": ua.str.contains(BOT_UA_PATTERN).astype("int8"),
        },
        index=events.index,
    )


def build_features(meta: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build per-cookie features using only events inside each observation window."""
    required_meta = set(META_COLUMNS)
    required_events = set(EVENT_COLUMNS)
    if missing := required_meta.difference(meta.columns):
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if missing := required_events.difference(events.columns):
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    if meta["cookie_id"].duplicated().any():
        raise ValueError("Metadata must contain exactly one row per cookie_id")

    cookies = meta[META_COLUMNS].copy()
    for column in ("cookie_created_at", "window_start_ts", "window_end_ts"):
        cookies[column] = pd.to_datetime(cookies[column], errors="coerce")

    selected_events = events[EVENT_COLUMNS].copy()
    selected_events["event_ts"] = pd.to_datetime(selected_events["event_ts"], errors="coerce")
    selected_events = selected_events.dropna(subset=["event_ts"])
    selected_events = selected_events.merge(
        cookies[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id",
        how="inner",
        validate="many_to_one",
    )
    in_window = selected_events["event_ts"].ge(selected_events["window_start_ts"])
    in_window &= selected_events["event_ts"].lt(selected_events["window_end_ts"])
    ev = selected_events.loc[in_window].copy()

    categorical_source = {
        "platform": "platform",
        "event_name": "event_name",
        "item_category": "item_category",
        "item_location": "item_location",
        "seller_type": "seller_type",
    }
    for column in categorical_source:
        ev[column] = ev[column].fillna("__missing__").astype(str).str.strip().str.lower()
        ev[column] = ev[column].replace("", "__missing__")
    ev["user_agent"] = ev["user_agent"].fillna("").astype(str).str.strip()
    ev["search_query"] = ev["search_query"].fillna("").astype(str).str.strip().str.lower()
    ev["search_page"] = pd.to_numeric(ev["search_page"], errors="coerce")
    ev["pointer_x"] = pd.to_numeric(ev["pointer_x"], errors="coerce")
    ev["pointer_y"] = pd.to_numeric(ev["pointer_y"], errors="coerce")
    ev = ev.join(_parse_user_agent(ev))
    ev["is_duplicate"] = ev.duplicated(subset=EVENT_COLUMNS, keep="first").astype("int8")
    ev = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort")
    duplicate_counts = ev.groupby("cookie_id", sort=False)["is_duplicate"].sum().rename(
        "duplicate_events"
    )
    ev = ev.drop_duplicates(subset=EVENT_COLUMNS, keep="first")

    grouped = ev.groupby("cookie_id", sort=False)
    features = grouped.agg(
        n_events=("event_ts", "size"),
        n_event_types=("event_name", "nunique"),
        n_items=("item_id", "nunique"),
        n_categories=("item_category", "nunique"),
        n_locations=("item_location", "nunique"),
        n_seller_types=("seller_type", "nunique"),
        n_platforms=("platform", "nunique"),
        n_user_agents=("user_agent", "nunique"),
        first_event_ts=("event_ts", "min"),
        last_event_ts=("event_ts", "max"),
        pointer_events=("pointer_x", "count"),
        search_page_mean=("search_page", "mean"),
        search_page_max=("search_page", "max"),
        search_page_std=("search_page", "std"),
        pointer_x_nunique=("pointer_x", "nunique"),
        pointer_y_nunique=("pointer_y", "nunique"),
        ua_automation_hint=("ua_automation_hint", "max"),
    )
    mode_sources = {
        "ua_mode": "ua_mode",
        "browser_mode": "browser",
        "os_mode": "operating_system",
        "platform_mode": "platform",
        "event_mode": "event_name",
        "category_mode": "item_category",
        "location_mode": "item_location",
        "seller_mode": "seller_type",
    }
    for feature_name, source_column in mode_sources.items():
        features = features.join(_group_mode(ev, source_column, feature_name))
    searched = ev.loc[ev["search_query"].ne("")].groupby("cookie_id")["search_query"].nunique()
    features["n_search_queries"] = searched
    features = features.join(duplicate_counts)
    features["event_span_seconds"] = (
        features["last_event_ts"] - features["first_event_ts"]
    ).dt.total_seconds()

    gaps = grouped["event_ts"].diff().dt.total_seconds()
    gap_frame = ev[["cookie_id"]].copy()
    gap_frame["gap_seconds"] = gaps
    valid_gaps = gap_frame.dropna(subset=["gap_seconds"])
    if not valid_gaps.empty:
        gap_features = valid_gaps.groupby("cookie_id")["gap_seconds"].agg(
            gap_mean_seconds="mean",
            gap_median_seconds="median",
            gap_std_seconds="std",
            gap_min_seconds="min",
            gap_p90_seconds=lambda values: values.quantile(0.9),
        )
        gap_features["gap_under_2s_share"] = valid_gaps.groupby("cookie_id")[
            "gap_seconds"
        ].apply(lambda values: values.le(2).mean())
        gap_features["gap_under_10s_share"] = valid_gaps.groupby("cookie_id")[
            "gap_seconds"
        ].apply(lambda values: values.le(10).mean())
        features = features.join(gap_features)

    for column, prefix in [
        ("event_name", "event_entropy"),
        ("item_category", "category_entropy"),
        ("item_location", "location_entropy"),
        ("platform", "platform_entropy"),
        ("seller_type", "seller_entropy"),
        ("browser", "browser_entropy"),
    ]:
        features = features.join(_group_entropy(ev, column, prefix))

    event_counts = pd.crosstab(ev["cookie_id"], ev["event_name"]).add_prefix("event_count_")
    features = features.join(event_counts)
    if not ev.empty:
        hour_counts = pd.crosstab(ev["cookie_id"], ev["event_ts"].dt.hour)
        active_hours = (hour_counts > 0).sum(axis=1).rename("active_hours")
        night_count = hour_counts.reindex(columns=range(0, 6), fill_value=0).sum(axis=1)
        night_share = (night_count / hour_counts.sum(axis=1)).rename("night_event_share")
        hour_entropy = hour_counts.apply(_entropy, axis=1).rename("hour_entropy")
        features = features.join([active_hours, night_share, hour_entropy])

        pointer_events = ev.dropna(subset=["pointer_x", "pointer_y"])
        if not pointer_events.empty:
            pointer_events = pointer_events.copy()
            pointer_events["step_distance"] = np.hypot(
                pointer_events.groupby("cookie_id")["pointer_x"].diff(),
                pointer_events.groupby("cookie_id")["pointer_y"].diff(),
            )
            pointer_features = pointer_events.groupby("cookie_id").agg(
                pointer_valid_events=("step_distance", "size"),
                pointer_path_length=("step_distance", "sum"),
                pointer_step_mean=("step_distance", "mean"),
                pointer_step_max=("step_distance", "max"),
            )
            features = features.join(pointer_features)

    features = features.reset_index()
    features = cookies.merge(features, on="cookie_id", how="left", validate="one_to_one")

    duration = (features["window_end_ts"] - features["window_start_ts"]).dt.total_seconds()
    age = (features["window_start_ts"] - features["cookie_created_at"]).dt.total_seconds()
    features["window_duration_hours"] = duration / 3600
    features["cookie_age_days"] = age / 86400
    features["cookie_created_hour"] = features["cookie_created_at"].dt.hour
    features["window_start_hour"] = features["window_start_ts"].dt.hour
    features["window_start_weekday"] = features["window_start_ts"].dt.dayofweek
    features["first_event_offset_seconds"] = (
        features["first_event_ts"] - features["window_start_ts"]
    ).dt.total_seconds()
    features["last_event_offset_seconds"] = (
        features["window_end_ts"] - features["last_event_ts"]
    ).dt.total_seconds()
    features["events_per_active_hour"] = features["n_events"] / features["active_hours"].replace(0, np.nan)

    event_count_columns = [column for column in features if column.startswith("event_count_")]
    for column in event_count_columns:
        features[f"{column}_share"] = features[column] / features["n_events"].replace(0, np.nan)

    dropped = META_COLUMNS[1:] + ["first_event_ts", "last_event_ts"]
    features = features.drop(columns=dropped)
    for column in CATEGORICAL_FEATURES:
        if column not in features:
            features[column] = "__missing__"
        features[column] = features[column].fillna("__missing__").astype(str)
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    features[numeric_columns] = features[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return features


def make_model(
    numeric_features: list[str],
    categorical_features: list[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    transformers = []
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=False,
                ),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric_features)
        )
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.06,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_catboost_model(random_seed: int = RANDOM_STATE) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function="Logloss",
        verbose=False,
        random_seed=random_seed,
        thread_count=4,
        allow_writing_files=False,
    )


def catboost_frame(
    features: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    model_features = features[categorical_features + numeric_features].copy()
    for column in categorical_features:
        model_features[column] = model_features[column].fillna("__missing__").astype(str)
    return model_features


def average_rank(scores: list[np.ndarray]) -> np.ndarray:
    ranks = [pd.Series(values).rank(method="average", pct=True).to_numpy() for values in scores]
    return np.mean(ranks, axis=0)


def split_feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numeric = [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column not in EXCLUDED_MODEL_FEATURES
    ]
    return categorical, numeric


def fit_and_evaluate(
    train: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    dates = pd.to_datetime(train["window_start_ts"]).dt.normalize()
    last_date = dates.max()
    windows = [
        (last_date - pd.Timedelta(days=8), last_date - pd.Timedelta(days=5)),
        (last_date - pd.Timedelta(days=5), last_date - pd.Timedelta(days=2)),
        (last_date - pd.Timedelta(days=2), last_date + pd.Timedelta(days=1)),
    ]
    baseline_columns = ["n_events", "n_items"]
    categorical, numeric = split_feature_columns(features)

    for validation_start, validation_end in windows:
        is_valid = dates.ge(validation_start) & dates.lt(validation_end)
        is_train = dates.lt(validation_start)
        if not is_valid.any() or not is_train.any():
            raise ValueError("Temporal validation split is empty; check train window dates")

        labels = train.loc[is_valid, "target"].to_numpy()
        training_labels = train.loc[is_train, "target"]
        baseline = make_model(baseline_columns, categorical_features=[])
        baseline.fit(features.loc[is_train, baseline_columns], training_labels)
        baseline_probabilities = baseline.predict_proba(
            features.loc[is_valid, baseline_columns]
        )[:, 1]

        model = make_model(numeric)
        model.fit(features.loc[is_train], training_labels)
        hgb_probabilities = model.predict_proba(features.loc[is_valid])[:, 1]

        cat_features = catboost_frame(features, categorical, numeric)
        cat_seed_probabilities = []
        for seed in MODEL_SEEDS:
            cat_model = make_catboost_model(seed)
            cat_model.fit(
                cat_features.loc[is_train],
                training_labels,
                cat_features=categorical,
            )
            cat_seed_probabilities.append(
                cat_model.predict_proba(cat_features.loc[is_valid])[:, 1]
            )
        cat_rank_average = average_rank(cat_seed_probabilities)

        print(
            f"Validation {validation_start.date()} to "
            f"{(validation_end - pd.Timedelta(days=1)).date()} "
            f"({int(is_valid.sum())} cookies, bot share={labels.mean():.4f})"
        )
        for name, scores in [
            ("Baseline (event/item counts)", baseline_probabilities),
            ("HistGradientBoosting", hgb_probabilities),
            ("CatBoost seed 42", cat_seed_probabilities[0]),
            ("CatBoost 3-seed rank average", cat_rank_average),
        ]:
            print(
                f"{name}: P@R>=0.70={precision_at_recall(labels, scores):.4f}, "
                f"PR-AUC={average_precision_score(labels, scores):.4f}, "
                f"ROC-AUC={roc_auc_score(labels, scores):.4f}, "
                f"Recall@FPR<=0.01={recall_at_fpr(labels, scores):.4f}"
            )


def create_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    output_path: Path,
) -> None:
    categorical, numeric = split_feature_columns(train_features)
    train_cat_features = catboost_frame(train_features, categorical, numeric)
    test_cat_features = catboost_frame(test_features, categorical, numeric)
    cat_seed_scores = []
    for seed in MODEL_SEEDS:
        cat_model = make_catboost_model(seed)
        cat_model.fit(train_cat_features, train["target"], cat_features=categorical)
        cat_seed_scores.append(cat_model.predict_proba(test_cat_features)[:, 1])
    submission = pd.DataFrame(
        {
            "cookie_id": test["cookie_id"].to_numpy(),
            "score": average_rank(cat_seed_scores),
        }
    )
    if len(submission) != len(test) or submission["cookie_id"].duplicated().any():
        raise ValueError("Submission must contain exactly one row per test cookie")
    if not submission["score"].between(0, 1).all():
        raise ValueError("Submission contains scores outside [0, 1]")
    if set(submission["cookie_id"]) != set(test["cookie_id"]):
        raise ValueError("Submission cookie IDs do not match test.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Avito bot-detection model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--mode", choices=("all", "validate", "submit"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.data_dir / "train.csv", parse_dates=META_COLUMNS[1:])
    test = pd.read_csv(args.data_dir / "test.csv", parse_dates=META_COLUMNS[1:])
    events = pd.read_csv(args.data_dir / "events.csv.gz", parse_dates=["event_ts"])
    all_meta = pd.concat(
        [train[META_COLUMNS], test[META_COLUMNS]],
        ignore_index=True,
    )
    all_features = build_features(all_meta, events)
    train_features = train[["cookie_id"]].merge(
        all_features, on="cookie_id", how="left", validate="one_to_one"
    ).drop(columns="cookie_id")
    test_features = test[["cookie_id"]].merge(
        all_features, on="cookie_id", how="left", validate="one_to_one"
    ).drop(columns="cookie_id")

    if args.mode in ("all", "validate"):
        fit_and_evaluate(train, train_features)
    if args.mode in ("all", "submit"):
        create_submission(train, test, train_features, test_features, args.output)


if __name__ == "__main__":
    main()