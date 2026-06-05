import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def plot_temperature(df):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["Timestamp"],
        df["Temperature (°C)"]
    )

    ax.set_title(
        "Temperature Trend"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")

    return fig


def plot_vibration(df):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["Timestamp"],
        df["Vibration (m/s²)"]
    )

    ax.set_title(
        "Vibration Trend"
    )

    return fig


def plot_failure_distribution(df):

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Fault Detected",
        data=df,
        ax=ax
    )

    ax.set_title(
        "Failure Distribution"
    )

    return fig


def plot_correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    return fig


def create_wordcloud(df):

    if "Failure Type" not in df.columns:
        return None

    text = " ".join(
        df["Failure Type"]
        .astype(str)
    )

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.imshow(wc)

    ax.axis("off")

    ax.set_title(
        "Failure Type Word Cloud"
    )

    return fig


def plot_anomaly_trend(df):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        df["Timestamp"],
        df["Temperature (°C)"],
        label="Temperature"
    )

    anomaly_df = df[
        df["Anomaly_Alert"] == 1
    ]

    ax.scatter(
        anomaly_df["Timestamp"],
        anomaly_df["Temperature (°C)"]
    )

    ax.legend()

    ax.set_title(
        "Anomaly Detection Trend"
    )

    return fig
