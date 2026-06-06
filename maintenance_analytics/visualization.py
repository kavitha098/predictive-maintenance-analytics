import plotly.express as px


def plot_temperature(df):

    fig = px.line(
        df,
        x=df.index,
        y="Temperature (°C)",
        title="Temperature Trend"
    )

    return fig


def plot_vibration(df):

    fig = px.line(
        df,
        x=df.index,
        y="Vibration (m/s²)",
        title="Vibration Trend"
    )

    return fig
