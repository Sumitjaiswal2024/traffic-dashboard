import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Traffic Monitoring Pipeline Dashboard", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
COUNTS_FILE = DATA_DIR / "cleaned_vehicle_counts.csv"
EVENTS_FILE = DATA_DIR / "cleaned_traffic_events.json"
IMAGES_FILE = DATA_DIR / "image_validation.csv"
LOG_FILE = DATA_DIR / "cleaning_log.txt"

COUNT_COLUMNS = ["car", "motorbike", "bus", "truck"]
COLORS = {"car": "#2563eb", "motorbike": "#f59e0b", "bus": "#10b981", "truck": "#ef4444"}
COUNTS_REQUIRED = ["timestamp", "Direction", "total_vehicles", "is_simulated"] + COUNT_COLUMNS
EVENTS_REQUIRED = ["timestamp", "event_type", "vehicle_count"]
IMAGES_REQUIRED = ["filename", "status"]


@st.cache_data
def load_counts():
    if not COUNTS_FILE.exists():
        return None, f"File not found: data/{COUNTS_FILE.name}"
    try:
        df = pd.read_csv(COUNTS_FILE)
        missing = [c for c in COUNTS_REQUIRED if c not in df.columns]
        if missing:
            return None, f"data/{COUNTS_FILE.name} is missing column(s): {missing}"
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, None
    except Exception as error:
        return None, f"Could not read data/{COUNTS_FILE.name}: {error}"


@st.cache_data
def load_events():
    if not EVENTS_FILE.exists():
        return None, {}, f"File not found: data/{EVENTS_FILE.name}"
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        records = data.get("events", []) if isinstance(data, dict) else data
        df = pd.DataFrame(records)
        missing = [c for c in EVENTS_REQUIRED if c not in df.columns]
        if missing:
            return None, metadata, f"data/{EVENTS_FILE.name} is missing field(s): {missing}"
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, metadata, None
    except Exception as error:
        return None, {}, f"Could not read data/{EVENTS_FILE.name}: {error}"


@st.cache_data
def load_images():
    if not IMAGES_FILE.exists():
        return None, f"File not found: data/{IMAGES_FILE.name}"
    try:
        df = pd.read_csv(IMAGES_FILE)
        missing = [c for c in IMAGES_REQUIRED if c not in df.columns]
        if missing:
            return None, f"data/{IMAGES_FILE.name} is missing column(s): {missing}"
        return df, None
    except Exception as error:
        return None, f"Could not read data/{IMAGES_FILE.name}: {error}"


@st.cache_data
def load_log():
    if not LOG_FILE.exists():
        return None, f"File not found: data/{LOG_FILE.name}"
    try:
        return LOG_FILE.read_text(encoding="utf-8"), None
    except Exception as error:
        return None, f"Could not read data/{LOG_FILE.name}: {error}"


def simulated_banner(counts, metadata):
    counts_simulated = counts is not None and bool(counts["is_simulated"].all())
    events_simulated = str(metadata.get("data_origin", "")).lower() == "simulated"
    if counts_simulated or events_simulated:
        st.warning(
            "SIMULATED DATA: the traffic counts and events were created to demonstrate "
            "the pipeline and do not represent real observed traffic. The photos are real "
            "but are not linked to these records."
        )


def page_overview(counts, events, images):
    st.header("Overview")
    if counts is None:
        st.info("Traffic-count data is not available, so the overview cannot be shown.")
        return

    per_interval = counts.groupby("timestamp", as_index=False)["total_vehicles"].sum()
    first, last = counts["timestamp"].min(), counts["timestamp"].max()

    a, b, c = st.columns(3)
    a.metric("Total vehicles", f"{int(counts['total_vehicles'].sum()):,}")
    b.metric("Time intervals", int(counts["timestamp"].nunique()))
    c.metric("Traffic-count rows", len(counts))

    d, e, f = st.columns(3)
    d.metric("Events", len(events) if events is not None else "n/a")
    if images is not None:
        valid = int((images["status"] == "valid").sum())
        e.metric("Valid images", f"{valid} of {len(images)}")
    else:
        e.metric("Valid images", "n/a")
    f.metric("Directions", int(counts["Direction"].nunique()))

    if first.date() == last.date():
        st.caption(f"Data covers {first:%d %b %Y}, {first:%H:%M} to {last:%H:%M}.")
    else:
        st.caption(f"Data covers {first:%d %b %Y %H:%M} to {last:%d %b %Y %H:%M}.")

    left, right = st.columns([3, 2])
    with left:
        fig = px.line(
            per_interval, x="timestamp", y="total_vehicles", markers=True,
            title="Total vehicles per interval",
            labels={"timestamp": "Time", "total_vehicles": "Vehicles"},
        )
        st.plotly_chart(fig)
    with right:
        type_totals = counts[COUNT_COLUMNS].sum().reset_index()
        type_totals.columns = ["Vehicle type", "Total"]
        fig = px.bar(
            type_totals, x="Vehicle type", y="Total", color="Vehicle type",
            color_discrete_map=COLORS, title="Vehicles by type",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

    valid_intervals = per_interval.dropna(subset=["total_vehicles"])
    if not valid_intervals.empty:
        busiest = valid_intervals.loc[valid_intervals["total_vehicles"].idxmax()]
        quietest = valid_intervals.loc[valid_intervals["total_vehicles"].idxmin()]
        x, y = st.columns(2)
        x.success(f"Busiest interval: {busiest['timestamp']:%d %b %H:%M} with {int(busiest['total_vehicles'])} vehicles")
        y.info(f"Quietest interval: {quietest['timestamp']:%d %b %H:%M} with {int(quietest['total_vehicles'])} vehicles")

    st.subheader("Events by type")
    if events is None:
        st.info("Event data is not available.")
    else:
        by_type = events["event_type"].value_counts()
        for column, (name, number) in zip(st.columns(len(by_type)), by_type.items()):
            column.metric(str(name).replace("_", " ").title(), int(number))


def coming_soon(name):
    st.header(name)
    st.info(f"The {name} page will be added in a later step.")


PAGES = ["Overview", "Pipeline", "Analytics", "Records", "Status"]

if "loaded_at" not in st.session_state:
    st.session_state["loaded_at"] = datetime.now()

st.sidebar.title("Traffic Pipeline")
page = st.sidebar.radio("Go to", PAGES)
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.session_state["loaded_at"] = datetime.now()
    st.rerun()
st.sidebar.caption(f"Data loaded at {st.session_state['loaded_at']:%H:%M:%S} (server time)")

counts, counts_error = load_counts()
events, event_metadata, events_error = load_events()
images, images_error = load_images()
log_text, log_error = load_log()

st.title("Traffic Monitoring Data Pipeline")
simulated_banner(counts, event_metadata)
for message in (counts_error, events_error, images_error, log_error):
    if message:
        st.error(message)

if page == "Overview":
    page_overview(counts, events, images)
else:
    coming_soon(page)
