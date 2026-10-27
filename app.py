import json
import re
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
IMAGES_DIR = Path(__file__).parent / "images"
PAGES = ["Overview", "Pipeline", "Analytics", "Records", "Traffic Images", "Status"]
LOG_LABELS = {
    "Rows before cleaning": "csv_before", "Rows after cleaning": "csv_after",
    "Exact duplicate rows removed": "csv_exact",
    "Same time/junction/direction duplicates removed": "csv_key",
    "Invalid rows rejected": "csv_rejected", "Junction_ID filled 'UNKNOWN'": "junction_filled",
    "Non-numeric counts -> missing": "non_numeric", "Rows kept with a missing count": "partial_missing",
    "Arithmetic check": "arithmetic", "Missing values found (per column)": "missing_text",
    "Records before cleaning": "json_before", "Exact duplicate records removed": "json_dupes",
    "Invalid records rejected": "json_rejected", "Records after cleaning": "json_after",
    "Images found": "images_found", "Images validated OK": "images_ok",
    "Image validation failures": "image_failures", "Raw files unchanged": "raw_unchanged",
}


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


def fmt(value):
    if value is None:
        return "n/a"
    return f"{value:,}" if isinstance(value, int) else str(value)


def metric_card(container, label, value, delta=None, help_text=None):
    kwargs = {"help": help_text}
    if delta is not None:
        kwargs.update(delta=delta, delta_color="off")
    try:
        container.metric(label, value, border=True, **kwargs)
    except TypeError:
        container.metric(label, value, **kwargs)


def style_fig(fig, title, height=360):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.25, title_text=""))
    return fig


def section(title, caption=None):
    st.divider()
    st.subheader(title)
    if caption:
        st.caption(caption)


def parse_log(text):
    info = {"reasons": {"csv": {}, "json": {}}, "output_table": [], "join_lines": []}
    if not text:
        return info
    part = None
    for line in text.splitlines():
        if line.startswith("--- CSV"):
            part = "csv"
        elif line.startswith("--- JSON"):
            part = "json"
        elif line.startswith("--- IMAGES"):
            part = "images"
        elif line.startswith("--- JOIN"):
            part = "join"
        elif line.startswith("--- OUTPUT"):
            part = "output"
        elif line.startswith("Raw files were") or line.startswith("Files saved in"):
            part = "end"
        elif part == "join" and line.strip():
            info["join_lines"].append(line.strip())
        reason = re.match(r"^\s+- (.+): (\d+)$", line)
        if reason and part in ("csv", "json"):
            info["reasons"][part][reason.group(1)] = int(reason.group(2))
            continue
        if part == "output":
            cells = line.split()
            if len(cells) >= 5 and cells[1].isdigit() and cells[2].isdigit():
                info["output_table"].append({"File": cells[0], "Expected": int(cells[1]), "Found": int(cells[2]),
                                             "Size (KB)": cells[3], "Status": " ".join(cells[4:])})
        if ":" in line:
            label, value = line.split(":", 1)
            key = LOG_LABELS.get(label.strip())
            if key:
                number = re.match(r"\s*(\d+)", value)
                info[key] = int(number.group(1)) if number else value.strip()
    return info


def time_range_filter(times, key):
    labels = [f"{t:%d %b %H:%M}" for t in times]
    start, end = st.select_slider("Time range", options=labels, value=(labels[0], labels[-1]), key=key)
    return times[labels.index(start)], times[labels.index(end)]


def search_filter(df, query):
    if not query:
        return df
    mask = df.astype(str).apply(lambda c: c.str.contains(query, case=False, regex=False, na=False)).any(axis=1)
    return df[mask]


def find_thumbnail(filename, thumb_index):
    if thumb_index is not None and "thumbnail_file" in thumb_index.columns:
        row = thumb_index[thumb_index["filename"] == filename]
        if not row.empty and (IMAGES_DIR / str(row.iloc[0]["thumbnail_file"])).exists():
            return IMAGES_DIR / str(row.iloc[0]["thumbnail_file"])
    for ext in (".jpg", ".jpeg", ".png"):
        path = IMAGES_DIR / (Path(filename).stem + ext)
        if path.exists():
            return path
    return None


@st.cache_data
def load_thumb_index():
    path = IMAGES_DIR / "thumbnail_index.csv"
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None


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
    metric_card(a, "Total vehicles", f"{int(counts['total_vehicles'].sum()):,}")
    metric_card(b, "Time intervals", int(counts["timestamp"].nunique()))
    metric_card(c, "Traffic-count rows", len(counts))

    d, e, f = st.columns(3)
    metric_card(d, "Events", len(events) if events is not None else "n/a")
    if images is not None:
        valid = int((images["status"] == "valid").sum())
        metric_card(e, "Valid images", f"{valid} of {len(images)}")
    else:
        metric_card(e, "Valid images", "n/a")
    metric_card(f, "Directions", int(counts["Direction"].nunique()))

    if first.date() == last.date():
        st.caption(f"Data covers {first:%d %b %Y}, {first:%H:%M} to {last:%H:%M}.")
    else:
        st.caption(f"Data covers {first:%d %b %Y %H:%M} to {last:%d %b %Y %H:%M}.")

    valid_intervals = per_interval.dropna(subset=["total_vehicles"]).reset_index(drop=True)
    by_direction = (counts.groupby("Direction", as_index=False)["total_vehicles"].sum()
                    .sort_values("total_vehicles", ascending=False).reset_index(drop=True))
    grand_total = int(counts["total_vehicles"].sum())

    section("Traffic statistics", "Calculated from the cleaned traffic-count table.")
    g, h, i, j = st.columns(4)
    metric_card(g, "Average per interval", f"{valid_intervals['total_vehicles'].mean():.1f}",
                help_text="Mean of the total vehicles in each time interval.")
    metric_card(h, "Busiest direction", str(by_direction.iloc[0]["Direction"]),
                delta=f"{int(by_direction.iloc[0]['total_vehicles']):,} vehicles")
    metric_card(i, "Quietest direction", str(by_direction.iloc[-1]["Direction"]),
                delta=f"{int(by_direction.iloc[-1]['total_vehicles']):,} vehicles")
    if len(valid_intervals) > 1:
        change = valid_intervals["total_vehicles"].diff()
        pos = int(change.abs().idxmax())
        metric_card(j, "Largest change between intervals", f"{int(change.iloc[pos]):+d} vehicles",
                    delta=f"{valid_intervals['timestamp'].iloc[pos - 1]:%H:%M} to {valid_intervals['timestamp'].iloc[pos]:%H:%M}")
    else:
        metric_card(j, "Largest change between intervals", "n/a")

    left, right = st.columns([3, 2])
    with left:
        fig = px.line(
            per_interval, x="timestamp", y="total_vehicles", markers=True,
            title="Total vehicles per interval",
            labels={"timestamp": "Time", "total_vehicles": "Vehicles"},
        )
        st.plotly_chart(style_fig(fig, "Total vehicles per interval"))
    with right:
        type_totals = counts[COUNT_COLUMNS].sum().reset_index()
        type_totals.columns = ["Vehicle type", "Total"]
        fig = px.bar(
            type_totals, x="Vehicle type", y="Total", color="Vehicle type",
            color_discrete_map=COLORS, title="Vehicles by type",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, "Vehicles by type"))

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
            metric_card(column, str(name).replace("_", " ").title(), int(number))

    section("Traffic by direction", "Total vehicles counted in each direction, with each direction's share of all traffic.")
    left, right = st.columns([3, 2])
    by_direction["Share (%)"] = (by_direction["total_vehicles"] / grand_total * 100).round(1) if grand_total else 0
    with left:
        fig = px.bar(by_direction, x="Direction", y="total_vehicles", color="Direction",
                     labels={"total_vehicles": "Vehicles"}, text="total_vehicles")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, "Total vehicles by direction"))
    with right:
        st.dataframe(by_direction.rename(columns={"total_vehicles": "Vehicles"}), hide_index=True)

    section("Vehicle types: counts and percentages", "Share of all counted vehicles, by type.")
    type_table = counts[COUNT_COLUMNS].sum().reset_index()
    type_table.columns = ["Vehicle type", "Count"]
    type_all = int(type_table["Count"].sum())
    type_table["Percentage (%)"] = (type_table["Count"] / type_all * 100).round(1) if type_all else 0
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(type_table, hide_index=True)
    with right:
        fig = px.pie(type_table, names="Vehicle type", values="Count", hole=0.5,
                     color="Vehicle type", color_discrete_map=COLORS)
        st.plotly_chart(style_fig(fig, "Vehicle-type share"))

    section("Traffic trend over time", "Interval totals with a 3-interval moving average (calculated; it starts at the third interval).")
    trend = valid_intervals.copy()
    trend["3-interval moving average"] = trend["total_vehicles"].rolling(3, min_periods=3).mean()
    trend = trend.rename(columns={"total_vehicles": "Interval total"})
    fig = px.line(trend, x="timestamp", y=["Interval total", "3-interval moving average"],
                  labels={"timestamp": "Time", "value": "Vehicles", "variable": ""})
    st.plotly_chart(style_fig(fig, "Traffic trend with moving average"))
    if len(trend) >= 2:
        half = len(trend) // 2
        first_half, second_half = trend["Interval total"].iloc[:half].mean(), trend["Interval total"].iloc[half:].mean()
        st.caption(f"Average per interval: {first_half:.1f} in the first half of the period vs {second_half:.1f} in the second half.")


def page_analytics(counts, events):
    st.header("Traffic Analytics")
    if counts is None:
        st.info("Traffic-count data is not available, so analytics cannot be shown.")
        return
    times = [pd.Timestamp(t) for t in sorted(counts["timestamp"].unique())]
    all_directions = sorted(counts["Direction"].unique())
    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        directions = f1.multiselect("Directions", all_directions, default=all_directions, key="an_dirs")
        types = f2.multiselect("Vehicle types", COUNT_COLUMNS, default=COUNT_COLUMNS, key="an_types")
        start, end = time_range_filter(times, "an_range")
    fdf = counts[counts["Direction"].isin(directions) & counts["timestamp"].between(start, end)].copy()
    if fdf.empty or not types:
        st.info("No data matches the selected filters.")
        return
    fdf["selected_total"] = fdf[types].sum(axis=1)
    st.caption(f"{len(fdf)} rows selected. Totals below include only the selected vehicle types.")

    per_dir_time = fdf.groupby(["timestamp", "Direction"], as_index=False)["selected_total"].sum()
    per_interval = fdf.groupby("timestamp", as_index=False)["selected_total"].sum()

    left, right = st.columns(2)
    with left:
        fig = px.line(per_dir_time, x="timestamp", y="selected_total", color="Direction", markers=True,
                      labels={"timestamp": "Time", "selected_total": "Vehicles"})
        st.plotly_chart(style_fig(fig, "Traffic over time by direction"))
    with right:
        pivot = per_dir_time.pivot(index="Direction", columns="timestamp", values="selected_total")
        pivot.columns = [f"{t:%H:%M}" for t in pivot.columns]
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                        labels={"x": "Time", "y": "Direction", "color": "Vehicles"})
        st.plotly_chart(style_fig(fig, "Heatmap: direction by time"))

    left, right = st.columns(2)
    with left:
        mix = fdf.groupby("timestamp", as_index=False)[types].sum().melt(id_vars="timestamp", var_name="Vehicle type", value_name="Vehicles")
        fig = px.bar(mix, x="timestamp", y="Vehicles", color="Vehicle type", color_discrete_map=COLORS,
                     labels={"timestamp": "Time"})
        st.plotly_chart(style_fig(fig, "Vehicle mix over time"))
    with right:
        dir_type = fdf.groupby("Direction", as_index=False)[types].sum().melt(id_vars="Direction", var_name="Vehicle type", value_name="Vehicles")
        fig = px.bar(dir_type, x="Direction", y="Vehicles", color="Vehicle type", barmode="group", color_discrete_map=COLORS)
        st.plotly_chart(style_fig(fig, "Direction comparison by vehicle type"))

    section("Trend and change", "Both charts are calculated from the interval totals above.")
    left, right = st.columns(2)
    with left:
        trend = per_interval.copy()
        trend["3-interval moving average"] = trend["selected_total"].rolling(3, min_periods=3).mean()
        trend = trend.rename(columns={"selected_total": "Interval total"})
        fig = px.line(trend, x="timestamp", y=["Interval total", "3-interval moving average"],
                      labels={"timestamp": "Time", "value": "Vehicles", "variable": ""})
        st.plotly_chart(style_fig(fig, "Moving average trend"))
    with right:
        change = per_interval.copy()
        change["Change"] = change["selected_total"].diff()
        fig = px.bar(change.dropna(subset=["Change"]), x="timestamp", y="Change",
                     labels={"timestamp": "Time", "Change": "Change vs previous interval"})
        st.plotly_chart(style_fig(fig, "Change from the previous interval"))

    section("Per-direction statistics", "Calculated per direction across the selected intervals.")
    rows = []
    for direction, group in per_dir_time.groupby("Direction"):
        peak = group.loc[group["selected_total"].idxmax()]
        rows.append({"Direction": direction, "Total": int(group["selected_total"].sum()),
                     "Average per interval": round(group["selected_total"].mean(), 1),
                     "Lowest interval": int(group["selected_total"].min()),
                     "Peak interval": int(peak["selected_total"]), "Peak time": f"{peak['timestamp']:%d %b %H:%M}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    section("Events", "From the validated event records, within the selected time range.")
    if events is None:
        st.info("Event data is not available.")
        return
    ev = events[events["timestamp"].between(start, end)]
    if ev.empty:
        st.info("No events fall in the selected time range.")
        return
    left, right = st.columns(2)
    with left:
        by_type = ev["event_type"].value_counts().reset_index()
        by_type.columns = ["Event type", "Events"]
        fig = px.bar(by_type, x="Event type", y="Events", color="Event type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, "Events by type"))
    with right:
        totals = counts[counts["timestamp"].between(start, end)].groupby("timestamp", as_index=False)["total_vehicles"].sum()
        merged = ev.merge(totals, on="timestamp", how="left", suffixes=("", "_csv"))
        fig = px.line(totals, x="timestamp", y="total_vehicles", labels={"timestamp": "Time", "total_vehicles": "Vehicles"})
        marks = px.scatter(merged.dropna(subset=["total_vehicles"]), x="timestamp", y="total_vehicles", color="event_type",
                           hover_data=["event_id"])
        for trace in marks.data:
            fig.add_trace(trace)
        st.plotly_chart(style_fig(fig, "Events on the traffic line"))
    if "junction_id" in ev.columns and "Junction_ID" in counts.columns and set(ev["junction_id"]) & set(counts["Junction_ID"]):
        matched = int(merged["total_vehicles"].notna().sum())
        equal = int((merged["vehicle_count"] == merged["total_vehicles"]).sum())
        st.caption(f"{matched} of {len(ev)} events share a timestamp and junction with the counts; {equal} have the same vehicle total. "
                   "The simulated events and counts were designed to align, so this is not an observed link.")
    else:
        st.caption("Events and counts share no junction id, so they are not compared.")


def page_records(counts, events, images):
    st.header("Records")
    tab_counts, tab_events, tab_images = st.tabs(["Traffic counts", "Events", "Image metadata"])
    with tab_counts:
        if counts is None:
            st.info("Traffic-count data is not available.")
        else:
            times = [pd.Timestamp(t) for t in sorted(counts["timestamp"].unique())]
            all_directions = sorted(counts["Direction"].unique())
            c1, c2, c3 = st.columns(3)
            directions = c1.multiselect("Direction", all_directions, default=all_directions, key="rc_dirs")
            types = c2.multiselect("Vehicle-type columns to show", COUNT_COLUMNS, default=COUNT_COLUMNS, key="rc_types")
            query = c3.text_input("Search all columns", key="rc_q")
            start, end = time_range_filter(times, "rc_range")
            view = counts[counts["Direction"].isin(directions) & counts["timestamp"].between(start, end)]
            view = search_filter(view, query)
            wanted = ["timestamp", "Junction_ID", "Direction"] + types + ["total_vehicles", "is_simulated", "has_missing_count"]
            view = view[[c for c in wanted if c in view.columns]]
            st.caption(f"Showing {len(view)} of {len(counts)} rows.")
            st.dataframe(view, hide_index=True)
            st.download_button("Download these rows as CSV", view.to_csv(index=False).encode("utf-8"),
                               file_name="traffic_counts_filtered.csv", mime="text/csv", key="dl_counts")
    with tab_events:
        if events is None:
            st.info("Event data is not available.")
        else:
            e1, e2 = st.columns(2)
            kinds = sorted(events["event_type"].unique())
            chosen = e1.multiselect("Event type", kinds, default=kinds, key="re_kinds")
            query = e2.text_input("Search all columns", key="re_q")
            view = search_filter(events[events["event_type"].isin(chosen)], query)
            st.caption(f"Showing {len(view)} of {len(events)} events.")
            st.dataframe(view, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("Download as CSV", view.to_csv(index=False).encode("utf-8"),
                               file_name="traffic_events_filtered.csv", mime="text/csv", key="dl_events_csv")
            d2.download_button("Download as JSON", view.to_json(orient="records", indent=2, date_format="iso"),
                               file_name="traffic_events_filtered.json", mime="application/json", key="dl_events_json")
    with tab_images:
        if images is None:
            st.info("Image metadata is not available.")
        else:
            statuses = sorted(images["status"].unique())
            chosen = st.multiselect("Status", statuses, default=statuses, key="ri_status")
            query = st.text_input("Search all columns", key="ri_q")
            view = search_filter(images[images["status"].isin(chosen)], query)
            st.caption(f"Showing {len(view)} of {len(images)} images. These photos are not linked to the counts or events.")
            st.dataframe(view, hide_index=True)
            st.download_button("Download as CSV", view.to_csv(index=False).encode("utf-8"),
                               file_name="image_metadata_filtered.csv", mime="text/csv", key="dl_images")


def page_images(images):
    st.header("Traffic Images")
    st.info("These are the actual input photos with the metadata your pipeline recorded when it validated them. "
            "The photos are real but are not linked in time or place to the simulated counts and events.")
    if images is None:
        st.warning("Image metadata is not available.")
        return
    thumb_index = load_thumb_index()
    f1, f2 = st.columns(2)
    statuses = sorted(images["status"].unique())
    chosen = f1.multiselect("Status", statuses, default=statuses, key="ti_status")
    query = f2.text_input("Search filename", key="ti_q")
    view = images[images["status"].isin(chosen)]
    if query:
        view = view[view["filename"].str.contains(query, case=False, regex=False)]
    a, b, c = st.columns(3)
    metric_card(a, "Images shown", f"{len(view)} of {len(images)}")
    metric_card(b, "Total size", f"{images['size_bytes'].sum() / 1e6:.1f} MB")
    metric_card(c, "Image formats", ", ".join(sorted(images["format"].dropna().astype(str).unique())) or "n/a",
                help_text="File format detected by the pipeline's validation step.")
    st.subheader("Gallery")
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        st.warning("No image thumbnails were found in the 'images' folder of this app, so only the metadata is shown below. "
                   "Create the thumbnails with the helper cell in Colab and upload them as the 'images' folder.")
    for start in range(0, len(view), 4):
        for column, (_, row) in zip(st.columns(4), view.iloc[start:start + 4].iterrows()):
            with column:
                path = find_thumbnail(row["filename"], thumb_index)
                if path is not None:
                    st.image(str(path), caption=row["filename"])
                else:
                    st.caption(f"{row['filename']} (thumbnail not found)")
                st.caption(f"{int(row['width'])} x {int(row['height'])} px | {row['format']} | "
                           f"{row['size_bytes'] / 1e6:.1f} MB | {row['status']}")
                if thumb_index is not None and "metadata_datetime" in thumb_index.columns:
                    match = thumb_index[thumb_index["filename"] == row["filename"]]
                    if not match.empty and pd.notna(match.iloc[0]["metadata_datetime"]):
                        st.caption(f"File metadata date: {match.iloc[0]['metadata_datetime']}")
    if thumb_index is not None and "metadata_datetime" in thumb_index.columns:
        st.caption("'File metadata date' is the date stored inside the photo file. It may reflect editing time rather than capture time.")
    section("Detection results and per-image vehicle counts")
    st.info("Not available. The pipeline does not run vehicle detection, and it does not store a capture time or a vehicle count "
            "for each image, so none are shown. Showing them would require adding a real detection or manual-counting step to the pipeline.")


def stage_html(stages):
    boxes = []
    for i, (title, subtitle, lines) in enumerate(stages):
        body = "".join(f"<div style='font-size:0.85rem'>{line}</div>" for line in lines)
        boxes.append("<div style='flex:1;min-width:150px;border:1px solid rgba(128,128,128,0.4);border-radius:10px;"
                     "padding:12px;background:rgba(37,99,235,0.08)'>"
                     f"<div style='font-weight:700'>{title}</div>"
                     f"<div style='font-size:0.78rem;opacity:0.75;margin-bottom:6px'>{subtitle}</div>{body}</div>")
        if i < len(stages) - 1:
            boxes.append("<div style='align-self:center;font-size:1.4rem;padding:0 2px'>&#10132;</div>")
    return "<div style='display:flex;gap:6px;flex-wrap:wrap;align-items:stretch'>" + "".join(boxes) + "</div>"


def page_pipeline(counts, events, images, log):
    st.header("Pipeline")
    if not log.get("csv_before") and log.get("csv_before") != 0:
        st.info("The cleaning log is not available or could not be read, so stage counts cannot be shown.")
    ok_files = sum(1 for r in log["output_table"] if r["Status"] in ("OK", "SKIPPED (optional)"))
    st.markdown(stage_html([
        ("1. Raw sources", "Never modified", ["Images (JPG/PNG)", "Traffic-count CSV", "Traffic-event JSON"]),
        ("2. Extract", "Read and check inputs", [f"Images found: {fmt(log.get('images_found'))}",
                                                  f"CSV rows read: {fmt(log.get('csv_before'))}",
                                                  f"JSON records read: {fmt(log.get('json_before'))}"]),
        ("3. Transform", "Clean and validate", [f"CSV rows kept: {fmt(log.get('csv_after'))}",
                                                 f"JSON records kept: {fmt(log.get('json_after'))}",
                                                 f"Images valid: {fmt(log.get('images_ok'))}"]),
        ("4. Output", "Save and verify", [f"Files verified OK: {ok_files} of {len(log['output_table'])}" if log["output_table"] else "Files verified: n/a"]),
        ("5. Dashboard", "This app", [f"CSV rows shown: {fmt(len(counts)) if counts is not None else 'n/a'}",
                                      f"Events shown: {fmt(len(events)) if events is not None else 'n/a'}"]),
    ]), unsafe_allow_html=True)

    section("Input and output record counts", "Taken from the pipeline's cleaning log.")
    st.dataframe(pd.DataFrame([
        {"Dataset": "Traffic-count CSV", "Input records": log.get("csv_before"), "Exact duplicates removed": log.get("csv_exact"),
         "Other duplicates removed": log.get("csv_key"), "Invalid rejected": log.get("csv_rejected"), "Output records": log.get("csv_after")},
        {"Dataset": "Traffic-event JSON", "Input records": log.get("json_before"), "Exact duplicates removed": log.get("json_dupes"),
         "Other duplicates removed": None, "Invalid rejected": log.get("json_rejected"), "Output records": log.get("json_after")},
        {"Dataset": "Images", "Input records": log.get("images_found"), "Exact duplicates removed": None,
         "Other duplicates removed": None, "Invalid rejected": log.get("image_failures"), "Output records": log.get("images_ok")},
    ]).astype(object).where(lambda d: d.notna(), "n/a"), hide_index=True)

    section("Missing values and rejections", "How the transform stage handled problem data.")
    a, b, c = st.columns(3)
    metric_card(a, "Junction_ID filled as UNKNOWN", fmt(log.get("junction_filled")))
    metric_card(b, "Non-numeric counts set to missing", fmt(log.get("non_numeric")))
    metric_card(c, "Rows kept with a missing count", fmt(log.get("partial_missing")))
    st.caption(f"Missing values found per column: {log.get('missing_text', 'n/a')}")
    for name in ("csv", "json"):
        if log["reasons"][name]:
            st.write(f"Rejection reasons ({name.upper()}):")
            st.dataframe(pd.DataFrame(list(log["reasons"][name].items()), columns=["Reason", "Records"]), hide_index=True)

    if log["join_lines"]:
        section("Dataset relationships", "From the pipeline's join check. Nothing is merged by the pipeline.")
        for line in log["join_lines"]:
            st.write(line)

    section("Not recorded by the pipeline")
    st.info("Processing duration, per-stage run times and the last execution time are not available, because the pipeline does not "
            "write them to its outputs. They are not estimated here. Recording them would need a small addition to the pipeline.")


def status_checks(log, counts, events, images, load_errors):
    checks = []

    def add(name, result, detail):
        checks.append({"Check": name, "Result": result, "Detail": detail})

    raw = log.get("raw_unchanged")
    add("Raw files unchanged", "UNKNOWN" if raw is None else ("PASS" if raw == "YES" else "FAIL"), f"Log reports: {fmt(raw)}")
    arith = log.get("arithmetic")
    add("CSV record arithmetic (before - removed = after)", "UNKNOWN" if arith is None else ("PASS" if arith == "OK" else "FAIL"), f"Log reports: {fmt(arith)}")
    pairs = [("Dashboard CSV rows = pipeline output", counts, log.get("csv_after")),
             ("Dashboard events = pipeline output", events, log.get("json_after")),
             ("Dashboard images = images found", images, log.get("images_found"))]
    for name, frame, expected in pairs:
        if frame is None or expected is None:
            add(name, "UNKNOWN", "Data or log value not available")
        else:
            add(name, "PASS" if len(frame) == expected else "FAIL", f"Dashboard has {len(frame)}, log says {expected}")
    rejected = [log.get("csv_rejected"), log.get("json_rejected")]
    if None in rejected:
        add("Invalid records rejected", "UNKNOWN", "Log value not available")
    else:
        add("Invalid records rejected", "PASS" if sum(rejected) == 0 else "ATTENTION", f"CSV: {rejected[0]}, JSON: {rejected[1]}")
    failures = log.get("image_failures")
    add("Image validation failures", "UNKNOWN" if failures is None else ("PASS" if failures == 0 else "FAIL"), f"Failures: {fmt(failures)}")
    table = log["output_table"]
    if not table:
        add("Output files verified", "UNKNOWN", "No output verification table in the log")
    else:
        bad = [r["File"] for r in table if r["Status"] not in ("OK", "SKIPPED (optional)")]
        add("Output files verified", "PASS" if not bad else "FAIL", f"{len(table) - len(bad)} of {len(table)} OK" + (f"; problems: {bad}" if bad else ""))
    add("Dashboard data files readable", "PASS" if not load_errors else "FAIL", "All four files read" if not load_errors else "; ".join(load_errors))
    return checks


def page_status(counts, events, images, log_text, log, load_errors):
    st.header("Pipeline Status")
    checks = status_checks(log, counts, events, images, load_errors)
    results = [c["Result"] for c in checks]
    if "FAIL" in results:
        st.error("One or more checks FAILED. See the table below.")
    elif "ATTENTION" in results or "UNKNOWN" in results:
        st.warning("No failures, but some checks need attention or could not be evaluated.")
    else:
        st.success("All checks passed.")
    a, b, c, d = st.columns(4)
    metric_card(a, "Passed", results.count("PASS"))
    metric_card(b, "Failed", results.count("FAIL"))
    metric_card(c, "Attention", results.count("ATTENTION"))
    metric_card(d, "Unknown", results.count("UNKNOWN"))
    st.dataframe(pd.DataFrame(checks), hide_index=True)

    section("Output verification", "The table the pipeline's Output stage wrote to its log.")
    if log["output_table"]:
        st.dataframe(pd.DataFrame(log["output_table"]), hide_index=True)
    else:
        st.info("No output verification table found in the cleaning log.")

    section("Errors and rejected records")
    if load_errors:
        for message in load_errors:
            st.error(message)
    shown = False
    for name in ("csv", "json"):
        for reason, number in log["reasons"][name].items():
            st.write(f"{name.upper()}: {number} record(s) rejected - {reason}")
            shown = True
    if log.get("image_failures"):
        st.write(f"Images: {log['image_failures']} failed validation.")
        shown = True
    if not shown and not load_errors:
        st.success("No errors or rejected records were recorded in the log.")

    section("Last run information")
    st.info("The last execution time and processing duration are not recorded by the pipeline, so they are not shown.")
    if log_text:
        with st.expander("Full cleaning log"):
            st.text(log_text)


def coming_soon(name):
    st.header(name)
    st.info(f"The {name} page will be added in a later step.")


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

log = parse_log(log_text)
load_errors = [m for m in (counts_error, events_error, images_error, log_error) if m]

if page == "Overview":
    page_overview(counts, events, images)
elif page == "Pipeline":
    page_pipeline(counts, events, images, log)
elif page == "Analytics":
    page_analytics(counts, events)
elif page == "Records":
    page_records(counts, events, images)
elif page == "Traffic Images":
    page_images(images)
elif page == "Status":
    page_status(counts, events, images, log_text, log, load_errors)
else:
    coming_soon(page)
