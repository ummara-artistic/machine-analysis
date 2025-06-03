import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from collections import Counter
import plotly.graph_objects as go
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
import streamlit as st

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="🎨 Machine Analysis", layout="wide")
# --- Load JSON Data ---
@st.cache_data
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# --- Parse Record ---
def parse_record(record):
    try:
        rd = record.get("Report Detail", {})
        params = record.get("Parameters", {})
        ld = record.get("Lot Detail", {})
        lot_info = record.get("Lot Info", {})


        creation_date = datetime.strptime(lot_info.get("Creation Date", ""), "%d-%b-%Y %H:%M:%S") \
            if lot_info.get("Creation Date") else None

        moisture_vals = list(map(float, rd.get("Moisture", "0,0,0").split(',')))
        moisture_mean = np.mean(moisture_vals)
        moisture_std = np.std(moisture_vals)

        front_left = np.mean(params.get("Front Tension Left", [0]))
        front_right = np.mean(params.get("Front Tension Right", [0]))
        rare_left = np.mean(params.get("Rare Tension Left", [0]))
        rare_right = np.mean(params.get("Rare Tension Right", [0]))

        tension_avg = np.mean([front_left, front_right, rare_left, rare_right]) # the check for yarn is it too lose or too tight
        tension_imbalance = abs(front_left + rare_left - front_right - rare_right) #absolute value

        actual_len = rd.get("Actual Length", 0) #getting exact value
        plan_len = ld.get("Plan Length", 1)  #getting exact value
        overrun_pct = ((actual_len - plan_len) / plan_len) * 100 if plan_len else 0 #plan to get overrun percentage ,
        #in textile we use over run to check for production of more material which is required or already planned of 

        per_creel_balls = rd.get("Per Creel Balls", 0)
        cone_weight = ld.get("Cone Weight", 0)
        total_yarn_used = cone_weight * per_creel_balls # yarn is like , creel is a machine frame that holds multiple yarn process during the warping process , preparing for weaving

        return {
            "Creation_Date": creation_date,
            "Moisture_Mean": moisture_mean,
            "Moisture_Std": moisture_std,
            "Front_Left_Tension": front_left,
            "Front_Right_Tension": front_right,
            "Rare_Left_Tension": rare_left,
            "Rare_Right_Tension": rare_right,
            "Tension_Avg": tension_avg,
            "Tension_Imbalance": tension_imbalance,
            "Overrun_Pct": overrun_pct,
            "Per_Creel_Balls": per_creel_balls,
            "Cone_Weight": cone_weight,
            "Total_Yarn_Used": total_yarn_used,
            "Humidity": rd.get("Dept Humidity", 0),
            "Tolerance": rd.get("Tolerance", ""),
            "Jog_Tension": params.get("Jog Tension", 0),
            "Speed_MMin": rd.get("Speed (M/Min)", 0),
            "Supplier": lot_info.get("Supplier", "Unknown"),
            "Batch_Number": lot_info.get("Batch Number", "Unknown"),
            "Machine_No": lot_info.get("Machine No", -1),
            "Remarks": lot_info.get("Remarks", "")
        }

    except Exception as e:
        st.error(f"Error parsing record: {e}")
        return {}

# --- Load & Process ---
file_path = os.path.join(os.getcwd(), "D:\ball_warping\all_machine_records.json")

if not os.path.exists(file_path):
    st.error("File not found: D:\ball_warping\all_machine_records.json")
    st.stop()
    
data = load_data(file_path)
df = pd.DataFrame([parse_record(r) for r in data])


df.dropna(subset=["Creation_Date"], inplace=True)
df.sort_values("Creation_Date", inplace=True)

# --- UI: Title ---
st.title("📈 Machine Quality & Performance Dashboard")

# --- Machine Selection ---
machine_input = st.sidebar.text_input("Enter Machine No (e.g. 6):", placeholder="Type machine number...")

#Work for remarks based on list of remarks 

# Fill NaN remarks
df["Remarks"] = df["Remarks"].fillna("").astype(str)

# Function to label remarks automatically
def auto_label_remark(remark):
    r = remark.lower()
    if any(kw in r for kw in ["ok to proceed", "no issue", "without incident", "all good", "smooth", "no problem", "normal", "proceed"]):
        return 0  # Positive
    elif any(kw in r for kw in ["fluctuated", "slightly", "monitor", "observe", "humidity", "check", "review"]):
        return 1  # Neutral
    elif any(kw in r for kw in ["inspection required", "irregular", "error", "fault", "repair", "maintenance", "stopped", "not working"]):
        return 2  # Negative
    else:
        return None

# Build training data from common remarks
top_n = 30
remark_counts = Counter(df["Remarks"])
most_common_remarks = [r for r, _ in remark_counts.most_common(top_n)]

labeled_remarks = [(r, auto_label_remark(r)) for r in most_common_remarks]
labeled_remarks = [(r, label) for r, label in labeled_remarks if label is not None]

# Train model if enough data
if len(labeled_remarks) >= 5:
    X_train, y_train = zip(*labeled_remarks)
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))
    model.fit(X_train, y_train)
    model_ready = True
else:
    model_ready = False

if machine_input:
    try:
        machine_no = int(machine_input)
        
        df_mach = df[df["Machine_No"] == machine_no]

        if df_mach.empty:
            st.warning(f"No data found for machine number {machine_no}.") # if there is no machine no found, no data will be appear
        else:
            latest = df_mach.iloc[-1]
            st.header(f"🧵 Machine {machine_no} Summary ({len(df_mach)} records)")
            st.write(f"Total Yarn Used: {latest['Total_Yarn_Used']:.2f}")
            st.write(f"Tension Imbalance: {latest['Tension_Imbalance']:.2f}")
            st.write(f"Overrun %: {latest['Overrun_Pct']:.2f}") 

            batch_df = df[df["Batch_Number"] == latest["Batch_Number"]]
            if len(batch_df) > 1:
                st.info(f"Batch has {len(batch_df)} lots. Avg Moisture: {batch_df['Moisture_Mean'].mean():.2f}")

            # Remarks and ML prediction
            remarks_list = df_mach["Remarks"].dropna().astype(str).tolist()
            if remarks_list and model_ready:
                combined_remarks = " ".join(remarks_list)
                prediction = model.predict([combined_remarks])[0]

                if prediction == 2:
                    st.error("❌ **Machine Alert:** Potential issue detected. Immediate inspection or maintenance is advised.")
                elif prediction == 1:
                    st.warning("⚠️ **Machine Status:** Minor fluctuations or irregularities. Recommend monitoring.")
                else:
                    st.success("✅ **Machine Status:** Operating within normal parameters. No action needed.")
            else:
                st.info("No remarks available or model not ready.")

            # --- Environment ---
            humidity = latest["Humidity"]
            st.write(f"Humidity: {humidity}%")
            if not (50 <= humidity <= 60):
                st.warning("⚠️ Humidity out of optimal range (50-60%)")

            df_mach_sorted = df_mach.sort_values("Creation_Date")
            X = df_mach_sorted["Creation_Date"].map(datetime.toordinal).values.reshape(-1, 1)
            y_moisture = df_mach_sorted["Moisture_Mean"].values
            y_tension = df_mach_sorted["Tension_Imbalance"].values
            y_overrun = df_mach_sorted["Overrun_Pct"].values
            y_speed = df_mach_sorted["Speed_MMin"].values

            if len(X) > 2:
                model_moisture = LinearRegression().fit(X, y_moisture)
                model_tension = LinearRegression().fit(X, y_tension)
                model_overrun = LinearRegression().fit(X, y_overrun)
                model_speed = LinearRegression().fit(X, y_speed)

                next_date = df_mach_sorted["Creation_Date"].max() + pd.Timedelta(days=1)
                st.subheader("🔮 Next Day Forecast")
                pred_moisture = model_moisture.predict([[next_date.toordinal()]])
                pred_tension = model_tension.predict([[next_date.toordinal()]])
                pred_overrun = model_overrun.predict([[next_date.toordinal()]])
                pred_speed = model_speed.predict([[next_date.toordinal()]])

                st.success(f"Predicted Moisture for {next_date.date()}: {pred_moisture[0]:.2f}")
                if pred_tension[0] > 10:
                    st.warning(f"⚠️ Predicted Tension Imbalance for {next_date.date()} is high: {pred_tension[0]:.2f}")
                if pred_overrun[0] > 100:
                    st.warning(f"⚠️ Predicted Overrun % for {next_date.date()} is unusually high: {pred_overrun[0]:.2f}")
                st.write(f"Predicted Speed (M/Min) for {next_date.date()}: {pred_speed[0]:.2f}")

                # Forecast for 6 months
                st.subheader("📈 6-Month Forecasts")
                future_dates = pd.date_range(start=next_date, periods=180, freq='D')
                future_X = future_dates.map(datetime.toordinal).values.reshape(-1, 1)

                future_moisture = model_moisture.predict(future_X)
                future_tension = model_tension.predict(future_X)
                future_overrun = model_overrun.predict(future_X)
                future_speed = model_speed.predict(future_X)

                # Graphs in Two Columns
                col1, col2 = st.columns(2)

                with col1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_mach_sorted["Creation_Date"], y=y_moisture,
                                              mode='lines+markers', name='Actual Moisture'))
                    fig1.add_trace(go.Scatter(x=future_dates, y=future_moisture,
                                              mode='lines', name='Forecast Moisture', line=dict(color='red')))
                    fig1.update_layout(title="Moisture Forecast", xaxis_title="Date", yaxis_title="Moisture")
                    st.plotly_chart(fig1)

                with col2:
                    fig2 = plt.figure(figsize=(8, 4))
                    plt.plot(df_mach_sorted["Creation_Date"], y_tension, marker='o', label="Actual Tension")
                    plt.plot(future_dates, future_tension, linestyle='--', color='orange', label="Forecast Tension")
                    plt.title("Tension Imbalance Forecast")
                    plt.xlabel("Date")
                    plt.ylabel("Tension Imbalance")
                    plt.legend()
                    st.pyplot(fig2)

                col3, col4 = st.columns(2)
                with col3:
                    plt.figure(figsize=(8, 4))
                    sns.lineplot(x=df_mach_sorted["Creation_Date"], y=y_overrun, label="Actual Overrun")
                    sns.lineplot(x=future_dates, y=future_overrun, label="Forecast Overrun", linestyle='--', color="green")
                    plt.title("Overrun % Forecast")
                    plt.xlabel("Date")
                    plt.ylabel("Overrun %")
                    plt.legend()
                    st.pyplot(plt.gcf())

                with col4:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=df_mach_sorted["Creation_Date"], y=y_speed,
                                              fill='tozeroy', mode='none', name='Actual Speed', fillcolor='lightblue'))
                    fig3.add_trace(go.Scatter(x=future_dates, y=future_speed,
                                              fill='tonexty', mode='none', name='Forecast Speed', fillcolor='lavender'))
                    fig3.update_layout(title="Speed Forecast (Area)", xaxis_title="Date", yaxis_title="Speed")
                    st.plotly_chart(fig3)
            else:
                st.info("Not enough data for predictions.")

            # --- Quality Section ---
            col5, col6 = st.columns(2)
            with col5:
                st.write(f"Mean Moisture: {latest['Moisture_Mean']:.2f}")
                st.write(f"Moisture Std Dev: {latest['Moisture_Std']:.2f}")
                if not 5.5 <= latest['Moisture_Mean'] <= 7.0:
                    st.error("⚠️ Moisture level out of acceptable range (5.5 - 7.0)")
                fig4 = go.Figure()
                fig4.add_trace(go.Violin(y=df_mach["Moisture_Mean"], box_visible=True, meanline_visible=True, name="Moisture"))
                fig4.update_layout(title="Moisture Distribution (Violin Plot)")
                st.plotly_chart(fig4)

            with col6:
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(x=df_mach_sorted["Creation_Date"], y=df_mach_sorted["Moisture_Mean"], name="Moisture"))
                if len(X) > 2:
                    pred_line = model_moisture.predict(X)
                    fig5.add_trace(go.Scatter(x=df_mach_sorted["Creation_Date"], y=pred_line, name="Predicted", mode='lines'))
                fig5.update_layout(title="Moisture Trend (Bar + Prediction)", barmode='overlay')
                st.plotly_chart(fig5)

            # --- Tension Section ---
            col7, col8 = st.columns(2)
            with col7:
                if latest['Tension_Imbalance'] > 10:
                    st.warning("⚠️ Significant tension imbalance!")
                fig6 = go.Figure()
                fig6.add_trace(go.Bar(name='Front Left Tension', x=df_mach_sorted["Creation_Date"], y=df_mach_sorted["Front_Left_Tension"]))
                fig6.add_trace(go.Bar(name='Front Right Tension', x=df_mach_sorted["Creation_Date"], y=df_mach_sorted["Front_Right_Tension"]))
                fig6.update_layout(barmode='group', title="Front Tension Comparison")
                st.plotly_chart(fig6)

            with col8:
                fig7 = go.Figure()
                fig7.add_trace(go.Scatter(x=df_mach_sorted["Creation_Date"], y=df_mach_sorted["Tension_Imbalance"],
                                          mode='lines', name="Actual", line_shape='hv'))
                if len(X) > 2:
                    pred_tension_line = model_tension.predict(X)
                    fig7.add_trace(go.Scatter(x=df_mach_sorted["Creation_Date"], y=pred_tension_line,
                                              mode='lines', name="Predicted", line_shape='hv', line=dict(dash='dot')))
                fig7.update_layout(title="Tension Imbalance Trend (Step)", xaxis_title="Date")
                st.plotly_chart(fig7)

            # --- Performance Section ---
            st.markdown("## 🏁 Performance Metrics")
            col9, col10 = st.columns(2)
            with col9:
                fig8 = go.Figure()
                fig8.add_trace(go.Scatter(x=df_mach_sorted["Creation_Date"], y=df_mach_sorted["Overrun_Pct"],
                                          mode='markers', marker=dict(size=8), name='Overrun'))
                if len(X) > 2:
                    pred_overrun_line = model_overrun.predict(X)
                    fig8.add_trace(go.Scatter(x=df_mach_sorted["Creation_Date"], y=pred_overrun_line,
                                              mode='lines', name='Predicted'))
                fig8.update_layout(title="Overrun % Trend (Dots + Line)")
                st.plotly_chart(fig8)

            with col10:
                st.write(f"Speed (M/Min): {latest['Speed_MMin']}")
                fig9 = go.Figure()
                fig9.add_trace(go.Histogram(x=df_mach_sorted["Speed_MMin"], nbinsx=20, name="Speed Distribution"))
                fig9.update_layout(title="Speed M/Min Distribution", xaxis_title="Speed (M/Min)", yaxis_title="Count")
                st.plotly_chart(fig9)

    except ValueError:
        st.error("Please enter a valid integer for Machine No.")
