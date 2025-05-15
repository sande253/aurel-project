import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import os

st.set_page_config(page_title="Demand Forecast", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container { padding: 2rem; }
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    .st-bb { background-color: transparent; }
    .st-at { background-color: #f0f2f6; }
    .st-ae { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🔮 Automated Demand Forecasting")
    st.markdown("Upload your cleaned data CSV to generate Prophet forecasts")

    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read and process data
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates(subset=["date", "keyword"]).reset_index(drop=True)

            # Keyword selection
            keywords = df['keyword'].unique().tolist()
            selected_keywords = st.multiselect("Select keywords to forecast", options=keywords)

            if selected_keywords:
                st.success(f"Selected {len(selected_keywords)} keywords for forecasting")
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, kw in enumerate(selected_keywords):
                    progress = (i + 1) / len(selected_keywords)
                    progress_bar.progress(progress)
                    
                    with st.status(f"Processing: {kw}", expanded=True) as status:
                        try:
                            # Filter data
                            keyword_df = df[df['keyword'] == kw]
                            prophet_df = keyword_df[['date', 'value']].rename(
                                columns={'date': 'ds', 'value': 'y'}
                            )

                            # Data validation
                            if len(prophet_df) < 10:
                                st.warning(f"Skipped {kw}: Need at least 10 data points")
                                continue

                            # Modeling
                            model = Prophet()
                            model.fit(prophet_df)
                            future = model.make_future_dataframe(periods=30)
                            forecast = model.predict(future)

                            # Plotting
                            st.subheader(f"Forecast for: {kw}")
                            fig = plot_plotly(model, forecast)
                            fig.update_layout(
                                width=1000,
                                height=600,
                                xaxis_title="Date",
                                yaxis_title="Search Interest",
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Show forecast components
                            st.markdown("**Forecast Components**")
                            components_fig = model.plot_components(forecast)
                            st.pyplot(components_fig)

                            status.update(label=f"Completed: {kw}", state="complete", expanded=False)
                            
                        except Exception as e:
                            st.error(f"Error processing {kw}: {str(e)}")
                            status.update(label=f"Failed: {kw}", state="error")
                            continue

                progress_bar.empty()
                status_text.success("Forecast complete!")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
