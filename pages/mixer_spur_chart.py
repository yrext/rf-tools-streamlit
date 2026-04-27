import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Constants ---
POPULAR_BANDS = {
    "Custom": [0, 0], # Placeholder for custom input
    # 3GPP Europe
    "3GPP n1 (Europe 2100)": [1920, 2170],
    "3GPP n3 (Europe 1800)": [1710, 1880],
    "3GPP n7 (Europe 2600)": [2500, 2690],
    "3GPP n8 (Europe 900)": [880, 960],
    "3GPP n20 (Europe 800)": [791, 862],
    "3GPP n28 (Europe 700)": [703, 803],
    # 3GPP US
    "3GPP n2 (US 1900)": [1850, 1990],
    "3GPP n5 (US 850)": [824, 894],
    "3GPP n12 (US 700)": [699, 746],
    "3GPP n30 (US 2300)": [2305, 2360],
    "3GPP n41 (US 2.5G)": [2496, 2690],
    "3GPP n66 (US AWS)": [1710, 2200],
    "3GPP n71 (US 600)": [617, 698],
    # 3GPP Global / TDD
    "3GPP n77 (C-Band)": [3300, 4200],
    "3GPP n78 (3.5GHz)": [3300, 3800],
    # Connectivity
    "WiFi 2.4GHz": [2400, 2483.5],
    "WiFi 5GHz": [5150, 5850],
    "WiFi 6GHz": [5925, 7125],
    "Bluetooth": [2400, 2483.5],
    "ISM 915MHz (US)": [902, 928],
    "ISM 868MHz (EU)": [863, 870],
    # GNSS
    "GNSS L1": [1559, 1610], # MHz
    "GNSS L2": [1215, 1240], # MHz
    "GNSS L5": [1164, 1189], # MHz
    # Aerospace
    "Aerospace L-Band": [960, 1215],
    "Aerospace S-Band": [2300, 2500],
    "Aerospace C-Band": [5030, 5091],
}

class MixerSpurChart:
    def __init__(self):
        pass

    def run(self):
        st.title("📊 Mixer Spur Chart Generator")

        st.info("""
        **Quick Guide**
        - Define the **RF**, **LO**, and **Desired IF** frequency ranges for your mixer.
        - Select a **Target Band** where unwanted spurs should be avoided.
        - The chart will display lines for mixer spurs (M·RF ± N·LO) that fall within the selected Target Band.
        - The primary IF output (1·LO ± 1·RF) is also highlighted.
        """)

        # --- Input Configuration ---
        st.markdown("##### Input Frequencies (MHz)")
        col_rf, col_lo, col_if = st.columns(3)
        with col_rf:
            st.subheader("RF Input")
            rf_min = st.number_input("RF Min (MHz)", value=2000.0, min_value=0.1, step=10.0, format="%.1f", key="rf_min")
            rf_max = st.number_input("RF Max (MHz)", value=2200.0, min_value=rf_min + 0.1, step=10.0, format="%.1f", key="rf_max")
        with col_lo:
            st.subheader("LO Input")
            lo_min = st.number_input("LO Min (MHz)", value=2300.0, min_value=0.1, step=10.0, format="%.1f", key="lo_min")
            lo_max = st.number_input("LO Max (MHz)", value=2500.0, min_value=lo_min + 0.1, step=10.0, format="%.1f", key="lo_max")
        with col_if:
            st.subheader("Desired IF Output")
            if_min = st.number_input("IF Min (MHz)", value=200.0, min_value=0.1, step=10.0, format="%.1f", key="if_min")
            if_max = st.number_input("IF Max (MHz)", value=400.0, min_value=if_min + 0.1, step=10.0, format="%.1f", key="if_max")

        st.markdown("---")
        st.markdown("##### Spur Configuration")
        col_target_band, col_order = st.columns(2)
        with col_target_band:
            target_band_name = st.selectbox("Target Band for Unwanted Spurs", list(POPULAR_BANDS.keys()), index=0)
            if target_band_name == "Custom":
                custom_target_min = st.number_input("Custom Target Min (MHz)", value=100.0, min_value=0.1, step=10.0, format="%.1f", key="custom_target_min")
                custom_target_max = st.number_input("Custom Target Max (MHz)", value=500.0, min_value=custom_target_min + 0.1, step=10.0, format="%.1f", key="custom_target_max")
                target_band_min, target_band_max = custom_target_min, custom_target_max
            else:
                target_band_min, target_band_max = POPULAR_BANDS[target_band_name]
                st.write(f"Selected band: {target_band_min} MHz to {target_band_max} MHz")

        with col_order:
            max_order = st.slider("Max Spur Order (M, N)", min_value=1, max_value=10, value=3, step=1)

        st.markdown("---")

        # --- Generate Chart ---
        self._render_spur_chart(rf_min, rf_max, lo_min, lo_max, if_min, if_max, target_band_min, target_band_max, max_order)

    def _render_spur_chart(self, rf_min, rf_max, lo_min, lo_max, if_min, if_max, target_band_min, target_band_max, max_order):
        fig = go.Figure()

        # Define the RF and LO ranges for plotting
        rf_plot_range = np.linspace(rf_min, rf_max, 200) # Increased points for smoother lines

        # Add operating region rectangle
        fig.add_shape(
            type="rect",
            x0=rf_min, y0=lo_min, x1=rf_max, y1=lo_max,
            line=dict(color="RoyalBlue", width=2),
            fillcolor="LightSkyBlue",
            opacity=0.2,
            name="Operating Region"
        )
        # Add a dummy trace for the legend entry for the operating region
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="LightSkyBlue", width=10), # Make it visible in legend
            name="Operating Region",
            hoverinfo="skip"
        ))


        # Plot desired IF lines (1*LO - 1*RF = IF, 1*RF - 1*LO = IF, 1*LO + 1*RF = IF)
        # These are the fundamental mixing products (M=1, N=1)

        # Desired IF (LO - RF = IF) -> f_LO = f_RF + IF
        fig.add_trace(go.Scatter(
            x=rf_plot_range,
            y=rf_plot_range + if_min,
            mode='lines',
            name=f'Desired IF (LO-RF) = {if_min} MHz',
            line=dict(color='cyan', width=3, dash='solid'),
            hoverinfo='skip',
            legendgroup="Fundamental"
        ))
        fig.add_trace(go.Scatter(
            x=rf_plot_range,
            y=rf_plot_range + if_max,
            mode='lines',
            name=f'Desired IF (LO-RF) = {if_max} MHz',
            line=dict(color='cyan', width=3, dash='solid'),
            hoverinfo='skip',
            legendgroup="Fundamental"
        ))
        # Desired IF (RF - LO = IF) -> f_LO = f_RF - IF
        fig.add_trace(go.Scatter(
            x=rf_plot_range,
            y=rf_plot_range - if_min,
            mode='lines',
            name=f'Desired IF (RF-LO) = {if_min} MHz',
            line=dict(color='deepskyblue', width=3, dash='solid'),
            hoverinfo='skip',
            legendgroup="Fundamental"
        ))
        fig.add_trace(go.Scatter(
            x=rf_plot_range,
            y=rf_plot_range - if_max,
            mode='lines',
            name=f'Desired IF (RF-LO) = {if_max} MHz',
            line=dict(color='deepskyblue', width=3, dash='solid'),
            hoverinfo='skip',
            legendgroup="Fundamental"
        ))
        # Desired IF (RF + LO = IF) -> f_LO = IF - f_RF
        fig.add_trace(go.Scatter(
            x=rf_plot_range,
            y=if_min - rf_plot_range,
            mode='lines',
            name=f'Desired IF (RF+LO) = {if_min} MHz',
            line=dict(color='magenta', width=3, dash='solid'),
            hoverinfo='skip',
            legendgroup="Fundamental"
        ))
        fig.add_trace(go.Scatter(
            x=rf_plot_range,
            y=if_max - rf_plot_range,
            mode='lines',
            name=f'Desired IF (RF+LO) = {if_max} MHz',
            line=dict(color='magenta', width=3, dash='solid'),
            hoverinfo='skip',
            legendgroup="Fundamental"
        ))


        # Plot unwanted spur lines for each M, N combination
        # M and N are absolute orders, so iterate from 1
        spur_colors = ['#EF553B', '#FFA15A', '#FECB52', '#00CC96', '#19D3F3', '#636EFA', '#B6E880', '#FF97FF', '#FF6692', '#AB63FA']
        spur_color_idx = 0

        for m in range(1, max_order + 1):
            for n in range(1, max_order + 1):
                # Skip the fundamental IF products (M=1, N=1) as they are handled above
                if (m == 1 and n == 1):
                    continue

                # Cycle through spur_colors for unwanted spurs
                current_color = spur_colors[spur_color_idx % len(spur_colors)]
                line_dash_style = 'solid' if (m + n) % 2 == 0 else 'dash' # Alternate dash style for clarity

                # Spur product: M*f_RF + N*f_LO = F_target
                # f_LO = (F_target - M*f_RF) / N
                if n != 0:
                    for target_f_edge in [target_band_min, target_band_max]:
                        y_vals = (target_f_edge - m * rf_plot_range) / n
                        # Only plot if y_vals are within LO range and positive
                        valid_indices = (y_vals >= lo_min) & (y_vals <= lo_max) & (y_vals > 0)
                        if np.any(valid_indices):
                            fig.add_trace(go.Scatter(
                                x=rf_plot_range[valid_indices],
                                y=y_vals[valid_indices],
                                mode='lines',
                                name=f'Spur {m}RF + {n}LO',
                                line=dict(color=current_color, width=1.5, dash=line_dash_style),
                                hoverinfo='text',
                                text=[f'M={m}, N={n}<br>Spur: {m}RF + {n}LO<br>Target: {target_f_edge} MHz' for _ in rf_plot_range[valid_indices]], # No legendgroup for individual spurs
                                showlegend=False # Hide individual spur traces from legend to avoid clutter
                            ))
                
                # Spur product: |M*f_RF - N*f_LO| = F_target
                # This splits into two cases: M*f_RF - N*f_LO = F_target  AND  N*f_LO - M*f_RF = F_target

                # Case A: M*f_RF - N*f_LO = F_target  => f_LO = (M*f_RF - F_target) / N
                if n != 0:
                    for target_f_edge in [target_band_min, target_band_max]:
                        y_vals = (m * rf_plot_range - target_f_edge) / n
                        valid_indices = (y_vals >= lo_min) & (y_vals <= lo_max) & (y_vals > 0)
                        if np.any(valid_indices):
                            fig.add_trace(go.Scatter(
                                x=rf_plot_range[valid_indices],
                                y=y_vals[valid_indices],
                                mode='lines',
                                name=f'Spur {m}RF - {n}LO',
                                line=dict(color=current_color, width=1.5, dash=line_dash_style),
                                hoverinfo='text',
                                text=[f'M={m}, N={n}<br>Spur: {m}RF - {n}LO<br>Target: {target_f_edge} MHz' for _ in rf_plot_range[valid_indices]], # No legendgroup for individual spurs
                                showlegend=False # Hide individual spur traces from legend to avoid clutter
                            ))
                
                # Case B: N*f_LO - M*f_RF = F_target  => f_LO = (F_target + M*f_RF) / N
                if n != 0:
                    for target_f_edge in [target_band_min, target_band_max]:
                        y_vals = (target_f_edge + m * rf_plot_range) / n
                        valid_indices = (y_vals >= lo_min) & (y_vals <= lo_max) & (y_vals > 0)
                        if np.any(valid_indices):
                            fig.add_trace(go.Scatter(
                                x=rf_plot_range[valid_indices],
                                y=y_vals[valid_indices],
                                mode='lines',
                                name=f'Spur {n}LO - {m}RF',
                                line=dict(color=current_color, width=1.5, dash=line_dash_style),
                                hoverinfo='text',
                                text=[f'M={m}, N={n}<br>Spur: {n}LO - {m}RF<br>Target: {target_f_edge} MHz' for _ in rf_plot_range[valid_indices]], # No legendgroup for individual spurs
                                showlegend=False # Hide individual spur traces from legend to avoid clutter
                            ))
                # Add a dummy trace for the legend entry for the spur order
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=current_color, width=2, dash=line_dash_style),
                                         name=f'Spur Order {m+n}', legendgroup=f"Order {m+n}"))
                spur_color_idx += 1


        fig.update_layout(
            title="Mixer Spur Chart (RF vs LO)",
            xaxis_title="RF Frequency (MHz)",
            yaxis_title="LO Frequency (MHz)",
            xaxis_range=[rf_min, rf_max],
            yaxis_range=[lo_min, lo_max],
            hovermode="closest",
            template="plotly_dark",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)) # Smaller legend font
        )
        st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    MixerSpurChart().run()