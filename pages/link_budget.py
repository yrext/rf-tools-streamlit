import streamlit as st
import numpy as np
import plotly.graph_objects as go

class LinkBudgetCalculator:
    """Link Budget Calculator using the Friis Transmission Equation."""

    def run(self):
        # Set page config if not already set by parent
        try:
            st.set_page_config(page_title="Link Budget Calculator", layout="wide")
        except Exception:
            pass

        st.title("📡 Link Budget Calculator")

        # Top instructions
        st.info("""
        **Instructions**
        - Adjust the **Transmitter**, **Receiver**, and **Environmental** parameters below.
        - The calculator uses the Friis Transmission Equation for Free Space Path Loss (FSPL).
        - **Link Margin** = (Received Power) - (Receiver Sensitivity) - (Fade Margin).
        - A positive Link Margin indicates a viable connection.
        """)

        # --- Inputs (Main Page) ---
        col_tx, col_env, col_rx = st.columns(3)

        with col_tx:
            st.markdown("#### 📤 Transmitter")
            tx_pwr = st.number_input("Transmit Power (dBm)", value=20.0, step=1.0, help="Power at the output of the transmitter")
            tx_loss = st.number_input("TX Cable Loss (dB)", value=1.0, min_value=0.0, step=0.1, help="Losses between transmitter and antenna")
            tx_gain = st.number_input("TX Antenna Gain (dBi)", value=5.0, step=0.5, help="Gain of the transmitting antenna")

        with col_env:
            st.markdown("#### 🌍 Environment")
            freq_mhz = st.number_input("Frequency (MHz)", value=2400.0, min_value=1.0, step=10.0)
            distance = st.number_input("Distance (m)", value=500.0, min_value=0.1, step=10.0)
            fade_margin = st.number_input("Fade Margin (dB)", value=15.0, min_value=0.0, step=1.0, help="Safety buffer for fading/obstructions")

        with col_rx:
            st.markdown("#### 📥 Receiver")
            rx_gain = st.number_input("RX Antenna Gain (dBi)", value=5.0, step=0.5)
            rx_loss = st.number_input("RX Cable Loss (dB)", value=1.0, min_value=0.0, step=0.1)
            rx_sens = st.number_input("RX Sensitivity (dBm)", value=-90.0, step=1.0, help="Minimum power required by the receiver")

        # --- Calculations ---
        c = 299792458.0  # Speed of light in m/s
        freq_hz = freq_mhz * 1e6
        
        # Free Space Path Loss (FSPL) in dB
        if distance > 0:
            fspl = 20 * np.log10(distance) + 20 * np.log10(freq_hz) + 20 * np.log10(4 * np.pi / c)
        else:
            fspl = 0

        # Link Budget Equation
        eirp = tx_pwr - tx_loss + tx_gain
        rx_pwr = eirp - fspl + rx_gain - rx_loss
        link_margin = rx_pwr - rx_sens - fade_margin

        # --- Results Display ---
        st.markdown("---")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        
        res_col1.metric("EIRP", f"{eirp:.2f} dBm")
        res_col2.metric("Path Loss (FSPL)", f"{fspl:.2f} dB")
        res_col3.metric("Received Power", f"{rx_pwr:.2f} dBm")
        
        margin_color = "normal" if link_margin >= 0 else "inverse"
        res_col4.metric("Link Margin", f"{link_margin:.2f} dB", delta=f"{link_margin:.2f} dB", delta_color=margin_color)

        if link_margin < 0:
            st.error(f"⚠️ **Link Budget Failed**: The signal is {abs(link_margin):.2f} dB below the required threshold.")
        else:
            st.success(f"✅ **Link Budget OK**: You have {link_margin:.2f} dB of headroom.")

        # --- Visualization: Path Loss Chart ---
        st.markdown("### 📈 Path Loss Visualization")
        dist_scale = st.radio("Distance Axis Scale", ["Logarithmic", "Linear"], horizontal=True)
        
        # Generate distance points for curve (1m to 5x distance or 1km)
        d_max = max(distance * 5, 1000)
        d_min = 1.0
        if dist_scale == "Logarithmic":
            d_axis = np.logspace(np.log10(d_min), np.log10(d_max), 500)
        else:
            d_axis = np.linspace(d_min, d_max, 500)
        
        # Calculate power vs distance curve
        fspl_axis = 20 * np.log10(d_axis) + 20 * np.log10(freq_hz) + 20 * np.log10(4 * np.pi / c)
        rx_pwr_axis = eirp - fspl_axis + rx_gain - rx_loss

        fig = go.Figure()

        # Received Power Curve
        fig.add_trace(go.Scatter(
            x=d_axis, y=rx_pwr_axis,
            name="Received Power (dBm)",
            line=dict(color="#0891b2", width=3),
            hovertemplate="Distance: %{x:.1f} m<br>Power: %{y:.2f} dBm<extra></extra>"
        ))

        # Sensitivity Threshold
        fig.add_hline(y=rx_sens, line_dash="dash", line_color="#ef4444", 
                      annotation_text="Sensitivity", annotation_position="bottom right")
        
        # Sensitivity + Margin Threshold
        fig.add_hline(y=rx_sens + fade_margin, line_dash="dot", line_color="#f97316",
                      annotation_text="Required with Margin", annotation_position="top right")

        # Current Operating Point
        fig.add_trace(go.Scatter(
            x=[distance], y=[rx_pwr],
            mode="markers",
            marker=dict(size=12, color="white", line=dict(width=2, color="#0891b2")),
            name="Current Point",
            hovertemplate="Target Distance: %{x} m<br>Actual Rx: %{y:.2f} dBm<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title=f"Distance (meters) - {dist_scale}",
            yaxis_title="Received Power (dBm)",
            xaxis_type="log" if dist_scale == "Logarithmic" else "linear",
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    LinkBudgetCalculator().run()