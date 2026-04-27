import streamlit as st
import numpy as np
import plotly.graph_objects as go
import itertools

# --- Constants ---
POPULAR_BANDS = {
    "Custom": [0, 0],
    "3GPP n1 (Europe 2100)": [1920, 2170],
    "3GPP n3 (Europe 1800)": [1710, 1880],
    "3GPP n7 (Europe 2600)": [2500, 2690],
    "3GPP n8 (Europe 900)": [880, 960],
    "3GPP n20 (Europe 800)": [791, 862],
    "3GPP n28 (Europe 700)": [703, 803],
    "3GPP n2 (US 1900)": [1850, 1990],
    "3GPP n5 (US 850)": [824, 894],
    "3GPP n12 (US 700)": [699, 746],
    "3GPP n30 (US 2300)": [2305, 2360],
    "3GPP n41 (US 2.5G)": [2496, 2690],
    "3GPP n66 (US AWS)": [1710, 2200],
    "3GPP n71 (US 600)": [617, 698],
    "WiFi 2.4GHz": [2400, 2483.5],
    "WiFi 5GHz": [5150, 5850],
    "WiFi 6GHz": [5925, 7125],
    "GNSS L1": [1559, 1610],
    "Aerospace L-Band": [960, 1215],
}

class ClockSpurChart:
    def run(self):
        st.title("⏰ Spur Chart Generator")

        st.info(r"""
        **Quick Guide**
        - Enter one or more **Frequencies** (comma-separated).
        - Define the **Chart Range** (f_min to f_max) to focus on your band of interest.
        - The chart calculates harmonics and intermodulation products: $f_{spur} = | \sum m_i \cdot f_i |$.
        - Bars are scaled by $1/Order$ for better visual identification.
        """)

        # --- Input Configuration ---
        st.markdown("##### Sources")
        clock_input = st.text_input("Enter Frequencies (MHz, comma-separated)", value="38.4, 19.2")
        
        try:
            clocks = [float(x.strip()) for x in clock_input.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid input. Please enter numeric values separated by commas.")
            st.stop()

        if not clocks:
            st.warning("Please enter at least one frequency.")
            st.stop()

        st.markdown("---")
        st.markdown("##### Plot Settings")
        c1, c2, c3 = st.columns(3)
        with c1:
            f_min = st.number_input("f_min (MHz)", value=0.0, min_value=0.0, step=10.0)
        with c2:
            f_max = st.number_input("f_max (MHz)", value=3000.0, min_value=f_min + 1.0, step=10.0)
        with c3:
            max_order = st.slider("Max Combined Order", min_value=1, max_value=12, value=5)

        col_target_band, _ = st.columns([1, 1])
        with col_target_band:
            target_band_name = st.selectbox("Target Band Highlight", list(POPULAR_BANDS.keys()), index=0)
            if target_band_name == "Custom":
                cb1, cb2 = st.columns(2)
                with cb1:
                    target_min = st.number_input("Target Min (MHz)", value=2400.0)
                with cb2:
                    target_max = st.number_input("Target Max (MHz)", value=2500.0)
            else:
                target_min, target_max = POPULAR_BANDS[target_band_name]

        # --- Calculations ---
        spurs = self._calculate_spurs(clocks, max_order, f_min, f_max)

        # --- Render Chart ---
        self._render_chart(spurs, f_min, f_max, target_min, target_max, target_band_name, clocks)

    def _calculate_spurs(self, clocks, max_order, f_min, f_max):
        """
        Calculates all unique positive spur frequencies within range.
        Spur = |m0*f0 + m1*f1 + ...| where sum(|mi|) <= max_order
        """
        num_clocks = len(clocks)
        # Generate range of coefficients for each clock
        coeff_range = range(-max_order, max_order + 1)
        
        # Use a dict to store the 'best' (lowest order) way to get a frequency
        unique_spurs = {}

        # Iterate through all combinations of coefficients
        for coeffs in itertools.product(coeff_range, repeat=num_clocks):
            if not any(c > 0 for c in coeffs):
                continue
                
            order = sum(abs(c) for c in coeffs)
            if order == 0 or order > max_order:
                continue
            
            freq = abs(sum(c * f for c, f in zip(coeffs, clocks)))
            
            # Filter by chart range and round to avoid floating point duplicates
            if f_min <= freq <= f_max:
                rounded_f = round(freq, 6)
                if rounded_f not in unique_spurs or order < unique_spurs[rounded_f]['order']:
                    unique_spurs[rounded_f] = {
                        'freq': freq,
                        'order': order,
                        'coeffs': coeffs
                    }
        
        return sorted(unique_spurs.values(), key=lambda x: x['freq'])

    def _render_chart(self, spurs, f_min, f_max, t_min, t_max, t_name, clocks):
        if not spurs:
            st.warning("No spurs found in this range with the current order limit.")
            return

        fig = go.Figure()

        # Highlight target band
        if t_max > t_min:
            fig.add_vrect(
                x0=t_min, x1=t_max,
                fillcolor="rgba(59, 130, 246, 0.3)",
                layer="below", line_width=0,
                annotation_text=f"Target: {t_name}",
                annotation_position="top left"
            )

        freqs = [s['freq'] for s in spurs]
        # We use 1/order as a pseudo-magnitude for visualization
        mags = [1.0 / s['order'] for s in spurs]
        
        hover_texts = []
        for s in spurs:
            comp_parts = []
            for i, c in enumerate(s['coeffs']):
                if c != 0:
                    comp_parts.append(f"{c} * {clocks[i]}MHz")
            
            txt = (f"Frequency: {s['freq']:.3f} MHz<br>"
                   f"Total Order: {s['order']}<br>"
                   f"Composition: {' + '.join(comp_parts)}")
            hover_texts.append(txt)

        # Use Bar chart for discrete "combs"
        fig.add_trace(go.Bar(
            x=freqs,
            y=mags,
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                color='orange',
                line=dict(width=0)
            ),
            width=0.5, # Narrow bars to look like spikes
            name="Clock Spurs"
        ))

        fig.update_layout(
            title="Spur Spectrum",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Relative Magnitude (1/Order)",
            xaxis_range=[f_min, f_max],
            yaxis_range=[0, 1.1],
            template="plotly_dark",
            height=600,
            hovermode="closest",
            showlegend=False
        )

        st.plotly_chart(fig, width='stretch')

        # --- Target Band Table ---
        target_spurs = [s for s in spurs if t_min <= s['freq'] <= t_max]
        if target_spurs:
            st.markdown(f"### 🎯 Spurs Overlapping {t_name}")
            table_data = []
            for s in target_spurs:
                comp_parts = [f"{c}*({clocks[i]}MHz)" for i, c in enumerate(s['coeffs']) if c != 0]
                table_data.append({
                    "Frequency (MHz)": round(s['freq'], 4),
                    "Clock Content": " + ".join(comp_parts),
                    "Total Order": s['order']
                })
            st.table(table_data)
        elif t_max > t_min:
            st.info(f"No spurs detected in the {t_name} range.")

if __name__ == "__main__":
    ClockSpurChart().run()