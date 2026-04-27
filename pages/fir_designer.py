import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal

TOPOLOGIES = ["Windowed Sinc", "Equiripple (Parks-McClellan)", "CIC (Cascaded Integrator-Comb)"]
FILTER_TYPES = ["Lowpass", "Highpass", "Bandpass", "Bandstop"]
WINDOWS = ["hamming", "hann", "blackman", "bartlett", "kaiser"]

TRACE_MAG_COLOR   = "#0891b2"
TRACE_PHASE_COLOR = "#f97316"
TRACE_GD_COLOR    = "#16a34a"

class FirDesigner:
    """Digital FIR filter designer with multi-topology support."""

    def run(self):
        st.title("🔢 FIR Filter Designer")

        st.info("""
        **Quick Guide**
        - **Windowed Sinc**: Fast design using windowing. High Taps = Sharper roll-off.
        - **Equiripple**: Optimal PM algorithm. Use 'Transition Width' to control steepness.
        - **CIC**: Multiplier-free, used for decimation. Response is (sin x / x)^N.
        """)

        # --- Configuration Row 1 ---
        st.markdown("##### Configuration")
        c1, c2, c3, c4, _ = st.columns([0.98, 0.98, 0.75, 0.75, 1.14])
        with c1:
            topo_selected = st.selectbox("Design Method", TOPOLOGIES, index=0)
        with c2:
            ftype = st.selectbox("Filter Type", FILTER_TYPES, index=0, disabled=("CIC" in topo_selected))
        with c3:
            fs = st.number_input("Fs (Hz)", value=1000.0, min_value=1.0, step=100.0)
        with c4:
            if "CIC" in topo_selected:
                N_stages = st.number_input("Stages (N)", value=3, min_value=1, max_value=10)
            else:
                N_taps = st.number_input("Taps (N)", value=31, min_value=1, max_value=1000, step=2)

        # --- Configuration Row 2 ---
        st.markdown("##### Frequencies & Tuning")
        if "CIC" in topo_selected:
            c2_1, c2_2, c2_3, _ = st.columns([0.75, 0.75, 0.75, 1.75])
            with c2_1:
                R = st.number_input("Decimation (R)", value=8, min_value=1)
            with c2_2:
                M = st.selectbox("Diff Delay (M)", [1, 2], index=0)
            f1, f2 = None, None
        else:
            if ftype in ["Lowpass", "Highpass"]:
                c2_1, c2_2, _ = st.columns([0.75, 0.75, 2.50])
                with c2_1:
                    f1 = st.number_input("Cutoff (Hz)", value=100.0, min_value=0.1, max_value=fs/2-0.1)
                f2 = None
            else:
                c2_1, c2_2, _ = st.columns([0.75, 0.75, 2.50])
                with c2_1:
                    f1 = st.number_input("Freq 1 (Hz)", value=100.0, min_value=0.1, max_value=fs/2-1.0)
                with c2_2:
                    f2 = st.number_input("Freq 2 (Hz)", value=200.0, min_value=f1+0.1, max_value=fs/2-0.1)

            c3_1, c3_2, _ = st.columns([0.75, 0.75, 2.50])
            with c3_1:
                if "Window" in topo_selected:
                    win = st.selectbox("Window", WINDOWS)
                else:
                    trans_w = st.number_input("Trans. Width (Hz)", value=20.0, min_value=1.0)

        # --- Design & Computation ---
        freq_axis = np.linspace(0, fs/2, 1024)
        b = None

        try:
            if "Window" in topo_selected:
                pass_zero = (ftype in ["Lowpass", "Bandstop"])
                cutoffs = f1 if f2 is None else [f1, f2]
                b = signal.firwin(N_taps, cutoffs, fs=fs, pass_zero=pass_zero, window=win)
            elif "Equiripple" in topo_selected:
                if ftype == "Lowpass":
                    bands, desired = [0, f1, f1+trans_w, fs/2], [1, 0]
                elif ftype == "Highpass":
                    bands, desired = [0, f1-trans_w, f1, fs/2], [0, 1]
                elif ftype == "Bandpass":
                    bands = [0, f1-trans_w, f1, f2, f2+trans_w, fs/2]
                    desired = [0, 1, 0]
                elif ftype == "Bandstop":
                    bands = [0, f1, f1+trans_w, f2-trans_w, f2, fs/2]
                    desired = [1, 0, 1]

                # Validate bands: must be strictly increasing and within [0, fs/2]
                bands = np.clip(bands, 0, fs/2)
                if any(np.diff(bands) <= 0):
                    raise ValueError("Transition width is too large or frequencies are too close for Equiripple design.")

                b = signal.remez(N_taps, bands, desired, fs=fs)
            elif "CIC" in topo_selected:
                # Generate equivalent FIR coefficients for a CIC filter
                # A single stage is a moving average of length R*M
                single_rect = np.ones(R * M)
                b_cic = single_rect
                for _ in range(N_stages - 1):
                    b_cic = np.convolve(b_cic, single_rect)
                # Normalize for DC gain of 1
                b = b_cic / (R * M)**N_stages

            # Compute frequency response from coefficients
            w, h = signal.freqz(b, [1], worN=freq_axis, fs=fs)

        except Exception as e:
            st.error(f"Design Constraint Error: {e}")
            st.stop()

        # --- Plots ---
        col_m, col_p = st.columns(2)
        with col_m:
            fig_m = go.Figure()
            mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
            fig_m.add_trace(go.Scatter(x=w, y=mag_db, line=dict(color=TRACE_MAG_COLOR, width=2)))
            fig_m.update_layout(title="Magnitude Response", xaxis_title="Freq (Hz)", yaxis_title="Gain (dB)", height=380, template="plotly_dark")
            st.plotly_chart(fig_m, width='stretch')
        
        with col_p:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=w, y=np.rad2deg(np.unwrap(np.angle(h))), line=dict(color=TRACE_PHASE_COLOR, width=2)))
            fig_p.update_layout(title="Unwrapped Phase", xaxis_title="Freq (Hz)", yaxis_title="Phase (°)", height=380, template="plotly_dark")
            st.plotly_chart(fig_p, width='stretch')

        fig_gd = go.Figure()
        gd = -np.gradient(np.unwrap(np.angle(h)), 2 * np.pi * w)
        fig_gd.add_trace(go.Scatter(x=w, y=gd * 1e3, line=dict(color=TRACE_GD_COLOR, width=2)))
        fig_gd.update_layout(title="Group Delay", xaxis_title="Freq (Hz)", yaxis_title="Delay (ms)", height=300, template="plotly_dark")
        st.plotly_chart(fig_gd, width='stretch')

        # --- Coefficients & Mathematical Equation ---
        st.markdown("---")
        col_eq, col_taps = st.columns([1.2, 0.8])

        with col_eq:
            st.subheader("Mathematical Representation")
            if "CIC" in topo_selected:
                st.latex(r"H(z) = \left( \frac{1 - z^{-RM}}{1 - z^{-1}} \right)^N")
                st.info(f"Recursive form: R={R}, M={M}, N={N_stages}")
            else:
                st.latex(r"H(z) = \sum_{n=0}^{N-1} b_n z^{-n}")
                st.info(f"Direct-form FIR with {len(b)} coefficients")

        with col_taps:
            st.subheader("Filter Taps")
            taps_df = [{"n": i, "Value (b[n])": val} for i, val in enumerate(b)]
            st.dataframe(taps_df, height=220, width='stretch', hide_index=True)
            
            csv_out = "n,coefficient\n" + "\n".join([f"{i},{v:.18e}" for i, v in enumerate(b)])
            st.download_button(
                "Download Taps (CSV)",
                data=csv_out,
                file_name=f"fir_coeffs_{topo_selected.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

        # --- Diagrams ---
        st.markdown("---")
        st.subheader("Implementation Diagram")
        if "CIC" in topo_selected:
            st.markdown(self._cic_svg(), unsafe_allow_html=True)
        else:
            st.markdown(self._fir_svg(), unsafe_allow_html=True)

    def _fir_svg(self):
        return """
        <svg width="100%" height="250" viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="background:#111; border-radius:8px;">
            <text x="20" y="30" fill="#888" font-size="12" font-weight="bold">DIRECT FORM FIR STRUCTURE</text>
            <!-- Input -->
            <text x="10" y="75" fill="#fff" font-size="14">x[n]</text>
            <line x1="40" y1="70" x2="80" y2="70" stroke="#fff" stroke-width="2" marker-end="url(#arrow)"/>
            <!-- Delay Chain -->
            <rect x="100" y="50" width="40" height="40" fill="#222" stroke="#fff" stroke-width="2"/>
            <text x="120" y="75" text-anchor="middle" fill="#fff" font-size="12">z⁻¹</text>
            <line x1="140" y1="70" x2="180" y2="70" stroke="#fff" stroke-width="2" marker-end="url(#arrow)"/>
            <rect x="180" y="50" width="40" height="40" fill="#222" stroke="#fff" stroke-width="2"/>
            <text x="200" y="75" text-anchor="middle" fill="#fff" font-size="12">z⁻¹</text>
            <line x1="220" y1="70" x2="260" y2="70" stroke="#fff" stroke-width="2" stroke-dasharray="4"/>
            <!-- Multipliers (Triangles) -->
            <path d="M 60,70 L 60,110 L 50,110 L 60,130 L 70,110 L 60,110" fill="#0891b2" stroke="#fff"/>
            <text x="45" y="125" text-anchor="end" fill="#0891b2" font-size="11">h[0]</text>
            <path d="M 160,70 L 160,110 L 150,110 L 160,130 L 170,110 L 160,110" fill="#0891b2" stroke="#fff"/>
            <text x="145" y="125" text-anchor="end" fill="#0891b2" font-size="11">h[1]</text>
            <path d="M 240,70 L 240,110 L 230,110 L 240,130 L 250,110 L 240,110" fill="#0891b2" stroke="#fff"/>
            <text x="225" y="125" text-anchor="end" fill="#0891b2" font-size="11">h[2]</text>
            <!-- Summation Bus -->
            <line x1="60" y1="160" x2="520" y2="160" stroke="#fff" stroke-width="2"/>
            <circle cx="160" cy="160" r="10" fill="#111" stroke="#fff" stroke-width="2"/>
            <text x="160" y="164" text-anchor="middle" fill="#fff" font-size="14">+</text>
            <circle cx="240" cy="160" r="10" fill="#111" stroke="#fff" stroke-width="2"/>
            <text x="240" y="164" text-anchor="middle" fill="#fff" font-size="14">+</text>
            <!-- Output -->
            <line x1="520" y1="160" x2="560" y2="160" stroke="#fff" stroke-width="2" marker-end="url(#arrow)"/>
            <text x="570" y="165" fill="#fff" font-size="14">y[n]</text>
            <!-- Definitions -->
            <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#fff" />
                </marker>
            </defs>
        </svg>
        """

    def _cic_svg(self):
        return """
        <svg width="100%" height="250" viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="background:#111; border-radius:8px;">
            <text x="20" y="30" fill="#888" font-size="12" font-weight="bold">CIC DECIMATOR (N-STAGES)</text>
            <!-- Integrator Stage -->
            <text x="80" y="50" text-anchor="middle" fill="#16a34a" font-size="10">INTEGRATOR</text>
            <circle cx="80" cy="100" r="12" fill="#111" stroke="#fff" stroke-width="2"/>
            <text x="80" y="105" text-anchor="middle" fill="#fff" font-size="16">+</text>
            <line x1="92" y1="100" x2="130" y2="100" stroke="#fff" stroke-width="2"/>
            <path d="M 130,100 L 130,150 L 80,150 L 80,112" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="2"/>
            <rect x="100" y="135" width="30" height="30" fill="#222" stroke="#fff"/>
            <text x="115" y="155" text-anchor="middle" fill="#fff" font-size="10">z⁻¹</text>
            <!-- Rate Change -->
            <line x1="130" y1="100" x2="200" y2="100" stroke="#fff" stroke-width="2"/>
            <rect x="200" y="80" width="40" height="40" rx="20" fill="#f97316" stroke="#fff"/>
            <text x="220" y="105" text-anchor="middle" fill="#fff" font-size="12">↓R</text>
            <!-- Comb Stage -->
            <text x="340" y="50" text-anchor="middle" fill="#0891b2" font-size="10">COMB (M-DELAY)</text>
            <line x1="240" y1="100" x2="300" y2="100" stroke="#fff" stroke-width="2"/>
            <circle cx="280" cy="100" r="4" fill="#fff"/>
            <line x1="280" y1="100" x2="280" y2="150" stroke="#fff" stroke-width="2"/>
            <rect x="300" y="135" width="40" height="30" fill="#222" stroke="#fff"/>
            <text x="320" y="155" text-anchor="middle" fill="#fff" font-size="10">z⁻ᴹ</text>
            <line x1="340" y1="150" x2="380" y2="150" stroke="#fff" stroke-width="2"/>
            <line x1="380" y1="150" x2="380" y2="112" stroke="#fff" stroke-width="2"/>
            <circle cx="380" cy="100" r="12" fill="#111" stroke="#fff" stroke-width="2"/>
            <text x="380" y="105" text-anchor="middle" fill="#fff" font-size="16">−</text>
            <line x1="300" y1="100" x2="368" y2="100" stroke="#fff" stroke-width="2"/>
            <!-- Output -->
            <line x1="392" y1="100" x2="450" y2="100" stroke="#fff" stroke-width="2"/>
            <text x="460" y="105" fill="#fff" font-size="14">y[n]</text>
            <text x="160" y="185" text-anchor="middle" fill="#888" font-size="10 italic">Repeating N times...</text>
        </svg>
        """

if __name__ == "__main__":
    FirDesigner().run()