import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import textwrap

TOPOLOGIES = ["T-Pad", "Pi-Pad", "Bridged-T Pad"]

TRACE_MAG_COLOR = "#0891b2"
TRACE_RL_COLOR = "#f97316"


class AttenuatorDesigner:
    """RF Attenuator (Pad) designer with component value calculation and S-parameter plots."""

    def run(self):
        st.title("🧮 Attenuator Pad Designer")

        st.info("""
        **Quick Guide**
        - Select a **Topology**, set the desired **Attenuation**, and the **Source (Zs)** and **Load (ZL)** impedances.
        - The attenuator components are calculated assuming the attenuator is designed for the **Source Impedance (Zs)**.
        - If **Zs ≠ ZL**, the S-parameter plots will show the effect of the impedance mismatch at the load.
        - Component values will be calculated and displayed.
        - S-parameter plots verify performance (S₂₁ Gain and S₁₁ Return Loss).
        """)

        # --- Configuration ---
        st.markdown("##### Configuration")
        c1, c2, c3, c4 = st.columns(4) # Removed the Z0 input, so 4 columns instead of 5
        with c1:
            topology = st.selectbox("Topology", TOPOLOGIES, index=0)
        with c2:
            attenuation_db = st.number_input("Attenuation (dB)", value=10.0, min_value=0.1, max_value=60.0, step=1.0)
        with c3:
            source_impedance_zs = st.number_input("Source Zs (Ω)", value=50.0, min_value=1.0, step=10.0)
        with c4:
            load_impedance_zl = st.number_input("Load ZL (Ω)", value=50.0, min_value=1.0, step=10.0)

        # --- Calculations ---
        K = 10**(attenuation_db / 20)

        # Use source_impedance_zs as the design impedance for the attenuator.
        # This assumes a symmetric attenuator designed for Zs, and ZL is just the load connected.
        # If Zs != ZL, the attenuator will not be perfectly matched at the load side.
        design_impedance_z0 = source_impedance_zs

        try:
            if topology == "T-Pad":
                R1 = design_impedance_z0 * (K - 1) / (K + 1)
                R2 = 2 * design_impedance_z0 * K / (K**2 - 1)
                resistor_values = {"R1": R1, "R2": R2}
            elif topology == "Pi-Pad":
                R1 = design_impedance_z0 * (K + 1) / (K - 1) 
                R2 = design_impedance_z0 * (K**2 - 1) / (2 * K)
                resistor_values = {"R1": R1, "R2": R2}
            elif topology == "Bridged-T Pad":
                R1 = design_impedance_z0
                R2 = design_impedance_z0 / (K - 1)
                R3 = design_impedance_z0 * (K - 1)
                resistor_values = {"R1": R1, "R2": R2, "R3": R3}
            else:
                st.error("Unknown topology.")
                st.stop()
            
            # Calculate ABCD matrix for a symmetric attenuator with design_impedance_z0 and K
            A = (K**2 + 1) / (2 * K)
            B = design_impedance_z0 * (K**2 - 1) / (2 * K)
            C = (1 / design_impedance_z0) * (K**2 - 1) / (2 * K)
            D = A
            
            S11, S21, S22 = self._abcd_to_s_params(A, B, C, D, source_impedance_zs, load_impedance_zl)

            s11_db_actual = 20 * np.log10(np.abs(S11))
            s21_db_actual = 20 * np.log10(np.abs(S21))

        except Exception as e:
            st.error(f"Calculation Error: {e}. Please check your inputs.")
            st.stop()

        # Warning for Zs != ZL
        if source_impedance_zs != load_impedance_zl:
            st.warning(f"⚠️ **Impedance Mismatch**: The attenuator is designed for Zs ({source_impedance_zs:.0f} Ω). "
                       f"Connecting a load of ZL ({load_impedance_zl:.0f} Ω) will result in a mismatch at the output.")

        # --- Display Component Values ---
        st.markdown("##### Component Values")
        cols = st.columns(len(resistor_values))
        for i, (key, value) in enumerate(resistor_values.items()):
            with cols[i]:
                label = key
                if topology == "T-Pad":
                    label = "R1 (Series)" if key == "R1" else "R2 (Shunt)"
                elif topology == "Pi-Pad":
                    label = "R1 (Shunt)" if key == "R1" else "R2 (Series)"
                elif topology == "Bridged-T Pad":
                    if key == "R1": label = "R1 (Series)"
                    elif key == "R2": label = "R2 (Shunt)"
                    elif key == "R3": label = "R3 (Bridge)"
                st.metric(label, f"{value:.2f} Ω")

        # --- Diagrams ---
        st.markdown("---")
        st.subheader("Implementation Schematic")
        if topology == "T-Pad":
            st.markdown(self._t_pad_svg(resistor_values, source_impedance_zs, load_impedance_zl), unsafe_allow_html=True)
        elif topology == "Pi-Pad":
            st.markdown(self._pi_pad_svg(resistor_values, source_impedance_zs, load_impedance_zl), unsafe_allow_html=True)
        elif topology == "Bridged-T Pad":
            st.markdown(self._bridged_t_pad_svg(resistor_values, source_impedance_zs, load_impedance_zl), unsafe_allow_html=True)

        # --- S-Parameter Plots ---
        st.markdown("---")
        st.markdown("##### S-Parameter Plots")
        freq_axis_mhz = np.linspace(0, 1000, 100) # 0 MHz to 1 GHz for plotting (linear axis)

        # S21 (Magnitude)
        fig_s21 = go.Figure()
        fig_s21.add_trace(go.Scatter(
            x=freq_axis_mhz,
            y=np.full_like(freq_axis_mhz, s21_db_actual),
            mode='lines',
            name='S21 Magnitude',
            line=dict(color=TRACE_MAG_COLOR, width=2)
        ))
        fig_s21.update_layout(
            title="S₂₁ Transmission Coefficient",
            xaxis_title="Frequency (MHz)",
            yaxis_title="S₂₁ (dB)",
            yaxis_range=[s21_db_actual - 5, 0],
            height=380,
            template="plotly_dark"
        )
        st.plotly_chart(fig_s21, width='stretch')

        # S11 (Return Loss)
        fig_s11 = go.Figure()
        fig_s11.add_trace(go.Scatter(
            x=freq_axis_mhz,
            y=np.full_like(freq_axis_mhz, s11_db_actual),
            mode='lines',
            name='S11 Return Loss',
            line=dict(color=TRACE_RL_COLOR, width=2)
        ))
        fig_s11.update_layout(
            title="S₁₁ Input Reflection",
            xaxis_title="Frequency (MHz)",
            yaxis_title="S₁₁ (dB)",
            yaxis_range=[-80, 0],
            height=380,
            template="plotly_dark"
        )
        st.plotly_chart(fig_s11, width='stretch')

    # ==========================================================================
    # S-parameter calculations
    # ==========================================================================
    def _abcd_to_s_params(self, A, B, C, D, Zs, ZL):
        Zs = float(Zs)
        ZL = float(ZL)
        if Zs <= 0 or ZL <= 0:
            return complex(1,0), complex(0,0), complex(1,0)
        sqrt_zs_zl = math.sqrt(Zs * ZL)
        denom = A * ZL + B + C * Zs * ZL + D * Zs
        if abs(denom) < 1e-12:
            return complex(1,0), complex(0,0), complex(1,0)
        S11 = (A * ZL + B - C * Zs * ZL - D * Zs) / denom
        S21 = (2 * sqrt_zs_zl) / denom
        S22 = (A * Zs + B - C * Zs * ZL - D * ZL) / denom
        return S11, S21, S22

    # ==========================================================================
    # SVG Diagrams
    # ==========================================================================
    RESISTOR_COLOR = "#f97316"

    def _fmt_resistor_value(self, value):
        if value >= 1e3:
            return f"{value / 1e3:.1f} kΩ"
        return f"{value:.1f} Ω"

    def _t_pad_svg(self, values, zs, zl):
        r1_val = self._fmt_resistor_value(values.get("R1", 0))
        r2_val = self._fmt_resistor_value(values.get("R2", 0))
        return textwrap.dedent(f"""
        <svg width="100%" height="220" viewBox="0 0 600 220" xmlns="http://www.w3.org/2000/svg" style="background:#111; border-radius:8px; display:block; margin:auto;">
            <text x="20" y="40" fill="#888" font-size="16" font-weight="bold">T-PAD ATTENUATOR</text>
            <text x="10" y="105" fill="#fff" font-size="18">IN</text>
            <text x="560" y="105" fill="#fff" font-size="18">OUT</text>
            <text x="40" y="140" text-anchor="middle" fill="#888" font-size="18">Zs={zs:.0f}Ω</text>
            <text x="560" y="140" text-anchor="middle" fill="#888" font-size="18">ZL={zl:.0f}Ω</text>
            <line x1="60" y1="100" x2="150" y2="100" stroke="#fff" stroke-width="3"/>
            <line x1="250" y1="100" x2="350" y2="100" stroke="#fff" stroke-width="3"/>
            <line x1="450" y1="100" x2="540" y2="100" stroke="#fff" stroke-width="3"/>
            {self._draw_resistor(150, 100, 100, 0, r1_val)}
            <line x1="300" y1="100" x2="300" y2="140" stroke="#fff" stroke-width="3"/>
            {self._draw_resistor(300, 140, 0, 40, r2_val)}
            {self._draw_ground(300, 180)}
            {self._draw_resistor(350, 100, 100, 0, r1_val)}
        </svg>
        """).strip()

    def _pi_pad_svg(self, values, zs, zl):
        r1_val = self._fmt_resistor_value(values.get("R1", 0))
        r2_val = self._fmt_resistor_value(values.get("R2", 0))
        return textwrap.dedent(f"""
        <svg width="100%" height="220" viewBox="0 0 600 220" xmlns="http://www.w3.org/2000/svg" style="background:#111; border-radius:8px; display:block; margin:auto;">
            <text x="20" y="40" fill="#888" font-size="16" font-weight="bold">PI-PAD ATTENUATOR</text>
            <text x="10" y="105" fill="#fff" font-size="18">IN</text>
            <text x="560" y="105" fill="#fff" font-size="18">OUT</text>
            <text x="40" y="140" text-anchor="middle" fill="#888" font-size="18">Zs={zs:.0f}Ω</text>
            <text x="560" y="140" text-anchor="middle" fill="#888" font-size="18">ZL={zl:.0f}Ω</text>
            <line x1="60" y1="100" x2="150" y2="100" stroke="#fff" stroke-width="3"/>
            <line x1="450" y1="100" x2="540" y2="100" stroke="#fff" stroke-width="3"/>
            <line x1="150" y1="100" x2="150" y2="140" stroke="#fff" stroke-width="3"/>
            {self._draw_resistor(150, 140, 0, 40, r1_val)}
            {self._draw_ground(150, 180)}
            {self._draw_resistor(150, 100, 300, 0, r2_val)}
            <line x1="450" y1="100" x2="450" y2="140" stroke="#fff" stroke-width="3"/>
            {self._draw_resistor(450, 140, 0, 40, r1_val)}
            {self._draw_ground(450, 180)}
        </svg>
        """).strip()

    def _bridged_t_pad_svg(self, values, zs, zl):
        r1_val = self._fmt_resistor_value(values.get("R1", 0))
        r2_val = self._fmt_resistor_value(values.get("R2", 0))
        r3_val = self._fmt_resistor_value(values.get("R3", 0))
        return textwrap.dedent(f"""
        <svg width="100%" height="280" viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg" style="background:#111; border-radius:8px; display:block; margin:auto;">
            <text x="20" y="40" fill="#888" font-size="16" font-weight="bold">BRIDGED-T PAD ATTENUATOR</text>
            <text x="10" y="145" fill="#fff" font-size="18">IN</text>
            <text x="560" y="145" fill="#fff" font-size="18">OUT</text>
            <text x="40" y="180" text-anchor="middle" fill="#888" font-size="18">Zs={zs:.0f}Ω</text>
            <text x="560" y="180" text-anchor="middle" fill="#888" font-size="18">ZL={zl:.0f}Ω</text>
            <line x1="60" y1="140" x2="150" y2="140" stroke="#fff" stroke-width="3"/>
            <line x1="250" y1="140" x2="350" y2="140" stroke="#fff" stroke-width="3"/>
            <line x1="450" y1="140" x2="540" y2="140" stroke="#fff" stroke-width="3"/>
            {self._draw_resistor(150, 140, 100, 0, r1_val)}
            <line x1="300" y1="140" x2="300" y2="180" stroke="#fff" stroke-width="3"/>
            {self._draw_resistor(300, 180, 0, 40, r2_val)}
            {self._draw_ground(300, 220)}
            <line x1="150" y1="140" x2="150" y2="80" stroke="#fff" stroke-width="3"/>
            <line x1="450" y1="140" x2="450" y2="80" stroke="#fff" stroke-width="3"/>
            {self._draw_resistor(150, 80, 300, 0, r3_val)}
            {self._draw_resistor(350, 140, 100, 0, r1_val)}
        </svg>
        """).strip()

    @staticmethod
    def _draw_resistor(x, y, length, height, label):
        """Draws a resistor symbol and its label."""
        # Resistor zig-zag path
        if length > 0: # Horizontal
            path = f"M {x},{y} "
            seg_len = length / 6
            for i in range(6):
                path += f"l {seg_len/2},{(-1 if i%2==0 else 1)*8} "
                path += f"l {seg_len/2},{(-1 if i%2==0 else 1)*-8} "
            text_x = x + length / 2
            text_y = y - 25
            anchor = "middle"
        else: # Vertical
            path = f"M {x},{y} "
            seg_len = height / 6
            for i in range(6):
                path += f"l {(-1 if i%2==0 else 1)*5},{seg_len/2} "
                path += f"l {(-1 if i%2==0 else 1)*-5},{seg_len/2} "
            text_x = x + 35
            text_y = y + height / 2
            anchor = "start"

        return (f'<path d="{path}" fill="none" stroke="{AttenuatorDesigner.RESISTOR_COLOR}" stroke-width="3"/>'
                f'<text x="{text_x}" y="{text_y}" text-anchor="{anchor}" fill="{AttenuatorDesigner.RESISTOR_COLOR}" font-size="18">{label}</text>')

    @staticmethod
    def _draw_ground(x, y):
        return (f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+8}" stroke="#fff" stroke-width="3"/>'
                f'<line x1="{x-15}" y1="{y+8}" x2="{x+15}" y2="{y+8}" stroke="#fff" stroke-width="2.5" opacity="0.7"/>'
                f'<line x1="{x-10}" y1="{y+16}" x2="{x+10}" y2="{y+16}" stroke="#fff" stroke-width="2.5" opacity="0.7"/>'
                f'<line x1="{x-5}" y1="{y+24}" x2="{x+5}" y2="{y+24}" stroke="#fff" stroke-width="2.5" opacity="0.7"/>')


if __name__ == "__main__":
    AttenuatorDesigner().run()