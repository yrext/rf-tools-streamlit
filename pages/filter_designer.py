import streamlit as st
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from scipy.signal import butter, cheby1, cheby2, bessel, ellip, zpk2tf, freqs


TOPOLOGIES    = ["Butterworth", "Chebyshev I", "Chebyshev II", "Bessel", "Elliptic"]
FILTER_TYPES  = ["Lowpass", "Highpass", "Bandpass", "Bandstop"]

# Tabulated doubly-terminated Bessel g-values for maximally-flat group delay
# (R_s = R_L = 1, -3dB at ω = 1 after scaling). Index: N=1..10.
BESSEL_G = {
    1:  [1.0000],
    2:  [1.5774, 0.4226],
    3:  [1.2550, 0.5528, 0.1922],
    4:  [1.0598, 0.5116, 0.3181, 0.1104],
    5:  [0.9303, 0.4577, 0.3312, 0.2090, 0.0718],
    6:  [0.8377, 0.4116, 0.3158, 0.2364, 0.1480, 0.0505],
    7:  [0.7677, 0.3744, 0.2944, 0.2378, 0.1778, 0.1104, 0.0375],
    8:  [0.7125, 0.3446, 0.2735, 0.2297, 0.1867, 0.1387, 0.0855, 0.0289],
    9:  [0.6678, 0.3203, 0.2547, 0.2184, 0.1859, 0.1506, 0.1111, 0.0682, 0.0230],
    10: [0.6305, 0.3002, 0.2384, 0.2066, 0.1808, 0.1539, 0.1240, 0.0911, 0.0557, 0.0187],
}

TRACE_MAG_COLOR   = "#0891b2"
TRACE_PHASE_COLOR = "#f97316"
TRACE_GD_COLOR    = "#16a34a"
SPEC_LINE_COLOR   = "rgba(239, 68, 68, 0.55)"


class FilterDesigner:
    """Analog prototype filter designer with live plots and component/Touchstone export."""

    def __init__(self):
        try:
            st.set_page_config(page_title="Filter Designer", layout="wide")
        except Exception:
            pass

    # ==========================================================================
    # UI
    # ==========================================================================
    def run(self):
        st.title("🎛️ RF Filter Designer")

        st.info("""
        **Quick Guide**
        - **Approximation**: Select a topology (Butterworth, Chebyshev, etc.) to define the roll-off.
        - **Frequency**: Define cutoffs or edges. f₀ and Bandwidth are derived automatically.
        - **Implementation**: View the physical LC ladder schematic and download components as CSV.
        """)

        # === Design controls ================================================
        # Row 1 — Topology / type / order / impedance
        st.markdown("##### Design")
        r1c1, r1c2, r1c3, r1c4, _ = st.columns([0.98, 0.98, 0.75, 0.75, 1.14])
        with r1c1:
            topology = st.selectbox("Approximation", TOPOLOGIES, index=0)
        with r1c2:
            ftype = st.selectbox("Response type", FILTER_TYPES, index=0)
        with r1c3:
            N = st.slider("Order N", min_value=1, max_value=10, value=3, step=1)
        with r1c4:
            r0 = st.number_input("R₀ (Ω)", value=50.0, min_value=1.0, step=5.0, format="%.1f")

        # Row 2 — Frequencies (1 or 2 fields depending on ftype) + derived info
        st.markdown("##### Frequencies")
        if ftype in ("Lowpass", "Highpass"):
            r2c1, r2c2, r2c3, _ = st.columns([0.75, 0.75, 1.50, 1.00])
            with r2c1:
                fc_mhz = st.number_input("Cutoff f_c (MHz)", value=100.0, min_value=0.001, step=10.0, format="%.3f")
            f1_mhz, f2_mhz = fc_mhz, None
            f0_mhz, bw_mhz = None, None
            with r2c2:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.caption(f"{ftype} · cutoff = {fc_mhz:.3f} MHz")
        else:
            r2c1, r2c2, r2c3, _ = st.columns([0.75, 0.75, 1.50, 1.00])
            with r2c1:
                f1_mhz = st.number_input("Lower edge f₁ (MHz)", value=80.0, min_value=0.001, step=5.0, format="%.3f")
            with r2c2:
                f2_mhz = st.number_input("Upper edge f₂ (MHz)", value=120.0, min_value=0.001, step=5.0, format="%.3f")
            if f2_mhz <= f1_mhz:
                st.error("Upper edge must be greater than lower edge.")
                st.stop()
            f0_mhz = float(np.sqrt(f1_mhz * f2_mhz))
            bw_mhz = f2_mhz - f1_mhz
            with r2c3:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.caption(f"f₀ = {f0_mhz:.3f} MHz · BW = {bw_mhz:.3f} MHz · Q = {f0_mhz / bw_mhz:.2f}")

        # Row 3 — Ripple / attenuation / sweep
        st.markdown("##### Ripple, attenuation & sweep")
        needs_rip = topology in ("Chebyshev I", "Elliptic")
        needs_att = topology in ("Chebyshev II", "Elliptic")
        r3c1, r3c2, r3c3, r3c4, _ = st.columns([0.75, 0.75, 0.75, 0.75, 1.00])
        with r3c1:
            rip_db = st.number_input(
                "Passband ripple Aₚ (dB)", value=0.5, min_value=0.001, step=0.1,
                format="%.3f", disabled=not needs_rip,
            )
        with r3c2:
            att_db = st.number_input(
                "Stopband Aₛ (dB)", value=40.0, min_value=1.0, step=1.0,
                format="%.2f", disabled=not needs_att,
            )
        with r3c3:
            decades = st.slider("Sweep span (±decades)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        with r3c4:
            n_pts = st.slider("Sweep points", min_value=128, max_value=4096, value=801, step=1)

        st.markdown("---")

        # --- Design: scipy produces the zpk for the requested btype directly.
        btype_map = {
            "Lowpass":  "lowpass",
            "Highpass": "highpass",
            "Bandpass": "bandpass",
            "Bandstop": "bandstop",
        }
        btype = btype_map[ftype]
        if ftype in ("Lowpass", "Highpass"):
            Wn = 2 * np.pi * f1_mhz * 1e6
        else:
            Wn = [2 * np.pi * f1_mhz * 1e6, 2 * np.pi * f2_mhz * 1e6]

        try:
            common = dict(btype=btype, analog=True, output="zpk")
            if topology == "Butterworth":
                z_t, p_t, k_t = butter(N, Wn, **common)
            elif topology == "Chebyshev I":
                z_t, p_t, k_t = cheby1(N, rip_db, Wn, **common)
            elif topology == "Chebyshev II":
                z_t, p_t, k_t = cheby2(N, att_db, Wn, **common)
            elif topology == "Bessel":
                z_t, p_t, k_t = bessel(N, Wn, norm="mag", **common)
            elif topology == "Elliptic":
                z_t, p_t, k_t = ellip(N, rip_db, att_db, Wn, **common)
            else:
                st.error("Unknown topology.")
                st.stop()
        except Exception as e:
            st.error(f"Design failed: {e}")
            st.stop()
        z_t = np.asarray(z_t, dtype=complex)
        p_t = np.asarray(p_t, dtype=complex)
        k_t = float(k_t)

        # --- Sweep frequencies --------------------------------------------
        if ftype in ("Lowpass", "Highpass"):
            f_center = f1_mhz * 1e6
        else:
            f_center = f0_mhz * 1e6
        f_lo = max(f_center / (10 ** decades), 1.0)
        f_hi = f_center * (10 ** decades)
        sweep_hz = np.logspace(np.log10(f_lo), np.log10(f_hi), int(n_pts))

        H = self.eval_H(z_t, p_t, k_t, sweep_hz)

        # --- Plots --------------------------------------------------------
        col_mag, col_phase = st.columns(2)
        with col_mag:
            st.plotly_chart(self._mag_fig(sweep_hz, H, ftype, f1_mhz, f2_mhz), width='stretch')
        with col_phase:
            st.plotly_chart(self._phase_fig(sweep_hz, H), width='stretch')
        st.plotly_chart(self._group_delay_fig(sweep_hz, H), width='stretch')

        # --- Summary metrics ----------------------------------------------
        mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-300))
        m_cols = st.columns(4)
        m_cols[0].metric("Peak gain", f"{np.max(mag_db):.2f} dB")
        m_cols[1].metric("Min gain (in sweep)", f"{np.min(mag_db):.2f} dB")
        center_idx = int(np.argmin(np.abs(sweep_hz - f_center)))
        m_cols[2].metric("Gain @ center", f"{mag_db[center_idx]:.2f} dB")
        gd = self._group_delay(sweep_hz, H)
        m_cols[3].metric("Group delay @ center", f"{gd[center_idx] * 1e9:.2f} ns")

        # --- Exports ------------------------------------------------------
        st.markdown("---")
        st.subheader("Export")

        # LC ladder
        components = self._synthesize_ladder(topology, ftype, N, rip_db,
                                             f1_mhz, f2_mhz, f0_mhz, bw_mhz, r0)
        ladder_cols = st.columns([2, 1])
        with ladder_cols[0]:
            st.markdown("#### Component Values (LC Ladder)")
            if components is None:
                st.info(
                    "LC ladder synthesis is only provided for Butterworth, Chebyshev I, "
                    "and Bessel topologies. Use the Touchstone export for the matched "
                    "response of this filter."
                )
            else:
                self._render_component_table(components)

        if components is not None:
            st.markdown("#### Implementation Schematic")
            st.markdown(self.render_ladder_svg(components, r0), unsafe_allow_html=True)
            
            csv_text = self._components_to_csv(components, topology, ftype, r0,
                                                f1_mhz, f2_mhz, f0_mhz, bw_mhz)
            st.download_button(
                "Download components (CSV)",
                data=csv_text,
                file_name=f"{topology.lower().replace(' ', '')}_{ftype.lower()}_N{N}.csv",
                mime="text/csv",
            )

        with ladder_cols[1]:
            st.markdown("#### Touchstone (.s2p)")
            s2p_freqs_hz = np.linspace(f_lo, f_hi, min(int(n_pts), 2001))
            H_s2p = self.eval_H(z_t, p_t, k_t, s2p_freqs_hz)
            s2p_text = self._build_s2p(s2p_freqs_hz, H_s2p, r0)
            st.download_button(
                "Download .s2p",
                data=s2p_text,
                file_name=f"{topology.lower().replace(' ', '')}_{ftype.lower()}_N{N}.s2p",
                mime="text/plain",
            )
            st.caption(
                "S21 comes directly from the transfer function. S11 is computed from "
                "|S11|² = 1 − |S21|² (lossless assumption), with phase set to 0°."
            )

    # ==========================================================================
    # Plotting
    # ==========================================================================
    @staticmethod
    def _mag_fig(freqs, H, ftype, f1_mhz, f2_mhz):
        mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-300))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs / 1e6, y=mag_db, mode="lines",
            line=dict(color=TRACE_MAG_COLOR, width=2),
            name="|H| (dB)",
        ))
        # Spec lines
        fig.add_hline(y=-3, line=dict(color=SPEC_LINE_COLOR, dash="dot"))
        if ftype in ("Bandpass", "Bandstop") and f2_mhz is not None:
            for edge in (f1_mhz, f2_mhz):
                fig.add_vline(x=edge, line=dict(color=SPEC_LINE_COLOR, dash="dot"))
        else:
            fig.add_vline(x=f1_mhz, line=dict(color=SPEC_LINE_COLOR, dash="dot"))
        fig.update_layout(
            title="Magnitude response",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Gain (dB)",
            xaxis_type="log",
            yaxis_range=[-100, 5],
            height=380,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig

    @staticmethod
    def _phase_fig(freqs, H):
        phase = np.unwrap(np.angle(H))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs / 1e6, y=np.rad2deg(phase), mode="lines",
            line=dict(color=TRACE_PHASE_COLOR, width=2),
            name="Phase",
        ))
        fig.update_layout(
            title="Phase response (unwrapped)",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase (°)",
            xaxis_type="log",
            height=380,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig

    @staticmethod
    def _group_delay(freqs, H):
        phase = np.unwrap(np.angle(H))
        omega = 2 * np.pi * freqs
        # Negative derivative of phase w.r.t. omega
        gd = -np.gradient(phase, omega)
        return gd

    def _group_delay_fig(self, freqs, H):
        gd_ns = self._group_delay(freqs, H) * 1e9
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs / 1e6, y=gd_ns, mode="lines",
            line=dict(color=TRACE_GD_COLOR, width=2),
            name="Group delay",
        ))
        fig.update_layout(
            title="Group delay",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Group delay (ns)",
            xaxis_type="log",
            height=340,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig

    # ==========================================================================
    # Transfer function evaluation
    # ==========================================================================
    @staticmethod
    def eval_H(z, p, k, freqs_hz):
        """Evaluate analog H(jω) on the given frequency grid. Uses scipy.signal.freqs."""
        b, a = zpk2tf(z, p, k)
        _, H = freqs(b, a, worN=2 * np.pi * np.asarray(freqs_hz, dtype=float))
        return H

    # ==========================================================================
    # LC ladder synthesis
    # ==========================================================================
    @staticmethod
    def butterworth_g(N):
        k_idx = np.arange(1, N + 1)
        return list(2 * np.sin((2 * k_idx - 1) * np.pi / (2 * N)))

    @staticmethod
    def chebyshev1_g(N, ripple_db):
        beta = np.log(1.0 / np.tanh(ripple_db / 17.37))
        gamma = np.sinh(beta / (2 * N))
        g = [0.0] * N
        a = [np.sin((2 * k - 1) * np.pi / (2 * N)) for k in range(1, N + 1)]
        b = [gamma ** 2 + np.sin(k * np.pi / N) ** 2 for k in range(1, N + 1)]
        g[0] = 2 * a[0] / gamma
        for k in range(2, N + 1):
            g[k - 1] = 4 * a[k - 2] * a[k - 1] / (b[k - 2] * g[k - 2])
        # Termination ratio
        if N % 2 == 1:
            g_last = 1.0
        else:
            g_last = (1.0 / np.tanh(beta / 4)) ** 2
        return g, g_last

    def _synthesize_ladder(self, topology, ftype, N, rip_db,
                           f1_mhz, f2_mhz, f0_mhz, bw_mhz, r0):
        """Return list of dicts: [{'kind': 'L'|'C', 'value': H|F, 'position': 'series'|'shunt', ...}]
        Source → Load order. Returns None if synthesis isn't supported for this topology."""
        if topology == "Butterworth":
            g = self.butterworth_g(N)
            r_ratio = 1.0
        elif topology == "Chebyshev I":
            g, r_ratio = self.chebyshev1_g(N, rip_db)
        elif topology == "Bessel":
            if N not in BESSEL_G:
                return None
            g = list(BESSEL_G[N])
            r_ratio = 1.0
        else:
            return None

        # Build the LPF prototype ladder: alternating series-L / shunt-C
        # First element is series inductor g[0]; alternates.
        if ftype == "Lowpass":
            wc = 2 * np.pi * f1_mhz * 1e6
            return self._ladder_lpf(g, r0, wc, r_ratio)
        if ftype == "Highpass":
            wc = 2 * np.pi * f1_mhz * 1e6
            return self._ladder_hpf(g, r0, wc, r_ratio)
        if ftype == "Bandpass":
            w0 = 2 * np.pi * f0_mhz * 1e6
            bw = 2 * np.pi * bw_mhz * 1e6
            return self._ladder_bpf(g, r0, w0, bw, r_ratio)
        if ftype == "Bandstop":
            w0 = 2 * np.pi * f0_mhz * 1e6
            bw = 2 * np.pi * bw_mhz * 1e6
            return self._ladder_bsf(g, r0, w0, bw, r_ratio)
        return None

    @staticmethod
    def _ladder_lpf(g, r0, wc, r_ratio):
        comps = []
        for i, gv in enumerate(g):
            if i % 2 == 0:  # series L
                L = gv * r0 / wc
                comps.append({"kind": "L", "value": L, "position": "series", "label": f"L{i + 1}"})
            else:           # shunt C
                C = gv / (r0 * wc)
                comps.append({"kind": "C", "value": C, "position": "shunt", "label": f"C{i + 1}"})
        return comps

    @staticmethod
    def _ladder_hpf(g, r0, wc, r_ratio):
        # LPF → HPF: series L → series C (value 1/(wc²·L)), shunt C → shunt L (value 1/(wc²·C))
        comps = []
        for i, gv in enumerate(g):
            if i % 2 == 0:  # series (was L, now C)
                C = 1.0 / (gv * r0 * wc)
                comps.append({"kind": "C", "value": C, "position": "series", "label": f"C{i + 1}"})
            else:           # shunt (was C, now L)
                L = r0 / (gv * wc)
                comps.append({"kind": "L", "value": L, "position": "shunt", "label": f"L{i + 1}"})
        return comps

    @staticmethod
    def _ladder_bpf(g, r0, w0, bw, r_ratio):
        # LPF series L → series L-C (tank in series); LPF shunt C → shunt L-C (parallel tank)
        comps = []
        for i, gv in enumerate(g):
            if i % 2 == 0:  # series branch: series L + series C (becomes series resonant)
                L = gv * r0 / bw
                C = bw / (gv * r0 * w0 ** 2)
                comps.append({"kind": "L", "value": L, "position": "series", "label": f"L{i + 1}s"})
                comps.append({"kind": "C", "value": C, "position": "series", "label": f"C{i + 1}s"})
            else:           # shunt branch: parallel L || C
                L = bw / (gv * w0 ** 2) * r0
                C = gv / (r0 * bw)
                comps.append({"kind": "L", "value": L, "position": "shunt", "label": f"L{i + 1}p"})
                comps.append({"kind": "C", "value": C, "position": "shunt", "label": f"C{i + 1}p"})
        return comps

    @staticmethod
    def _ladder_bsf(g, r0, w0, bw, r_ratio):
        # LPF series L → series (L || C); LPF shunt C → shunt (L series C)
        comps = []
        for i, gv in enumerate(g):
            if i % 2 == 0:  # series: parallel L || C
                L = gv * r0 * bw / (w0 ** 2)
                C = 1.0 / (gv * r0 * bw)
                comps.append({"kind": "L", "value": L, "position": "series", "label": f"L{i + 1}p"})
                comps.append({"kind": "C", "value": C, "position": "series", "label": f"C{i + 1}p"})
            else:           # shunt: series L+C
                L = r0 / (gv * bw)
                C = gv * bw / (r0 * w0 ** 2)
                comps.append({"kind": "L", "value": L, "position": "shunt", "label": f"L{i + 1}s"})
                comps.append({"kind": "C", "value": C, "position": "shunt", "label": f"C{i + 1}s"})
        return comps

    # ==========================================================================
    # Component rendering & export
    # ==========================================================================
    @staticmethod
    def _fmt_value(c):
        v = c["value"]
        if c["kind"] == "L":
            if v >= 1e-6:   return f"{v * 1e6:.3f} µH"
            if v >= 1e-9:   return f"{v * 1e9:.3f} nH"
            return f"{v * 1e12:.3f} pH"
        else:  # C
            if v >= 1e-9:   return f"{v * 1e9:.3f} nF"
            if v >= 1e-12:  return f"{v * 1e12:.3f} pF"
            return f"{v * 1e15:.3f} fF"

    def _render_component_table(self, components):
        rows = []
        for c in components:
            rows.append({
                "Ref": c["label"],
                "Kind": c["kind"],
                "Topology": c["position"].capitalize(),
                "Value": self._fmt_value(c),
                "SI (base units)": f"{c['value']:.6e} {'H' if c['kind'] == 'L' else 'F'}",
            })
        st.dataframe(rows, width='stretch', hide_index=True)

    @staticmethod
    def _components_to_csv(components, topology, ftype, r0, f1_mhz, f2_mhz, f0_mhz, bw_mhz):
        buf = StringIO()
        buf.write(f"# Topology,{topology}\n")
        buf.write(f"# Type,{ftype}\n")
        buf.write(f"# Z0 (ohm),{r0}\n")
        if ftype in ("Lowpass", "Highpass"):
            buf.write(f"# Cutoff (MHz),{f1_mhz}\n")
        else:
            buf.write(f"# f1 (MHz),{f1_mhz}\n")
            buf.write(f"# f2 (MHz),{f2_mhz}\n")
            buf.write(f"# f0 (MHz),{f0_mhz}\n")
            buf.write(f"# BW (MHz),{bw_mhz}\n")
        buf.write("Ref,Kind,Position,Value,Unit\n")
        for c in components:
            val_si = c["value"]
            unit = "H" if c["kind"] == "L" else "F"
            buf.write(f"{c['label']},{c['kind']},{c['position']},{val_si:.9e},{unit}\n")
        return buf.getvalue()

    @staticmethod
    def _build_s2p(freqs_hz, H, r0):
        # S21 = H(jω); |S11|² = 1 − |S21|² (lossless, reciprocal 2-port assumption)
        s21 = H
        s21_mag = np.abs(s21)
        s11_mag_sq = np.clip(1.0 - s21_mag ** 2, 0.0, 1.0)
        s11_mag = np.sqrt(s11_mag_sq)
        s21_ang_deg = np.rad2deg(np.angle(s21))
        # Leave S11 phase at 0 and S12 = S21 (symmetric passive assumption)
        out = StringIO()
        out.write("! Touchstone v1 file — generated by filter-designer.py\n")
        out.write(f"! Z0 = {r0} Ω\n")
        out.write("! Format: Frequency (Hz), S11 (MA), S21 (MA), S12 (MA), S22 (MA)\n")
        out.write(f"# HZ S MA R {r0}\n")
        for f, m11, m21, a21 in zip(freqs_hz, s11_mag, s21_mag, s21_ang_deg):
            out.write(
                f"{f:.6e} {m11:.6e} 0.0 "
                f"{m21:.6e} {a21:.4f} "
                f"{m21:.6e} {a21:.4f} "
                f"{m11:.6e} 0.0\n"
            )
        return out.getvalue()

    # ==========================================================================
    # Schematic rendering (ladder) — Source ── [...] ── Load
    # ==========================================================================
    L_COLOR = "#5b8def"
    C_COLOR = "#f97316"

    def render_ladder_svg(self, components, r0):
        """Render the LC ladder. Groups consecutive 'series' then 'shunt' branches into cells."""
        # Reorganize into branches: each branch is either all-series (stacked) or all-shunt (stacked)
        # For LPF/HPF: one component per branch. For BPF/BSF: two components per branch.
        branches = []
        i = 0
        while i < len(components):
            pos = components[i]["position"]
            grp = [components[i]]
            j = i + 1
            while j < len(components) and components[j]["position"] == pos and self._same_branch(components[i], components[j]):
                grp.append(components[j])
                j += 1
            branches.append(grp)
            i = j

        cell_w = 180
        margin_x = 120
        wire_y = 100
        term_r = 12
        side_padding = 80
        source_x = side_padding
        has_shunt = any(b[0]["position"] == "shunt" for b in branches)
        height = 420 if has_shunt else 200
        width = max(900, margin_x * 2 + len(branches) * cell_w)
        load_x = width - side_padding

        positions = [margin_x + i * cell_w + cell_w // 2 for i in range(len(branches))]
        series_half_w = 40

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            f'style="display:block;margin:0 auto;max-width:100%;height:auto;'
            f'font-family:-apple-system,Segoe UI,sans-serif;">'
        ]

        # Main wire
        prev_x = source_x
        for i, b in enumerate(branches):
            cx = positions[i]
            if b[0]["position"] == "series":
                parts.append(
                    f'<line x1="{prev_x}" y1="{wire_y}" x2="{cx - series_half_w * len(b)}" y2="{wire_y}" '
                    f'stroke="currentColor" stroke-width="3" stroke-linecap="round" opacity="0.7"/>'
                )
                prev_x = cx + series_half_w * len(b)
        parts.append(
            f'<line x1="{prev_x}" y1="{wire_y}" x2="{load_x}" y2="{wire_y}" '
            f'stroke="currentColor" stroke-width="3" stroke-linecap="round" opacity="0.7"/>'
        )

        # Terminals
        parts.append(
            f'<circle cx="{source_x}" cy="{wire_y}" r="{term_r}" fill="none" '
            f'stroke="currentColor" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{source_x}" y="{wire_y - 25}" text-anchor="middle" font-size="18" '
            f'font-weight="700" fill="currentColor">Source</text>'
        )
        parts.append(
            f'<text x="{source_x}" y="{wire_y + 45}" text-anchor="middle" font-size="16" '
            f'fill="currentColor" opacity="0.75">{r0:.0f} Ω</text>'
        )
        parts.append(f'<circle cx="{load_x}" cy="{wire_y}" r="{term_r}" fill="currentColor"/>')
        parts.append(
            f'<text x="{load_x}" y="{wire_y - 25}" text-anchor="middle" font-size="18" '
            f'font-weight="700" fill="currentColor">Load</text>'
        )
        parts.append(
            f'<text x="{load_x}" y="{wire_y + 45}" text-anchor="middle" font-size="16" '
            f'fill="currentColor" opacity="0.75">{r0:.0f} Ω</text>'
        )

        # Components
        for i, b in enumerate(branches):
            cx = positions[i]
            if b[0]["position"] == "series":
                # Draw each component inline, spaced along the wire
                count = len(b)
                total_span = count * 2 * series_half_w
                start_x = cx - total_span / 2
                for j, c in enumerate(b):
                    slot_cx = start_x + (j + 0.5) * 2 * series_half_w
                    color = self.L_COLOR if c["kind"] == "L" else self.C_COLOR
                    if c["kind"] == "L":
                        parts.extend(self._series_L_svg(slot_cx, wire_y, series_half_w, color))
                    else:
                        parts.extend(self._series_C_svg(slot_cx, wire_y, series_half_w, color))
                    label = f'{c["label"]} = {self._fmt_value(c)}'
                    parts.append(
                        f'<text x="{slot_cx}" y="{wire_y - 35}" text-anchor="middle" font-size="16" '
                        f'font-weight="600" fill="{color}">{label}</text>'
                    )
            else:  # shunt branch — stack components vertically from wire to ground
                top_y = wire_y
                segment_y = top_y
                for j, c in enumerate(b):
                    color = self.L_COLOR if c["kind"] == "L" else self.C_COLOR
                    if c["kind"] == "L":
                        svg_parts, next_y = self._shunt_L_segment(cx, segment_y, color)
                    else:
                        svg_parts, next_y = self._shunt_C_segment(cx, segment_y, color)
                    parts.extend(svg_parts)
                    label = f'{c["label"]} = {self._fmt_value(c)}'
                    parts.append(
                        f'<text x="{cx + 35}" y="{(segment_y + next_y) / 2 + 6}" text-anchor="start" '
                        f'font-size="16" font-weight="600" fill="{color}">{label}</text>'
                    )
                    segment_y = next_y
                # Ground symbol at bottom
                gy = segment_y + 40
                parts.append(
                    f'<line x1="{cx}" y1="{segment_y}" x2="{cx}" y2="{gy}" '
                    f'stroke="currentColor" stroke-width="3" stroke-linecap="round"/>'
                )
                parts.append(
                    f'<line x1="{cx - 15}" y1="{gy}" x2="{cx + 15}" y2="{gy}" '
                    f'stroke="currentColor" stroke-width="2.5" opacity="0.7"/>'
                )
                parts.append(
                    f'<line x1="{cx - 10}" y1="{gy + 8}" x2="{cx + 10}" y2="{gy + 8}" '
                    f'stroke="currentColor" stroke-width="2.5" opacity="0.7"/>'
                )
                parts.append(
                    f'<line x1="{cx - 5}" y1="{gy + 16}" x2="{cx + 5}" y2="{gy + 16}" '
                    f'stroke="currentColor" stroke-width="2.5" opacity="0.7"/>'
                )

        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _same_branch(a, b):
        """Two consecutive components are in the same branch if they share a branch label stem
        (e.g. L1s and C1s, or L1p and C1p). For LPF/HPF the labels are unique → always separate."""
        la, lb = a.get("label", ""), b.get("label", "")
        if not la or not lb:
            return False
        return la[:-1] == lb[:-1] and la[-1] in ("s", "p") and lb[-1] == la[-1]

    @staticmethod
    def _series_L_svg(cx, cy, half_w, color):
        r = 8
        coil_w = 8 * r
        x0 = cx - coil_w / 2
        x1 = cx + coil_w / 2
        path = f"M {x0},{cy} " + " ".join(f"a {r},{r} 0 0,1 {2 * r},0" for _ in range(4))
        return [
            f'<line x1="{cx - half_w}" y1="{cy}" x2="{x0}" y2="{cy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>',
            f'<line x1="{x1}" y1="{cy}" x2="{cx + half_w}" y2="{cy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
        ]

    @staticmethod
    def _series_C_svg(cx, cy, half_w, color):
        gap = 9
        plate_h = 32
        return [
            f'<line x1="{cx - half_w}" y1="{cy}" x2="{cx - gap}" y2="{cy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<line x1="{cx - gap}" y1="{cy - plate_h / 2}" x2="{cx - gap}" y2="{cy + plate_h / 2}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
            f'<line x1="{cx + gap}" y1="{cy - plate_h / 2}" x2="{cx + gap}" y2="{cy + plate_h / 2}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
            f'<line x1="{cx + gap}" y1="{cy}" x2="{cx + half_w}" y2="{cy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
        ]

    @staticmethod
    def _shunt_L_segment(cx, y_top, color):
        stub = 40
        r = 8
        coil_h = 8 * r
        y_start = y_top + stub
        y_end = y_start + coil_h
        path = f"M {cx},{y_start} " + " ".join(f"a {r},{r} 0 0,0 0,{2 * r}" for _ in range(4))
        return [
            f'<line x1="{cx}" y1="{y_top}" x2="{cx}" y2="{y_start}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>',
        ], y_end

    @staticmethod
    def _shunt_C_segment(cx, y_top, color):
        stub = 45
        plate_gap = 16
        plate_w = 32
        y_p1 = y_top + stub
        y_p2 = y_p1 + plate_gap
        return [
            f'<line x1="{cx}" y1="{y_top}" x2="{cx}" y2="{y_p1}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<line x1="{cx - plate_w / 2}" y1="{y_p1}" x2="{cx + plate_w / 2}" y2="{y_p1}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
            f'<line x1="{cx - plate_w / 2}" y1="{y_p2}" x2="{cx + plate_w / 2}" y2="{y_p2}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
        ], y_p2


if __name__ == "__main__":
    FilterDesigner().run()
