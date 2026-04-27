import streamlit as st
import numpy as np
import math
from pages.smith_component import smith_chart


TRACE_MEASURED_COLOR = "#08CE30"   # violet — raw S11 / reference
TRACE_MATCHED_COLOR  = "#0891b2"   # teal   — post-match simulated S11


class RfMatcher():
    """Smith-chart L-section matcher with optional S1P load."""

    def __init__(self):
        ss = st.session_state
        if "points" not in ss:           ss.points = []          # committed Z values
        if "step" not in ss:             ss.step = 1
        if "last_click_id" not in ss:    ss.last_click_id = 0
        if "s1p" not in ss:              ss.s1p = None           # dict or None
        if "_last_freq_hz" not in ss:    ss._last_freq_hz = None
        if "_last_s1p_name" not in ss:   ss._last_s1p_name = None

        try:
            st.set_page_config(page_title="Smith Chart Matcher", layout="wide")
        except Exception:
            # Host app already called set_page_config — safe to skip when embedded.
            pass

    # ==========================================================================
    # UI
    # ==========================================================================
    def run(self):
        st.title("🎯 Simple RF Matcher")

        st.info("""
        **Quick Guide**
        - **Place Load**: Upload an **S1P file** or click the chart to place the Load manually.
        - **Series/Shunt**: Move the mouse to preview paths; click to commit components. Sections alternate.
        - **Undo**: **Right-click** the chart to remove the latest node.
        """)

        # ---- Controls row ---------------------------------------------------
        ctl_freq, ctl_btn, _ = st.columns([1, 0.33, 2.67])
        with ctl_freq:
            freq_mhz = st.number_input("Target Frequency (MHz)", value=2400.0, min_value=1.0, step=1.0)
            freq_hz = freq_mhz * 1e6
            show_vswr = st.radio(
                "VSWR circles (1.5 · 2 · 3)",
                ["Off", "On"], index=0, horizontal=True,
            ) == "On"
        with ctl_btn:
            st.markdown("<div style='height:1.7em'></div>", unsafe_allow_html=True)
            if st.button("Reset", width='stretch'):
                st.session_state.points = []
                st.session_state.step = 1
                st.session_state.last_click_id = 0
                if "smith_top" in st.session_state:
                    del st.session_state["smith_top"]
                st.rerun()

        # ---- S1P uploader ---------------------------------------------------
        uploaded = st.file_uploader(
            "Load S1P file (optional — if empty, click the chart to place the Load manually)",
            type=["s1p"],
        )
        self._sync_s1p_state(uploaded)

        # ---- Freq-change wipe (only when S1P loaded) ------------------------
        prev_freq = st.session_state._last_freq_hz
        st.session_state._last_freq_hz = freq_hz
        if st.session_state.s1p is not None and prev_freq is not None and abs(prev_freq - freq_hz) > 1e-3:
            st.session_state.points = []

        # ---- Auto-place Load from S1P at target freq ------------------------
        if st.session_state.s1p is not None and not st.session_state.points:
            s1p = st.session_state.s1p
            idx = int(np.argmin(np.abs(s1p["freqs"] - freq_hz)))
            s11_at_target = complex(s1p["s11"][idx])
            z_load = s1p["z0"] * (1 + s11_at_target) / (1 - s11_at_target)
            st.session_state.points.append(complex(float(z_load.real), float(z_load.imag)))

        # Step always derived from current point count.
        st.session_state.step = self._step_from_count(len(st.session_state.points))

        # ---- Top chart ------------------------------------------------------
        trace_measured, trace_highlight_idx = self._measured_trace(freq_hz)

        step_titles = {1: "Place Load", 2: "Series Match (Const R)", 3: "Shunt Match (Const G)"}
        st.subheader(f"Step {st.session_state.step}: {step_titles.get(st.session_state.step, '')}")

        nodes_for_js = [[z.real, z.imag] for z in st.session_state.points]
        result = smith_chart(
            nodes=nodes_for_js, freq_mhz=freq_mhz,
            step=st.session_state.step, show_vswr=show_vswr,
            trace=trace_measured,
            trace_color=TRACE_MEASURED_COLOR,
            trace_highlight_idx=trace_highlight_idx,
            key="smith_top",
        )

        self._handle_click(result)

        # ---- Node table -----------------------------------------------------
        if st.session_state.points:
            st.markdown("### Circuit Nodes")
            cols = st.columns(len(st.session_state.points))
            for idx, point in enumerate(st.session_state.points):
                with cols[idx]:
                    sign = "+" if point.imag >= 0 else "-"
                    q = self.node_q(point)
                    q_str = "∞" if math.isinf(q) else f"{q:.2f}"
                    body = (
                        f"**Node {idx}**\n\n"
                        f"Z = {point.real:.1f} {sign} j{abs(point.imag):.1f} Ω\n\n"
                        f"Q = {q_str}\n\n"
                        f"S11 = {self.get_s11_db(point):.2f} dB"
                    )
                    if idx > 0:
                        prev = st.session_state.points[idx - 1]
                        is_series = (idx % 2 != 0)
                        body += f"\n\n{'Series' if is_series else 'Shunt'}: {self.calc_component(prev, point, is_series, freq_hz)}"
                    st.info(body)

        # ---- Matching-network schematic -------------------------------------
        components = self.build_component_list(st.session_state.points, freq_hz)
        if components:
            # build_component_list returns Load→Source; schematic draws Source→Load.
            st.markdown("### Matching Network")
            st.markdown(
                self.render_schematic_svg(list(reversed(components)), st.session_state.points[0]),
                unsafe_allow_html=True,
            )

        # ---- Bottom chart: simulated matched response -----------------------
        if st.session_state.s1p is not None:
            st.markdown("---")
            st.subheader("Simulated Matched Response (across full S1P sweep)")
            s1p = st.session_state.s1p
            matched_gamma = self.simulate_matched_s11(
                s1p["freqs"], np.asarray(s1p["s11"], dtype=complex), components, z0=s1p["z0"],
            )
            trace_matched = self._sanitize_trace(matched_gamma)
            smith_chart(
                nodes=[], freq_mhz=freq_mhz, step=1,
                show_vswr=show_vswr,
                trace=trace_matched,
                trace_color=TRACE_MATCHED_COLOR,
                trace_highlight_idx=trace_highlight_idx,
                trace_ref=trace_measured,
                trace_ref_color=TRACE_MEASURED_COLOR,
                readonly=True,
                key="smith_bottom",
            )

    # ==========================================================================
    # S1P + state helpers
    # ==========================================================================
    def _sync_s1p_state(self, uploaded):
        """Reconcile file-uploader state with session_state.s1p. Wipes points on change."""
        ss = st.session_state
        if uploaded is not None:
            if ss._last_s1p_name != uploaded.name:
                try:
                    text = uploaded.read().decode("utf-8", errors="ignore")
                    freqs, s11, z0 = self.parse_s1p(text)
                except Exception as e:
                    st.error(f"Failed to parse S1P: {e}")
                    return
                if len(freqs) == 0:
                    st.error("Invalid S1P: no data rows found.")
                    return
                ss.s1p = {"freqs": freqs, "s11": s11, "z0": z0, "name": uploaded.name}
                ss._last_s1p_name = uploaded.name
                ss.points = []
        elif ss._last_s1p_name is not None:
            # User cleared the uploader — fall back to manual flow.
            ss.s1p = None
            ss._last_s1p_name = None
            ss.points = []

    def _measured_trace(self, freq_hz):
        """Returns (trace_as_list_of_[gx,gy], highlight_idx) or ([], -1) when no S1P."""
        s1p = st.session_state.s1p
        if s1p is None:
            return [], -1
        gamma50 = self._s11_to_gamma50(np.asarray(s1p["s11"], dtype=complex), s1p["z0"])
        trace = self._sanitize_trace(gamma50)
        idx = int(np.argmin(np.abs(s1p["freqs"] - freq_hz)))
        # Map idx onto the (potentially filtered) trace: only valid if we kept all points.
        # Sanitize shouldn't drop anything for well-formed S1P data.
        return trace, idx if idx < len(trace) else -1

    @staticmethod
    def _s11_to_gamma50(s11_array, z0):
        if abs(z0 - 50.0) < 1e-9:
            return s11_array
        z = z0 * (1 + s11_array) / (1 - s11_array)
        return (z - 50.0) / (z + 50.0)

    @staticmethod
    def _sanitize_trace(gamma_array):
        pts = []
        for g in gamma_array:
            if np.isfinite(g.real) and np.isfinite(g.imag):
                pts.append([float(g.real), float(g.imag)])
        return pts

    # ==========================================================================
    # Click handling
    # ==========================================================================
    def _handle_click(self, result):
        if not result:
            return
        click_id = int(result.get("click_id", 0))
        if click_id <= st.session_state.last_click_id:
            return
        st.session_state.last_click_id = click_id
        action = result.get("type", "click")
        has_s1p = st.session_state.s1p is not None

        if action == "undo":
            # Don't let right-click delete the S1P-derived Load.
            min_points = 1 if has_s1p else 0
            if len(st.session_state.points) > min_points:
                st.session_state.points.pop()

        else:  # click → place next node
            g_click = complex(float(result["x"]), float(result["y"]))
            z_raw = self.gamma_to_z(g_click)
            step = st.session_state.step
            if step == 1:
                # Manual Load placement — only allowed when no S1P is loaded.
                if not has_s1p:
                    st.session_state.points.append(z_raw)
            elif step == 2:
                prev_z = st.session_state.points[-1]
                st.session_state.points.append(complex(prev_z.real, z_raw.imag))
            elif step == 3:
                prev_y = 1 / st.session_state.points[-1]
                raw_y = 1 / z_raw
                constrained_y = complex(prev_y.real, raw_y.imag)
                st.session_state.points.append(1 / constrained_y)

        st.session_state.step = self._step_from_count(len(st.session_state.points))
        st.rerun()

    # ==========================================================================
    # S1P parser (Touchstone v1, 1-port)
    # ==========================================================================
    @staticmethod
    def parse_s1p(text):
        """Parse a Touchstone v1 .s1p file. Returns (freqs_hz, s11_complex, z0)."""
        freq_unit = "GHZ"   # Touchstone spec default
        fmt = "MA"          # spec default
        z0 = 50.0
        unit_map = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
        freqs, s11 = [], []
        for raw in text.splitlines():
            line = raw.split("!", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                toks = line[1:].upper().split()
                i = 0
                while i < len(toks):
                    t = toks[i]
                    if t in unit_map:
                        freq_unit = t
                    elif t in ("MA", "DB", "RI"):
                        fmt = t
                    elif t == "R" and i + 1 < len(toks):
                        z0 = float(toks[i + 1])
                        i += 1
                    # "S" (parameter type) and other tokens are ignored
                    i += 1
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            f_scale = unit_map.get(freq_unit, 1.0)
            f = float(parts[0]) * f_scale
            a, b = float(parts[1]), float(parts[2])
            if fmt == "MA":
                s = a * np.exp(1j * np.deg2rad(b))
            elif fmt == "DB":
                s = (10 ** (a / 20)) * np.exp(1j * np.deg2rad(b))
            else:  # RI
                s = complex(a, b)
            freqs.append(f)
            s11.append(s)
        return np.asarray(freqs, dtype=float), np.asarray(s11, dtype=complex), float(z0)

    # ==========================================================================
    # Matching-network simulation
    # ==========================================================================
    @staticmethod
    def build_component_list(points, freq_hz):
        """Reconstruct L/C components from the committed node sequence.
        Order is Load→Source, matching the Smith-chart walk."""
        if len(points) < 2:
            return []
        omega = 2 * np.pi * freq_hz
        comps = []
        for i in range(1, len(points)):
            prev, cur = points[i - 1], points[i]
            is_series = (i % 2 != 0)
            if is_series:
                dx = cur.imag - prev.imag
                if dx > 1e-12:
                    comps.append({"kind": "L", "value": dx / omega, "position": "series"})
                elif dx < -1e-12:
                    comps.append({"kind": "C", "value": 1.0 / (omega * abs(dx)), "position": "series"})
            else:
                db = (1 / cur).imag - (1 / prev).imag
                if db > 1e-12:
                    comps.append({"kind": "C", "value": db / omega, "position": "shunt"})
                elif db < -1e-12:
                    comps.append({"kind": "L", "value": 1.0 / (omega * abs(db)), "position": "shunt"})
        return comps

    @staticmethod
    def simulate_matched_s11(freqs_hz, s11_array, components, z0=50.0):
        """Apply components (Load→Source order) to S11(f) and return Γ at 50 Ω."""
        z_cur = z0 * (1 + s11_array) / (1 - s11_array)
        omega = 2 * np.pi * freqs_hz
        for c in components:
            if c["kind"] == "L":
                z_comp = 1j * omega * c["value"]
            else:  # C
                z_comp = 1.0 / (1j * omega * c["value"])
            if c["position"] == "series":
                z_cur = z_cur + z_comp
            else:  # shunt — combine admittances
                z_cur = 1.0 / (1.0 / z_cur + 1.0 / z_comp)
        return (z_cur - 50.0) / (z_cur + 50.0)

    # ==========================================================================
    # Pure math (per-node)
    # ==========================================================================
    def z_to_gamma(self, z, z0=50.0):
        if z == complex(float("inf"), float("inf")):
            return complex(1, 0)
        return (z - z0) / (z + z0)

    def gamma_to_z(self, g, z0=50.0):
        if abs(1 - g) < 1e-6:
            return complex(1e6, 1e6)
        return z0 * (1 + g) / (1 - g)

    def get_s11_db(self, z):
        g = self.z_to_gamma(z)
        mag = abs(g)
        return 20 * math.log10(mag) if mag > 1e-10 else -100.0

    def node_q(self, z):
        # Nodal Q = |X| / R. Equals |B|/G in admittance — identical number for either view.
        if abs(z.real) < 1e-9:
            return float("inf")
        return abs(z.imag) / abs(z.real)

    def calc_component(self, z_start, z_end, is_series, freq_hz):
        omega = 2 * np.pi * freq_hz
        if is_series:
            delta_x = z_end.imag - z_start.imag
            if delta_x > 0:
                return f"L = {delta_x / omega * 1e9:.2f} nH"
            if delta_x < 0:
                return f"C = {1 / (omega * abs(delta_x)) * 1e12:.2f} pF"
            return "—"
        y_start, y_end = 1 / z_start, 1 / z_end
        delta_b = y_end.imag - y_start.imag
        if delta_b > 0:
            return f"C = {delta_b / omega * 1e12:.2f} pF"
        if delta_b < 0:
            return f"L = {1 / (omega * abs(delta_b)) * 1e9:.2f} nH"
        return "—"

    def _step_from_count(self, n):
        # 0 points → place Load; odd → next is Series; even (>0) → next is Shunt
        if n == 0:
            return 1
        return 2 if n % 2 == 1 else 3

    # ==========================================================================
    # Schematic (SVG) — Source ──[components]── Load
    # ==========================================================================
    L_COLOR = "#5b8def"
    C_COLOR = "#f97316"

    def render_schematic_svg(self, components_s2l, load_z):
        """SVG schematic string. components_s2l is Source→Load order."""
        n = len(components_s2l)
        if n == 0:
            return ""

        # Layout constants
        cell_w = 180
        margin_x = 120
        wire_y = 100
        term_r = 12
        side_padding = 80
        source_x = side_padding
        width = max(900, margin_x * 2 + n * cell_w)
        load_x = width - side_padding
        has_shunt = any(c["position"] == "shunt" for c in components_s2l)
        height = 300 if has_shunt else 200

        positions = [margin_x + i * cell_w + cell_w // 2 for i in range(n)]
        series_half_w = 40

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            f'style="display:block;margin:0 auto;max-width:100%;height:auto;'
            f'font-family:-apple-system,Segoe UI,sans-serif;">'
        ]

        # Main wire: continuous, broken at each series component
        prev_x = source_x
        for i, c in enumerate(components_s2l):
            cx = positions[i]
            if c["position"] == "series":
                parts.append(
                    f'<line x1="{prev_x}" y1="{wire_y}" x2="{cx - series_half_w}" y2="{wire_y}" '
                    f'stroke="currentColor" stroke-width="3.5" stroke-linecap="round" opacity="0.7"/>'
                )
                prev_x = cx + series_half_w
        parts.append(
            f'<line x1="{prev_x}" y1="{wire_y}" x2="{load_x}" y2="{wire_y}" '
            f'stroke="currentColor" stroke-width="3.5" stroke-linecap="round" opacity="0.7"/>'
        )

        # Source terminal (open circle + label + 50 Ω)
        parts.append(
            f'<circle cx="{source_x}" cy="{wire_y}" r="{term_r}" fill="none" '
            f'stroke="currentColor" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{source_x}" y="{wire_y - 30}" text-anchor="middle" font-size="18" '
            f'font-weight="700" fill="currentColor">Source</text>'
        )
        parts.append(
            f'<text x="{source_x}" y="{wire_y + 45}" text-anchor="middle" font-size="16" '
            f'fill="currentColor" opacity="0.75">50 Ω</text>'
        )

        # Load terminal (filled circle + Z label)
        parts.append(
            f'<circle cx="{load_x}" cy="{wire_y}" r="{term_r}" fill="currentColor"/>'
        )
        parts.append(
            f'<text x="{load_x}" y="{wire_y - 30}" text-anchor="middle" font-size="18" '
            f'font-weight="700" fill="currentColor">Load</text>'
        )
        sign = "+" if load_z.imag >= 0 else "-"
        parts.append(
            f'<text x="{load_x}" y="{wire_y + 45}" text-anchor="middle" font-size="16" '
            f'fill="currentColor" opacity="0.75">{load_z.real:.0f} {sign} j{abs(load_z.imag):.0f} Ω</text>'
        )

        # Components + labels
        for i, c in enumerate(components_s2l):
            cx = positions[i]
            color = self.L_COLOR if c["kind"] == "L" else self.C_COLOR
            label = f'{c["kind"]} = {self._fmt_component_value(c)}'

            if c["position"] == "series":
                if c["kind"] == "L":
                    parts.extend(self._series_L_svg(cx, wire_y, series_half_w, color))
                else:
                    parts.extend(self._series_C_svg(cx, wire_y, series_half_w, color))
                parts.append(
                    f'<text x="{cx}" y="{wire_y - 40}" text-anchor="middle" font-size="16" '
                    f'font-weight="700" fill="{color}">{label}</text>'
                )
            else:  # shunt
                if c["kind"] == "L":
                    parts.extend(self._shunt_L_svg(cx, wire_y, color))
                else:
                    parts.extend(self._shunt_C_svg(cx, wire_y, color))
                parts.append(
                    f'<text x="{cx + 35}" y="{wire_y + 85}" text-anchor="start" font-size="16" '
                    f'font-weight="700" fill="{color}">{label}</text>'
                )

        parts.append('</svg>')
        return "\n".join(parts)

    @staticmethod
    def _fmt_component_value(c):
        if c["kind"] == "L":
            return f"{c['value'] * 1e9:.2f} nH"
        return f"{c['value'] * 1e12:.2f} pF"

    @staticmethod
    def _series_L_svg(cx, cy, half_w, color):
        """Inline inductor — 4 semicircular humps + short stubs."""
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
        """Inline capacitor — two vertical plates with small gap."""
        gap = 9
        plate_h = 32
        return [
            f'<line x1="{cx - half_w}" y1="{cy}" x2="{cx - gap}" y2="{cy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<line x1="{cx - gap}" y1="{cy - plate_h / 2}" x2="{cx - gap}" y2="{cy + plate_h / 2}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
            f'<line x1="{cx + gap}" y1="{cy - plate_h / 2}" x2="{cx + gap}" y2="{cy + plate_h / 2}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
            f'<line x1="{cx + gap}" y1="{cy}" x2="{cx + half_w}" y2="{cy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
        ]

    @staticmethod
    def _shunt_L_svg(cx, wire_y, color):
        """Shunt inductor dropping to ground — vertical coil with stubs on both sides."""
        tap_h = 40        # stub between main wire and coil
        bot_stub = 40     # stub between coil and ground
        r = 8
        coil_h = 8 * r
        y_start = wire_y + tap_h
        y_end = y_start + coil_h
        path = f"M {cx},{y_start} " + " ".join(f"a {r},{r} 0 0,0 0,{2 * r}" for _ in range(4))
        gy = y_end + bot_stub
        return [
            f'<line x1="{cx}" y1="{wire_y}" x2="{cx}" y2="{y_start}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>',
            f'<line x1="{cx}" y1="{y_end}" x2="{cx}" y2="{gy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            # Ground symbol: three horizontal lines of decreasing width
            f'<line x1="{cx - 15}" y1="{gy}" x2="{cx + 15}" y2="{gy}" stroke="currentColor" stroke-width="2.5" opacity="0.7"/>',
            f'<line x1="{cx - 10}" y1="{gy + 6}" x2="{cx + 10}" y2="{gy + 6}" stroke="currentColor" stroke-width="2.5" opacity="0.7"/>',
            f'<line x1="{cx - 5}" y1="{gy + 12}" x2="{cx + 5}" y2="{gy + 12}" stroke="currentColor" stroke-width="2.5" opacity="0.7"/>',
        ]

    @staticmethod
    def _shunt_C_svg(cx, wire_y, color):
        """Shunt capacitor dropping to ground — two horizontal plates with stubs on both sides."""
        tap_h = 50       # stub between main wire and top plate
        plate_gap = 16    # separation between plates
        bot_stub = 50     # stub between bottom plate and ground
        plate_w = 32
        y_top = wire_y + tap_h
        y_bot = y_top + plate_gap
        gy = y_bot + bot_stub
        return [
            f'<line x1="{cx}" y1="{wire_y}" x2="{cx}" y2="{y_top}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<line x1="{cx - plate_w / 2}" y1="{y_top}" x2="{cx + plate_w / 2}" y2="{y_top}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
            f'<line x1="{cx - plate_w / 2}" y1="{y_bot}" x2="{cx + plate_w / 2}" y2="{y_bot}" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>',
            f'<line x1="{cx}" y1="{y_bot}" x2="{cx}" y2="{gy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<line x1="{cx - 15}" y1="{gy}" x2="{cx + 15}" y2="{gy}" stroke="currentColor" stroke-width="2.5" opacity="0.7"/>',
            f'<line x1="{cx - 10}" y1="{gy + 6}" x2="{cx + 10}" y2="{gy + 6}" stroke="currentColor" stroke-width="2.5" opacity="0.7"/>',
            f'<line x1="{cx - 5}" y1="{gy + 12}" x2="{cx + 5}" y2="{gy + 12}" stroke="currentColor" stroke-width="2.5" opacity="0.7"/>',
        ]


if __name__ == "__main__":
    RfMatcher().run()
