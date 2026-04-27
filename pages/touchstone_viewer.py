"""
# Name:        touchstone_viewer.py
# Purpose:     streamlit touchston viewer app
#
# Author:      Yannish RAMGULAM
#
# Copyright:   (c) Y.Ramgulam 2026
# Licence:     CC-BY-4.0
#
#
"""

import streamlit as st
import skrf as rf
import plotly.graph_objects as go
import numpy as np
import tempfile
import os

class TouchStoneViewer():
    def __init__(self):
        pass

    def run(self):

        st.set_page_config(page_title="RF Touchstone Viewer", layout="wide")

        st.title("📡 Touchstone File Viewer")
        st.write("Upload an .s1p or .s2p file to visualize S-parameters.")

        uploaded_file = st.file_uploader("Choose a Touchstone file", type=["s1p", "s2p"])

        if uploaded_file is not None:
            # scikit-rf requires a file path, so we save the uploaded buffer to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # Load the network
                network = rf.Network(tmp_path)
                os.unlink(tmp_path)  # Clean up temp file

                st.success(f"Loaded: {network.name}")
                
                # Selection for S-parameters
                # Generate labels like 'S11', 'S21' based on port count
                num_ports = network.nports
                params = []
                for i in range(num_ports):
                    for j in range(num_ports):
                        params.append(f"S{i+1}{j+1}")
                
                selected_param = st.selectbox("Select Parameter", params)
                
                # Map string selection back to indices
                row = int(selected_param[1]) - 1
                col = int(selected_param[2]) - 1
                
                # Extract data
                # Robust frequency auto-scaling: scikit-rf stores .f in Hz regardless of source.
                # We detect the best unit (Hz, kHz, MHz, GHz) based on the maximum frequency.
                f_max = np.max(network.f)
                if f_max >= 1e9:
                    f_factor, f_unit = 1e9, "GHz"
                elif f_max >= 1e6:
                    f_factor, f_unit = 1e6, "MHz"
                elif f_max >= 1e3:
                    f_factor, f_unit = 1e3, "kHz"
                else:
                    f_factor, f_unit = 1.0, "Hz"
                    
                f_disp = network.f / f_factor
                s_data = network.s[:, row, col]
                mag_db = 20 * np.log10(np.abs(s_data))
                ang_deg = np.angle(s_data, deg=True)
                z_data = network.z[:, row, col]
                z_str = [f"{z.real:.2f}{z.imag:+.2f}j" for z in z_data]
                
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"Magnitude (dB) - {selected_param}")
                    fig_mag = go.Figure()
                    
                    mag_hover = [f"{f:.2f} {f_unit}<br>{m:.2f} dB" for f, m in zip(f_disp, mag_db)]
                    
                    fig_mag.add_trace(go.Scatter(
                        x=f_disp, y=mag_db, mode='lines', name=selected_param,
                        text=mag_hover,
                        hovertemplate="%{text}<extra></extra>"
                    ))
                    fig_mag.update_layout(
                        xaxis_title=f"Frequency ({f_unit})",
                        yaxis_title="Magnitude (dB)",
                        template="plotly_dark",
                        hovermode="x",
                        height=600
                    )
                    st.plotly_chart(fig_mag, width='stretch')

                with col2:
                    st.subheader(f"Smith Chart - {selected_param}")
                    
                    fig_smith = go.Figure()

                    # --- Smith Chart Grid ---
                    grid_color = "rgba(150, 150, 150, 0.3)"
                    
                    # -- Constant Resistance Circles (r)
                    # These are naturally contained within the unit circle for r >= 0
                    rs = [0, 0.2, 0.5, 1.0, 2.0, 5.0]
                    for r in rs:
                        radius = 1 / (r + 1)
                        center_x = r / (r + 1)
                        theta_g = np.linspace(0, 2*np.pi, 100)
                        fig_smith.add_trace(go.Scatter(
                            x=center_x + radius * np.cos(theta_g),
                            y=radius * np.sin(theta_g),
                            mode='lines',
                            line=dict(color=grid_color, width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                    # -- Constant Reactance Arcs (x)
                    # We calculate intersection with the unit circle to keep arcs inside
                    xs = [0.2, 0.5, 1.0, 2.0, 5.0]
                    for x in xs:
                        for sign in [1, -1]:
                            x_val = x * sign
                            center_y = 1 / x_val
                            radius = 1 / abs(x_val)
                            
                            # Intersection point with the unit circle (u^2 + v^2 = 1)
                            u_int = (x_val**2 - 1) / (x_val**2 + 1)
                            v_int = 2 * x_val / (x_val**2 + 1)
                            
                            # Calculate angles from the circle center (1, 1/x)
                            angle_end = np.arctan2(v_int - center_y, u_int - 1) % (2 * np.pi)
                            
                            if sign > 0:
                                # Upper arcs: from intersection to the point (1,0) at 270 deg
                                t = np.linspace(angle_end, 1.5 * np.pi, 60)
                            else:
                                # Lower arcs: from the point (1,0) at 90 deg to intersection
                                t = np.linspace(0.5 * np.pi, angle_end, 60)

                            fig_smith.add_trace(go.Scatter(
                                x=1 + radius * np.cos(t),
                                y=center_y + radius * np.sin(t),
                                mode='lines',
                                line=dict(color=grid_color, width=1),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                    
                    # -- Real Axis (x=0 line)
                    fig_smith.add_trace(go.Scatter(
                        x=[-1, 1], y=[0, 0],
                        mode='lines',
                        line=dict(color=grid_color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # --- Data Trace ---
                    real = np.real(s_data)
                    imag = np.imag(s_data)
                    
                    smith_hover = [
                        f"f = {f:.6g} {f_unit}<br>"
                        f"Z = {zs} Ω<br>"
                        f"|{selected_param}| = {mdb:.2f} dB<br>"
                        f"∠{selected_param} = {adeg:.2f}°"
                        for f, zs, mdb, adeg in zip(f_disp, z_str, mag_db, ang_deg)
                    ]

                    fig_smith.add_trace(go.Scatter(
                        x=real, y=imag, 
                        mode='lines', 
                        name=selected_param,
                        line=dict(color='orange', width=2),
                        text=smith_hover,
                        hovertemplate="%{text}<extra></extra>"
                    ))

                    fig_smith.update_layout(
                        xaxis=dict(range=[-1.1, 1.1], constraintoward='center', scaleanchor="y", scaleratio=1, 
                                showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(range=[-1.1, 1.1], showgrid=False, zeroline=False, visible=False),
                        width=600,
                        height=600,
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    
                    st.plotly_chart(fig_smith, width='stretch')

                # Display raw data summary
                with st.expander("View Network Summary"):
                    st.text(str(network))

            except Exception as e:
                st.error(f"Error processing file: {e}")

        else:
            st.info("Please upload a .s1p or .s2p file to begin.")