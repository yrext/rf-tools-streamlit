import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import simpson

class AntennaPatternViewer:
    def __init__(self):
        self.c = 299792458  # Speed of light

    def calc_pattern(self, freq_ghz, L_mm, W_mm, h_mm, er, N, M, dx_mm, dy_mm, theta_scan_deg, phi_scan_deg):
        lam = (self.c / (freq_ghz * 1e9)) * 1000  # wavelength in mm
        k = 2 * np.pi / lam
        
        # Theta/Phi grid
        theta = np.linspace(0.01, np.pi/2, 91) # Hemisphere only for patch
        phi = np.linspace(0, 2*np.pi, 181)
        TH, PH = np.meshgrid(theta, phi)

        # Scan direction unit vector components
        th_s = np.radians(theta_scan_deg)
        ph_s = np.radians(phi_scan_deg)
        u_s = np.sin(th_s) * np.cos(ph_s)
        v_s = np.sin(th_s) * np.sin(ph_s)

        # --- Single Element Pattern (Simplified Patch Model) ---
        # E-plane (phi=0) and H-plane (phi=90) approximations
        X = (k * h_mm / 2) * np.sin(TH) * np.cos(PH)
        Y = (k * W_mm / 2) * np.sin(TH) * np.sin(PH)
        
        # Element Factor
        # We use a standard approximation for a rectangular patch element
        element_pattern = np.abs(np.cos(k * L_mm / 2 * np.sin(TH) * np.cos(PH)) * np.sinc(Y/np.pi)) * np.cos(TH)

        # --- Array Factor ---
        af = np.zeros_like(TH, dtype=complex)
        for n in range(N):
            for m in range(M):
                # Apply progressive phase shift for beamsteering
                phase = k * (
                    n * dx_mm * (np.sin(TH) * np.cos(PH) - u_s) + 
                    m * dy_mm * (np.sin(TH) * np.sin(PH) - v_s))
                af += np.exp(1j * phase)
        
        af_abs = np.abs(af)
        total_pattern = element_pattern * af_abs
        
        # Normalize
        total_pattern_lin = total_pattern / np.max(total_pattern)
        total_pattern_db = 20 * np.log10(total_pattern_lin + 1e-9)
        
        return theta, phi, total_pattern_lin, total_pattern_db, lam

    def run(self):
        st.header("📡 Antenna Pattern Viewer (Patch Array)")
        
        st.info("""
        **How to use:**
        * **Set Frequency**: Adjust the operating frequency to see wavelength effects.
        * **Define Patch & Grid**: Set the physical dimensions (L, W) and array configuration (NxM).
        * **Analyze**: Explore the 3D gain pattern and check 2D polar cuts for 3dB beamwidth and directivity.
        * **Steer**: Use the Scan sliders to tilt the beam electronically.
        """)

        col_in, col_geom = st.columns([1, 1])
        
        with col_in:
            st.subheader("Design Parameters")
            freq = st.slider("Frequency (GHz)", 0.5, 60.0, 2.4, 0.1)
            
            st.markdown("**Patch Geometry**")
            col_l, col_w = st.columns(2)
            with col_l: L = st.number_input("Length L (mm)", 1.0, 100.0, 38.0)
            with col_w: W = st.number_input("Width W (mm)", 1.0, 100.0, 45.0)
            
            st.markdown("**Array Configuration**")
            col_n, col_m = st.columns(2)
            with col_n: N = st.number_input("N patches (X)", 1, 16, 2)
            with col_m: M = st.number_input("M patches (Y)", 1, 16, 2)
            
            col_dx, col_dy = st.columns(2)
            with col_dx: dx = st.number_input("Spacing dx (mm)", 1.0, 200.0, 62.0)
            with col_dy: dy = st.number_input("Spacing dy (mm)", 1.0, 200.0, 62.0)

            st.markdown("**Beam Steering**")
            col_ts, col_ps = st.columns(2)
            with col_ts: theta_s = st.slider("Scan Theta (°)", 0.0, 60.0, 0.0)
            with col_ps: phi_s = st.slider("Scan Phi (°)", 0.0, 360.0, 0.0)

        with col_geom:
            st.subheader("Array Layout")
            fig_geo, ax_geo = plt.subplots(figsize=(5, 5))
            for n in range(N):
                for m in range(M):
                    x_pos = n * dx - (N-1)*dx/2 - W/2
                    y_pos = m * dy - (M-1)*dy/2 - L/2
                    rect = plt.Rectangle((x_pos, y_pos), W, L, color='#d4af37', alpha=0.8)
                    ax_geo.add_patch(rect)
            
            limit = max(N*dx, M*dy, 100) / 1.5
            ax_geo.set_xlim(-limit, limit)
            ax_geo.set_ylim(-limit, limit)
            ax_geo.set_aspect('equal')
            ax_geo.set_xlabel("x (mm)")
            ax_geo.set_ylabel("y (mm)")
            ax_geo.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_geo)

        # Calculations
        theta, phi, pat_lin, pat_db, lam = self.calc_pattern(freq, L, W, 1.6, 4.4, N, M, dx, dy, theta_s, phi_s)

        # --- Metrics ---
        st.divider()
        m1, m2, m3 = st.columns(3)
        
        # Directivity approx (Numerical integration over hemisphere)
        # D = 4pi / integral(|f|^2 sin theta dtheta dphi)
        d_theta = theta[1] - theta[0]
        d_phi = phi[1] - phi[0]
        integrand = (pat_lin**2) * np.sin(theta)
        integral = np.sum(integrand) * d_theta * d_phi
        directivity = (4 * np.pi) / integral if integral > 0 else 0
        
        m1.metric("Approx. Directivity", f"{10*np.log10(directivity):.2f} dBi")
        m2.metric("Wavelength (λ)", f"{lam:.2f} mm")
        m3.metric("Array Size", f"{N}x{M} ({N*M} elements)")

        # --- Visualizations ---
        # --- 3D Radiation Pattern ---
        # Convert to Spherical for Plotly
        # Note: We cap DB for visualization
        R = np.maximum(pat_db, -40) + 40 
        
        TH, PH = np.meshgrid(theta, phi)
        X = R * np.sin(TH) * np.cos(PH)
        Y = R * np.sin(TH) * np.sin(PH)
        Z = R * np.cos(TH)
        
        fig_3d = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z, 
            surfacecolor=pat_db, 
            colorscale='Jet',
            cmin=-30,  # Autoscale color to focus on the top 30dB
            cmax=0,
            colorbar=dict(title="Gain (dB)")
        )])
        fig_3d.update_layout(
            title='3D Radiation Pattern (dB)',
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_3d, width='stretch')

        # --- 2D Polar Cuts ---
        st.subheader("2D Polar Cuts")
        col_pol1, col_pol2 = st.columns(2)
        
        # E-Plane (phi = 0)
        e_plane_idx = 0
        h_plane_idx = 90 # phi = 90 degrees
        
        with col_pol1:
            fig_p1, ax_p1 = plt.subplots(subplot_kw={'projection': 'polar'})
            ax_p1.plot(theta, pat_db[e_plane_idx, :], color='red', label='E-Plane (φ=0°)')
            ax_p1.plot(-theta, pat_db[e_plane_idx, :], color='red')
            ax_p1.set_theta_zero_location("N")
            ax_p1.set_thetamin(-90)
            ax_p1.set_thetamax(90)
            ax_p1.set_ylim(-40, 0)
            ax_p1.set_title("E-Plane Cut")
            st.pyplot(fig_p1)
            
            # Beamwidth E-Plane
            main_lobe = pat_db[e_plane_idx, :]
            bw_indices = np.where(main_lobe >= -3)[0]
            if len(bw_indices) > 0:
                bw = np.degrees(theta[bw_indices[-1]] - theta[bw_indices[0]]) * 2
                st.caption(f"E-Plane 3dB Beamwidth: ~{bw:.1f}°")

        with col_pol2:
            fig_p2, ax_p2 = plt.subplots(subplot_kw={'projection': 'polar'})
            ax_p2.plot(theta, pat_db[h_plane_idx, :], color='blue', label='H-Plane (φ=90°)')
            ax_p2.plot(-theta, pat_db[h_plane_idx, :], color='blue')
            ax_p2.set_theta_zero_location("N")
            ax_p2.set_thetamin(-90)
            ax_p2.set_thetamax(90)
            ax_p2.set_ylim(-40, 0)
            ax_p2.set_title("H-Plane Cut")
            st.pyplot(fig_p2)

            # Beamwidth H-Plane
            main_lobe_h = pat_db[h_plane_idx, :]
            bw_indices_h = np.where(main_lobe_h >= -3)[0]
            if len(bw_indices_h) > 0:
                bw_h = np.degrees(theta[bw_indices_h[-1]] - theta[bw_indices_h[0]]) * 2
                st.caption(f"H-Plane 3dB Beamwidth: ~{bw_h:.1f}°")

        st.info("""
        **Model Notes:**
        - **Element Factor:** Uses the two-slot model for a rectangular microstrip patch.
        - **Array Factor:** Calculated for a uniform rectangular array (URA) with user-defined spacing.
        - **Hemisphere:** Pattern is plotted for the upper hemisphere ($z > 0$) as patches are typically backed by a ground plane.
        """)