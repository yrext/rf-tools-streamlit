import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

class TransmissionLineCalc:
    def __init__(self):
        pass

    def calc_microstrip(self, w, h, er, t):
        """Hammerstad and Jensen model for Microstrip Impedance."""
        if t > 0:
            we = w + (t / np.pi) * np.log(1 + (4 * np.e) / (t / h * (1 / np.tanh(np.sqrt(6.517 * w / h)))**2))
            we = (we + w) / 2
        else:
            we = w

        u = we / h
        er_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 / u)**-0.5
        
        if u <= 1:
            z0 = (60 / np.sqrt(er_eff)) * np.log(8 / u + 0.25 * u)
        else:
            z0 = (120 * np.pi) / (np.sqrt(er_eff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
        
        return z0, er_eff

    def calc_stripline(self, w, b, er, t):
        """Standard Stripline model (IPC-2221). b = ground spacing."""
        z0 = (60 / np.sqrt(er)) * np.log((1.9 * b) / (0.8 * w + t))
        return z0, er

    def calc_cpw(self, w, s, er):
        """Conformal mapping for CPW with infinite substrate."""
        k = w / (w + 2 * s)
        k_sq = k**2
        kp_sq = 1 - k_sq
        
        # Complete elliptic integral of the first kind
        kk = ellipk(k_sq)
        kkp = ellipk(kp_sq)
        
        er_eff = (er + 1) / 2
        z0 = (30 * np.pi / np.sqrt(er_eff)) * (kkp / kk)
        return z0, er_eff

    def run(self):
        st.header("📏 Transmission Line Calculator")
        st.write("Calculate characteristic impedance and visualize PCB cross-sections.")

        col_p, col_v = st.columns([1, 1.5])

        with col_p:
            if 'tl_unit' not in st.session_state:
                st.session_state.tl_unit = "Imperial (mil)"
                st.session_state.tl_w = 121
                st.session_state.tl_h = 63
                st.session_state.tl_s = 5.0

            unit_sys = st.radio("Unit System", ["Imperial (mil)", "Metric (mm)"], horizontal=True)
            
            # Handle Unit Conversion on Toggle
            if unit_sys != st.session_state.tl_unit:
                conv = 0.0254 if unit_sys == "Metric (mm)" else 1/0.0254
                st.session_state.tl_w *= conv
                st.session_state.tl_h *= conv
                st.session_state.tl_s *= conv
                st.session_state.tl_unit = unit_sys

            is_metric = unit_sys == "Metric (mm)"
            u_lab = "mm" if is_metric else "mil"
            to_mil = 1/0.0254 if is_metric else 1.0
            
            line_type = st.selectbox("Geometry", ["Microstrip", "Stripline", "Coplanar Waveguide"])
            
            if is_metric:
                w = st.slider(f"Trace Width (W) [{u_lab}]", 0.05, 10.0, float(np.clip(st.session_state.tl_w, 0.05, 10.0)), 0.01)
                h = st.slider(f"Dielectric Height (H) [{u_lab}]", 0.02, 10.0, float(np.clip(st.session_state.tl_h, 0.02, 10.0)), 0.01)
            else:
                w = st.slider(f"Trace Width (W) [{u_lab}]", 2.0, 200.0, float(np.clip(st.session_state.tl_w, 2.0, 200.0)), 0.5)
                h = st.slider(f"Dielectric Height (H) [{u_lab}]", 1.0, 200.0, float(np.clip(st.session_state.tl_h, 1.0, 200.0)), 0.5)
            
            # Store current values to session state
            st.session_state.tl_w = w
            st.session_state.tl_h = h

            er = st.slider("Dielectric Constant (εr)", 1.0, 20.0, 4.4, 0.1)
            
            w_mil, h_mil = w * to_mil, h * to_mil
            s_mil = 0.0
            
            if line_type == "Coplanar Waveguide":
                if is_metric:
                    s = st.slider(f"Gap Spacing (S) [{u_lab}]", 0.05, 5.0, float(np.clip(st.session_state.tl_s, 0.05, 5.0)), 0.01)
                else:
                    s = st.slider(f"Gap Spacing (S) [{u_lab}]", 2.0, 100.0, float(np.clip(st.session_state.tl_s, 2.0, 100.0)), 0.5)
                st.session_state.tl_s = s
                s_mil = s * to_mil
                z0, eff = self.calc_cpw(w_mil, s_mil, er)
            elif line_type == "Microstrip":
                z0, eff = self.calc_microstrip(w_mil, h_mil, er, 0)
            else: # Stripline
                z0, eff = self.calc_stripline(w_mil, h_mil, er, 0)

            st.divider()
            st.metric("Characteristic Impedance (Z₀)", f"{z0:.2f} Ω")
            st.metric("Effective Dielectric (ε_eff)", f"{eff:.3f}")

        with col_v:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor('#f8f9fa')
            
            # Static drawing constants for consistent proportions
            W_DRAW = 40
            H_DRAW = 60
            S_DRAW = 20
            T_DRAW = 5
            
            ax.set_xlim(-100, 100)
            ax.set_ylim(-20, 100)

            # Draw Substrate
            ax.add_patch(plt.Rectangle((-100, 0), 200, H_DRAW, color='#a8d5ba', alpha=0.5))
            
            # Draw Conductors
            gold = '#d4af37'
            if line_type == "Microstrip":
                ax.add_patch(plt.Rectangle((-100, -T_DRAW), 200, T_DRAW, color=gold)) # Bottom GND
                ax.add_patch(plt.Rectangle((-W_DRAW/2, H_DRAW), W_DRAW, T_DRAW, color=gold)) # Trace

                # Annotations
                ax.annotate('', xy=(-W_DRAW/2, H_DRAW + T_DRAW + 5), xytext=(W_DRAW/2, H_DRAW + T_DRAW + 5), 
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(0, H_DRAW + T_DRAW + 10, 'W', ha='center', fontweight='bold')
                ax.annotate('', xy=(-85, 0), xytext=(-85, H_DRAW), 
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(-90, H_DRAW/2, 'H', ha='right', va='center', fontweight='bold')

            elif line_type == "Stripline":
                ax.add_patch(plt.Rectangle((-100, -T_DRAW), 200, T_DRAW, color=gold)) # Bottom GND
                ax.add_patch(plt.Rectangle((-100, H_DRAW), 200, T_DRAW, color=gold))   # Top GND
                ax.add_patch(plt.Rectangle((-W_DRAW/2, H_DRAW/2 - T_DRAW/2), W_DRAW, T_DRAW, color=gold)) # Center Trace

                # Annotations
                ax.annotate('', xy=(-W_DRAW/2, H_DRAW/2 + T_DRAW/2 + 5), xytext=(W_DRAW/2, H_DRAW/2 + T_DRAW/2 + 5), 
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(0, H_DRAW/2 + T_DRAW/2 + 10, 'W', ha='center', fontweight='bold')
                ax.annotate('', xy=(-85, 0), xytext=(-85, H_DRAW), 
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(-90, H_DRAW/2, 'H', ha='right', va='center', fontweight='bold')

            elif line_type == "Coplanar Waveguide":
                ax.add_patch(plt.Rectangle((-W_DRAW/2, H_DRAW), W_DRAW, T_DRAW, color=gold)) # Signal
                ax.add_patch(plt.Rectangle((-100, H_DRAW), 100 - W_DRAW/2 - S_DRAW, T_DRAW, color=gold)) # Left GND
                ax.add_patch(plt.Rectangle((W_DRAW/2 + S_DRAW, H_DRAW), 100 - W_DRAW/2 - S_DRAW, T_DRAW, color=gold)) # Right GND

                # Annotations
                ax.annotate('', xy=(-W_DRAW/2, H_DRAW + T_DRAW + 5), xytext=(W_DRAW/2, H_DRAW + T_DRAW + 5), 
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(0, H_DRAW + T_DRAW + 10, 'W', ha='center', fontweight='bold')
                ax.annotate('', xy=(W_DRAW/2, H_DRAW + T_DRAW + 5), xytext=(W_DRAW/2 + S_DRAW, H_DRAW + T_DRAW + 5), 
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(W_DRAW/2 + S_DRAW/2, H_DRAW + T_DRAW + 10, 'S', ha='center', fontweight='bold')
                ax.annotate('', xy=(-85, 0), xytext=(-85, H_DRAW), 
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(-90, H_DRAW/2, 'H', ha='right', va='center', fontweight='bold')

            ax.set_aspect('equal')
            ax.axis('off')
            st.pyplot(fig)
            
        st.info("Formulas assume infinite ground planes and standard quasi-static approximations.")