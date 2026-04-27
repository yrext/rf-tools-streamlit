import streamlit as st

from pages.touchstone_viewer import TouchStoneViewer
from pages.smith_match import RfMatcher
from pages.filter_designer import FilterDesigner
from pages.attenuator_designer import AttenuatorDesigner
from pages.mixer_spur_chart import MixerSpurChart
from pages.clock_spur_chart import ClockSpurChart
from pages.fir_designer import FirDesigner
from pages.link_budget import LinkBudgetCalculator
from pages.transmission_line_calc import TransmissionLineCalc
from pages.antenna_pattern_viewer import AntennaPatternViewer


# ----------------------------------------
# Define mini-tools as functions
# ----------------------------------------

def tool_about():
    st.header("ℹ️ About This App")
    st.write("This is a collection of handy RF tools built in a single Streamlit app.")

# ----------------------------------------
# Create the sidebar menu
# ----------------------------------------

st.sidebar.title("🧰 RF Toolkit")
st.sidebar.write("Select a tool below:")

# The selectbox returns the string the user clicked
selected_tool = st.sidebar.radio(
    "Navigation", 
    ["Touchstone Viewer", 
     "Simple RF Match", 
     "Transmission Line Calc",
     "Analog filter designer", 
     "FIR filter designer", 
     "Attenuator Designer",
     "Spur Chart",
     "Antenna Link Budget", 
     "Patch Antenna Array Viewer",
     "About"]
)

# ----------------------------------------
# Route to the correct tool
# ----------------------------------------

if selected_tool == "Touchstone Viewer":
    TouchStoneViewer().run()
elif selected_tool == "Simple RF Match":
    RfMatcher().run()
elif selected_tool == "Analog filter designer":
    FilterDesigner().run()
elif selected_tool == "Attenuator Designer":
    AttenuatorDesigner().run()
elif selected_tool == "Transmission Line Calc":
    TransmissionLineCalc().run()
elif selected_tool == "Mixer Spur Chart":
    MixerSpurChart().run()
elif selected_tool == "Spur Chart":
    ClockSpurChart().run()
elif selected_tool == "FIR filter designer":
    FirDesigner().run()
elif selected_tool == "Antenna Link Budget":
    LinkBudgetCalculator().run()
elif selected_tool == "Patch Antenna Array Viewer":
    AntennaPatternViewer().run()
elif selected_tool == "About":
    tool_about()
else:
    pass
