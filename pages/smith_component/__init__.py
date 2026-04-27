import os
import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component_func = components.declare_component("smith_chart", path=_FRONTEND_DIR)


def smith_chart(
    nodes, freq_mhz, step,
    show_vswr=False,
    trace=None, trace_color="#8b5cf6", trace_highlight_idx=-1,
    trace_ref=None, trace_ref_color="#8b5cf6",
    readonly=False,
    key="smith",
):
    """Interactive client-side Smith chart with rubber-band preview.

    nodes:               list of [re, im] floats (committed impedance points)
    freq_mhz:            target frequency in MHz (used for component-value labels)
    step:                1=place Load, 2=Series (const R), 3=Shunt (const G)
    show_vswr:           overlay VSWR=1.5/2/3 reference circles around Γ=0
    trace:               optional list of [gx, gy] to draw as a solid Γ-plane trace
    trace_color:         CSS color for the primary trace
    trace_highlight_idx: index into `trace` to mark with a dot (target frequency)
    trace_ref:           optional secondary trace drawn faded + dashed (e.g. pre-match
                         reference on the simulation chart)
    trace_ref_color:     CSS color for the secondary trace
    readonly:            when True, disables click/right-click and the step hint —
                         intended for the post-match simulation view.

    Returns the most recent event from the JS as
        {"type": "click"|"undo", "x": gamma_real, "y": gamma_imag, "click_id": int}
    or None if the user hasn't interacted yet. click_id increments on each
    click/right-click so the caller can distinguish new events from stale state.
    """
    return _component_func(
        nodes=nodes, freq_mhz=freq_mhz, step=step, show_vswr=show_vswr,
        trace=trace or [], trace_color=trace_color,
        trace_highlight_idx=trace_highlight_idx,
        trace_ref=trace_ref or [], trace_ref_color=trace_ref_color,
        readonly=readonly,
        key=key, default=None,
    )
