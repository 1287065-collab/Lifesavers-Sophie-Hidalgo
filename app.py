import json
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration & Constants (Pathway NUR-3 | AP Calculus | Advanced)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Career Venture Lab | NUR-3 Health Education Studio",
    page_icon="🩺",
    layout="wide",
)

DEFAULT_TEAM_ID = "NURSES-IN-HEALTHCARE"
PATHWAY_ID = "NUR-3"
COURSE_NAME = "AP Calculus"
BUSINESS_NAME = "Nurse-led health education simulation"
OFFERINGS = [
    "General wellness education workshop",
    "Caregiver information session",
    "First-aid awareness demonstration",
]


def to_money(val) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def init_default_state() -> dict:
    return {
        "team_id": DEFAULT_TEAM_ID,
        "course": COURSE_NAME,
        "pathway_id": PATHWAY_ID,
        "selected_skill": "Constrained Profit Optimization & Accumulation",
        "selected_level": "Advanced",
        "snapshot_id": "SNAP-001",
        "state_version": 1,
        "simulation_date": "2027-01-31",
        "base_period": "Monthly",
        "working_days": 20,
        "currency": "USD",
        "random_seed": 2027,
        "processed_run_ids": [],
        "startup": {
            "owner_contribution": 6000.00,
            "equipment_assets": 1500.00,
            "pre_opening_expense": 500.00,
            "opening_inventory": 0.00,
            "post_startup_cash": 4000.00,
            "retained_earnings_start": -500.00,
            "loan_balance": 0.00,
        },
        "inputs": {
            "base_price": 25.00,
            "D0": 80,
            "staff_hours": 30.0,
            "hours_per_unit": 0.5,
            "venue_seats": 60,
            "no_show_rate": 0.0,
            "satisfaction_mean": 4.0,
            "var_educator_delivery": 2.00,
            "var_materials": 2.00,
            "var_cleaning_consumables": 1.00,
            "fix_venue": 250.00,
            "fix_scheduling_software": 100.00,
            "fix_accessibility_support": 200.00,
            "fix_admin_insurance": 150.00,
            "ad_spend": 0.00,
            "refund_amount": 0.00,
            "loan_principal_payment": 0.00,
            "interest_expense": 0.00,
            "additional_capex": 0.00,
            "new_owner_cash": 0.00,
            "new_loan_cash": 0.00,
        },
        "ledger": [],
        "recommendations": [],
        "decisions": [],
        "assumptions": [
            "Fictional nonclinical educational workshops only; no medical diagnosis, treatment, or patient data.",
            "Linear baseline demand q(p) = 2*D0 - (D0/p0)*p = 160 - 3.2p on 0 <= p <= 50.",
            "Payment collected and variable per-seat cost incurred only for attended seats.",
            "Baseline scenario excludes taxes, depreciation, and receivables unless explicitly modeled.",
        ],
        "sources": [
            "Career Venture Lab Student Handbook (November 2026) - Pathway NUR-3 & NUR Package",
            "AP Calculus Course Build Adapter",
        ],
    }


if "biz_state" not in st.session_state:
    st.session_state.biz_state = init_default_state()


# ---------------------------------------------------------------------------
# Deterministic Simulation & Math Engine
# ---------------------------------------------------------------------------
def compute_unit_variable_cost(inputs: dict) -> Decimal:
    return (
        to_money(inputs["var_educator_delivery"])
        + to_money(inputs["var_materials"])
        + to_money(inputs["var_cleaning_consumables"])
    )


def compute_monthly_fixed_cost(inputs: dict) -> Decimal:
    return (
        to_money(inputs["fix_venue"])
        + to_money(inputs["fix_scheduling_software"])
        + to_money(inputs["fix_accessibility_support"])
        + to_money(inputs["fix_admin_insurance"])
    )


def simulate_month(
    inputs: dict,
    opening_cash: Decimal,
    opening_debt: Decimal,
    requested_override: int | None = None,
    capacity_override: int | None = None,
    ad_cost_override: float | None = None,
) -> dict:
    price = to_money(inputs["base_price"])
    requested = int(requested_override) if requested_override is not None else int(inputs["D0"])
    if requested < 0:
        raise ValueError("Requested demand must be nonnegative.")

    if capacity_override is not None:
        capacity = int(capacity_override)
        bottleneck = "Fixture Capacity Override"
    else:
        staff_cap = math.floor(float(inputs["staff_hours"]) / float(inputs["hours_per_unit"]))
        venue_cap = int(inputs["venue_seats"])
        capacity = min(staff_cap, venue_cap)
        if staff_cap < venue_cap:
            bottleneck = f"Educator Hours ({staff_cap} seats)"
        elif venue_cap < staff_cap:
            bottleneck = f"Venue Seats ({venue_cap} seats)"
        else:
            bottleneck = f"Balanced ({capacity} seats)"

    booked = min(requested, capacity)
    unserved_demand = max(0, requested - booked)
    no_show_rate = float(inputs["no_show_rate"])
    attended = math.floor(booked * (1.0 - no_show_rate))
    no_shows = booked - attended
    sold = attended

    v_unit = compute_unit_variable_cost(inputs)
    f_month = compute_monthly_fixed_cost(inputs)
    marketing = to_money(ad_cost_override) if ad_cost_override is not None else to_money(inputs["ad_spend"])
    refunds = to_money(inputs["refund_amount"])
    interest = to_money(inputs["interest_expense"])
    principal = to_money(inputs["loan_principal_payment"])
    capex = to_money(inputs["additional_capex"])
    new_owner = to_money(inputs["new_owner_cash"])
    new_loan = to_money(inputs["new_loan_cash"])

    gross_revenue = (price * Decimal(sold)).quantize(Decimal("0.01"))
    net_revenue = gross_revenue - refunds
    variable_expense = (v_unit * Decimal(sold)).quantize(Decimal("0.01"))
    fixed_expense = f_month
    total_operating_expense = variable_expense + fixed_expense + marketing
    operating_profit = net_revenue - total_operating_expense
    net_profit = operating_profit - interest

    ending_cash = (
        opening_cash
        + net_revenue
        - variable_expense
        - fixed_expense
        - marketing
        - interest
        - principal
        - capex
        + new_owner
        + new_loan
    )
    ending_debt = max(Decimal("0.00"), opening_debt + new_loan - principal)

    return {
        "requested": requested,
        "capacity": capacity,
        "bottleneck": bottleneck,
        "booked": booked,
        "unserved_demand": unserved_demand,
        "no_shows": no_shows,
        "sold": sold,
        "price": float(price),
        "v_unit": float(v_unit),
        "gross_revenue": float(gross_revenue),
        "refunds": float(refunds),
        "net_revenue": float(net_revenue),
        "variable_expense": float(variable_expense),
        "fixed_expense": float(fixed_expense),
        "marketing_expense": float(marketing),
        "total_expense": float(total_operating_expense),
        "operating_profit": float(operating_profit),
        "interest_expense": float(interest),
        "net_profit": float(net_profit),
        "opening_cash": float(opening_cash),
        "ending_cash": float(ending_cash),
        "ending_debt": float(ending_debt),
        "utilization_pct": round((booked / capacity * 100.0), 2) if capacity > 0 else None,
        "attendance_pct": round((attended / booked * 100.0), 2) if booked > 0 else None,
    }


def record_simulation_run(
    state: dict,
    run_id: str,
    sim_date: str,
    requested_override: int | None = None,
    capacity_override: int | None = None,
    ad_cost_override: float | None = None,
    mode_label: str = "Normal",
) -> tuple[bool, str, dict | None]:
    if run_id in state["processed_run_ids"]:
        return False, f"Duplicate run_id '{run_id}' rejected: no duplicate transactions created.", None

    current_cash = to_money(state["startup"]["post_startup_cash"])
    current_debt = to_money(state["startup"]["loan_balance"])
    if state["ledger"]:
        last_tx = state["ledger"][-1]
        current_cash = to_money(last_tx["ending_cash"])
        current_debt = to_money(last_tx["ending_debt"])

    res = simulate_month(
        state["inputs"],
        opening_cash=current_cash,
        opening_debt=current_debt,
        requested_override=requested_override,
        capacity_override=capacity_override,
        ad_cost_override=ad_cost_override,
    )

    state["state_version"] += 1
    ver = state["state_version"]
    state["simulation_date"] = sim_date
    state["snapshot_id"] = f"SNAP-{ver:03d}"
    state["processed_run_ids"].append(run_id)

    tx = {
        "transaction_id": f"TX-{len(state['ledger']) + 1:04d}",
        "run_id": run_id,
        "date": sim_date,
        "mode": mode_label,
        "offering_id": "NUR-WORKSHOP-01",
        "state_version": ver,
        **res,
    }
    state["ledger"].append(tx)
    return True, f"Recorded run '{run_id}' -> state_version v{ver}.", tx


def run_nur3_acceptance_tests() -> pd.DataFrame:
    results = []

    base = init_default_state()
    r_zero = simulate_month(base["inputs"], Decimal("4000.00"), Decimal("0.00"), requested_override=0)
    t1_pass = (
        r_zero["sold"] == 0
        and r_zero["net_revenue"] == 0.0
        and r_zero["variable_expense"] == 0.0
        and r_zero["operating_profit"] == -700.0
        and r_zero["utilization_pct"] == 0.0
        and r_zero["attendance_pct"] is None
    )
    results.append({
        "Test Case": "1. Zero Sales (D=0)",
        "Expected": "Rev=$0, Var=$0, Profit=-$700, No div-by-zero",
        "Actual": f"Rev=${r_zero['net_revenue']:.2f}, Var=${r_zero['variable_expense']:.2f}, Profit=${r_zero['operating_profit']:.2f}",
        "Status": "PASS" if t1_pass else "FAIL",
    })

    r_cap = simulate_month(base["inputs"], Decimal("4000.00"), Decimal("0.00"))
    t2_pass = r_cap["booked"] == 60 and r_cap["unserved_demand"] == 20 and r_cap["operating_profit"] == 500.0
    results.append({
        "Test Case": "2. Demand Exceeds Capacity (D=80, Cap=60)",
        "Expected": "Sold=60, Unserved=20, Profit=$500",
        "Actual": f"Sold={r_cap['sold']}, Unserved={r_cap['unserved_demand']}, Profit=${r_cap['operating_profit']:.2f}",
        "Status": "PASS" if t2_pass else "FAIL",
    })

    s_fix = init_default_state()
    fixture_rows = [
        ("FIX-2027-01", "2027-01-31", 80, 60, 0.0),
        ("FIX-2027-02", "2027-02-28", 88, 60, 100.0),
        ("FIX-2027-03", "2027-03-31", 72, 45, 0.0),
    ]
    for rid, dt, req, cap, ad in fixture_rows:
        record_simulation_run(s_fix, rid, dt, requested_override=req, capacity_override=cap, ad_cost_override=ad, mode_label="Fixture")
    l = s_fix["ledger"]
    t3_pass = (
        len(l) == 3
        and (l[0]["sold"], l[0]["net_revenue"], l[0]["total_expense"], l[0]["operating_profit"], l[0]["ending_cash"]) == (60, 1500.0, 1000.0, 500.0, 4500.0)
        and (l[1]["sold"], l[1]["net_revenue"], l[1]["total_expense"], l[1]["operating_profit"], l[1]["ending_cash"]) == (60, 1500.0, 1100.0, 400.0, 4900.0)
        and (l[2]["sold"], l[2]["net_revenue"], l[2]["total_expense"], l[2]["operating_profit"], l[2]["ending_cash"]) == (45, 1125.0, 925.0, 200.0, 5100.0)
    )
    results.append({
        "Test Case": "3. NUR 3-Month Ledger Fixture",
        "Expected": "Closing Cash: Jan $4,500 | Feb $4,900 | Mar $5,100",
        "Actual": f"Jan ${l[0]['ending_cash']:.0f} | Feb ${l[1]['ending_cash']:.0f} | Mar ${l[2]['ending_cash']:.0f}",
        "Status": "PASS" if t3_pass else "FAIL",
    })

    s_ns = init_default_state()
    s_ns["inputs"]["no_show_rate"] = 0.20
    r_ns = simulate_month(s_ns["inputs"], Decimal("4000.00"), Decimal("0.00"))
    t4_pass = (
        r_ns["booked"] == 60
        and r_ns["sold"] == 48
        and r_ns["net_revenue"] == 1200.0
        and r_ns["variable_expense"] == 240.0
        and r_ns["operating_profit"] == 260.0
    )
    results.append({
        "Test Case": "4. NUR Business-Specific Test (20% No-Show)",
        "Expected": "Booked=60, Attended=48, Rev=$1200, Var=$240, Profit=$260",
        "Actual": f"Booked={r_ns['booked']}, Attended={r_ns['sold']}, Rev=${r_ns['net_revenue']:.0f}, Var=${r_ns['variable_expense']:.0f}, Profit=${r_ns['operating_profit']:.0f}",
        "Status": "PASS" if t4_pass else "FAIL",
    })

    ok_dup, _, _ = record_simulation_run(s_fix, "FIX-2027-01", "2027-01-31", requested_override=80, capacity_override=60, ad_cost_override=0.0)
    t5_pass = (ok_dup is False) and (len(s_fix["ledger"]) == 3)
    results.append({
        "Test Case": "5. Duplicate Run Protection",
        "Expected": "Second run with same run_id rejected; ledger length stays 3",
        "Actual": f"Rejected={not ok_dup}, Ledger rows={len(s_fix['ledger'])}",
        "Status": "PASS" if t5_pass else "FAIL",
    })

    t_grid = np.linspace(0, 10, 10001)
    rev_int = float(np.trapezoid(100 + 20 * t_grid, t_grid))
    cost_int = float(np.trapezoid(60 + 5 * t_grid, t_grid))
    prof_int = rev_int - cost_int
    t6_pass = abs(prof_int - 1150.0) < 1e-4
    results.append({
        "Test Case": "6. AP Calculus Accumulation Integral [0,10]",
        "Expected": "Rev=$2,000, Cost=$850, Net=$1,150",
        "Actual": f"Rev=${rev_int:.2f}, Cost=${cost_int:.2f}, Net=${prof_int:.2f}",
        "Status": "PASS" if t6_pass else "FAIL",
    })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Sidebar: Snapshot, Save/Load JSON, & Scope Notice
# ---------------------------------------------------------------------------
state = st.session_state.biz_state
with st.sidebar:
    st.title("🩺 Career Venture Lab")
    st.caption(f"**Team:** `{state['team_id']}`  \n**Pathway:** `{state['pathway_id']}` ({state['course']})  \n**Level:** `{state['selected_level']}`")
    st.info(
        f"**Active Snapshot:** `{state['snapshot_id']}` (v{state['state_version']})  \n"
        f"**Sim Date:** `{state['simulation_date']}`  \n"
        f"**Ledger Entries:** `{len(state['ledger'])}`"
    )

    st.subheader("💾 Durable Save / Restore")
    snapshot_json = json.dumps(state, indent=2)
    st.download_button(
        label="Export Snapshot JSON",
        data=snapshot_json,
        file_name=f"{state['team_id']}_{state['pathway_id']}_v{state['state_version']}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_json = st.file_uploader("Restore / Import Snapshot JSON", type=["json"])
    if uploaded_json is not None:
        try:
            imported = json.load(uploaded_json)
            required_keys = {"team_id", "course", "pathway_id", "state_version", "startup", "inputs", "ledger"}
            if not required_keys.issubset(imported.keys()):
                st.error("Import rejected: missing required snapshot fields.")
            elif imported.get("team_id") != state["team_id"]:
                st.error(f"Import rejected: team_id '{imported.get('team_id')}' does not match '{state['team_id']}'.")
            elif imported.get("pathway_id") != PATHWAY_ID:
                st.error(f"Import rejected: pathway_id '{imported.get('pathway_id')}' does not match '{PATHWAY_ID}'.")
            else:
                st.session_state.biz_state = imported
                st.success(f"Restored {imported.get('snapshot_id')} (v{imported['state_version']}).")
                st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON file: {exc}")

    if st.button("Reset to NUR-3 Baseline", use_container_width=True):
        st.session_state.biz_state = init_default_state()
        st.rerun()

    st.divider()
    st.warning(
        "**Educational Simulation Only:** Fictional nonclinical health education workshops. "
        "No medical diagnosis, treatment, real patient data, or real payments."
    )

# ---------------------------------------------------------------------------
# Main Header & Navigation Tabs (Stage 1 Core + Placeholders)
# ---------------------------------------------------------------------------
st.header("Nurse-Led Health Education Simulation (`NUR-3`)")
st.markdown(
    "**Primary Offering:** General wellness education workshop *(secondary offerings: Caregiver information session; First-aid awareness demonstration)*"
)

tabs = st.tabs([
    "1. Overview & Inputs",
    "2. AP Calculus Math Panel",
    "3. Operations & Ledger",
    "4. AI Handoffs & Diagnostics",
    "5. Stage 2-4 Roadmap",
])

# ---------------------------------------------------------------------------
# TAB 1: Overview & Editable Inputs
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Startup Financing & Opening Balance Check (Fictional Editable Baseline)")
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
    s_col1.metric("Owner Contribution", f"${state['startup']['owner_contribution']:,.2f}")
    s_col2.metric("Equipment Assets", f"${state['startup']['equipment_assets']:,.2f}")
    s_col3.metric("Pre-Opening Expense", f"${state['startup']['pre_opening_expense']:,.2f}")
    s_col4.metric("Post-Startup Cash", f"${state['startup']['post_startup_cash']:,.2f}")

    current_cash_val = (
        state["ledger"][-1]["ending_cash"] if state["ledger"] else state["startup"]["post_startup_cash"]
    )
    s_col5.metric("Current Spendable Cash", f"${current_cash_val:,.2f}")

    st.divider()
    st.subheader("Monthly Operating & Cost Parameters (Scenario Sandbox)")
    st.caption("Adjusting sliders previews the current month without modifying the official ledger until you approve a run.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Pricing, Demand & Capacity**")
        state["inputs"]["base_price"] = st.number_input(
            "Workshop Seat Price ($)", min_value=1.0, max_value=200.0, value=float(state["inputs"]["base_price"]), step=1.0
        )
        state["inputs"]["D0"] = st.number_input(
            "Base Monthly Demand D0 (seats)", min_value=0, max_value=500, value=int(state["inputs"]["D0"]), step=5
        )
        state["inputs"]["staff_hours"] = st.number_input(
            "Educator Staff Hours / Month", min_value=0.0, max_value=300.0, value=float(state["inputs"]["staff_hours"]), step=1.0
        )
        state["inputs"]["hours_per_unit"] = st.number_input(
            "Educator Hours per Seat", min_value=0.1, max_value=10.0, value=float(state["inputs"]["hours_per_unit"]), step=0.1
        )
        state["inputs"]["venue_seats"] = st.number_input(
            "Venue Seat Limit / Month", min_value=0, max_value=300, value=int(state["inputs"]["venue_seats"]), step=5
        )
        state["inputs"]["no_show_rate"] = st.slider(
            "No-Show Rate", min_value=0.0, max_value=0.50, value=float(state["inputs"]["no_show_rate"]), step=0.05
        )

    with col_b:
        st.markdown("**Variable Costs ($ per Attended Seat)**")
        state["inputs"]["var_educator_delivery"] = st.number_input(
            "Educator Delivery ($/seat)", min_value=0.0, value=float(state["inputs"]["var_educator_delivery"]), step=0.5
        )
        state["inputs"]["var_materials"] = st.number_input(
            "Workshop Materials ($/seat)", min_value=0.0, value=float(state["inputs"]["var_materials"]), step=0.5
        )
        state["inputs"]["var_cleaning_consumables"] = st.number_input(
            "Cleaning Consumables ($/seat)", min_value=0.0, value=float(state["inputs"]["var_cleaning_consumables"]), step=0.5
        )
        st.info(f"**Total Variable Cost ($v$):** `${float(compute_unit_variable_cost(state['inputs'])):.2f} / attended seat`")

        st.markdown("**Campaign & Non-Operating Cash Items**")
        state["inputs"]["ad_spend"] = st.number_input(
            "Own Marketing Expense ($)", min_value=0.0, value=float(state["inputs"]["ad_spend"]), step=25.0
        )
        state["inputs"]["additional_capex"] = st.number_input(
            "Additional Equipment CapEx ($)", min_value=0.0, value=float(state["inputs"]["additional_capex"]), step=50.0
        )

    with col_c:
        st.markdown("**Monthly Fixed Expenses ($/Month)**")
        state["inputs"]["fix_venue"] = st.number_input(
            "Venue Rental ($/mo)", min_value=0.0, value=float(state["inputs"]["fix_venue"]), step=25.0
        )
        state["inputs"]["fix_scheduling_software"] = st.number_input(
            "Scheduling Software ($/mo)", min_value=0.0, value=float(state["inputs"]["fix_scheduling_software"]), step=25.0
        )
        state["inputs"]["fix_accessibility_support"] = st.number_input(
            "Accessibility & Translation Support ($/mo)", min_value=0.0, value=float(state["inputs"]["fix_accessibility_support"]), step=25.0
        )
        state["inputs"]["fix_admin_insurance"] = st.number_input(
            "Admin & Insurance Assumption ($/mo)", min_value=0.0, value=float(state["inputs"]["fix_admin_insurance"]), step=25.0
        )
        st.info(f"**Total Fixed Cost ($F$):** `${float(compute_monthly_fixed_cost(state['inputs'])):.2f} / month`")

    preview = simulate_month(state["inputs"], to_money(current_cash_val), Decimal("0.00"))
    st.subheader("Live Single-Month Preview (Uncommitted)")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Capacity / Bottleneck", f"{preview['capacity']} seats", preview["bottleneck"])
    p2.metric("Attended (Sold)", f"{preview['sold']} / {preview['requested']} req", f"{preview['unserved_demand']} unserved")
    p3.metric("Monthly Revenue", f"${preview['net_revenue']:,.2f}")
    p4.metric("Operating Profit", f"${preview['operating_profit']:,.2f}")
    p5.metric("Projected Closing Cash", f"${preview['ending_cash']:,.2f}")

# ---------------------------------------------------------------------------
# TAB 2: AP Calculus Math Skills & Challenge Level Panel
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Math Skills & Challenge Level Panel — AP Calculus")
    m1, m2, m3 = st.columns(3)
    m1.info(f"**Course:** {state['course']} (`{state['pathway_id']}`)")
    m2.info(f"**Approved Challenge Level:** {state['selected_level']}")
    m3.info("**Skill:** Constrained Optimization & Rate Accumulation")

    p0 = float(state["inputs"]["base_price"])
    D0 = float(state["inputs"]["D0"])
    v = float(compute_unit_variable_cost(state["inputs"]))
    F = float(compute_monthly_fixed_cost(state["inputs"]))
    cap = preview["capacity"]

    a = 2.0 * D0
    b = (D0 / p0) if p0 > 0 else 0.0
    p_max = 2.0 * p0
    p_corner = (a - cap) / b if b > 0 else 0.0
    p_unconstrained = (a + b * v) / (2.0 * b) if b > 0 else 0.0

    st.markdown("### 1. Piecewise Constrained Profit Optimization & Derivative Analysis")
    st.latex(
        rf"q(p) = {a:.1f} - {b:.2f}p \quad 	ext{{on }} 0 \le p \le {p_max:.2f}, \qquad "
        rf"s(p) = \min\left({cap},\, \max(0,\, q(p))ight)"
    )
    st.latex(
        rf"P(p) = egin{{cases}} "
        rf"{cap}(p - {v:.2f}) - {F:.2f} & 	ext{{if }} 0 \le p < {p_corner:.2f} 	ext{{ (capacity-constrained)}} \ "
        rf"(p - {v:.2f})({a:.1f} - {b:.2f}p) - {F:.2f} & 	ext{{if }} {p_corner:.2f} \le p \le {p_max:.2f} 	ext{{ (demand-constrained)}} "
        rf"\end{{cases}}"
    )
    st.latex(
        rf"P'(p) = egin{{cases}} "
        rf"{cap} 	ext{{ seats (\$/\$)}} & 	ext{{if }} 0 < p < {p_corner:.2f} \ "
        rf"	ext{{undefined (corner)}} & 	ext{{at }} p = {p_corner:.2f} \ "
        rf"{a + b * v:.2f} - {2 * b:.2f}p & 	ext{{if }} {p_corner:.2f} < p < {p_max:.2f} "
        rf"\end{{cases}}"
    )

    p_grid = np.linspace(0, p_max, 400)
    q_demand = np.maximum(0.0, a - b * p_grid)
    s_cap = np.minimum(float(cap), q_demand)
    profit_cont = (p_grid - v) * s_cap - F
    profit_int = (p_grid - v) * np.floor(s_cap) - F

    deriv_vals = np.where(p_grid < p_corner, float(cap), a + b * v - 2 * b * p_grid)
    if b > 0 and 0 <= p_corner <= p_max:
        corner_idx = int(np.argmin(np.abs(p_grid - p_corner)))
        deriv_vals[corner_idx] = np.nan

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"Continuous vs. Integer Profit P(p) [Snapshot {state['snapshot_id']} v{state['state_version']}]",
            "Marginal Profit Rate P'(p) ($ profit per $1 price increase)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=p_grid, y=profit_cont, name="Continuous Profit P(p)", line=dict(color="#1f77b4", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=p_grid, y=profit_int, name="Integer-Unit Profit (Floor)", line=dict(color="#ff7f0e", dash="dot", width=2)),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even ($0)", row=1, col=1)
    if 0 <= p_corner <= p_max:
        fig.add_vline(
            x=p_corner,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"Capacity Corner p=${p_corner:.2f}",
            row=1,
            col=1,
        )
        fig.add_vline(x=p_corner, line_dash="dash", line_color="#d62728", row=2, col=1)

    fig.add_trace(
        go.Scatter(x=p_grid, y=deriv_vals, name="Derivative P'(p)", line=dict(color="#2ca02c", width=3)),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_xaxes(title_text="Workshop Seat Price p (USD/seat)", row=2, col=1)
    fig.update_yaxes(title_text="Monthly Profit (USD)", row=1, col=1)
    fig.update_yaxes(title_text="P'(p) (USD / USD)", row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    p_opt_constrained = max(p_corner, p_unconstrained) if p_unconstrained < p_corner else p_unconstrained
    candidate_prices = sorted(set([0.0, round(p_corner, 2), round(p_unconstrained, 2), round(p_opt_constrained, 2), round(p_max, 2)]))
    audit_rows = []
    for cp in candidate_prices:
        if 0 <= cp <= p_max:
            qd = max(0.0, a - b * cp)
            sc = min(float(cap), qd)
            pr_c = (cp - v) * sc - F
            pr_i = (cp - v) * math.floor(sc) - F
            note = []
            if abs(cp - 0.0) < 1e-6:
                note.append("Left Endpoint")
            if abs(cp - p_max) < 1e-6:
                note.append("Right Endpoint")
            if abs(cp - p_corner) < 0.02:
                note.append("Capacity Corner (Constrained Optimum)")
            if abs(cp - p_unconstrained) < 0.02:
                note.append("Unconstrained Critical Point (P'=0)")
            audit_rows.append({
                "Candidate Price ($)": f"${cp:.2f}",
                "Point Type": ", ".join(note) if note else "Interior Point",
                "Unconstrained Demand q(p)": round(qd, 2),
                "Completed Seats s(p)": round(sc, 2),
                "Continuous Profit ($)": f"${pr_c:,.2f}",
                "Integer-Floored Profit ($)": f"${pr_i:,.2f}",
            })
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

    st.info(
        f"**Interpretation Prompt:** At the unconstrained critical point $p = \${p_unconstrained:.2f}$, demand is "
        f"$q({p_unconstrained:.2f}) = {max(0.0, a - b * p_unconstrained):.1f}$ seats, which exceeds your capacity of **{cap} seats**. "
        f"Because $P'(p) = {cap} > 0$ for all $p < \${p_corner:.2f}$ and $P'(p) < 0$ for $p > \${p_corner:.2f}$, "
        f"profit is maximized right at the capacity boundary **$p = \${p_corner:.2f}$**."
    )

    if state["selected_level"] == "Advanced":
        st.divider()
        st.markdown("### 2. Advanced Extension: Campaign Rate Accumulation (Definite Integral)")
        st.caption(
            "Isolated 10-day campaign rate exercise from the AP Calculus Adapter (kept separate from the monthly accounting ledger)."
        )
        t_end = st.slider("Campaign Window End Day T (days)", min_value=1, max_value=10, value=10)
        t_arr = np.linspace(0, t_end, 500)
        r_t = 100.0 + 20.0 * t_arr
        c_t = 60.0 + 5.0 * t_arr
        acc_rev = float(np.trapezoid(r_t, t_arr))
        acc_cost = float(np.trapezoid(c_t, t_arr))
        acc_net = acc_rev - acc_cost

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Accumulated Revenue ∫₀^{t_end} r(t) dt", f"${acc_rev:,.2f}")
        c2.metric(f"Accumulated Cost ∫₀^{t_end} c(t) dt", f"${acc_cost:,.2f}")
        c3.metric("Net Accumulated Gain", f"${acc_net:,.2f}")

        fig_int = go.Figure()
        fig_int.add_trace(go.Scatter(x=t_arr, y=r_t, name="Revenue Rate r(t) = 100 + 20t ($/day)", line=dict(color="#1f77b4", width=3)))
        fig_int.add_trace(go.Scatter(x=t_arr, y=c_t, name="Cost Rate c(t) = 60 + 5t ($/day)", fill="tonexty", line=dict(color="#d62728", width=3)))
        fig_int.update_layout(
            title=f"Shaded Net Accumulation Over [0, {t_end}] Days (Units: Dollars)",
            xaxis_title="Time t (days)",
            yaxis_title="Rate (USD / day)",
            height=360,
        )
        st.plotly_chart(fig_int, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Operations, Dated Ledger & Fixture Replay
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Dated Event Ledger & Simulation Runs")

    col_run, col_fix = st.columns(2)
    with col_run:
        st.markdown("#### Advance One Approved Simulation Month (Normal Mode)")
        run_date = st.text_input("Simulation Date (YYYY-MM-DD)", value="2027-01-31")
        custom_run_id = st.text_input("Unique Run ID", value=f"RUN-{len(state['ledger']) + 1:03d}")
        if st.button("Approve & Record Month in Ledger", type="primary"):
            ok, msg, _ = record_simulation_run(state, custom_run_id.strip(), run_date.strip(), mode_label="Normal")
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with col_fix:
        st.markdown("#### NUR Package 3-Month Fixture Replay (Test Mode)")
        st.warning("Fixture mode overrides capacity with `60, 60, 45` and adds `$100` ad cost in Feb 2027 to verify ledger math.")
        if st.button("Reset & Run 3-Month NUR Fixture"):
            fresh = init_default_state()
            fixture_rows = [
                ("FIX-2027-01", "2027-01-31", 80, 60, 0.0),
                ("FIX-2027-02", "2027-02-28", 88, 60, 100.0),
                ("FIX-2027-03", "2027-03-31", 72, 45, 0.0),
            ]
            for rid, dt, req, cap_ov, ad in fixture_rows:
                record_simulation_run(
                    fresh, rid, dt, requested_override=req, capacity_override=cap_ov, ad_cost_override=ad, mode_label="Fixture"
                )
            st.session_state.biz_state = fresh
            st.success("Replayed NUR 3-month fixture into clean ledger.")
            st.rerun()

    st.divider()
    if state["ledger"]:
        df_ledger = pd.DataFrame(state["ledger"])
        st.dataframe(df_ledger, use_container_width=True, hide_index=True)

        csv_buf = StringIO()
        df_ledger.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Ledger CSV",
            data=csv_buf.getvalue(),
            file_name=f"{state['team_id']}_NUR3_ledger_v{state['state_version']}.csv",
            mime="text/csv",
        )
    else:
        st.info("No transactions recorded yet. Approve a monthly run or replay the NUR 3-month fixture above.")

# ---------------------------------------------------------------------------
# TAB 4: Manual AI Handoffs & Stage 1 Acceptance Tests
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Stage 1 Automated Acceptance Checks")
    st.caption("Deterministic verification against the NUR Package and AP Calculus Adapter rules.")
    df_tests = run_nur3_acceptance_tests()
    st.dataframe(df_tests, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Manual AI Assistant Handoff Importer")
    st.caption("Paste a structured JSON recommendation from your Finance, Operations, Marketing, or Reviewer assistant. Stale versions or mismatched team IDs are automatically rejected.")
    rec_raw = st.text_area(
        "Recommendation JSON Payload",
        height=140,
        placeholder='{"role": "Finance", "team_id": "NURSES-IN-HEALTHCARE", "state_version": 1, "question": "...", "recommendation": "...", "approval_status": "pending"}',
    )
    if st.button("Validate & Import Recommendation"):
        try:
            rec = json.loads(rec_raw)
            if rec.get("team_id") != state["team_id"]:
                st.error(f"Rejected: team_id mismatch (expected '{state['team_id']}').")
            elif rec.get("state_version") != state["state_version"]:
                st.error(
                    f"Rejected stale/mismatched state_version v{rec.get('state_version')} "
                    f"(current dashboard is v{state['state_version']})."
                )
            else:
                rec["approval_status"] = "pending"
                rec["imported_at"] = datetime.now(timezone.utc).isoformat()
                state["recommendations"].append(rec)
                st.success("Recommendation imported with status = pending. Review and approve before changing inputs.")
        except Exception as exc:
            st.error(f"Invalid JSON: {exc}")

    if state["recommendations"]:
        for idx, r in enumerate(state["recommendations"]):
            with st.expander(f"Recommendation #{idx + 1} — {r.get('role', 'Specialist')} (v{r.get('state_version')}) [{r.get('approval_status')}]"):
                st.json(r)
                if r.get("approval_status") == "pending":
                    if st.button(f"Mark Recommendation #{idx + 1} Approved", key=f"app_rec_{idx}"):
                        state["recommendations"][idx]["approval_status"] = "approved"
                        st.rerun()

# ---------------------------------------------------------------------------
# TAB 5: Stage 2-4 Implementation Roadmap
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Honest Implementation Status (Staged Rollout)")
    roadmap = pd.DataFrame([
        {"Stage": "Stage 1 (Current)", "Modules": "NUR-3 Inputs, AP Calculus Optimization & Integral Panel, Dated Ledger, JSON Save/Restore, Fixture & Stale-Version Tests", "Status": "Implemented"},
        {"Stage": "Stage 2 (Next)", "Modules": "Full 8-Report Center (Income Statement, Cash Flow, Balance Sheet, Variance), Costed Marketing Studio, Two-Report Decision Memo Builder", "Status": "Not Yet Implemented"},
        {"Stage": "Stage 3", "Modules": "Cross-Course Connections Panel (Health Literacy & Economics), Frozen-Snapshot Practice Assessment Generator", "Status": "Not Yet Implemented"},
        {"Stage": "Stage 4 (Optional)", "Modules": "Fictional Demonstration Storefront (No Real Payments)", "Status": "Not Yet Implemented"},
    ])
    st.table(roadmap)
