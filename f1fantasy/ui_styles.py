"""Lightweight visual system. Native widgets own behaviour; CSS owns layout.

Selectors use our explicit container keys, semantic roles and Streamlit test IDs.
The mobile subviews remain mounted so changing width never resets user inputs.
"""

DASHBOARD_CSS = """
<style>
:root {
    --f1-bg: #0e1116;
    --f1-panel: #151a21;
    --f1-panel-soft: #1c232d;
    --f1-border: #303946;
    --f1-text: #f2f5f8;
    --f1-muted: #acb6c5;
    --f1-red: #ef5350;
    --f1-red-soft: #302022;
    --f1-green: #79d7aa;
    --f1-radius: 8px;
}
.stApp { background: var(--f1-bg); color: var(--f1-text); }
[data-testid="stHeader"] { background: var(--f1-bg); }
[data-testid="stMainBlockContainer"] {
    max-width: 1360px;
    padding: 2.5rem 2.5rem 3rem;
}
[data-testid="stVerticalBlock"] { gap: 1rem; }
[data-testid="stColumn"], [data-testid="stVerticalBlock"],
[data-testid="stElementContainer"] { min-width: 0; }
h1, h2, h3, h4, h5 { color: var(--f1-text); letter-spacing: -0.025em; line-height: 1.25; }
[data-testid="stMarkdownContainer"] h1 { font-size: 1.8rem; font-weight: 750; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.55rem; }
[data-testid="stMarkdownContainer"] h3 { font-size: 1.2rem; }
[data-testid="stMarkdownContainer"] h4 { font-size: 1.1rem; }
[data-testid="stMarkdownContainer"] h5 { font-size: 1rem; }
p { line-height: 1.5; }
[data-testid="stCaptionContainer"] { color: var(--f1-muted); font-size: .875rem; }
.f1-app-header h1 { margin: .1rem 0; padding: 0; line-height: 1.25; }
.f1-app-header p { margin: 0; font-size: .9rem; color: var(--f1-muted); }
.f1-wordmark { color: var(--f1-red); font-size: .75rem; font-weight: 750; letter-spacing: .14em; }
.st-key-app_header { padding-bottom: .35rem; }
.st-key-app_header button { white-space: nowrap; }
.f1-race-card {
    display: grid; grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center; gap: 1rem; padding: .7rem 1rem;
    background: var(--f1-panel); border-radius: var(--f1-radius);
    border-left: 3px solid var(--f1-red);
}
.f1-round { font-size: .8rem; font-weight: 650; color: var(--f1-muted); white-space: nowrap; }
.f1-race-name { font-size: 1.2rem; line-height: 1.35; font-weight: 700; }
.f1-race-sub { color: var(--f1-muted); font-size: .75rem; letter-spacing: .03em; }
.f1-race-date { font-size: .85rem; color: var(--f1-muted); }
.f1-race-card > div:last-child { text-align: right; }
.f1-race-countdown { font-size: .9rem; font-weight: 650; }
.st-key-race_context { gap: .5rem; }
.st-key-data_notices:empty { display: none; }
.st-key-data_notices [data-testid="stAlert"] { padding: .6rem .8rem; }
.st-key-data_notices [data-testid="stMarkdownContainer"] p { font-size: .85rem; }
/* Hide marker-only blocks so they do not create empty flex gaps. */
[data-testid="stElementContainer"]:has(.f1-optimise-view-marker),
[data-testid="stElementContainer"]:has([class^="f1-universe-desktop-"]) { display: none; }
/* Semantic roles work across the BaseWeb and React Aria tab implementations. */
[data-testid="stTabs"] [role="tablist"] {
    gap: .4rem; border-bottom: 1px solid var(--f1-border);
    overflow-x: auto; flex-wrap: nowrap; scrollbar-width: thin;
}
[data-testid="stTabs"] [role="tab"] {
    min-height: 46px; flex: 0 0 auto; padding: .55rem 1rem;
    font-size: .95rem; font-weight: 550; color: var(--f1-muted);
    display: flex; justify-content: center; align-items: center; text-align: center;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--f1-text); box-shadow: inset 0 -2px var(--f1-red);
}
/* The native moving indicator retains stale widths after responsive resizing. */
[data-testid="stTabs"] .react-aria-SelectionIndicator,
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
.st-key-primary_navigation > div > [role="tablist"] > [role="tab"] { font-weight: 650; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: .65rem; }
.f1-section-kicker {
    color: var(--f1-muted); font-size: .73rem; font-weight: 650;
    letter-spacing: .11em; margin: .2rem 0; text-transform: uppercase;
}
[data-testid="stWidgetLabel"] p { font-size: .9rem; font-weight: 550; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stMultiSelect"] [role="combobox"] { min-height: 44px; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] { background: #273242; color: var(--f1-text); border-radius: 4px; max-width: 100%; min-height: 34px; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] > span[title] { max-width: none; white-space: normal; overflow-wrap: anywhere; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] [role="presentation"] { min-width: 28px; min-height: 30px; }
.st-key-current_team_driver_ids [data-baseweb="select"] > div,
.st-key-current_team_constructor_ids [data-baseweb="select"] > div,
.st-key-current_team_driver_ids div:has(> [data-baseweb="tag"]),
.st-key-current_team_constructor_ids div:has(> [data-baseweb="tag"]) { max-height: none; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    border-radius: 6px; font-weight: 600; padding: .55rem .9rem;
}
button[kind="primary"] { background: #d84343; color: white; border-color: #d84343; }
button[kind="primary"]:hover { background: #e45151; border-color: #e45151; color: white; }
button:focus-visible, [role="slider"]:focus-visible, input:focus-visible,
[role="combobox"]:focus-visible { outline: 2px solid #a9c7ff; outline-offset: 3px; }
[data-testid="stButtonGroup"] button { min-height: 44px; font-size: .9rem; padding: .5rem .85rem; }
[data-testid="stExpander"] { border-radius: var(--f1-radius); border-color: var(--f1-border); }
[data-testid="stExpander"] summary { min-height: 48px; }
[data-testid="stMetric"] {
    background: var(--f1-panel); padding: .9rem 1rem;
    border-radius: 6px; font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] { color: var(--f1-muted); }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 650; }
[data-testid="stMetricValue"] > div { white-space: normal; overflow-wrap: anywhere; }
[data-testid="stAlert"] { border-radius: 6px; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { max-width: 100%; }
.st-key-optimiser_quick_setup { background: var(--f1-panel); border-color: var(--f1-border); }
.st-key-optimiser_controls_view, .st-key-optimiser_teams_view { gap: .75rem; }
.f1-setup-summary { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: .8rem; }
.f1-setup-summary span { display: block; color: var(--f1-muted); font-size: .8rem; }
.f1-setup-summary strong { font-size: .95rem; font-weight: 650; }
.f1-setup-hint { color: var(--f1-muted); font-size: .85rem; margin: .7rem 0 0; }
.f1-empty-state { padding: 2rem; background: var(--f1-panel); border-radius: var(--f1-radius); }
.f1-empty-state [data-testid="stMarkdownContainer"] h3 { font-size: 1.5rem; margin: .65rem 0; }
.f1-empty-state p { max-width: 48ch; margin: .65rem 0 0; }
.f1-empty-detail { color: var(--f1-muted); font-size: .9rem; }
.f1-empty-table { color: var(--f1-muted); padding: 1rem; background: var(--f1-panel); border-radius: 6px; }
/* Shared identity: names remain visible on touch screens, with team colour secondary. */
.f1-asset-identity { display: inline-flex; align-items: center; gap: .5rem; min-width: 0; max-width: 100%; }
.f1-asset-id {
    display: inline-flex; align-items: center; justify-content: center;
    flex: 0 0 auto; min-width: 2.6rem; min-height: 1.8rem;
    padding: .15rem .3rem; border-radius: 4px; font-size: .75rem; font-weight: 750;
    letter-spacing: .025em; white-space: nowrap;
}
.f1-asset-text { display: flex; flex-direction: column; min-width: 0; line-height: 1.3; }
.f1-asset-name { color: var(--f1-text); font-size: .9rem; font-weight: 600; overflow-wrap: anywhere; }
.f1-asset-team { color: var(--f1-muted); font-size: .75rem; margin-top: .12rem; }
.f1-gain-positive { color: #79d7aa !important; }
.f1-gain-negative { color: #ff9aa3 !important; }
.f1-gain-neutral { color: #c8d0db !important; }
.f1-gain-missing { color: var(--f1-muted) !important; }
.f1-boost { padding: .12rem .3rem; background: var(--f1-red-soft); color: #ffaca9; font-size: .75rem; font-weight: 750; border-radius: 3px; white-space: nowrap; }
.f1-availability-muted { color: var(--f1-muted); font-size: .7rem; font-weight: 500; }
/* Ranked teams: readable names and explicit metric labels. */
.f1-ranked-team {
    padding: 1rem; margin: 0 0 1rem; border: 1px solid var(--f1-border);
    border-radius: var(--f1-radius); background: var(--f1-panel);
    font-variant-numeric: tabular-nums;
}
.f1-ranked-team[data-rank="1"] { border-top: 3px solid var(--f1-red); }
.f1-team-header { display: grid; grid-template-columns: 2rem minmax(0,1fr); gap: .8rem; align-items: center; margin-bottom: .75rem; }
.f1-team-rank { display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: 4px; color: var(--f1-text); background: var(--f1-panel-soft); font-weight: 700; }
.f1-team-summary { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: .7rem; }
.f1-team-stat { min-width: 0; }
.f1-team-stat > span { display: block; color: var(--f1-muted); font-size: .75rem; line-height: 1.3; }
.f1-team-stat strong { display: block; margin-top: .2rem; font-size: 1.05rem; font-weight: 650; white-space: nowrap; }
.f1-team-stat:last-child strong { font-size: 1.2rem; }
.f1-team-assets { display: grid; gap: .8rem; }
.f1-team-section-label { color: var(--f1-muted); font-size: .72rem; font-weight: 600; letter-spacing: .06em; margin: 0 0 .25rem; text-transform: uppercase; }
.f1-card-grid { display: grid; grid-template-columns: 1fr; gap: 0; }
.f1-driver-card {
    display: grid; grid-template-columns: minmax(0,1fr) 9.3rem 5rem; align-items: center;
    gap: .8rem; padding: .6rem .55rem; border-bottom: 1px solid #29313b;
    border-left: 2px solid var(--team-color, #64748b); min-width: 0;
}
.f1-driver-card:last-child { border-bottom: 0; }
.f1-card-grid .f1-driver-card:not(:first-child) .f1-card-label { display: none; }
.f1-card-top, .f1-card-identity { min-width: 0; }
.f1-card-identity { display: flex; align-items: center; gap: .35rem; }
.f1-card-middle { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: .75rem; text-align: right; }
.f1-card-label { display: block; color: var(--f1-muted); font-size: .7rem; font-weight: 450; white-space: nowrap; }
.f1-card-value { display: block; font-size: .88rem; font-weight: 600; white-space: nowrap; }
.f1-card-points { text-align: right; }
.f1-card-points .f1-card-value { color: var(--f1-text); font-weight: 700; }
/* The inline lock/exclude matrix deliberately remains a table at all widths. */
.f1-universe-heading { display: block; text-align: right; color: var(--f1-muted); font-size: .7rem; font-weight: 600; white-space: nowrap; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stColumn"]:first-child .f1-universe-heading { text-align: left; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stColumn"]:nth-child(n+5) .f1-universe-heading { text-align: center; }
.f1-universe-number { display: block; font-size: .83rem; font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }
[class*="st-key-optimiser_universe_scroll"] { padding: .65rem !important; gap: .35rem; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 3px !important; align-items: center; border-bottom: 1px solid #29313b; min-height: 48px; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 0 !important; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stVerticalBlock"] { gap: .35rem; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stMarkdownContainer"] p { margin: 0; }
[class*="st-key-optimiser_universe_scroll"] .f1-asset-identity { display: block; }
[class*="st-key-optimiser_universe_scroll"] .f1-asset-id { display: none; }
[class*="st-key-optimiser_universe_scroll"] .f1-asset-name { font-size: .78rem; }
[class*="st-key-optimiser_universe_scroll"] .f1-asset-team { display: none; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stElementContainer"]:has(> [data-testid="stCheckbox"]) { width: 100% !important; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stCheckbox"] { display: flex; justify-content: center; width: 100%; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stCheckbox"] label { min-height: 44px; align-items: center; justify-content: center; padding: 0; width: 100%; margin: 0; }
[class*="st-key-optimiser_universe_scroll"] [data-testid="stCheckbox"] label > span { margin: 0; }
/* Tables scroll inside their own boundary. Numeric precision stays in the render helpers. */
.f1-table-scroll { width: 100%; overflow-x: auto; border: 1px solid var(--f1-border); border-radius: 6px; margin: .35rem 0 .75rem; }
.f1-compact-table { width: 100%; border-collapse: collapse; font-size: .9rem; font-variant-numeric: tabular-nums; }
.f1-compact-table th, .f1-compact-table td { padding: .7rem .8rem; text-align: right; border: 0; border-bottom: 1px solid #29313b; vertical-align: middle; }
.f1-compact-table th { color: var(--f1-muted); background: var(--f1-panel); font-size: .78rem; font-weight: 600; white-space: nowrap; }
.f1-compact-table th:first-child, .f1-compact-table td:first-child { text-align: left; background: var(--f1-panel); }
.f1-compact-table tbody tr:last-child td { border-bottom: 0; }
.f1-compact-table td:not(:first-child) { white-space: nowrap; }
.f1-compact-table small { color: var(--f1-muted); display: block; font-size: .75rem; }
.f1-price-change-table { min-width: 720px; }
.f1-price-change-table th:first-child, .f1-price-change-table td:first-child { min-width: 12rem; position: sticky; left: 0; z-index: 1; }
.f1-table-note { color: var(--f1-muted); font-size: .8rem; padding: .6rem; }
/* Both Market views share a compact reading width and consistent column rhythm. */
.st-key-market_outlook, .f1-market-table-wrap, .f1-universe-scroll, .st-key-market_sort_controls { max-width: 1040px; }
.f1-market-table, .f1-universe-table { table-layout: fixed; }
.f1-market-table th, .f1-market-table td,
.f1-universe-scroll th, .f1-universe-scroll td { padding: .5rem .65rem; }
.f1-threshold-table th:first-child, .f1-threshold-table td:first-child { width: 32%; }
.f1-universe-table th:first-child, .f1-universe-table td:first-child { width: 40%; }
.f1-market-table .f1-asset-name, .f1-universe-table .f1-asset-name { font-size: .85rem; }
.f1-band-terrible { background: #52282b; }
.f1-band-poor { background: #3e2b30; }
.f1-band-good { background: #283d32; }
.f1-band-great { background: #1e4934; }
.f1-transfer-row { display: grid; grid-template-columns: minmax(0,1fr) auto minmax(0,1fr); gap: 1rem; align-items: center; margin-bottom: 1rem; }
.f1-transfer-row .f1-driver-card { display: flex; flex-direction: column; align-items: stretch; }
.f1-transfer-arrow { font-size: 1.5rem; color: var(--f1-muted); text-align: center; }
.f1-mobile-table, .st-key-optimise_mobile_subview,
.st-key-optimiser_teams_action,
[class*="st-key-sprint_diagnostics_mobile"] { display: none; }
[data-testid="stLayoutWrapper"]:has(> .st-key-optimiser_teams_action) { display: none; }
@media (min-width: 769px) {
    /* Desktop uses its width for side-by-side assets; phones keep legible rows. */
    .f1-ranked-team .f1-driver-grid { grid-template-columns: repeat(5, minmax(0,1fr)); gap: .45rem; }
    .f1-ranked-team .f1-constructor-grid { grid-template-columns: repeat(2, minmax(0,1fr)); gap: .45rem; width: calc(40% - .27rem); margin-inline: auto; }
    .f1-ranked-team .f1-driver-card {
        display: flex; flex-direction: column; align-items: stretch; justify-content: space-between;
        padding: .6rem; gap: .6rem; border: 0; border-top: 2px solid var(--team-color, #64748b);
        background: var(--f1-panel-soft); border-radius: 3px; min-height: 130px;
    }
    .f1-ranked-team .f1-team-section:nth-child(2) .f1-team-section-label { text-align: center; }
    .f1-ranked-team .f1-asset-id { display: none; }
    .f1-ranked-team .f1-asset-name { font-size: .85rem; }
    .f1-ranked-team .f1-asset-team { font-size: .7rem; }
    .f1-ranked-team .f1-card-identity { flex-wrap: wrap; }
    .f1-ranked-team .f1-card-middle { gap: .35rem; }
    .f1-ranked-team .f1-card-price { text-align: left; }
    .f1-ranked-team .f1-card-points { display: flex; justify-content: space-between; align-items: baseline; gap: .25rem; }
    .f1-ranked-team .f1-card-grid .f1-driver-card .f1-card-label { display: block; font-size: .68rem; }
    .f1-ranked-team .f1-card-value { font-size: .8rem; }
    body:has(.f1-universe-desktop-drivers) .st-key-optimiser_constructors_view,
    body:has(.f1-universe-desktop-constructors) .st-key-optimiser_drivers_view { display: none; }
}
@media (min-width: 769px) and (max-width: 1100px) {
    .f1-ranked-team .f1-driver-card { padding: .5rem .35rem; }
    [data-testid="stMainBlockContainer"] { padding-inline: 1.5rem; }
    .st-key-optimiser_primary_controls [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-optimiser_primary_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 calc(50% - 1rem) !important; min-width: 0 !important; }
    .f1-driver-card { grid-template-columns: minmax(0,1fr) 7.4rem 4.5rem; gap: .4rem; padding-inline: .4rem; }
    .f1-driver-card .f1-asset-id { display: none; }
    .f1-team-summary { gap: .45rem; }
    .f1-team-stat strong { font-size: .95rem; }
    .f1-team-header { grid-template-columns: 1.6rem minmax(0,1fr); gap: .5rem; }
}
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] { padding: 2.7rem 1.1rem calc(2rem + env(safe-area-inset-bottom)); }
    /* Stack control groups, with explicit exceptions for metrics and asset rows. */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .85rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100%; min-width: 100%; }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"] { flex: 1 1 calc(50% - .85rem); min-width: 0; }
    .st-key-app_header [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: .6rem; }
    .st-key-app_header [data-testid="stColumn"]:first-child { flex: 1 1 auto; min-width: 0; }
    .st-key-app_header [data-testid="stColumn"]:last-child { flex: 0 0 8rem; min-width: 0; }
    [data-testid="stMarkdownContainer"] .f1-app-header h1 { font-size: 1.3rem; }
    .f1-wordmark { font-size: .67rem; }
    .f1-app-header p { display: none; }
    .st-key-app_header button { font-size: .8rem; padding-inline: .55rem; }
    .f1-race-card { grid-template-columns: minmax(0,1fr) auto; padding: .85rem; gap: .25rem .65rem; }
    .f1-round { grid-column: 1; font-size: .72rem; }
    .f1-race-card > div:nth-child(2) { grid-column: 1; grid-row: 2; }
    .f1-race-card > div:last-child { grid-column: 2; grid-row: 1 / span 2; }
    .f1-race-card > div:nth-child(2) .f1-race-sub { display: none; }
    .f1-race-name { font-size: 1.05rem; }
    .f1-race-date, .f1-race-countdown { font-size: .78rem; }
    .st-key-primary_navigation > div > [role="tablist"] { gap: 0; }
    .st-key-primary_navigation > div > [role="tablist"] > [role="tab"] { flex: 1 1 0; min-width: 0; padding-inline: .5rem; }
    [data-testid="stTabs"] [role="tab"] { font-size: .9rem; padding-inline: .75rem; }
    [data-testid="stMetric"] { padding: .8rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    input, [role="combobox"] { font-size: 16px !important; }
    .st-key-optimise_mobile_subview { display: block; width: 100% !important; }
    .st-key-optimise_mobile_subview [data-testid="stButtonGroup"] { width: 100% !important; }
    .st-key-optimise_mobile_subview [data-testid="stButtonGroup"] > div { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); width: 100%; max-width: none; gap: 0; }
    .st-key-optimise_mobile_subview button { min-width: 0; padding: .5rem .2rem; font-size: .8rem; }
    .st-key-optimise_mobile_subview button p { font-size: .78rem; }
    .st-key-optimiser_controls_view, .st-key-optimiser_teams_view,
    .st-key-optimiser_drivers_view, .st-key-optimiser_constructors_view { display: none; }
    body:has(.f1-optimise-view-teams) .st-key-optimiser_teams_action,
    body:has(.f1-optimise-view-teams) .st-key-optimiser_teams_view,
    body:has(.f1-optimise-view-drivers) .st-key-optimiser_drivers_view,
    body:has(.f1-optimise-view-constructors) .st-key-optimiser_constructors_view,
    body:has(.f1-optimise-view-controls) .st-key-optimiser_controls_view { display: block; }
    [data-testid="stLayoutWrapper"]:has(> .st-key-optimiser_controls_view),
    [data-testid="stLayoutWrapper"]:has(> .st-key-optimiser_universe_selector) { display: none; }
    body:has(.f1-optimise-view-teams) [data-testid="stLayoutWrapper"]:has(> .st-key-optimiser_teams_action),
    body:has(.f1-optimise-view-controls) [data-testid="stLayoutWrapper"]:has(> .st-key-optimiser_controls_view) { display: block; }
    .st-key-optimiser_teams_action { padding: 1rem; background: var(--f1-panel); border-radius: var(--f1-radius); }
    .st-key-optimiser_teams_action [data-testid="stButton"] { margin-top: .85rem; }
    .st-key-optimiser_universe_selector { display: none; }
    /* Remove invisible layout columns and nested result scrolling on phones. */
    .st-key-optimiser_dashboard > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:not(:has(.st-key-optimiser_teams_view)) { display: none; }
    body:has(.f1-optimise-view-drivers, .f1-optimise-view-constructors) .st-key-optimiser_dashboard > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:not(:has(.st-key-optimiser_teams_view)) { display: block; }
    body:has(.f1-optimise-view-drivers, .f1-optimise-view-constructors, .f1-optimise-view-controls) .st-key-optimiser_dashboard > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-optimiser_teams_view) { display: none; }
    .st-key-optimiser_results_scroll, [class*="st-key-optimiser_universe_scroll"],
    [data-testid="stLayoutWrapper"]:has(> .st-key-optimiser_results_scroll),
    [data-testid="stLayoutWrapper"]:has(> [class*="st-key-optimiser_universe_scroll"]) { height: auto !important; max-height: none !important; overflow: visible !important; }
    [class*="st-key-optimiser_universe_scroll"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 0; min-width: 0 !important; }
    [class*="st-key-optimiser_universe_scroll"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child { flex-grow: 2; }
    [class*="st-key-optimiser_universe_scroll"] .f1-asset-id { display: inline-flex; min-width: 2.5rem; }
    [class*="st-key-optimiser_universe_scroll"] .f1-asset-text { display: none; }
    .st-key-optimiser_primary_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 100% !important; }
    .st-key-optimiser_primary_controls [data-testid="stColumn"]:nth-child(2),
    .st-key-optimiser_primary_controls [data-testid="stColumn"]:nth-child(3) { flex: 1 1 calc(50% - .85rem) !important; }
    .f1-ranked-team { padding: .85rem; }
    .f1-team-header { grid-template-columns: 1.7rem minmax(0,1fr); align-items: start; gap: .6rem; }
    .f1-team-rank { width: 1.7rem; height: 1.7rem; font-size: .85rem; }
    .f1-team-summary { grid-template-columns: repeat(2,minmax(0,1fr)); gap: .65rem 1rem; }
    .f1-team-stat strong { font-size: 1rem; }
    .f1-team-stat:last-child strong { font-size: 1.15rem; }
    .f1-driver-card { grid-template-columns: minmax(0,1fr) 7.1rem 4.1rem; gap: .4rem; padding: .65rem .35rem; }
    .f1-driver-card .f1-asset-id { display: none; }
    .f1-driver-card .f1-asset-name { font-size: .84rem; }
    .f1-driver-card .f1-asset-team { font-size: .72rem; }
    .f1-card-middle { gap: .5rem; }
    .f1-card-label { font-size: .7rem; }
    .f1-card-value { font-size: .84rem; }
    .f1-card-identity { flex-wrap: wrap; gap: .15rem; }
    .f1-boost { font-size: .65rem; }
    .f1-empty-state { padding: 1.25rem; }
    .f1-empty-state [data-testid="stMarkdownContainer"] h3 { font-size: 1.3rem; }
    .f1-desktop-table, [class*="st-key-sprint_diagnostics_desktop"] { display: none; }
    .f1-mobile-table, [class*="st-key-sprint_diagnostics_mobile"] { display: block; }
    .f1-mobile-schema { width: 100%; min-width: 0; table-layout: fixed; font-size: .85rem; }
    .f1-mobile-schema th, .f1-mobile-schema td { padding: .65rem .4rem; }
    .f1-mobile-schema th:first-child, .f1-mobile-schema td:first-child { width: 40%; }
    .f1-mobile-schema td:not(:first-child) { white-space: normal; overflow-wrap: break-word; }
    .f1-mobile-schema .f1-asset-id { display: none; }
    .f1-mobile-schema .f1-asset-name { font-size: .83rem; }
    .f1-mobile-schema .f1-asset-team { font-size: .72rem; }
    .f1-mobile-schema .f1-availability-muted { display: block; }
    .f1-threshold-table { font-size: .75rem; }
    .f1-threshold-table th, .f1-threshold-table td { padding: .55rem .25rem; white-space: normal; }
    .f1-threshold-table th { font-size: .7rem; }
    .f1-threshold-table th:first-child, .f1-threshold-table td:first-child { width: 16%; }
    .f1-threshold-table th:nth-child(2), .f1-threshold-table td:nth-child(2) { width: 13%; }
    .f1-threshold-table .f1-asset-text, .f1-projection-mobile .f1-asset-text { display: none; }
    .f1-threshold-table .f1-asset-id, .f1-projection-mobile .f1-asset-id { display: inline-flex; min-width: 2.4rem; font-size: .7rem; }
    .f1-projection-mobile th:first-child, .f1-projection-mobile td:first-child { width: 22%; }
    .f1-projection-mobile th { font-size: .73rem; }
    .f1-projection-mobile .f1-availability-muted { font-size: .65rem; line-height: 1.2; }
    .st-key-market_sort_controls [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; }
    .st-key-market_sort_controls [data-testid="stColumn"] { flex: 2 1 0; min-width: 0; }
    .st-key-market_sort_controls [data-testid="stColumn"]:last-child { flex: 1 1 0; }
    .f1-transfer-row { grid-template-columns: 1fr; gap: .5rem; }
    .f1-transfer-arrow { transform: rotate(90deg); }
}
@media (max-width: 400px) {
    [data-testid="stMainBlockContainer"] { padding-inline: .85rem; }
    .f1-ranked-team { padding: .7rem; }
    .f1-driver-card { grid-template-columns: minmax(0,1fr) 6.6rem 3.7rem; gap: .3rem; }
    .f1-card-value { font-size: .82rem; }
    .f1-card-label { font-size: .7rem; }
    .f1-team-stat > span { font-size: .73rem; }
}
</style>
"""
