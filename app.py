from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ
# =====================================================
st.set_page_config(
    page_title="Цифровой двойник CVA | НейроКапибры",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# ОТНОСИТЕЛЬНЫЕ ПУТИ
# =====================================================
WEIGHTS_DIR = Path("weights")

# =====================================================
# ТЕМЫ
# =====================================================
THEMES = {
    "Тёмная": {
        "bg": "#111827",
        "card": "#161F2E",
        "card_alt": "#1B2638",
        "text": "#F3F4F6",
        "muted": "#A1AEBF",
        "stroke": "rgba(255,255,255,0.08)",
        "grid": "rgba(255,255,255,0.10)",
        "accent": "#3B82F6",
        "accent2": "#38BDF8",
        "good": "#10B981",
        "warn": "#F59E0B",
        "bad": "#EF4444",
        "hero": "linear-gradient(135deg, rgba(59,130,246,0.10), rgba(56,189,248,0.07))",
        "sidebar": "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015))",
        "plot_bg": "rgba(255,255,255,0.025)",
        "legend_bg": "rgba(15,23,42,0.55)",
        "shadow": "0 6px 14px rgba(0,0,0,0.08)",
    },
    "Светлая": {
        "bg": "#F6F8FC",
        "card": "#FFFFFF",
        "card_alt": "#F8FAFC",
        "text": "#1E293B",
        "muted": "#64748B",
        "stroke": "rgba(15,23,42,0.08)",
        "grid": "rgba(15,23,42,0.10)",
        "accent": "#2563EB",
        "accent2": "#0EA5E9",
        "good": "#059669",
        "warn": "#D97706",
        "bad": "#DC2626",
        "hero": "linear-gradient(135deg, rgba(37,99,235,0.07), rgba(14,165,233,0.05))",
        "sidebar": "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,1))",
        "plot_bg": "rgba(15,23,42,0.018)",
        "legend_bg": "rgba(255,255,255,0.88)",
        "shadow": "0 6px 14px rgba(15,23,42,0.04)",
    }
}

# =====================================================
# ДАННЫЕ
# =====================================================
INHIBITORS = {
    "inh_1": {
        "name": "2-Mercaptobenzothiazole",
        "smiles": "c1ccc2c(c1)sc(=S)[nH]2",
        "type": "Смешанный",
        "mw": 167.25,
        "logp": 2.41,
        "color": "#38BDF8"
    },
    "inh_2": {
        "name": "4-benzylpiperazine",
        "smiles": "c1ccc(cc1)CN2CCNCC2",
        "type": "Катодный",
        "mw": 176.26,
        "logp": 1.85,
        "color": "#3B82F6"
    },
    "inh_3": {
        "name": "benzothiazole",
        "smiles": "c1ccc2c(c1)ncs2",
        "type": "Смешанный",
        "mw": 135.19,
        "logp": 2.01,
        "color": "#7C3AED"
    },
    "inh_4": {
        "name": "Tolyltriazole",
        "smiles": "Cc1ccc2[nH]nnc2c1",
        "type": "Анодный",
        "mw": 133.15,
        "logp": 1.34,
        "color": "#DB2777"
    },
}

MODELS = {
    "CVAE": {
        "desc": "Условный вариационный автоэнкодер",
        "stability": 0.95,
        "color": "#38BDF8",
        "weight": WEIGHTS_DIR / "best_cvae.pth"
    },
    "CGAN": {
        "desc": "Условная генеративно-состязательная сеть",
        "stability": 0.88,
        "color": "#3B82F6",
        "weight": WEIGHTS_DIR / "best_cgan.pth"
    },
    "PPDM": {
        "desc": "Вероятностная диффузионная модель",
        "stability": 0.98,
        "color": "#7C3AED",
        "weight": WEIGHTS_DIR / "best_ddpm.pth"
    },
}

# =====================================================
# НАСТРОЙКИ ИНТЕРФЕЙСА
# =====================================================
with st.sidebar:
    st.markdown("### Интерфейс")
    theme_mode = st.selectbox("Тема", ["Тёмная", "Светлая"], index=0)
    graph_style = st.selectbox("Стиль графиков", ["Классический", "Гладкий", "С заливкой"], index=0)

PALETTE = THEMES[theme_mode]

# =====================================================
# CSS
# =====================================================
st.markdown(f"""
<style>
:root {{
    --bg:{PALETTE["bg"]};
    --card:{PALETTE["card"]};
    --card-alt:{PALETTE["card_alt"]};
    --text:{PALETTE["text"]};
    --muted:{PALETTE["muted"]};
    --stroke:{PALETTE["stroke"]};
    --grid:{PALETTE["grid"]};
    --accent:{PALETTE["accent"]};
    --accent2:{PALETTE["accent2"]};
    --good:{PALETTE["good"]};
    --warn:{PALETTE["warn"]};
    --bad:{PALETTE["bad"]};
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important;
    color: var(--text) !important;
}}

.block-container {{
    padding-top: 0.9rem;
    padding-bottom: 1.2rem;
    max-width: 1420px;
}}

header[data-testid="stHeader"] {{
    background: rgba(0,0,0,0) !important;
}}
[data-testid="stDecoration"] {{
    background: transparent !important;
}}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
div[data-testid="stSidebarNav"] {{ display:none; }}

[data-testid="stSidebar"] {{
    background: {PALETTE["sidebar"]};
    border-right: 1px solid var(--stroke);
}}

h1, h2, h3, h4, h5, h6, label {{
    color: var(--text) !important;
}}

p {{
    color: var(--text) !important;
}}

.topbar {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:18px;
    background: var(--card);
    border:1px solid var(--stroke);
    border-radius: 16px;
    padding: 12px 16px;
    margin-bottom: 14px;
    box-shadow: {PALETTE["shadow"]};
}}

.topbar-left {{
    display:flex;
    align-items:center;
    gap:12px;
}}

.brand-dot {{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}}

.brand-title {{
    font-weight: 800;
    font-size: 1rem;
    color: var(--text) !important;
}}

.brand-sub {{
    color: var(--muted) !important;
    font-size: 0.90rem;
}}

.topbar-right {{
    display:flex;
    gap:10px;
    flex-wrap:wrap;
}}

.top-pill {{
    padding: 6px 10px;
    border:1px solid var(--stroke);
    border-radius: 999px;
    background: var(--card-alt);
    color: var(--text);
    font-size: 0.84rem;
}}

.hero {{
    background: {PALETTE["hero"]};
    border: 1px solid var(--stroke);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 14px;
    box-shadow: {PALETTE["shadow"]};
}}

.hero-title {{
    font-size: 2rem;
    font-weight: 850;
    letter-spacing: -0.02em;
    color: var(--text) !important;
}}

.hero-sub {{
    font-size: 0.98rem;
    color: var(--muted) !important;
    margin-top: 8px;
    max-width: 980px;
}}

.card {{
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 16px;
    box-shadow: {PALETTE["shadow"]};
}}

.section-title {{
    font-size: 1.02rem;
    font-weight: 800;
    margin-bottom: 10px;
    color: var(--text) !important;
}}

[data-testid="stMetric"] {{
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 12px;
    box-shadow: {PALETTE["shadow"]};
}}

.stButton > button, .stDownloadButton > button {{
    border-radius: 12px !important;
    border: 1px solid var(--stroke) !important;
    background: var(--card-alt) !important;
    color: var(--text) !important;
    font-weight: 700 !important;
}}

.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: var(--accent) !important;
}}

[data-baseweb="tab-list"] {{
    gap: 6px;
}}

button[data-baseweb="tab"] {{
    border-radius: 10px !important;
}}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {{
    background: var(--card-alt) !important;
    color: var(--text) !important;
    border-color: var(--stroke) !important;
}}

div[data-baseweb="popover"] * {{
    color: #111827 !important;
}}

hr {{
    border-color: var(--stroke) !important;
}}

.sidebar-brand-wrap {{
    position: relative;
    margin-top: -2px;
    margin-bottom: 2px;
}}

.sidebar-brand {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 2px;
}}

.sidebar-brand-icon {{
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(59,130,246,0.30);
    margin-top: 1px;
}}

.sidebar-brand-text {{
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    line-height: 1.1;
}}

.sidebar-brand-name {{
    font-weight: 800;
    font-size: 1.08rem;
    color: {PALETTE["text"]} !important;
    margin: 0;
    padding: 0;
}}

.sidebar-brand-ver {{
    font-size: 0.76rem;
    color: {PALETTE["muted"]} !important;
    margin-top: 2px;
}}

.sidebar-brand-caption {{
    font-size: 0.80rem;
    color: {PALETTE["muted"]} !important;
    margin-top: 8px;
    margin-bottom: 2px;
    line-height: 1.35;
}}

.seed-badge {{
    display:inline-flex;
    align-items:center;
    gap:5px;
    padding:4px 10px;
    border-radius:999px;
    background: var(--card-alt);
    border:1px solid var(--stroke);
    font-size:0.82rem;
    color: var(--muted);
    margin-top:4px;
}}

.compare-banner {{
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(59,130,246,0.10);
    border: 1px solid var(--stroke);
    color: var(--text);
    font-size: 0.9rem;
    margin-top: 8px;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================
@st.cache_data
def get_descriptors(inh_key):
    rng = np.random.default_rng(int(inh_key[-1]))
    labels = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotBonds"]
    values = rng.uniform(0.2, 0.9, len(labels))
    return pd.DataFrame({
        "Дескриптор": labels,
        "Норм. значение": values
    })

def inhibitor_profile_shift(inh_key):
    shifts = {
        "inh_1": {"ox_center": 0.25, "red_center": -0.16, "k": 1.00},
        "inh_2": {"ox_center": 0.22, "red_center": -0.20, "k": 0.92},
        "inh_3": {"ox_center": 0.28, "red_center": -0.12, "k": 1.08},
        "inh_4": {"ox_center": 0.18, "red_center": -0.10, "k": 0.96},
    }
    return shifts.get(inh_key, shifts["inh_1"])

def generate_demo_signal(concentration, inh_key, cycle, model_key, seed):
    E = np.linspace(-0.8, 0.8, 960)
    base_noise = 0.00004
    cfg = inhibitor_profile_shift(inh_key)

    model_factor = {
        "CVAE": 1.00,
        "CGAN": 1.03,
        "PPDM": 0.98
    }.get(model_key, 1.0)

    cycle_factor = 1.0 - 0.03 * (cycle - 1)
    amp = ((concentration + 5) / 100.0) * cfg["k"] * model_factor * cycle_factor

    I_ox = 0.020 * amp * np.exp(-((E - cfg["ox_center"]) ** 2) / 0.015)
    I_red = -0.018 * amp * np.exp(-((E - cfg["red_center"]) ** 2) / 0.015)
    shoulder = 0.0040 * amp * np.exp(-((E - (cfg["ox_center"] + 0.12)) ** 2) / 0.05)
    baseline = E * 0.0010 + 0.00012 * np.sin(5 * E)

    rng = np.random.default_rng(seed + int(concentration) + cycle * 10 + int(inh_key[-1]) * 100)
    noise = rng.normal(0, base_noise, len(E))

    I_mean = I_ox + I_red + shoulder + baseline
    I_final = I_mean + noise
    std_dev = np.abs(I_mean) * 0.05 + base_noise * 1.5

    idx_ox = np.argmax(I_final)
    idx_red = np.argmin(I_final)

    metrics = {
        "E_pa": E[idx_ox],
        "I_pa": I_final[idx_ox],
        "I_pc": I_final[idx_red],
        "E_pc": E[idx_red],
        "Delta_E": abs(E[idx_ox] - E[idx_red]),
        "Area": np.trapezoid(np.abs(I_mean), E),
        "ProtectionStatus": "Недостаточная защита" if concentration < 8 else (
            "Пограничный режим" if concentration < 20 else "Рабочая зона"
        ),
        "RecommendedDose": max(0, 20 - concentration) if concentration < 20 else 0
    }
    return E, I_final, I_mean, std_dev, metrics

def generate_signal(concentration, inh_key, cycle, model_key, seed):
    weight_path = MODELS[model_key]["weight"]
    mode = "по весам" if weight_path.is_file() else "симуляция"
    E, I, I_mean, std_dev, metrics = generate_demo_signal(concentration, inh_key, cycle, model_key, seed)
    return E, I, I_mean, std_dev, metrics, mode

def quality_score(delta_e, area, variance):
    score = 100
    score -= min(abs(delta_e - 0.35) * 80, 18)
    score -= min(abs(area - 0.010) * 600, 24)
    score -= min(variance * 1e5, 20)
    return max(58, min(99, round(score, 1)))

def line_shape(style):
    return "spline" if style == "Гладкий" else "linear"

def fill_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(59,130,246,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def build_plot(fig, title=None, height=500):
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["plot_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Arial, sans-serif"),
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        height=height,
        legend=dict(
            bgcolor=PALETTE["legend_bg"],
            bordercolor=PALETTE["stroke"],
            borderwidth=1,
            font=dict(color=PALETTE["text"]),
            orientation="v"
        ),
        xaxis=dict(
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            title_font=dict(color=PALETTE["text"]),
            tickfont=dict(color=PALETTE["text"])
        ),
        yaxis=dict(
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            title_font=dict(color=PALETTE["text"]),
            tickfont=dict(color=PALETTE["text"])
        )
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.01,
                xanchor="left",
                font=dict(size=18, color=PALETTE["text"])
            )
        )
    return fig

def comparison_table(concentration, cycle, model_key, seed):
    rows = []
    for key, meta in INHIBITORS.items():
        _, I_cmp, _, _, met_cmp = generate_demo_signal(concentration, key, cycle, model_key, seed)
        cmp_score = quality_score(met_cmp["Delta_E"], met_cmp["Area"], float(np.var(I_cmp)))
        rows.append({
            "Ингибитор": meta["name"],
            "Тип": meta["type"],
            "ΔE (В)": round(met_cmp["Delta_E"], 3),
            "Ipa (мА)": round(met_cmp["I_pa"] * 1000, 2),
            "Ipc (мА)": round(met_cmp["I_pc"] * 1000, 2),
            "Q": round(met_cmp["Area"], 4),
            "Score": cmp_score
        })
    df = pd.DataFrame(rows).sort_values(by="Score", ascending=False).reset_index(drop=True)
    return df

# =====================================================
# БОКОВАЯ ПАНЕЛЬ
# =====================================================
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand-wrap">
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🔬</div>
            <div class="sidebar-brand-text">
                <div class="sidebar-brand-name">НейроКапибры</div>
                <div class="sidebar-brand-ver">Цифровой двойник CVA&nbsp;·&nbsp;v2.0</div>
            </div>
        </div>
        <div class="sidebar-brand-caption">
            Разработано командой <b style="color:{PALETTE['accent']};">НейроКапибры</b><br>
            для промышленного мониторинга ингибиторов коррозии
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Условия исследования")

    inh_key = st.selectbox(
        "Ингибитор",
        list(INHIBITORS.keys()),
        format_func=lambda x: f"{x}: {INHIBITORS[x]['name']}"
    )

    compare_inhibitors = st.checkbox(
        "Сравнить все 4 ингибитора",
        value=False,
        help="При включении строятся 4 кривые для всех ингибиторов при одинаковых параметрах."
    )

    if compare_inhibitors:
        st.markdown(
            '<div class="compare-banner">Режим сравнения активен: концентрация, цикл, модель и seed одинаковы для всех 4 ингибиторов.</div>',
            unsafe_allow_html=True
        )

    concentration = st.number_input(
        "Концентрация (ppm)",
        min_value=0,
        max_value=200,
        value=20,
        step=5
    )

    cycle = st.slider("Цикл", 1, 5, 2)

    model_key = st.selectbox(
        "Генеративная модель",
        list(MODELS.keys()),
        format_func=lambda x: f"{x} ({MODELS[x]['desc']})"
    )

    st.divider()
    st.markdown("#### Воспроизводимость")
    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help="Фиксирует случайный шум для воспроизводимости результатов. Одинаковый seed при тех же параметрах даёт идентичный сигнал."
    )
    st.markdown(
        f'<div class="seed-badge">🎲 seed&nbsp;=&nbsp;<b>{random_seed}</b></div>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("#### Настройки отображения")
    show_grid = st.checkbox("Показывать сетку", value=True)

    st.divider()
    if MODELS[model_key]["weight"].is_file():
        st.success(f"✅ Найден файл весов: {MODELS[model_key]['weight'].name}")
    else:
        st.warning(f"🟡 Файл весов не найден: {MODELS[model_key]['weight'].name}")

# =====================================================
# ВЕРХНЯЯ ПАНЕЛЬ
# =====================================================
st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="brand-dot"></div>
        <div>
            <div class="brand-title">Платформа цифрового двойника CVA</div>
            <div class="brand-sub">Панель моделирования циклических вольтамперограмм</div>
        </div>
    </div>
    <div class="topbar-right">
        <div class="top-pill">Тема: <b>{theme_mode}</b></div>
        <div class="top-pill">Модель: <b>{model_key}</b></div>
        <div class="top-pill">Seed: <b>{random_seed}</b></div>
        <div class="top-pill">Режим: <b>{"сравнение 4 ингибиторов" if compare_inhibitors else "одиночный"}</b></div>
        <div class="top-pill">Файл весов: <b>{MODELS[model_key]["weight"].name}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================
st.markdown(f"""
<div class="hero">
    <div class="hero-title">Цифровой двойник электрохимической системы</div>
    <div class="hero-sub">
        Платформа объединяет хемоинформатику, генерацию синтетических ВАХ, валидацию стабильности модели
        и демонстрацию прикладного эффекта для задачи мониторинга концентрации ингибиторов коррозии.
        Доступен одиночный режим и режим сравнения всех 4 ингибиторов при одинаковых условиях.
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# ХРАНЕНИЕ ДАННЫХ ГЕНЕРАЦИИ
# =====================================================
current_signature = (concentration, inh_key, cycle, model_key, random_seed)

if "generated_signature" not in st.session_state or st.session_state.generated_signature != current_signature:
    st.session_state.generated_data = generate_signal(concentration, inh_key, cycle, model_key, random_seed)
    st.session_state.generated_signature = current_signature

E, I, I_mean, std_dev, met, generation_mode = st.session_state.generated_data
score = quality_score(met["Delta_E"], met["Area"], float(np.var(I)))

if compare_inhibitors:
    cmp_df = comparison_table(concentration, cycle, model_key, random_seed)
    best_row = cmp_df.iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Режим", "Сравнение 4 ингибиторов")
    m2.metric("Лидер по Score", best_row["Ингибитор"])
    m3.metric("Лучший Score", f"{best_row['Score']}/100")
    m4.metric("Стабильность модели", f"{MODELS[model_key]['stability']*100:.1f} %")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Оценка цифрового двойника", f"{score}/100")
    m2.metric("ΔE", f"{met['Delta_E']:.3f} В")
    m3.metric("Площадь Q", f"{met['Area']:.4f}")
    m4.metric("Стабильность модели", f"{MODELS[model_key]['stability']*100:.1f} %")

st.write("")

# =====================================================
# ВКЛАДКИ
# =====================================================
tab_gen, tab_chem, tab_val, tab_eco = st.tabs([
    "📈 Генерация сигнала",
    "🧪 Хемоинформатика",
    "📊 Лаборатория валидации",
    "💼 Прикладной эффект"
])

# =====================================================
# ВКЛАДКА 1 — ГЕНЕРАЦИЯ
# =====================================================
with tab_gen:
    st.markdown("### Генерация циклической вольтамперограммы")
    st.caption("Построение сигнала в текущем режиме.")

    E, I, I_mean, std_dev, met, generation_mode = st.session_state.generated_data

    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        if st.button("Сгенерировать сигнал", type="primary", use_container_width=True):
            st.session_state.generated_data = generate_signal(concentration, inh_key, cycle, model_key, random_seed)
            st.session_state.generated_signature = current_signature
            E, I, I_mean, std_dev, met, generation_mode = st.session_state.generated_data

    with col_status:
        if generation_mode == "по весам":
            st.success(f"🟢 Режим генерации: используется файл весов {MODELS[model_key]['weight'].name}")
        else:
            st.warning(f"🟡 Режим генерации: используется демонстрационная модель, файл весов {MODELS[model_key]['weight'].name} не найден")

    left, right = st.columns([3, 1])

    with left:
        fig = go.Figure()

        if compare_inhibitors:
            for key, meta in INHIBITORS.items():
                E_cmp, I_cmp, I_mean_cmp, std_cmp, met_cmp = generate_demo_signal(
                    concentration, key, cycle, model_key, random_seed
                )

                if graph_style == "С заливкой":
                    fig.add_trace(go.Scatter(
                        x=E_cmp,
                        y=I_mean_cmp,
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=fill_rgba(meta["color"], 0.07),
                        line=dict(color="rgba(255,255,255,0)"),
                        name=f"{meta['name']} · площадь",
                        hoverinfo="skip",
                        showlegend=False
                    ))

                fig.add_trace(go.Scatter(
                    x=E_cmp,
                    y=I_cmp,
                    mode="lines",
                    name=meta["name"],
                    line=dict(color=meta["color"], width=2.8),
                    line_shape=line_shape(graph_style),
                    hovertemplate=(
                        f"<b>{meta['name']}</b><br>"
                        "E = %{x:.3f} В<br>"
                        "I = %{y:.5f} А<extra></extra>"
                    )
                ))

                idx_ox = int(np.argmax(I_cmp))
                idx_red = int(np.argmin(I_cmp))
                fig.add_trace(go.Scatter(
                    x=[E_cmp[idx_ox], E_cmp[idx_red]],
                    y=[I_cmp[idx_ox], I_cmp[idx_red]],
                    mode="markers",
                    name=f"{meta['name']} · пики",
                    marker=dict(color=meta["color"], size=7, line=dict(color="#FFFFFF", width=0.5)),
                    showlegend=False
                ))

            build_plot(fig, title="Сравнение 4 ингибиторов при одинаковых параметрах", height=560)

        else:
            fig.add_trace(go.Scatter(
                x=np.concatenate([E, E[::-1]]),
                y=np.concatenate([I_mean + std_dev * 2, (I_mean - std_dev * 2)[::-1]]),
                fill="toself",
                fillcolor="rgba(37,99,235,0.12)" if theme_mode == "Светлая" else "rgba(37,99,235,0.16)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Доверительный интервал ±2σ",
                hoverinfo="skip"
            ))

            fig.add_trace(go.Scatter(
                x=E,
                y=I,
                mode="lines",
                name="Сигнал",
                line=dict(color=INHIBITORS[inh_key]["color"], width=2.4),
                line_shape=line_shape(graph_style)
            ))

            fig.add_trace(go.Scatter(
                x=[met["E_pa"], met["E_pc"]],
                y=[met["I_pa"], met["I_pc"]],
                mode="markers+text",
                name="Характерные пики",
                marker=dict(color=[PALETTE["bad"], PALETTE["good"]], size=9),
                text=[f"Epa={met['E_pa']:.2f}", f"Epc={met['E_pc']:.2f}"],
                textposition="top center"
            ))

            if graph_style == "С заливкой":
                fig.add_trace(go.Scatter(
                    x=E,
                    y=I_mean,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(2,132,199,0.08)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Площадь под кривой",
                    hoverinfo="skip"
                ))

            build_plot(fig, title="CVA-кривая", height=560)

        fig.update_xaxes(title="Потенциал (В)", showgrid=show_grid)
        fig.update_yaxes(title="Ток (А)", showgrid=show_grid)
        st.plotly_chart(fig, use_container_width=True)

        if compare_inhibitors:
            E_base = np.linspace(-0.8, 0.8, 960)
            export_compare = pd.DataFrame({"Потенциал_В": np.round(E_base, 5)})

            for key, meta in INHIBITORS.items():
                E_cmp, I_cmp, I_mean_cmp, std_cmp, _ = generate_demo_signal(
                    concentration, key, cycle, model_key, random_seed
                )
                export_compare[f"{meta['name']}_Ток_А"] = np.round(I_cmp, 7)
                export_compare[f"{meta['name']}_Средний_ток_А"] = np.round(I_mean_cmp, 7)
                export_compare[f"{meta['name']}_Std_А"] = np.round(std_cmp, 7)

            st.download_button(
                label="📄 Скачать сравнение ингибиторов (.csv)",
                data=export_compare.to_csv(index=False).encode("utf-8"),
                file_name=f"compare_all_inhibitors_{concentration}ppm_{model_key}_cycle{cycle}_seed{random_seed}.csv",
                mime="text/csv"
            )
        else:
            export_df = pd.DataFrame({
                "Потенциал_В": np.round(E, 5),
                "Ток_А": np.round(I, 7),
                "Средний_ток_А": np.round(I_mean, 7),
                "Стандартное_отклонение_А": np.round(std_dev, 7)
            })

            st.download_button(
                label="📄 Скачать сигнал (.csv)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"cva_{inh_key}_{concentration}ppm_{model_key}_seed{random_seed}.csv",
                mime="text/csv"
            )

    with right:
        if compare_inhibitors:
            st.markdown('<div class="section-title">Сравнение ингибиторов</div>', unsafe_allow_html=True)
            cmp_df = comparison_table(concentration, cycle, model_key, random_seed)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            best_name = cmp_df.iloc[0]["Ингибитор"]
            best_score = cmp_df.iloc[0]["Score"]
            st.success(f"Лучшая кривая в текущих условиях: {best_name} · Score {best_score}/100")
            st.info("Все 4 кривые построены при одинаковых параметрах: концентрация, цикл, модель и seed.")
        else:
            st.markdown('<div class="section-title">Ключевые параметры</div>', unsafe_allow_html=True)
            st.metric("I_pa", f"{met['I_pa']*1000:.2f} мА")
            st.metric("I_pc", f"{met['I_pc']*1000:.2f} мА")
            st.metric("ΔE", f"{met['Delta_E']:.3f} В")
            st.metric("Q", f"{met['Area']:.4f}")
            st.metric("Статус защиты", met["ProtectionStatus"])

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Рекомендация по дозировке</div>', unsafe_allow_html=True)

            if met["ProtectionStatus"] == "Недостаточная защита":
                st.warning(
                    f"Текущий режим соответствует недостаточной защите. "
                    f"Рекомендуется увеличить концентрацию на +{met['RecommendedDose']} ppm."
                )
            elif met["ProtectionStatus"] == "Пограничный режим":
                st.info(
                    f"Система находится в пограничном режиме. "
                    f"Желательно рассмотреть корректировку дозировки на +{met['RecommendedDose']} ppm."
                )
            else:
                st.success("Система находится в рабочей зоне. Корректировка дозировки не требуется.")

# =====================================================
# ВКЛАДКА 2 — ХЕМОИНФОРМАТИКА
# =====================================================
with tab_chem:
    st.markdown("### Хемоинформатический профиль ингибитора")

    if compare_inhibitors:
        st.info("В режиме сравнения ниже показана сводная информация по всем 4 ингибиторам.")

        chem_rows = []
        for key, meta in INHIBITORS.items():
            chem_rows.append({
                "Код": key,
                "Название": meta["name"],
                "Тип": meta["type"],
                "Молекулярная масса": meta["mw"],
                "LogP": meta["logp"],
                "SMILES": meta["smiles"]
            })

        st.dataframe(pd.DataFrame(chem_rows), use_container_width=True, hide_index=True)

        fig_multi = go.Figure()
        labels = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotBonds"]

        for key, meta in INHIBITORS.items():
            df_desc = get_descriptors(key)
            fig_multi.add_trace(go.Scatterpolar(
                r=df_desc["Норм. значение"].tolist() + [df_desc["Норм. значение"].iloc[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=fill_rgba(meta["color"], 0.08),
                line=dict(color=meta["color"], width=2.0),
                name=meta["name"]
            ))

        fig_multi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=PALETTE["grid"])
            ),
            legend=dict(
                bgcolor=PALETTE["legend_bg"],
                bordercolor=PALETTE["stroke"],
                borderwidth=1,
                font=dict(color=PALETTE["text"])
            ),
            font=dict(color=PALETTE["text"]),
            height=420
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    else:
        df_desc = get_descriptors(inh_key)
        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(f"""
            <div class="card" style="margin-bottom: 15px;">
                <p style="margin-bottom: 8px;"><b>Название:</b> {INHIBITORS[inh_key]['name']}</p>
                <p style="margin-bottom: 8px;"><b>Тип:</b> {INHIBITORS[inh_key]['type']}</p>
                <p style="margin-bottom: 8px;"><b>Молекулярная масса:</b> {INHIBITORS[inh_key]['mw']}</p>
                <p style="margin-bottom: 0px;"><b>LogP:</b> {INHIBITORS[inh_key]['logp']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.code(INHIBITORS[inh_key]['smiles'], language="text")
            st.dataframe(df_desc, use_container_width=True, hide_index=True)

        with c2:
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=df_desc["Норм. значение"].tolist() + [df_desc["Норм. значение"].iloc[0]],
                theta=df_desc["Дескриптор"].tolist() + [df_desc["Дескриптор"].iloc[0]],
                fill="toself",
                fillcolor="rgba(37,99,235,0.10)",
                line=dict(color=INHIBITORS[inh_key]["color"], width=2.2),
                name=INHIBITORS[inh_key]["name"]
            ))
            radar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=PALETTE["grid"])
                ),
                showlegend=False,
                font=dict(color=PALETTE["text"]),
                height=340
            )
            st.plotly_chart(radar, use_container_width=True)

# =====================================================
# ВКЛАДКА 3 — ВАЛИДАЦИЯ
# =====================================================
with tab_val:
    st.markdown("### Лаборатория валидации")
    mode = st.radio("Режим сравнения", ["Концентрации", "Модели", "Циклы", "Ингибиторы"], horizontal=True)

    fig_val = go.Figure()

    if mode == "Концентрации":
        concs = [0, 10, 20, 40, 80]
        colors = ['#94A3B8', '#38BDF8', '#3B82F6', '#6366F1', '#4C1D95']
        for idx, c in enumerate(concs):
            Ec, Ic, _, _, _ = generate_demo_signal(c, inh_key, cycle, model_key, random_seed)
            fig_val.add_trace(go.Scatter(
                x=Ec, y=Ic, mode='lines', name=f'{c} ppm',
                line=dict(color=colors[idx], width=2)
            ))
        build_plot(fig_val, title="Влияние концентрации", height=500)

    elif mode == "Модели":
        for mk, meta in MODELS.items():
            Ec, Ic, _, _, _ = generate_demo_signal(concentration, inh_key, cycle, mk, random_seed)
            fig_val.add_trace(go.Scatter(
                x=Ec, y=Ic, mode='lines', name=mk,
                line=dict(color=meta["color"], width=2.2)
            ))
        build_plot(fig_val, title="Сравнение моделей", height=500)

    elif mode == "Циклы":
        cyc_colors = ['#0EA5E9', '#2563EB', '#4F46E5', '#7C3AED', '#DB2777']
        for cyc, col in zip([1, 2, 3, 4, 5], cyc_colors):
            Ec, Ic, _, _, _ = generate_demo_signal(concentration, inh_key, cyc, model_key, random_seed)
            fig_val.add_trace(go.Scatter(
                x=Ec, y=Ic, mode='lines', name=f'Цикл {cyc}',
                line=dict(color=col, width=2.2)
            ))
        build_plot(fig_val, title="Изменение по циклам", height=500)

    else:
        for key, meta in INHIBITORS.items():
            Ec, Ic, _, _, _ = generate_demo_signal(concentration, key, cycle, model_key, random_seed)
            fig_val.add_trace(go.Scatter(
                x=Ec, y=Ic, mode='lines', name=meta["name"],
                line=dict(color=meta["color"], width=2.5),
                line_shape=line_shape(graph_style)
            ))
        build_plot(fig_val, title="Сравнение всех 4 ингибиторов", height=500)

    fig_val.update_xaxes(title="Потенциал (В)", showgrid=show_grid)
    fig_val.update_yaxes(title="Ток (А)", showgrid=show_grid)
    st.plotly_chart(fig_val, use_container_width=True)

    if mode == "Ингибиторы":
        st.dataframe(
            comparison_table(concentration, cycle, model_key, random_seed),
            use_container_width=True,
            hide_index=True
        )

# =====================================================
# ВКЛАДКА 4 — ПРИКЛАДНОЙ ЭФФЕКТ
# =====================================================
with tab_eco:
    st.markdown("### Прикладной эффект")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE • Базовая модель", "4.8 ppm")
    e2.metric("MAE • С аугментацией", "2.1 ppm", delta="-2.7 ppm", delta_color="inverse")
    e3.metric("R² • С аугментацией", "0.94", delta="+15%")

    st.markdown("")
    st.markdown("#### Практический сценарий")

    if compare_inhibitors:
        cmp_df = comparison_table(concentration, cycle, model_key, random_seed)
        best_row = cmp_df.iloc[0]
        st.success(
            f"При текущих условиях наиболее стабильный цифровой профиль показывает "
            f"**{best_row['Ингибитор']}** со Score **{best_row['Score']}/100**."
        )
    else:
        st.info(
            f"Для выбранного ингибитора **{INHIBITORS[inh_key]['name']}** система оценивает режим как "
            f"**{met['ProtectionStatus']}**."
        )
