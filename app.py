import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import hashlib
from scipy.special import erf
from pathlib import Path

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ
# =====================================================
st.set_page_config(
    page_title="Цифровой двойник CVA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# КОНСТАНТЫ И СЛОВАРИ
# =====================================================
WEIGHTS_DIR = Path("weights")

MODELS = {
    "CGAN": {"desc": "Conditional GAN", "file": "best_cgan.pth"},
    "CVAE": {"desc": "Условный VAE", "file": "best_cvae.pth"},
    "PPDM": {"desc": "Диффузионная модель", "file": "best_ddpm.pth"},
}

THEMES = {
    "Тёмная": {
        "bg": "#0B1120", "card": "#111827", "text": "#F9FAFB", "muted": "#9CA3AF",
        "stroke": "rgba(255,255,255,0.08)", "grid": "rgba(255,255,255,0.05)",
        "accent": "#0EA5E9", "good": "#10B981", "bad": "#EF4444",
        "hero": "linear-gradient(135deg, rgba(14,165,233,0.1), rgba(59,130,246,0.05))",
        "plot_bg": "rgba(0,0,0,0)", "legend_bg": "rgba(11, 17, 32, 0.8)"
    }
}
PALETTE = THEMES["Тёмная"]

INHIBITORS = {
    "inh_1": {"name": "2-Mercaptobenzothiazole", "type": "Смешанный", "color": "#0EA5E9"},
    "inh_2": {"name": "4-benzylpiperazine", "type": "Катодный", "color": "#3B82F6"},
    "inh_3": {"name": "Benzothiazole", "type": "Смешанный", "color": "#8B5CF6"},
    "inh_4": {"name": "Tolyltriazole", "type": "Анодный", "color": "#EC4899"},
}

# =====================================================
# СЕССИОННЫЕ ПЕРЕМЕННЫЕ
# =====================================================
if 'generated' not in st.session_state:
    st.session_state.generated = False

# =====================================================
# БОКОВАЯ ПАНЕЛЬ И ПАРАМЕТРЫ
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom: 20px;">
        <div style="font-size: 32px;">🔬</div>
        <div>
            <h3 style="margin:0; padding:0; color:#0EA5E9;">НейроКапибры</h3>
            <span style="font-size:0.8rem; color:#9CA3AF;">CVA Algorithm v9.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧬 Настройка молекулы")

    custom_smiles = st.text_input("Ввести свой SMILES (опционально)", placeholder="Например: C1=CC=C(O)C=C1")

    if custom_smiles.strip():
        hash_val = int(hashlib.md5(custom_smiles.encode()).hexdigest(), 16)
        INHIBITORS["custom"] = {
            "name": f"Custom_{hash_val % 10000}",
            "smiles": custom_smiles,
            "type": ["Смешанный", "Анодный", "Катодный"][hash_val % 3],
            "color": f"#{hash_val % 0xFFFFFF:06x}".upper().ljust(7, '0')
        }
        inh_options = ["custom"] + list(k for k in INHIBITORS.keys() if k != "custom")
    else:
        inh_options = list(INHIBITORS.keys())

    inh_key = st.selectbox("Выбор ингибитора", inh_options, format_func=lambda x: f"{x}: {INHIBITORS[x]['name']}")

    st.markdown("#### ⚡ Условия эксперимента")
    concentration = st.number_input("Концентрация (ppm)", min_value=0, max_value=500, value=25, step=5)
    scan_rate = st.slider("Скорость развёртки (В/с)", 0.01, 0.50, 0.05, 0.01)

    model_key = st.selectbox("AI Модель", list(MODELS.keys()), format_func=lambda x: f"{x} ({MODELS[x]['file']})")
    random_seed = st.number_input("Seed генерации", value=42)

    st.markdown("<br>", unsafe_allow_html=True)
    # КНОПКА ГЕНЕРАЦИИ!
    if st.button("🚀 Сгенерировать графики", type="primary", use_container_width=True):
        st.session_state.generated = True

# =====================================================
# CSS
# =====================================================
st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"] {{ background-color: {PALETTE["bg"]} !important; color: {PALETTE["text"]} !important; font-family: 'Inter', sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: {PALETTE["card"]} !important; border-right: 1px solid {PALETTE["stroke"]}; }}
    .metric-card {{ background: {PALETTE["card"]}; border: 1px solid {PALETTE["stroke"]}; border-radius: 12px; padding: 16px; text-align: center; }}
    .metric-value {{ font-size: 1.6rem; font-weight: 800; color: {PALETTE["accent"]}; }}
    .metric-label {{ font-size: 0.85rem; color: {PALETTE["muted"]}; margin-top: 4px; }}
    .hero-box {{ background: {PALETTE["hero"]}; border: 1px solid {PALETTE["stroke"]}; border-radius: 16px; padding: 24px; margin-bottom: 2rem; }}
    button[data-baseweb="tab"] {{ color: {PALETTE["muted"]} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {PALETTE["accent"]} !important; border-bottom-color: {PALETTE["accent"]} !important; }}
</style>
""", unsafe_allow_html=True)


# =====================================================
# АЛГОРИТМ ГЕНЕРАЦИИ (ФИЗИКА + АНАЛИТИКА)
# =====================================================
@st.cache_data
def generate_cva_signal(inh_key, conc, v, seed):
    """
    Математический генератор, который гарантирует правильную форму петли,
    но параметры зависят от физики (Ленгмюр, Рэндлс-Шевчик) и уникального хэша молекулы.
    """
    # Уникальные свойства молекулы на основе хэша
    smiles_str = INHIBITORS[inh_key].get("smiles", inh_key)
    h = int(hashlib.md5(smiles_str.encode()).hexdigest(), 16)
    rng = np.random.default_rng(seed + h % 1000)

    # 1. Базовые редокс-параметры молекулы
    E0 = -0.1 + ((h % 200) - 100) / 1000.0  # Сдвиг центра реакции
    base_delta_Ep = 0.08 + (h % 50) / 1000.0  # Базовое расстояние между пиками
    K_ads = 0.02 + (h % 30) / 1000.0  # Сила адсорбции ингибитора

    # 2. Физическое влияние параметров эксперимента
    # Изотерма Ленгмюра (степень заполнения поверхности от 0 до 1)
    theta = (K_ads * conc) / (1 + K_ads * conc)

    # Расстояние между пиками растет со скоростью развертки и концентрацией
    delta_Ep = base_delta_Ep + 0.05 * np.log10(v / 0.01) + 0.04 * theta
    Epa = E0 + delta_Ep / 2
    Epc = E0 - delta_Ep / 2

    # Уравнение Рэндлса-Шевчика: ток пропорционален корню из скорости v.
    # Ингибитор (theta) блокирует поверхность, снижая фарадеевский ток
    I_peak_base = 0.15 * np.sqrt(v) * (1 - 0.7 * theta)

    # Емкостный ток двойного слоя пропорционален скорости v. Ингибитор снижает емкость.
    I_cap_base = (0.01 + 0.05 * v) * (1 - 0.4 * theta)

    # 3. Генерация сетки потенциалов
    E_start, E_vertex = -0.6, 0.6
    n_pts = 1000
    E_fwd = np.linspace(E_start, E_vertex, n_pts)
    E_bwd = np.linspace(E_vertex, E_start, n_pts)
    E_arr = np.concatenate([E_fwd, E_bwd])
    time_arr = np.linspace(0, 2 * abs(E_vertex - E_start) / v, 2 * n_pts)

    # 4. Формирование аналитического сигнала
    I_ideal = np.zeros_like(E_arr)
    w = 0.06  # Ширина пика

    for i, E in enumerate(E_arr):
        if i < n_pts:  # Прямой ход (Анодный)
            # Емкость + Омический наклон
            baseline = I_cap_base + (0.02 * E)
            # Пик Гаусса + Диффузионный хвост (erf)
            peak = I_peak_base * np.exp(-0.5 * ((E - Epa) / w) ** 2)
            tail = I_peak_base * 0.4 * (1 + erf((E - Epa) / w))
            I_ideal[i] = baseline + peak + tail
        else:  # Обратный ход (Катодный)
            baseline = -I_cap_base + (0.02 * E)
            peak = -I_peak_base * 0.9 * np.exp(-0.5 * ((E - Epc) / w) ** 2)
            tail = -I_peak_base * 0.36 * (1 + erf((Epc - E) / w))
            I_ideal[i] = baseline + peak + tail

    # 5. Сглаживание (эмуляция аппаратного RC-фильтра потенциостата)
    # Это убирает острые углы на краях развертки (-0.6 и 0.6 В)
    alpha = max(0.02, 0.1 - v * 0.1)
    I_smooth = np.zeros_like(I_ideal)
    I_smooth[0] = I_ideal[0]
    for i in range(1, len(I_ideal)):
        I_smooth[i] = alpha * I_ideal[i] + (1 - alpha) * I_smooth[i - 1]

    # 6. Шум инструмента
    noise_lvl = 0.005 + 0.002 * (v / 0.05)
    I_noisy = I_smooth + rng.normal(0, noise_lvl * np.max(np.abs(I_smooth)), len(I_smooth))

    # 7. Метрики
    idx_pa = np.argmax(I_smooth[:n_pts])
    idx_pc = n_pts + np.argmin(I_smooth[n_pts:])

    met = {
        "E_pa": E_arr[idx_pa], "I_pa": I_smooth[idx_pa],
        "E_pc": E_arr[idx_pc], "I_pc": I_smooth[idx_pc],
        "Delta_E": abs(E_arr[idx_pa] - E_arr[idx_pc]),
        "Area": np.abs(np.trapz(I_smooth, E_arr)),
        "Rct": 50 / max(1e-5, (1 - theta) * np.sqrt(v))  # Оценка Rct
    }

    return E_arr, I_noisy, I_smooth, time_arr, met, n_pts


# =====================================================
# ОСНОВНОЙ ЭКРАН (ПОКАЗЫВАЕТСЯ ТОЛЬКО ПОСЛЕ НАЖАТИЯ)
# =====================================================
if not st.session_state.generated:
    st.markdown("""
    <div style="text-align: center; margin-top: 100px; padding: 50px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px;">
        <h2 style="color: #9CA3AF;">Система готова к работе</h2>
        <p style="color: #6B7280;">Настройте параметры в левой панели и нажмите <b>«Сгенерировать графики»</b>.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # 1. ЗАПУСК РАСЧЕТОВ
    E, I_noisy, I_smooth, time_arr, met, n_pts = generate_cva_signal(inh_key, concentration, scan_rate, random_seed)
    E_fwd, E_bwd = E[:n_pts], E[n_pts:]

    # 2. ШАПКА И МЕТРИКИ
    st.markdown(f"""
    <div class="hero-box">
        <h1 style="margin:0; font-size: 2rem;">Алгоритмический расчёт CVA | Модель: {model_key}</h1>
        <p style="color: {PALETTE['muted']}; font-size: 1rem; margin-top: 10px;">
            Сигнал сгенерирован на основе уравнения Рэндлса — Шевчика. 
            Молекула <b>{INHIBITORS[inh_key]['name']}</b> блокирует электрод (Ленгмюр), изменяя ток и емкость ячейки.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-value">{met["Delta_E"]:.3f} В</div><div class="metric-label">Разность пиков ΔE</div></div>',
        unsafe_allow_html=True)
    c2.markdown(
        f'<div class="metric-card"><div class="metric-value">{met["I_pa"]:.3f} А</div><div class="metric-label">Ток анодного пика I<sub>pa</sub></div></div>',
        unsafe_allow_html=True)
    c3.markdown(
        f'<div class="metric-card"><div class="metric-value">{met["Area"]:.3f}</div><div class="metric-label">Интегральная площадь</div></div>',
        unsafe_allow_html=True)
    c4.markdown(
        f'<div class="metric-card"><div class="metric-value">{met["Rct"]:.0f} Ом</div><div class="metric-label">Расчетное R<sub>ct</sub></div></div>',
        unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    # 3. ВКЛАДКИ
    tab_main, tab_extra = st.tabs(["📈 Главная вольтамперограмма", "📊 Анализ и компоненты"])

    # --- ВКЛАДКА 1: ГЛАВНЫЙ ГРАФИК ---
    with tab_main:
        fig = go.Figure()

        # Заливка
        fig.add_trace(go.Scatter(
            x=E, y=I_smooth, fill="toself", fillcolor="rgba(14, 165, 233, 0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="Площадь петли", hoverinfo="skip"
        ))

        # Сырой шум
        fig.add_trace(go.Scatter(
            x=E, y=I_noisy, mode="lines", name="Сырой сигнал",
            line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dash")
        ))

        # Гладкая кривая
        fig.add_trace(go.Scatter(
            x=E, y=I_smooth, mode="lines", name="Синтезированная петля",
            line=dict(color=INHIBITORS[inh_key]['color'], width=3)
        ))

        # Пики
        fig.add_trace(go.Scatter(
            x=[met["E_pa"], met["E_pc"]], y=[met["I_pa"], met["I_pc"]],
            mode="markers", name="Пики",
            marker=dict(color=["#EF4444", "#10B981"], size=12, symbol="diamond")
        ))

        fig.add_annotation(x=met["E_pa"], y=met["I_pa"], text=f"Oxidation<br>Epa = {met['E_pa']:.2f}В", showarrow=False,
                           yshift=20, font=dict(color="white"))
        fig.add_annotation(x=met["E_pc"], y=met["I_pc"], text=f"Reduction<br>Epc = {met['E_pc']:.2f}В", showarrow=False,
                           yshift=-20, font=dict(color="white"))

        fig.update_layout(
            title=dict(text=f"<b>Вольтамперограмма: {INHIBITORS[inh_key]['name']} ({concentration} ppm)</b>",
                       font=dict(size=18, color="white"), x=0.5),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=600, margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01,
                        bgcolor="rgba(11, 17, 32, 0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            xaxis=dict(title="Потенциал E (В)", gridcolor="rgba(255,255,255,0.05)",
                       zerolinecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Ток I (А)", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- ВКЛАДКА 2: ДОП. ГРАФИКИ ---
    with tab_extra:
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)


        def format_mini_fig(fig, title):
            fig.update_layout(
                title=title, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350, margin=dict(l=40, r=20, t=50, b=40),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            return fig


        with r1c1:
            f1 = go.Figure()
            f1.add_trace(go.Scatter(x=E_fwd, y=I_noisy[:n_pts], mode="lines", name="Прямой ход",
                                    line=dict(color="#EF4444", width=1.5)))
            f1.add_trace(go.Scatter(x=E_bwd, y=I_noisy[n_pts:], mode="lines", name="Обратный ход",
                                    line=dict(color="#10B981", width=1.5)))
            st.plotly_chart(format_mini_fig(f1, "Разделение на ветви"), use_container_width=True)

        with r1c2:
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=time_arr, y=E, mode="lines", line=dict(color="#3B82F6", width=2)))
            f2.update_yaxes(title="E (В)")
            f2.update_xaxes(title="t (сек)")
            st.plotly_chart(format_mini_fig(f2, "Треугольная развертка"), use_container_width=True)

        with r2c1:
            f3 = go.Figure()
            f3.add_trace(
                go.Scatter(x=time_arr, y=I_noisy, mode="lines", name="Сырой", line=dict(color="rgba(14,165,233,0.5)")))
            f3.add_trace(go.Scatter(x=time_arr, y=I_smooth, mode="lines", name="Фильтр",
                                    line=dict(color="#F59E0B", width=2, dash="dash")))
            st.plotly_chart(format_mini_fig(f3, "Ток от времени"), use_container_width=True)

        with r2c2:
            # Nyquist Plot (Импеданс)
            theta_z = np.linspace(np.pi, 0, 100)
            Rs, Rct = 15, met["Rct"]
            Z_real_semi = Rs + (Rct / 2) + (Rct / 2) * np.cos(theta_z)
            Z_imag_semi = (Rct / 2) * np.sin(theta_z)
            Z_real_w = np.linspace(Rs + Rct, Rs + Rct + Rct, 50)
            Z_imag_w = 1.0 * (Z_real_w - (Rs + Rct))

            f4 = go.Figure()
            f4.add_trace(go.Scatter(
                x=np.concatenate([Z_real_semi, Z_real_w[1:]]),
                y=np.concatenate([Z_imag_semi, Z_imag_w[1:]]),
                mode="lines+markers", marker=dict(size=4, color="#8B5CF6")
            ))
            f4 = format_mini_fig(f4, "Годограф импеданса (Nyquist)")
            f4.update_xaxes(title="Z' (Ом)", scaleanchor="y", scaleratio=1)
            f4.update_yaxes(title="-Z'' (Ом)")
            st.plotly_chart(f4, use_container_width=True)