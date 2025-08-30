# poligonal_app.py  — reemplaza por completo tu archivo con este contenido
import math
import pandas as pd
import streamlit as st

# ----------------- Utilidades DMS -----------------
def dms_to_deg(s: str | float | int | None) -> float | None:
    """'ddd°mm\'ss.ss\"' -> grados decimales. Devuelve None si s no tiene forma válida."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s or "°" not in s:
        return None
    try:
        d_part, rest = s.split("°", 1)
        d = float(d_part)
        if "'" in rest:
            m_part, rest2 = rest.split("'", 1)
            m = float(m_part)
        else:
            m = 0.0
            rest2 = rest
        if '"' in rest2:
            s_part = rest2.replace('"', "").strip()
            sec = float(s_part) if s_part else 0.0
        else:
            sec = 0.0
        sign = -1.0 if d < 0 else 1.0
        d = abs(d)
        return sign * (d + m/60.0 + sec/3600.0)
    except Exception:
        return None

def deg_to_dms(deg: float | None, sec_dec: int = 2) -> str:
    """grados decimales -> 'ddd°mm'ss.ss\"' (sec_dec decimales)."""
    if deg is None or pd.isna(deg):
        return ""
    sign = "-" if deg < 0 else ""
    x = abs(deg) % 360.0
    d = int(x)
    rem = (x - d) * 60.0
    m = int(rem)
    s = (rem - m) * 60.0
    fmt = f"{{:.{sec_dec}f}}"
    s_txt = fmt.format(s)
    # redondeo que puede empujar a 60.00
    s_val = float(s_txt)
    if s_val >= 60.0 - 10**(-(sec_dec+1)):
        s_val = 0.0
        m += 1
    if m >= 60:
        m = 0
        d += 1
    return f"{sign}{d}°{m:02d}'{s_val:0{2+1+sec_dec}.{sec_dec}f}\""  # 2 dígitos + '.' + dec

def wrap360(x: float) -> float:
    return x % 360.0

def wrap180(x: float) -> float:
    y = (x + 180.0) % 360.0 - 180.0
    return y

# ----------------- Cálculos de poligonal -----------------
def dist_from_stadia(row, K: float, reduce_by_v: bool):
    """Si Dist H == 0, calcula por pelos sup/inf y opcionalmente reduce por cos(V)."""
    d = float(row.get("Dist H (m)") or 0.0)
    if d and d > 0:
        return d
    sup = row.get("Sup (m)") or 0.0
    inf = row.get("Inf (m)") or 0.0
    v_deg = dms_to_deg(row.get("Vert (DMS)"))
    d_stadia = K * (float(sup) - float(inf))
    if reduce_by_v and v_deg is not None:
        d_stadia *= math.cos(math.radians(v_deg))
    return float(d_stadia)

def build_results(df_in: pd.DataFrame, sec_dec=2, K=100.0, reduce_by_v=False):
    # orden y nombres de columnas tal cual el cuadro
    df = df_in.copy()

    # ---------- Preparación de columnas numéricas ----------
    n = len(df)
    # Distancias horizontales (ingresadas o por pelos)
    d_m = [dist_from_stadia(df.iloc[i], K, reduce_by_v) for i in range(n)]

    # Ángulos a la derecha (grados)
    ang_deg = [dms_to_deg(df.loc[i, "Ángulo (DMS)"]) for i in range(n)]
    # Azimut medido (sólo PQ y RS normalmente)
    az_med_deg = [dms_to_deg(df.loc[i, "Azimut medido (DMS)"]) for i in range(n)]

    # ---------- Azimut calculado (sin compensar) ----------
    az_calc = [None] * n
    # exige que el 1er azimut esté medido (PQ)
    az0 = az_med_deg[0]
    if az0 is None:
        raise ValueError("Debes ingresar el 'Azimut medido (DMS)' de la primera fila (PQ).")
    az_calc[0] = az0
    for i in range(1, n):
        a = ang_deg[i]
        if a is None:
            az_calc[i] = None
        else:
            # Regla del cuadro: Az(i) = Az(i-1) + Ángulo_derecha(i) - 180°
            az_calc[i] = wrap360(az_calc[i-1] + a - 180.0)

    # ---------- Cierre angular y reparto por ángulo ----------
    # error angular total (en segundos): az_med_RS - az_calc_RS
    if az_med_deg[-1] is None or az_calc[-1] is None:
        err_sec = 0.0
    else:
        err_sec = wrap180(az_med_deg[-1] - az_calc[-1]) * 3600.0
    # índices de las filas donde hay ángulo
    idx_ang = [i for i in range(n) if ang_deg[i] is not None]
    m = len(idx_ang)
    comp_sec = [0.0] * n
    if m > 0:
        # reparto con redondeo a centésimas y ajuste en la última para cerrar exacto
        cuota = err_sec / m
        acumul = 0.0
        for k, i in enumerate(idx_ang):
            if k < m - 1:
                c = round(cuota, 2)
                comp_sec[i] = c
                acumul += c
            else:
                comp_sec[i] = round(err_sec - acumul, 2)

    # Ángulos compensados y Azimut compensado
    ang_comp_deg = [None if ang_deg[i] is None else (ang_deg[i] + comp_sec[i] / 3600.0) for i in range(n)]
    az_comp = [None] * n
    az_comp[0] = az_calc[0]
    for i in range(1, n):
        a = ang_comp_deg[i]
        az_comp[i] = None if a is None else wrap360(az_comp[i-1] + a - 180.0)
    # Si el último fue medido, forzamos concordancia por formato
    if az_med_deg[-1] is not None:
        az_comp[-1] = az_med_deg[-1]

    # ---------- Proyecciones (con azimut compensado) ----------
    dN = []
    dE = []
    for i in range(n):
        if az_comp[i] is None or not d_m[i]:
            dN.append(0.0)
            dE.append(0.0)
        else:
            z = math.radians(az_comp[i])
            dN.append(d_m[i] * math.cos(z))
            dE.append(d_m[i] * math.sin(z))

    # ---------- Cierre lineal en RS y reparto por distancia ----------
    # Coordenadas medidas de PQ y RS
    N0 = float(df.loc[0, "Coord. medidas — NORTE (m)"] or 0.0)
    E0 = float(df.loc[0, "Coord. medidas — ESTE (m)"] or 0.0)
    N_RS_meas = float(df.loc[n-1, "Coord. medidas — NORTE (m)"] or 0.0)
    E_RS_meas = float(df.loc[n-1, "Coord. medidas — ESTE (m)"] or 0.0)

    # Predicción RS SIN compensación lineal (acumulando dN/dE desde QA hasta RS)
    sum_dN_no_comp = sum(dN[1:])  # desde la 2ª fila
    sum_dE_no_comp = sum(dE[1:])
    N_RS_pred = N0 + sum_dN_no_comp
    E_RS_pred = E0 + sum_dE_no_comp

    eN = N_RS_meas - N_RS_pred
    eE = E_RS_meas - E_RS_pred

    # reparto por distancia
    perimetro = sum(d_m[1:])  # no cuenta PQ
    compN = []
    compE = []
    for i in range(n):
        if i == 0 or d_m[i] == 0 or perimetro == 0:
            compN.append(0.0)
            compE.append(0.0)
        else:
            w = d_m[i] / perimetro
            compN.append(w * eN)
            compE.append(w * eE)

    # Proyecciones compensadas
    dN_c = [dN[i] + compN[i] for i in range(n)]
    dE_c = [dE[i] + compE[i] for i in range(n)]

    # ---------- Pre coordenadas y coordenadas absolutas ----------
    # Pre coordenadas absolutas (en QA valen a las medidas de PQ)
    preN = [None] * n
    preE = [None] * n
    # para QA (i=1) son las medidas de PQ
    if n > 1:
        preN[1] = N0
        preE[1] = E0
    # desde AB (i=2) acumulando las proyecciones compensadas de la fila anterior
    for i in range(2, n):
        preN[i] = preN[i-1] + dN_c[i-1]
        preE[i] = preE[i-1] + dE_c[i-1]

    # Coordenadas absolutas (acumulación hasta cada punto)
    absN = [None] * n
    absE = [None] * n
    # QA: base
    if n > 1:
        absN[1] = N0 + dN_c[1]
        absE[1] = E0 + dE_c[1]
    for i in range(2, n):
        absN[i] = absN[i-1] + dN_c[i]
        absE[i] = absE[i-1] + dE_c[i]
    # aseguramos que en RS salgan exactamente las medidas (cierre)
    if n > 0 and N_RS_meas and E_RS_meas:
        absN[-1] = N_RS_meas
        absE[-1] = E_RS_meas

    # ---------- Salida en el mismo orden del cuadro ----------
    # Formateo DMS
    def fdms(v): return deg_to_dms(v, sec_dec)
    def f3(x):   return "" if x is None else f"{x:.3f}"

    out = pd.DataFrame({
        "EST.- PV.": df["EST.- PV."],
        "Ángulo a la derecha (DMS)": [df.loc[i, "Ángulo (DMS)"] for i in range(n)],
        "Ángulo comp. (DMS)": [deg_to_dms(ang_comp_deg[i], sec_dec) if ang_comp_deg[i] is not None else "" for i in range(n)],
        "Azimut calculado (DMS)": [fdms(az_calc[i]) for i in range(n)],
        "Azimut medido (DMS)": [df.loc[i, "Azimut medido (DMS)"] if pd.notna(df.loc[i, "Azimut medido (DMS)"]) else "" for i in range(n)],
        "Comp. (seg)": [f"{comp_sec[i]:.2f}" if ang_deg[i] is not None else "" for i in range(n)],
        "Ángulo a la derecha compensado (DMS)": [deg_to_dms(ang_comp_deg[i], sec_dec) if ang_comp_deg[i] is not None else "" for i in range(n)],
        "Azimut compensado (DMS)": [fdms(az_comp[i]) for i in range(n)],
        "Distancia H (m)": [f3(d_m[i]) for i in range(n)],
        "ΔN (m)": [f3(dN[i]) for i in range(n)],
        "ΔE (m)": [f3(dE[i]) for i in range(n)],
        "Pre N (m)": [f3(preN[i]) for i in range(n)],
        "Pre E (m)": [f3(preE[i]) for i in range(n)],
        "Cx (m)": [f3(compN[i]) for i in range(n)],
        "Cy (m)": [f3(compE[i]) for i in range(n)],
        "ΔN comp. (m)": [f3(dN_c[i]) for i in range(n)],
        "ΔE comp. (m)": [f3(dE_c[i]) for i in range(n)],
        "N abs. (m)": [f3(absN[i]) for i in range(n)],
        "E abs. (m)": [f3(absE[i]) for i in range(n)],
    })
    # totales que verás debajo
    tot = {
        "Σ ΔN (m)": sum(dN),
        "Σ ΔE (m)": sum(dE),
        "Perímetro (m)": perimetro,
        "Comp. por ángulo (seg)": err_sec / m if m else 0.0,
        "e_N (m)": eN,  # cierre distribuido en N
        "e_E (m)": eE,  # cierre distribuido en E
    }
    return out, tot

# ----------------- UI -----------------
st.set_page_config(
    page_title="Te odio Alejandro",  # ← título de la pestaña
    page_icon=None,                     # ← sin emoji/ícono (usa el default de Streamlit)
    layout="wide",
)

st.sidebar.header("Parámetros")
sec_dec = st.sidebar.number_input("Decimales en segundos (formato DMS)", 0, 3, 2, 1)
Kconst = st.sidebar.number_input("Constante K (pelos: K·(sup−inf))", 0.0, 1000.0, 100.0, 0.1)
reduce_v = st.sidebar.checkbox("Reducir por ángulo vertical (DMS → cos(V))", value=False)

cols = [
    "EST.- PV.",
    "Ángulo (DMS)",
    "Azimut medido (DMS)",
    "Dist H (m)",
    "Sup (m)",
    "Inf (m)",
    "Vert (DMS)",
    "Coord. medidas — NORTE (m)",
    "Coord. medidas — ESTE (m)"
]

ejemplo = [
    ["PQ", "",        "93°40'36.69\"", 0.0,   "", "", "", 8523648.917, 597951.333],
    ["QA", "222°53'37\"", "",            39.992, "", "", "", "", ""],
    ["AB", "125°49'02\"", "",           507.894, "", "", "", "", ""],
    ["BC", "242°53'24\"", "",          1487.535, "", "", "", "", ""],
    ["CD", "191°31'39\"", "",           548.826, "", "", "", "", ""],
    ["DE", "189°20'26\"", "",           405.318, "", "", "", "", ""],
    ["ER", "173°31'13\"", "",           252.490, "", "", "", "", ""],
    ["RS", "118°58'42\"", "98°38'47.96\"", 0.0, "", "", "", 8521329.633, 599730.308],
]

st.title("Poligonal (abierta/ligada) — cuadro idéntico al PDF(Te odio Alejandro)")

st.write("**Complete la entrada exactamente como el cuadro del PDF (orden PQ, QA, AB, BC, CD, DE, ER, RS).**")

df_in = pd.DataFrame(ejemplo, columns=cols)
df_in = st.data_editor(
    df_in,
    use_container_width=True,
    num_rows="dynamic",  # ← ahora puedes agregar y quitar filas con el botón “+”
    column_config={
        "EST.- PV.": st.column_config.TextColumn(width="small"),
        "Ángulo (DMS)": st.column_config.TextColumn(help="Ángulo a la derecha (DMS)"),
        "Azimut medido (DMS)": st.column_config.TextColumn(help="Sólo PQ y RS normalmente"),
        "Dist H (m)": st.column_config.NumberColumn(format="%.3f"),
        "Sup (m)": st.column_config.NumberColumn(format="%.3f"),
        "Inf (m)": st.column_config.NumberColumn(format="%.3f"),
        "Vert (DMS)": st.column_config.TextColumn(),
        "Coord. medidas — NORTE (m)": st.column_config.NumberColumn(format="%.3f"),
        "Coord. medidas — ESTE (m)": st.column_config.NumberColumn(format="%.3f"),
    },
    key="entrada",
)


try:
    tabla, tot = build_results(df_in, sec_dec=sec_dec, K=Kconst, reduce_by_v=reduce_v)
    st.subheader("Resultados")
    st.dataframe(tabla, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Σ ΔN (m)", f"{tot['Σ ΔN (m)']:.3f}")
    c2.metric("Σ ΔE (m)", f"{tot['Σ ΔE (m)']:.3f}")
    c3.metric("Perímetro (m)", f"{tot['Perímetro (m)']:.3f}")
    c4.metric("Comp. por ángulo (seg)", f"{tot['Comp. por ángulo (seg)']:.2f}\"")
    c5.metric("e_N / e_E (m)", f"{tot['e_N (m)']:.3f} / {tot['e_E (m)']:.3f}")

except Exception as e:
    st.error(str(e))
