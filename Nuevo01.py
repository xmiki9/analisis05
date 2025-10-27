# -*- coding: utf-8 -*-
# ==========================================================
# REPORTE A4 "BONITO": PÓRTICO 3D paramétrico (OpenSeesPy)
# CARGAS: peso propio (vigas/columnas) + losa + acabados + tabiquería + viva (E.020)
# - Plantas por nivel con columnas rectangulares
# - Vista 3D con prismas (vigas/columnas) y flechas de carga
# - Páginas por elemento: Vigas (Vz, My), Columnas (My, Mz)
# - Diagramas con relleno de líneas VERTICALES
# ==========================================================
# pip install openseespy numpy matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
import textwrap
import openseespy.opensees as ops

# Constantes de maquetación A4 con márgenes específicos
CM_TO_INCH = 1.0 / 2.54
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69

MARGIN_LEFT_CM = 1.0
MARGIN_TOP_CM = 1.0
MARGIN_RIGHT_CM = 1.0
MARGIN_BOTTOM_CM = 1.0

PAGE_LEFT = (MARGIN_LEFT_CM * CM_TO_INCH) / A4_WIDTH_IN
PAGE_RIGHT = 1.0 - (MARGIN_RIGHT_CM * CM_TO_INCH) / A4_WIDTH_IN
PAGE_TOP = 1.0 - (MARGIN_TOP_CM * CM_TO_INCH) / A4_HEIGHT_IN
PAGE_BOTTOM = (MARGIN_BOTTOM_CM * CM_TO_INCH) / A4_HEIGHT_IN

# Posiciones auxiliares para encabezados y pies siempre dentro del marco util
HEADER_OFFSET = 0.018
SUBHEADER_GAP = 0.050
FOOTER_OFFSET = 0.018
PAGE_NUMBER_OFFSET = 0.085

FIGURE_CAPTION_GAP = 0.16
TABLE_CAPTION_GAP = 0.006

PAGE_HEADER_Y = PAGE_TOP - HEADER_OFFSET
PAGE_SUBHEADER_Y = max(PAGE_BOTTOM + FOOTER_OFFSET, PAGE_HEADER_Y - SUBHEADER_GAP)
PAGE_FOOTER_Y = PAGE_BOTTOM + FOOTER_OFFSET


def apply_page_margins(fig):
    fig.subplots_adjust(left=PAGE_LEFT, right=PAGE_RIGHT,
                        top=PAGE_TOP, bottom=PAGE_BOTTOM)


# Registro global de pies de figura y tabla para generar leyendas consistentes
figure_registry = []
table_registry = []
_figure_labels = {}
_table_labels = {}

page_outline_plan = []
page_outline_actual = []
_page_counter = 0


def _reset_caption_registry():
    global figure_registry, table_registry, _figure_labels, _table_labels
    figure_registry = []
    table_registry = []
    _figure_labels = {}
    _table_labels = {}


def reset_page_counter():
    global page_outline_plan, page_outline_actual, _page_counter
    page_outline_plan = []
    page_outline_actual = []
    _page_counter = 0


def register_figure(key, title):
    if key in _figure_labels:
        return _figure_labels[key]['label']
    label = f"Figura N°{len(figure_registry)+1:02d}"
    entry = {'label': label, 'title': title}
    _figure_labels[key] = entry
    figure_registry.append(entry)
    return label


def register_table(key, title):
    if key in _table_labels:
        return _table_labels[key]['label']
    label = f"Tabla N°{len(table_registry)+1:02d}"
    entry = {'label': label, 'title': title}
    _table_labels[key] = entry
    table_registry.append(entry)
    return label


def figure_caption(key):
    entry = _figure_labels.get(key)
    if entry is None:
        raise KeyError(f"Figura con clave {key} no registrada")
    return f"{entry['label']}: {entry['title']}"


def table_caption(key):
    entry = _table_labels.get(key)
    if entry is None:
        raise KeyError(f"Tabla con clave {key} no registrada")
    return f"{entry['label']}: {entry['title']}"


def finalize_page(pdf, fig, page_info, footer_left=None, footer_center=None, footer_above_right=None):
    global _page_counter
    _page_counter += 1
    page_number = _page_counter
    fig.text(PAGE_RIGHT, PAGE_BOTTOM - PAGE_NUMBER_OFFSET,
             f'Página {page_number}', ha='right', va='center', fontsize=9)
    if footer_left:
        fig.text(PAGE_LEFT, PAGE_FOOTER_Y, footer_left,
                 ha='left', va='center', fontsize=9)
    if footer_center:
        fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_FOOTER_Y,
                 footer_center, ha='center', va='center', fontsize=9)
    if footer_above_right:
        fig.text(PAGE_RIGHT, PAGE_FOOTER_Y + 0.03,
                 footer_above_right, ha='right', va='center', fontsize=9)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    recorded = dict(page_info)
    recorded['page'] = page_number
    page_outline_actual.append(recorded)

def page_spectrum(modal_x, modal_y):
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    ax = fig.add_subplot(1, 1, 1)

    modal_periods = []
    for modal in (modal_x, modal_y):
        for mode in modal.get('modes', []):
            modal_periods.append(mode['T'])

    Tmax = max([0.5, Tl * 1.4] + modal_periods)
    periods = np.linspace(0.01, Tmax, 800)
    Sa = spectral_accel_e030(periods) / g_grav
    ax.plot(periods, Sa, color='tab:blue', lw=2.5, label='Espectro E.030 (zeta=5%)')

    def plot_modal_points(modal, color, label_prefix):
        modes = modal.get('modes', [])
        if not modes:
            return
        for idx, mode in enumerate(modes[:3], start=1):
            label = f"{label_prefix} modo {idx}: T={mode['T']:.2f}s" if idx == 1 else None
            ax.scatter(mode['T'], mode['Sa'] / g_grav, color=color, s=60, marker='o', label=label)

    plot_modal_points(modal_x, 'tab:orange', 'X')
    plot_modal_points(modal_y, 'tab:green', 'Y')

    ax.set_title('Espectro elastico E.030 (direcciones X e Y)', fontsize=14, weight='bold')
    ax.set_xlabel('Periodo T [s]')
    ax.set_ylabel('Sa [g]')
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)

    resume = [
        f'T1x={Tx1:.3f}s  masa mod={mode_mass_x_pct:.1f}%  Vb={base_shear_x_ton:.2f} t',
        f'T1y={Ty1:.3f}s  masa mod={mode_mass_y_pct:.1f}%  Vb={base_shear_y_ton:.2f} t',
        f'Deriva max: X={drift_x_pct:.2f}%  Y={drift_y_pct:.2f}%'
    ]
    footer = "\n".join(resume)
    return fig, footer



# ------------------ PARÁMETROS EDITABLES ------------------
# Discretización
nx, ny, nz = 1, 1, 1                       # vanos X, vanos Y, pisos
Lx_spans = 5                # len = nx
Ly_spans = 6.0                             # escalar -> se replica ny veces
H_levels = 3                     # len = nz

# Cargas de piso (kN/m2)
gamma_conc = 24.0                           # peso específico hormigón ~24 kN/m3
t_losa     = 0.2                          # espesor de losa (m)
q_losa     = gamma_conc * t_losa            # kN/m2 (peso propio de losa)
q_acab     = 1.00                           # acabados (ajusta)
q_tabiq    = 1.00                           # tabiquería distribuida (ajusta)
q_viva     = 2.00                           # E.020 (vivienda por defecto). Cambia según uso real.

# Por si cada nivel tiene uso distinto: acepta escalar o lista len=nz
qv_levels  = q_viva                         # carga viva por nivel
qd_extra   = q_acab + q_tabiq + q_losa      # DL de piso adicional (sin vigas/columnas)

# ¿Qué niveles reciben las cargas de piso? (p.ej. todos)
loaded_levels = list(range(1, nz+1))        # [1..nz]

# Secciónes (para modelo y dibujo)
E, nu = 25e6, 0.20
G      = E/(2*(1+nu))

# Columnas (sección rectangular b×h) — para 3D
sec_col_b, sec_col_h = 0.35, 0.35          # (b local y, h local z)
sec_col_plan = (0.35, 0.35)                # (bx, by) en planta
Acol  = sec_col_b*sec_col_h
Iycol = sec_col_b*sec_col_h**3/12
Izcol = sec_col_h*sec_col_b**3/12
Jcol  = 1e-3

# Vigas (rectangular b×h)
sec_beam_b, sec_beam_h = 0.30, 0.50
Abeam  = sec_beam_b*sec_beam_h
Iybeam = sec_beam_b*sec_beam_h**3/12
Izbeam = sec_beam_h*sec_beam_b**3/12
Jbeam  = 1e-3

# Banda en planta para dibujar vigas
beam_plan_width = 0.25

# Diseño de columnas (materiales y refuerzo)
fc_col = 28.0          # MPa
fy_col = 420.0         # MPa
Es_col = 200000.0      # MPa
clear_cover_col = 0.04 # m (cara de concreto a estribo)
stirrup_diam = 0.010   # m
bar_diam_main = 0.016  # m (aprox. barra 5/8")
n_bars_main = 8
phi_axial_col = 0.65   # factor resistencia para columna arriostrada
eps_cu = 0.003

# Diseño de vigas (materiales y opciones de refuerzo)
fc_beam = 28.0         # MPa
fy_beam = 420.0        # MPa
phi_flex_beam = 0.90   # ACI 318 flexión controlada
clear_cover_beam = 0.04    # m
stirrup_diam_beam = 0.010  # m

beam_bar_sizes = {
    'N4': 0.0127,
    'N5': 0.0159,
    'N6': 0.0190,
    'N8': 0.0254,
}

# Soportes en base
base_fix = (1,1,1,1,1,1)

# Salida
PDF_NAME = "Portico3D_Reporte_A4.pdf"

# Sismo (E.030) y combinaciones (E.060)
Z_sismo = 0.45            # Factor de zona sísmica
U_importancia = 1.00      # Factor de uso/importancia
S_suelo = 1.00            # Factor de suelo
R_respuesta = 5.00        # Factor de reducción de respuesta (marcos de concreto)
Tp = 0.40                 # Periodo Tp del espectro (s)
Tl = 2.50                 # Periodo Tl del espectro (s)
C_min = 0.10              # Valor mínimo del coeficiente C
xi_modal = 0.05           # Amortiguamiento modal (5%)
psi_live = 0.25           # Fracción de carga viva que participa en la masa
max_modal_modes = None    # Cambia a un entero para forzar cantidad de modos

# ------------------ UTILIDADES ------------------
def as_list(x, n):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != n: raise ValueError(f"Se esperaban {n} valores y recibí {len(x)}")
        return list(x)
    return [x]*n

Lx = as_list(Lx_spans, nx)
Ly = as_list(Ly_spans, ny)
H  = as_list(H_levels, nz)
qv = as_list(qv_levels, nz)   # viva por nivel

X = np.cumsum([0.0] + Lx)
Y = np.cumsum([0.0] + Ly)
Z = np.cumsum([0.0] + H)

def node_id(ix, iy, iz):
    return 1 + ix + (nx+1)*iy + (nx+1)*(ny+1)*iz

def tributary_width_X_row(iy):
    # para viga // X en fila iy → ancho tributario en Y
    dy_left  = (Y[iy]   - Y[iy-1]) if iy > 0   else (Y[1]-Y[0])
    dy_right = (Y[iy+1] - Y[iy])   if iy < ny  else (Y[ny]-Y[ny-1])
    return 0.5*(dy_left + dy_right)

def tributary_width_Y_col(ix):
    # para viga // Y en columna ix → ancho tributario en X
    dx_left  = (X[ix]   - X[ix-1]) if ix > 0   else (X[1]-X[0])
    dx_right = (X[ix+1] - X[ix])   if ix < nx  else (X[nx]-X[nx-1])
    return 0.5*(dx_left + dx_right)

# ------------------ MODELO OPENSEES ------------------
ops.wipe()
ops.model('basic','-ndm',3,'-ndf',6)

# Nodos
for iz in range(nz+1):
    for iy in range(ny+1):
        for ix in range(nx+1):
            ops.node(node_id(ix,iy,iz), X[ix], Y[iy], Z[iz])

# Apoyos base
for iy in range(ny+1):
    for ix in range(nx+1):
        ops.fix(node_id(ix,iy,0), *base_fix)

# Transformaciones
ops.geomTransf('Linear', 1, 0,0,1)  # vigas (z local = Z global)
ops.geomTransf('Linear', 2, 0,1,0)  # columnas

# Elementos
eleTag = 1
columns = []   # (ele, ix, iy, iz)  iz = nivel superior (entre iz-1 → iz)
beamsX  = []   # (ele, iz, iy, ix)
beamsY  = []   # (ele, iz, ix, iy)

# Columnas
for iz in range(1, nz+1):
    for iy in range(ny+1):
        for ix in range(nx+1):
            nd_i = node_id(ix,iy,iz-1)
            nd_j = node_id(ix,iy,iz)
            ops.element('elasticBeamColumn', eleTag, nd_i, nd_j,
                        Acol, E, G, Jcol, Iycol, Izcol, 2)
            columns.append((eleTag, ix, iy, iz))
            eleTag += 1

# Vigas X
for iz in range(1, nz+1):
    for iy in range(ny+1):
        for ix in range(nx):
            nd_i = node_id(ix,   iy, iz)
            nd_j = node_id(ix+1, iy, iz)
            ops.element('elasticBeamColumn', eleTag, nd_i, nd_j,
                        Abeam, E, G, Jbeam, Iybeam, Izbeam, 1)
            beamsX.append((eleTag, iz, iy, ix))
            eleTag += 1

# Vigas Y
for iz in range(1, nz+1):
    for ix in range(nx+1):
        for iy in range(ny):
            nd_i = node_id(ix, iy,   iz)
            nd_j = node_id(ix, iy+1, iz)
            ops.element('elasticBeamColumn', eleTag, nd_i, nd_j,
                        Abeam, E, G, Jbeam, Iybeam, Izbeam, 1)
            beamsY.append((eleTag, iz, ix, iy))
            eleTag += 1

# ------------------ CARGAS BASE, DIAFRAGMA RÍGIDO Y MASAS ------------------
g_grav = 9.80665  # m/s2
kN_TO_TON = 1.0 / 9.80665  # 1 kN = 0.10197 toneladas-fuerza
TON_TO_KN = 1.0 / kN_TO_TON

# Peso propio de vigas (lineal) y mapas de carga por caso
w_self_beam = gamma_conc * Abeam  # kN/m
beam_tags = [ele for (ele, _, _, _) in beamsX] + [ele for (ele, _, _, _) in beamsY]
w_dead_map = {}
w_live_map = {}
for (ele, iz, iy, ix) in beamsX:
    trib = tributary_width_X_row(iy)
    w_dead = w_self_beam
    w_live = 0.0
    if iz in loaded_levels:
        w_dead += qd_extra * trib
        w_live += qv[iz-1] * trib
    w_dead_map[ele] = w_dead
    w_live_map[ele] = w_live
for (ele, iz, ix, iy) in beamsY:
    trib = tributary_width_Y_col(ix)
    w_dead = w_self_beam
    w_live = 0.0
    if iz in loaded_levels:
        w_dead += qd_extra * trib
        w_live += qv[iz-1] * trib
    w_dead_map[ele] = w_dead
    w_live_map[ele] = w_live
w_total_map = {ele: w_dead_map.get(ele, 0.0) + w_live_map.get(ele, 0.0) for ele in beam_tags}
beam_zero_map = {ele: 0.0 for ele in beam_tags}

# Peso propio de columnas como carga nodal (nivel superior)
column_dead_loads = {}
for (ele, ix, iy, iz) in columns:
    nd_j = node_id(ix, iy, iz)
    Lc = Z[iz] - Z[iz-1]
    w_self_col = gamma_conc * Acol
    P_col = w_self_col * Lc
    column_dead_loads[nd_j] = column_dead_loads.get(nd_j, 0.0) + P_col

# Nodos maestros del diafragma rígido (uno por nivel)
floor_area = X[-1] * Y[-1]
x_cg = 0.5 * X[-1]
y_cg = 0.5 * Y[-1]
floor_master = {}
node_tag_counter = node_id(nx, ny, nz)
for iz in range(1, nz+1):
    node_tag_counter += 1
    master_tag = node_tag_counter
    ops.node(master_tag, x_cg, y_cg, Z[iz])
    ops.fix(master_tag, 0, 0, 1, 1, 1, 0)
    floor_master[iz] = master_tag
    for iy in range(ny+1):
        for ix in range(nx+1):
            slave = node_id(ix, iy, iz)
            ops.equalDOF(master_tag, slave, 1, 2, 6)

# Masas concentradas por piso (kg equivalentes en kN*s^2/m)
floor_mass_data = {iz: {'mass': 0.0, 'Jz': 0.0} for iz in range(1, nz+1)}
radius_sq_plate = (X[-1]**2 + Y[-1]**2) / 12.0

def add_point_mass(level, mass, x, y):
    if level not in floor_mass_data or mass <= 0.0:
        return
    r2 = (x - x_cg)**2 + (y - y_cg)**2
    floor_mass_data[level]['mass'] += mass
    floor_mass_data[level]['Jz'] += mass * r2

def add_uniform_plate_mass(level, mass):
    if level not in floor_mass_data or mass <= 0.0:
        return
    floor_mass_data[level]['mass'] += mass
    floor_mass_data[level]['Jz'] += mass * radius_sq_plate

# Cargas de piso (DL + ψ*LL)
for iz in range(1, nz+1):
    if iz in loaded_levels:
        dead_mass = qd_extra * floor_area / g_grav
        live_mass = psi_live * qv[iz-1] * floor_area / g_grav
        add_uniform_plate_mass(iz, dead_mass + live_mass)

# Masa de vigas (peso propio)
mass_per_length_beam = w_self_beam / g_grav
for (ele, iz, iy, ix) in beamsX:
    L = X[ix+1] - X[ix]
    m = mass_per_length_beam * L
    xc = 0.5 * (X[ix] + X[ix+1])
    yc = Y[iy]
    add_point_mass(iz, m, xc, yc)
for (ele, iz, ix, iy) in beamsY:
    L = Y[iy+1] - Y[iy]
    m = mass_per_length_beam * L
    xc = X[ix]
    yc = 0.5 * (Y[iy] + Y[iy+1])
    add_point_mass(iz, m, xc, yc)

# Masa de columnas (mitad a cada extremo)
mass_per_length_col = gamma_conc * Acol / g_grav
for (ele, ix, iy, iz) in columns:
    Lc = Z[iz] - Z[iz-1]
    m = mass_per_length_col * Lc
    share = 0.5 * m
    x = X[ix]; y = Y[iy]
    add_point_mass(iz, share, x, y)
    if iz-1 >= 1:
        add_point_mass(iz-1, share, x, y)

floor_mass_summary = {}
total_mass = 0.0
for iz, data in floor_mass_data.items():
    mass = data['mass']
    Jz = data['Jz']
    node_tag = floor_master[iz]
    ops.mass(node_tag, mass, mass, 0.0, 0.0, 0.0, Jz)
    floor_mass_summary[iz] = {'node': node_tag, 'mass': mass, 'Jz': Jz}
    total_mass += mass
total_weight = total_mass * g_grav

beam_lengths = {}
for (ele, iz, iy, ix) in beamsX:
    beam_lengths[ele] = X[ix+1] - X[ix]
for (ele, iz, ix, iy) in beamsY:
    beam_lengths[ele] = Y[iy+1] - Y[iy]

# ------------------ FUNCIONES DE ANÁLISIS ------------------
def spectral_c_e030(T):
    T = np.asarray(T, dtype=float)
    c = np.piecewise(
        T,
        [T < 0.2 * Tp,
         (T >= 0.2 * Tp) & (T < Tp),
         (T >= Tp) & (T < Tl),
         T >= Tl],
        [
            lambda val: 1.0 + 7.5 * (val / Tp),
            2.5,
            lambda val: 2.5 * (Tp / val),
            lambda val: 2.5 * (Tp * Tl / (val**2))
        ]
    )
    return np.maximum(c, C_min)

def spectral_accel_e030(T):
    C = spectral_c_e030(T)
    Sa = (Z_sismo * U_importancia * S_suelo / R_respuesta) * C * g_grav
    return Sa

def configure_linear_static():
    ops.wipeAnalysis()
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.test('NormUnbalance', 1e-9, 20, 0)
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')

def collect_case_results(beam_w_map):
    case_data = {'beams': {}, 'columns': {}, 'displacements': {}}
    for (ele, _, _, _) in beamsX + beamsY:
        f = np.array(ops.eleResponse(ele, 'localForce'), dtype=float)
        Vi = f[2] * kN_TO_TON
        Vj = -f[8] * kN_TO_TON
        Mi = f[4] * kN_TO_TON
        Mj = f[10] * kN_TO_TON
        case_data['beams'][ele] = {
            'Vi': Vi,
            'Vj': Vj,
            'Mi': Mi,
            'Mj': Mj,
            'w': beam_w_map.get(ele, 0.0) * kN_TO_TON
        }
    for (ele, ix, iy, iz) in columns:
        f = np.array(ops.eleResponse(ele, 'localForce'), dtype=float)
        case_data['columns'][ele] = {
            'Pi': f[0] * kN_TO_TON,
            'Vyi': f[1] * kN_TO_TON,
            'Vzi': f[2] * kN_TO_TON,
            'Pj': -f[6] * kN_TO_TON,
            'Vyj': -f[7] * kN_TO_TON,
            'Vzj': -f[8] * kN_TO_TON,
            'Myi': f[4] * kN_TO_TON,
            'Mzi': f[5] * kN_TO_TON,
            'Myj': f[10] * kN_TO_TON,
            'Mzj': f[11] * kN_TO_TON,
            'nivel': iz
        }
    for iz, node in floor_master.items():
        disp = [ops.nodeDisp(node, dof) for dof in (1, 2, 6)]
        case_data['displacements'][iz] = np.array(disp, dtype=float)
    return case_data

def apply_beam_distributed_loads(load_map):
    for ele, w in load_map.items():
        if abs(w) > 0.0:
            ops.eleLoad('-ele', ele, '-type', '-beamUniform', 0.0, -w)

def apply_column_loads(load_map):
    for node, P in load_map.items():
        if abs(P) > 0.0:
            ops.load(node, 0.0, 0.0, -P, 0.0, 0.0, 0.0)

def apply_seismic_pattern(loads_dict, direction):
    for node, vec in loads_dict.items():
        Fx, Fy, Mz = vec
        if direction == 'X':
            ops.load(node, Fx, 0.0, 0.0, 0.0, 0.0, Mz)
        else:
            ops.load(node, 0.0, Fy, 0.0, 0.0, 0.0, Mz)

def scale_seismic_loads(loads_dict, factor):
    return {node: np.array(vec, dtype=float) * factor for node, vec in loads_dict.items()}

def run_load_case(beam_w_map, column_loads=None, seismic_loads=None, direction=None):
    run_load_case.counter += 1
    ts_tag = run_load_case.counter
    ops.timeSeries('Linear', ts_tag)
    ops.pattern('Plain', ts_tag, ts_tag)
    if beam_w_map:
        apply_beam_distributed_loads(beam_w_map)
    if column_loads:
        apply_column_loads(column_loads)
    if seismic_loads:
        apply_seismic_pattern(seismic_loads, direction)
    configure_linear_static()
    ret = ops.analyze(1)
    if ret != 0:
        ops.test('NormUnbalance', 1e-8, 50, 0)
        ops.algorithm('Newton')
        ret = ops.analyze(1)
        if ret != 0:
            raise RuntimeError("No convergió el análisis estático para el caso de carga.")
    results = collect_case_results(beam_w_map if beam_w_map is not None else beam_zero_map)
    ops.loadConst('-time', 0.0)
    ops.remove('loadPattern', ts_tag)
    ops.remove('timeSeries', ts_tag)
    return results
run_load_case.counter = 100

def hermite_moment_diagram(L, Mi, Mj, Vi, Vj, npts=401):
    if L <= 0.0:
        x = np.zeros(npts)
        return x, np.zeros_like(x)
    x = np.linspace(0.0, L, npts)
    xi = x / L
    h1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    h2 = 3.0 * xi**2 - 2.0 * xi**3
    h3 = xi - 2.0 * xi**2 + xi**3
    h4 = -xi**2 + xi**3
    M = Mi * h1 + Mj * h2 + L * Vi * h3 + L * Vj * h4
    return x, M


def beta1_factor(fc_mpa):
    if fc_mpa <= 28.0:
        return 0.85
    reduction = 0.05 * max(0.0, (fc_mpa - 28.0) / 7.0)
    return max(0.65, 0.85 - reduction)


def build_beam_rebar_options():
    options = []
    for size, diam in beam_bar_sizes.items():
        area_bar = 0.25 * np.pi * diam**2
        for count in range(2, 7):
            label = f"{count}{size}"
            options.append({
                'label': label,
                'size': size,
                'count': count,
                'diam': diam,
                'area': count * area_bar
            })
    options.sort(key=lambda item: item['area'])
    return options


beam_rebar_options = build_beam_rebar_options()

MPA_TO_KGF_CM2 = 10.197162129779


def build_column_rebar_layout(b, h, cover, stirrup, bar_diam, n_bars):
    cover_to_bar = cover + stirrup + bar_diam / 2.0
    if cover_to_bar >= min(b, h) / 2.0:
        raise ValueError("La combinacion de recubrimiento y diametros deja sin espacio al acero longitudinal.")
    y = b / 2.0 - cover_to_bar
    z = h / 2.0 - cover_to_bar
    if n_bars != 8:
        raise ValueError("Esta rutina esta configurada para 8 barras longitudinales.")
    layout_coords = [
        (-y, -z), (-y,  z), ( y, -z), ( y,  z),
        (-y,  0.0), ( y,  0.0), ( 0.0, -z), ( 0.0,  z)
    ]
    area_bar = 0.25 * np.pi * bar_diam**2
    return [{'y': yi, 'z': zi, 'area': area_bar} for (yi, zi) in layout_coords]


def plot_column_section(ax, b, h, cover, stirrup, bar_diam, layout):
    half_b = b / 2.0
    half_h = h / 2.0
    rect = Rectangle((-half_b, -half_h), b, h, linewidth=1.2, edgecolor='k', facecolor='0.9')
    ax.add_patch(rect)
    stirrup_offset = cover + stirrup / 2.0
    inner = Rectangle((-half_b + stirrup_offset, -half_h + stirrup_offset),
                      b - 2 * stirrup_offset, h - 2 * stirrup_offset,
                      linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--')
    ax.add_patch(inner)
    for bar in layout:
        circ = Circle((bar['y'], bar['z']), bar_diam / 2.0, color='tab:blue', ec='k', lw=0.6)
        ax.add_patch(circ)

    steel_note = f"{len(layout)}Ø{bar_diam*1000:.0f}"
    ax.text(0.0, half_h + 0.05 * h, steel_note, ha='center', va='bottom',
            fontsize=8.6, color='tab:blue', weight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-half_b * 1.12, half_b * 1.12)
    ax.set_ylim(-half_h * 1.12, half_h * 1.22)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def bar_positions(count, width, cover, stirrup, bar_diam):
    if count <= 1:
        return [0.0]
    free = width - 2 * (cover + stirrup + bar_diam / 2.0)
    if free <= 0.0:
        spacing = 0.0
    else:
        spacing = free / (count - 1) if count > 1 else 0.0
    start = -free / 2.0
    return [start + i * spacing for i in range(count)]


def plot_beam_section(ax, width, depth, cover, stirrup, bottom_opt, top_opt):
    ax.set_title('Sección transversal', fontsize=10, weight='bold')
    half_b = width / 2.0
    half_h = depth / 2.0
    ax.add_patch(Rectangle((-half_b, -half_h), width, depth, linewidth=1.2, edgecolor='k', facecolor='0.9'))
    stirrup_offset = cover + stirrup / 2.0
    ax.add_patch(Rectangle((-half_b + stirrup_offset, -half_h + stirrup_offset),
                           width - 2 * stirrup_offset, depth - 2 * stirrup_offset,
                           linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--'))

    def place_bars(option, z_coord):
        if option['count'] == 0:
            return
        xs = bar_positions(option['count'], width, cover, stirrup, option['diam'])
        for x in xs:
            ax.add_patch(Circle((x, z_coord), option['diam'] / 2.0, color='tab:blue', ec='k', lw=0.6))

    z_bottom = -half_h + cover + stirrup + (bottom_opt['diam'] / 2.0 if bottom_opt['count'] > 0 else 0.0)
    z_top = half_h - cover - stirrup - (top_opt['diam'] / 2.0 if top_opt['count'] > 0 else 0.0)
    place_bars(bottom_opt, z_bottom)
    place_bars(top_opt, z_top)

    def annotate_face(count, diam, z_coord, valign):
        if count <= 0:
            return
        note = f"{count}Ø{diam*1000:.0f}"
        ax.text(0.0, z_coord + (0.04 if valign == 'bottom' else -0.04) * depth,
                note, ha='center', va=valign, fontsize=9, color='tab:blue', weight='bold')

    annotate_face(bottom_opt['count'], bottom_opt['diam'], z_bottom, 'bottom')
    annotate_face(top_opt['count'], top_opt['diam'], z_top, 'top')

    ax.axhline(0.0, color='0.5', lw=0.6, ls=':')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-half_b * 1.25, half_b * 1.25)
    ax.set_ylim(-half_h * 1.35, half_h * 1.35)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def design_beam_flexure(Mu_ton_m, tension_face, options, b, h, cover, stirrup,
                        fc_mpa, fy_mpa, phi):
    Mu_ton_m = max(float(Mu_ton_m), 0.0)
    Mu_Nm = Mu_ton_m * TON_TO_KN * 1e3
    fc_pa = fc_mpa * 1e6
    fy_pa = fy_mpa * 1e6
    beta1 = beta1_factor(fc_mpa)

    result = {
        'Mu_ton_m': Mu_ton_m,
        'Mu_Nm': Mu_Nm,
        'tension_face': tension_face,
        'option': None,
        'As_req': 0.0,
        'As_min': 0.0,
        'As_max': 0.0,
        'phiMn_Nm': 0.0,
        'phiMn_ton_m': 0.0,
        'ok': False,
        'ratio': 0.0,
        'needs_resize': False
    }

    if Mu_ton_m < 1e-6:
        Mu_Nm = 0.0

    def effective_depth(diam):
        return h - cover - stirrup - diam / 2.0

    fc_kgf = fc_mpa * MPA_TO_KGF_CM2
    fy_kgf = fy_mpa * MPA_TO_KGF_CM2

    diam_ref = options[0]['diam'] if options else 0.02
    d_ref = effective_depth(diam_ref)
    b_cm = b * 100.0
    d_cm = d_ref * 100.0
    rho_min = 0.7 * np.sqrt(fc_kgf) / fy_kgf
    As_min_cm2 = rho_min * b_cm * d_cm
    As_min = As_min_cm2 * 1e-4
    rho_b = (0.85 * fc_kgf * beta1 / fy_kgf) * (6000.0 / (6000.0 + fy_kgf))
    As_balanced_cm2 = rho_b * b_cm * d_cm
    As_max = 0.75 * As_balanced_cm2 * 1e-4

    result['As_min'] = As_min
    result['As_max'] = As_max

    if Mu_Nm <= 0.0:
        required_As = As_min
    else:
        d_guess = d_ref
        denom = phi * 0.85 * fc_pa * b * d_guess**2
        demand_ratio = min(1.0, (2.0 * Mu_Nm) / max(denom, 1e-9))
        if demand_ratio >= 1.0 - 1e-6:
            required_As = np.inf
        else:
            As_calc = (0.85 * fc_pa * b * d_guess / fy_pa) * (1.0 - np.sqrt(1.0 - demand_ratio))
            required_As = max(As_calc, As_min)
    result['As_req'] = required_As

    selected = None
    phiMn_selected = 0.0
    for opt in options:
        d_eff = effective_depth(opt['diam'])
        if d_eff <= 0.0:
            continue
        a = opt['area'] * fy_pa / (0.85 * fc_pa * b)
        Mn_Nm = opt['area'] * fy_pa * max(d_eff - a / 2.0, 0.0)
        phiMn = phi * Mn_Nm
        if required_As <= opt['area'] + 1e-10 and phiMn >= Mu_Nm - 1e-6:
            selected = opt
            phiMn_selected = phiMn
            break
    if selected is None and options:
        selected = options[-1]
        d_eff = effective_depth(selected['diam'])
        a = selected['area'] * fy_pa / (0.85 * fc_pa * b)
        Mn_Nm = selected['area'] * fy_pa * max(d_eff - a / 2.0, 0.0)
        phiMn_selected = phi * Mn_Nm
        if phiMn_selected < Mu_Nm - 1e-6:
            result['needs_resize'] = True

    if selected is None:
        return result

    result['option'] = selected
    result['phiMn_Nm'] = phiMn_selected
    result['phiMn_ton_m'] = phiMn_selected / (TON_TO_KN * 1e3)
    result['As_prov'] = selected['area']
    if Mu_Nm > 0.0:
        result['ratio'] = Mu_Nm / max(phiMn_selected, 1e-6)
    else:
        result['ratio'] = required_As / max(selected['area'], 1e-9)
    result['ok'] = phiMn_selected >= Mu_Nm - 1e-6 and (selected['area'] >= As_min - 1e-9)
    if selected['area'] > As_max + 1e-9:
        result['needs_resize'] = True
    return result


Pn0_limit_ton = None


def compute_modal_forces(direction_label, dir_index):
    master_nodes = [floor_master[iz] for iz in sorted(floor_master)]
    if not master_nodes:
        return {'forces': {}, 'modes': [], 'base_shear': 0.0, 'total_mass_dir': 0.0}
    ops.wipeAnalysis()
    ops.system('FullGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    node_masses = {node: np.array(ops.nodeMass(node), dtype=float) for node in master_nodes}
    total_mass_dir = sum(node_masses[node][dir_index] for node in master_nodes)
    target_modes = len(master_nodes) * 3
    if max_modal_modes is not None:
        target_modes = min(target_modes, max_modal_modes)
    eigen_vals = ops.eigen('-fullGenLapack', target_modes)
    modes = []
    for mode_idx, lam in enumerate(eigen_vals, start=1):
        if lam <= 0.0:
            continue
        omega = lam**0.5
        T = 2.0 * np.pi / omega
        phi = {}
        for node in master_nodes:
            phi[node] = np.array([ops.nodeEigenvector(node, mode_idx, dof) for dof in (1, 2, 6)], dtype=float)
        num = sum(node_masses[node][dir_index] * phi[node][dir_index] for node in master_nodes)
        den = sum(node_masses[node][dir_index] * phi[node][dir_index]**2 for node in master_nodes)
        if den <= 0.0 or abs(num) < 1e-12:
            continue
        Gamma = num / den
        Sa = spectral_accel_e030(T)
        forces = {}
        for node in master_nodes:
            masses = node_masses[node]
            phi_vec = phi[node]
            Fx = masses[0] * phi_vec[0] * Gamma * Sa
            Fy = masses[1] * phi_vec[1] * Gamma * Sa
            Mz = masses[5] * phi_vec[2] * Gamma * Sa
            forces[node] = np.array([Fx, Fy, Mz], dtype=float)
        modal_mass_eff = Gamma * num
        mass_ratio = modal_mass_eff / total_mass_dir if total_mass_dir > 0 else 0.0
        modes.append({
            'id': mode_idx,
            'T': T,
            'omega': omega,
            'Gamma': Gamma,
            'Sa': Sa,
            'forces': forces,
            'mass_ratio': mass_ratio
        })
    if not modes:
        return {'forces': {}, 'modes': [], 'base_shear': 0.0, 'total_mass_dir': total_mass_dir}
    dominant_idx = max(range(len(modes)), key=lambda i: modes[i]['mass_ratio'])
    combined = {node: np.zeros(3) for node in master_nodes}
    for comp in range(3):
        for node in master_nodes:
            sum_sq = sum(m['forces'][node][comp]**2 for m in modes)
            sign = np.sign(modes[dominant_idx]['forces'][node][comp])
            if sign == 0.0:
                sign = 1.0
            combined[node][comp] = sign * np.sqrt(sum_sq)
    for idx, mode in enumerate(modes):
        mode['cum_mass_ratio'] = sum(m['mass_ratio'] for m in modes[:idx+1])
    base_shear = sum(combined[node][dir_index] for node in master_nodes)
    return {
        'forces': combined,
        'modes': modes,
        'base_shear': base_shear,
        'total_mass_dir': total_mass_dir,
        'direction': direction_label
    }

# ------------------ CASOS DE CARGA BÁSICOS ------------------
modal_X = compute_modal_forces('X', 0)
modal_Y = compute_modal_forces('Y', 1)

case_results = {}
case_results['D'] = run_load_case(w_dead_map, column_loads=column_dead_loads)
case_results['L'] = run_load_case(w_live_map)
case_results['E_X_POS'] = run_load_case(beam_zero_map, seismic_loads=modal_X['forces'], direction='X')
case_results['E_X_NEG'] = run_load_case(beam_zero_map, seismic_loads=scale_seismic_loads(modal_X['forces'], -1.0), direction='X')
case_results['E_Y_POS'] = run_load_case(beam_zero_map, seismic_loads=modal_Y['forces'], direction='Y')
case_results['E_Y_NEG'] = run_load_case(beam_zero_map, seismic_loads=scale_seismic_loads(modal_Y['forces'], -1.0), direction='Y')

# ------------------ COMBINACIONES E.060 Y ENVOLVENTES ------------------
load_combinations = {
    '1.4CM+1.7CV': {'D': 1.40, 'L': 1.70},
    '1.25(CM+CV)+CSx': {'D': 1.25, 'L': 1.25, 'E_X_POS': 1.00},
    '1.25(CM+CV)-CSx': {'D': 1.25, 'L': 1.25, 'E_X_NEG': 1.00},
    '1.25(CM+CV)+CSy': {'D': 1.25, 'L': 1.25, 'E_Y_POS': 1.00},
    '1.25(CM+CV)-CSy': {'D': 1.25, 'L': 1.25, 'E_Y_NEG': 1.00},
    '0.9CM+CSx': {'D': 0.90, 'E_X_POS': 1.00},
    '0.9CM-CSx': {'D': 0.90, 'E_X_NEG': 1.00},
    '0.9CM+CSy': {'D': 0.90, 'E_Y_POS': 1.00},
    '0.9CM-CSy': {'D': 0.90, 'E_Y_NEG': 1.00},
}

beam_envelopes = {}
for ele in beam_tags:
    L = beam_lengths[ele]
    x = np.linspace(0.0, L, 401)
    V_pos = np.full_like(x, -np.inf)
    V_neg = np.full_like(x, np.inf)
    M_pos = np.full_like(x, -np.inf)
    M_neg = np.full_like(x, np.inf)
    for combo, factors in load_combinations.items():
        Vi = Vj = Mi = Mj = 0.0
        for case, coef in factors.items():
            data = case_results[case]
            blk = data['beams'][ele]
            Vi += coef * blk['Vi']
            Vj += coef * blk['Vj']
            Mi += coef * blk['Mi']
            Mj += coef * blk['Mj']
        x_local, M_curve = hermite_moment_diagram(L, Mi, Mj, Vi, Vj)
        V_curve = np.gradient(M_curve, x_local, edge_order=2) if L > 0 else np.zeros_like(M_curve)
        V_pos = np.maximum(V_pos, V_curve)
        V_neg = np.minimum(V_neg, V_curve)
        M_pos = np.maximum(M_pos, M_curve)
        M_neg = np.minimum(M_neg, M_curve)
    beam_envelopes[ele] = {
        'x': x,
        'V_pos': V_pos,
        'V_neg': V_neg,
        'M_pos': M_pos,
        'M_neg': M_neg
    }

beam_design_results = {}
for ele in beam_tags:
    env = beam_envelopes[ele]
    Mu_pos = float(np.max(env['M_pos'])) if env['M_pos'].size else 0.0
    Mu_neg = float(-np.min(env['M_neg'])) if env['M_neg'].size else 0.0
    bottom_design = design_beam_flexure(Mu_pos, 'bottom', beam_rebar_options,
                                        sec_beam_b, sec_beam_h,
                                        clear_cover_beam, stirrup_diam_beam,
                                        fc_beam, fy_beam, phi_flex_beam)
    top_design = design_beam_flexure(Mu_neg, 'top', beam_rebar_options,
                                     sec_beam_b, sec_beam_h,
                                     clear_cover_beam, stirrup_diam_beam,
                                     fc_beam, fy_beam, phi_flex_beam)
    beam_design_results[ele] = {
        'Mu_pos': Mu_pos,
        'Mu_neg': Mu_neg,
        'bottom': bottom_design,
        'top': top_design,
        'ok': bottom_design['ok'] and top_design['ok'],
        'needs_resize': bottom_design['needs_resize'] or top_design['needs_resize']
    }

beam_summary_rows = []
beam_order_entries = []
for (ele, iz, iy, ix) in beamsX:
    design = beam_design_results[ele]
    bottom_label = design['bottom']['option']['label'] if design['bottom']['option'] else '—'
    top_label = design['top']['option']['label'] if design['top']['option'] else '—'
    max_ratio = max(design['bottom']['ratio'], design['top']['ratio'])
    beam_summary_rows.append([
        ele,
        'X',
        f"nivel {iz}, y={iy}, vano x={ix}",
        f"{design['Mu_pos']:.2f}",
        f"{design['Mu_neg']:.2f}",
        bottom_label,
        top_label,
        f"{max_ratio:.2f}",
        'OK' if design['ok'] else 'No cumple'
    ])
    beam_order_entries.append((max_ratio, ele, f"Viga-X (nivel {iz}, y={iy}, vano x={ix}, ele={ele})"))

for (ele, iz, ix, iy) in beamsY:
    design = beam_design_results[ele]
    bottom_label = design['bottom']['option']['label'] if design['bottom']['option'] else '—'
    top_label = design['top']['option']['label'] if design['top']['option'] else '—'
    max_ratio = max(design['bottom']['ratio'], design['top']['ratio'])
    beam_summary_rows.append([
        ele,
        'Y',
        f"nivel {iz}, x={ix}, vano y={iy}",
        f"{design['Mu_pos']:.2f}",
        f"{design['Mu_neg']:.2f}",
        bottom_label,
        top_label,
        f"{max_ratio:.2f}",
        'OK' if design['ok'] else 'No cumple'
    ])
    beam_order_entries.append((max_ratio, ele, f"Viga-Y (nivel {iz}, x={ix}, vano y={iy}, ele={ele})"))

ordered_beams = sorted(beam_order_entries, reverse=True)

column_envelopes = {}
for (ele, ix, iy, iz) in columns:
    L = Z[iz] - Z[iz-1]
    z = np.linspace(0.0, L, 401)
    My_pos = np.full_like(z, -np.inf)
    My_neg = np.full_like(z, np.inf)
    Mz_pos = np.full_like(z, -np.inf)
    Mz_neg = np.full_like(z, np.inf)
    Vy_pos = np.full_like(z, -np.inf)
    Vy_neg = np.full_like(z, np.inf)
    Vz_pos = np.full_like(z, -np.inf)
    Vz_neg = np.full_like(z, np.inf)
    for combo, factors in load_combinations.items():
        Myi = Myj = Mzi = Mzj = 0.0
        Vyi = Vyj = Vzi = Vzj = 0.0
        for case, coef in factors.items():
            blk = case_results[case]['columns'][ele]
            Myi += coef * blk['Myi']
            Myj += coef * blk['Myj']
            Mzi += coef * blk['Mzi']
            Mzj += coef * blk['Mzj']
            Vyi += coef * blk['Vyi']
            Vyj += coef * blk['Vyj']
            Vzi += coef * blk['Vzi']
            Vzj += coef * blk['Vzj']
        if L > 0:
            My_curve = Myi + (Myj - Myi) * (z / L)
            Mz_curve = Mzi + (Mzj - Mzi) * (z / L)
            Vy_curve = Vyi + (Vyj - Vyi) * (z / L)
            Vz_curve = Vzi + (Vzj - Vzi) * (z / L)
        else:
            My_curve = np.full_like(z, Myi)
            Mz_curve = np.full_like(z, Mzi)
            Vy_curve = np.full_like(z, Vyi)
            Vz_curve = np.full_like(z, Vzi)
        My_pos = np.maximum(My_pos, My_curve)
        My_neg = np.minimum(My_neg, My_curve)
        Mz_pos = np.maximum(Mz_pos, Mz_curve)
        Mz_neg = np.minimum(Mz_neg, Mz_curve)
        Vy_pos = np.maximum(Vy_pos, Vy_curve)
        Vy_neg = np.minimum(Vy_neg, Vy_curve)
        Vz_pos = np.maximum(Vz_pos, Vz_curve)
        Vz_neg = np.minimum(Vz_neg, Vz_curve)
    column_envelopes[ele] = {
        'z': z,
        'My_pos': My_pos,
        'My_neg': My_neg,
        'Mz_pos': Mz_pos,
        'Mz_neg': Mz_neg,
        'Vy_pos': Vy_pos,
        'Vy_neg': Vy_neg,
        'Vz_pos': Vz_pos,
        'Vz_neg': Vz_neg,
        'nivel': iz
    }

combo_names = list(load_combinations.keys())
column_combo_forces = {}
for (ele, ix, iy, iz) in columns:
    column_combo_forces[ele] = {
        'i': {},
        'j': {},
        'meta': {
            'ix': ix,
            'iy': iy,
            'iz': iz,
            'z_bottom': Z[iz-1],
            'z_top': Z[iz]
        }
    }
    for combo, factors in load_combinations.items():
        Pi = Pj = Myi = Myj = Mzi = Mzj = 0.0
        Vyi = Vyj = Vzi = Vzj = 0.0
        for case, coef in factors.items():
            blk = case_results[case]['columns'][ele]
            Pi += coef * blk['Pi']
            Pj += coef * blk['Pj']
            Myi += coef * blk['Myi']
            Myj += coef * blk['Myj']
            Mzi += coef * blk['Mzi']
            Mzj += coef * blk['Mzj']
            Vyi += coef * blk['Vyi']
            Vyj += coef * blk['Vyj']
            Vzi += coef * blk['Vzi']
            Vzj += coef * blk['Vzj']
        Pu = 0.5 * (Pi + Pj)
        column_combo_forces[ele]['i'][combo] = {
            'Pu': Pu,
            'P_end': Pi,
            'Vy': Vyi,
            'Vz': Vzi,
            'My': Myi,
            'Mz': Mzi
        }
        column_combo_forces[ele]['j'][combo] = {
            'Pu': Pu,
            'P_end': Pj,
            'Vy': Vyj,
            'Vz': Vzj,
            'My': Myj,
            'Mz': Mzj
        }

def section_response_uniaxial(axis, c, rebar_layout, fc=fc_col, fy=fy_col, Es=Es_col, eps_c=eps_cu):
    depth = sec_col_h if axis == 'y' else sec_col_b
    width = sec_col_b if axis == 'y' else sec_col_h
    coord_key = 'z' if axis == 'y' else 'y'
    extreme = depth / 2.0
    beta1 = beta1_factor(fc)
    c_eff = max(c, 1e-6)
    a = beta1 * c_eff
    if a <= 0.0:
        Cc_kN = 0.0
        uc = extreme
    elif a >= depth:
        Cc_kN = 0.85 * fc * width * depth * 1000.0
        uc = 0.0
    else:
        Cc_kN = 0.85 * fc * width * a * 1000.0
        uc = extreme - a / 2.0
    Pn_kN = Cc_kN
    Mn_kN_m = Cc_kN * uc
    eps_tension = 0.0
    eps_y = fy / Es
    for bar in rebar_layout:
        coord = bar[coord_key]
        dist = extreme - coord
        strain = eps_c * (1.0 - dist / c_eff)
        stress = np.clip(Es * strain, -fy, fy)
        Fs_kN = stress * bar['area'] * 1000.0
        Pn_kN += Fs_kN
        Mn_kN_m += Fs_kN * coord
        eps_tension = min(eps_tension, strain)
    eps_t = abs(min(eps_tension, 0.0))
    if eps_t <= eps_y:
        phi_m = 0.65
    elif eps_t >= eps_y + 0.003:
        phi_m = 0.90
    else:
        phi_m = 0.65 + (eps_t - eps_y) * (0.25 / 0.003)
    phi_m = np.clip(phi_m, 0.65, 0.90)
    Pn_ton = Pn_kN * kN_TO_TON
    Mn_ton_m = Mn_kN_m * kN_TO_TON
    phiMn_ton_m = phi_m * abs(Mn_ton_m)
    compression_limit = Pn0_limit_ton if Pn0_limit_ton is not None else np.inf
    if Pn_ton >= 0.0:
        phiPn_ton = phi_m * min(Pn_ton, compression_limit)
    else:
        phiPn_ton = phi_m * Pn_ton
    return {
        'Pn': Pn_ton,
        'Mn': abs(Mn_ton_m),
        'phiMn': phiMn_ton_m,
        'phiPn': phiPn_ton,
        'phi_m': phi_m,
        'eps_t': eps_t
    }

def compute_uniaxial_curve(axis, rebar_layout, npts=240):
    depth = sec_col_h if axis == 'y' else sec_col_b
    c_values = np.linspace(0.01, depth * 6.0, npts)
    data = {'Pn': [], 'phiPn': [], 'Mn': [], 'phiMn': [], 'phi_m': [], 'eps_t': []}
    for c in c_values:
        res = section_response_uniaxial(axis, c, rebar_layout)
        for key in data:
            data[key].append(res[key])
    steel_area = sum(bar['area'] for bar in rebar_layout)
    tension_capacity_ton = -fy_col * steel_area * 1000.0 * kN_TO_TON
    phi_tension = 0.90
    data['Pn'].append(tension_capacity_ton)
    data['phiPn'].append(phi_tension * tension_capacity_ton)
    data['Mn'].append(0.0)
    data['phiMn'].append(0.0)
    data['phi_m'].append(phi_tension)
    data['eps_t'].append(fy_col / Es_col + 0.003)
    return {key: np.array(values, dtype=float) for key, values in data.items()}

class MomentCapacityCurve:
    def __init__(self, P_values, phiM_values, phiPn0):
        P_aug = np.append(P_values, phiPn0)
        M_aug = np.append(phiM_values, 0.0)
        order = np.argsort(P_aug)
        P_sorted = P_aug[order]
        M_sorted = M_aug[order]
        unique_P = []
        unique_M = []
        for p, m in zip(P_sorted, M_sorted):
            if unique_P and abs(p - unique_P[-1]) < 1e-4:
                unique_M[-1] = max(unique_M[-1], m)
            else:
                unique_P.append(p)
                unique_M.append(max(m, 0.0))
        self.P = np.array(unique_P, dtype=float)
        self.M = np.array(unique_M, dtype=float)

    def phiMn(self, Pu):
        if Pu <= self.P[0]:
            return self.M[0]
        if Pu >= self.P[-1]:
            return self.M[-1]
        return float(np.interp(Pu, self.P, self.M))

def bresler_alpha(Pu, phiPn0):
    if phiPn0 <= 1e-6:
        return 1.0
    ratio = max(0.0, Pu) / phiPn0
    if ratio <= 0.2:
        return 1.0
    ratio = min(ratio, 0.99)
    return min(10.0, 1.0 / (1.0 - ratio))

def evaluate_bresler(Pu, My, Mz, curve_y, curve_z, phiPn0):
    cap_y = curve_y.phiMn(Pu)
    cap_z = curve_z.phiMn(Pu)
    if cap_y <= 1e-6 or cap_z <= 1e-6:
        return np.inf, 1.0, cap_y, cap_z
    alpha = bresler_alpha(Pu, phiPn0) if Pu >= 0.0 else 1.0
    ratio_y = abs(My) / cap_y
    ratio_z = abs(Mz) / cap_z
    demand = ratio_y**alpha + ratio_z**alpha
    return demand, alpha, cap_y, cap_z

def bresler_boundary(Pu, curve_y, curve_z, phiPn0, npts=361):
    cap_y = curve_y.phiMn(Pu)
    cap_z = curve_z.phiMn(Pu)
    if cap_y <= 1e-6 or cap_z <= 1e-6:
        return None
    alpha = bresler_alpha(Pu, phiPn0) if Pu >= 0.0 else 1.0
    xs = np.linspace(-cap_y, cap_y, npts)
    base = np.clip(1.0 - np.power(np.abs(xs) / max(cap_y, 1e-9), alpha), 0.0, 1.0)
    ys = cap_z * np.power(base, 1.0 / alpha)
    return xs, ys, alpha

column_rebar_layout = build_column_rebar_layout(sec_col_b, sec_col_h,
                                                clear_cover_col, stirrup_diam,
                                                bar_diam_main, n_bars_main)
As_total_col = sum(bar['area'] for bar in column_rebar_layout)
rho_long_col = As_total_col / (sec_col_b * sec_col_h)
Ag_col = sec_col_b * sec_col_h
Pn0_nom_kN = 0.85 * fc_col * (Ag_col - As_total_col) * 1000.0 + fy_col * As_total_col * 1000.0
phiPn0_ton = phi_axial_col * Pn0_nom_kN * kN_TO_TON
Pn0_limit_ton = 0.8 * Pn0_nom_kN * kN_TO_TON

uniaxial_y_curve = compute_uniaxial_curve('y', column_rebar_layout)
uniaxial_z_curve = compute_uniaxial_curve('z', column_rebar_layout)
moment_curve_y = MomentCapacityCurve(uniaxial_y_curve['Pn'], uniaxial_y_curve['phiMn'], phiPn0_ton)
moment_curve_z = MomentCapacityCurve(uniaxial_z_curve['Pn'], uniaxial_z_curve['phiMn'], phiPn0_ton)

def build_interaction_plot_data(uniaxial_curve):
    order = np.argsort(uniaxial_curve['Pn'])
    P_sorted = uniaxial_curve['Pn'][order]
    phiP_sorted = uniaxial_curve['phiPn'][order]
    M_sorted = uniaxial_curve['Mn'][order]
    phiM_sorted = uniaxial_curve['phiMn'][order]
    P_vals, phiP_vals, M_vals, phiM_vals = [], [], [], []
    last_P = None
    for P_val, phiP_val, M_val, phiM_val in zip(P_sorted, phiP_sorted, M_sorted, phiM_sorted):
        if last_P is not None and abs(P_val - last_P) < 1e-4:
            if abs(M_val) > abs(M_vals[-1]):
                M_vals[-1] = M_val
                phiM_vals[-1] = phiM_val
            if abs(phiP_val) > abs(phiP_vals[-1]):
                phiP_vals[-1] = phiP_val
        else:
            P_vals.append(P_val)
            phiP_vals.append(phiP_val)
            M_vals.append(M_val)
            phiM_vals.append(phiM_val)
            last_P = P_val
    return {
        'Pn': np.array(P_vals, dtype=float),
        'phiPn': np.array(phiP_vals, dtype=float),
        'Mn': np.array(M_vals, dtype=float),
        'phiMn': np.array(phiM_vals, dtype=float)
    }

interaction_curve_plot = {
    'y': build_interaction_plot_data(uniaxial_y_curve),
    'z': build_interaction_plot_data(uniaxial_z_curve)
}

column_design_checks = {}
for (ele, ix, iy, iz) in columns:
    meta = column_combo_forces[ele]['meta']
    column_design_checks[ele] = {
        'i': {},
        'j': {},
        'summary': {
            'ix': ix,
            'iy': iy,
            'nivel_top': iz,
            'nivel_bottom': iz-1,
            'z_top': meta['z_top'],
            'z_bottom': meta['z_bottom']
        }
    }
    ratios_all = []
    Pu_all = []
    for end in ('i', 'j'):
        for combo in combo_names:
            data = column_combo_forces[ele][end][combo]
            Pu = data['Pu']
            My = data['My']
            Mz = data['Mz']
            demand, alpha, cap_y, cap_z = evaluate_bresler(Pu, My, Mz, moment_curve_y, moment_curve_z, phiPn0_ton)
            check = {
                'Pu': Pu,
                'My': My,
                'Mz': Mz,
                'ratio': demand,
                'alpha': alpha,
                'cap_y': cap_y,
                'cap_z': cap_z,
                'ratio_y': abs(My) / cap_y if cap_y > 1e-6 else np.inf,
                'ratio_z': abs(Mz) / cap_z if cap_z > 1e-6 else np.inf
            }
            column_design_checks[ele][end][combo] = check
            ratios_all.append(demand)
            Pu_all.append(Pu)
    column_design_checks[ele]['summary']['max_ratio'] = max(ratios_all) if ratios_all else 0.0
    column_design_checks[ele]['summary']['ok'] = column_design_checks[ele]['summary']['max_ratio'] <= 1.0 + 1e-6
    if Pu_all:
        column_design_checks[ele]['summary']['Pu_max_abs'] = max(abs(p) for p in Pu_all)
        column_design_checks[ele]['summary']['Pu_max_comp'] = max((p for p in Pu_all if p >= 0.0), default=0.0)
        column_design_checks[ele]['summary']['Pu_min_ten'] = min((p for p in Pu_all if p <= 0.0), default=0.0)
    else:
        column_design_checks[ele]['summary']['Pu_max_abs'] = 0.0
        column_design_checks[ele]['summary']['Pu_max_comp'] = 0.0
        column_design_checks[ele]['summary']['Pu_min_ten'] = 0.0
    unique_P = sorted({float(f"{p:.4f}") for p in Pu_all})
    max_levels = 8
    if len(unique_P) > max_levels:
        idxs = np.linspace(0, len(unique_P)-1, max_levels).astype(int)
        Pu_plot_levels = [unique_P[i] for i in idxs]
    else:
        Pu_plot_levels = unique_P
    if 0.0 not in Pu_plot_levels:
        Pu_plot_levels.append(0.0)
    column_design_checks[ele]['summary']['Pu_levels'] = sorted(Pu_plot_levels)

displacement_envelope = {iz: {'UX_max': -np.inf, 'UX_min': np.inf,
                              'UY_max': -np.inf, 'UY_min': np.inf,
                              'RZ_max': -np.inf, 'RZ_min': np.inf}
                         for iz in floor_master}
columns_sorted = sorted(columns, key=lambda item: column_design_checks[item[0]]['summary']['max_ratio'], reverse=True)
column_summary_rows = []
for (ele, ix, iy, iz) in columns_sorted:
    summary = column_design_checks[ele]['summary']
    status = 'OK' if summary['ok'] else 'No cumple'
    column_summary_rows.append([
        ele,
        f"({ix},{iy})",
        f"{summary['nivel_bottom']} -> {summary['nivel_top']}",
        f"{summary['Pu_max_comp']:.2f}",
        f"{summary['Pu_min_ten']:.2f}",
        f"{summary['max_ratio']:.2f}",
        status
    ])
drift_envelope = {iz: {'X_max': -np.inf, 'X_min': np.inf,
                       'Y_max': -np.inf, 'Y_min': np.inf}
                  for iz in floor_master}
for combo, factors in load_combinations.items():
    combo_disp = {}
    for iz, node in floor_master.items():
        ux = uy = rz = 0.0
        for case, coef in factors.items():
            disp = case_results[case]['displacements'][iz]
            ux += coef * disp[0]
            uy += coef * disp[1]
            rz += coef * disp[2]
        combo_disp[iz] = (ux, uy, rz)
        env = displacement_envelope[iz]
        env['UX_max'] = max(env['UX_max'], ux)
        env['UX_min'] = min(env['UX_min'], ux)
        env['UY_max'] = max(env['UY_max'], uy)
        env['UY_min'] = min(env['UY_min'], uy)
        env['RZ_max'] = max(env['RZ_max'], rz)
        env['RZ_min'] = min(env['RZ_min'], rz)
    for iz in range(1, nz+1):
        ux_curr, uy_curr, _ = combo_disp.get(iz, (0.0, 0.0, 0.0))
        ux_prev, uy_prev = (0.0, 0.0) if iz == 1 else combo_disp.get(iz-1, (0.0, 0.0, 0.0))[:2]
        drift_x = ux_curr - ux_prev
        drift_y = uy_curr - uy_prev
        env = drift_envelope[iz]
        env['X_max'] = max(env['X_max'], drift_x)
        env['X_min'] = min(env['X_min'], drift_x)
        env['Y_max'] = max(env['Y_max'], drift_y)
        env['Y_min'] = min(env['Y_min'], drift_y)

dominant_mode_X = max(modal_X['modes'], key=lambda m: m['mass_ratio']) if modal_X['modes'] else None
dominant_mode_Y = max(modal_Y['modes'], key=lambda m: m['mass_ratio']) if modal_Y['modes'] else None
mass_ratio_total_X = modal_X['modes'][-1]['cum_mass_ratio'] if modal_X['modes'] else 0.0
mass_ratio_total_Y = modal_Y['modes'][-1]['cum_mass_ratio'] if modal_Y['modes'] else 0.0

roof_disp_x = roof_disp_y = 0.0
if nz in displacement_envelope:
    roof_data = displacement_envelope[nz]
    roof_disp_x = max(abs(roof_data['UX_max']), abs(roof_data['UX_min']))
    roof_disp_y = max(abs(roof_data['UY_max']), abs(roof_data['UY_min']))

max_drift_ratio_x = 0.0
max_drift_ratio_y = 0.0
for iz in range(1, nz+1):
    env = drift_envelope[iz]
    drift_x = max(abs(env['X_max']), abs(env['X_min']))
    drift_y = max(abs(env['Y_max']), abs(env['Y_min']))
    h = H[iz-1] if iz-1 < len(H) else 1.0
    if h > 0:
        max_drift_ratio_x = max(max_drift_ratio_x, drift_x / h)
        max_drift_ratio_y = max(max_drift_ratio_y, drift_y / h)

base_shear_x = abs(modal_X['base_shear'])
base_shear_y = abs(modal_Y['base_shear'])
base_shear_x_ton = base_shear_x * kN_TO_TON
base_shear_y_ton = base_shear_y * kN_TO_TON
total_mass_ton = total_weight / g_grav
total_weight_ton = total_weight * kN_TO_TON
Tx1 = dominant_mode_X['T'] if dominant_mode_X else 0.0
Ty1 = dominant_mode_Y['T'] if dominant_mode_Y else 0.0
mode_mass_x_pct = dominant_mode_X['mass_ratio'] * 100 if dominant_mode_X else 0.0
mode_mass_y_pct = dominant_mode_Y['mass_ratio'] * 100 if dominant_mode_Y else 0.0
mass_ratio_total_X_pct = mass_ratio_total_X * 100
mass_ratio_total_Y_pct = mass_ratio_total_Y * 100
drift_x_pct = max_drift_ratio_x * 100
drift_y_pct = max_drift_ratio_y * 100
roof_disp_x_mm = roof_disp_x * 1000
roof_disp_y_mm = roof_disp_y * 1000
# ------------------ RESPUESTAS Y DIBUJOS ------------------
def ele_nodes(e):
    i,j = ops.eleNodes(e)
    xi,yi,zi = ops.nodeCoord(i); xj,yj,zj = ops.nodeCoord(j)
    L = ((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)**0.5
    return (i,j), (np.array([xi,yi,zi]), np.array([xj,yj,zj])), L

def local_end_forces(e):
    f = np.array(ops.eleResponse(e,'localForce'), dtype=float)
    if f.size == 12: return f
    if f.size == 6:  return np.concatenate([f, -f])
    g = np.array(ops.eleResponse(e,'force'), dtype=float)
    return g if g.size==12 else np.concatenate([g[:6], -g[:6]])

def vertical_fill(ax, x, y, base=0, density=250, color='k', lw=0.7):
    step = max(1, int(len(x)/density))
    segs = [[(x[i], base), (x[i], y[i])] for i in range(0, len(x), step)]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=0.9))


def column_diagram_envelope(ax, z, pos_curve, neg_curve, color, xlabel, set_ylabel=True):
    z = np.asarray(z)
    pos_curve = np.asarray(pos_curve, dtype=float)
    neg_curve = np.asarray(neg_curve, dtype=float)
    pos_curve = np.where(np.isfinite(pos_curve), pos_curve, 0.0)
    neg_curve = np.where(np.isfinite(neg_curve), neg_curve, 0.0)
    pos_curve = np.maximum(pos_curve, 0.0)
    neg_curve = np.minimum(neg_curve, 0.0)

    ax.fill_betweenx(z, 0.0, pos_curve, color=color, alpha=0.35)
    ax.fill_betweenx(z, 0.0, neg_curve, color=color, alpha=0.35)
    ax.plot(pos_curve, z, color=color, lw=2.0)
    ax.plot(neg_curve, z, color=color, lw=2.0)
    ax.axvline(0.0, color='0.25', lw=1.2)

    span = max(np.max(np.abs(pos_curve)), np.max(np.abs(neg_curve)), 1e-6)
    ax.set_xlim(-1.15 * span, 1.15 * span)
    ax.set_ylim(z[0], z[-1])
    if set_ylabel:
        ax.set_ylabel('z local [m]')
    ax.set_xlabel(xlabel)
    ax.grid(True, ls=':', alpha=0.35)

# -------- Planta por nivel (columnas rectangulares + vigas banda) ---------
def draw_plan_level(ax, level_z, highlight=None):
    ax.set_title(f"Planta — Nivel z = {level_z:g} m", fontsize=10, weight='bold')
    bx, by = sec_col_plan
    # Columnas (rectángulos)
    for iy in range(ny+1):
        for ix in range(nx+1):
            cx, cy = X[ix], Y[iy]
            rect = np.array([[cx-bx/2, cy-by/2],
                             [cx+bx/2, cy-by/2],
                             [cx+bx/2, cy+by/2],
                             [cx-bx/2, cy+by/2],
                             [cx-bx/2, cy-by/2]])
            ax.plot(rect[:,0], rect[:,1], color='k', lw=1.5)
            ax.fill(rect[:,0], rect[:,1], color='0.85', zorder=1)
    # Vigas X
    for (ele, iz, iy, ix) in beamsX:
        if Z[iz] != level_z: continue
        x0, x1 = X[ix], X[ix+1]; y = Y[iy]; w = beam_plan_width
        poly = np.array([[x0, y - w/2],[x1, y - w/2],[x1, y + w/2],[x0, y + w/2],[x0, y - w/2]])
        col = 'tab:blue' if highlight==ele else '0.25'
        ax.plot(poly[:,0], poly[:,1], color=col, lw=1.4)
        ax.fill(poly[:,0], poly[:,1], color=(0.7,0.8,1.0) if highlight==ele else '0.9', zorder=0)
    # Vigas Y
    for (ele, iz, ix, iy) in beamsY:
        if Z[iz] != level_z: continue
        y0, y1 = Y[iy], Y[iy+1]; x = X[ix]; w = beam_plan_width
        poly = np.array([[x - w/2, y0],[x - w/2, y1],[x + w/2, y1],[x + w/2, y0],[x - w/2, y0]])
        col = 'tab:blue' if highlight==ele else '0.25'
        ax.plot(poly[:,0], poly[:,1], color=col, lw=1.4)
        ax.fill(poly[:,0], poly[:,1], color=(0.7,0.8,1.0) if highlight==ele else '0.9', zorder=0)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.5, X[-1]+0.5); ax.set_ylim(-0.5, Y[-1]+0.5)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.grid(True, ls=':', alpha=0.35)

# ------------------ 3D con prismas -----------------------
def orthonormal_basis_along(p, q, up_hint=np.array([0,0,1.0])):
    ex = q - p; L = np.linalg.norm(ex)
    if L == 0: return None, None, None, 0.0
    ex = ex / L
    uz = up_hint / np.linalg.norm(up_hint)
    if abs(np.dot(ex, uz)) > 0.98:
        uz = np.array([1,0,0], dtype=float)
    ey = np.cross(uz, ex); n = np.linalg.norm(ey)
    if n < 1e-8: ey = np.array([0,1,0]); n = 1.0
    ey = ey / n
    uz = np.cross(ex, ey)
    uz = uz / np.linalg.norm(uz)
    return ex, ey, uz, L

def prism_faces(p, q, half_y, half_z, up_hint=np.array([0,0,1.0])):
    ex, ey, uz, L = orthonormal_basis_along(p, q, up_hint)
    if L == 0: return []
    c = []
    for end in [p, q]:
        c.append(end + (-half_y)*ey + (-half_z)*uz)
        c.append(end + (+half_y)*ey + (-half_z)*uz)
        c.append(end + (+half_y)*ey + (+half_z)*uz)
        c.append(end + (-half_y)*ey + (+half_z)*uz)
    faces = [
        [c[0],c[1],c[2],c[3]],
        [c[4],c[5],c[6],c[7]],
        [c[0],c[1],c[5],c[4]],
        [c[1],c[2],c[6],c[5]],
        [c[2],c[3],c[7],c[6]],
        [c[3],c[0],c[4],c[7]],
    ]
    return faces

def draw_frame_3d(ax, highlight=None):
    # Columnas
    for (ele, ix, iy, iz) in columns:
        (_, (p,q), _) = ele_nodes(ele)
        faces = prism_faces(p, q, sec_col_b/2, sec_col_h/2, up_hint=np.array([1,0,0]))
        col = (0.6,0.6,0.6) if highlight!=ele else (1.0,0.8,0.2)
        pc = Poly3DCollection(faces, facecolors=col, edgecolors='k', linewidths=0.6, alpha=0.95)
        ax.add_collection3d(pc)
    # Vigas
    for (ele, iz, iy, ix) in beamsX:
        (_, (p,q), _) = ele_nodes(ele)
        faces = prism_faces(p, q, sec_beam_b/2, sec_beam_h/2, up_hint=np.array([0,0,1.0]))
        col = (0.75,0.8,1.0) if highlight!=ele else (0.26,0.52,1.0)
        pc = Poly3DCollection(faces, facecolors=col, edgecolors='k', linewidths=0.5, alpha=0.95)
        ax.add_collection3d(pc)
    for (ele, iz, ix, iy) in beamsY:
        (_, (p,q), _) = ele_nodes(ele)
        faces = prism_faces(p, q, sec_beam_b/2, sec_beam_h/2, up_hint=np.array([0,0,1.0]))
        col = (0.75,0.8,1.0) if highlight!=ele else (0.26,0.52,1.0)
        pc = Poly3DCollection(faces, facecolors=col, edgecolors='k', linewidths=0.5, alpha=0.95)
        ax.add_collection3d(pc)

    ax.set_box_aspect([X[-1], Y[-1], Z[-1]])
    ax.set_xlim(-0.6, X[-1]+0.6); ax.set_ylim(-0.6, Y[-1]+0.6); ax.set_zlim(-0.2, Z[-1]+0.8)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')

# 3D con flechas de carga (longitud proporcional a w)
def draw_frame_3d_with_loads(ax):
    draw_frame_3d(ax)
    if not w_total_map:
        ax.set_title('3D (sin cargas gravitacionales)', fontsize=11, weight='bold')
        return
    wmax = max(w_total_map.values())
    if wmax <= 0:
        ax.set_title('3D (sin cargas gravitacionales)', fontsize=11, weight='bold')
        return

    def add_quivers(p, q, w, n_arrows=7):
        xs = np.linspace(p[0], q[0], n_arrows+2)[1:-1]
        ys = np.linspace(p[1], q[1], n_arrows+2)[1:-1]
        zs = np.linspace(p[2], q[2], n_arrows+2)[1:-1]
        L = 0.8 * max(0.25, (w / wmax))  # flecha proporcional con limite visual
        for x, y, z in zip(xs, ys, zs):
            ax.quiver(x, y, z, 0, 0, -L, arrow_length_ratio=0.25, color='k', linewidth=1.2)
    for (ele, iz, iy, ix) in beamsX:
        add_quivers(*ele_nodes(ele)[1], w_total_map.get(ele, 0.0) * kN_TO_TON)
    for (ele, iz, ix, iy) in beamsY:
        add_quivers(*ele_nodes(ele)[1], w_total_map.get(ele, 0.0) * kN_TO_TON)
    ax.set_title('3D (cargas gravitacionales)', fontsize=11, weight='bold')

# ------------------ Páginas de diagramas ------------------
def page_beam(ele, tag_txt, env):
    (_, (p, q), L) = ele_nodes(ele)
    x = env['x']
    V_pos = env['V_pos']
    V_neg = env['V_neg']
    M_pos = env['M_pos']
    M_neg = env['M_neg']

    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0], hspace=0.35)

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.set_title(f'Viga - {tag_txt}', fontsize=12, weight='bold')
    draw_frame_3d(ax3d, highlight=ele)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title('Cortante $V_z$ - envolvente', fontsize=11)
    ax1.add_patch(plt.Rectangle((0, -sec_beam_h/2), L, sec_beam_h, ec='k', fc='0.9', lw=1.0))
    ax1.axhline(0, color='k', lw=0.8)
    ax1.plot(x, V_pos, color='tab:blue', lw=2.0)
    ax1.plot(x, V_neg, color='tab:blue', lw=2.0)
    vertical_fill(ax1, x, V_pos, color='tab:blue')
    vertical_fill(ax1, x, V_neg, color='tab:blue')
    Vy = max(np.max(np.abs(V_pos)), np.max(np.abs(V_neg)), 1e-6)
    ax1.set_xlim(-0.02 * L, 1.02 * L)
    ax1.set_ylim(-1.25 * Vy, 1.25 * Vy)
    ax1.set_xlabel('x local [m]')
    ax1.set_ylabel('$V_z$ [t]')
    ax1.grid(True, ls=':', alpha=0.35)

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_title('Momento $M_y$ - envolvente', fontsize=11)
    ax2.add_patch(plt.Rectangle((0, -sec_beam_h/2), L, sec_beam_h, ec='k', fc='0.9', lw=1.0))
    ax2.axhline(0, color='k', lw=0.8)
    ax2.plot(x, M_pos, color='tab:red', lw=2.0)
    ax2.plot(x, M_neg, color='tab:red', lw=2.0)
    vertical_fill(ax2, x, M_pos, color='tab:red')
    vertical_fill(ax2, x, M_neg, color='tab:red')
    Myabs = max(np.max(np.abs(M_pos)), np.max(np.abs(M_neg)), 1e-6)
    ax2.set_xlim(-0.02 * L, 1.02 * L)
    ax2.set_ylim(-1.25 * Myabs, 1.25 * Myabs)
    ax2.set_xlabel('x local [m]')
    ax2.set_ylabel('$M_y$ [t-m]')
    ax2.grid(True, ls=':', alpha=0.35)

    Vmax = np.max(V_pos)
    Vmin = np.min(V_neg)
    Mmax = np.max(M_pos)
    Mmin = np.min(M_neg)
    footer = (f"L={L:.3f}  Vmax={Vmax:.3f} t  Vmin={Vmin:.3f} t  "
              f"Mmax={Mmax:.3f} t-m  Mmin={Mmin:.3f} t-m")
    fig.subplots_adjust(left=PAGE_LEFT, right=PAGE_RIGHT,
                        top=PAGE_TOP - 0.03, bottom=PAGE_BOTTOM + 0.08)
    return fig, footer


def _beam_layout_from_option(design_face):
    opt = design_face['option']
    if opt is None:
        return {'count': 0, 'diam': beam_bar_sizes['N6']}
    return {'count': opt['count'], 'diam': opt['diam']}


def draw_beam_design_block(fig, slot_spec, ele, tag_txt):
    design = beam_design_results[ele]
    bottom = design['bottom']
    top = design['top']

    block = slot_spec.subgridspec(3, 1, height_ratios=[0.46, 0.18, 0.36], hspace=0.16)

    ax_section = fig.add_subplot(block[0, 0])
    plot_beam_section(ax_section, sec_beam_b, sec_beam_h,
                      clear_cover_beam, stirrup_diam_beam,
                      _beam_layout_from_option(bottom), _beam_layout_from_option(top))
    ax_section.set_anchor('C')
    ax_section.set_title(tag_txt, fontsize=11, weight='bold', pad=10)

    ax_info = fig.add_subplot(block[1, 0])
    ax_info.axis('off')
    ax_info.set_anchor('N')
    info_lines = [
        f"fc'={fc_beam:.1f} MPa, fy={fy_beam:.0f} MPa, ϕ={phi_flex_beam:.2f}",
        f"Sección {sec_beam_b:.2f}×{sec_beam_h:.2f} m",
        f"Recubrimiento={clear_cover_beam*1000:.0f} mm",
        f"Mu(+)= {design['Mu_pos']:.2f} t-m  Mu(-)= {design['Mu_neg']:.2f} t-m",
        f"Estado global: {'OK' if design['ok'] else 'No cumple'}"
    ]
    if design['needs_resize']:
        info_lines.append('⚠️ Requiere revisar sección o combinación de refuerzo (ϕMn < Mu).')
    ax_info.text(0.5, 0.96, "\n".join(info_lines), ha='center', va='top', fontsize=9.4,
               transform=ax_info.transAxes, wrap=True)

    def format_face_row(label, face_design):
        opt = face_design['option']
        As_req_cm2 = face_design['As_req'] * 1e4
        As_min_cm2 = face_design['As_min'] * 1e4
        As_prov_cm2 = face_design.get('As_prov', 0.0) * 1e4
        phiMn = face_design['phiMn_ton_m']
        Mu = face_design['Mu_ton_m']
        ratio = face_design['ratio'] if Mu > 1e-6 else (As_req_cm2 / max(As_prov_cm2, 1e-6))
        ref = opt['label'] if opt else '—'
        estado = 'OK' if face_design['ok'] else 'No cumple'
        return [
            label,
            f"{Mu:.2f}",
            f"{phiMn:.2f}",
            f"{As_req_cm2:.2f}",
            f"{As_min_cm2:.2f}",
            f"{As_prov_cm2:.2f}",
            ref,
            f"{ratio:.2f}",
            estado
        ]

    ax_table = fig.add_subplot(block[2, 0])
    ax_table.axis('off')
    table_cols = ['Cara', 'Mu [t-m]', 'ϕMn [t-m]', 'As req [cm²]', 'As min [cm²]',
                  'As prov [cm²]', 'Refuerzo', 'Demanda', 'Estado']
    table_rows = [
        format_face_row('Inferior (+)', bottom),
        format_face_row('Superior (-)', top)
    ]
    col_widths = [0.12, 0.11, 0.15, 0.11, 0.10, 0.10, 0.12, 0.07, 0.12]
    table = ax_table.table(cellText=table_rows, colLabels=table_cols,
                           colWidths=col_widths, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(0.98, 1.15)
    for r, row in enumerate(table_rows):
        if row[-1] != 'OK':
            for c in range(len(table_cols)):
                table[(r+1, c)].set_facecolor('#f8d7da')
    ax_table.text(0.5, -TABLE_CAPTION_GAP,
                  table_caption(('beam_faces', ele)),
                  transform=ax_table.transAxes, ha='center', va='top', fontsize=9)


def build_beam_design_page(entries):
    rows = len(entries)
    if rows == 0:
        return None
    fig = plt.figure(figsize=(8.27, 11.69))
    outer = fig.add_gridspec(rows, 1, height_ratios=[1] * rows, hspace=0.32)
    for idx, (ele, tag_txt) in enumerate(entries):
        slot = outer[idx, 0]
        draw_beam_design_block(fig, slot, ele, tag_txt)
    apply_page_margins(fig)
    return fig


def prepare_caption_registry():
    _reset_caption_registry()
    if beam_summary_rows:
        register_table(('beam_summary', 'general'), 'Resumen de diseño de vigas')
    for (_, ele, tag_txt) in ordered_beams:
        register_table(('beam_faces', ele), f'Revisión de refuerzo por cara · {tag_txt}')
    if column_summary_rows:
        register_table(('column_summary', 'general'), 'Resumen de demandas y verificación de columnas')
    for (ele, ix, iy, iz) in columns_sorted:
        tag_txt = f"(x={ix}, y={iy}, piso {iz}, ele={ele})"
        register_table(('column_combos', ele), f'Combinaciones de carga y verificaciones · Columna {tag_txt}')
        register_figure(('column_interaction_my', ele), f'Interacción My - Pu · Columna {tag_txt}')
        register_figure(('column_interaction_mz', ele), f'Interacción Mz - Pu · Columna {tag_txt}')


def page_column(ele, tag_txt, env):
    (_, (p, q), L) = ele_nodes(ele)
    z = env['z']
    My_pos = env['My_pos']
    My_neg = env['My_neg']
    Mz_pos = env['Mz_pos']
    Mz_neg = env['Mz_neg']

    Vy_pos = env['Vy_pos']
    Vy_neg = env['Vy_neg']
    Vz_pos = env['Vz_pos']
    Vz_neg = env['Vz_neg']

    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.32, wspace=0.28)

    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    ax3d.set_title(f'Columna - {tag_txt}', fontsize=12, weight='bold')
    draw_frame_3d(ax3d, highlight=ele)

    ax_vy = fig.add_subplot(gs[1, 0])
    ax_vy.set_title('Cortante $V_y$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_vy, z, Vy_pos, Vy_neg, color='tab:orange', xlabel='$V_y$ [t]')

    ax_vz = fig.add_subplot(gs[1, 1])
    ax_vz.set_title('Cortante $V_z$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_vz, z, Vz_pos, Vz_neg, color='tab:blue', xlabel='$V_z$ [t]', set_ylabel=False)

    ax_my = fig.add_subplot(gs[2, 0])
    ax_my.set_title('Momento $M_y$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_my, z, My_pos, My_neg, color='tab:red', xlabel='$M_y$ [t·m]')

    ax_mz = fig.add_subplot(gs[2, 1])
    ax_mz.set_title('Momento $M_z$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_mz, z, Mz_pos, Mz_neg, color='tab:purple', xlabel='$M_z$ [t·m]', set_ylabel=False)

    Vy_max = float(np.max(Vy_pos))
    Vy_min = float(np.min(Vy_neg))
    Vz_max = float(np.max(Vz_pos))
    Vz_min = float(np.min(Vz_neg))
    My_max = float(np.max(My_pos))
    My_min = float(np.min(My_neg))
    Mz_max = float(np.max(Mz_pos))
    Mz_min = float(np.min(Mz_neg))
    footer = (f"L={L:.3f}  Vy(+/-)=({Vy_max:.3f}, {Vy_min:.3f}) t  "
              f"Vz(+/-)=({Vz_max:.3f}, {Vz_min:.3f}) t  "
              f"My(+/-)=({My_max:.3f}, {My_min:.3f}) t·m  "
              f"Mz(+/-)=({Mz_max:.3f}, {Mz_min:.3f}) t·m")
    return fig, footer


def page_column_design(ele, tag_txt):
    checks = column_design_checks[ele]
    summary = checks['summary']
    Pu_levels = summary.get('Pu_levels', [0.0])
    Pu_levels_plot = [lvl for lvl in Pu_levels if lvl <= phiPn0_ton + 1e-6]
    if len(Pu_levels_plot) == 0:
        Pu_levels_plot = [0.0]
    colors_levels = plt.cm.Greys(np.linspace(0.35, 0.85, len(Pu_levels_plot)))

    def plot_interaction(ax, axis_key, title):
        curve = interaction_curve_plot[axis_key]
        moment_key = 'My' if axis_key == 'y' else 'Mz'
        ratio_key = 'ratio_y' if axis_key == 'y' else 'ratio_z'
        xlabel = 'My [t-m]' if axis_key == 'y' else 'Mz [t-m]'
        ax.set_title(title, fontsize=11, weight='bold', pad=10)
        ax.plot(curve['Mn'], curve['Pn'], color='tab:blue', lw=2.0, label='Pn-Mn (ϕ=1.0)')
        ax.plot(-curve['Mn'], curve['Pn'], color='tab:blue', lw=2.0)
        ax.plot(curve['phiMn'], curve['phiPn'], color='tab:red', lw=2.0, label='ϕPn-ϕMn')
        ax.plot(-curve['phiMn'], curve['phiPn'], color='tab:red', lw=2.0)
        ax.axhline(phiPn0_ton, color='0.4', lw=0.8, ls='--', label=f'ϕPn0={phiPn0_ton:.1f} t')
        ax.axhline(0.0, color='0.7', lw=0.8, ls=':')

        demand_moments = []
        demand_axial = []
        for combo in combo_names:
            for end_key in ('i', 'j'):
                data = checks[end_key][combo]
                Pu = data['Pu']
                M_val = data[moment_key]
                ratio_val = data[ratio_key]
                ok_point = ratio_val <= 1.0 + 1e-6
                point_color = 'tab:green' if ok_point else 'tab:red'
                ax.scatter(M_val, Pu, color=point_color, marker='o', edgecolor='none', s=8, zorder=4)
                demand_moments.append(abs(M_val))
                demand_axial.append(Pu)

        moment_span = max(demand_moments + curve['Mn'].tolist() + curve['phiMn'].tolist() + [1.0])
        axial_values = demand_axial + curve['Pn'].tolist() + curve['phiPn'].tolist() + [phiPn0_ton, 0.0]
        axial_max = max(axial_values) if axial_values else 0.0
        axial_min = min(axial_values) if axial_values else 0.0
        margin_m = 0.15 * moment_span if moment_span > 1e-6 else 1.0
        margin_p = 0.05 * max(abs(axial_max), abs(axial_min), 1.0)
        ax.set_xlim(-moment_span - margin_m, moment_span + margin_m)
        ax.set_ylim(axial_min - margin_p, axial_max + margin_p)

        for idx, Pu in enumerate(Pu_levels_plot):
            if abs(Pu) <= max(abs(axial_max), abs(axial_min)) + 5 * margin_p:
                ax.axhline(Pu, color=colors_levels[idx], ls=':', lw=0.8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Pu [t]')
        ax.grid(True, ls=':', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    # Hoja 1: propiedades, sección y tabla de combinaciones
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    fig.suptitle('Diseño de Columnas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
    subheader = (
        f"Columna {tag_txt} · Metodología seccional con refuerzo longitudinal uniforme"
    )
    fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
             subheader, ha='center', va='center', fontsize=11)
    fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.055)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.36, 0.2, 0.44], hspace=0.16)

    ax_section = fig.add_subplot(gs[0, 0])
    plot_column_section(ax_section, sec_col_b, sec_col_h, clear_cover_col,
                        stirrup_diam, bar_diam_main, column_rebar_layout)
    ax_section.set_anchor('C')
    ax_section.set_title('Sección transversal de la columna', fontsize=11, weight='bold', pad=10)
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    ax_info.set_anchor('N')
    info_lines = [
        f"Ubicación: nivel {summary['nivel_bottom']} → {summary['nivel_top']} | z={summary['z_bottom']:.2f}→{summary['z_top']:.2f} m",
        f"fc'={fc_col:.1f} MPa, fy={fy_col:.0f} MPa, Es={Es_col/1000:.0f} GPa",
        f"Sección {sec_col_b:.2f}×{sec_col_h:.2f} m  Recubrimiento={clear_cover_col*1000:.0f} mm",
        f"Refuerzo: {n_bars_main} barras Ø{bar_diam_main*1000:.0f} mm (As={As_total_col*1e4:.2f} cm², ρ={rho_long_col*100:.2f}%)",
        f"ϕPn0={phiPn0_ton:.2f} t  Pu,max={summary['Pu_max_comp']:.2f} t  Pu,min={summary['Pu_min_ten']:.2f} t",
        f"Índice max Bresler={summary['max_ratio']:.2f} ({'OK' if summary['ok'] else 'No cumple'})"
    ]
    ax_info.text(0.5, 0.96, "\n".join(info_lines), ha='center', va='top', fontsize=9.6,
                transform=ax_info.transAxes, wrap=True)

    ax_table = fig.add_subplot(gs[2, 0])
    ax_table.axis('off')
    table_cols = ['Sección', 'Combo', 'Pu [t]', 'My [t-m]', 'Mz [t-m]', 'alpha', 'Índice Bresler', 'My/ϕMn', 'Mz/ϕMn']
    table_rows = []
    for end_key, end_label in [('i', 'Base'), ('j', 'Cabeza')]:
        for combo in combo_names:
            data = checks[end_key][combo]
            table_rows.append([
                end_label,
                combo,
                f"{data['Pu']:.2f}",
                f"{data['My']:.2f}",
                f"{data['Mz']:.2f}",
                f"{data['alpha']:.2f}",
                f"{data['ratio']:.2f}",
                f"{data['ratio_y']:.2f}",
                f"{data['ratio_z']:.2f}"
            ])
    col_widths = [0.1, 0.18, 0.11, 0.11, 0.11, 0.08, 0.13, 0.09, 0.09]
    table = ax_table.table(cellText=table_rows, colLabels=table_cols,
                           colWidths=col_widths, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(0.98, 1.08)
    for r, row in enumerate(table_rows):
        ratio_vals = [float(row[-3]), float(row[-2]), float(row[-1])]
        if any(val > 1.0 + 1e-3 for val in ratio_vals):
            for c in range(len(table_cols)):
                table[(r+1, c)].set_facecolor('#f8d7da')
    ax_table.text(0.5, -TABLE_CAPTION_GAP,
                  table_caption(('column_combos', ele)),
                  transform=ax_table.transAxes, ha='center', va='top', fontsize=9)
    fig1 = fig

    # Hoja 2: diagramas de interacción apilados
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    fig.suptitle('Diseño de Columnas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
    fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
             f"Columna {tag_txt} · Diagramas de interacción biaxial",
             ha='center', va='center', fontsize=11)
    fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.06)
    gs_inter = fig.add_gridspec(2, 1, height_ratios=[0.5, 0.5], hspace=0.32)
    ax_inter_y = fig.add_subplot(gs_inter[0, 0])
    plot_interaction(ax_inter_y, 'y', 'Interacción My - Pu')
    ax_inter_y.text(0.5, -FIGURE_CAPTION_GAP,
                    figure_caption(('column_interaction_my', ele)),
                    transform=ax_inter_y.transAxes, ha='center', va='top', fontsize=9)
    ax_inter_z = fig.add_subplot(gs_inter[1, 0])
    plot_interaction(ax_inter_z, 'z', 'Interacción Mz - Pu')
    ax_inter_z.text(0.5, -FIGURE_CAPTION_GAP,
                    figure_caption(('column_interaction_mz', ele)),
                    transform=ax_inter_z.transAxes, ha='center', va='top', fontsize=9)
    fig2 = fig

    return fig1, fig2


# ------------------ CONSTRUCCIÓN DEL PDF A4 ------------------
prepare_caption_registry()
reset_page_counter()


def _append_plan(title, subtitle='', level=1):
    page_outline_plan.append({'title': title, 'subtitle': subtitle, 'level': level})


_append_plan('Portada', 'Datos generales del proyecto', level=1)
_append_plan('Vistas Generales', 'Perfiles ortogonales y vista 3D', level=1)
for iz in range(1, nz+1):
    subtitle = f'nivel {iz} · z={Z[iz-1]:.2f}→{Z[iz]:.2f} m'
    _append_plan(f'Planta Nivel {iz}', subtitle, level=2)
_append_plan('Espectro Modal', 'Espectro E.030 y modos dominantes', level=1)
for (ele, iz, iy, ix) in beamsX:
    tag = f'nivel {iz}, y={iy}, vano x={ix}'
    _append_plan(f'Viga-X ele {ele}', tag, level=2)
for (ele, iz, ix, iy) in beamsY:
    tag = f'nivel {iz}, x={ix}, vano y={iy}'
    _append_plan(f'Viga-Y ele {ele}', tag, level=2)
if beam_summary_rows:
    _append_plan('Diseño de Vigas - Resumen', 'Flexión unidireccional por elemento', level=1)
for idx in range(0, len(ordered_beams), 2):
    batch = ordered_beams[idx:idx+2]
    tags = [tag for (_, _, tag) in batch]
    subtitle = ' · '.join(tags)
    _append_plan('Diseño de Vigas - Detalle', subtitle, level=2)
for (ele, ix, iy, iz) in columns:
    tag = f'(x={ix}, y={iy}, piso {iz}, ele={ele})'
    _append_plan(f'Columna ele {ele} - Envolventes', tag, level=2)
if column_summary_rows:
    _append_plan('Diseño de Columnas - Resumen', 'Demandas Bresler y estado', level=1)
for (ele, ix, iy, iz) in columns_sorted:
    tag = f'(x={ix}, y={iy}, piso {iz}, ele={ele})'
    _append_plan(f'Columna ele {ele} - Propiedades', tag, level=2)
    _append_plan(f'Columna ele {ele} - Diagramas', tag, level=2)


plan_iter = iter(page_outline_plan)

with PdfPages(PDF_NAME) as pdf:
    # Portada
    page_info = next(plan_iter)
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_HEADER_Y,
             "REPORTE DE PÓRTICO 3D", ha='center', fontsize=20, weight='bold')
    gamma_conc_t = gamma_conc * kN_TO_TON
    q_losa_t = q_losa * kN_TO_TON
    q_acab_t = q_acab * kN_TO_TON
    q_tabiq_t = q_tabiq * kN_TO_TON
    qv_t = [val * kN_TO_TON for val in qv]
    span_x_desc = ", ".join(f"{val:.2f} m" for val in Lx)
    span_y_desc = ", ".join(f"{val:.2f} m" for val in Ly)
    level_desc = ", ".join(f"{val:.2f} m" for val in H)
    qv_desc = ", ".join(f"{val:.3f} t/m²" for val in qv_t)

    info_header = f"Fecha: {datetime.now():%Y-%m-%d %H:%M}"

    paragraph_1 = (
        f"El pórtico 3D paramétrico (OpenSeesPy) se compone de {nx} vanos en X (∑Lx={sum(Lx):.2f} m; tramos {span_x_desc}) y "
        f"{ny} vanos en Y (∑Ly={sum(Ly):.2f} m; tramos {span_y_desc}), distribuidos en {nz} niveles con alturas {level_desc}. "
        f"Las columnas rectangulares cuentan con b={sec_col_b:.2f} m × h={sec_col_h:.2f} m y las vigas con b={sec_beam_b:.2f} m × h={sec_beam_h:.2f} m "
        f"(ancho tributario en planta={beam_plan_width:.2f} m). Los apoyos base se modelan como {base_fix} (restricciones aplicadas en la base)."
    )

    paragraph_2 = (
        f"La losa de espesor t_losa={t_losa:.3f} m y peso específico γ={gamma_conc_t:.3f} t/m³ produce q_losa={q_losa_t:.3f} t/m². "
        f"A ello se añaden acabados q_acab={q_acab_t:.3f} t/m², tabiquería q_tabiq={q_tabiq_t:.3f} t/m² y cargas vivas reducidas q_viva=({qv_desc}). "
        f"Se consideran los niveles cargados {loaded_levels} para el análisis gravitacional y sísmico."
    )

    paragraph_3 = (
        f"El peso sísmico equivalente asciende a {total_weight_ton:.2f} t (masa={total_mass_ton:.2f} t) con ψ_live={psi_live:.2f}. "
        f"Los parámetros del espectro E.030 son Z={Z_sismo:.2f}, U={U_importancia:.2f}, S={S_suelo:.2f}, R={R_respuesta:.2f}, Tp={Tp:.2f} s y Tl={Tl:.2f} s. "
        f"Los modos fundamentales resultan T₁x={Tx1:.3f} s y T₁y={Ty1:.3f} s, con participaciones de masa de {mode_mass_x_pct:.1f}% y {mode_mass_y_pct:.1f}% (acumulado {mass_ratio_total_X_pct:.1f}% y {mass_ratio_total_Y_pct:.1f}%). "
        f"Los cortantes basales son Vbx={base_shear_x_ton:.2f} t y Vby={base_shear_y_ton:.2f} t, mientras que las derivas máximas de entrepiso alcanzan {drift_x_pct:.2f}% en X y {drift_y_pct:.2f}% en Y con desplazamientos de cubierta Δtecho-X={roof_disp_x_mm:.1f} mm y Δtecho-Y={roof_disp_y_mm:.1f} mm."
    )

    wrapped_paragraphs = [textwrap.fill(p, width=110) for p in (paragraph_1, paragraph_2, paragraph_3)]
    info = "\n\n".join([info_header] + wrapped_paragraphs)
    fig.text(PAGE_LEFT, PAGE_TOP - 0.06, info, ha='left', va='top', fontsize=10)
    ax3d = fig.add_subplot(2,1,2, projection='3d')
    ax3d.set_title('Vista 3D (general)', fontsize=12)
    draw_frame_3d(ax3d)
    finalize_page(pdf, fig, page_info)

    # Vistas globales (perfiles y 3D con cargas)
    page_info = next(plan_iter)
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    gs  = fig.add_gridspec(2,2, hspace=0.35, wspace=0.25)
    # Perfil XZ
    axXZ = fig.add_subplot(gs[0,0])
    axXZ.set_title('Perfil-X (proyección X–Z)', fontsize=11, weight='bold')
    for ix in range(nx+1):
        for iz in range(1, nz+1):
            axXZ.plot([X[ix], X[ix]], [Z[iz-1], Z[iz]], color='k', lw=2)
    for iz in range(1, nz+1):
        for ix in range(nx):
            axXZ.plot([X[ix], X[ix+1]], [Z[iz], Z[iz]], color='0.3', lw=2)
    axXZ.set_xlim(-0.5, X[-1]+0.5); axXZ.set_ylim(-0.2, Z[-1]+0.8)
    axXZ.set_xlabel('X [m]'); axXZ.set_ylabel('Z [m]'); axXZ.grid(True, ls=':', alpha=0.35)
    # Perfil YZ
    axYZ = fig.add_subplot(gs[0,1])
    axYZ.set_title('Perfil-Y (proyección Y–Z)', fontsize=11, weight='bold')
    for iy in range(ny+1):
        for iz in range(1, nz+1):
            axYZ.plot([Y[iy], Y[iy]], [Z[iz-1], Z[iz]], color='k', lw=2)
    for iz in range(1, nz+1):
        for iy in range(ny):
            axYZ.plot([Y[iy], Y[iy+1]], [Z[iz], Z[iz]], color='0.3', lw=2)
    axYZ.set_xlim(-0.5, Y[-1]+0.5); axYZ.set_ylim(-0.2, Z[-1]+0.8)
    axYZ.set_xlabel('Y [m]'); axYZ.set_ylabel('Z [m]'); axYZ.grid(True, ls=':', alpha=0.35)
    # 3D general
    ax3d1 = fig.add_subplot(gs[1,0], projection='3d')
    ax3d1.set_title('3D', fontsize=11, weight='bold')
    draw_frame_3d(ax3d1)
    # 3D con CARGAS aplicadas (flechas proporcionales a w)
    ax3d2 = fig.add_subplot(gs[1,1], projection='3d')
    draw_frame_3d_with_loads(ax3d2)
    finalize_page(pdf, fig, page_info)

    # Plantas por nivel (una pagina por nivel)
    for iz in range(1, nz+1):
        page_info = next(plan_iter)
        zlev = Z[iz]
        fig = plt.figure(figsize=(8.27, 11.69))
        apply_page_margins(fig)
        axP = fig.add_subplot(1,1,1)
        draw_plan_level(axP, zlev)
        fig.suptitle(f'Planta - Nivel {iz}', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
        finalize_page(pdf, fig, page_info)

    # Espectro modal (antes de diagramas de elementos)
    page_info = next(plan_iter)
    fig, footer = page_spectrum(modal_X, modal_Y)
    finalize_page(pdf, fig, page_info, footer_left=footer)

    # Todas las VIGAS // X
    for (ele, iz, iy, ix) in beamsX:
        page_info = next(plan_iter)
        tag = f"Viga-X (nivel {iz}, y={iy}, vano x={ix}, ele={ele})"
        fig, footer = page_beam(ele, tag, beam_envelopes[ele])
        finalize_page(pdf, fig, page_info, footer_left=footer)

    # Todas las VIGAS // Y
    for (ele, iz, ix, iy) in beamsY:
        page_info = next(plan_iter)
        tag = f"Viga-Y (nivel {iz}, x={ix}, vano y={iy}, ele={ele})"
        fig, footer = page_beam(ele, tag, beam_envelopes[ele])
        finalize_page(pdf, fig, page_info, footer_left=footer)

    # Resumen de diseño de vigas
    if beam_summary_rows:
        page_info = next(plan_iter)
        fig = plt.figure(figsize=(8.27, 11.69))
        apply_page_margins(fig)
        fig.suptitle('Diseño de Vigas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
        fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
                 'Resumen general de diseño flexional', ha='center', va='center', fontsize=11)
        fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.06)
        gs_beam = fig.add_gridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.05)
        ax_text = fig.add_subplot(gs_beam[0, 0])
        ax_text.axis('off')
        ax_text.text(0.0, 1.0,
                     "Metodología: flexión unidireccional con ϕ=0.90, refuerzo mínimo según ACI 318-19.",
                     ha='left', va='top', fontsize=10)
        ax_table = fig.add_subplot(gs_beam[1, 0])
        ax_table.axis('off')
        table_cols = ['Ele', 'Orientación', 'Posición', 'Mu+ [t-m]', 'Mu- [t-m]',
                      'Ref. inf', 'Ref. sup', 'Demanda max', 'Estado']
        table = ax_table.table(cellText=beam_summary_rows, colLabels=table_cols, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.96, 1.10)
        for r, row in enumerate(beam_summary_rows):
            if row[-1] != 'OK':
                for c in range(len(table_cols)):
                    table[(r+1, c)].set_facecolor('#f8d7da')
        if beam_summary_rows:
            ax_table.text(0.5, -TABLE_CAPTION_GAP,
                          table_caption(('beam_summary', 'general')),
                          transform=ax_table.transAxes, ha='center', va='top', fontsize=9)
        finalize_page(pdf, fig, page_info)

    # Paginas detalladas de diseño de vigas (ordenadas por demanda)
    for idx in range(0, len(ordered_beams), 2):
        page_info = next(plan_iter)
        batch = ordered_beams[idx:idx+2]
        entries = [(ele, tag) for (_, ele, tag) in batch]
        fig = build_beam_design_page(entries)
        if fig is not None:
            finalize_page(pdf, fig, page_info)

    # Todas las COLUMNAS
    for (ele, ix, iy, iz) in columns:
        page_info = next(plan_iter)
        tag = f"(x={ix}, y={iy}, piso {iz}, ele={ele})"
        fig, footer = page_column(ele, tag, column_envelopes[ele])
        finalize_page(pdf, fig, page_info, footer_left=footer)

    # Resumen de diseño de columnas (Bresler)
    if column_summary_rows:
        page_info = next(plan_iter)
        fig = plt.figure(figsize=(8.27, 11.69))
        apply_page_margins(fig)
        fig.suptitle('Diseño de Columnas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
        fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
                 'Metodología Bresler (ACI 318-19) para columnas rectangulares',
                 ha='center', va='center', fontsize=11)
        fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.06)
        summary_text = [
            "Metodología: aproximación de Bresler basada en ACI 318-19 (ecuación 22.5.1.2).",
            f"Sección rectangular {sec_col_b:.2f} x {sec_col_h:.2f} m con refuerzo uniforme.",
            f"Refuerzo adoptado: {n_bars_main} barras diam.{bar_diam_main*1000:.0f} mm (As={As_total_col*1e4:.2f} cm², ρ={rho_long_col*100:.2f}%).",
            f"Materiales: fc'={fc_col:.1f} MPa, fy={fy_col:.0f} MPa; factor phi_axial={phi_axial_col:.2f}.",
            f"Capacidad axial factorizada: phiPn0={phiPn0_ton:.2f} t."
        ]
        gs_summary = fig.add_gridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.05)
        ax_text = fig.add_subplot(gs_summary[0, 0])
        ax_text.axis('off')
        ax_text.text(0.0, 1.0, "\n".join(summary_text), ha='left', va='top', fontsize=10)
        ax_table = fig.add_subplot(gs_summary[1, 0])
        ax_table.axis('off')
        table_cols = ['Ele', '(ix,iy)', 'Nivel', 'Pu_max [t]', 'Pu_min [t]', 'Índice max', 'Estado']
        table = ax_table.table(cellText=column_summary_rows, colLabels=table_cols, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.96, 1.12)
        for r, row in enumerate(column_summary_rows):
            if row[-1] != 'OK':
                for c in range(len(table_cols)):
                    table[(r+1, c)].set_facecolor('#f8d7da')
        if column_summary_rows:
            ax_table.text(0.5, -TABLE_CAPTION_GAP,
                          table_caption(('column_summary', 'general')),
                          transform=ax_table.transAxes, ha='center', va='top', fontsize=9)
        finalize_page(pdf, fig, page_info)

    # Paginas detalladas por columna con diagramas biaxiales
    for (ele, ix, iy, iz) in columns_sorted:
        tag = f"(x={ix}, y={iy}, piso {iz}, ele={ele})"
        fig1, fig2 = page_column_design(ele, tag)
        page_info = next(plan_iter)
        finalize_page(pdf, fig1, page_info)
        page_info = next(plan_iter)
        finalize_page(pdf, fig2, page_info)

try:
    extra_info = next(plan_iter)
    raise RuntimeError(f'Sobraron entradas en el plan de páginas: {extra_info}')
except StopIteration:
    pass

print(f"PDF generado: {PDF_NAME}")
print(f"Modal X: T1={Tx1:.3f}s, masa mod={mode_mass_x_pct:.1f}%, masa acum={mass_ratio_total_X_pct:.1f}%, Vb={base_shear_x_ton:.2f} t")
print(f"Modal Y: T1={Ty1:.3f}s, masa mod={mode_mass_y_pct:.1f}%, masa acum={mass_ratio_total_Y_pct:.1f}%, Vb={base_shear_y_ton:.2f} t")