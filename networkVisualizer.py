import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

COLOR_NODE        = "#1f1f1f"
COLOR_NODE_BORDER = "#d9d9d9"
COLOR_FORWARD     = "#22c55e"
COLOR_OUTPUT      = "#16a34a"
COLOR_BACKPROP    = "#86efac"
COLOR_EDGE        = "#2563eb"
COLOR_EDGE_ACTIVE = "#22c55e"
COLOR_EDGE_BACK   = "#4ade80"
BG_COLOR          = "#000000"
TEXT_COLOR        = "white"
SUBTEXT_COLOR     = "#cbd5e1"

def visualizeNetwork(synapticWeights, sampleInput, sampleRaw, samplePrediction, day=1, unit="F"):
    w1 = float(np.array(synapticWeights).flatten()[0])
    w2 = float(np.array(synapticWeights).flatten()[1])
    tmax_n, tmin_n = float(sampleInput[0]), float(sampleInput[1])
    tmax_f, tmin_f = float(sampleRaw[0]),   float(sampleRaw[1])
    tmax_c = round((tmax_f-32)*5/9, 1) if unit=="C" else round(tmax_f, 1)
    tmin_c = round((tmin_f-32)*5/9, 1) if unit=="C" else round(tmin_f, 1)
    prod1  = tmax_n * w1
    prod2  = tmin_n * w2
    ws     = prod1 + prod2
    pred_show = round((samplePrediction-32)*5/9, 1) if unit=="C" else round(samplePrediction, 1)

    # ── Canvas ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-2, 8); ax.set_ylim(-4.5, 5.5); ax.axis("off")

    title_txt    = ax.text(3, 5.0, "", color=TEXT_COLOR,    fontsize=22, ha='center', fontweight='bold')
    subtitle_txt = ax.text(3, 4.3, "", color=SUBTEXT_COLOR, fontsize=12, ha='center')
    formula_txt  = ax.text(3,-3.8, "", color=COLOR_FORWARD, fontsize=12, ha='center', fontweight='bold')

    ax.text(0, 3.8, "Entrada",  color="white", fontsize=15, ha='center', fontweight='bold')
    ax.text(5, 3.8, "Salida",   color="white", fontsize=15, ha='center', fontweight='bold')

    # Nodos
    node_pos = {(0,0): (0, 1.1), (0,1): (0,-1.1), (1,0): (5, 0)}

    edge_lines   = {}
    weight_txts  = {}
    weights_vals = {((0,0),(1,0)): w1, ((0,1),(1,0)): w2}

    for edge, wv in weights_vals.items():
        (l1,i1),(l2,i2) = edge
        x1,y1 = node_pos[(l1,i1)]
        x2,y2 = node_pos[(l2,i2)]
        line, = ax.plot([x1,x2],[y1,y2], color=COLOR_EDGE, lw=1.2, alpha=0.4, zorder=1)
        edge_lines[edge] = line
        mx,my = (x1+x2)/2, (y1+y2)/2
        wt = ax.text(mx-0.2, my+0.3, f"w={wv:.4f}", color='#a78bfa', fontsize=11,
                     ha='center', alpha=0.0, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='#1e1b4b', ec='#6366f1', lw=1.2))
        weight_txts[edge] = wt

    node_circles = {}
    node_texts   = {}
    for key, (x,y) in node_pos.items():
        c = plt.Circle((x,y), 0.55, color=COLOR_NODE, ec=COLOR_NODE_BORDER, lw=2, zorder=3)
        ax.add_patch(c)
        node_circles[key] = c
        t = ax.text(x, y, "", color="white", fontsize=11, ha='center', va='center',
                    zorder=4, fontweight='bold')
        node_texts[key] = t

    # Etiquetas externas
    ax.text(-1.7, 1.1,  f"{tmax_c}°{unit}\ntmax", color=COLOR_FORWARD, fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#052e16', ec=COLOR_FORWARD, lw=1.3))
    ax.text(-1.7,-1.1,  f"{tmin_c}°{unit}\ntmin", color='#3b82f6',     fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#0c1a3a', ec='#3b82f6',     lw=1.3))
    ax.text( 7.0, 0,    f"tmax_tomorrow\n≈{pred_show}°{unit}", color=COLOR_OUTPUT, fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#052e16', ec=COLOR_OUTPUT,   lw=1.3))

    # ── Helpers ───────────────────────────────────────────────────
    def rgba_hex(h, a):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2],16)/255 for i in (0,2,4)) + (a,)

    def node_color(v, mode):
        a = 0.2 + 0.8*min(abs(v), 1.0)
        return rgba_hex(COLOR_FORWARD if mode=='forward' else COLOR_BACKPROP, a)

    def reset():
        for ln in edge_lines.values():
            ln.set_color(COLOR_EDGE); ln.set_linewidth(1.2); ln.set_alpha(0.4)
        for c in node_circles.values():
            c.set_facecolor(COLOR_NODE); c.set_edgecolor(COLOR_NODE_BORDER); c.set_linewidth(2)
        for t in node_texts.values():
            t.set_text("")
        for wt in weight_txts.values():
            wt.set_alpha(0.0)
        title_txt.set_text(""); subtitle_txt.set_text(""); formula_txt.set_text("")
        formula_txt.set_color(COLOR_FORWARD)

    def show_node(key, val, mode='forward'):
        node_circles[key].set_facecolor(node_color(val, mode))
        node_texts[key].set_text(f"{val:.3f}")

    def activate_edge(edge):
        ln = edge_lines[edge]
        ln.set_color(COLOR_EDGE_ACTIVE); ln.set_linewidth(3.5); ln.set_alpha(0.95)
        weight_txts[edge].set_alpha(1.0)

    # ── Frames ────────────────────────────────────────────────────
    def update(f):
        reset()

        if f < 25:
            title_txt.set_text(f"Red Neuronal · Predicción Día #{day}")
            subtitle_txt.set_text("Mira cómo los datos viajan y se transforman hasta la predicción")

        elif f < 55:
            title_txt.set_text("① Entrada del usuario")
            subtitle_txt.set_text(f"tmax = {tmax_c}°{unit}   |   tmin = {tmin_c}°{unit}")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            formula_txt.set_text(f"tmax = {tmax_c}°{unit}    tmin = {tmin_c}°{unit}")

        elif f < 85:
            title_txt.set_text("② Conversión °C → °F (interna)")
            subtitle_txt.set_text("La red trabaja internamente en Fahrenheit")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            formula_txt.set_text(f"{tmax_c}°C × 9/5 + 32 = {tmax_f:.1f}°F    |    {tmin_c}°C × 9/5 + 32 = {tmin_f:.1f}°F")

        elif f < 115:
            title_txt.set_text("③ Normalización  (x − min) / (max − min)")
            subtitle_txt.set_text("Escalar a [0,1] para que la red pueda procesar los valores")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            formula_txt.set_text(f"({tmax_f:.1f}−20)/(105−20) = {tmax_n:.4f}    |    ({tmin_f:.1f}−10)/(90−10) = {tmin_n:.4f}")

        elif f < 145:
            title_txt.set_text("④ Pesos sinápticos")
            subtitle_txt.set_text("Cada valor viaja por su conexión multiplicado por su peso")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            activate_edge(((0,0),(1,0))); activate_edge(((0,1),(1,0)))
            formula_txt.set_text(f"w₁ = {w1:.4f}    |    w₂ = {w2:.4f}")

        elif f < 165:
            title_txt.set_text("⑤ Multiplicación")
            subtitle_txt.set_text("Cada entrada × su peso antes de llegar al nodo de salida")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            activate_edge(((0,0),(1,0))); activate_edge(((0,1),(1,0)))
            formula_txt.set_text(f"{tmax_n:.4f} × {w1:.4f} = {prod1:.4f}    |    {tmin_n:.4f} × {w2:.4f} = {prod2:.4f}")

        elif f < 185:
            title_txt.set_text("⑥ Suma ponderada")
            subtitle_txt.set_text("El nodo de salida suma todos los productos recibidos")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            activate_edge(((0,0),(1,0))); activate_edge(((0,1),(1,0)))
            show_node((1,0), ws)
            formula_txt.set_text(f"Σ = {prod1:.4f} + {prod2:.4f} = {ws:.4f}")

        elif f < 205:
            title_txt.set_text("⑦ Desnormalización → Predicción")
            subtitle_txt.set_text("La suma vuelve a temperatura real en °C")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            activate_edge(((0,0),(1,0))); activate_edge(((0,1),(1,0)))
            show_node((1,0), ws)
            formula_txt.set_text(f"Σ={ws:.4f} × 85 + 20 = {samplePrediction:.1f}°F  →  {pred_show}°{unit}")

        else:
            title_txt.set_text(f"Prediccion: {pred_show} °{unit} para manana")
            subtitle_txt.set_text("La red completo el proceso exitosamente")
            show_node((0,0), tmax_n); show_node((0,1), tmin_n)
            activate_edge(((0,0),(1,0))); activate_edge(((0,1),(1,0)))
            show_node((1,0), ws)
            node_circles[(1,0)].set_edgecolor("#ffffff")
            node_circles[(1,0)].set_linewidth(3.5)
            pulse = 1.8 + 1.0*np.sin((f-205)*0.5)
            for ln in edge_lines.values():
                ln.set_linewidth(max(1.2, pulse))
            formula_txt.set_text(f"Entrada: {tmax_c}°{unit}, {tmin_c}°{unit}  →  Prediccion: {pred_show}°{unit}")
            formula_txt.set_color("#fbbf24")

        return [title_txt, subtitle_txt, formula_txt] + list(edge_lines.values()) + \
               list(node_circles.values()) + list(node_texts.values()) + list(weight_txts.values())

    ani = FuncAnimation(fig, update, frames=220, interval=120, blit=False, repeat=True)
    plt.tight_layout()
    ani.save("animacion_temperatura.gif", writer="pillow", fps=12)
    plt.show()