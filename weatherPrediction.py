import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

data = pd.read_csv("./data/clean_weather.csv", index_col = 0)
data = data.ffill()
tInputs = np.array([np.array(data["tmax"]), np.array(data["tmin"])]).T
tOutputs = np.array([data["tmax_tomorrow"]])
tInputsNorm = (tInputs - np.min(tInputs, axis=0)) / (np.max(tInputs, axis=0) - np.min(tInputs, axis=0))
tOutputsNorm = (tOutputs - np.min(tOutputs)) / (np.max(tOutputs) - np.min(tOutputs))
learning_rate = 0.00000001

def sigmoid (x):
    return (2 / (1 + np.exp(-x))) - 1

def calculateError (expected, outputs):
    return expected - outputs

def trainNeuralNetwork (iterations):
    if (os.path.exists("./data/wieghts.json")):
        weights = pd.read_json("./data/wieghts.json")
    else:
        np.random.seed(1)
        weights = np.random.random((2, 1)) * 0.01
    for iteration in range(iterations):
        outputs = np.dot(tInputsNorm, weights)
        error = calculateError(tOutputsNorm.reshape(-1, 1), outputs)
        adjustments = np.dot(tInputsNorm.T, error) * learning_rate / len(tInputsNorm)
        weights += adjustments
    pd.DataFrame(weights).to_json("./data/wieghts.json")
    print(f"La presición del programa es del : {round((1 - np.mean(np.abs(error))) * 100, 2)}%")
    return weights

def predict (inputs, synapticWeights):
    inputsNorm = (inputs - np.min(tInputs, axis=0)) / (np.max(tInputs, axis=0) - np.min(tInputs, axis=0))
    weightedSum = np.dot(inputsNorm, synapticWeights)
    outputs = weightedSum * (np.max(tOutputs) - np.min(tOutputs)) + np.min(tOutputs)
    return outputs

def visualizeNetwork(synapticWeights, sampleInput, sampleRaw, samplePrediction, day=1, unit="F"):
    w1 = float(np.array(synapticWeights).flatten()[0])
    w2 = float(np.array(synapticWeights).flatten()[1])
    tmax_n, tmin_n = float(sampleInput[0]), float(sampleInput[1])
    tmax_f, tmin_f = float(sampleRaw[0]), float(sampleRaw[1])
    tmax_c = (tmax_f - 32) * 5/9 if unit == "C" else tmax_f
    tmin_c = (tmin_f - 32) * 5/9 if unit == "C" else tmin_f
    prod1 = tmax_n * w1
    prod2 = tmin_n * w2
    ws = prod1 + prod2
    pred_show = round((samplePrediction - 32) * 5/9, 1) if unit == "C" else round(samplePrediction, 1)

    G='#22c55e'; B='#3b82f6'; O='#f97316'; P='#a78bfa'; W='#f1f5f9'; Y='#fbbf24'; LG='#94a3b8'

    class Diapositivas:
        def __init__(self):
            self.slide = 0
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            self.fig.patch.set_facecolor('#0f172a')
            self.ax.set_facecolor('#0f172a')

            ax_prev = plt.axes([0.2, 0.02, 0.1, 0.07])
            ax_next = plt.axes([0.7, 0.02, 0.1, 0.07])
            self.btn_prev = Button(ax_prev, '← Anterior', color='#6366f1', hovercolor='#8b5cf6')
            self.btn_next = Button(ax_next, 'Siguiente →', color='#6366f1', hovercolor='#8b5cf6')
            self.btn_prev.on_clicked(self.prev_slide)
            self.btn_next.on_clicked(self.next_slide)
            self.draw()

        def prev_slide(self, event):
            if self.slide > 0:
                self.slide -= 1; self.ax.clear(); self.draw()

        def next_slide(self, event):
            if self.slide < 5:
                self.slide += 1; self.ax.clear(); self.draw()

        def draw(self):
            ax = self.ax
            ax.set_xlim(0,14); ax.set_ylim(0,9); ax.axis('off'); ax.set_facecolor('#0f172a')
            slides = [s0, s1, s2, s3, s4, s5]
            slides[self.slide](ax)
            self.fig.canvas.draw_idle()

    def mk_circle(ax, x, y, col, lbl, sub=""):
        c = plt.Circle((x,y), 0.8, color=col, ec='white', lw=2.5, zorder=5)
        ax.add_patch(c)
        ax.text(x, y+(0.2 if sub else 0), lbl, ha='center', va='center',
                fontsize=13, color='white', fontweight='bold', zorder=6)
        if sub:
            ax.text(x, y-0.3, sub, ha='center', va='center',
                    fontsize=10, color='white', zorder=6)

    def mk_txt(ax, x, y, s, sz=11, col=W, bold=False, bg=None):
        kw = dict(ha='center', va='center', fontsize=sz, color=col, zorder=7,
                  fontweight='bold' if bold else 'normal')
        if bg: kw['bbox'] = dict(boxstyle='round,pad=0.45', fc=bg[0], ec=bg[1], lw=1.5)
        ax.text(x, y, s, **kw)

    def mk_arrow(ax, x1, y1, x2, y2, w=2, col='#6366f1'):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle="-|>", lw=w, color=col, mutation_scale=22))

    def base(ax, out_lbl='?'):
        mk_circle(ax, 3.5, 5.5, G, 'tmax')
        mk_circle(ax, 3.5, 2.5, B, 'tmin')
        mk_circle(ax, 11,  4.0, O, out_lbl)
        ax.text(3.5,7.1,'ENTRADA',ha='center',fontsize=10,color=LG,fontweight='bold')
        ax.text(11, 7.1,'SALIDA', ha='center',fontsize=10,color=LG,fontweight='bold')

    def slide_num(ax, n):
        ax.text(7, 8.7, f"Paso {n}/6", ha='center', fontsize=10, color='#475569')

    # ── Slide 0: Entrada ─────────────────────────────────────────────
    def s0(ax):
        ax.text(7, 8.3, "① ENTRADA: Usuario ingresa temperaturas",
                ha='center', fontsize=16, color=Y, fontweight='bold')
        base(ax)
        mk_txt(ax, 0.6, 5.5, f"{tmax_c:.1f} °{unit}", sz=15, col=G, bold=True, bg=('#052e16','#22c55e'))
        mk_txt(ax, 0.6, 2.5, f"{tmin_c:.1f} °{unit}", sz=15, col=B, bold=True, bg=('#0c1a3a','#3b82f6'))
        mk_arrow(ax, 1.5, 5.5, 2.7, 5.5, w=4, col=G)
        mk_arrow(ax, 1.5, 2.5, 2.7, 2.5, w=4, col=B)
        mk_txt(ax, 7, 1.0, "Los valores entran a los nodos de la red.", sz=12, col=LG, bg=('#1e293b','#334155'))
        slide_num(ax, 1)

    # ── Slide 1: °C → °F ────────────────────────────────────────────
    def s1(ax):
        ax.text(7, 8.3, "② CONVERSIÓN: °C → °F (proceso interno)",
                ha='center', fontsize=16, color=Y, fontweight='bold')
        base(ax)
        mk_txt(ax, 0.6, 5.5, f"{tmax_c:.1f} °{unit}", sz=15, col=G, bold=True, bg=('#052e16','#22c55e'))
        mk_txt(ax, 0.6, 2.5, f"{tmin_c:.1f} °{unit}", sz=15, col=B, bold=True, bg=('#0c1a3a','#3b82f6'))
        mk_arrow(ax, 1.5, 5.5, 2.7, 5.5, w=4, col=G)
        mk_arrow(ax, 1.5, 2.5, 2.7, 2.5, w=4, col=B)
        # Resultado dentro del nodo
        ax.text(3.5, 5.2, f"{tmax_f:.1f}°F", ha='center', fontsize=10, color='#bbf7d0', zorder=6)
        ax.text(3.5, 2.2, f"{tmin_f:.1f}°F", ha='center', fontsize=10, color='#bfdbfe', zorder=6)
        mk_txt(ax, 7, 6.2, "°F = °C × 9/5 + 32", sz=13, col=LG, bold=True, bg=('#1e293b','#334155'))
        mk_txt(ax, 7, 1.0, f"{tmax_c:.1f} × 9/5 + 32 = {tmax_f:.1f}°F\n{tmin_c:.1f} × 9/5 + 32 = {tmin_f:.1f}°F",
               sz=12, col=W, bg=('#1e293b','#334155'))
        slide_num(ax, 2)

    # ── Slide 2: Normalización ───────────────────────────────────────
    def s2(ax):
        ax.text(7, 8.3, "③ NORMALIZACIÓN: Escalar a [0, 1]",
                ha='center', fontsize=16, color=Y, fontweight='bold')
        base(ax)
        mk_txt(ax, 0.6, 5.5, f"{tmax_f:.1f}°F", sz=14, col=G, bold=True, bg=('#052e16','#22c55e'))
        mk_txt(ax, 0.6, 2.5, f"{tmin_f:.1f}°F", sz=14, col=B, bold=True, bg=('#0c1a3a','#3b82f6'))
        mk_arrow(ax, 1.5, 5.5, 2.7, 5.5, w=4, col=G)
        mk_arrow(ax, 1.5, 2.5, 2.7, 2.5, w=4, col=B)
        ax.text(3.5, 5.2, f"{tmax_n:.4f}", ha='center', fontsize=11, color='#86efac', zorder=6)
        ax.text(3.5, 2.2, f"{tmin_n:.4f}", ha='center', fontsize=11, color='#93c5fd', zorder=6)
        mk_txt(ax, 7, 6.2, "(x − min) / (max − min)", sz=13, col=LG, bold=True, bg=('#1e293b','#334155'))
        mk_txt(ax, 7, 1.0, f"tmax: ({tmax_f:.1f} − 20) / (105 − 20) = {tmax_n:.4f}\ntmin: ({tmin_f:.1f} − 10) / (90 − 10)  = {tmin_n:.4f}",
               sz=11, col=W, bg=('#1e293b','#334155'))
        slide_num(ax, 3)

    # ── Slide 3: Pesos ───────────────────────────────────────────────
    def s3(ax):
        ax.text(7, 8.3, "④ PESOS: valor_norm × peso sináptico",
                ha='center', fontsize=16, color=Y, fontweight='bold')
        base(ax, '∑')
        mk_txt(ax, 0.6, 5.5, f"{tmax_n:.4f}", sz=13, col=G, bold=True, bg=('#052e16','#22c55e'))
        mk_txt(ax, 0.6, 2.5, f"{tmin_n:.4f}", sz=13, col=B, bold=True, bg=('#0c1a3a','#3b82f6'))
        mk_arrow(ax, 1.5, 5.5, 2.7, 5.5, w=3, col=G)
        mk_arrow(ax, 1.5, 2.5, 2.7, 2.5, w=3, col=B)
        mk_arrow(ax, 4.3, 5.5, 10.2, 4.3, w=w1*12, col=P)
        mk_arrow(ax, 4.3, 2.5, 10.2, 3.7, w=w2*12, col=P)
        mk_txt(ax, 6.5, 5.9, f"w₁ = {w1:.4f}", sz=12, col=P, bold=True, bg=('#1e1b4b','#6366f1'))
        mk_txt(ax, 6.5, 2.1, f"w₂ = {w2:.4f}", sz=12, col=P, bold=True, bg=('#1e1b4b','#6366f1'))
        mk_txt(ax, 8.8, 5.1, f"= {prod1:.4f}", sz=11, col='#c4b5fd')
        mk_txt(ax, 8.8, 2.9, f"= {prod2:.4f}", sz=11, col='#c4b5fd')
        mk_txt(ax, 7, 1.0, f"{tmax_n:.4f} × {w1:.4f} = {prod1:.4f}\n{tmin_n:.4f} × {w2:.4f} = {prod2:.4f}",
               sz=12, col=W, bg=('#1e293b','#334155'))
        slide_num(ax, 4)

    # ── Slide 4: Suma ────────────────────────────────────────────────
    def s4(ax):
        ax.text(7, 8.3, "⑤ SUMA PONDERADA en el nodo de salida",
                ha='center', fontsize=16, color=Y, fontweight='bold')
        base(ax, '∑')
        mk_arrow(ax, 4.3, 5.5, 10.2, 4.3, w=w1*12, col=P)
        mk_arrow(ax, 4.3, 2.5, 10.2, 3.7, w=w2*12, col=P)
        mk_txt(ax, 6.5, 5.9, f"{prod1:.4f}", sz=12, col=P, bold=True, bg=('#1e1b4b','#6366f1'))
        mk_txt(ax, 6.5, 2.1, f"{prod2:.4f}", sz=12, col=P, bold=True, bg=('#1e1b4b','#6366f1'))
        ax.text(11, 3.65, f"Σ={ws:.4f}", ha='center', fontsize=11, color='#fed7aa', zorder=6)
        mk_txt(ax, 7, 1.0, f"Σ = {prod1:.4f} + {prod2:.4f} = {ws:.4f}",
               sz=12, col=W, bg=('#1e293b','#334155'))
        slide_num(ax, 5)

    # ── Slide 5: Resultado ───────────────────────────────────────────
    def s5(ax):
        ax.text(7, 8.3, "⑥ RESULTADO: Desnormalización → °C",
                ha='center', fontsize=16, color=Y, fontweight='bold')
        base(ax, '✓')
        ax.text(11, 3.65, f"{pred_show}°{unit}", ha='center', fontsize=12,
                color='#fed7aa', fontweight='bold', zorder=6)
        mk_txt(ax, 7, 6.2,
               f"Σ={ws:.4f}  →  ×(max−min)+min  →  {samplePrediction:.1f}°F  →  ÷1.8−32  →  {pred_show}°C",
               sz=11, col=W, bg=('#1e293b','#334155'))
        ax.text(11, 5.5, f"Predicción\ndía #{day}:", ha='center', fontsize=12,
                color=O, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.45', fc='#431407', ec=O, lw=1.5))
        ax.text(11, 1.8, f"{pred_show} °{unit}",
                ha='center', fontsize=22, color=O, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='#431407', ec=O, lw=2))
        slide_num(ax, 6)

    Diapositivas()
    plt.show()

def tc_convert(temp_f, unit):
    """Convierte de °F a °C si el usuario eligió Celsius, o lo deja en °F."""
    if unit == "C":
        return (temp_f - 32) * (5 / 9)
    return temp_f