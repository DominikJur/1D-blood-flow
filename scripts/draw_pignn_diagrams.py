import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, x, y, w, h, text, color='lightblue'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec="black")
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color='gray'))

# Diagram 1: Topology
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
ax.text(5, 5.5, "Arterial Topology", ha='center', va='center', fontsize=14, fontweight='bold')
draw_box(ax, 1, 4.5, 2, 0.8, "Inlet", "lightcyan")
draw_box(ax, 4, 4.5, 2, 0.8, "Bifurcation", "lightcyan")
draw_box(ax, 7, 5, 2, 0.8, "Outlet 1", "lightcyan")
draw_box(ax, 7, 4, 2, 0.8, "Outlet 2", "lightcyan")
draw_arrow(ax, 3, 4.9, 4, 4.9); ax.text(3.5, 5.1, "Vessel 1", ha='center', fontsize=9)
draw_arrow(ax, 6, 4.9, 7, 5.4); ax.text(6.5, 5.3, "Vessel 2", ha='center', fontsize=9)
draw_arrow(ax, 6, 4.9, 7, 4.4); ax.text(6.5, 4.5, "Vessel 3", ha='center', fontsize=9)

ax.text(5, 3.5, "GNN Node Representation", ha='center', va='center', fontsize=14, fontweight='bold')
nodes = [(1.5,2.5, "Node 1"), (3.0,2.5, "Node 2"), (4.5,2.5, "Node 3"), (6.0,3.2, "Node 4"), (7.5,3.2, "Node 5"), (6.0,1.8, "Node 6"), (7.5,1.8, "Node 7")]
for x, y, txt in nodes:
    c = plt.Circle((x, y), 0.4, color='lightgreen', ec='black')
    ax.add_patch(c)
    ax.text(x, y, txt, ha='center', va='center', fontsize=9, fontweight='bold')
draw_arrow(ax, 1.9, 2.5, 2.6, 2.5)
draw_arrow(ax, 3.4, 2.5, 4.1, 2.5)
draw_arrow(ax, 4.8, 2.7, 5.7, 3.1)
draw_arrow(ax, 6.4, 3.2, 7.1, 3.2)
draw_arrow(ax, 4.8, 2.3, 5.7, 1.9)
draw_arrow(ax, 6.4, 1.8, 7.1, 1.8)
os.makedirs("research_papers", exist_ok=True)
plt.savefig("research_papers/diagram1_topology.png", bbox_inches='tight', dpi=150)
plt.close()

# Diagram 2: Architecture
fig, ax = plt.subplots(figsize=(6, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
draw_box(ax, 2, 8.5, 6, 1, "Input Graph (n x 3)\n[A_0, beta, delta X]", "#e1f5fe")
draw_box(ax, 2.5, 7, 5, 0.8, "ARMA GNN Layer 1 (32 ch)", "white")
draw_box(ax, 2.5, 5.5, 5, 0.8, "ARMA GNN Layer 2 (32 ch)", "white")
draw_box(ax, 4.5, 4.2, 1, 0.5, "...", "white")
draw_box(ax, 2.5, 2.5, 5, 0.8, "ARMA GNN Layer 5 (32 ch)", "white")
draw_box(ax, 1.5, 0.5, 7, 1, "Output Graph (n x 400)\n[A(t_1..t_200), u(t_1..t_200)]", "#e8f5e9")
draw_arrow(ax, 5, 8.5, 5, 7.8)
draw_arrow(ax, 5, 7.0, 5, 6.3)
draw_arrow(ax, 5, 5.5, 5, 4.7)
draw_arrow(ax, 5, 4.2, 5, 3.3)
draw_arrow(ax, 5, 2.5, 5, 1.5)
plt.savefig("research_papers/diagram2_architecture.png", bbox_inches='tight', dpi=150)
plt.close()

# Diagram 3: Loss
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
draw_box(ax, 0.5, 3.5, 3, 1, "PIGNN Output\nA(t), u(t)", "lightyellow")
draw_box(ax, 5, 6, 3, 1, "Measurement Loss", "mistyrose")
draw_box(ax, 5, 3.5, 3, 1, "PDE Collocation Loss", "mistyrose")
draw_box(ax, 5, 1, 3, 1, "Interface Loss", "mistyrose")

draw_box(ax, 1, 6.5, 3, 0.8, "Clinical Data", "white")
draw_box(ax, 1, 5.2, 3, 0.6, "Continuity Eq", "white")
draw_box(ax, 1, 4.5, 3, 0.6, "Momentum Eq", "white")
draw_box(ax, 1, 2.5, 3, 0.6, "Mass Conservation", "white")
draw_box(ax, 1, 1.8, 3, 0.6, "Pressure Continuity", "white")

draw_box(ax, 9.5, 3.5, 2, 1, "Total Loss", "lightcoral")

# Arrows from output
draw_arrow(ax, 3.5, 4, 5, 6.5)
draw_arrow(ax, 3.5, 4, 5, 4)
draw_arrow(ax, 3.5, 4, 5, 1.5)

# Arrows from data/physics to losses
draw_arrow(ax, 4, 6.9, 5, 6.6)
draw_arrow(ax, 4, 5.5, 5, 4.3)
draw_arrow(ax, 4, 4.8, 5, 4.1)
draw_arrow(ax, 4, 2.8, 5, 1.7)
draw_arrow(ax, 4, 2.1, 5, 1.5)

# Arrows to Total Loss
draw_arrow(ax, 8, 6.5, 9.5, 4.2)
draw_arrow(ax, 8, 4.0, 9.5, 4.0)
draw_arrow(ax, 8, 1.5, 9.5, 3.8)

plt.savefig("research_papers/diagram3_loss.png", bbox_inches='tight', dpi=150)
plt.close()
