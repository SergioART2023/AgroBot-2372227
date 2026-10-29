"""AgroBot - Búsqueda no informada e informada con interfaz gráfica (Tkinter)."""
import heapq
import time
import tkinter as tk
from collections import deque, namedtuple
from itertools import count
from tkinter import filedialog, messagebox, ttk

COMB_MAX = 30
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INF = float("inf")

# pos, fert, secos (frozenset), en_av, comb, av_disp (avioneta aún sin tomar)
Estado = namedtuple("Estado", "pos fert secos en_av comb av_disp")


class Nodo:
    __slots__ = ("estado", "padre", "g", "prof")

    def __init__(self, estado, padre, g, prof):
        self.estado, self.padre, self.g, self.prof = estado, padre, g, prof


# ----------------------------------------------------------------- MUNDO
class Mundo:
    def __init__(self, grid):
        self.grid = grid
        self.filas, self.cols = len(grid), len(grid[0])
        secos = []
        self.inicio = self.deposito = self.avioneta = None
        for r, fila in enumerate(grid):
            for c, v in enumerate(fila):
                if v == 2:
                    self.inicio = (r, c)
                elif v == 3:
                    secos.append((r, c))
                elif v == 4:
                    self.deposito = (r, c)
                elif v == 5:
                    self.avioneta = (r, c)
        self.secos0 = frozenset(secos)

    @staticmethod
    def desde_archivo(ruta):
        with open(ruta, encoding="utf-8") as f:
            grid = [[int(x) for x in ln.split()] for ln in f if ln.strip()]
        if len(grid) != 10 or any(len(f) != 10 for f in grid):
            raise ValueError("El mundo debe ser una matriz de 10x10.")
        planos = [v for f in grid for v in f]
        if any(v not in range(6) for v in planos):
            raise ValueError("Solo se permiten los valores 0 a 5.")
        for val, nombre in ((2, "AgroBot"), (4, "depósito"), (5, "avioneta")):
            if planos.count(val) != 1:
                raise ValueError(f"Debe haber exactamente un {nombre} ({val}).")
        return Mundo(grid)

    def inicial(self):
        return Estado(self.inicio, 0, self.secos0, False, 0, True)

    def es_meta(self, e):
        return not e.secos

    def sucesores(self, e):
        costo = 0.5 if e.en_av else 1.0
        for dr, dc in DIRS:
            r, c = e.pos[0] + dr, e.pos[1] + dc
            if not (0 <= r < self.filas and 0 <= c < self.cols) or self.grid[r][c] == 1:
                continue
            p = (r, c)
            fert, secos, en_av, comb, av_disp = e.fert, e.secos, e.en_av, e.comb, e.av_disp
            if en_av:
                comb -= 1
                if comb == 0:
                    en_av = False          # se acabó el combustible
            if p == self.avioneta and av_disp:
                en_av, comb, av_disp = True, COMB_MAX, False
            if p == self.deposito:
                fert = 2
            if p in secos and fert > 0:
                fert -= 1
                secos = secos - {p}
            yield Estado(p, fert, secos, en_av, comb, av_disp), costo


# ------------------------------------------------------------ HEURÍSTICA
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def mst_cost(puntos):
    if len(puntos) <= 1:
        return 0
    visitados = {0}
    dist = [manhattan(puntos[0], p) for p in puntos]
    total = 0
    while len(visitados) < len(puntos):
        j = min((i for i in range(len(puntos)) if i not in visitados), key=lambda i: dist[i])
        total += dist[j]
        visitados.add(j)
        for i in range(len(puntos)):
            if i not in visitados:
                dist[i] = min(dist[i], manhattan(puntos[j], puntos[i]))
    return total


def heuristica(e, deposito):
    if not e.secos:
        return 0
    puntos = [e.pos] + list(e.secos)
    if len(e.secos) > e.fert:
        puntos.append(deposito)
    L = mst_cost(puntos)
    F = e.comb if e.en_av else (COMB_MAX if e.av_disp else 0)
    return L - 0.5 * min(L, F)


# ---------------------------------------------------------------- BÚSQUEDA
def buscar(mundo, alg):
    """Devuelve dict con camino (lista de (estado, g)), nodos, profundidad, tiempo, costo."""
    t0 = time.perf_counter()
    ini = mundo.inicial()
    raiz = Nodo(ini, None, 0.0, 0)
    expandidos, prof_max, meta = 0, 0, None
    contador = count()

    if alg in ("Amplitud", "Profundidad evitando ciclos"):
        frontera, visitados = deque([raiz]), {ini}
        while frontera:
            n = frontera.popleft() if alg == "Amplitud" else frontera.pop()
            if mundo.es_meta(n.estado):
                meta = n
                break
            expandidos += 1
            for est, costo in mundo.sucesores(n.estado):
                if est in visitados:
                    continue
                visitados.add(est)
                h = Nodo(est, n, n.g + costo, n.prof + 1)
                prof_max = max(prof_max, h.prof)
                frontera.append(h)
    else:
        def prioridad(g, est):
            if alg == "Costo uniforme":
                return g
            h = heuristica(est, mundo.deposito)
            return h if alg == "Avara" else g + h

        mejor = {ini: 0.0}
        heap = [(prioridad(0.0, ini), next(contador), raiz)]
        while heap:
            _, _, n = heapq.heappop(heap)
            if alg != "Avara" and n.g > mejor.get(n.estado, INF):
                continue                   # entrada obsoleta
            if mundo.es_meta(n.estado):
                meta = n
                break
            expandidos += 1
            for est, costo in mundo.sucesores(n.estado):
                g = n.g + costo
                if alg == "Avara":
                    if est in mejor:       # sin reabrir
                        continue
                elif g >= mejor.get(est, INF):
                    continue               # A*/Costo reabren si g mejora
                mejor[est] = g
                h = Nodo(est, n, g, n.prof + 1)
                prof_max = max(prof_max, h.prof)
                heapq.heappush(heap, (prioridad(g, est), next(contador), h))

    res = {"nodos": expandidos, "prof": prof_max, "tiempo": time.perf_counter() - t0,
           "camino": None, "costo": None}
    if meta:
        camino, n = [], meta
        while n:
            camino.append((n.estado, n.g))
            n = n.padre
        res["camino"], res["costo"] = camino[::-1], meta.g
    return res


# --------------------------------------------------------------------- GUI
CELDA = 54
COLORES = {0: "white", 1: "#333333", 2: "white", 3: "#8B5A2B", 4: "#4A90D9", 5: "#F2C94C"}


class App:
    def __init__(self, root):
        self.root = root
        root.title("AgroBot")
        self.mundo, self.camino, self.anim_id = None, None, None

        barra = ttk.Frame(root, padding=6)
        barra.pack(fill="x")
        ttk.Button(barra, text="Cargar mundo", command=self.cargar).pack(side="left")
        self.tipo = ttk.Combobox(barra, state="readonly", width=12,
                                 values=["No informada", "Informada"])
        self.tipo.current(0)
        self.tipo.pack(side="left", padx=4)
        self.tipo.bind("<<ComboboxSelected>>", self.actualizar_algs)
        self.alg = ttk.Combobox(barra, state="readonly", width=26)
        self.alg.pack(side="left", padx=4)
        self.actualizar_algs()
        ttk.Button(barra, text="Ejecutar", command=self.ejecutar).pack(side="left", padx=4)
        ttk.Button(barra, text="Reiniciar", command=self.reiniciar).pack(side="left")
        ttk.Label(barra, text="Velocidad:").pack(side="left", padx=(10, 2))
        self.vel = tk.Scale(barra, from_=50, to=800, orient="horizontal", length=90, showvalue=False)
        self.vel.set(250)
        self.vel.pack(side="left")

        self.canvas = tk.Canvas(root, width=10 * CELDA, height=10 * CELDA, bg="white")
        self.canvas.pack(padx=6)
        self.estado_lbl = ttk.Label(root, text="Cargue un mundo para comenzar.")
        self.estado_lbl.pack(anchor="w", padx=6)
        self.reporte = ttk.Label(root, text="", justify="left")
        self.reporte.pack(anchor="w", padx=6, pady=6)

    def actualizar_algs(self, _=None):
        vals = (["Amplitud", "Costo uniforme", "Profundidad evitando ciclos"]
                if self.tipo.get() == "No informada" else ["Avara", "A*"])
        self.alg.config(values=vals)
        self.alg.current(0)

    def cargar(self):
        ruta = filedialog.askopenfilename(filetypes=[("Texto", "*.txt"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            self.mundo = Mundo.desde_archivo(ruta)
        except Exception as ex:
            messagebox.showerror("Archivo inválido", str(ex))
            return
        self.reiniciar()

    def reiniciar(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        if self.mundo:
            self.reporte.config(text="")
            self.dibujar(self.mundo.inicial(), 0.0)

    def dibujar(self, e, g):
        m, cv = self.mundo, self.canvas
        cv.delete("all")
        for r in range(m.filas):
            for c in range(m.cols):
                v = m.grid[r][c]
                color, txt = COLORES[v], ""
                if v == 3:
                    if (r, c) not in e.secos:
                        color = "#4CAF50"          # cultivo tratado
                    txt = "C"
                elif v == 4:
                    txt = "D"
                elif v == 5:
                    txt = "A" if e.av_disp else ""
                    if not e.av_disp:
                        color = "white"
                x, y = c * CELDA, r * CELDA
                cv.create_rectangle(x, y, x + CELDA, y + CELDA, fill=color, outline="#999")
                if txt:
                    cv.create_text(x + CELDA / 2, y + CELDA / 2, text=txt, fill="white",
                                   font=("Arial", 16, "bold"))
        r, c = e.pos
        x, y = c * CELDA, r * CELDA
        cv.create_oval(x + 8, y + 8, x + CELDA - 8, y + CELDA - 8,
                       fill="#FF8C00" if e.en_av else "#D32F2F", outline="black", width=2)
        cv.create_text(x + CELDA / 2, y + CELDA / 2, text="R", fill="white", font=("Arial", 14, "bold"))
        self.estado_lbl.config(
            text=f"Fertilizante: {e.fert}   Combustible avioneta: {e.comb if e.en_av else 0}   "
                 f"En avioneta: {'sí' if e.en_av else 'no'}   Costo acumulado: {g:.1f}")

    def ejecutar(self):
        if not self.mundo:
            messagebox.showinfo("AgroBot", "Primero cargue un mundo.")
            return
        self.reiniciar()
        alg = self.alg.get()
        self.reporte.config(text=f"Ejecutando {alg}...")
        self.root.update_idletasks()
        res = buscar(self.mundo, alg)
        txt = (f"Algoritmo: {alg}\nNodos expandidos: {res['nodos']}\n"
               f"Profundidad del árbol: {res['prof']}\nTiempo de cómputo: {res['tiempo']:.4f} s")
        if res["camino"] is None:
            txt += "\nNo se encontró solución."
            self.reporte.config(text=txt)
            return
        txt += f"\nPasos de la solución: {len(res['camino']) - 1}"
        if alg in ("Costo uniforme", "A*"):
            txt += f"\nCosto de la solución: {res['costo']:.1f}"
        self.reporte.config(text=txt)
        self.camino, self.paso = res["camino"], 0
        self.animar()

    def animar(self):
        e, g = self.camino[self.paso]
        self.dibujar(e, g)
        self.paso += 1
        if self.paso < len(self.camino):
            self.anim_id = self.root.after(self.vel.get(), self.animar)
        else:
            self.anim_id = None


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()