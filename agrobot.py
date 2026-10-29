"""AgroBot - Búsqueda no informada e informada con interfaz gráfica (Tkinter)."""
import heapq
import math
import os
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
ZOOM = 3                       # sprites de 16x16 -> 48x48 px por casilla
CELDA = 16 * ZOOM
MARGEN = 6                     # holgura alrededor del mapa para que no se recorten los bordes
BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_SPRITES = os.path.join(BASE, "sprites")
CARPETA_REPORTES = os.path.join(BASE, "reportes")
NOMBRES_SPRITES = ["Agrobot", "AgrobotSostenidoPorAvioneta", "Avioneta", "Muro",
                   "Piso", "CultivoSeco", "CultivoFertilizado", "Fertilizante"]

# Imagen de la pantalla de inicio (título "Agrobot" + sprite), dentro de la carpeta sprites.
# Cambia el nombre si tu archivo se llama distinto.
ARCHIVO_PORTADA = "Portada.png"
PORTADA_ANCHO_MAX = 460        # si la imagen es más ancha, se reduce automáticamente

BG = "#2b1d26"                 # color de fondo de la ventana
FG = "#f2e9d8"                 # color del texto


class Tooltip:
    """Globo de ayuda que aparece al dejar el mouse sobre un widget."""

    def __init__(self, widget, texto, retraso=400):
        self.widget, self.texto, self.retraso = widget, texto, retraso
        self.ventana, self.id = None, None
        widget.bind("<Enter>", self._programar, add="+")
        widget.bind("<Leave>", self._ocultar, add="+")
        widget.bind("<ButtonPress>", self._ocultar, add="+")

    def _programar(self, _=None):
        self._cancelar()
        self.id = self.widget.after(self.retraso, self._mostrar)

    def _cancelar(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def _mostrar(self):
        if self.ventana:
            return
        x = min(self.widget.winfo_rootx() + 10, self.widget.winfo_screenwidth() - 290)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.ventana = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.texto, justify="left", bg="#fff8dc", fg="#222222",
                 relief="solid", borderwidth=1, padx=6, pady=3, wraplength=260).pack()

    def _ocultar(self, _=None):
        self._cancelar()
        if self.ventana:
            self.ventana.destroy()
            self.ventana = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("AgroBot")
        root.configure(bg=BG)

        estilo = ttk.Style()
        estilo.theme_use("clam")
        estilo.configure("TFrame", background=BG)
        estilo.configure("TLabel", background=BG, foreground=FG)
        estilo.configure("TButton", padding=4)
        estilo.configure("Grande.TButton", padding=(18, 8), font=("Arial", 14, "bold"))

        self.mundo, self.camino, self.anim_id = None, None, None
        self.ruta_mundo = ""
        self.cargar_sprites()
        self.construir_portada()
        self.construir_juego()
        self.mostrar_portada()

    # ------------------------------------------------------ pantalla de inicio
    def construir_portada(self):
        self.portada = tk.Frame(self.root, bg=BG)
        centro = tk.Frame(self.portada, bg=BG)
        centro.place(relx=0.5, rely=0.5, anchor="center")     # todo centrado

        if self.img_portada:
            tk.Label(centro, image=self.img_portada, bg=BG, bd=0).pack(pady=(0, 24))
        else:                                                  # respaldo si falta la imagen
            tk.Label(centro, text="AgroBot", font=("Arial", 40, "bold"),
                     bg=BG, fg=FG).pack(pady=(0, 10))
            if self.img.get("Agrobot"):
                tk.Label(centro, image=self.img["Agrobot"], bg=BG, bd=0).pack(pady=(0, 24))

        ttk.Button(centro, text="Comenzar", style="Grande.TButton",
                   command=self.cargar).pack()

    def mostrar_portada(self):
        self.juego.pack_forget()
        self.portada.pack(fill="both", expand=True)
        ancho = 10 * CELDA + 2 * MARGEN + 24
        self.root.geometry(f"{ancho}x{ancho + 60}")
        self.root.minsize(ancho, ancho + 60)
        self.centrar_ventana()

    def mostrar_juego(self):
        self.portada.pack_forget()
        self.juego.pack(fill="both", expand=True)
        self.root.minsize(1, 1)
        self.root.geometry("")                  # tamaño según el contenido
        self.centrar_ventana()

    def ajustar_ventana(self):
        """Agranda la ventana (solo si hace falta) para que se vea todo el contenido."""
        self.root.update_idletasks()
        w = max(self.root.winfo_width(), self.root.winfo_reqwidth())
        h = max(self.root.winfo_height(), self.root.winfo_reqheight())
        x, y = self.root.winfo_x(), self.root.winfo_y()
        y = max(0, min(y, self.root.winfo_screenheight() - h - 60))   # que no se salga de la pantalla
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def centrar_ventana(self):
        """Ubica la ventana en el centro de la pantalla."""
        self.root.update_idletasks()
        w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        if self.root.winfo_width() > 1:
            w, h = max(w, self.root.winfo_width()), max(h, self.root.winfo_height())
        x = (self.root.winfo_screenwidth() - w) // 2
        y = max(0, (self.root.winfo_screenheight() - h) // 2 - 30)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ----------------------------------------------------------- pantalla juego
    def construir_juego(self):
        self.juego = ttk.Frame(self.root)
        barra = ttk.Frame(self.juego, padding=6)
        barra.pack(fill="x", pady=(6, 18))        # espacio entre las opciones y el mundo
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
        lbl_vel = ttk.Label(barra, text="Velocidad:")
        lbl_vel.pack(side="left", padx=(10, 2))
        self.vel = tk.Scale(barra, from_=50, to=800, orient="horizontal", length=90,
                            showvalue=False, bg=BG, fg=FG, highlightthickness=0,
                            troughcolor="#45283c")
        self.vel.set(250)
        self.vel.pack(side="left")
        ayuda = ("Entre más a la derecha esté la barra, más despacio anda el robot; "
                 "entre más a la izquierda, más rápido.")
        Tooltip(lbl_vel, ayuda)
        Tooltip(self.vel, ayuda)

        self.canvas = tk.Canvas(self.juego, width=10 * CELDA + 2 * MARGEN,
                                height=10 * CELDA + 2 * MARGEN,
                                bg="#45283c", highlightthickness=0, bd=0)
        self.canvas.pack(padx=6, pady=(0, 18))    # espacio entre el mundo y el reporte
        self.estado_lbl = ttk.Label(self.juego, text="")
        self.estado_lbl.pack(anchor="w", padx=6, pady=(0, 8))
        self.reporte = ttk.Label(self.juego, text="", justify="left",
                                 wraplength=10 * CELDA + 2 * MARGEN - 12)
        self.reporte.pack(anchor="w", padx=6, pady=(0, 10))

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
            mundo = Mundo.desde_archivo(ruta)
        except Exception as ex:
            messagebox.showerror("Archivo inválido", str(ex))
            return
        self.mundo, self.ruta_mundo = mundo, ruta
        if self.portada.winfo_ismapped():       # primera carga: pasar de la portada al juego
            self.mostrar_juego()
        self.reiniciar()

    def reiniciar(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        if self.mundo:
            self.reporte.config(text="")
            self.dibujar(self.mundo.inicial(), 0.0)

    def cargar_sprites(self):
        """Carga sprites/<Nombre>.png ampliados con zoom entero (se ven nítidos).
        Si falta alguno, esa casilla se dibuja con colores como respaldo."""
        self.img = {}
        for nombre in NOMBRES_SPRITES:
            ruta = os.path.join(RUTA_SPRITES, nombre + ".png")
            try:
                self.img[nombre] = tk.PhotoImage(file=ruta).zoom(ZOOM)
            except tk.TclError:
                self.img[nombre] = None

        # Imagen de portada (se reduce si es demasiado ancha)
        try:
            portada = tk.PhotoImage(file=os.path.join(RUTA_SPRITES, ARCHIVO_PORTADA))
            if portada.width() > PORTADA_ANCHO_MAX:
                portada = portada.subsample(math.ceil(portada.width() / PORTADA_ANCHO_MAX))
            self.img_portada = portada
        except tk.TclError:
            self.img_portada = None

    def _celda(self, x, y, nombre, color, txt=""):
        """Dibuja un sprite o, si no existe, un rectángulo de color."""
        if self.img.get(nombre):
            self.canvas.create_image(x, y, image=self.img[nombre], anchor="nw")
        else:
            self.canvas.create_rectangle(x, y, x + CELDA, y + CELDA, fill=color, outline="#999")
            if txt:
                self.canvas.create_text(x + CELDA / 2, y + CELDA / 2, text=txt, fill="white",
                                        font=("Arial", 14, "bold"))

    def dibujar(self, e, g):
        m, cv = self.mundo, self.canvas
        cv.delete("all")
        for r in range(m.filas):
            for c in range(m.cols):
                v, x, y = m.grid[r][c], MARGEN + c * CELDA, MARGEN + r * CELDA
                if v == 1:
                    self._celda(x, y, "Muro", "#333333")
                    continue
                self._celda(x, y, "Piso", "white")          # fondo de toda casilla libre
                if v == 3:
                    if (r, c) in e.secos:
                        self._celda(x, y, "CultivoSeco", "#8B5A2B", "C")
                    else:
                        self._celda(x, y, "CultivoFertilizado", "#4CAF50", "C")
                elif v == 4:
                    self._celda(x, y, "Fertilizante", "#4A90D9", "D")
                elif v == 5 and e.av_disp:
                    self._celda(x, y, "Avioneta", "#F2C94C", "A")
        x, y = MARGEN + e.pos[1] * CELDA, MARGEN + e.pos[0] * CELDA
        if e.en_av:
            if self.img.get("AgrobotSostenidoPorAvioneta"):
                self._celda(x, y, "AgrobotSostenidoPorAvioneta", "")
            else:
                cv.create_oval(x + 6, y + 6, x + CELDA - 6, y + CELDA - 6, fill="#FF8C00", outline="black", width=2)
        elif self.img.get("Agrobot"):
            self._celda(x, y, "Agrobot", "")
        else:
            cv.create_oval(x + 6, y + 6, x + CELDA - 6, y + CELDA - 6, fill="#D32F2F", outline="black", width=2)
        self.estado_lbl.config(
            text=f"Fertilizante: {e.fert}   Combustible avioneta: {e.comb if e.en_av else 0}   "
                 f"En avioneta: {'sí' if e.en_av else 'no'}   Costo acumulado: {g:.1f}")

    # ----------------------------------------------------------------- REPORTE
    def guardar_reporte(self, alg, res):
        """Genera el reporte como archivo .txt en la carpeta 'reportes'. Devuelve la ruta."""
        os.makedirs(CARPETA_REPORTES, exist_ok=True)
        nombre = alg.replace("*", "Estrella").replace(" ", "_")
        marca = time.strftime("%Y%m%d_%H%M%S")
        ruta = os.path.join(CARPETA_REPORTES, f"reporte_{nombre}_{marca}.txt")

        lineas = [
            "AgroBot - Reporte de ejecución:",
            "=" * 40,
            f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mundo: {os.path.basename(self.ruta_mundo)}",
            f"Algoritmo: {alg}",
            f"Nodos expandidos: {res['nodos']}",
            f"Profundidad del árbol: {res['prof']}",
            f"Tiempo de cómputo: {res['tiempo']:.4f} s",
        ]
        if res["camino"] is None:
            lineas.append("Resultado: No se encontró la solución.")
        else:
            lineas.append(f"Pasos de la solución: {len(res['camino']) - 1}")
            if alg in ("Costo uniforme", "A*"):
                lineas.append(f"Costo de la solución: {res['costo']:.1f}")
            recorrido = " -> ".join(f"({e.pos[0]},{e.pos[1]})" for e, _ in res["camino"])
            lineas += ["", "Recorrido (fila,columna):", recorrido]

        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")
        return ruta

    def ejecutar(self):
        if not self.mundo:
            messagebox.showinfo("AgroBot", "Primero cargá un mundo rey.")
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
        else:
            txt += f"\nPasos de la solución: {len(res['camino']) - 1}"
            if alg in ("Costo uniforme", "A*"):
                txt += f"\nCosto de la solución: {res['costo']:.1f}"

        try:
            ruta = self.guardar_reporte(alg, res)
            txt += f"\nReporte guardado en: {ruta}"
        except OSError as ex:
            txt += f"\nNo se pudo guardar el reporte: {ex}"
        self.reporte.config(text=txt)
        self.ajustar_ventana()

        if res["camino"] is None:
            return
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
    try:
        icono = tk.PhotoImage(file=os.path.join(RUTA_SPRITES, "Agrobot.png"))
        raiz.iconphoto(True, icono)
    except tk.TclError:
        pass                                   # si no existe, queda el icono por defecto
    App(raiz)
    raiz.mainloop()