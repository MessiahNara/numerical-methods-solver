import math
import re
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def build_fx(eq_str):
    x = sp.symbols("x")
    eq_str = eq_str.replace("^", "**")
    expr = sp.sympify(eq_str)
    return sp.lambdify(x, expr, "numpy")


def build_fx_and_derivative(eq_str):
    x = sp.symbols("x")
    eq_str = eq_str.replace("^", "**")
    expr = sp.sympify(eq_str)
    derivative = sp.diff(expr, x)
    return sp.lambdify(x, expr, "numpy"), sp.lambdify(x, derivative, "numpy")


def build_fxy(eq_str):
    expr = eq_str.strip().replace("^", "**")
    expr = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)
    expr = re.sub(r"([a-zA-Z)])(\d)", r"\1*\2", expr)

    math_names = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }

    def f(x, y):
        local_names = math_names.copy()
        local_names["x"] = x
        local_names["y"] = y
        return eval(expr, {"__builtins__": {}}, local_names)

    f(1.0, 1.0)
    return f


def bisection_solver(f, a, b, tol, max_iter):
    steps = []

    if f(a) * f(b) >= 0:
        return steps, "Error: Root not bracketed."

    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(b - a)

        steps.append((a, b, c, fc, error))

        if error < tol or fc == 0:
            return steps, "Converged!"

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return steps, "Did not converge."


def secant_solver(f, x0, x1, tol, max_iter):
    steps = []

    for _ in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)

        if abs(fx1 - fx0) < 1e-12:
            return steps, "Error: Denominator is zero."

        next_x = x1 - fx1 * ((x1 - x0) / (fx1 - fx0))
        error = abs(next_x - x1)

        steps.append((x0, x1, fx1, next_x, error))

        if error < tol:
            return steps, "Converged!"

        x0 = x1
        x1 = next_x

    return steps, "Did not converge."


def newton_solver(f, df, x0, tol, max_iter):
    steps = []
    current_x = x0

    for _ in range(max_iter):
        fx = f(current_x)
        dfx = df(current_x)

        if abs(dfx) < 1e-12:
            return steps, "Error: Derivative too small."

        next_x = current_x - (fx / dfx)
        error = abs(next_x - current_x)

        steps.append((current_x, fx, dfx, next_x, error))

        if error < tol:
            return steps, "Converged!"

        current_x = next_x

    return steps, "Did not converge."


def euler_solver(f, x0, y0, x_end, h, max_steps):
    if h <= 0:
        return [], "Error: Step size must be positive."

    if x_end <= x0:
        return [], "Error: Final x must be greater than initial x."

    steps = []
    current_x = x0
    current_y = y0

    for _ in range(max_steps):
        if current_x >= x_end - 1e-12:
            return steps, "Converged!"

        step_h = min(h, x_end - current_x)
        slope = f(current_x, current_y)

        next_x = current_x + step_h
        next_y = current_y + step_h * slope
        dy = next_y - current_y

        steps.append((current_x, current_y, slope, step_h, next_x, next_y, dy))

        current_x = next_x
        current_y = next_y

    return steps, "Did not converge."


class NumericalMethodsSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Solver")
        self.root.geometry("1320x760")
        self.root.minsize(1150, 680)
        self.root.configure(bg="#e9eef5")

        self.colors = {
            "Bisection": "#d62728",
            "Newton": "#1f77b4",
            "Secant": "#2ca02c",
            "Euler": "#9467bd",
        }

        self.pages = {}
        self.nav_buttons = {}

        self.main = tk.Frame(root, bg="#e9eef5")
        self.main.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(self.main, bg="#111827", width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = tk.Frame(self.main, bg="#e9eef5")
        self.content.pack(side="right", fill="both", expand=True)

        self.setup_styles()
        self.create_pages()
        self.create_sidebar()
        self.show_page("Bisection")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        for name, color in self.colors.items():
            style.configure(
                f"{name}.Treeview",
                font=("Consolas", 10),
                rowheight=28,
                borderwidth=1,
                relief="solid",
                background="white",
                fieldbackground="white"
            )

            style.configure(
                f"{name}.Treeview.Heading",
                font=("Arial", 10, "bold"),
                background=color,
                foreground="white",
                relief="flat"
            )

            style.map(
                f"{name}.Treeview",
                background=[("selected", "#dbeafe")]
            )

    def create_sidebar(self):
        tk.Label(
            self.sidebar,
            text="Methods",
            bg="#111827",
            fg="white",
            font=("Arial", 20, "bold")
        ).pack(anchor="w", padx=22, pady=(24, 24))

        tk.Label(
            self.sidebar,
            text="ROOT FINDER",
            bg="#111827",
            fg="#9ca3af",
            font=("Arial", 9, "bold")
        ).pack(anchor="w", padx=22, pady=(4, 6))

        self.add_nav_button("Bisection", "Bisection")
        self.add_nav_button("Newton", "Newton")
        self.add_nav_button("Secant", "Secant")

        tk.Label(
            self.sidebar,
            text="ODE",
            bg="#111827",
            fg="#9ca3af",
            font=("Arial", 9, "bold")
        ).pack(anchor="w", padx=22, pady=(24, 6))

        self.add_nav_button("Euler", "Euler")

    def add_nav_button(self, text, page_name):
        button = tk.Button(
            self.sidebar,
            text=text,
            bg="#243244",
            fg="white",
            activebackground=self.colors[page_name],
            activeforeground="white",
            relief="flat",
            font=("Arial", 11, "bold"),
            command=lambda: self.show_page(page_name)
        )
        button.pack(fill="x", padx=18, pady=5, ipady=8)
        self.nav_buttons[page_name] = button

    def show_page(self, page_name):
        for page in self.pages.values():
            page.pack_forget()

        self.pages[page_name].pack(fill="both", expand=True)

        for name, button in self.nav_buttons.items():
            button.config(bg=self.colors[name] if name == page_name else "#243244")

    def create_pages(self):
        self.create_bisection_page()
        self.create_newton_page()
        self.create_secant_page()
        self.create_euler_page()

    def create_layout(self, page_name, title, subtitle):
        page = tk.Frame(self.content, bg="#e9eef5")
        self.pages[page_name] = page

        color = self.colors[page_name]

        header = tk.Frame(page, bg=color, height=88)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text=title,
            bg=color,
            fg="white",
            font=("Arial", 22, "bold")
        ).pack(anchor="w", padx=26, pady=(14, 0))

        tk.Label(
            header,
            text=subtitle,
            bg=color,
            fg="white",
            font=("Arial", 10)
        ).pack(anchor="w", padx=26)

        body = tk.Frame(page, bg="#e9eef5")
        body.pack(fill="both", expand=True, padx=18, pady=18)

        left = tk.Frame(body, bg="white", padx=16, pady=16, bd=1, relief="solid")
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)
        left.configure(width=500)

        right = tk.Frame(body, bg="white", padx=10, pady=10, bd=1, relief="solid")
        right.pack(side="right", fill="both", expand=True)

        figure = Figure(figsize=(6, 5), dpi=100)
        ax = figure.add_subplot(111)

        canvas = FigureCanvasTkAgg(figure, right)
        NavigationToolbar2Tk(canvas, right)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        return left, ax, canvas

    def add_input_grid(self, parent, fields):
        entries = {}

        input_frame = tk.Frame(parent, bg="white")
        input_frame.pack(fill="x")

        for index, (key, label, default) in enumerate(fields):
            row = index // 2
            col = index % 2

            block = tk.Frame(input_frame, bg="white")
            block.grid(row=row, column=col, sticky="ew", padx=(0, 10), pady=(0, 8))

            tk.Label(
                block,
                text=label,
                bg="white",
                fg="#111827",
                font=("Arial", 9, "bold")
            ).pack(anchor="w")

            entry = tk.Entry(
                block,
                font=("Consolas", 10),
                relief="solid",
                bd=1
            )
            entry.insert(0, default)
            entry.pack(fill="x")

            entries[key] = entry

        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)

        return entries

    def create_button(self, parent, text, color, command):
        button = tk.Button(
            parent,
            text=text,
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief="flat",
            font=("Arial", 11, "bold"),
            command=command
        )
        button.pack(fill="x", pady=(8, 12), ipady=7)

    def create_table(self, parent, page_name):
        color = self.colors[page_name]

        tk.Label(
            parent,
            text="Iteration Table",
            bg="white",
            fg="#111827",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", pady=(4, 4))

        status_label = tk.Label(
            parent,
            text="STATUS: Waiting for calculation",
            bg="white",
            fg=color,
            font=("Consolas", 10, "bold")
        )
        status_label.pack(anchor="w", pady=(0, 5))

        table_frame = tk.Frame(parent, bg="white", bd=1, relief="solid")
        table_frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(
            table_frame,
            show="headings",
            style=f"{page_name}.Treeview"
        )

        y_scroll = tk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        x_scroll = tk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)

        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        tree.tag_configure("even", background="#ffffff")
        tree.tag_configure("odd", background="#f3f4f6")

        return status_label, tree

    def fill_table(self, tree, status_label, status, color, columns, rows):
        status_label.config(text=f"STATUS: {status}", fg=color)

        tree.delete(*tree.get_children())
        tree["columns"] = columns

        total_width = 450
        step_width = 60
        other_width = max(95, int((total_width - step_width) / max(1, len(columns) - 1)))

        for column in columns:
            tree.heading(column, text=column)

            if column == "Step":
                tree.column(
                    column,
                    width=step_width,
                    minwidth=step_width,
                    anchor="center",
                    stretch=False
                )
            else:
                tree.column(
                    column,
                    width=other_width,
                    minwidth=90,
                    anchor="e",
                    stretch=True
                )

        for index, row in enumerate(rows):
            tag = "even" if index % 2 == 0 else "odd"
            tree.insert("", "end", values=row, tags=(tag,))

    def create_bisection_page(self):
        left, ax, canvas = self.create_layout(
            "Bisection",
            "Root Finder - Bisection Method",
            "Finds a root by repeatedly cutting an interval in half."
        )

        entries = self.add_input_grid(left, [
            ("eq", "Equation f(x)", "x^3 - x - 2"),
            ("a", "Left Bound (a)", "1.0"),
            ("b", "Right Bound (b)", "2.0"),
            ("acc", "Decimals", "6"),
            ("max_iter", "Max Iterations", "100"),
        ])

        self.create_button(
            left,
            "Calculate and Graph",
            self.colors["Bisection"],
            lambda: self.solve_bisection(entries, status_label, table, ax, canvas)
        )

        status_label, table = self.create_table(left, "Bisection")

    def solve_bisection(self, entries, status_label, table, ax, canvas):
        try:
            f = build_fx(entries["eq"].get())
            a = float(entries["a"].get())
            b = float(entries["b"].get())
            decimals = int(entries["acc"].get())
            max_iter = int(entries["max_iter"].get())

            steps, status = bisection_solver(f, a, b, 10 ** (-decimals), max_iter)

            rows = []
            for i, (av, bv, cv, fc, err) in enumerate(steps, start=1):
                rows.append([
                    i,
                    f"{av:.{decimals}f}",
                    f"{bv:.{decimals}f}",
                    f"{cv:.{decimals}f}",
                    f"{fc:.{decimals}f}",
                    f"{err:.{decimals}f}",
                ])

            self.fill_table(
                table,
                status_label,
                status,
                self.colors["Bisection"],
                ["Step", "a", "b", "c", "f(c)", "Error"],
                rows
            )

            ax.clear()
            x_vals = np.linspace(a - 1, b + 1, 400)
            ax.plot(x_vals, f(x_vals), color="#1f77b4", label="f(x)")
            ax.axhline(0, color="black")
            ax.axvline(a, color="#d62728", linestyle="--", label="a and b")
            ax.axvline(b, color="#d62728", linestyle="--")

            if steps:
                ax.plot(steps[-1][2], 0, "go", label="Approx Root")

            ax.set_title("Bisection Method Visualization")
            ax.grid(True, linestyle="--")
            ax.legend()
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_newton_page(self):
        left, ax, canvas = self.create_layout(
            "Newton",
            "Root Finder - Newton's Method",
            "Finds a root using one starting guess and the derivative."
        )

        entries = self.add_input_grid(left, [
            ("eq", "Equation f(x)", "x^3 - x - 2"),
            ("x0", "Initial Guess (x0)", "-2"),
            ("acc", "Decimals", "6"),
            ("max_iter", "Max Iterations", "100"),
        ])

        self.create_button(
            left,
            "Calculate and Graph",
            self.colors["Newton"],
            lambda: self.solve_newton(entries, status_label, table, ax, canvas)
        )

        status_label, table = self.create_table(left, "Newton")

    def solve_newton(self, entries, status_label, table, ax, canvas):
        try:
            f, df = build_fx_and_derivative(entries["eq"].get())
            x0 = float(entries["x0"].get())
            decimals = int(entries["acc"].get())
            max_iter = int(entries["max_iter"].get())

            steps, status = newton_solver(f, df, x0, 10 ** (-decimals), max_iter)

            rows = []
            for i, (xn, fxn, dfx, xnext, err) in enumerate(steps, start=1):
                rows.append([
                    i,
                    f"{xn:.{decimals}f}",
                    f"{fxn:.{decimals}f}",
                    f"{dfx:.{decimals}f}",
                    f"{xnext:.{decimals}f}",
                    f"{err:.{decimals}f}",
                ])

            self.fill_table(
                table,
                status_label,
                status,
                self.colors["Newton"],
                ["Step", "x", "f(x)", "f'(x)", "Next x", "Error"],
                rows
            )

            ax.clear()

            if steps:
                xs = [s[0] for s in steps] + [s[3] for s in steps]
                x_vals = np.linspace(min(xs) - 2, max(xs) + 2, 400)

                ax.plot(x_vals, f(x_vals), color="#1f77b4", label="f(x)")
                ax.axhline(0, color="black")

                for xn, fxn, dfx, xnext, err in steps[:5]:
                    ax.plot(xn, fxn, "ro")
                    ax.plot([xn, xnext], [fxn, 0], "k--", alpha=0.5)

            ax.set_title("Newton's Method Visualization")
            ax.grid(True, linestyle="--")
            ax.legend()
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_secant_page(self):
        left, ax, canvas = self.create_layout(
            "Secant",
            "Root Finder - Secant Method",
            "Finds a root using two starting guesses."
        )

        entries = self.add_input_grid(left, [
            ("eq", "Equation f(x)", "x^3 - x - 2"),
            ("x0", "x0", "1.0"),
            ("x1", "x1", "2.0"),
            ("acc", "Decimals", "6"),
            ("max_iter", "Max Iterations", "100"),
        ])

        self.create_button(
            left,
            "Calculate and Graph",
            self.colors["Secant"],
            lambda: self.solve_secant(entries, status_label, table, ax, canvas)
        )

        status_label, table = self.create_table(left, "Secant")

    def solve_secant(self, entries, status_label, table, ax, canvas):
        try:
            f = build_fx(entries["eq"].get())
            x0 = float(entries["x0"].get())
            x1 = float(entries["x1"].get())
            decimals = int(entries["acc"].get())
            max_iter = int(entries["max_iter"].get())

            steps, status = secant_solver(f, x0, x1, 10 ** (-decimals), max_iter)

            rows = []
            for i, (xn_1, xn, fxn, xnext, err) in enumerate(steps, start=1):
                rows.append([
                    i,
                    f"{xn_1:.{decimals}f}",
                    f"{xn:.{decimals}f}",
                    f"{fxn:.{decimals}f}",
                    f"{xnext:.{decimals}f}",
                    f"{err:.{decimals}f}",
                ])

            self.fill_table(
                table,
                status_label,
                status,
                self.colors["Secant"],
                ["Step", "x old", "x", "f(x)", "Next x", "Error"],
                rows
            )

            ax.clear()

            if steps:
                xs = [s[0] for s in steps] + [s[1] for s in steps] + [s[3] for s in steps]
                x_vals = np.linspace(min(xs) - 2, max(xs) + 2, 400)

                ax.plot(x_vals, f(x_vals), color="#1f77b4", label="f(x)")
                ax.axhline(0, color="black")

                for xn_1, xn, fxn, xnext, err in steps[:5]:
                    ax.plot(xn, fxn, "ro")
                    ax.plot([xn_1, xnext], [f(xn_1), 0], "k--", alpha=0.5)

            ax.set_title("Secant Method Visualization")
            ax.grid(True, linestyle="--")
            ax.legend()
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_euler_page(self):
        left, ax, canvas = self.create_layout(
            "Euler",
            "ODE - Euler's Method",
            "Approximates y values for dy/dx = f(x,y)."
        )

        entries = self.add_input_grid(left, [
            ("eq", "dy/dx = f(x,y)", "x + y"),
            ("x0", "Initial x", "0"),
            ("y0", "Initial y", "1"),
            ("x_end", "Final x", "2"),
            ("h", "Step Size h", "0.1"),
            ("acc", "Decimals", "6"),
            ("max_steps", "Max Steps", "100"),
        ])

        self.create_button(
            left,
            "Calculate and Graph",
            self.colors["Euler"],
            lambda: self.solve_euler(entries, status_label, table, ax, canvas)
        )

        status_label, table = self.create_table(left, "Euler")

    def solve_euler(self, entries, status_label, table, ax, canvas):
        try:
            f = build_fxy(entries["eq"].get())
            x0 = float(entries["x0"].get())
            y0 = float(entries["y0"].get())
            x_end = float(entries["x_end"].get())
            h = float(entries["h"].get())
            decimals = int(entries["acc"].get())
            max_steps = int(entries["max_steps"].get())

            steps, status = euler_solver(f, x0, y0, x_end, h, max_steps)

            rows = []
            for i, (xn, yn, slope, step_h, xnext, ynext, dy) in enumerate(steps):
                rows.append([
                    i,
                    f"{xn:.{decimals}f}",
                    f"{yn:.{decimals}f}",
                    f"{slope:.{decimals}f}",
                    f"{dy:.{decimals}f}",
                ])

            self.fill_table(
                table,
                status_label,
                status,
                self.colors["Euler"],
                ["Step", "x", "y", "f(x,y)", "dy"],
                rows
            )

            ax.clear()

            if steps:
                x_vals = [x0] + [s[4] for s in steps]
                y_vals = [y0] + [s[5] for s in steps]

                ax.plot(
                    x_vals,
                    y_vals,
                    "o-",
                    color="#1f77b4",
                    label="Euler approximation"
                )

                for index, (xn, yn, slope, step_h, xnext, ynext, dy) in enumerate(steps):
                    ax.plot(
                        [xn, xnext],
                        [yn, yn + slope * step_h],
                        color="#f97316",
                        linestyle="--",
                        alpha=0.75,
                        label="Tangent / slope line" if index == 0 else ""
                    )

                ax.plot(x_vals[-1], y_vals[-1], "ro", label="Final point")
            else:
                ax.plot([x0], [y0], "ro", label="Initial point")

            ax.set_title("Euler's Method Visualization")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, linestyle="--")
            ax.legend()
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = NumericalMethodsSolver(root)
    root.mainloop()