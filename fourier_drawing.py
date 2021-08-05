from manim import *
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from svgpathtools import svg2paths

def get_complex_coefficients(svg_file: str, N: int = 10, total_num_of_points: int = 100, save_point_plot: str = None, sort_vectors_desc: bool = None):
    svg = svg2paths(svg_file_location=svg_file)

    PATH = svg[0][0]

    assert PATH.isclosed(), 'PATH NOT CLOSED'

    total_length = 0
    num_of_paths = 0

    for p in PATH:
        total_length += p.length()
        num_of_paths += 1

    final = []

    for p in PATH:
        num_of_points = round(p.length() / total_length * (total_num_of_points + num_of_paths))
        final.extend([p.point(t) for t in np.linspace(start=0, stop=1, num=num_of_points)][:-1])

    final_arr = np.array(final)

    X = np.real(final_arr)
    X = X - X.mean()
    Y = np.imag(final_arr)
    Y = Y - Y.mean()
    mx = max(X.max(), Y.max())

    X = X / mx
    Y = Y / mx

    min_fr = -int((N - 1) / 2)
    max_fr = -min_fr if N % 2 == 1 else -min_fr + 1

    from math import pi
    time = np.linspace(start=0, stop=2 * pi, num=len(X))

    s = []
    for f in range(min_fr, max_fr + 1):
        s.append((f, sum([(X[i] + Y[i] * 1j) * np.exp(f * 1j * -1 * time[i]) * (2 * pi / len(time)) for i in range(len(X))]) / (2 * pi)))

    freqs = np.array([i[0] for i in s])
    cs = np.array([[i[1].real, i[1].imag] for i in s])

    coefs = np.hstack((cs, freqs.reshape(-1, 1)))

    df = pd.DataFrame(data=coefs, columns=['Re', 'Im', 'Freq'])

    if save_point_plot is not None:
        sns.scatterplot(x=X, y=Y, hue=time, palette=sns.color_palette('mako', as_cmap=True))
        plt.axis('scaled')
        plt.savefig(save_point_plot)

    if sort_vectors_desc is None:
        return df.values
    else:
        df['Magnitude'] = np.sqrt(df['Re'] ** 2 + df['Im'] ** 2)
        return df.sort_values(by='Magnitude', ascending=(not sort_vectors_desc)).drop(columns='Magnitude').values

class Draw(Scene):

    SETTINGS: Dict[str, Any] = {
        'file_path': './img/cowboy.svg',
        'n_vectors': 401,
        'n_points': 80000,
        'vector_scale': 2.5,
        'cycles_per_s': 1 / 10,
        'run_time_s': 11,
        'plot_save_path': './img/cowboy.png'
    }

    DEFAULT_SETTINGS: Dict[str, Any] = {
        'file_path': None,
        'n_vectors': 11,
        'n_points': 100,
        'plot_save_path': None,
        'run_time_s': 10,
        'vector_scale': 1,
        'cycles_per_s': 1,
        'sort_vectors_desc': True
    }

    def get_settings(self):
        CONFIG: Dict[str, Any] = {}
        for k in self.DEFAULT_SETTINGS:
            if k in self.SETTINGS.keys():
                CONFIG[k] = self.SETTINGS[k]
            else:
                CONFIG[k] = self.DEFAULT_SETTINGS[k]
        return CONFIG

    def construct(self):
        CONFIG = self.get_settings()

        coefs = get_complex_coefficients(svg_file=CONFIG['file_path'], N=CONFIG['n_vectors'], total_num_of_points=CONFIG['n_points'], save_point_plot=CONFIG['plot_save_path'], sort_vectors_desc=CONFIG['sort_vectors_desc'])

        tracker = ValueTracker()

        vecs: List[Vector] = [Vector(direction=[c[0], c[1]]) for c in coefs] # type:ignore

        vecs[0].add_updater(lambda x: x.rotate(angle=coefs[0 ,2] * tracker.get_value(), about_point=x.get_start())) # type:ignore

        for i in range(1, len(vecs)):
            vecs[i].add_updater(lambda v, i=i: v.move_to(vecs[i - 1].get_end() + v.get_center() - v.get_start()).rotate(angle=coefs[i, 2] * tracker.get_value(), about_point=v.get_start())) # type:ignore

        trace = TracedPath(vecs[-1].get_end)

        freq = CONFIG['cycles_per_s']
        # self.play(Write(Text('Writing the letter E').shift(3 * LEFT + 2 * UP)))

        self.add(*[v.scale(CONFIG['vector_scale']) for v in vecs], trace)
        tracker.increment_value(PI / 30 * freq)
        self.play(*[v.animate for v in vecs], run_time=CONFIG['run_time_s'])

class Morph(Scene):
    SETTINGS: Dict[str, Any] = {
        'file_path': './img/e.svg',
        'n_vectors': 50,
        'n_points': 10000,
        'vector_scale': 3,
        'run_time_s': 11,
        'n_vectors_interval': 1,
        'plot_save_path': './img/cowboy.png'
    }

    DEFAULT_SETTINGS: Dict[str, Any] = {
        'file_path': None,
        'n_vectors': 11,
        'n_points': 100,
        'plot_save_path': None,
        'run_time_s': 10,
        'vector_scale': 1,
        'n_vectors_interval': 1,
        'sort_vectors_desc': True
    }

    def get_settings(self):
        CONFIG: Dict[str, Any] = {}
        for k in self.DEFAULT_SETTINGS:
            if k in self.SETTINGS.keys():
                CONFIG[k] = self.SETTINGS[k]
            else:
                CONFIG[k] = self.DEFAULT_SETTINGS[k]
        return CONFIG

    def construct(self):
        CONFIG = self.get_settings()

        scale = CONFIG['vector_scale']
        axes = Axes(
            x_range=[-1, 1, 1],
            x_length=2 * scale,
            y_range=[-1, 1, 1],
            y_length=2 * scale
        )

        coefs = get_complex_coefficients(svg_file=CONFIG['file_path'], N=CONFIG['n_vectors'], total_num_of_points=CONFIG['n_points'], save_point_plot=CONFIG['plot_save_path'], sort_vectors_desc=CONFIG['sort_vectors_desc'])

        freqs = []

        # print(coefs)

        for i in range(coefs.shape[0]):
            freqs.append(lambda x, i=i: (coefs[i, 0] + coefs[i, 1] * 1j) * np.exp(coefs[i, 2] * x * 1j))

        # graph = axes.get_parametric_curve(lambda t: np.array([
        #     np.real([freqs[0](t)]).sum(),
        #     np.imag([freqs[0](t)]).sum()
        # ]), t_range=[0, 2 * PI])

        # graph2 = axes.get_parametric_curve(lambda t: np.array([
        #     np.real([freqs[0](t), freqs[1](t)]).sum(),
        #     np.imag([freqs[0](t), freqs[1](t)]).sum()
        # ]), t_range=[0, 2 * PI])

        graphs = []

        for i in range(len(freqs)):
            graphs.append(axes.get_parametric_curve(lambda t, i=i: np.array([
            np.real([f(t) for f in freqs[:i+1]]).sum(),
            np.imag([f(t) for f in freqs[:i+1]]).sum()
        ]), t_range=[0, 2 * PI]))

        curr_graph = graphs[0]
        curr_num = Text(str(1)).shift(4 * RIGHT + 3 * UP)
        self.add(curr_graph)

        for i, g in enumerate(graphs[1:]):
            # self.wait(duration=0.1)
            self.play(Transform(curr_num, Text(str(i + 2)).shift(4 * RIGHT + 3 * UP)), Transform(curr_graph, g), run_time=1/5)

        # self.wait(duration=1)
        # self.play(Transform(graph, graph2))
        # self.wait(duration=1)