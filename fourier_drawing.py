from manim import *
from typing import Any, Dict, List, Callable, Literal, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from math import pi


def preview(X: np.ndarray, Y: np.ndarray, T: np.ndarray, file_save_path: str):
    sns.scatterplot(
        x=X,
        y=Y,
        hue=T,
        palette=sns.color_palette('mako', as_cmap=True)
    )
    plt.axis('scaled')
    plt.savefig(file_save_path)


def get_points(
    svg_file: str,
    total_num_of_points: int = 100,
    centre_kind: Union[Literal['mass'], Literal['midpoint']] = 'midpoint',
    flip_vertically: bool = True,
):
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
        num_of_points = round(p.length() / total_length *
                              (total_num_of_points + num_of_paths))
        final.extend([p.point(t) for t in np.linspace(
            start=0, stop=1, num=num_of_points)][:-1])

    final_arr = np.array(final)

    # Centre the data points
    X = np.real(final_arr)
    x_centre = (X.max() + X.min()) / \
        2 if centre_kind == 'midpoint' else X.mean()
    X -= x_centre

    Y = np.imag(final_arr)
    y_centre = (Y.max() + Y.min()) / \
        2 if centre_kind == 'midpoint' else Y.mean()
    Y -= y_centre

    if flip_vertically:
        Y *= -1

    # Normalize the data points
    mx = max(X.max(), Y.max())

    X /= mx
    Y /= mx

    # Generate evenly spaced time intervals
    T = np.linspace(start=0, stop=2 * pi, num=len(X))

    return X, Y, T


def get_complex_coefficients(
    svg_file: str,
    max_vector_fr: int = 5,
    total_num_of_points: int = 100,
    centre_kind: Union[Literal['mass'], Literal['midpoint']] = 'midpoint',
    save_point_plot: str = None,
    vector_sort_fn: Union[str, Callable[[
        pd.DataFrame], pd.Series]] = 'Magnitude',
    vector_sort_order: Union[Literal['asc'], Literal['desc']] = 'desc'
):
    # Retrieve data points
    X, Y, T = get_points(
        svg_file=svg_file,
        total_num_of_points=total_num_of_points,
        centre_kind=centre_kind
    )

    # Preview the data points
    if save_point_plot is not None:
        preview(X=X, Y=Y, T=T, file_save_path=save_point_plot)

    # Numerical Integration
    s = []
    for f in range(-max_vector_fr, max_vector_fr + 1):
        s.append((f, sum([(X[i] + Y[i] * 1j) * np.exp(f * 1j * -1 * T[i])
                          * (2 * pi / len(T)) for i in range(len(X))]) / (2 * pi)))

    # Build the dataframe
    coefs = np.hstack((
        np.array([[i[1].real, i[1].imag] for i in s]),
        np.array([i[0] for i in s]).reshape(-1, 1)
    ))
    df = pd.DataFrame(data=coefs, columns=['Re', 'Im', 'Freq'])

    # Return the sorted dataframe
    if vector_sort_fn is not None:
        if type(vector_sort_fn) is str:
            if vector_sort_fn == 'Magnitude':
                df['sort_sequence'] = np.sqrt(df['Re'] ** 2 + df['Im'] ** 2)
            elif vector_sort_fn == 'Frequency':
                df['sort_sequence'] = np.abs(df['Freq'])
            else:
                raise NotImplementedError
        else:
            df['sort_sequence'] = vector_sort_fn(df)  # type: ignore
        return df.sort_values(by='sort_sequence', ascending=(vector_sort_order == 'asc')).drop(columns='sort_sequence').values
    else:
        return df.values


class DrawingFourierTransform(Scene):
    def preview(
        self,
        svg_file: str,
        save_plot_to: str,
        total_num_of_points: int = 1000,
        centre_kind: Union[Literal['mass'], Literal['midpoint']] = 'midpoint'
    ):
        X, Y, T = get_points(
            svg_file=svg_file,
            total_num_of_points=total_num_of_points,
            centre_kind=centre_kind
        )
        preview(X=X, Y=Y, T=T, file_save_path=save_plot_to)

    def draw(
        self,
        svg_file: str,
        max_vector_fr: int = 5,
        total_num_of_points: int = 1000,
        save_point_plot: str = None,
        cycles_per_s: float = 1.0,
        vector_scale: float = 2.0,
        run_time_s: float = 10.0,
        position: np.ndarray = ORIGIN,
        **kwargs
    ):
        coefs = get_complex_coefficients(
            svg_file=svg_file,
            max_vector_fr=max_vector_fr,
            total_num_of_points=total_num_of_points,
            save_point_plot=save_point_plot,
            **kwargs
        )

        tracker = ValueTracker()

        vecs: List[Vector] = [Vector(direction=[c[0], c[1]])
                              for c in coefs]  # type: ignore

        vecs[0].shift(position)
        vecs[0].add_updater(lambda x: x.rotate(
            angle=coefs[0, 2] * tracker.get_value(), about_point=x.get_start()))  # type:ignore

        for i in range(1, len(vecs)):
            vecs[i].add_updater(lambda v, i=i: v.move_to(vecs[i - 1].get_end() + v.get_center() - v.get_start(
            )).rotate(angle=coefs[i, 2] * tracker.get_value(), about_point=v.get_start()))  # type:ignore

        trace = TracedPath(vecs[-1].get_end)

        freq = cycles_per_s

        self.add(*[v.scale(vector_scale) for v in vecs], trace)
        tracker.increment_value(PI / 30 * freq)
        self.play(*[v.animate for v in vecs], run_time=run_time_s)

    def morph(
        self,
        svg_file: str,
        max_vector_fr: int = 5,
        total_num_of_points: int = 1000,
        save_point_plot: str = None,
        initial_delay_s: float = 0.0,
        delay_btw_transformations_s: float = 0.1,
        delay_each_transformation_s: float = 0.1,
        vector_scale: float = 2.0,
        position: np.ndarray = ORIGIN,
        n_vectors_interval: int = 1,
        outline_colour: np.ndarray = WHITE,
        **kwargs
    ):

        axes = Axes(
            x_range=[-1, 1, 1],
            x_length=2 * vector_scale,
            y_range=[-1, 1, 1],
            y_length=2 * vector_scale
        )

        coefs = get_complex_coefficients(
            svg_file=svg_file,
            total_num_of_points=total_num_of_points,
            max_vector_fr=max_vector_fr,
            save_point_plot=save_point_plot,
            **kwargs
        )

        freqs = []

        for i in range(coefs.shape[0]):
            freqs.append(lambda x, i=i: (
                coefs[i, 0] + coefs[i, 1] * 1j) * np.exp(coefs[i, 2] * x * 1j))

        graphs = []

        for i in range(len(freqs)):
            graphs.append(axes.plot_parametric_curve(lambda t, i=i: np.array([
                np.real([f(t) for f in freqs[:i+1]]).sum(),
                np.imag([f(t) for f in freqs[:i+1]]).sum()
            ]), t_range=[0, 2 * PI], color=outline_colour).shift(position))

        curr_graph = graphs[0]
        self.add(curr_graph)
        self.wait(duration=initial_delay_s)

        for i, g in enumerate(graphs[1::n_vectors_interval]):
            self.wait(duration=delay_btw_transformations_s)
            self.play(Transform(curr_graph, g),
                      run_time=delay_each_transformation_s)
