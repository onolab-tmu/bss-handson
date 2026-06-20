import matplotlib.pyplot as plt


def apply_axis_style(ax) -> None:
    ax.tick_params(direction=plt.rcParams["xtick.direction"])
    ax.grid(
        visible=plt.rcParams["axes.grid"],
        linestyle=plt.rcParams["grid.linestyle"],
        linewidth=plt.rcParams["grid.linewidth"],
        alpha=plt.rcParams["grid.alpha"],
    )
