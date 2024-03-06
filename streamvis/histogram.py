import bottleneck as bn
import numpy as np
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    ColumnDataSource,
    DataRange1d,
    Grid,
    LinearAxis,
    PanTool,
    Plot,
    Quad,
    ResetTool,
    SaveTool,
    Spinner,
    Toggle,
    WheelZoomTool,
)

STEP = 1


class Histogram:
    def __init__(self, nplots, plot_height=350, plot_width=700, lower=0, upper=1000, nbins=100):
        """Initialize histogram plots.

        Args:
            nplots (int): Number of histogram plots that will share common controls.
            plot_height (int, optional): Height of plot area in screen pixels. Defaults to 350.
            plot_width (int, optional): Width of plot area in screen pixels. Defaults to 700.
            lower (int, optional): Initial lower range of the bins. Defaults to 0.
            upper (int, optional): Initial upper range of the bins. Defaults to 1000.
            nbins (int, optional): Initial number of the bins. Defaults to 100.
        """
        self._counts = [0 for _ in range(nplots)]

        # Histogram plots
        self.plots = []
        self._plot_sources = []
        for ind in range(nplots):
            plot = Plot(
                x_range=DataRange1d(),
                y_range=DataRange1d(),
                plot_height=plot_height,
                plot_width=plot_width,
                toolbar_location="left",
            )

            # ---- tools
            plot.toolbar.logo = None
            # share 'pan', 'boxzoom', and 'wheelzoom' tools between all plots
            if ind == 0:
                pantool = PanTool()
                boxzoomtool = BoxZoomTool()
                wheelzoomtool = WheelZoomTool()
            plot.add_tools(pantool, boxzoomtool, wheelzoomtool, SaveTool(), ResetTool())

            # ---- axes
            plot.add_layout(LinearAxis(), place="below")
            plot.add_layout(LinearAxis(major_label_orientation="vertical"), place="left")

            # ---- grid lines
            plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
            plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

            # ---- quad (single bin) glyph
            plot_source = ColumnDataSource(dict(left=[], right=[], top=[]))
            plot.add_glyph(
                plot_source,
                Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"),
            )

            self.plots.append(plot)
            self._plot_sources.append(plot_source)

        # Histogram controls
        # ---- histogram range toggle button
        def auto_toggle_callback(state):
            if state:  # Automatic
                lower_spinner.disabled = True
                upper_spinner.disabled = True

            else:  # Manual
                lower_spinner.disabled = False
                upper_spinner.disabled = False

        auto_toggle = Toggle(label="Auto Histogram Range", active=True)
        auto_toggle.on_click(auto_toggle_callback)
        self.auto_toggle = auto_toggle

        # ---- histogram lower range
        def lower_spinner_callback(_attr, _old_value, new_value):
            self.upper_spinner.low = new_value + STEP
            self._counts = [0 for _ in range(nplots)]

        lower_spinner = Spinner(
            title="Lower Range:",
            high=upper - STEP,
            value=lower,
            step=STEP,
            disabled=auto_toggle.active,
            default_size=145,
        )
        lower_spinner.on_change("value", lower_spinner_callback)
        self.lower_spinner = lower_spinner

        # ---- histogram upper range
        def upper_spinner_callback(_attr, _old_value, new_value):
            self.lower_spinner.high = new_value - STEP
            self._counts = [0 for _ in range(nplots)]

        upper_spinner = Spinner(
            title="Upper Range:",
            low=lower + STEP,
            value=upper,
            step=STEP,
            disabled=auto_toggle.active,
            default_size=145,
        )
        upper_spinner.on_change("value", upper_spinner_callback)
        self.upper_spinner = upper_spinner

        # ---- histogram number of bins
        def nbins_spinner_callback(_attr, _old_value, _new_value):
            self._counts = [0 for _ in range(nplots)]

        nbins_spinner = Spinner(title="Number of Bins:", low=1, value=nbins, default_size=145)
        nbins_spinner.on_change("value", nbins_spinner_callback)
        self.nbins_spinner = nbins_spinner

        # ---- histogram log10 of counts toggle button
        def log10counts_toggle_callback(state):
            self._counts = [0 for _ in range(nplots)]
            for plot in self.plots:
                if state:
                    plot.yaxis[0].axis_label = "log⏨(Counts)"
                else:
                    plot.yaxis[0].axis_label = "Counts"

        log10counts_toggle = Toggle(label="log⏨(Counts)", button_type="default", default_size=145)
        log10counts_toggle.on_click(log10counts_toggle_callback)
        self.log10counts_toggle = log10counts_toggle

    @property
    def lower(self):
        """Lower range of the bins (readonly)
        """
        return self.lower_spinner.value

    @property
    def upper(self):
        """Upper range of the bins (readonly)
        """
        return self.upper_spinner.value

    @property
    def nbins(self):
        """Number of the bins (readonly)
        """
        return self.nbins_spinner.value

    def update(self, input_data, accumulate=False):
        """Trigger an update for the histogram plots.

        Args:
            input_data (ndarray): Source values for histogram plots.
            accumulate (bool, optional): Add together bin values of the previous and current data.
                Defaults to False.
        """
        if self.auto_toggle.active and not accumulate:  # automatic
            # find the lowest and the highest value in input data
            lower = 0
            upper = 1

            for data in input_data:
                min_val = bn.nanmin(data)
                min_val = 0 if np.isnan(min_val) else min_val
                lower = min(lower, min_val)

                max_val = bn.nanmax(data)
                max_val = 1 if np.isnan(max_val) else max_val
                upper = max(upper, max_val)

            self.lower_spinner.value = int(np.floor(lower))
            self.upper_spinner.value = int(np.ceil(upper))

        # get histogram counts and update plots
        for data, counts, plot_source in zip(input_data, self._counts, self._plot_sources):
            next_counts, edges = np.histogram(data, bins=self.nbins, range=(self.lower, self.upper))

            if self.log10counts_toggle.active:
                next_counts = np.log10(next_counts, where=next_counts > 0)

            if accumulate:
                counts += next_counts
            else:
                counts = next_counts

            plot_source.data.update(left=edges[:-1], right=edges[1:], top=counts)
