import bottleneck as bn
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Spacer, Title, Div

import streamvis as sv

doc = curdoc()

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 1030 + 53
MAIN_CANVAS_HEIGHT = 514 + 67

ZOOM_CANVAS_WIDTH = 514 + 53
ZOOM_CANVAS_HEIGHT = 514 + 28


# Create streamvis components
sv_main = sv.ImageView(
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    image_height=514,
    image_width=1030,
)

sv_zoom = sv.ImageView(
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    image_height=514,
    image_width=1030,
    x_start=258,
    x_end=772,
)

sv_zoom.proj_toggle = sv_main.proj_toggle
sv_main.add_as_zoom(sv_zoom)

sv_colormapper = sv.ColorMapper([sv_main, sv_zoom])
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_main.plot.add_layout(sv_colormapper.color_bar, place="below")

sv_saturated_pixels = sv.SaturatedPixels([sv_main, sv_zoom])

sv_hist = sv.Histogram(nplots=2, plot_height=200, plot_width=700)
sv_hist.plots[0].title = Title(text="Full image")
sv_hist.plots[1].title = Title(text="Roi")
sv_hist.auto_toggle.width = 200

sv_streamgraph = sv.StreamGraph(nplots=2, plot_height=200, plot_width=700)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")

sv_streamctrl = sv.StreamControl()

sv_image_processor = sv.ImageProcessor()

sv_metadata = sv.MetadataHandler(datatable_width=500)
sv_metadata.issues_datatable.height = 100


# Final layouts
layout_utility = column(
    gridplot(
        sv_streamgraph.plots, ncols=1, toolbar_location="left", toolbar_options=dict(logo=None)
    ),
    row(
        sv_streamgraph.moving_average_spinner,
        column(Spacer(height=19), sv_streamgraph.reset_button),
    ),
)

show_overlays_div = Div(text="Show Overlays:")

layout_controls = column(
    row(sv_image_processor.threshold_min_spinner, sv_image_processor.threshold_max_spinner),
    sv_image_processor.threshold_toggle,
    row(
        sv_image_processor.aggregate_time_spinner,
        sv_image_processor.aggregate_time_counter_textinput,
    ),
    sv_image_processor.aggregate_toggle,
    doc.stats.auxiliary_apps_dropdown,
    row(sv_colormapper.select, sv_colormapper.high_color, sv_colormapper.mask_color),
    sv_colormapper.scale_radiobuttongroup,
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    sv_colormapper.auto_toggle,
    show_overlays_div,
    row(sv_saturated_pixels.toggle, sv_main.proj_toggle),
    sv_streamctrl.datatype_select,
    sv_streamctrl.conv_opts_cbbg,
    sv_streamctrl.toggle,
)

layout_metadata = column(
    sv_metadata.issues_datatable, sv_metadata.datatable, row(sv_metadata.show_all_toggle)
)

layout_hist = column(
    gridplot(sv_hist.plots, ncols=1, toolbar_location="left", toolbar_options=dict(logo=None)),
    row(
        column(Spacer(height=19), sv_hist.auto_toggle),
        sv_hist.lower_spinner,
        sv_hist.upper_spinner,
        sv_hist.nbins_spinner,
    ),
)

final_layout = column(
    row(sv_main.plot, sv_zoom.plot, Spacer(width=30), layout_controls),
    row(layout_metadata, layout_utility, layout_hist),
)

doc.add_root(final_layout)


async def internal_periodic_callback():
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        sv_rt.metadata, sv_rt.image = sv_streamctrl.get_stream_data(-1)
        sv_rt.thresholded_image, sv_rt.aggregated_image, sv_rt.reset = sv_image_processor.update(
            sv_rt.image
        )

    if sv_rt.image.shape == (1, 1):
        # skip client update if the current image is dummy
        return

    _, metadata = sv_rt.image, sv_rt.metadata
    thr_image, reset, aggr_image = sv_rt.thresholded_image, sv_rt.reset, sv_rt.aggregated_image

    sv_colormapper.update(aggr_image)
    sv_main.update(aggr_image)

    # Statistics
    im_block = aggr_image[sv_zoom.y_start : sv_zoom.y_end, sv_zoom.x_start : sv_zoom.x_end]
    total_sum_zoom = bn.nansum(im_block)

    # Deactivate auto histogram range if aggregation is on
    if sv_image_processor.aggregate_toggle.active:
        sv_hist.auto_toggle.active = False

    # Update histogram
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if reset:
            sv_hist.update([thr_image, im_block])
        else:
            im_block = thr_image[sv_zoom.y_start : sv_zoom.y_end, sv_zoom.x_start : sv_zoom.x_end]
            sv_hist.update([thr_image, im_block], accumulate=True)

    # Update total intensities plots
    sv_streamgraph.update([bn.nansum(aggr_image), total_sum_zoom])

    # Parse and update metadata
    metadata_toshow = sv_metadata.parse(metadata)

    sv_saturated_pixels.update(metadata)

    sv_metadata.update(metadata_toshow)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
