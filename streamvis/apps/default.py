from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, CustomJS, Spacer, Title

import streamvis as sv

doc = curdoc()

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 800 + 55
MAIN_CANVAS_HEIGHT = 800 + 60

ZOOM_CANVAS_WIDTH = 600 + 55
ZOOM_CANVAS_HEIGHT = 600 + 30

APP_FPS = 1


# Main plot
sv_mainview = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)

# ---- add zoom plot
sv_zoomview = sv.ImageView(plot_height=ZOOM_CANVAS_HEIGHT, plot_width=ZOOM_CANVAS_WIDTH)

sv_mainview.add_as_zoom(sv_zoomview)

sv_zoom_proj_v = sv.Projection(sv_zoomview, 'vertical')
sv_zoom_proj_h = sv.Projection(sv_zoomview, 'horizontal')


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.location = (0, -5)
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place='below')


# Add mask to all plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview])


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview])


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=400, plot_width=700)


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(nplots=2, plot_height=200, plot_width=700, rollover=36000)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")


# Open statistics button
open_stats_button = Button(label='Open Statistics')
open_stats_button.js_on_click(CustomJS(code="window.open('/statistics');"))


# Stream toggle button
sv_streamctrl = sv.StreamControl()


# Image processor
sv_image_processor = sv.ImageProcessor()


# Metadata datatable
sv_metadata = sv.MetadataHandler()


# Final layouts
colormap_panel = column(
    sv_colormapper.select,
    sv_colormapper.scale_radiobuttongroup,
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
)

layout_zoom = gridplot(
    [[sv_zoom_proj_v.plot, None], [sv_zoomview.plot, sv_zoom_proj_h.plot]], merge_tools=False
)

layout_utility = column(
    gridplot(
        sv_streamgraph.plots, ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)
    ),
    row(
        sv_streamgraph.moving_average_spinner,
        column(Spacer(height=19), sv_streamgraph.reset_button),
    ),
)

layout_controls = column(
    colormap_panel,
    sv_mask.toggle,
    open_stats_button,
    sv_intensity_roi.toggle,
    sv_streamctrl.datatype_select,
    sv_streamctrl.toggle,
)

layout_threshold_aggr = column(
    sv_image_processor.threshold_toggle,
    sv_image_processor.threshold_max_spinner,
    sv_image_processor.threshold_min_spinner,
    Spacer(height=30),
    sv_image_processor.aggregate_toggle,
    sv_image_processor.aggregate_time_spinner,
    sv_image_processor.aggregate_time_counter_textinput,
)

layout_metadata = column(
    sv_metadata.datatable, row(sv_metadata.show_all_toggle, sv_metadata.issues_dropdown)
)

final_layout = column(
    row(sv_mainview.plot, layout_controls, column(layout_metadata, layout_utility)),
    row(layout_zoom, layout_threshold_aggr, sv_hist.plots[0]),
)

doc.add_root(final_layout)


async def update_client(image, metadata, reset, aggr_image):
    sv_colormapper.update(aggr_image)
    sv_mainview.update(aggr_image)

    sv_zoom_proj_v.update(aggr_image)
    sv_zoom_proj_h.update(aggr_image)

    # Statistics
    y_start = int(np.floor(sv_zoomview.y_start))
    y_end = int(np.ceil(sv_zoomview.y_end))
    x_start = int(np.floor(sv_zoomview.x_start))
    x_end = int(np.ceil(sv_zoomview.x_end))

    im_block = aggr_image[y_start:y_end, x_start:x_end]
    total_sum_zoom = np.sum(im_block)

    # Update histogram
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if reset:
            sv_hist.update([aggr_image])
        else:
            im_block = image[y_start:y_end, x_start:x_end]
            sv_hist.update([im_block], accumulate=True)

    # Update total intensities plots
    sv_streamgraph.update([np.sum(aggr_image, dtype=np.float), total_sum_zoom])

    # Parse and update metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update mask
    sv_mask.update(sv_metadata)

    sv_intensity_roi.update(metadata, sv_metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        sv_rt.current_metadata, sv_rt.current_image = sv_streamctrl.get_stream_data(-1)
        sv_rt.aggregated_image, sv_rt.reset = sv_image_processor.update(sv_rt.current_image)

    if sv_rt.current_image.shape != (1, 1):
        doc.add_next_tick_callback(
            partial(
                update_client,
                image=sv_rt.current_image,
                metadata=sv_rt.current_metadata,
                reset=sv_rt.reset,
                aggr_image=sv_rt.aggregated_image,
            )
        )


doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)