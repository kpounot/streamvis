from functools import partial

import h5py
import jungfrau_utils as ju
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Spacer, Title, Toggle

import streamvis as sv

receiver = sv.receiver.current
doc = curdoc()
doc.title = sv.receiver.args.page_title

# Expected image sizes for the detector
IMAGE_SIZE_X = 1030
IMAGE_SIZE_Y = 1554

current_gain_file = ''
current_pedestal_file = ''
jf_calib = None

sv_rt = sv.Runtime()

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = IMAGE_SIZE_X // 2 + 55 + 40
MAIN_CANVAS_HEIGHT = IMAGE_SIZE_Y // 2 + 86 + 60

ZOOM_CANVAS_WIDTH = 388 + 55
ZOOM_CANVAS_HEIGHT = 388 + 62

DEBUG_INTENSITY_WIDTH = 700

APP_FPS = 1

util_plot_size = 160

ZOOM_WIDTH = 500
ZOOM_HEIGHT = 500

ZOOM1_LEFT = 265
ZOOM1_BOTTOM = 800
ZOOM1_RIGHT = ZOOM1_LEFT + ZOOM_WIDTH
ZOOM1_TOP = ZOOM1_BOTTOM + ZOOM_HEIGHT

ZOOM2_LEFT = 265
ZOOM2_BOTTOM = 200
ZOOM2_RIGHT = ZOOM2_LEFT + ZOOM_WIDTH
ZOOM2_TOP = ZOOM2_BOTTOM + ZOOM_HEIGHT


# Main plot
sv_mainview = sv.ImageView(
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
)

sv_mainview.plot.title = Title(text=' ')

# ---- add zoom plot 1
sv_zoomview1 = sv.ImageView(
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
    x_start=ZOOM1_LEFT,
    x_end=ZOOM1_RIGHT,
    y_start=ZOOM1_BOTTOM,
    y_end=ZOOM1_TOP,
)

sv_zoomview1.plot.title = Title(text='Signal roi', text_color='red')
sv_mainview.add_as_zoom(sv_zoomview1, line_color='red')

# ---- add zoom plot 2
sv_zoomview2 = sv.ImageView(
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
    x_start=ZOOM2_LEFT,
    x_end=ZOOM2_RIGHT,
    y_start=ZOOM2_BOTTOM,
    y_end=ZOOM2_TOP,
)

sv_zoomview2.plot.title = Title(text='Background roi', text_color='green')
sv_mainview.add_as_zoom(sv_zoomview2, line_color='green')


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(
    nplots=2, plot_height=util_plot_size, plot_width=DEBUG_INTENSITY_WIDTH, rollover=36000
)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Normalized signal−background Intensity")
sv_streamgraph.glyphs[1].line_color = 'red'


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview1, sv_zoomview2])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.height = 10
sv_colormapper.color_bar.location = (0, -5)
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place='below')


# Add mask to all plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview1, sv_zoomview2])


# Histogram plots
sv_hist = sv.Histogram(nplots=3, plot_height=300, plot_width=600)
sv_hist.plots[0].title = Title(text="Full image")
sv_hist.plots[1].title = Title(text="Signal roi", text_color='red')
sv_hist.plots[2].title = Title(text="Background roi", text_color='green')
sv_hist.auto_toggle.width = 300


# Stream toggle button
def stream_button_callback(state):
    global connected
    if state:
        connected = True
        stream_button.label = 'Connecting'
        stream_button.button_type = 'default'

    else:
        connected = False
        stream_button.label = 'Connect'
        stream_button.button_type = 'default'


stream_button = Toggle(label="Connect", button_type='default')
stream_button.on_click(stream_button_callback)


# Colormapper panel
colormap_panel = column(
    sv_colormapper.select,
    Spacer(height=10),
    sv_colormapper.scale_radiobuttongroup,
    Spacer(height=10),
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
)


# Metadata datatable
sv_metadata = sv.MetadataHandler(
    datatable_height=130, datatable_width=700, check_shape=(IMAGE_SIZE_Y, IMAGE_SIZE_X)
)


# Final layouts
layout_main = column(Spacer(), sv_mainview.plot)

layout_zoom = column(sv_zoomview1.plot, sv_zoomview2.plot, Spacer())

hist_layout = row(sv_hist.plots[0], sv_hist.plots[1], sv_hist.plots[2])

hist_controls = row(
    Spacer(width=20),
    column(Spacer(height=19), sv_hist.auto_toggle),
    sv_hist.lower_spinner,
    sv_hist.upper_spinner,
    sv_hist.nbins_spinner,
    column(Spacer(height=19), sv_hist.log10counts_toggle),
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

layout_controls = row(
    Spacer(width=45), column(colormap_panel, sv_mask.toggle), Spacer(width=45), stream_button
)

layout_metadata = column(
    sv_metadata.datatable, row(sv_metadata.show_all_toggle, sv_metadata.issues_dropdown)
)

final_layout = column(
    row(
        layout_main,
        Spacer(width=15),
        column(layout_zoom),
        Spacer(width=15),
        column(layout_metadata, layout_utility, layout_controls),
    ),
    column(hist_layout, hist_controls),
)

doc.add_root(final_layout)


async def update_client(image, metadata):
    sv_colormapper.update(image)
    sv_mainview.update(image)

    # Signal roi and intensity
    sig_y_start = int(np.floor(sv_zoomview1.y_start))
    sig_y_end = int(np.ceil(sv_zoomview1.y_end))
    sig_x_start = int(np.floor(sv_zoomview1.x_start))
    sig_x_end = int(np.ceil(sv_zoomview1.x_end))

    im_block1 = image[sig_y_start:sig_y_end, sig_x_start:sig_x_end]
    sig_sum = np.sum(im_block1, dtype=np.float)
    sig_area = (sig_y_end - sig_y_start) * (sig_x_end - sig_x_start)

    # Background roi and intensity
    bkg_y_start = int(np.floor(sv_zoomview2.y_start))
    bkg_y_end = int(np.ceil(sv_zoomview2.y_end))
    bkg_x_start = int(np.floor(sv_zoomview2.x_start))
    bkg_x_end = int(np.ceil(sv_zoomview2.x_end))

    im_block2 = image[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end]
    bkg_sum = np.sum(im_block2, dtype=np.float)
    bkg_area = (bkg_y_end - bkg_y_start) * (bkg_x_end - bkg_x_start)

    # Update histogram
    sv_hist.update([image, im_block1, im_block2])

    # correct the backgroud roi sum by subtracting overlap area sum
    overlap_y_start = max(sig_y_start, bkg_y_start)
    overlap_y_end = min(sig_y_end, bkg_y_end)
    overlap_x_start = max(sig_x_start, bkg_x_start)
    overlap_x_end = min(sig_x_end, bkg_x_end)
    if (overlap_y_end - overlap_y_start > 0) and (overlap_x_end - overlap_x_start > 0):
        # else no overlap
        bkg_sum -= np.sum(
            image[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end], dtype=np.float
        )
        bkg_area -= (overlap_y_end - overlap_y_start) * (overlap_x_end - overlap_x_start)

    if bkg_area == 0:
        # background area is fully surrounded by signal area
        bkg_int = 0
    else:
        bkg_int = bkg_sum / bkg_area

    # Corrected signal intensity
    sig_sum -= bkg_int * sig_area

    # Update total intensities plots
    sv_streamgraph.update([np.sum(image, dtype=np.float), sig_sum])

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update mask
    sv_mask.update(metadata.get('pedestal_file'), metadata.get('detector_name'), sv_metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    global current_gain_file, current_pedestal_file, jf_calib

    if connected:
        if receiver.state == 'polling':
            stream_button.label = 'Polling'
            stream_button.button_type = 'warning'

        elif receiver.state == 'receiving':
            stream_button.label = 'Receiving'
            stream_button.button_type = 'success'

            sv_rt.current_metadata, sv_rt.current_image = receiver.buffer[-1]

            if sv_rt.current_image.dtype != np.float16 and sv_rt.current_image.dtype != np.float32:
                gain_file = sv_rt.current_metadata.get('gain_file')
                pedestal_file = sv_rt.current_metadata.get('pedestal_file')
                detector_name = sv_rt.current_metadata.get('detector_name')
                is_correction_data_present = gain_file and pedestal_file and detector_name

                if is_correction_data_present:
                    if current_gain_file != gain_file or current_pedestal_file != pedestal_file:
                        # Update gain/pedestal filenames and JungfrauCalibration
                        current_gain_file = gain_file
                        current_pedestal_file = pedestal_file

                        with h5py.File(current_gain_file, 'r') as h5gain:
                            gain = h5gain['/gains'][:]

                        with h5py.File(current_pedestal_file, 'r') as h5pedestal:
                            pedestal = h5pedestal['/gains'][:]
                            pixel_mask = h5pedestal['/pixel_mask'][:].astype(np.int32)

                        jf_calib = ju.JungfrauCalibration(gain, pedestal, pixel_mask)

                    sv_rt.current_image = jf_calib.apply_gain_pede(sv_rt.current_image)
                    sv_rt.current_image = ju.apply_geometry(sv_rt.current_image, detector_name)
            else:
                sv_rt.current_image = sv_rt.current_image.astype('float32', copy=True)

    if sv_rt.current_image.shape != (1, 1):
        doc.add_next_tick_callback(
            partial(update_client, image=sv_rt.current_image, metadata=sv_rt.current_metadata)
        )


doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
