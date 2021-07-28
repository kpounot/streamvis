import numpy as np
from bokeh.models import Asterisk, CheckboxGroup, ColumnDataSource


class SaturatedPixels:
    def __init__(self, image_views, sv_metadata):
        """Initialize a saturated pixels overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_metadata (MetadataHandler): A metadata handler to report metadata issues.
        """
        self._sv_metadata = sv_metadata

        # ---- saturated pixel markers
        self._source = ColumnDataSource(dict(x=[], y=[]))

        marker_glyph = Asterisk(x="x", y="y", size=20, line_color="white", line_width=2)

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, marker_glyph)

        # ---- toggle button
        toggle = CheckboxGroup(labels=["Saturated Pixels"], active=[0], default_size=145)
        self.toggle = toggle

    def _clear(self):
        if len(self._source.data["x"]):
            self._source.data.update(x=[], y=[])

    def update(self, metadata):
        """Trigger an update for the saturated pixels overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        if not self.toggle.active:
            self._clear()
            return

        saturated_pixels_y = metadata.get("saturated_pixels_y")
        saturated_pixels_x = metadata.get("saturated_pixels_x")

        if saturated_pixels_y is None or saturated_pixels_x is None:
            self._sv_metadata.add_issue("Metadata does not contain data for saturated pixels")
            self._clear()
            return

        # convert coordinates to numpy arrays, because if these values were received as a part
        # of a zmq message, they will be lists (ndarray is not JSON serializable)
        saturated_pixels_y = np.array(saturated_pixels_y, copy=False)
        saturated_pixels_x = np.array(saturated_pixels_x, copy=False)
        self._source.data.update(x=saturated_pixels_x + 0.5, y=saturated_pixels_y + 0.5)
