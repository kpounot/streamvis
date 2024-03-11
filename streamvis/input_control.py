import numpy as np
from bokeh.io import curdoc
from bokeh.models import CheckboxButtonGroup, Select, Toggle


class StreamControl:
    def __init__(self):
        """Initialize a stream control widget.
        """
        doc = curdoc()
        self.receiver = doc.receiver

        # connect toggle button
        def toggle_callback(_active):
            self._update_toggle_view()

        toggle = Toggle(label="Connect", button_type="primary")
        toggle.on_click(toggle_callback)
        self.toggle = toggle

        # data type select
        datatype_select = Select(
            title="Data type:", value="Processed", options=["Raw image", "Gains", "Processed"], default_size=145
        )
        self.datatype_select = datatype_select

        # conversion options
        conv_opts_cbbg = CheckboxButtonGroup(
            labels=["Mask", "Gap pixels", "Geometry"], active=[0, 1, 2]
        )
        self.conv_opts_cbbg = conv_opts_cbbg

        # rotate image select
        rotate_values = ["0", "90", "180", "270"]
        rotate_image = Select(
            title="Rotate image (deg):",
            value=rotate_values[0],
            options=rotate_values,
            default_size=145,
        )
        self.rotate_image = rotate_image


        # rotate image select
        sc = list(np.arange(16).astype(str)) + ['any']
        storage_cell = Select(
            title="Storage cell:",
            value=sc[0],
            options=sc,
            default_size=145,
        )
        self.storage_cell = storage_cell

        doc.add_periodic_callback(self._update_toggle_view, 1000)

    @property
    def is_activated(self):
        """Return the stream toggle state (readonly)
        """
        return self.toggle.active

    @property
    def is_receiving(self):
        """Return the stream receiver state (readonly)
        """
        return self.receiver.state == "receiving"

    def get_stream_data(self, index):
        """Get data from the stream receiver.

        Args:
            index (int): index into data buffer of receiver

        Returns:
            (dict, ndarray): metadata and image at index
        """
        active_opts = list(self.conv_opts_cbbg.active)
        mask = 0 in active_opts
        gap_pixels = 1 in active_opts
        geometry = 2 in active_opts

        if self.datatype_select.value == "Raw image":
            metadata, image = self.receiver.get_image(
                index, mask=mask, gap_pixels=gap_pixels, geometry=geometry
            )
        elif self.datatype_select.value == "Gains":
            metadata, image = self.receiver.get_image_gains(
                index, mask=mask, gap_pixels=gap_pixels, geometry=geometry
            )
        if self.datatype_select.value == "Processed":
            metadata, image = self.receiver.get_image(
                index, processed=True
            )
            return metadata, image

        n_rot = int(self.rotate_image.value) // 90
        if n_rot:
            image = np.rot90(image, k=n_rot)

        image = np.ascontiguousarray(image, dtype=np.float32)

        return metadata, image

    def _update_toggle_view(self):
        """Update label and button type of the toggle
        """
        if self.is_activated:
            if self.is_receiving:
                label = "Receiving"
                button_type = "success"
            else:
                label = "Polling"
                button_type = "warning"
        else:
            label = "Connect"
            button_type = "primary"

        self.toggle.label = label
        self.toggle.button_type = button_type
