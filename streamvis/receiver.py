import logging
from collections import deque
from datetime import datetime

import numpy as np
import zmq
from jungfrau_utils import JFDataHandler

logger = logging.getLogger(__name__)


class Receiver:
    def __init__(self, on_receive=None, buffer_size=1):
        """Initialize a jungfrau receiver.

        Args:
            on_receive (function, optional): Execute function with each received metadata and image
                as input arguments. Defaults to None.
            buffer_size (int, optional): A number of last received zmq messages to keep in memory.
                Defaults to 1.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.state = "polling"
        self.on_receive = on_receive

        self.jf_adapter = StreamAdapter()

    def start(self, connection_mode, address):
        """[summary]

        Args:
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (str): The address string, e.g. 'tcp://127.0.0.1:9001'.

        Raises:
            RuntimeError: Unknown connection mode.
        """
        zmq_context = zmq.Context(io_threads=2)
        zmq_socket = zmq_context.socket(zmq.SUB)  # pylint: disable=E1101
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # pylint: disable=E1101

        if connection_mode == "connect":
            zmq_socket.connect(address)
        elif connection_mode == "bind":
            zmq_socket.bind(address)
        else:
            raise RuntimeError("Unknown connection mode {connection_mode}")

        poller = zmq.Poller()
        poller.register(zmq_socket, zmq.POLLIN)

        while True:
            events = dict(poller.poll(1000))
            if zmq_socket not in events:
                self.state = "polling"
                continue

            time_poll = datetime.now()
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=False, track=False)
            metadata["time_poll"] = time_poll
            metadata["time_recv"] = datetime.now() - time_poll

            dtype = metadata.get("type")
            shape = metadata.get("shape")
            if dtype is None or shape is None:
                logger.error("Cannot find 'type' and/or 'shape' in received metadata")
                continue

            image = np.frombuffer(image.buffer, dtype=dtype).reshape(shape)

            if self.on_receive is not None:
                self.on_receive(metadata, image)

            # add to buffer only if the recieved image is not dummy
            if image.shape != (2, 2):
                self.buffer.append((metadata, image))

            self.state = "receiving"

    def get_image(self, index, mask=True, gap_pixels=True, geometry=True):
        """Get metadata and image with the index.
        """
        metadata, raw_image = self.buffer[index]
        image = self.jf_adapter.process(
            raw_image, metadata, mask=mask, gap_pixels=gap_pixels, geometry=geometry
        )

        if (
            self.jf_adapter.handler
            and "saturated_pixels" not in metadata
            and raw_image.dtype == np.uint16
        ):
            saturated_pixels_coord = self.jf_adapter.handler.get_saturated_pixels(
                raw_image, mask=mask, gap_pixels=gap_pixels, geometry=geometry
            )

            metadata["saturated_pixels_coord"] = saturated_pixels_coord
            metadata["saturated_pixels"] = len(saturated_pixels_coord[0])

        return metadata, image

    def get_image_gains(self, index, mask=True, gap_pixels=True, geometry=True):
        """Get metadata and gains of image with the index.
        """
        metadata, image = self.buffer[index]
        if image.dtype != np.uint16:
            return metadata, image

        if self.jf_adapter.handler:
            image = self.jf_adapter.handler.get_gains(
                image, mask=mask, gap_pixels=gap_pixels, geometry=geometry
            )

        return metadata, image


class StreamAdapter:
    def __init__(self):
        # a placeholder for jf data handler to be initiated with detector name
        self.handler = None
        self._mask_double_pixels = None

    @property
    def mask_double_pixels(self):
        """Current flag for masking double pixels.
        """
        return self._mask_double_pixels

    @mask_double_pixels.setter
    def mask_double_pixels(self, value):
        value = bool(value)
        self._mask_double_pixels = value
        if self.handler is not None:
            self.handler.mask_double_pixels = value

    def process(self, image, metadata, mask=True, gap_pixels=True, geometry=True):
        """Perform jungfrau detector data processing on an image received via stream.

        Args:
            image (ndarray): An image to be processed.
            metadata (dict): A corresponding image metadata.

        Returns:
            ndarray: Resulting image.
        """
        # as a first step, try to set the detector_name, skip if detector_name is empty
        detector_name = metadata.get("detector_name")
        if detector_name:
            # check if jungfrau data handler is already set for this detector
            if self.handler is None or self.handler.detector_name != detector_name:
                try:
                    self.handler = JFDataHandler(detector_name)
                    if self.mask_double_pixels is not None:
                        self.handler.mask_double_pixels = self.mask_double_pixels
                except KeyError:
                    logging.exception(f"Error creating data handler for detector {detector_name}")
                    self.handler = None
        else:
            self.handler = None

        # return a copy of input image if
        # 1) its data type differs from 'uint16' (probably, it is already been processed)
        # 2) jf data handler failed to be created for that detector_name
        if image.dtype != np.uint16 or self.handler is None:
            return np.copy(image)

        # parse metadata
        self._update_handler(metadata)

        # skip conversion step if jungfrau data handler cannot do it, thus avoiding Exception raise
        conversion = self.handler.can_convert()

        # skip masking step if pixel_mask is None
        mask = mask and self.handler.pixel_mask is not None

        return self.handler.process(
            image, conversion=conversion, mask=mask, gap_pixels=gap_pixels, geometry=geometry
        )

    def _update_handler(self, md_dict):
        # gain file
        gain_file = md_dict.get("gain_file", "")
        try:
            self.handler.gain_file = gain_file
        except Exception:
            logging.exception(f"Error loading gain file {gain_file}")
            self.handler.gain_file = ""

        # pedestal file
        pedestal_file = md_dict.get("pedestal_file", "")
        try:
            self.handler.pedestal_file = pedestal_file
        except Exception:
            logging.exception(f"Error loading pedestal file {pedestal_file}")
            self.handler.pedestal_file = ""

        # module map
        module_map = md_dict.get("module_map")
        self.handler.module_map = None if (module_map is None) else np.array(module_map)

        # highgain
        daq_rec = md_dict.get("daq_rec")
        self.handler.highgain = False if (daq_rec is None) else bool(daq_rec & 0b1)
