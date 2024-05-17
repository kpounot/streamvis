import os

import logging
from collections import deque
from datetime import datetime

from pathlib import Path

import json

import numpy as np
import zmq
from jungfrau_utils import JFDataHandler
from numba import njit

from datastorage import DataStorage

from id09.detectors.jungfrau import correct

logger = logging.getLogger(__name__)


class Receiver:
    def __init__(
            self, 
            on_receive=None, 
            buffer_size=1, 
            publisher_address=None,
            burst=False,
        ):
        """Initialize a jungfrau receiver.

        Args:
            on_receive (function, optional): Execute function with each received metadata and image
                as input arguments. Defaults to None.
            buffer_size (int, optional): A number of last received zmq messages to keep in memory.
                Defaults to 1.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.raw = deque(maxlen=buffer_size)
        self.state = "polling"
        self.on_receive = on_receive
        self.publisher = publisher_address
        
        self.burst = burst

        self.sc = range(16)

        self.jf_adapter = StreamAdapter()

    def start(self, io_threads, connection_mode, address, burst=False,
              timeout=1000,
              exp_folder="/data/visitor/blc15385/id09/20240513/"):
        """Start the receiver loop.

        Args:
            io_threads (int): The size of the zmq thread pool to handle I/O operations.
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (list of str): The addresses strings, e.g. '[tcp://127.0.0.1:9001]'.

        Raises:
            RuntimeError: Unknown connection mode.
        """
        zmq_context = zmq.Context(io_threads=io_threads)

        gain_folder = Path("/data/id09/archive/calibrations/jf1m")
        exp_folder = Path(exp_folder)
        if not exp_folder.exists():
            raise Exception(f"'exp_folder' ({str(exp_folder)}) does not exist")
        if burst:
            print("Using burst")
            gains_file = str(gain_folder / "gain_maps_m593_m601_sc.h5")
            darks_file = str(exp_folder / "PROCESSED_DATA" / "darks" /
                             "darks_last_sc.h5")
        else:
            gains_file = str(gain_folder / "gain_maps_m593_m601.h5")
            darks_file = str(exp_folder / "PROCESSED_DATA" / "darks" /
                             "darks_last.h5")

        darks_mtime = os.lstat(darks_file).st_mtime

        gains = DataStorage(gains_file).gains
        darks = DataStorage(darks_file).darks
        
        sockets = []
        poller = zmq.Poller()
        for adrs in address:
            zmq_socket = zmq_context.socket(zmq.SUB)
            if connection_mode == "connect":
                zmq_socket.connect(adrs)
            elif connection_mode == "bind":
                zmq_socket.bind(adrs)
            else:
                raise RuntimeError("Unknown connection mode {connection_mode}")

            zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            poller.register(zmq_socket, zmq.POLLIN)
            sockets.append(zmq_socket)

        while True:
            if darks_mtime != os.lstat(darks_file).st_mtime:
                darks = DataStorage(darks_file).darks
                darks_mtime = os.lstat(darks_file).st_mtime
                print(f"Using a newer dark: {os.readlink(darks_file)}")
            events = dict(poller.poll(timeout))
            time_poll = datetime.now()
            is_receiving = True
            while is_receiving:
                metas, imgs = [], []
                for socket in sockets:
                    if socket not in events:
                        self.state = "polling"
                        is_receiving = False

                    ans = socket.recv_multipart()
                    if len(ans) == 2:  # check if it is (metadata, data) for one module
                        metadata, image = ans
                        self.raw.append((metadata, image))
                        metadata = json.loads(metadata.decode('utf-8'))
                        metadata["time_poll"] = time_poll
                        metadata["time_recv"] = datetime.now() - time_poll
                        dtype = metadata.get("type" )
                        shape = metadata.get("shape")
                        if dtype is None and shape is None:
                            logger.error("Cannot find 'type' and 'shape' in received metadata")
                            continue

                        if dtype is None:
                            dtype = np.uint16

                        if len(image) % 16 == 0:
                            image = np.frombuffer(image, dtype=dtype).reshape(shape[::-1])
                        else:
                            print("WARNING: image data size is not 16 bits per pixel")
                            continue

                        if image.shape != (2, 2):
                            metas.append(metadata)
                            imgs.append(image)

                    else:
                        is_receiving = False

                    if len(metas) == 2:
                        if metas[0]['frameIndex'] != metas[1]['frameIndex']:
                            continue
                        frameIdx = metas[-1]['frameIndex'] % 16 if burst else 0
                        metadata = metas[-1]
                        if frameIdx in self.sc:
                            image = np.array(imgs[-2:])
                            try: 
                                image = correct(
                                    image, 
                                    darks=darks[frameIdx], 
                                    gains=gains[frameIdx], 
                                    geometry=True
                                )
                                # image = image
                                self.buffer.append((metadata, image))
                            except:
                                print("Check if dark is in the right BURST/STANDARD mode")
                            self.state = "receiving"

                            if self.on_receive is not None:
                                self.on_receive(metadata, image)
        
    def get_image(self, index, mask=True, gap_pixels=True, geometry=True, processed=False):
        """Get metadata and image with the index.
        """
        metadata, raw_image = self.buffer[index]

        if not processed:
            image = self.jf_adapter.process(
                raw_image, metadata, mask=mask, gap_pixels=gap_pixels, geometry=geometry
            )
        else:
            image = raw_image

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

        # Buffer image mask data
        self._inv_mask = None
        self._pedestal_file = ""
        self._mask_gap_pixels = None
        self._mask_geometry = None
        self._module_map = None

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

        # return a copy of input image if jf data handler creation failed for that detector_name
        if self.handler is None:
            return np.copy(image)

        # still try to apply mask if data type differs from 'uint16' (probably, it is already been
        # processed)
        if image.dtype != np.uint16:
            image = image.astype(np.float32, copy=True)
            if mask:
                image = self._apply_mask(image, gap_pixels, geometry)
            return image

        # parse metadata
        self._update_handler(metadata)

        # skip conversion step if jungfrau data handler cannot do it, thus avoiding Exception raise
        conversion = self.handler.can_convert()

        proc_image = self.handler.process(
            image, conversion=conversion, mask=False, gap_pixels=gap_pixels, geometry=geometry
        )

        if mask:
            proc_image = self._apply_mask(proc_image, gap_pixels, geometry)

        return proc_image

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

    def _apply_mask(self, image, gap_pixels, geometry):
        # assign masked values to np.nan
        if self.handler.pixel_mask is not None:
            # check if mask needs to be refreshed
            if (
                self._pedestal_file != self.handler.pedestal_file
                or np.any(self._module_map != self.handler.module_map)
                or self._gap_pixels != gap_pixels
                or self._geometry != geometry
            ):
                self._inv_mask = np.invert(self.handler.get_pixel_mask(gap_pixels, geometry))
                self._pedestal_file = self.handler.pedestal_file
                self._module_map = self.handler.module_map
                self._gap_pixels = gap_pixels
                self._geometry = geometry

            # cast to np.float32 in case there was no conversion, but mask should still be applied
            if image.dtype != np.float32:
                image = image.astype(np.float32)

            if image.shape == self._inv_mask.shape:
                _apply_mask_core(image, self._inv_mask)
            else:
                raise ValueError("Image and mask shapes are not the same")

        else:
            self._inv_mask = None
            self._pedestal_file = ""
            self._module_map = None
            self._gap_pixels = None
            self._geometry = None

        return image


@njit
def _apply_mask_core(image, mask):
    sy, sx = image.shape
    for i in range(sx):
        for j in range(sy):
            if mask[j, i]:
                image[j, i] = np.nan


if __name__ == "__main__":
    rcv = Receiver()

    rcv.start(1, 'connect', ['tcp://172.29.10.100:30001', 'tcp://172.29.10.100:30002'])
