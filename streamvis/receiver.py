import copy
import logging
from collections import deque
from threading import RLock

import h5py
import numpy as np
import zmq

from jungfrau_utils import JFDataHandler

logger = logging.getLogger(__name__)


class StatisticsHandler:
    def __init__(self, hit_threshold, buffer_size=1):
        self.hit_threshold = hit_threshold
        self.current_run_name = None
        self.last_hit = (None, None)
        self.peakfinder_buffer = deque(maxlen=buffer_size)
        self.hitrate_buffer_fast = deque(maxlen=50)
        self.hitrate_buffer_slow = deque(maxlen=500)
        self._lock = RLock()

        self.data = dict(
            run_names=[],
            nframes=[],
            bad_frames=[],
            sat_pix_nframes=[],
            laser_on_nframes=[],
            laser_on_hits=[],
            laser_on_hits_ratio=[],
            laser_off_nframes=[],
            laser_off_hits=[],
            laser_off_hits_ratio=[],
        )

        self.sum_data = copy.deepcopy(self.data)
        for key, val in self.sum_data.items():
            if key == 'run_names':
                val.append("Summary")
            else:
                val.append(0)

    def parse(self, metadata, image):
        number_of_spots = metadata.get('number_of_spots')
        is_hit = number_of_spots and number_of_spots > self.hit_threshold

        run_name = metadata.get('run_name')
        if run_name:
            with self._lock:
                if run_name != self.current_run_name:
                    self.peakfinder_buffer.clear()
                    self.current_run_name = run_name
                    for key, val in self.data.items():
                        if key == 'run_names':
                            val.append(run_name)
                        else:
                            val.append(0)

                swissmx_x = metadata.get('swissmx_x')
                swissmx_y = metadata.get('swissmx_y')
                frame = metadata.get('frame')
                if swissmx_x and swissmx_y and frame and number_of_spots:
                    self.peakfinder_buffer.append(
                        np.array([swissmx_x, swissmx_y, frame, number_of_spots])
                    )

                self.increment('nframes')

                if 'is_good_frame' in metadata and not metadata['is_good_frame']:
                    self.increment('bad_frames')

                if 'saturated_pixels' in metadata:
                    if metadata['saturated_pixels'] != 0:
                        self.increment('sat_pix_nframes')
                else:
                    self.data['sat_pix_nframes'][-1] = np.nan

                laser_on = metadata.get('laser_on')
                if laser_on is not None:
                    switch = 'laser_on' if laser_on else 'laser_off'

                    self.increment(f'{switch}_nframes')

                    if is_hit:
                        self.increment(f'{switch}_hits')

                    self.data[f'{switch}_hits_ratio'][-1] = (
                        self.data[f'{switch}_hits'][-1] / self.data[f'{switch}_nframes'][-1]
                    )
                    self.sum_data[f'{switch}_hits_ratio'][-1] = (
                        self.sum_data[f'{switch}_hits'][-1] / self.sum_data[f'{switch}_nframes'][-1]
                    )
                else:
                    self.data['laser_on_nframes'][-1] = np.nan
                    self.data['laser_on_hits'][-1] = np.nan
                    self.data['laser_on_hits_ratio'][-1] = np.nan
                    self.data['laser_off_nframes'][-1] = np.nan
                    self.data['laser_off_hits'][-1] = np.nan
                    self.data['laser_off_hits_ratio'][-1] = np.nan

        if is_hit:
            self.last_hit = (metadata, image)
            self.hitrate_buffer_fast.append(1)
            self.hitrate_buffer_slow.append(1)
        else:
            self.hitrate_buffer_fast.append(0)
            self.hitrate_buffer_slow.append(0)

    def increment(self, key):
        self.data[key][-1] += 1
        self.sum_data[key][-1] += 1

    def reset(self):
        with self._lock:
            self.current_run_name = None

            for val in self.data.values():
                val.clear()

            for key, val in self.sum_data.items():
                if key != 'run_names':
                    val[0] = 0


class Receiver:
    def __init__(self, stats, on_receive=None, buffer_size=1):
        self.buffer = deque(maxlen=buffer_size)
        self.state = 'polling'
        self.on_receive = on_receive

        self.gain_file = ''
        self.pedestal_file = ''
        self.jf_handler = None

        self.stats = stats

    def start(self, connection_mode, address):
        zmq_context = zmq.Context(io_threads=2)
        zmq_socket = zmq_context.socket(zmq.SUB)  # pylint: disable=E1101
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # pylint: disable=E1101

        if connection_mode == 'connect':
            zmq_socket.connect(address)
        elif connection_mode == 'bind':
            zmq_socket.bind(address)
        else:
            raise RuntimeError("Unknown connection mode {connection_mode}")

        poller = zmq.Poller()
        poller.register(zmq_socket, zmq.POLLIN)

        while True:
            events = dict(poller.poll(1000))
            if zmq_socket in events:
                metadata = zmq_socket.recv_json(flags=0)
                image = zmq_socket.recv(flags=0, copy=False, track=False)

                dtype = metadata.get('type')
                shape = metadata.get('shape')
                if dtype is None or shape is None:
                    logger.error("Cannot find 'type' and/or 'shape' in received metadata")
                    continue

                image = np.frombuffer(image.buffer, dtype=dtype).reshape(shape)

                if self.on_receive is not None:
                    self.on_receive(metadata, image)

                self.buffer.append((metadata, image))
                self.state = 'receiving'

            else:
                self.state = 'polling'

    def get_image(self, index):
        metadata, image = self.buffer[index]
        return self.apply_jf_conversion(metadata, image)

    def get_last_hit(self):
        metadata, image = self.stats.last_hit
        return self.apply_jf_conversion(metadata, image)

    def get_image_gains(self, index):
        metadata, image = self.buffer[index]
        if image.dtype == np.float16 or image.dtype == np.float32:
            return metadata, image

        image = self.get_gains(image)
        return metadata, self.jf_handler.apply_geometry(image)

    def get_last_hit_gains(self):
        metadata, image = self.stats.last_hit
        if image.dtype == np.float16 or image.dtype == np.float32:
            return metadata, image

        image = self.get_gains(image)
        return metadata, self.jf_handler.apply_geometry(image)

    def apply_jf_conversion(self, metadata, image):
        if image.dtype == np.float16 or image.dtype == np.float32:
            # do not apply any conversions or corrections, because for dtype float16 or float32
            # it is already done either on a detector backend or elsewhere
            image = image.astype('float32', copy=True)
            return metadata, image

        detector_name = metadata.get('detector_name')
        # initial setup of JFDataHandler for a particular detector_name
        if self.jf_handler is None or self.jf_handler.detector_name != detector_name:
            try:
                self.jf_handler = JFDataHandler(detector_name)
            except KeyError:
                # do not apply any conversions or corrections, because detector_name is unknown
                image = image.astype('float32', copy=True)
                return metadata, image

            self.gain_file = ''
            self.pedestal_file = ''

        self._set_gain_file(metadata.get('gain_file'))
        self._set_pedestal_file(metadata.get('pedestal_file'))

        module_map = metadata.get('module_map')
        if module_map:
            self.jf_handler.module_map = np.array(module_map)

        if self.jf_handler.G is not None and self.jf_handler.P is not None:
            image = self.jf_handler.apply_gain_pede(image)

        image = self.jf_handler.apply_geometry(image)

        return metadata, image

    @staticmethod
    def get_gains(image):
        return image >> 14

    def _set_gain_file(self, filename):
        if filename == self.gain_file:
            return

        if filename:
            try:
                with h5py.File(filename, 'r') as h5gain:
                    gain = h5gain['/gains'][:]
            except:
                logger.exception(f'Can not read gain file {filename}')
                gain = None
        else:
            gain = None

        self.gain_file = filename
        self.jf_handler.G = gain

    def _set_pedestal_file(self, filename):
        if filename == self.pedestal_file:
            return

        if filename:
            try:
                with h5py.File(filename, 'r') as h5pedestal:
                    pedestal = h5pedestal['/gains'][:]
                    pixel_mask = h5pedestal['/pixel_mask'][:]
            except:
                logger.exception(f'Can not read pedestal file {filename}')
                pedestal = None
                pixel_mask = None
        else:
            pedestal = None
            pixel_mask = None

        self.pedestal_file = filename
        self.jf_handler.P = pedestal
        self.jf_handler.pixel_mask = pixel_mask
