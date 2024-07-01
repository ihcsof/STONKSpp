import mosaik_api
import json
from cosima_core.util.general_config import CONNECT_ATTR
from cosima_core.util.util_functions import log
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#import logging
#logging.basicConfig(filename='collector.log', level=logging.INFO, format='%(asctime)s %(message)s')

# The simulator meta data that we return in "init()":
META = {
    'api_version': '3.0',
    'type': 'event-based',
    'models': {
        'Collector': {
            'public': True,
            'params': [],
            'attrs': ['message'],
        },
    },
}

class Collector(mosaik_api.Simulator):

    def __init__(self):
        super().__init__(META)
        self._sid = None
        self._client_name = None
        self._msg_counter = 0
        self._outbox = []
        self._output_time = 0
        self._simulator = None

    def init(self, sid, **sim_params):
        self._sid = sid
        if 'client_name' in sim_params.keys():
            self.meta['models']['Collector']['attrs'].append(f'{CONNECT_ATTR}{sim_params["client_name"]}')
            self._client_name = sim_params['client_name']
            # the prosumer that this collector represents
            self.whoami = int(self._client_name[len("client"):])
            # log filename
            self.log_filename = f'collectorLogs/collector_log_{self.whoami}.log'
        if 'simulator' in sim_params.keys():
            self._simulator = sim_params['simulator']
        return META

    def create(self, num, model, **model_conf):
        return [{'eid': self._sid, 'type': model}]

    def step(self, time, inputs, max_advance):
        # Extracting the content of the message received
        content = inputs[f'Collector-{self.whoami}'][f'message_with_delay_for_client{self.whoami}']["CommunicationSimulator-0.CommunicationSimulator"][0]["content"]

        # Log the data to the file
        tosave = json.loads(content)
        with open(self.log_filename, 'a') as f:
            for content_item in tosave:
                f.write(f'{content_item["src"]},{time}\n')

        self._outbox.append({'msg_id': f'{self._client_name}_{self._msg_counter}',
                             'max_advance': max_advance,
                             'sim_time': time + 1,
                             'sender': self._client_name,
                             'receiver': self._simulator,
                             'content': content,
                             'creation_time': time,
                             })
        self._msg_counter += 1
        self._output_time = time + 1
        return None

    def get_data(self, outputs):
        data = {}
        if self._outbox:
            data = {self._sid: {f'message': self._outbox}, 'time': self._output_time}
            self._outbox = []

        return data

    def finalize(self):
        log(str(self._msg_counter)+" messages received from "+str(self._client_name))
        log('Finalize Collector')
