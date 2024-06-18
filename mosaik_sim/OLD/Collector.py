import mosaik_api
import logging
import json
from cosima_core.util.general_config import CONNECT_ATTR
from cosima_core.util.util_functions import log
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

logging.basicConfig(filename='collector.log', level=logging.INFO, format='%(asctime)s %(message)s')

# The simulator meta data that we return in "init()":
META = {
    'api_version': '3.0',
    'type': 'event-based',
    'models': {
        'Collector': {
            'public': True,
            'params': [],
            'attrs': ['message'], #'stats'
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
        self.stats = {}

    def init(self, sid, **sim_params):
        self._sid = sid
        if 'client_name' in sim_params.keys():
            self.meta['models']['Collector']['attrs'].append(f'{CONNECT_ATTR}{sim_params["client_name"]}')
            self._client_name = sim_params['client_name']
        if 'simulator' in sim_params.keys():
            self._simulator = sim_params['simulator']
        return META

    def create(self, num, model, **model_conf):
        return [{'eid': self._sid, 'type': model}]

    def step(self, time, inputs, max_advance):
        # Extracting the content of the message received
        content = inputs["Collector-0"]["message_with_delay_for_client1"]["CommunicationSimulator-0.CommunicationSimulator"][0]["content"]

        # save stats of latencies for each prosumer
        new_content = json.loads(content)
        [self.stats.setdefault(content_item["src"], []).append(time) for content_item in new_content]

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

    # Function to calculate mean latencies
    def calculate_mean_latencies(self, data):
        mean_latencies = {}
        for prosumer, times in data.items():
            latencies = np.diff(times)  # Calculate the differences between consecutive timesteps
            if len(latencies) > 0:
                mean_latency = np.mean(latencies)  # Calculate the mean latency
            else:
                mean_latency = 0
            mean_latencies[prosumer] = mean_latency
        return mean_latencies


    def finalize(self):
        log('Finalize Collector')
        mean_latencies = self.calculate_mean_latencies(self.stats)

        # Plotting
        prosumer_ids = list(mean_latencies.keys())
        latencies = list(mean_latencies.values())

        plt.figure(figsize=(14, 8))
        plt.bar(prosumer_ids, latencies, color='skyblue')
        plt.xlabel('Prosumer ID')
        plt.ylabel('Mean Latency')
        plt.title('Mean Latency for Each Prosumer')
        plt.xticks(prosumer_ids)
        plt.grid(True)
        plt.show()

        # Calculate latencies
        latencies_data = {prosumer: np.diff(times) for prosumer, times in self.stats.items()}

        # Plotting
        plt.figure(figsize=(14, 8))

        for prosumer, latencies in latencies_data.items():
            steps = range(1, len(latencies) + 1)
            plt.plot(steps, latencies, marker='o', label=f'Prosumer {prosumer}')

        plt.xlabel('Step Number')
        plt.ylabel('Latency')
        plt.title('Latency Evolution for Each Prosumer Over Steps')
        plt.legend()
        plt.grid(True)
        plt.show()
