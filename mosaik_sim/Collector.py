import mosaik_api
import json
from cosima_core.util.general_config import CONNECT_ATTR
from cosima_core.util.util_functions import log
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random
import re

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
        self._nMessagesFrom = {}
        self.loss_prob = np.zeros(200)

    def init(self, sid, **sim_params):
        self._sid = sid
        # id of the run among multiple runs (see runs.py)
        if 'run' in sim_params.keys():
            self._run_id = sim_params['run']
        else:
            self._run_id = 1
        if 'client_name' in sim_params.keys():
            self.meta['models']['Collector']['attrs'].append(f'{CONNECT_ATTR}{sim_params["client_name"]}')
            self._client_name = sim_params['client_name']
            # the prosumer that this collector represents
            self.whoami = int(self._client_name[len("client"):])
            # log filenames
            self.log_filename = f'collectorLogs{self._run_id}/collector_log_{self.whoami}.log'
            self.log_filename_msgs = f'messagesLogs/message_log_{self.whoami}.log'
        if 'simulator' in sim_params.keys():
            self._simulator = sim_params['simulator']
        if 'loss_prob' in sim_params.keys():
            if len(sim_params['loss_prob']) == 1:
                # Repeat the single element 200 times
                defined_loss_prob = [sim_params['loss_prob'][0]] * 200
            else:
                # Use the provided loss_prob array as is
                defined_loss_prob = sim_params['loss_prob']
        else:
            defined_loss_prob = []
        
        self.loss_prob[:len(defined_loss_prob)] = defined_loss_prob
        return META
    
    def append_or_increment_msg_id(self, msg_id):
        # Check if there is already a number at the end of msg_id
        match = re.match(r"^(.*?_\d+)(?:_(\d+))?$", msg_id)
        
        if match:
            base_id = match.group(1)  # clientname_msgcounter part
            last_number = match.group(2)  # the last number after _msgcounter
            
            if last_number is None:
                # No number present after clientname_msgcounter, append '_1'
                new_msg_id = f"{base_id}_1"
            else:
                # Increment the existing number
                new_msg_id = f"{base_id}_{int(last_number) + 1}"
        else:
            # In case the msg_id doesn't match the expected pattern
            raise ValueError("msg_id format is incorrect")
        
        return new_msg_id

    def create(self, num, model, **model_conf):
        return [{'eid': self._sid, 'type': model}]

    def step(self, time, inputs, max_advance):
        # Extracting the content of the message received
        msg_id = inputs[f'Collector-{self.whoami}'][f'message_with_delay_for_client{self.whoami}']["CommunicationSimulator-0.CommunicationSimulator"][0]["msg_id"]
        prev_time = inputs[f'Collector-{self.whoami}'][f'message_with_delay_for_client{self.whoami}']["CommunicationSimulator-0.CommunicationSimulator"][0]["sim_time"]
        content = inputs[f'Collector-{self.whoami}'][f'message_with_delay_for_client{self.whoami}']["CommunicationSimulator-0.CommunicationSimulator"][0]["content"]

        # Log the latency data to the file (event steps for each prosumer)
        #tosave = json.loads(content)
        
        '''with open(self.log_filename, 'a') as f:
            for content_item in tosave:
                # with calc sim_time (if using SDCWithCalcLatencies)
                #f.write(f'{content_item["src"]},{time + content_item["real_time"]}\n')
                f.write(f'{content_item["src"]},{time}\n')'''

        # Count the number of messages received from each prosumer
        '''for content_item in tosave:
            pros = content_item["dest"]
            if pros in self._nMessagesFrom:
                self._nMessagesFrom[pros] += 1
            else:
                self._nMessagesFrom[pros] = 1'''

        # Easily erase data loss probabilities
        # loss_prob = []
        if random.random() < self.loss_prob[self.whoami]:
            acc_msg_id = self.append_or_increment_msg_id(msg_id)

            # (1) Retransimmiting:
            log("Collector: Data loss: retransmitting message")
            self._outbox.append({'msg_id': acc_msg_id,
                             'max_advance': max_advance,
                             'sim_time': time + 1,
                             'sender': self._client_name,
                             'receiver': self._client_name,
                             'content': content,
                             'creation_time': prev_time,
                             })

            # (2) Lost:
            '''log("Collector: Data loss: lost message")
            self._outbox.append({'msg_id': acc_msg_id,
                                'max_advance': max_advance,
                                'sim_time': time + 2,
                                'sender': self._client_name,
                                'receiver': self._simulator,
                                'content': "[{\"src\": -1, \"dest\": -1, \"trade\": -1}]",
                                'creation_time': time,
                                })'''
        else:
            self._outbox.append({'msg_id': f'id:{msg_id}',
                                'max_advance': max_advance,
                                'sim_time': time + 1,
                                'sender': self._client_name,
                                'receiver': self._simulator,
                                'content': content,
                                'creation_time': prev_time,
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
        log("Messages received from each prosumer:")
        log(str(self._nMessagesFrom))
        # log the number of messages received from each prosumer to the file
        '''with open(self.log_filename_msgs, 'w') as f:
            for prosumer, nMessages in self._nMessagesFrom.items():
                f.write(f'{prosumer},{nMessages}\n')'''
        log('Finalize Collector')
