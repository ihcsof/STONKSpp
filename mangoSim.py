import asyncio
import math
import json
import numpy as np
import mosaik_api
from mango import create_container
from mango.agent.role import RoleAgent
from mango.agent.role import Role
from cosima_core.messages.mango_messages import SimulatorMessage
from ProsumerGUROBI_FIX import Prosumer, Manager

def log(msg, level='info'):
    print(f"[{level.upper()}] {msg}")

try:
    from cosima_core.util.general_config import CONNECT_ATTR
except ImportError:
    CONNECT_ATTR = 'connect_'

META = {
    'api_version': '3.0',
    'type': 'event-based',
    'models': {
        'MangoProsumerModel': {
            'public': True,
            'params': [],
            'attrs': ['message'],
        },
    },
}

class ADMMRole(Role):
    def __init__(self, MGraph=None, maximum_iteration=2000, penaltyfactor=0.01,
                 residual_primal=1e-4, residual_dual=1e-4):
        super().__init__()
        self.MGraph = MGraph
        self.maximum_iteration = maximum_iteration
        self.penaltyfactor = penaltyfactor
        self.residual_primal = residual_primal
        self.residual_dual = residual_dual
        self.players = {}
        self.nag = 0
        self.Trades = None
        self.iteration = 0
        self.has_finished = False
        self.prim = float('inf')
        self.dual = float('inf')

    def setup(self):
        super().setup()
        log(f'[ADMMRole] setup for agent {self.context.aid}')
        if self.MGraph:
            self.nag = len(self.MGraph.vs)
            self.Trades = np.zeros((self.nag, self.nag))
            for v in self.MGraph.vs:
                partners = np.zeros((self.nag,))
                out_edges = self.MGraph.es.select(_source=v.index)
                for e in out_edges:
                    partners[e.target] = 1
                preferences = np.zeros((self.nag,))
                if v['Type'] == 'Manager':
                    self.players[v.index] = Manager(agent=v, partners=partners,
                                                    preferences=preferences,
                                                    rho=self.penaltyfactor)
                else:
                    self.players[v.index] = Prosumer(agent=v, partners=partners,
                                                     preferences=preferences,
                                                     rho=self.penaltyfactor)
            self.context.schedule_instant_task(self._start_admm)
        else:
            log('[ADMMRole] No graph (MGraph) provided. Nothing to optimize.')
        self.context.schedule_instant_task(self._admm_iteration)

    def handle_simulator_message(self, content, meta):
        msg_bytes = content.message
        data_str = msg_bytes.decode()
        data = json.loads(data_str)
        log(f'[ADMMRole] handle_simulator_message: {data}')
        self.context.schedule_instant_task(self._admm_iteration)

    def handle_simulator_input(self, values):
        log(f'[ADMMRole] handle_simulator_input: {values}')

    async def _start_admm(self):
        if self.nag == 0:
            return
        self.iteration = 0
        self.context.schedule_instant_task(self._admm_iteration)

    async def _admm_iteration(self):
        if self.has_finished:
            return
        self.iteration += 1
        log(f'[ADMMRole] ADMM iteration #{self.iteration}')
        for i in range(self.nag):
            self.Trades[i, :] = self.players[i].optimize(self.Trades[i, :])
        self.prim = sum(p.Res_primal for p in self.players.values())
        self.dual = sum(p.Res_dual for p in self.players.values())
        log(f'[ADMMRole] iteration={self.iteration}, prim={self.prim}, dual={self.dual}')
        if ((self.prim <= self.residual_primal and self.dual <= self.residual_dual) 
                or (self.iteration >= self.maximum_iteration)):
            self.has_finished = True
            log('[ADMMRole] ADMM finished.')
            return
        self.context.schedule_task(1, self._admm_iteration)

    def get_data(self, outputs):
        return {
            self.context.aid: {
                'admm_iteration': self.iteration,
                'admm_primal': self.prim,
                'admm_dual': self.dual,
            }
        }

class MangoProsumerSimulator(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self._sid = None
        self._loop = None
        self._container = None
        self._agent = None
        self._port = 0
        self._client_agent_mapping = {}
        self._agent_roles = []
        self._conversion_factor = 1.0
        self._outputs = {}
        self._buffered_mango_inputs = []
        self._message_counter = 0

    def init(self, sid, time_resolution=1.0, **sim_params):
        self._sid = sid
        self._loop = asyncio.get_event_loop()
        if 'client_name' in sim_params:
            self.meta['models']['MangoProsumerModel']['attrs'].append(
                f"{CONNECT_ATTR}{sim_params['client_name']}"
            )
        if 'port' in sim_params:
            self._port = sim_params['port']
        if 'conversion_factor' in sim_params:
            self._conversion_factor = sim_params['conversion_factor']
        if 'agent_roles' in sim_params:
            self._agent_roles = sim_params['agent_roles']
        if 'client_agent_mapping' in sim_params and 'codec' in sim_params:
            self._client_agent_mapping = sim_params['client_agent_mapping']
            codec = sim_params['codec']
            self._loop.run_until_complete(self._create_container_and_agent(codec))
        if 'connect_attributes' in sim_params:
            self.meta['models']['MangoProsumerModel']['attrs'].extend(
                sim_params['connect_attributes']
            )
        return self.meta

    async def _create_container_and_agent(self, codec):
        self._container = await create_container(
            addr=self._sid, 
            connection_type='mosaik',
            codec=codec
        )
        agent_id = self._client_agent_mapping.get(self._sid, f"agent_{self._sid}")
        self._agent = RoleAgent(self._container, suggested_aid=agent_id)
        for role in self._agent_roles:
            self._agent.add_role(role)

    def create(self, num, model, **model_conf):
        return [{'eid': self._sid, 'type': model}]

    def step(self, time, inputs, max_advance):
        log(f'[MangoProsumerSimulator {self._sid}] step at time={time}')
        mango_inputs = []
        for eid, attr_names in inputs.items():
            for attribute, sources_to_values in attr_names.items():
                for values in sources_to_values.values():
                    if isinstance(values, list):
                        for msg_dict in values:
                            byte_content = str.encode(msg_dict['content'])
                            mango_inputs.append(byte_content)
                    else:
                        for role in self._agent_roles:
                            if hasattr(role, 'handle_simulator_input'):
                                role.handle_simulator_input(sources_to_values)
        self._buffered_mango_inputs.extend(mango_inputs)
        container_time = time / self._conversion_factor
        output = self._loop.run_until_complete(
            self._container.step(container_time, self._buffered_mango_inputs)
        )
        self._buffered_mango_inputs = []
        duration = math.ceil(output.duration * self._conversion_factor)
        next_step = time + duration if output.next_activity else None
        for msg in output.messages:
            msg_output_time = math.ceil(msg.time * self._conversion_factor)
            out_msg = {
                'msg_id': f'Message_{self._sid}_{self._message_counter}',
                'max_advance': max_advance,
                'sim_time': msg_output_time + 1,
                'sender': self._sid,
                'receiver': msg.receiver,
                'content': msg.message.decode(),
                'creation_time': msg_output_time + 1,
            }
            self._message_counter += 1
            if msg_output_time not in self._outputs:
                self._outputs[msg_output_time] = []
            self._outputs[msg_output_time].append(out_msg)
        if not output.next_activity and not self._outputs and duration <= 1:
            return None
        if output.next_activity is None:
            next_step = time + 1
        else:
            next_step = math.ceil(output.next_activity * self._conversion_factor)
        return next_step

    def get_data(self, outputs):
        data = {}
        if self._outputs:
            earliest_time = min(self._outputs.keys())
            out_list = self._outputs[earliest_time]
            data = {
                self._sid: {'message': out_list},
                'time': earliest_time + 1
            }
            del self._outputs[earliest_time]
        for role in self._agent_roles:
            if hasattr(role, 'get_data'):
                agent_data = role.get_data(outputs=outputs)
                for key, val in agent_data.items():
                    if key in data:
                        data[key].update(val)
                    else:
                        data[key] = val
        return data

    def finalize(self):
        log(f'[MangoProsumerSimulator {self._sid}] finalize()')
        self._loop.run_until_complete(self._shutdown(self._agent, self._container))

    @staticmethod
    async def _shutdown(*args):
        futs = []
        for arg in args:
            futs.append(arg.shutdown())
        log('MangoProsumerSimulator: container + agent shutdown...')
        await asyncio.gather(*futs)
        log('MangoProsumerSimulator: done.')

