import logging
import heapq
import mosaik_api_v3 as mosaik

META = {
    'type': 'hybrid',
    'models': {
        'Prosumer': {
            'public': True,
            'params': ['init_val'], # input (unused for now)
            'attrs': ['src', 'dest', 'formatted_msg'],
        }
    }
}

class Simulation(mosaik.Simulator):
    """Subclass this to represent the simulation state.

    Here, self.t is the simulated time and self.events is the event queue.
    """

    def __init__(self):
        super().__init__(META)

        self.eid_prefix = 'Prosumer_'
        self.entities = {}  # Maps EIDs to model instances/entities
        self.t: float = 0
        self.temp_time: float = 0 
        self.events: list[tuple[float, "Event"]] = []

    def init(self, sid, time_resolution, eid_prefix=None):
        # TEMP
        self.sid = sid
        self.time_resolution = time_resolution
        if eid_prefix is not None:
            self.eid_prefix = eid_prefix
        return self.meta
    
    # create and get_data has to be implemented in the subclass
    def create(self, num, model, init_val):
        pass

    def get_data(self, outputs):
        pass

    def schedule(self, delay, event):
        """Add an event to the event queue after the required delay."""

        heapq.heappush(self.events, (self.t + delay, event))

    def run(self, max_t=float('inf')):
        # while self.events:
        if not self.events:
            return
        t, event = item = heapq.heappop(self.events) # MAYBE HERE NOT HEAPPOP
        if t > max_t:
            return
        self.t = t
        event.process(self)

    def step(self, time, inputs, max_advance):
        # FOR NOW TIME IS IGNORED
        self.temp_time = time
        # INPUTS WILL RECEIVE INPUTS FROM NS3
        self.run(max_advance)
        return time + 1

    def log_info(self, msg):
        logging.info(f'{self.t:.2f}: {msg}')


class Event:
    """
    Subclass this to represent your events.

    You may need to define __init__ to set up all the necessary information.
    """

    def process(self, sim: Simulation):
        raise NotImplementedError

    def __lt__(self, other):
        """Method needed to break ties with events happening at the same time."""

        return id(self) < id(other)
