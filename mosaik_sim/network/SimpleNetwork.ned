package networks;
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

import inet.common.scenario.ScenarioManager;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth10M;
import inet.node.ethernet.EtherLink;
import inet.node.ethernet.EtherSwitch;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import inet.visualizer.integrated.IntegratedCanvasVisualizer;
import modules.MosaikSchedulerModule;
import modules.MosaikScenarioManager;


channel MyEth10M extends EtherLink
{
    datarate = 10Mbps;
}

network ExpandedNetwork
{
    @display("bgb=3000,3000");
    submodules:
        scenarioManager: MosaikScenarioManager {
            @display("p=100,2850");
        }

        schedulerModule: MosaikSchedulerModule {
            @display("p=200,2850");
        }

        configurator: Ipv4NetworkConfigurator {
            @display("p=300,2850");
        }

        visualizer: IntegratedCanvasVisualizer {
            @display("p=400,2850");
        }

        mainSwitch: EtherSwitch {
            @display("p=1500,1500");
        }

        // 51 individual clients
        client0: StandardHost { @display("p=200,100"); }
        client1: StandardHost { @display("p=400,100"); }
        client2: StandardHost { @display("p=600,100"); }
        client3: StandardHost { @display("p=800,100"); }
        client4: StandardHost { @display("p=1000,100"); }
        client5: StandardHost { @display("p=1200,100"); }
        client6: StandardHost { @display("p=1400,100"); }
        client7: StandardHost { @display("p=1600,100"); }
        client8: StandardHost { @display("p=1800,100"); }
        client9: StandardHost { @display("p=2000,100"); }
        client10: StandardHost { @display("p=2200,100"); }
        client11: StandardHost { @display("p=2400,100"); }
        client12: StandardHost { @display("p=2600,100"); }
        client13: StandardHost { @display("p=2800,100"); }
        client14: StandardHost { @display("p=300,300"); }
        client15: StandardHost { @display("p=500,300"); }
        client16: StandardHost { @display("p=700,300"); }
        client17: StandardHost { @display("p=900,300"); }
        client18: StandardHost { @display("p=1100,300"); }
        client19: StandardHost { @display("p=1300,300"); }
        client20: StandardHost { @display("p=1500,300"); }
        client21: StandardHost { @display("p=1700,300"); }
        client22: StandardHost { @display("p=1900,300"); }
        client23: StandardHost { @display("p=2100,300"); }
        client24: StandardHost { @display("p=2300,300"); }
        client25: StandardHost { @display("p=2500,300"); }
        client26: StandardHost { @display("p=2700,300"); }
        client27: StandardHost { @display("p=2900,300"); }
        client28: StandardHost { @display("p=200,500"); }
        client29: StandardHost { @display("p=400,500"); }
        client30: StandardHost { @display("p=600,500"); }
        client31: StandardHost { @display("p=800,500"); }
        client32: StandardHost { @display("p=1000,500"); }
        client33: StandardHost { @display("p=1200,500"); }
        client34: StandardHost { @display("p=1400,500"); }
        client35: StandardHost { @display("p=1600,500"); }
        client36: StandardHost { @display("p=1800,500"); }
        client37: StandardHost { @display("p=2000,500"); }
        client38: StandardHost { @display("p=2200,500"); }
        client39: StandardHost { @display("p=2400,500"); }
        client40: StandardHost { @display("p=2600,500"); }
        client41: StandardHost { @display("p=2800,500"); }
        client42: StandardHost { @display("p=300,700"); }
        client43: StandardHost { @display("p=500,700"); }
        client44: StandardHost { @display("p=700,700"); }
        client45: StandardHost { @display("p=900,700"); }
        client46: StandardHost { @display("p=1100,700"); }
        client47: StandardHost { @display("p=1300,700"); }
        client48: StandardHost { @display("p=1500,700"); }
        client49: StandardHost { @display("p=1700,700"); }
        client50: StandardHost { @display("p=1900,700"); }

    connections:
        client0.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client1.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client2.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client3.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client4.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client5.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client6.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client7.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client8.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client9.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client10.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client11.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client12.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client13.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client14.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client15.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client16.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client17.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client18.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client19.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client20.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client21.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client22.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client23.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client24.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client25.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client26.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client27.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client28.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client29.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client30.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client31.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client32.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client33.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client34.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client35.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client36.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client37.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client38.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client39.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client40.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client41.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client42.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client43.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client44.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client45.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client46.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client47.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client48.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client49.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;
        client50.ethg++ <--> MyEth10M { length = 100m; } <--> mainSwitch.ethg++;

}

network SimpleNetwork
{
    @display("bgb=1332.5,1063.4");
    submodules:

        scenarioManager: MosaikScenarioManager {

            @display("p=97.5,317.19998");
        }

        schedulerModule: MosaikSchedulerModule {
            @display("p=200,100");
        }

        configurator: Ipv4NetworkConfigurator {
            @display("p=100,100");
        }

        client0: StandardHost {
            @display("p=721,100");

        }
        client2: StandardHost {
            @display("p=1016.6,659.1");
        }
        client1: StandardHost {
            @display("p=799.5,379.6");
        }
        router0: Router {
            @display("p=577.2,638.3");
        }
        router1: Router {
            @display("p=300.3,452.4");
        }
        switch0: EtherSwitch {
            @display("p=800.8,657.8");
        }
        switch1: EtherSwitch {
            @display("p=495.3,109.2");
        }

        visualizer: IntegratedCanvasVisualizer {
            @display("p=98,209");
        }

        router2: Router {
            @display("p=208,279.5");
        }
        router3: Router {
            @display("p=422.5,301.6");
        }
        router4: Router {
            @display("p=284.69998,128.7");
        }
        switch2: EtherSwitch {
            @display("p=613.6,315.9");
        }
        client3: StandardHost {
            @display("p=1030.9,942.5");
        }
        switch3: EtherSwitch {
            @display("p=688.32007,876.82263");
        }
    connections:
        switch0.ethg++ <--> Eth10M {  length = 1000km; } <--> client2.ethg++;
        switch0.ethg++ <--> Eth10M {  length = 1000km; } <--> router0.ethg++;
        router0.ethg++ <--> Eth10M {  length = 1000km; } <--> router1.ethg++;
        client0.ethg++ <--> Eth10M {  length = 500km; } <--> switch1.ethg++;
        router2.ethg++ <--> Eth10M {  length = 100km; } <--> router4.ethg++;
        router2.ethg++ <--> Eth10M {  length = 500km; } <--> router1.ethg++;
        router1.ethg++ <--> Eth10M {  length = 500km; } <--> router3.ethg++;
        router3.ethg++ <--> Eth10M {  length = 50km; } <--> router4.ethg++;
        router4.ethg++ <--> Eth10M {  length = 50km; } <--> switch1.ethg++;
        router2.ethg++ <--> Eth10M {  length = 50km; } <--> router3.ethg++;
        switch2.ethg++ <--> Eth10M {  length = 50km; } <--> client1.ethg++;
        switch2.ethg++ <--> Eth10M {  length = 50km; } <--> router3.ethg++;
        switch3.ethg++ <--> Eth10M {  length = 50km; } <--> client3.ethg++;
        router0.ethg++ <--> Eth10M {  length = 50km; }  <--> switch3.ethg++;
}

network DisconnectedClient2_SimpleNetwork
{
    submodules:

		scenarioManager: MosaikScenarioManager {
            @display("p=216.6825,136.52249");
        }
        schedulerModule: MosaikSchedulerModule {
            @display("p=200,100");
        }

        configurator: Ipv4NetworkConfigurator {
            @display("p=100,100");
        }

        client0: StandardHost {
            @display("p=721,100");

        }
        client2: StandardHost {
            @display("p=1016.6,659.1");
        }
        client1: StandardHost {
            @display("p=799.5,379.6");
        }
        router0: Router {
            @display("p=577.2,638.3");
        }
        router1: Router {
            @display("p=300.3,452.4");
        }
        switch0: EtherSwitch {
            @display("p=800.8,657.8");
        }
        switch1: EtherSwitch {
            @display("p=495.3,109.2");
        }

        visualizer: IntegratedCanvasVisualizer {
            @display("p=98,209");
        }

        router2: Router {
            @display("p=208,279.5");
        }
        router3: Router {
            @display("p=422.5,301.6");
        }
        router4: Router {
            @display("p=284.69998,128.7");
        }
        switch2: EtherSwitch {
            @display("p=613.6,315.9");
        }
    connections:
        switch0.ethg++ <--> Eth10M {  length = 1000km; } <--> router0.ethg++;
        router0.ethg++ <--> Eth10M {  length = 1000km; } <--> router1.ethg++;
        client0.ethg++ <--> Eth10M {  length = 500km; } <--> switch1.ethg++;
        router2.ethg++ <--> Eth10M {  length = 100km; } <--> router4.ethg++;
        router2.ethg++ <--> Eth10M {  length = 500km; } <--> router1.ethg++;
        router1.ethg++ <--> Eth10M {  length = 500km; } <--> router3.ethg++;
        router3.ethg++ <--> Eth10M {  length = 50km; } <--> router4.ethg++;
        router4.ethg++ <--> Eth10M {  length = 50km; } <--> switch1.ethg++;
        router2.ethg++ <--> Eth10M {  length = 50km; } <--> router3.ethg++;
        switch2.ethg++ <--> Eth10M {  length = 50km; } <--> client1.ethg++;
        switch2.ethg++ <--> Eth10M {  length = 50km; } <--> router3.ethg++;
}

network DisconnectedClient1_SimpleNetwork
{
    submodules:
		scenarioManager: MosaikScenarioManager {
            @display("p=216.6825,136.52249");
        }
        
        schedulerModule: MosaikSchedulerModule {
            @display("p=200,100");
        }

        configurator: Ipv4NetworkConfigurator {
            @display("p=100,100");
        }

        client0: StandardHost {
            @display("p=721,100");

        }
        client2: StandardHost {
            @display("p=1016.6,659.1");
        }
        client1: StandardHost {
            @display("p=799.5,379.6");
        }
        router0: Router {
            @display("p=577.2,638.3");
        }
        router1: Router {
            @display("p=300.3,452.4");
        }
        switch0: EtherSwitch {
            @display("p=800.8,657.8");
        }
        switch1: EtherSwitch {
            @display("p=495.3,109.2");
        }

        visualizer: IntegratedCanvasVisualizer {
            @display("p=98,209");
        }

        router2: Router {
            @display("p=208,279.5");
        }
        router3: Router {
            @display("p=422.5,301.6");
        }
        router4: Router {
            @display("p=284.69998,128.7");
        }
        switch2: EtherSwitch {
            @display("p=613.6,315.9");
        }
    connections:
        switch0.ethg++ <--> Eth10M {  length = 1000km; } <--> client2.ethg++;
        switch0.ethg++ <--> Eth10M {  length = 1000km; } <--> router0.ethg++;
        router0.ethg++ <--> Eth10M {  length = 1000km; } <--> router1.ethg++;
        client0.ethg++ <--> Eth10M {  length = 500km; } <--> switch1.ethg++;
        router2.ethg++ <--> Eth10M {  length = 100km; } <--> router4.ethg++;
        router2.ethg++ <--> Eth10M {  length = 500km; } <--> router1.ethg++;
        router1.ethg++ <--> Eth10M {  length = 500km; } <--> router3.ethg++;
        router3.ethg++ <--> Eth10M {  length = 50km; } <--> router4.ethg++;
        router4.ethg++ <--> Eth10M {  length = 50km; } <--> switch1.ethg++;
        router2.ethg++ <--> Eth10M {  length = 50km; } <--> router3.ethg++;
        switch2.ethg++ <--> Eth10M {  length = 50km; } <--> router3.ethg++;
}

network SmallNetwork
{
    submodules:
        scenarioManager: MosaikScenarioManager {
            @display("p=216.6825,136.52249");
        }

        schedulerModule: MosaikSchedulerModule {
            @display("p=200,100");
        }

        configurator: Ipv4NetworkConfigurator {
            @display("p=100,100");
        }

        client0: StandardHost {
            @display("p=183,404");

        }
        client2: StandardHost {
            @display("p=712,365");
        }
        client1: StandardHost {
            @display("p=633,564");
        }

        router0: Router {
            @display("p=463,359");
        }
        switch0: EtherSwitch {
            @display("p=582,358");
        }
        switch1: EtherSwitch {
            @display("p=307,341");
        }

        visualizer: IntegratedCanvasVisualizer {
            @display("p=98,209");
        }

        switch2: EtherSwitch {
            @display("p=543,477");
        }
    connections:
        switch0.ethg++ <--> Eth10M {  length = 67km; } <--> client2.ethg++;
        switch0.ethg++ <--> Eth10M {  length = 100km; } <--> router0.ethg++;

        client0.ethg++ <--> Eth10M {  length = 100km; } <--> switch1.ethg++;
        router0.ethg++ <--> Eth10M {  length = 100km; } <--> switch1.ethg++;
        router0.ethg++ <--> Eth10M {  length = 100km; } <--> switch2.ethg++;
        switch2.ethg++ <--> Eth10M {  length = 100km; } <--> client1.ethg++;

}

