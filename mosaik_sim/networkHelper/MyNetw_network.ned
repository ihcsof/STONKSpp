
network MyNetw
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

        router: Router {
            @display("p=577.2,638.3");
        }

        mySwitch: EtherSwitch {
            @display("p=1500,1500");
        }
    
        traffic_device_0: StandardHost {
            @display("p=714.89,206.998");
        }
        client0: StandardHost { @display("p=200,100"); }
        client1: StandardHost { @display("p=400,100"); }
        client2: StandardHost { @display("p=600,100"); }
        client3: StandardHost { @display("p=800,100"); }
        client4: StandardHost { @display("p=1000,100"); }
        client5: StandardHost { @display("p=1200,100"); }

    connections:
        mySwitch.ethg++ <--> Eth { length = 100km; } <--> router.ethg++;
    
        traffic_device_0.ethg++ <--> MyEth { length = 80m; } <--> mySwitch.ethg++;
    
        client0.ethg++ <--> MyEth { length = 80m; } <--> mySwitch.ethg++;
    
        client1.ethg++ <--> MyEth { length = 80m; } <--> mySwitch.ethg++;
    
        client2.ethg++ <--> MyEth { length = 80m; } <--> mySwitch.ethg++;
    
        client3.ethg++ <--> MyEth { length = 80m; } <--> mySwitch.ethg++;
    
        client4.ethg++ <--> MyEth { length = 80m; } <--> mySwitch.ethg++;
    
        client5.ethg++ <--> MyEth { length = 80m; } <--> mySwitch.ethg++;
    
}