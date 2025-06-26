# Ray Tracing-Based Large-Scale MIMO Channel Simulator

<img width="100%" alt="ray_simulator_architecture" src="https://github.com/user-attachments/assets/3bfbed87-7319-4ffc-a56d-966ba2a2ee5b" /><br/>

<p align="justify">
&emsp;Ray Tracing-Based Large-Scale MIMO Channel Simulator provides a deterministic and realistic channel modeling environment crucial for wireless communication studies. It constructs a comprehensive 3D virtual map from OpenStreetMap, integrating detailed representations of roads, buildings, and strategically deployed base stations (BSs) to ensure broad coverage. Mobility patterns of user equipment (UE), represented as vehicles, are realistically generated using the Simulation of Urban MObility (SUMO), which emulates traffic flow and vehicle behavior in urban environments. The integrated ray tracing module, Sionna RT, calculates accurate propagation paths between BSs and UEs, considering physical phenomena such as reflection, diffraction, and scattering. Configurable parameters in Sionna RT, including interaction types, antenna radiation patterns, and ray propagation depth, enable precise modeling of the MIMO-OFDM channel, providing detailed channel characteristics such as angles of arrival and departure, path coefficients, and delays.
</p><br/>

## Citation
<p align="justify">
If you use this dataset or any part of the code, please cite the paper below  
</p>

```

```

<br/>

## Virtual Environment Overview
* [Suwon](https://github.com/yongjun0711/Ray_tracing-based_large-scale_MIMO_channel_simulator/tree/main/scenario/Suwon)

<br/>

## Installation
We recommend that you run this simulator in a virtual environment using [Conda](https://www.anaconda.com/).<br/><br/>

### _1. Prerequisite_
#### _1-1. Environmnet preparation for Sionna_
**GPU support (NVIDIA GPUs Only)**

If you would like to run the code on a GPU, be sure to complete the following steps

- **Update NVIDIA Drivers:** Install the most recent driver release that matches your GPU model.
- **Install GPU-Enabled TensorFlow:** Choose and install the TensorFlow build compiled with CUDA support.
- **Set Up CUDA & cuDNN:** Download and configure both CUDA and cuDNN libraries—these are required for TensorFlow to offload operations onto the GPU.

You can refer to [here](https://www.tensorflow.org/install).<br/><br/>

**CPU support**

If you would like to run the code on a CPU, ensure [LLVM](https://llvm.org/) is installed on your system.<br/><br/>

#### _1-2. SUMO_
To run the mobility simulator, ensure [SUMO](https://sumo.dlr.de/docs/Downloads.php) is installed on your system.<br/>

<hr/>

### _2. Requirements_
You can install all necessary libraries required for this simulator using the command below
```
pip install -r requirements.txt
```

<br/>

## Usage
You can create a customized MIMO-OFDM channel dataset by following the steps below.<br/><br/>

### _1. Set config.yaml_
We provide examples that operate at [7.8 GHz](https://github.com/yongjun0711/Ray_tracing-based_large-scale_MIMO_channel_simulator/blob/main/scenario/Suwon/simulation/7_8G/config.yaml) and [15.1 GHz](https://github.com/yongjun0711/Ray_tracing-based_large-scale_MIMO_channel_simulator/blob/main/scenario/Suwon/simulation/15_1G/config.yaml), respectively. By referring to the table below and modifying your `config.yaml`, you can design your simulator.

| Parameter               | Value                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| simulation_time         | Duration of the mobility simulation in seconds.                                                              |
| simulation_step         | Interval in seconds for collecting vehicle data during the simulation.                                       |
| insertion_density       | Sets traffic density level; higher values increase congestion.                                               |
| min_distance            | Minimum travel distance for generated vehicles.                                                              |
| vehicle_height          | Height of the generated vehicles.                                                                            |
| frequency               | Operating frequency of the virtual environment.                                                              |
| diffraction             | Enables or disables diffraction in the propagation simulation.                                               |
| scattering              | Enables or disables scattering in the propagation simulation.                                                |
| edge_diffraction        | Enables or disables diffraction at surface edges.                                                            |
| scat_keep_prob          | Probability of retaining a scattered propagation path.                                                       |
| scat_random_phases      | Enables or disables random phase shifts due to scattering.                                                   |
| antenna_pattern         | Defines the radiation patterns of transmit and receive antennas.                                             |
| antenna_polarization    | Defines the polarization of transmit and receive antennas.                                                   |
| max_depth               | Maximum number of reflections for generated radio signals.                                                   |
| num_samples             | Number of candidate points for generating radio signals; higher values increase generation likelihood and computational load. |
| max_rx_positions        | Number of receivers simulated concurrently; higher values reduce total simulation runs but increase computations per run. |
| preprocess_score_thresh | Threshold used in preprocessing to index rays based on similarity.                                  |

<hr/>

### _2. Mobility Simulator_
You run Mobility Simulator with the command below, and the results are saved in `ue.pkl`.
```
python ./simulator/mobility_simulator.py
```

<hr/>

### _3. Ray Tracing Simulator_
You run Ray Tracing Simulator with the command below, and the results are saved in `ray_group.pkl` and `ray_{ue_name}.pkl`.
```
python ./simulator/ray_simulator.py
```

<hr/>

### _4. Animation for Ray Tracing Simulation_
We provide an animation tool to visualize ray tracing simulations based on [pythreejs](https://github.com/jupyter-widgets/pythreejs/). You can view the animation for your ray tracing simulation in `animation_script.ipynb`.
