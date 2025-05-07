# Competition Setup Guide

This guide will walk you through setting up and running the competition environment using Docker on both Ubuntu and Windows. The repository contains a Dockerfile, a Python class interface, and an evaluation script to ensure consistent evaluation for all participants.

## Participants' Instructions
1. **Implement the Class**:
   - Use `DiagnosisSystemClass.py` as a reference.
   - Implement your solution in a new Python file (e.g., `MyDiagnosisSystem.py`).
2. **Requirements**:
   - If your solution requires additional Python libraries, list them in `requirements.txt`.

### Input and Output
- **Input File**: specify file name in the command line
```
python RunDiagnoser.py data/training_data/wltp_NF.csv
```

- **Output File**: After running the command, `results/wltp_NF.csv` will be created

### Modifying `RunDiagnoser.py`
Participants need to modify `RunDiagnoser.py` to use their own diagnosis system. Change the following lines:

```python
from ExampleDiagnosisSystem import ExampleDiagnosisSystem # Change this line to use your own diagnosis system

# Create diagnosis system
ds = ExampleDiagnosisSystem() # Change this line to use your own diagnosis system
```
Do not change anything else in run_diagnoser.py!

### Timeout
The script has a timeout of 0.1 seconds for each sample. If the diagnosis takes longer than this, the script will print a timeout message and stop processing further samples.

The solutions will be evaluated on WSL2 Ubuntu 22.04, Docker version 24.0.7, Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz.

### Initialization in `__init__`
Participants can use the `__init__` method to initialize their diagnosis system. This can include loading models, precomputed parameters, or any other setup required for their diagnosis system.

Please note that all solutions will be evaluated inside a docker image created with the provided docker file. Please ensure that all precomputed files are compatible with this environment. It is strongly recommended that you run your final model training on the docker image created using the provided dockerfile.

Example how to save and load models and scalers is provided in ExampleDiagnosisSystem.py. Please make sure that your solution includes trained models (where applicable) in the data/resources directory and the source code required to train the models.


## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your machine (instructions provided below).
- Git installed to clone the repository.

### Docker Installation

#### Ubuntu
1. **Update your system**:
   ```bash
   sudo apt update
   ```

2. **Install Docker**:
   ```bash
   sudo apt install docker.io
   ```

3. **Start and Enable Docker**:
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

4. **Add your user to the Docker group** (optional, for running Docker without `sudo`):
   ```bash
   sudo usermod -aG docker $USER
   ```
   After this, log out and log back in for changes to take effect.

#### Windows
1. **Download Docker Desktop** from [Docker's official website](https://www.docker.com/products/docker-desktop).
2. **Install Docker Desktop** and follow the on-screen instructions.
3. **Enable WSL2 Integration** (recommended):
   - Open Docker Desktop settings.
   - Navigate to "Resources" > "WSL Integration" and enable integration for your desired WSL2 distributions.
4. **Verify Installation**:
   Open PowerShell or Command Prompt and run:
   ```powershell
   docker --version
   ```

## Repository Setup
1. **Clone the Repository**:
   ```bash
   git clone git@gitlab.com:daner29/dxc25liu-ice.git
   cd dxc25liu-ice
   ```

2. **Files Overview**:
   - `Dockerfile`: Defines the Docker image for the competition.
   - `DiagnosisSystemClass.py`: The base class interface for participants.
   - `ExampleDiagnosisSystem.py`: Example implementation of the DiagnosisSystemClass.py.
   - `ExampleDiagnosisSystem.py`: Example implementation of the DiagnosisSystemClass.py using random forests as a fault classifier.
   - `RunDiagnoser.py`: The script used to run participant submissions.
   - `requirements.txt`: Contains required Python packages.

## Running the Environment

### Build the Docker Image

In the directory where the Dockerfile is located, build the Docker image:

```bash
docker build -t competition_env .
```

### Running the Evaluation

To run the evaluation, mount the current directory and specify input file. The output file is saved in the results directory with the output_ prefix.

#### Ubuntu
```bash
docker run -v "$(pwd):/app" --network none competition_env RunDiagnoser.py data/training_data/wltp_NF.csv 
```

#### Windows (PowerShell)

```powershell
docker run -v "${PWD}:/app" --network none competition_env RunDiagnoser.py data/training_data/wltp_NF.csv
```


By using --network none, the container will be completely isolated from the network, ensuring it cannot connect to the internet or any external network.

### Additional Information

Please note that the provided training data is only for testing and explanatory purposes. The complete training data is **not provided** in this repository because of GitHub file size limits. Please use the training data provided on the competition website to train your models.

