# 🧠 NAS for Multi-Core Scheduling

A sophisticated Neural Architecture Search (NAS) system for optimizing multi-core task scheduling policies using ensemble surrogate models and active learning.

## 🎯 Project Overview

This project implements an advanced NAS framework that automatically discovers optimal neural network architectures for multi-core task scheduling. It combines:

- **Ensemble Surrogate Models** (Random Forest, Gradient Boosting, Neural Networks, Gaussian Processes)
- **Active Learning** with uncertainty-based exploration
- **Advanced Simulation Features** (heterogeneous cores, resource contention, energy modeling)
- **Interactive Dashboard** for real-time experimentation and visualization

## ✨ Key Features

### 🚀 Advanced Simulation Capabilities
- **Heterogeneous Cores**: Support for different core types (Performance, Efficient, Standard)
- **Resource Contention**: Simulate shared caches, memory bandwidth, and I/O bottlenecks
- **Energy Modeling**: Track and optimize for energy consumption alongside performance
- **Preemption/Interrupts**: Realistic task preemption and system interrupt simulation
- **Stochastic Arrivals**: Realistic task arrival patterns with configurable jitter
- **Task Dependencies**: DAG-based scheduling with dependency constraints

### 🧠 Intelligent Neural Architecture Search
- **Ensemble Surrogate Models**: Combines multiple ML models for robust predictions
- **Active Learning**: Uncertainty-based exploration for efficient architecture discovery
- **Expanded Search Space**: 13+ architectural parameters including layers, activations, regularization
- **Real-time Uncertainty Estimation**: Visual confidence metrics and improvement tracking

### 📊 Interactive Dashboard
- **Real-time Visualization**: Gantt charts, performance comparisons, uncertainty evolution
- **Advanced Configuration**: User-selectable simulation parameters and complexity levels
- **Active Learning Controls**: Interactive retraining with progress tracking
- **Performance Analysis**: Comprehensive metrics and improvement analysis

## 🏗️ Architecture

### Core Components

```
NAS Project/
├── dashboard.py          # Interactive Streamlit dashboard
├── nas_controller.py     # Ensemble surrogate model and NAS logic
├── simulator.py          # Advanced multi-core simulation engine
├── workload.py           # Workload generation with advanced features
├── policies/
│   ├── baseline.py       # Round-robin baseline policy
│   └── neural_policy.py  # Neural network-based scheduling policy
└── requirements.txt      # Python dependencies
```

### Technical Stack

- **Frontend**: Streamlit (Interactive Dashboard)
- **ML Framework**: PyTorch (Neural Networks), Scikit-learn (Ensemble Models)
- **Visualization**: Plotly (Interactive Charts)
- **Simulation**: Custom discrete-event simulator
- **Active Learning**: Uncertainty-based exploration strategies

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NAS
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Quick Start

1. **Launch the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Configure simulation parameters** in the sidebar:
   - Select workload type (balanced, bursty, real-time, memory-bound)
   - Choose complexity level (basic/advanced)
   - Set number of tasks and cores
   - Configure advanced options (if using advanced mode)

4. **Run simulation** and explore results!

### Advanced Features

#### Basic Mode
- Simple workload generation
- Standard simulation parameters
- Quick experimentation

#### Advanced Mode
- **Heterogeneous Cores**: Enable different core types
- **Resource Contention**: Configure cache, memory, and I/O contention levels
- **Energy Modeling**: Track energy consumption
- **Preemption/Interrupts**: Set preemption and interrupt frequencies
- **Stochastic Arrivals**: Add realistic arrival jitter

#### Active Learning
- **Pre-training**: Automatic 30-iteration surrogate model training
- **Retrain Button**: Run 25 additional active learning iterations
- **Uncertainty Tracking**: Visual confidence improvement metrics

## 📈 Performance Metrics

The system evaluates architectures based on multiple objectives:

- **Latency**: Average task completion time
- **Utilization**: Core resource utilization efficiency
- **Deadline Compliance**: Number of missed deadlines
- **Energy Efficiency**: Energy consumption per task
- **Resource Contention**: Cache misses, memory wait times, I/O bottlenecks

## 🔬 Technical Details

### Neural Architecture Search Space

The system explores architectures with the following parameters:

- **Network Structure**: Number of layers (1-6), hidden dimensions (16-512)
- **Activation Functions**: ReLU, Tanh, LeakyReLU, ELU, Swish
- **Regularization**: Dropout, Layer Normalization
- **Training**: Learning rates, optimizers, batch sizes
- **Advanced**: Attention heads, residual connections

### Ensemble Surrogate Model

Combines four different models with adaptive weighting:

1. **Random Forest**: Robust baseline performance
2. **Gradient Boosting**: High accuracy predictions
3. **Neural Network**: Complex non-linear relationships
4. **Gaussian Process**: Uncertainty estimation

### Active Learning Strategy

- **Exploration vs Exploitation**: Configurable balance (0.0-1.0)
- **Uncertainty Sampling**: Prioritizes architectures with high prediction uncertainty
- **Multi-objective Optimization**: Considers multiple performance metrics
- **Real-time Adaptation**: Model weights updated based on cross-validation performance

## 🎯 Use Cases

### Research Applications
- **Scheduling Algorithm Development**: Discover novel scheduling policies
- **Hardware-Software Co-design**: Optimize for specific hardware configurations
- **Performance Analysis**: Understand trade-offs between different objectives

### Educational Applications
- **Computer Architecture**: Learn about multi-core systems and scheduling
- **Machine Learning**: Understand NAS and active learning concepts
- **System Simulation**: Explore discrete-event simulation techniques

### Industrial Applications
- **Data Center Optimization**: Optimize task scheduling for cloud environments
- **Embedded Systems**: Real-time scheduling for IoT and edge computing
- **HPC Clusters**: High-performance computing workload optimization

## 🔧 Configuration

### Workload Types

- **Balanced**: Evenly distributed tasks with moderate complexity
- **Bursty**: Tasks arrive in bursts, simulating real-world patterns
- **Real-time**: Tasks with strict deadlines and priority levels
- **Memory-bound**: Tasks with high memory requirements

### Simulation Parameters

- **Number of Tasks**: 10-100 tasks per simulation
- **Number of Cores**: 2-16 cores with configurable types
- **Simulation Time**: Configurable maximum simulation duration
- **Random Seed**: Reproducible results for experimentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the interactive dashboard framework
- **PyTorch** for neural network implementation
- **Scikit-learn** for ensemble machine learning models
- **Plotly** for interactive visualizations

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a research and educational project. For production use, additional testing, validation, and optimization may be required. 