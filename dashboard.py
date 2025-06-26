import streamlit as st
from workload import generate_workload
from simulator import Simulator
from policies.baseline import RoundRobinPolicy
from policies.neural_policy import NeuralSchedulerPolicy
from nas_controller import SurrogateModel
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import List, Dict
import numpy as np

# Enhanced model building logic for expanded search space
def build_model(arch, num_cores):
    layers = []
    input_dim = 3  # arrival, duration, free_cores
    
    # Remove the problematic MultiheadAttention for now
    # We'll implement a simpler attention mechanism if needed
    # if arch.get("attention_heads", 0) > 0:
    #     attention_dim = arch["hidden_dim"]
    #     layers.append(nn.MultiheadAttention(attention_dim, arch["attention_heads"], batch_first=True))
    #     layers.append(nn.LayerNorm(attention_dim))
    
    for i in range(arch["num_layers"]):
        # Add residual connection if specified
        if arch.get("residual_connections", False) and i > 0:
            layers.append(ResidualBlock(input_dim, arch["hidden_dim"]))
        else:
            layers.append(nn.Linear(input_dim, arch["hidden_dim"]))
        
        # Add layer normalization if specified or if regularization is batch_norm
        if arch.get("layer_normalization", False) or arch.get("regularization") == "batch_norm":
            layers.append(nn.LayerNorm(arch["hidden_dim"]))
        
        # Add activation function
        if arch["activation"] == "relu":
            layers.append(nn.ReLU())
        elif arch["activation"] == "tanh":
            layers.append(nn.Tanh())
        elif arch["activation"] == "leaky_relu":
            layers.append(nn.LeakyReLU())
        elif arch["activation"] == "elu":
            layers.append(nn.ELU())
        elif arch["activation"] == "swish":
            layers.append(nn.SiLU())  # SiLU is the same as Swish
        
        # Add dropout if specified
        if arch.get("regularization") == "dropout" and arch.get("dropout_rate", 0.0) > 0:
            layers.append(nn.Dropout(arch["dropout_rate"]))
        
        input_dim = arch["hidden_dim"]
    
    # Output layer
    layers.append(nn.Linear(input_dim, num_cores))
    
    return nn.Sequential(*layers)

# Residual block for residual connections
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out

# Set page config for better UI
st.set_page_config(
    page_title="Neural Scheduling Architecture Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown('<h1 class="main-header"> NAS for Multi-Core Scheduling</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    scenario = st.selectbox(
        "Workload Type:",
        ["balanced", "bursty", "real_time", "memory_bound"],
        help="Choose the type of workload to simulate"
    )
    
    complexity_level = st.selectbox(
        "Simulation Complexity:",
        ["basic", "advanced"],
        help="Choose the complexity level of the simulation"
    )
    
    num_tasks = st.slider("Number of Tasks", 10, 100, 20, help="Total number of tasks to schedule")
    num_cores = st.slider("Number of Cores", 2, 16, 4, help="Number of available CPU cores")
    seed = st.number_input("Random Seed", value=42, help="Seed for reproducible results")
    
    # Advanced simulation options
    if complexity_level == "advanced":
        st.markdown("---")
        st.markdown("### 🚀 Advanced Simulation Options")
        
        # Heterogeneous cores
        st.session_state.heterogeneous_cores = st.checkbox("Heterogeneous Cores", 
                                                          value=st.session_state.get('heterogeneous_cores', True), 
                                                          help="Enable different core types (big.LITTLE, GPU/CPU)")
        
        if st.session_state.heterogeneous_cores:
            st.info("Core types will be automatically assigned: Performance, Efficient, and Standard cores")
        
        # Resource contention
        st.session_state.resource_contention = st.checkbox("Resource Contention", 
                                                          value=st.session_state.get('resource_contention', True),
                                                          help="Simulate shared caches, memory bandwidth, and I/O bottlenecks")
        
        if st.session_state.resource_contention:
            st.session_state.cache_contention = st.slider("Cache Contention Level", 0.0, 1.0, 
                                                        value=st.session_state.get('cache_contention', 0.3),
                                                        help="Level of cache contention (0=none, 1=high)")
            st.session_state.memory_contention = st.slider("Memory Contention Level", 0.0, 1.0, 
                                                         value=st.session_state.get('memory_contention', 0.2),
                                                         help="Level of memory bandwidth contention")
            st.session_state.io_contention = st.slider("I/O Contention Level", 0.0, 1.0, 
                                                     value=st.session_state.get('io_contention', 0.1),
                                                     help="Level of I/O bottleneck simulation")
        
        # Energy modeling
        st.session_state.energy_modeling = st.checkbox("Energy Modeling", 
                                                      value=st.session_state.get('energy_modeling', True),
                                                      help="Track and optimize for energy consumption")
        
        # Preemption and interrupts
        st.session_state.preemption_enabled = st.checkbox("Preemption/Interrupts", 
                                                         value=st.session_state.get('preemption_enabled', True),
                                                         help="Allow tasks to be paused/migrated")
        
        if st.session_state.preemption_enabled:
            st.session_state.preemption_frequency = st.slider("Preemption Frequency", 0.0, 1.0, 
                                                            value=st.session_state.get('preemption_frequency', 0.1),
                                                            help="Frequency of task preemptions")
            st.session_state.interrupt_frequency = st.slider("Interrupt Frequency", 0.0, 1.0, 
                                                           value=st.session_state.get('interrupt_frequency', 0.05),
                                                           help="Frequency of system interrupts")
        
        # Stochastic arrivals
        st.session_state.stochastic_arrivals = st.checkbox("Stochastic Arrivals", 
                                                          value=st.session_state.get('stochastic_arrivals', True),
                                                          help="Use realistic arrival patterns with jitter")
        
        if st.session_state.stochastic_arrivals:
            st.session_state.arrival_jitter = st.slider("Arrival Jitter", 0.0, 2.0, 
                                                      value=st.session_state.get('arrival_jitter', 0.5),
                                                      help="Random jitter in task arrival times")
    
    st.markdown("---")
    st.markdown("### Active Learning Controls")
    exploration_weight = st.slider("Exploration Weight", 0.0, 1.0, 0.3, 
                                  help="Balance between exploration (uncertainty) and exploitation (performance)")
    
    if st.button("🔄 Retrain Surrogate Model", key="retrain_btn"):
        st.session_state.retrain_model = True
    
    st.markdown("---")
    st.markdown("### Workload Scenarios:")
    st.markdown("- **Balanced**: Evenly distributed tasks")
    st.markdown("- **Bursty**: Tasks arrive in bursts")
    st.markdown("- **Real-time**: Tasks with deadlines")
    st.markdown("- **Memory-bound**: Tasks with memory requirements")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Simulation Parameters")
    st.info(f"**Scenario:** {scenario.title()}")
    st.info(f"**Tasks:** {num_tasks}")
    st.info(f"**Cores:** {num_cores}")

with col2:
    st.subheader("🚀 Run Simulation")
    if st.button("Start Simulation", type="primary", key="run_btn"):
        st.session_state.run_simulation = True

# Run simulation when button is clicked
if st.session_state.get('run_simulation', False):
    st.session_state.run_simulation = False
    
    with st.spinner("Running advanced simulation..."):
        # Get advanced simulation options
        advanced_options = {}
        if complexity_level == "advanced":
            # Store advanced options in session state
            if 'heterogeneous_cores' not in st.session_state:
                st.session_state.heterogeneous_cores = True
            if 'resource_contention' not in st.session_state:
                st.session_state.resource_contention = True
            if 'energy_modeling' not in st.session_state:
                st.session_state.energy_modeling = True
            if 'preemption_enabled' not in st.session_state:
                st.session_state.preemption_enabled = True
            if 'stochastic_arrivals' not in st.session_state:
                st.session_state.stochastic_arrivals = True
            if 'cache_contention' not in st.session_state:
                st.session_state.cache_contention = 0.3
            if 'memory_contention' not in st.session_state:
                st.session_state.memory_contention = 0.2
            if 'io_contention' not in st.session_state:
                st.session_state.io_contention = 0.1
            if 'preemption_frequency' not in st.session_state:
                st.session_state.preemption_frequency = 0.1
            if 'interrupt_frequency' not in st.session_state:
                st.session_state.interrupt_frequency = 0.05
            if 'arrival_jitter' not in st.session_state:
                st.session_state.arrival_jitter = 0.5
            
            advanced_options = {
                "heterogeneous_cores": st.session_state.heterogeneous_cores,
                "resource_contention": st.session_state.resource_contention,
                "energy_modeling": st.session_state.energy_modeling,
                "preemption_enabled": st.session_state.preemption_enabled,
                "stochastic_arrivals": st.session_state.stochastic_arrivals,
                "cache_contention": st.session_state.cache_contention,
                "memory_contention": st.session_state.memory_contention,
                "io_contention": st.session_state.io_contention,
                "preemption_frequency": st.session_state.preemption_frequency,
                "interrupt_frequency": st.session_state.interrupt_frequency,
                "arrival_jitter": st.session_state.arrival_jitter
            }
        
        # Generate workload with complexity level and advanced options
        workload_meta = {
            "scenario": scenario,
            "complexity_level": complexity_level,
            "num_tasks": num_tasks,
            "num_cores": num_cores,
            **advanced_options
        }
        
        # Calculate workload meta-features
        tasks = generate_workload(scenario, num_tasks=num_tasks, seed=seed, 
                                complexity_level=complexity_level, **advanced_options)
        
        # Extract workload features for surrogate
        durations = [t.duration for t in tasks]
        memory_demands = [t.memory_demand for t in tasks if t.memory_demand]
        priorities = [t.priority for t in tasks]
        dependencies = [len(t.dependencies) for t in tasks]
        
        workload_features = {
            "avg_task_duration": np.mean(durations),
            "task_duration_variance": np.var(durations),
            "memory_intensity": np.mean(memory_demands) if memory_demands else 0.5,
            "io_intensity": np.mean([getattr(t, 'io_operations', 0) for t in tasks]),
            "dependency_density": np.mean(dependencies),
            "priority_variance": np.var(priorities) if priorities else 0.5,
            "num_cores": num_cores,
            "core_heterogeneity": 0.5 if advanced_options.get('heterogeneous_cores', False) else 0.0,
            "memory_bandwidth": 25.6,
            "cache_size": 8
        }

        # Pre-train surrogate with 30 random architectures
        surrogate = SurrogateModel()
        pretrain_iters = 30
        pretrain_bar = st.progress(0, text="Pre-training surrogate model...")
        for i in range(pretrain_iters):
            arch = surrogate._generate_random_arch()
            model = build_model(arch, num_cores)
            policy = NeuralSchedulerPolicy(model)
            sim = Simulator(num_cores=num_cores, scheduling_policy=policy)
            sim.load_tasks(tasks)
            result = sim.run(max_time=100)
            # Use a comprehensive score including energy and resource contention
            score = -result["avg_latency"] - 2 * result["deadline_misses"]
            if advanced_options.get('energy_modeling', False):
                score -= 0.1 * result.get("total_energy", 0)
            if advanced_options.get('resource_contention', False):
                score -= 0.5 * result.get("resource_contention_score", 0)
            surrogate.update(arch, score, workload_meta=workload_features)
            pretrain_bar.progress((i+1)/pretrain_iters, text=f"Pre-training surrogate model... {i+1}/{pretrain_iters}")
        pretrain_bar.empty()

    # Baseline Simulation
    baseline_policy = RoundRobinPolicy()
    sim1 = Simulator(num_cores=num_cores, scheduling_policy=baseline_policy)
    sim1.load_tasks(tasks)
    baseline_result = sim1.run(max_time=100)

    # Surrogate NAS with active learning
    top_architectures = surrogate.suggest(
        n_candidates=20, 
        workload_meta=workload_features,
        exploration_weight=exploration_weight
    )
        
    # Use the best architecture for simulation
    best_arch, _, predicted_perf, uncertainty = top_architectures[0]
    model = build_model(best_arch, num_cores)
    neural_policy = NeuralSchedulerPolicy(model)
    sim2 = Simulator(num_cores=num_cores, scheduling_policy=neural_policy)
    sim2.load_tasks(tasks)
    neural_result = sim2.run(max_time=100)

    # Handle active learning retraining
    if st.session_state.get('retrain_model', False):
        st.session_state.retrain_model = False
        
        with st.spinner("Running active learning iterations..."):
            # Run additional iterations to improve the surrogate model
            active_learning_iters = 25
            al_progress_bar = st.progress(0, text="Active learning in progress...")
            
            for i in range(active_learning_iters):
                # Generate new architectures based on uncertainty
                candidates = surrogate.suggest(n_candidates=10, workload_meta=workload_features, 
                                            exploration_weight=0.7)  # Higher exploration
                
                for arch, _, _, uncertainty in candidates[:3]:  # Top 3 most uncertain
                    model = build_model(arch, num_cores)
                    policy = NeuralSchedulerPolicy(model)
                    sim = Simulator(num_cores=num_cores, scheduling_policy=policy)
                    sim.load_tasks(tasks)
                    result = sim.run(max_time=100)
                    
                    # Comprehensive scoring
                    score = -result["avg_latency"] - 2 * result["deadline_misses"]
                    if advanced_options.get('energy_modeling', False):
                        score -= 0.1 * result.get("total_energy", 0)
                    if advanced_options.get('resource_contention', False):
                        score -= 0.5 * result.get("resource_contention_score", 0)
                    
                    surrogate.update(arch, score, workload_meta=workload_features)
                
                al_progress_bar.progress((i+1)/active_learning_iters, 
                                       text=f"Active learning iteration {i+1}/{active_learning_iters}")
            
            al_progress_bar.empty()
            
            # Re-run NAS with improved surrogate
            top_architectures = surrogate.suggest(
                n_candidates=20, 
                workload_meta=workload_features,
                exploration_weight=exploration_weight
            )
            
            # Update results with new best architecture
            best_arch, _, predicted_perf, uncertainty = top_architectures[0]
            model = build_model(best_arch, num_cores)
    neural_policy = NeuralSchedulerPolicy(model)
    sim2 = Simulator(num_cores=num_cores, scheduling_policy=neural_policy)
    sim2.load_tasks(tasks)
    neural_result = sim2.run(max_time=100)

    st.success("✅ Active learning completed! Surrogate model improved.")

    # Display results
    st.success("✅ Advanced simulation completed!")
    
    # Model Performance Section
    st.subheader("🧠 Ensemble Model Performance")
    model_performance = surrogate.get_model_performance()
    
    if model_performance:
        perf_cols = st.columns(len(model_performance))
        for i, (model_name, perf) in enumerate(model_performance.items()):
            with perf_cols[i]:
                st.metric(f"{model_name.replace('_', ' ').title()}", 
                         f"{perf['r2']:.3f}", 
                         f"±{perf.get('std', 0):.3f}")
    
    # Top 5 Neural Architectures with Uncertainty
    st.subheader("🏆 Top 5 Neural Architectures (with Uncertainty)")
    
    # Create visual cards for top 5 architectures
    arch_cols = st.columns(5)
    
    for i, (arch, combined_score, predicted_perf, uncertainty) in enumerate(top_architectures[:5]):
        with arch_cols[i]:
            # Color based on uncertainty (red = high uncertainty, green = low)
            uncertainty_color = f"rgb({int(255 * uncertainty)}, {int(255 * (1-uncertainty))}, 0)"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {'#ff6b6b' if i==0 else '#4ecdc4' if i==1 else '#45b7d1' if i==2 else '#96c93d' if i==3 else '#f39c12'}, {'#ee5a52' if i==0 else '#44a08d' if i==1 else '#2c3e50' if i==2 else '#27ae60' if i==3 else '#e67e22'});
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                font-size: 0.9rem;
                border: 3px solid {uncertainty_color};
            ">
                <h3 style="margin: 0 0 1rem 0; font-size: 1.3rem;">{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '🏆' if i==3 else '🎯'} Rank #{i+1}</h3>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Layers:</strong> {arch["num_layers"]}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Hidden:</strong> {arch["hidden_dim"]}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Activation:</strong> {arch["activation"].upper()}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Predicted:</strong> {predicted_perf:.2f}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Uncertainty:</strong> {uncertainty:.3f}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Combined:</strong> {combined_score:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Best Architecture Details with expanded information
    st.subheader("🧠 Selected Architecture (Best)")
    
    # Create a detailed architecture display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Layers", best_arch["num_layers"])
        st.metric("Hidden Dimension", best_arch["hidden_dim"])
        st.metric("Activation", best_arch["activation"].upper())
    with col2:
        st.metric("Regularization", best_arch.get("regularization", "linear").upper())
        st.metric("Dropout Rate", f"{best_arch.get('dropout_rate', 0.0):.1f}")
        st.metric("Learning Rate", best_arch.get("learning_rate", 0.001))
    with col3:
        st.metric("Optimizer", best_arch.get("optimizer", "adam").upper())
        st.metric("Attention Heads", best_arch.get("attention_heads", 0))
        st.metric("Residual Connections", "✓" if best_arch.get("residual_connections", False) else "✗")

    # Enhanced Performance comparison with new metrics
    st.subheader("📈 Advanced Performance Comparison")
    
    # Create metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🔄 Baseline (Round Robin)")
        st.metric("Avg Latency", f"{baseline_result['avg_latency']:.2f}")
        st.metric("Avg Utilization", f"{baseline_result['avg_utilization']:.2%}")
        st.metric("Deadline Misses", baseline_result['deadline_misses'])
        st.metric("Energy Efficiency", f"{baseline_result.get('energy_efficiency', 0):.2f}")
        st.metric("Resource Contention", f"{baseline_result.get('resource_contention_score', 0):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🧠 Neural Policy")
        st.metric("Avg Latency", f"{neural_result['avg_latency']:.2f}")
        st.metric("Avg Utilization", f"{neural_result['avg_utilization']:.2%}")
        st.metric("Deadline Misses", neural_result['deadline_misses'])
        st.metric("Energy Efficiency", f"{neural_result.get('energy_efficiency', 0):.2f}")
        st.metric("Resource Contention", f"{neural_result.get('resource_contention_score', 0):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Performance improvement
    improvement = ((baseline_result['avg_latency'] - neural_result['avg_latency']) / baseline_result['avg_latency']) * 100
    if improvement > 0:
        st.success(f"🎉 Neural policy improved latency by {improvement:.1f}%!")
    else:
        st.warning(f"⚠️ Neural policy increased latency by {abs(improvement):.1f}%")

    # Metric Comparison Bar Chart - Fixed to properly show deadline misses
    st.subheader("📊 Detailed Metrics Comparison")
    metrics = ["avg_latency", "avg_utilization", "deadline_misses", "num_completed"]
    metric_labels = ["Average Latency", "Average Utilization", "Deadline Misses", "Completed Tasks"]
    baseline_vals = [baseline_result[m] for m in metrics]
    neural_vals = [neural_result[m] for m in metrics]

    # Normalize utilization for better visualization
    baseline_vals[1] *= 100
    neural_vals[1] *= 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Baseline',
        x=metric_labels,
        y=baseline_vals,
        marker_color='#1f77b4'
    ))
    fig.add_trace(go.Bar(
        name='Neural Policy',
        x=metric_labels,
        y=neural_vals,
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        xaxis_title="Metrics",
        yaxis_title="Values",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Task Schedule Table with proper deadline and memory values using getattr
    st.subheader("📅 Neural Policy Task Schedule")
    
    # Create task table with proper values using getattr like core_id
    task_table = []
    for t in sim2.completed_tasks:
        task_data = {
        "Task ID": t.task_id,
        "Arrival": t.arrival_time,
        "Duration": t.duration,
            "Start": t.start_time if t.start_time is not None else "Not started",
            "End": t.end_time if t.end_time is not None else "Not finished",
            "Core": getattr(t, 'core_id', 'Unassigned')
        }
        
        # Handle deadline display using getattr like core_id
        deadline = getattr(t, 'deadline', None)
        if deadline is not None:
            if t.end_time is not None and t.end_time > deadline:
                task_data["Deadline"] = f"{deadline} ⚠️ (Missed)"
            else:
                task_data["Deadline"] = f"{deadline} ✅"
        else:
            task_data["Deadline"] = "No deadline"
        
        # Handle memory demand display using getattr like core_id
        memory_demand = getattr(t, 'memory_demand', None)
        if memory_demand is not None:
            task_data["Memory"] = f"{memory_demand} GB"
        else:
            task_data["Memory"] = "Not specified"
        
        task_table.append(task_data)

    df_tasks = pd.DataFrame(task_table)
    st.dataframe(df_tasks, use_container_width=True)

    # Fixed Gantt Chart
    st.subheader("📅 Task Execution Timeline (Gantt Chart)")
    
    # Prepare data for Gantt chart
    gantt_data = []
    for t in sim2.completed_tasks:
        if t.start_time is not None and t.end_time is not None:
            gantt_data.append({
                "Task": f"Task {t.task_id}",
                "Core": f"Core {getattr(t, 'core_id', '?')}",
                "Start": t.start_time,
                "End": t.end_time,
                "Duration": t.end_time - t.start_time
            })

    if gantt_data:
        df_gantt = pd.DataFrame(gantt_data)
        
        # Create Gantt chart using plotly
        fig_gantt = go.Figure()
        
        # Add bars for each task
        for _, row in df_gantt.iterrows():
            fig_gantt.add_trace(go.Bar(
                name=row['Task'],
                x=[row['Duration']],
                y=[row['Core']],
                orientation='h',
                base=row['Start'],
                hovertemplate=f"Task: {row['Task']}<br>" +
                            f"Core: {row['Core']}<br>" +
                            f"Start: {row['Start']}<br>" +
                            f"End: {row['End']}<br>" +
                            f"Duration: {row['Duration']}<extra></extra>"
            ))
        
        fig_gantt.update_layout(
            title="Task Execution Timeline",
            xaxis_title="Time",
            yaxis_title="Cores",
            height=400,
            showlegend=False,
            barmode='stack'
        )
        
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.warning("No completed tasks to display in Gantt chart")

    # Additional insights
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if neural_result['deadline_misses'] < baseline_result['deadline_misses']:
            st.success("✅ Better deadline compliance")
        elif neural_result['deadline_misses'] > baseline_result['deadline_misses']:
            st.error("❌ Worse deadline compliance")
        else:
            st.info("➖ Same deadline compliance")
    
    with col2:
        if neural_result['avg_utilization'] > baseline_result['avg_utilization']:
            st.success("✅ Higher resource utilization")
        else:
            st.warning("⚠️ Lower resource utilization")
    
    with col3:
        if neural_result['num_completed'] > baseline_result['num_completed']:
            st.success("✅ More tasks completed")
        else:
            st.warning("⚠️ Fewer tasks completed")

    # New section: Active Learning Insights
    st.subheader("🔍 Active Learning Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Exploration Weight:** {exploration_weight:.1f}")
        st.info(f"**Model Uncertainty Range:** {min([a[3] for a in top_architectures]):.3f} - {max([a[3] for a in top_architectures]):.3f}")
        st.info(f"**Average Uncertainty:** {np.mean([a[3] for a in top_architectures]):.3f}")
        
        # Model confidence metrics
        if len(surrogate.uncertainty_history) > 1:
            initial_uncertainty = np.mean(surrogate.uncertainty_history[0])
            final_uncertainty = np.mean(surrogate.uncertainty_history[-1])
            confidence_improvement = (initial_uncertainty - final_uncertainty) / initial_uncertainty * 100
            st.success(f"**Confidence Improvement:** {confidence_improvement:.1f}%")
        
        # Ensemble model performance
        model_performance = surrogate.get_model_performance()
        if model_performance:
            st.info("**Ensemble Model Performance:**")
            for model_name, perf in model_performance.items():
                st.write(f"• {model_name.replace('_', ' ').title()}: R² = {perf['r2']:.3f}")
    
    with col2:
        # Uncertainty trend chart
        if len(surrogate.uncertainty_history) > 1:
            fig_uncertainty = go.Figure()
            
            # Plot uncertainty evolution for top 5 architectures
            for rank in range(min(5, len(surrogate.uncertainty_history[0]))):
                uncertainties = [history[rank] if rank < len(history) else 0 
                               for history in surrogate.uncertainty_history[-10:]]  # Last 10 iterations
                fig_uncertainty.add_trace(go.Scatter(
                    y=uncertainties,
                    mode='lines+markers',
                    name=f'Rank #{rank+1}',
                    line=dict(width=2)
                ))
            
            fig_uncertainty.update_layout(
                title="Uncertainty Evolution (Last 10 Iterations)",
                xaxis_title="Iteration",
                yaxis_title="Uncertainty",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_uncertainty, use_container_width=True)
    
    # Advanced simulation insights
    if complexity_level == "advanced":
        st.subheader("🚀 Advanced Simulation Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if advanced_options.get('heterogeneous_cores', False):
                st.success("✅ Heterogeneous cores enabled")
                st.info("Performance, Efficient, and Standard cores")
            else:
                st.info("ℹ️ Homogeneous cores")
        
        with col2:
            if advanced_options.get('resource_contention', False):
                st.success("✅ Resource contention simulation")
                st.info(f"Cache: {advanced_options.get('cache_contention', 0):.1f}")
                st.info(f"Memory: {advanced_options.get('memory_contention', 0):.1f}")
                st.info(f"I/O: {advanced_options.get('io_contention', 0):.1f}")
            else:
                st.info("ℹ️ No resource contention")
        
        with col3:
            if advanced_options.get('energy_modeling', False):
                st.success("✅ Energy modeling enabled")
                st.metric("Total Energy", f"{neural_result.get('total_energy', 0):.2f}")
                st.metric("Energy Efficiency", f"{neural_result.get('energy_efficiency', 0):.2f}")
            else:
                st.info("ℹ️ Energy modeling disabled")
    
    # Performance improvement analysis
    st.subheader("📊 Performance Improvement Analysis")
    
    improvements = {}
    for metric in ['avg_latency', 'avg_utilization', 'deadline_misses', 'num_completed']:
        baseline_val = baseline_result[metric]
        neural_val = neural_result[metric]
        
        if baseline_val != 0:
            if metric == 'avg_latency':
                # Lower is better for latency
                improvement = ((baseline_val - neural_val) / baseline_val) * 100
            else:
                # Higher is better for others
                improvement = ((neural_val - baseline_val) / baseline_val) * 100
        else:
            improvement = 0
        
        improvements[metric] = improvement
    
    # Create improvement visualization
    fig_improvements = go.Figure()
    
    metrics_labels = ['Latency', 'Utilization', 'Deadline Misses', 'Completed Tasks']
    improvements_list = [improvements['avg_latency'], improvements['avg_utilization'], 
                        improvements['deadline_misses'], improvements['num_completed']]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements_list]
    
    fig_improvements.add_trace(go.Bar(
        x=metrics_labels,
        y=improvements_list,
        marker_color=colors,
        text=[f"{imp:+.1f}%" for imp in improvements_list],
        textposition='auto'
    ))
    
    fig_improvements.update_layout(
        title="Performance Improvements (%)",
        xaxis_title="Metrics",
        yaxis_title="Improvement (%)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_improvements, use_container_width=True)
    
    # Summary of improvements
    positive_improvements = sum(1 for imp in improvements_list if imp > 0)
    total_metrics = len(improvements_list)
    
    if positive_improvements > total_metrics / 2:
        st.success(f"🎉 Neural policy improved {positive_improvements}/{total_metrics} metrics!")
    elif positive_improvements > 0:
        st.warning(f"⚠️ Neural policy improved {positive_improvements}/{total_metrics} metrics")
    else:
        st.error("❌ Neural policy did not improve any metrics")
