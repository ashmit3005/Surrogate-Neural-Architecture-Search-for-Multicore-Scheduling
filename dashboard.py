import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from workload import generate_workload
from simulator import Simulator
from policies.baseline import RoundRobinPolicy
from policies.neural_policy import NeuralSchedulerPolicy
from nas_controller import SurrogateModel
import torch.nn as nn
import torch
import random
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="NAS Neural Scheduler Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .improvement-positive {
        color: #28a745;
        font-weight: bold;
    }
    .improvement-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .architecture-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Define model building logic
def build_model(arch, num_cores):
    layers = []
    input_dim = 3
    for _ in range(arch["num_layers"]):
        layers.append(nn.Linear(input_dim, arch["hidden_dim"]))
        layers.append(nn.ReLU() if arch["activation"] == "relu" else nn.Tanh())
        input_dim = arch["hidden_dim"]
    layers.append(nn.Linear(input_dim, num_cores))
    return nn.Sequential(*layers)

# Enhanced simulator with detailed tracking
class EnhancedSimulator(Simulator):
    def __init__(self, num_cores, scheduling_policy, verbose=False):
        super().__init__(num_cores, scheduling_policy, verbose)
        self.core_utilization_history = []
        self.task_completion_history = []
        self.queue_length_history = []
        
    def run(self, max_time=1000):
        active_tasks = []
        while self.time < max_time:
            # Track queue length
            self.queue_length_history.append(len(active_tasks))
            
            # Add tasks that arrive at this time
            while self.task_queue and self.task_queue[0].arrival_time <= self.time:
                active_tasks.append(self.task_queue.popleft())

            # Ask the policy to decide task-core mapping
            assignments = self.policy.assign_tasks(active_tasks, self.cores, self.time)

            for task, core in assignments:
                if task.memory_demand and task.memory_demand > core.memory_capacity:
                    continue
                core.assign(task, self.time)
                active_tasks.remove(task)

            # Step all cores and track utilization
            core_utils = []
            for core in self.cores:
                finished = core.step(self.time)
                if finished:
                    self.completed_tasks.append(finished)
                    self.task_completion_history.append({
                        'time': self.time,
                        'task_id': finished.task_id,
                        'latency': self.time - finished.arrival_time
                    })
                core_utils.append(1 if core.current_task else 0)
            
            self.core_utilization_history.append(core_utils)
            self.time += 1

        return self.evaluate()

# Main dashboard
st.markdown('<h1 class="main-header">🚀 NAS Neural Scheduler Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")
scenario = st.sidebar.selectbox(
    "Workload Type:",
    ["balanced", "bursty", "real_time", "memory_bound"],
    help="Choose the type of workload to simulate"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    num_tasks = st.slider("Number of Tasks", 10, 100, 30)
    num_cores = st.slider("Number of Cores", 2, 16, 4)
with col2:
    max_time = st.slider("Max Time Steps", 50, 500, 200)
    seed = st.number_input("Random Seed", value=42)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    show_detailed_plots = st.checkbox("Show Detailed Plots", value=True)
    show_architecture_analysis = st.checkbox("Show Architecture Analysis", value=True)
    auto_refresh = st.checkbox("Auto-refresh on parameter change", value=False)

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary") or auto_refresh:
    with st.spinner("Running simulation..."):
        # Generate workload
        workload_meta = {"scenario": scenario}
        tasks = generate_workload(scenario, num_tasks=num_tasks, seed=seed)

        # Baseline Simulation
        baseline_policy = RoundRobinPolicy()
        sim1 = EnhancedSimulator(num_cores=num_cores, scheduling_policy=baseline_policy)
        sim1.load_tasks(tasks)
        baseline_result = sim1.run(max_time=max_time)

        # Neural Simulation
        surrogate = SurrogateModel()
        arch = surrogate.suggest(workload_meta=workload_meta)[0][0]
        model = build_model(arch, num_cores)
        neural_policy = NeuralSchedulerPolicy(model)
        sim2 = EnhancedSimulator(num_cores=num_cores, scheduling_policy=neural_policy)
        sim2.load_tasks(tasks)
        neural_result = sim2.run(max_time=max_time)

        # Store results in session state
        st.session_state.baseline_result = baseline_result
        st.session_state.neural_result = neural_result
        st.session_state.baseline_sim = sim1
        st.session_state.neural_sim = sim2
        st.session_state.architecture = arch
        st.session_state.tasks = tasks

# Display results if available
if hasattr(st.session_state, 'baseline_result'):
    baseline_result = st.session_state.baseline_result
    neural_result = st.session_state.neural_result
    baseline_sim = st.session_state.baseline_sim
    neural_sim = st.session_state.neural_sim
    architecture = st.session_state.architecture
    tasks = st.session_state.tasks

    # Header with key metrics
    st.markdown("## 📊 Performance Overview")
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Latency",
            f"{baseline_result['avg_latency']:.2f}",
            f"{neural_result['avg_latency'] - baseline_result['avg_latency']:.2f}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Avg Utilization",
            f"{baseline_result['avg_utilization']:.2%}",
            f"{neural_result['avg_utilization'] - baseline_result['avg_utilization']:.2%}"
        )
    
    with col3:
        st.metric(
            "Tasks Completed",
            baseline_result['num_completed'],
            neural_result['num_completed'] - baseline_result['num_completed']
        )
    
    with col4:
        st.metric(
            "Deadline Misses",
            baseline_result['deadline_misses'],
            neural_result['deadline_misses'] - baseline_result['deadline_misses'],
            delta_color="inverse"
        )

    # Architecture Analysis
    if show_architecture_analysis:
        st.markdown("## 🏗️ Neural Architecture Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Selected Architecture")
            arch_df = pd.DataFrame([architecture])
            st.dataframe(arch_df, use_container_width=True)
            
            # Architecture visualization
            fig = go.Figure()
            
            # Create network visualization
            layers = [3] + [architecture['hidden_dim']] * architecture['num_layers'] + [num_cores]
            layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(architecture['num_layers'])] + ['Output']
            
            for i, (layer_size, layer_name) in enumerate(zip(layers, layer_names)):
                y_positions = np.linspace(0, 1, layer_size)
                for j, y in enumerate(y_positions):
                    fig.add_trace(go.Scatter(
                        x=[i] * layer_size,
                        y=y_positions,
                        mode='markers',
                        marker=dict(size=10, color='lightblue'),
                        name=layer_name,
                        showlegend=False
                    ))
                    
                    # Add connections to next layer
                    if i < len(layers) - 1:
                        next_y_positions = np.linspace(0, 1, layers[i+1])
                        for next_y in next_y_positions:
                            fig.add_trace(go.Scatter(
                                x=[i, i+1],
                                y=[y, next_y],
                                mode='lines',
                                line=dict(color='gray', width=0.5),
                                showlegend=False
                            ))
            
            fig.update_layout(
                title="Neural Network Architecture",
                xaxis_title="Layers",
                yaxis_title="Neurons",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Architecture Details")
            st.markdown(f"""
            <div class="architecture-box">
            <h4>Configuration</h4>
            <p><strong>Layers:</strong> {architecture['num_layers']}</p>
            <p><strong>Hidden Dim:</strong> {architecture['hidden_dim']}</p>
            <p><strong>Activation:</strong> {architecture['activation'].upper()}</p>
            <p><strong>Parameters:</strong> {sum(layers[i] * layers[i+1] for i in range(len(layers)-1))}</p>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Performance Comparison
    st.markdown("## 📈 Detailed Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = {
        'Metric': ['Average Latency', 'Average Utilization', 'Tasks Completed', 'Deadline Misses'],
        'Baseline': [
            baseline_result['avg_latency'],
            baseline_result['avg_utilization'],
            baseline_result['num_completed'],
            baseline_result['deadline_misses']
        ],
        'Neural': [
            neural_result['avg_latency'],
            neural_result['avg_utilization'],
            neural_result['num_completed'],
            neural_result['deadline_misses']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Improvement'] = comparison_df['Neural'] - comparison_df['Baseline']
    comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Baseline'] * 100).round(2)
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance improvement visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Latency Comparison', 'Utilization Comparison', 
                       'Completion Rate', 'Deadline Misses'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Latency comparison
    fig.add_trace(
        go.Bar(x=['Baseline', 'Neural'], y=[baseline_result['avg_latency'], neural_result['avg_latency']],
               name='Avg Latency', marker_color=['#ff7f0e', '#1f77b4']),
        row=1, col=1
    )
    
    # Utilization comparison
    fig.add_trace(
        go.Bar(x=['Baseline', 'Neural'], y=[baseline_result['avg_utilization'], neural_result['avg_utilization']],
               name='Avg Utilization', marker_color=['#ff7f0e', '#1f77b4']),
        row=1, col=2
    )
    
    # Completion rate
    completion_rate_baseline = baseline_result['num_completed'] / num_tasks
    completion_rate_neural = neural_result['num_completed'] / num_tasks
    fig.add_trace(
        go.Bar(x=['Baseline', 'Neural'], y=[completion_rate_baseline, completion_rate_neural],
               name='Completion Rate', marker_color=['#ff7f0e', '#1f77b4']),
        row=2, col=1
    )
    
    # Deadline misses
    fig.add_trace(
        go.Bar(x=['Baseline', 'Neural'], y=[baseline_result['deadline_misses'], neural_result['deadline_misses']],
               name='Deadline Misses', marker_color=['#ff7f0e', '#1f77b4']),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Performance Metrics Comparison")
    st.plotly_chart(fig, use_container_width=True)

    # Time Series Analysis
    if show_detailed_plots:
        st.markdown("## ⏱️ Time Series Analysis")
        
        # Core utilization over time
        baseline_utils = np.array(baseline_sim.core_utilization_history)
        neural_utils = np.array(neural_sim.core_utilization_history)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Core Utilization Over Time - Baseline', 'Core Utilization Over Time - Neural'),
            vertical_spacing=0.1
        )
        
        for i in range(num_cores):
            fig.add_trace(
                go.Scatter(x=list(range(len(baseline_utils))), y=baseline_utils[:, i],
                          name=f'Core {i}', mode='lines'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(range(len(neural_utils))), y=neural_utils[:, i],
                          name=f'Core {i}', mode='lines'),
                row=2, col=1
            )
        
        fig.update_layout(height=600, title_text="Core Utilization Patterns")
        st.plotly_chart(fig, use_container_width=True)
        
        # Queue length comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(baseline_sim.queue_length_history))),
            y=baseline_sim.queue_length_history,
            name='Baseline Queue Length',
            line=dict(color='#ff7f0e')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(neural_sim.queue_length_history))),
            y=neural_sim.queue_length_history,
            name='Neural Queue Length',
            line=dict(color='#1f77b4')
        ))
        fig.update_layout(
            title="Task Queue Length Over Time",
            xaxis_title="Time Steps",
            yaxis_title="Queue Length",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Workload Analysis
    st.markdown("## 📋 Workload Analysis")
    
    # Task distribution
    task_df = pd.DataFrame([
        {
            'Task ID': task.task_id,
            'Arrival Time': task.arrival_time,
            'Duration': task.duration,
            'Priority': task.priority,
            'Memory Demand': task.memory_demand if task.memory_demand else 0,
            'Deadline': task.deadline if task.deadline else None
        }
        for task in tasks
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Task Distribution")
        fig = px.histogram(task_df, x='Duration', nbins=10, title='Task Duration Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Arrival Pattern")
        fig = px.scatter(task_df, x='Arrival Time', y='Duration', 
                        title='Task Arrival vs Duration',
                        color='Priority' if 'Priority' in task_df.columns else None)
        st.plotly_chart(fig, use_container_width=True)

    # Summary and Insights
    st.markdown("## 💡 Insights & Recommendations")
    
    # Calculate improvements
    latency_improvement = (baseline_result['avg_latency'] - neural_result['avg_latency']) / baseline_result['avg_latency'] * 100
    utilization_improvement = (neural_result['avg_utilization'] - baseline_result['avg_utilization']) / baseline_result['avg_utilization'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Findings")
        if latency_improvement > 0:
            st.success(f"✅ Neural scheduler reduced average latency by {latency_improvement:.1f}%")
        else:
            st.error(f"❌ Neural scheduler increased average latency by {abs(latency_improvement):.1f}%")
            
        if utilization_improvement > 0:
            st.success(f"✅ Neural scheduler improved utilization by {utilization_improvement:.1f}%")
        else:
            st.warning(f"⚠️ Neural scheduler reduced utilization by {abs(utilization_improvement):.1f}%")
    
    with col2:
        st.markdown("### Recommendations")
        recommendations = []
        if latency_improvement > 5:
            recommendations.append("🎯 Neural scheduler shows significant latency improvements")
        if neural_result['deadline_misses'] < baseline_result['deadline_misses']:
            recommendations.append("⏰ Better deadline compliance with neural approach")
        if neural_result['num_completed'] > baseline_result['num_completed']:
            recommendations.append("✅ Higher task completion rate achieved")
        
        if not recommendations:
            recommendations.append("🔍 Consider tuning neural architecture parameters")
            recommendations.append("📊 Analyze workload characteristics for optimization")
        
        for rec in recommendations:
            st.info(rec)

    # Export functionality
    st.markdown("## 📤 Export Results")
    
    if st.button("Export Results to CSV"):
        # Create comprehensive results dataframe
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario,
            'num_tasks': num_tasks,
            'num_cores': num_cores,
            'max_time': max_time,
            'seed': seed,
            'baseline_avg_latency': baseline_result['avg_latency'],
            'neural_avg_latency': neural_result['avg_latency'],
            'baseline_avg_utilization': baseline_result['avg_utilization'],
            'neural_avg_utilization': neural_result['avg_utilization'],
            'baseline_completed': baseline_result['num_completed'],
            'neural_completed': neural_result['num_completed'],
            'baseline_deadline_misses': baseline_result['deadline_misses'],
            'neural_deadline_misses': neural_result['deadline_misses'],
            'architecture_layers': architecture['num_layers'],
            'architecture_hidden_dim': architecture['hidden_dim'],
            'architecture_activation': architecture['activation']
        }
        
        export_df = pd.DataFrame([export_data])
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"nas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Configure your simulation parameters in the sidebar and click 'Run Simulation' to get started!")
    
    # Show sample architecture
    st.markdown("## 🏗️ Sample Neural Architecture")
    sample_arch = {
        "num_layers": 2,
        "hidden_dim": 64,
        "activation": "relu"
    }
    st.json(sample_arch)
    
    st.markdown("""
    ### What this dashboard provides:
    
    🚀 **Performance Comparison**: Side-by-side comparison of baseline vs neural scheduling
    
    📊 **Multiple Visualizations**: 
    - Time series analysis of core utilization
    - Queue length tracking
    - Performance metrics comparison
    - Architecture visualization
    
    💡 **Insights & Recommendations**: AI-powered analysis of results
    
    📤 **Export Capabilities**: Download results for further analysis
    
    🔧 **Interactive Configuration**: Adjust workload types, task counts, and core counts
    """)
