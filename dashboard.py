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

# Enhanced model building logic for expanded search space
def build_model(arch, num_cores):
    layers = []
    input_dim = 3  # arrival, duration, free_cores
    
    # Add attention mechanism if specified
    if arch.get("attention_heads", 0) > 0:
        attention_dim = arch["hidden_dim"]
        layers.append(nn.MultiheadAttention(attention_dim, arch["attention_heads"], batch_first=True))
        layers.append(nn.LayerNorm(attention_dim))
    
    for i in range(arch["num_layers"]):
        # Add residual connection if specified
        if arch.get("residual_connections", False) and i > 0:
            layers.append(ResidualBlock(input_dim, arch["hidden_dim"]))
        else:
            layers.append(nn.Linear(input_dim, arch["hidden_dim"]))
        
        # Add layer normalization if specified
        if arch.get("layer_normalization", False):
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
        
        # Add regularization
        if arch.get("regularization") == "dropout" and arch.get("dropout_rate", 0.0) > 0:
            layers.append(nn.Dropout(arch["dropout_rate"]))
        elif arch.get("regularization") == "batch_norm":
            layers.append(nn.BatchNorm1d(arch["hidden_dim"]))
        
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
    
    num_tasks = st.slider("Number of Tasks", 10, 50, 20, help="Total number of tasks to schedule")
    num_cores = st.slider("Number of Cores", 2, 8, 4, help="Number of available CPU cores")
    seed = st.number_input("Random Seed", value=42, help="Seed for reproducible results")
    
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
    
    with st.spinner("Running simulation..."):
        # Generate workload
        workload_meta = {"scenario": scenario}
        tasks = generate_workload(scenario, num_tasks=num_tasks, seed=seed)
        
        # Baseline Simulation
        baseline_policy = RoundRobinPolicy()
        sim1 = Simulator(num_cores=num_cores, scheduling_policy=baseline_policy)
        sim1.load_tasks(tasks)
        baseline_result = sim1.run(max_time=100)

        # Surrogate NAS - Get top 5 architectures
        surrogate = SurrogateModel()
        top_architectures = surrogate.suggest(workload_meta=workload_meta)
        
        # Use the best architecture for simulation
        best_arch = top_architectures[0][0]
        model = build_model(best_arch, num_cores)
        neural_policy = NeuralSchedulerPolicy(model)
        sim2 = Simulator(num_cores=num_cores, scheduling_policy=neural_policy)
        sim2.load_tasks(tasks)
        neural_result = sim2.run(max_time=100)

    # Display results
    st.success("✅ Simulation completed!")
    
    # Top 5 Neural Architectures Section
    st.subheader("🏆 Top 5 Neural Architectures")
    
    # Create visual cards for top 5 architectures
    arch_cols = st.columns(5)
    
    for i, (arch, predicted_score) in enumerate(top_architectures[:5]):
        with arch_cols[i]:
            # Create a card-like display with expanded information
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
            ">
                <h3 style="margin: 0 0 1rem 0; font-size: 1.3rem;">{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '🏆' if i==3 else '��️'} Rank #{i+1}</h3>
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
                    <strong>Reg:</strong> {arch.get("regularization", "linear").upper()}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Attention:</strong> {arch.get("attention_heads", 0)}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Residual:</strong> {'✓' if arch.get("residual_connections", False) else '✗'}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.3rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>Score:</strong> {predicted_score:.2f}
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

    # Performance comparison
    st.subheader("📈 Performance Comparison")
    
    # Create metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🔄 Baseline (Round Robin)")
        st.metric("Avg Latency", f"{baseline_result['avg_latency']:.2f}")
        st.metric("Avg Utilization", f"{baseline_result['avg_utilization']:.2%}")
        st.metric("Deadline Misses", baseline_result['deadline_misses'])
        st.metric("Completed Tasks", baseline_result['num_completed'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🧠 Neural Policy")
        st.metric("Avg Latency", f"{neural_result['avg_latency']:.2f}")
        st.metric("Avg Utilization", f"{neural_result['avg_utilization']:.2%}")
        st.metric("Deadline Misses", neural_result['deadline_misses'])
        st.metric("Completed Tasks", neural_result['num_completed'])
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
