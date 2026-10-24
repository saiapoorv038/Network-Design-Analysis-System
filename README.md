# Network Design & Analysis System (NET-DAS)

A Computer Networks project for interactive network topology design, structural analysis, Quality of Service (QoS) routing, performance evaluation, and AI-assisted bottleneck prediction.

## Overview

NET-DAS is a web-based network design and analysis tool that helps users construct network topologies, evaluate their connectivity and performance, and explore possible failure scenarios.

## Features

- **Interactive Network Design:** Create nodes and links and experiment with Hub-and-Spoke, Ring, and Mesh topologies.
- **Graph Analysis:** Evaluate network connectivity, centrality, articulation points, and node risk.
- **Network Performance Metrics:** Estimate link delay, utilization, packet loss, and throughput.
- **QoS Routing:** Compare routes using network link metrics.
- **AI-Assisted Bottleneck Prediction:** Classify potential link risks and generate recommendations.
- **Domain-Based Evaluation:** Assess network designs for campus, hospital, IoT, and enterprise scenarios.
- **Failure Simulation:** Test the impact of node failures and investigate network resilience.
- **Interactive Visualization:** Explore network layouts and packet-flow animations through a browser interface.

## Technology Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python and Flask
- Graph analysis: NetworkX
- Numerical computation: NumPy and SciPy
- Machine learning: Scikit-learn, with a NumPy-based fallback in the AI engine

## Project Structure

```text
Network-Design-Analysis-System/
├── indexC.html
├── CN_Server.py
├── ai_engine.py
├── requirements.txt
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/saiapoorv038/Network-Design-Analysis-System.git
cd Network-Design-Analysis-System
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Start the backend

```bash
python CN_Server.py
```

### 4. Open the frontend

Open `indexC.html` in your browser. The backend should be available at `http://localhost:5050`, matching the API address configured in the frontend.

## How to Use

1. Load a preset topology or create nodes and links.
2. Click **Run Analysis** to inspect network metrics and recommendations.
3. Explore the analysis, AI actions, domain-fit, and network-performance panels.
4. Select source and destination nodes to evaluate routing.
5. Use **Inject Failure** to simulate a node outage and examine its impact.

## Notes and Limitations

This project is an educational prototype. Its network performance metrics and failure scenarios are simulated or mathem
