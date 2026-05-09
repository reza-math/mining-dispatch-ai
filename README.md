# 🚜💎 AI-Driven Mine Dispatch Optimization

This project implements a Reinforcement Learning (RL) agent designed to optimize truck dispatching in a mining environment. The goal is to maximize production across multiple shovels while minimizing truck queue times and meeting specific production targets.

## 🚀 Overview
The simulation models a complex mining operation where an AI "Dispatcher" allocates trucks to different shovels. The agent learns through trial and error to balance:
- **Production Efficiency:** Reaching target tonnage for each shovel.
- **Operational Flow:** Minimizing bottlenecks and idle times in the queue.

## 📊 Performance Indicators
In recent training episodes (e.g., Episode 321), the model achieved:
- **Efficiency Score:** 651.00
- **Dynamic Balancing:** Successfully reaching 1500-ton targets on various shovels dynamically.
- **Automated Saving:** Model weights are saved as `mining_model.pth` for continuous learning.

## 🛠️ Technology Stack
- **Language:** Python 3.x
- **Frameworks:** PyTorch / Reinforcement Learning Logic
- **Logic:** Discrete Event Simulation for Mine Dispatching

## 🚀 How to Run
1. Install dependencies: `pip install torch` (and any other libraries).
2. Run the training script: `mining-dispatch-ai.py`.
3. The model will start training and automatically save the weights as `mining_model.pth`.

## 📚 References & Academic Background
This project is inspired by and built upon the following research papers in the field of mine dispatching and fleet management:

1. **Mirzaei-Nasirabad, H., Mohtasham, M., Askari-Nasab, H. et al. (2023).** *An optimization model for the real-time truck dispatching problem in open-pit mining operations.* Optim Eng 24, 2449–2473.
2. **Mirzaei Nasirabad, H., Mohtasham, M., Rahimzadeh-Nanekaran, F. (2023).** *Evaluating the Effect of Fleet Management on the Performance of Mining Operations Using Integer Linear Programming.* IJMS, 16(1), 139-155.
3. **Mohtasham, M., Mirzaei-Nasirabad, H., Askari-Nasab, H. & Alizadeh, B. (2021).** *A multi-objective model for fleet allocation schedule in open-pit mines.* International Journal of Mining, Reclamation and Environment, 35:10, 709-727.
4. **Chang, Y., Ren, H., & Wang, S. (2015).** *Modelling and Optimizing an Open-Pit Truck Scheduling Problem.* Discrete Dynamics in Nature and Society.
5. **Arelovich, F., Masson, O., Agamennoni, O. et al. (2010).** *Heuristic rule for truck dispatching in open-pit mines with local information-based decisions.* IEEE ITSC, 1408-1414.
6. **Temeng, V. A., Otuonye, F. O., & Frendewey, J. O. (1998).** *A Nonpreemptive Goal Programming Approach to Truck Dispatching in Open Pit Mines.* Mineral Resources Engineering, 7(2), 59-67.
