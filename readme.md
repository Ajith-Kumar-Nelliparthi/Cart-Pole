# 🎯 CartPole Q-Learning Agent

## 📝 Project Overview

This project implements a Q-learning reinforcement learning algorithm to balance a pole on a moving cart using the OpenAI Gymnasium CartPole-v1 environment

 ![cart_pole](https://github.com/user-attachments/assets/3a4f48b4-18c1-49f7-9a4f-14432e7c568e)



## 📌 How It Works

1. Environment Setup: The CartPole-v1 environment is created using Gymnasium.
2. Q-learning Algorithm:
   >A Q-table is initialized with values between 0 and 1.\
   >The state space is discretized into 20 bins for each of the four state variables.\
   >The agent selects actions using an epsilon-greedy policy.\
   >Rewards are updated based on the pole's angle to encourage balance.\
   >The Q-table is updated using the Bellman equation.
3. Training: The agent is trained over 20,000 episodes, with epsilon decreasing gradually.
4. Testing: The trained agent is tested with the learned Q-table.
5. Visualization: The best episode is recorded and saved as a video.

## 🚀 Installation

1️⃣ Clone the Repository
```
git clone https://github.com/Ajith-Kumar-Nelliparthi/Cart-Pole.git
cd Cart-Pole.git
```

2️⃣ Create a Virtual Environment (Optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

## 🏁 How to Train the Agent

Run the following command to train the Q-learning agent:
```
python cart_pole.py
```
## Training Process

1. The agent learns through 30,000 episodes (default)
2. Updates the Q-table based on rewards
3. Saves the best performing model
4. Saves a video (success.mp4) of the best episode

## 🎥 Viewing the Learned Policy

After training, you can visualize the best model using:
```
python cart_pole.py --test
```
This will run the trained agent in the environment without learning, using the saved cart_q_table.npy file.

## 🔧 Hyperparameters

You can adjust the training parameters in q.py:
```
LEARNING_RATE = 0.1    # Learning rate for Q-learning updates
DISCOUNT = 0.95        # Discount factor for future rewards
EPISODES = 30000       # Number of training episodes
SHOW_EVERY = 1000      # Render every N episodes
DISCRETE_OS_SIZE = [20, 20]  # Discretization of state space
```
## 📂 Project Structure
```
📁 MountainCar-Q-Learning
│── cart_pole.py                # Main training script
│── cart_q_table.npy            # Saved Q-table (after training)
│── requirements.txt            # Dependencies
│── results/                    # Folder for saving videos
```
## 📜 References

[OpenAI Gym](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

[Q-learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)

## 🤝 Contributing

Feel free to fork this repository and contribute with improvements!

## 📧 Contact

For any questions, reach out to: \

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/Ajith532542840)\
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nelliparthi-ajith-233803262)\
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](nelliparthi123@gmail.com)

## 🌟 If you like this project, give it a star! ⭐

