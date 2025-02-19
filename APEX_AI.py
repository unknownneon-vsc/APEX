import os

import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
import torch
import torchvision  # Add this import
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dqn_agent import DQNAgent
import gym
import numpy as np
import hashlib
import random  # Add this import
import openai  # Add this import

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("APEX")

# Define transformations for the training data and testing data
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

# Download and load the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define the classes in the CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network, define the criterion and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
def train(net, trainloader, criterion, optimizer, epochs=2):  # Reduced epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

# Test the network on the test data
def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# Function to display some sample test images and predictions
def display_predictions(net, testloader, classes, num_images=5):
    dataiter = iter(testloader)
    images, labels = next(dataiter)  # Use the next() function
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(num_images)))
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(num_images)))

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge (question TEXT, answer TEXT)''')
    conn.commit()
    conn.close()

# Store question and answer in the database
def store_knowledge(question, answer):
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute("INSERT INTO knowledge (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

# Retrieve answer from the database
def retrieve_knowledge(question):
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute("SELECT answer FROM knowledge WHERE question=?", (question,))
    answer = c.fetchone()
    conn.close()
    return answer[0] if answer else None

# Define the APEX class
class APEX:
    def __init__(self):
        self.knowledge_base = {}

    def answer(self, question):
        # Check if the answer is already in the knowledge base
        answer = retrieve_knowledge(question)
        if answer:
            return answer

        # Generate a response using OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                max_tokens=150,
                temperature=0.7,  # Adjust temperature for more creative responses
                top_p=0.9  # Use nucleus sampling
            )
            answer = response.choices[0].message['content'].strip()
        except Exception as e:
            answer = f"Sorry, I couldn't generate a response due to an error: {e}"

        store_knowledge(question, answer)
        return answer

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')  # Use environment variable for secret key
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Create an instance of the APEX class
apex = APEX()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# In-memory user store
users = {'admin': bcrypt.generate_password_hash('password').decode('utf-8')}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            user = User('admin')
            login_user(user)
            return redirect(url_for('guest'))
        elif username in users and bcrypt.check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('guest'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/guest')
@login_required
def guest():
    return render_template('guest.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/ask', methods=['GET', 'POST'])
@login_required
def ask():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question')
        response = apex.answer(question)
        return jsonify({'response': response})
    return render_template('ask.html')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        setting1 = request.form.get('setting1')
        setting2 = request.form.get('setting2')
        theme = request.form.get('theme')
        # Save settings logic here
        session['theme'] = theme
        return redirect(url_for('settings'))
    return render_template('settings.html', theme=session.get('theme', 'light'))

@app.context_processor
def inject_theme():
    return dict(theme=session.get('theme', 'light'))

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        # Save feedback logic here
        return redirect(url_for('feedback'))
    return render_template('feedback.html')

@app.context_processor
def inject_theme():
    return dict(theme=session.get('theme', 'light'))

# Track active users
active_users = set()

@app.before_request
def before_request():
    if current_user.is_authenticated:
        active_users.add(current_user.id)

@app.teardown_request
def teardown_request(exception):
    if current_user.is_authenticated:
        active_users.discard(current_user.id)

@app.route('/proxy', methods=['POST'])
@login_required
def proxy():
    data = request.get_json()
    url = data.get('url')
    method = data.get('method', 'GET')
    headers = data.get('headers', {})
    payload = data.get('payload', {})

    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=payload)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=payload)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=headers, json=payload)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers, json=payload)
        else:
            return jsonify({'error': 'Unsupported HTTP method'}), 400

        return jsonify({
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': response.json() if response.headers.get('Content-Type') == 'application/json' else response.text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    def test_torch():
        try:
            print(f"Torch version: {torch.__version__}")
            print("Torch library imported successfully.")
        except Exception as e:
            print(f"Error importing torch library: {e}")

    test_torch()
    
    # Initialize the database
    init_db()  # Add this line to initialize the database
    
    env = gym.make('CartPole-v1')

    # First determine state size from environment
    state = env.reset()
    if isinstance(state, tuple):
        state = np.concatenate([np.array(s, dtype=np.float32).flatten() if not isinstance(s, dict) else np.array(list(s.values()), dtype=np.float32).flatten() for s in state])
    else:
        state = np.array(state, dtype=np.float32).flatten() if not isinstance(state, dict) else np.array(list(state.values()), dtype=np.float32).flatten()
    state_size = state.shape[0]  # Get actual state size

    # Initialize agent with correct state size
    agent = DQNAgent(state_size=state_size, action_size=env.action_space.n)
    batch_size = 32

    # Train the neural network
    print("Starting neural network training...")  # Debug statement
    train(net, trainloader, criterion, optimizer, epochs=2)  # Reduced epochs
    test(net, testloader)
    display_predictions(net, testloader, classes)
    print("Neural network training completed.")  # Debug statement

    # Q-learning parameters
    num_episodes = 100  # Reduced episodes
    learning_rate = 0.1
    discount_factor = 0.99

    # Initialize Q-table with a fixed size
    Q_table_size = 10000  # Example size, adjust as needed
    Q = np.zeros([Q_table_size, env.action_space.n])

    # Function to convert state to a unique index
    def state_to_index(state, table_size):
        state_str = str(state)
        hash_object = hashlib.md5(state_str.encode())
        return int(hash_object.hexdigest(), 16) % table_size

    # Training loop
    print("Starting Q-learning training...")  # Debug statement
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = np.concatenate([np.array(s, dtype=np.float32).flatten() if not isinstance(s, dict) else np.array(list(s.values()), dtype=np.float32).flatten() for s in state])
        else:
            state = np.array(state, dtype=np.float32).flatten() if not isinstance(state, dict) else np.array(list(state.values()), dtype=np.float32).flatten()
        state_index = state_to_index(state, Q_table_size)
        done = False

        while not done:
            # Choose action based on epsilon-greedy policy
            if random.uniform(0, 1) < 0.1:  # Explore
                action = env.action_space.sample()
            else:  # Exploit
                action = np.argmax(Q[state_index, :])

            # Take action and observe new state and reward
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if isinstance(new_state, tuple):
                new_state = np.concatenate([np.array(s, dtype=np.float32).flatten() if not isinstance(s, dict) else np.array(list(s.values()), dtype=np.float32).flatten() for s in new_state])
            else:
                new_state = np.array(new_state, dtype=np.float32).flatten() if not isinstance(new_state, dict) else np.array(list(new_state.values()), dtype=np.float32).flatten()
            new_state_index = state_to_index(new_state, Q_table_size)

            # Update Q-value
            Q[state_index, action] = Q[state_index, action] + learning_rate * (reward + discount_factor * np.max(Q[new_state_index, :]) - Q[state_index, action])

            state_index = new_state_index

    # Close the environment
    env.close()
    print("Q-learning training completed.")  # Debug statement

    # Run Flask app on port 5001
    app.run(host='127.0.0.1', port=5001)