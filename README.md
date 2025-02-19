# APEX AI Project

## Setup

1. Obtain an OpenAI API key from [OpenAI](https://beta.openai.com/signup/).
2. Create a `.env` file in the project root directory and add your API key:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Run the Flask application:
    ```bash
    python APEX_AI.py
    ```
2. Open your web browser and navigate to `http://127.0.0.1:5001` to access the application.

### `APEX_AI.py`
This file contains the main application logic for the APEX AI project. It includes the following functionalities:
- User authentication using Flask-Login and Flask-Bcrypt
- Flask web application setup
- Neural network training and testing using PyTorch
- Q-learning implementation for the CartPole-v1 environment using OpenAI Gym
- Knowledge base management using SQLite

### `dqn_agent.py`
This file contains the implementation of the DQNAgent class, which is used for the Q-learning algorithm in the APEX AI project.

### `fine_tune_gpt2.py`
This file contains the code for fine-tuning a GPT-2 model using the Hugging Face Transformers library.

### `test_transformers.py`
This file contains a test script to verify the installation of the Hugging Face Transformers library.

### `test_torch.py`
This file contains a test script to verify the installation of the PyTorch library.

### `templates/`
This directory contains the HTML templates for the Flask web application. The following templates are included:
- `index.html`: The home page of the application.
- `login.html`: The login page for user authentication.
- `ask.html`: The page where users can ask questions to the APEX AI system.

### `static/styles.css`
This file contains the CSS styles for the HTML templates.