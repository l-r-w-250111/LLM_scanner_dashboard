# LLM_scanner_dashboard

A Streamlit-based dashboard to scan and display the status of running LLM/embedding models, GPU usage, and containerized environments.

## Features

- **Local Process Scanning**: Detects running LLM and embedding model processes based on a configurable set of rules.
- **GPU Monitoring**: Displays GPU usage details by running `nvidia-smi`.
- **Container Scanning**: Lists running Docker and Podman containers.
- **System Metrics**: Shows real-time CPU and memory usage.

## Requirements

- Python 3.9+

## Installation and Usage

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    pip install -r requirements.txt
    ```

2.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
