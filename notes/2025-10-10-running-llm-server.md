# Running vLLM Server in Background

This guide shows how to run the vLLM server in the background for long-running LLM analysis.

## Option 1: Using `screen` (Recommended)

Best for interactive management and easy reconnection.

```bash
# Start a new screen session for the server
screen -S vllm_server

# Start vllm inside the screen session
vllm serve openai/gpt-oss-120b

# Detach from screen: Press Ctrl+A, then D
# The server keeps running in the background

# Later, to check on it or view logs:
screen -r vllm_server

# To list all screen sessions:
screen -ls

# To kill the server:
screen -r vllm_server
# Then Ctrl+C to stop vllm
# Then exit or Ctrl+D to close the screen session
```

## Option 2: Using `tmux`

Similar to screen but with more features.

```bash
# Start tmux session
tmux new -s vllm_server

# Start vllm
vllm serve openai/gpt-oss-120b

# Detach: Ctrl+B, then D

# Reattach:
tmux attach -t vllm_server

# List sessions:
tmux ls

# Kill session:
tmux kill-session -t vllm_server
```

## Option 3: Using `nohup`

Simple background process with log file.

```bash
# Start vllm in background with logging
nohup vllm serve openai/gpt-oss-120b > vllm_server.log 2>&1 &

# Check if it's running:
ps aux | grep vllm

# View logs in real-time:
tail -f vllm_server.log

# Kill it later:
pkill -f "vllm serve"
# Or find PID and kill:
ps aux | grep vllm
kill <PID>
```

## Running Analysis Scripts

Once the server is running in background, you can run analysis in another screen/tmux session:

```bash
# Start a new screen session for analysis
screen -S llm_analysis

# Run analysis with many jets
uv run python scripts/analyze_llm_templates.py --num_jets 1000

# Detach: Ctrl+A, then D

# Check progress later:
screen -r llm_analysis
```

## Workflow Example

Complete workflow with both server and analysis in background:

```bash
# Terminal 1: Start vLLM server
screen -S vllm_server
vllm serve openai/gpt-oss-120b
# Press Ctrl+A, then D to detach

# Terminal 1: Start analysis
screen -S llm_analysis
uv run python scripts/analyze_llm_templates.py \
  --num_jets 100 \
  --reasoning_efforts low medium high \
  --templates simple_list with_summary_stats with_optimal_cut with_engineered_features
# Press Ctrl+A, then D to detach

# Check on analysis progress
screen -r llm_analysis
# Press Ctrl+A, then D to detach again

# When done, generate plots
uv run python scripts/plot_llm_analysis.py results/llm_analysis_*.json

# Clean up when finished
screen -r vllm_server
# Press Ctrl+C to stop server
# Type 'exit' to close screen session
```

## Checking Server Status

```bash
# Check if vLLM is running
ps aux | grep vllm

# Check if port 8000 is in use
lsof -i :8000

# Test server with curl
curl http://localhost:8000/v1/models

# Or use the analysis script's built-in check
uv run python scripts/analyze_llm_templates.py --num_jets 1
```

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Try a different port: `vllm serve openai/gpt-oss-120b --port 8001`
- Update analysis script: `--base_url http://localhost:8001/v1`

### Can't reconnect to screen
- List sessions: `screen -ls`
- If session shows as "Detached", reattach: `screen -r vllm_server`
- If session shows as "Attached", force detach: `screen -d -r vllm_server`

### Out of memory
- Reduce model size or use GPU with more memory
- Check memory usage: `nvidia-smi` (for GPU) or `free -h` (for CPU)

### Analysis script can't connect to server
- Verify server is running: `ps aux | grep vllm`
- Test connection: `curl http://localhost:8000/v1/models`
- Check firewall settings if running on HPC cluster
