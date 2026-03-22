#!/bin/bash
cd /Users/sasha/IdeaProjects/personal_projects/rag_challenge
nohup uv run python scripts/task_server.py > /tmp/task_server.log 2>&1 &
echo "Task server started, PID=$!, log at /tmp/task_server.log"
