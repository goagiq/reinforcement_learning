#!/bin/bash
# GPU Monitoring Script
# Displays GPU utilization and memory usage

nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

